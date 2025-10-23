"""
Fast LSTM Training with CUDA Support and Progress Bars

Optimized for:
- LSTM model only (prediction)
- Trajectory verification only
- CUDA GPU acceleration
- Real-time progress tracking with tqdm
- Minimal memory footprint
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import VesselDataPreprocessor
from src.trajectory_verification import TrajectoryVerifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_lstm_cuda.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FastLSTMTrainer:
    """Fast LSTM trainer with CUDA support."""
    
    def __init__(self, output_dir='training_logs_lstm', sample_size=100000):
        """Initialize trainer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.sample_size = sample_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        logger.info(f"🚀 FastLSTMTrainer initialized")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"💾 Output: {self.output_dir}")
    
    def load_and_sample_data(self, data_path):
        """Load and sample data with progress bar."""
        logger.info("\n" + "="*70)
        logger.info("[1/5] LOADING DATA")
        logger.info("="*70)
        
        # Load data
        logger.info(f"📂 Loading from: {data_path}")
        df = pd.read_csv(data_path)
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], format="%Y-%m-%dT%H:%M:%S", errors='coerce')
        
        logger.info(f"✓ Loaded {len(df):,} records, {df['MMSI'].nunique():,} vessels")
        self.results['raw_data'] = {
            'records': len(df),
            'vessels': df['MMSI'].nunique()
        }
        
        # Sample data
        logger.info(f"\n📊 Sampling to {self.sample_size:,} records...")
        df_sampled = df.groupby('MMSI', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(len(x) * self.sample_size / len(df))))),
            include_groups=False
        ).reset_index(drop=True)
        
        logger.info(f"✓ Sampled to {len(df_sampled):,} records")
        self.results['sampled_data'] = {'records': len(df_sampled)}
        
        return df_sampled
    
    def preprocess_data(self, df):
        """Preprocess data with progress bar."""
        logger.info("\n" + "="*70)
        logger.info("[2/5] PREPROCESSING DATA")
        logger.info("="*70)
        
        preprocessor = VesselDataPreprocessor()
        logger.info("🔧 Preprocessing...")
        df_processed = preprocessor.preprocess(df)
        
        logger.info(f"✓ Preprocessed to {len(df_processed):,} records")
        self.results['processed_data'] = {'records': len(df_processed)}
        
        return df_processed
    
    def engineer_features(self, df):
        """Engineer features with progress bar."""
        logger.info("\n" + "="*70)
        logger.info("[3/5] ENGINEERING FEATURES")
        logger.info("="*70)
        
        logger.info("🔨 Engineering 13 features...")
        
        # Sort by MMSI and time
        df = df.sort_values(['MMSI', 'BaseDateTime']).reset_index(drop=True)
        
        # Temporal features
        df['hour'] = df['BaseDateTime'].dt.hour
        df['day_of_week'] = df['BaseDateTime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Kinematic features
        df['speed_change'] = df.groupby('MMSI')['SOG'].diff().fillna(0)
        df['heading_change'] = df.groupby('MMSI')['COG'].diff().fillna(0)
        df['heading_change'] = df['heading_change'].apply(lambda x: min(abs(x), 360 - abs(x)))
        df['acceleration'] = df['speed_change'].rolling(window=2).mean().fillna(0)
        
        # Spatial features
        df['lat_change'] = df.groupby('MMSI')['LAT'].diff().fillna(0)
        df['lon_change'] = df.groupby('MMSI')['LON'].diff().fillna(0)
        
        # Statistical features
        df['rolling_mean_speed'] = df.groupby('MMSI')['SOG'].rolling(window=5, min_periods=1).mean().reset_index(drop=True)
        df['rolling_std_speed'] = df.groupby('MMSI')['SOG'].rolling(window=5, min_periods=1).std().fillna(0).reset_index(drop=True)
        
        logger.info(f"✓ Engineered 13 features")
        self.results['features'] = {'count': 13}
        
        return df
    
    def create_sequences(self, df, seq_length=60):
        """Create sequences for LSTM with progress bar."""
        logger.info("\n" + "="*70)
        logger.info("[4/5] CREATING SEQUENCES")
        logger.info("="*70)

        feature_cols = ['LAT', 'LON', 'SOG', 'COG', 'hour', 'day_of_week',
                       'is_weekend', 'speed_change', 'heading_change', 'acceleration',
                       'lat_change', 'lon_change', 'rolling_mean_speed']

        X, y = [], []

        logger.info(f"Creating sequences (length={seq_length})...")

        for mmsi in tqdm(df['MMSI'].unique(), desc="Processing vessels", unit="vessel"):
            vessel_data = df[df['MMSI'] == mmsi][feature_cols].values

            if len(vessel_data) < seq_length + 1:
                continue

            for i in tqdm(range(len(vessel_data) - seq_length), desc=f"MMSI {mmsi}", leave=False):
                X.append(vessel_data[i:i+seq_length])
                y.append(vessel_data[i+seq_length, :2])  # Next LAT, LON

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        logger.info(f"Created {len(X):,} sequences")
        self.results['sequences'] = {'count': len(X)}

        return X, y
    
    def train_lstm(self, X, y):
        """Train LSTM with PyTorch and CUDA."""
        logger.info("\n" + "="*70)
        logger.info("[5/5] TRAINING LSTM")
        logger.info("="*70)

        try:
            from sklearn.preprocessing import MinMaxScaler

            # Normalize data
            logger.info("Normalizing data...")
            scaler = MinMaxScaler()
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)

            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}")

            # Convert to PyTorch tensors
            logger.info("Converting to PyTorch tensors...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Device: {device}")

            X_train_t = torch.FloatTensor(X_train).to(device)
            y_train_t = torch.FloatTensor(y_train).to(device)
            X_val_t = torch.FloatTensor(X_val).to(device)
            y_val_t = torch.FloatTensor(y_val).to(device)

            # Build model
            logger.info("\nBuilding LSTM model...")
            model = self._build_pytorch_lstm(X.shape[1], X.shape[2]).to(device)
            logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

            # Training
            logger.info("\nTraining...")
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            patience = 3
            patience_counter = 0

            for epoch in tqdm(range(20), desc="Epochs"):
                # Train
                model.train()
                train_loss = 0
                batch_size = 32

                for i in tqdm(range(0, len(X_train_t), batch_size), desc=f"Epoch {epoch+1} batches", leave=False):
                    batch_X = X_train_t[i:i+batch_size]
                    batch_y = y_train_t[i:i+batch_size]

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= (len(X_train_t) // batch_size)
                train_losses.append(train_loss)

                # Validate
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()

                val_losses.append(val_loss)

                logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), self.output_dir / 'lstm_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

            # Load best model
            model.load_state_dict(torch.load(self.output_dir / 'lstm_model.pt'))
            logger.info("Model saved to lstm_model.pt")

            self.results['training'] = {
                'epochs': epoch + 1,
                'final_loss': float(train_losses[-1]),
                'final_val_loss': float(val_losses[-1])
            }

            return model, scaler

        except Exception as e:
            logger.error(f"Error during training: {e}")
            return None, None

    def _build_pytorch_lstm(self, seq_length, input_size):
        """Build PyTorch LSTM model."""
        class LSTMModel(torch.nn.Module):
            def __init__(self, input_size):
                super(LSTMModel, self).__init__()
                self.lstm1 = torch.nn.LSTM(input_size, 128, batch_first=True)
                self.dropout1 = torch.nn.Dropout(0.2)
                self.lstm2 = torch.nn.LSTM(128, 64, batch_first=True)
                self.dropout2 = torch.nn.Dropout(0.2)
                self.fc1 = torch.nn.Linear(64, 32)
                self.fc2 = torch.nn.Linear(32, 2)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                lstm_out1, _ = self.lstm1(x)
                lstm_out1 = self.dropout1(lstm_out1)
                lstm_out2, _ = self.lstm2(lstm_out1)
                lstm_out2 = self.dropout2(lstm_out2)
                last_hidden = lstm_out2[:, -1, :]
                fc_out = self.relu(self.fc1(last_hidden))
                output = self.fc2(fc_out)
                return output

        return LSTMModel(input_size)
    
    def verify_trajectories(self, df):
        """Verify trajectories with progress bar."""
        logger.info("\n" + "="*70)
        logger.info("TRAJECTORY VERIFICATION")
        logger.info("="*70)

        verifier = TrajectoryVerifier()

        logger.info("Verifying trajectories...")
        verification_results = []

        for mmsi in tqdm(df['MMSI'].unique(), desc="Verifying vessels", unit="vessel"):
            vessel_traj = df[df['MMSI'] == mmsi].sort_values('BaseDateTime')
            if len(vessel_traj) >= 3:
                result = verifier.verify_trajectory(vessel_traj)
                verification_results.append({
                    'mmsi': mmsi,
                    'smoothness': result['smoothness_score'],
                    'speed_consistency': result['speed_consistency'],
                    'heading_consistency': result['heading_consistency']
                })

        avg_smoothness = np.mean([r['smoothness'] for r in verification_results])
        logger.info(f"Average smoothness: {avg_smoothness:.4f}")

        self.results['verification'] = {
            'avg_smoothness': float(avg_smoothness),
            'vessels_verified': len(verification_results)
        }

        return verification_results
    
    def save_results(self):
        """Save results to JSON."""
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"✓ Results saved to {results_path}")


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("⚡ FAST LSTM TRAINING WITH CUDA")
    print("="*70 + "\n")
    
    # Initialize trainer
    trainer = FastLSTMTrainer(sample_size=100000)
    
    # Load data
    data_path = r"D:\Maritime_Vessel_monitoring\csv_extracted_data\AIS_2020_01_03\AIS_2020_01_03.csv"
    df = trainer.load_and_sample_data(data_path)
    
    # Preprocess
    df = trainer.preprocess_data(df)
    
    # Engineer features
    df = trainer.engineer_features(df)
    
    # Create sequences
    X, y = trainer.create_sequences(df)
    
    # Train LSTM
    model, scaler = trainer.train_lstm(X, y)
    
    # Verify trajectories
    verification = trainer.verify_trajectories(df)
    
    # Save results
    trainer.save_results()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Results saved to: {trainer.output_dir}/")
    print(f"🤖 Model saved to: {trainer.output_dir}/lstm_model.h5")


if __name__ == '__main__':
    main()

