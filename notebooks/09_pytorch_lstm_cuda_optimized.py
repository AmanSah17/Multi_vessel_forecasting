"""
PyTorch LSTM Training with CUDA - Ultra-Fast Version

Optimized for:
- PyTorch LSTM (faster than TensorFlow)
- CUDA GPU acceleration
- Real-time progress with tqdm
- Minimal memory usage
- Fast training (10-15 minutes)
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import VesselDataPreprocessor
from src.trajectory_verification import TrajectoryVerifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pytorch_cuda.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """PyTorch LSTM Model."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=2):
        """Initialize LSTM model."""
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        """Forward pass."""
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class PyTorchLSTMTrainer:
    """Fast PyTorch LSTM trainer."""
    
    def __init__(self, sample_size=100000, batch_size=64, epochs=15):
        """Initialize trainer."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_dir = Path('training_logs_pytorch')
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
        logger.info(f"🚀 PyTorch LSTM Trainer")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"⚙️  Batch size: {batch_size}, Epochs: {epochs}")
    
    def load_and_sample(self, data_path):
        """Load and sample data."""
        logger.info("\n" + "="*70)
        logger.info("[1/5] LOADING DATA")
        logger.info("="*70)

        logger.info(f"📂 Loading: {data_path}")
        df = pd.read_csv(data_path)
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], format="%Y-%m-%dT%H:%M:%S", errors='coerce')

        logger.info(f"✓ Loaded {len(df):,} records, {df['MMSI'].nunique():,} vessels")

        # Sample
        logger.info(f"📊 Sampling to {self.sample_size:,}...")
        if len(df) > self.sample_size:
            # Stratified sampling by MMSI
            sample_ratio = self.sample_size / len(df)
            df_list = []
            for mmsi in df['MMSI'].unique():
                vessel_df = df[df['MMSI'] == mmsi]
                sample_size_vessel = max(1, int(len(vessel_df) * sample_ratio))
                df_list.append(vessel_df.sample(n=min(sample_size_vessel, len(vessel_df))))
            df = pd.concat(df_list, ignore_index=True)

        logger.info(f"✓ Sampled to {len(df):,} records, {df['MMSI'].nunique():,} vessels")
        self.results['data'] = {'records': len(df), 'vessels': df['MMSI'].nunique()}

        return df
    
    def preprocess(self, df):
        """Preprocess data."""
        logger.info("\n" + "="*70)
        logger.info("[2/5] PREPROCESSING")
        logger.info("="*70)
        
        preprocessor = VesselDataPreprocessor()
        df = preprocessor.preprocess(df)
        logger.info(f"✓ Preprocessed to {len(df):,}")
        
        return df
    
    def engineer_features(self, df):
        """Engineer features."""
        logger.info("\n" + "="*70)
        logger.info("[3/5] FEATURE ENGINEERING")
        logger.info("="*70)
        
        df = df.sort_values(['MMSI', 'BaseDateTime']).reset_index(drop=True)
        
        # Temporal
        df['hour'] = df['BaseDateTime'].dt.hour
        df['day_of_week'] = df['BaseDateTime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Kinematic
        df['speed_change'] = df.groupby('MMSI')['SOG'].diff().fillna(0)
        df['heading_change'] = df.groupby('MMSI')['COG'].diff().fillna(0)
        df['heading_change'] = df['heading_change'].apply(lambda x: min(abs(x), 360 - abs(x)))
        df['acceleration'] = df['speed_change'].rolling(window=2).mean().fillna(0)
        
        # Spatial
        df['lat_change'] = df.groupby('MMSI')['LAT'].diff().fillna(0)
        df['lon_change'] = df.groupby('MMSI')['LON'].diff().fillna(0)
        
        # Statistical
        df['rolling_mean_speed'] = df.groupby('MMSI')['SOG'].rolling(window=5, min_periods=1).mean().reset_index(drop=True)
        df['rolling_std_speed'] = df.groupby('MMSI')['SOG'].rolling(window=5, min_periods=1).std().fillna(0).reset_index(drop=True)
        
        logger.info(f"✓ Engineered 13 features")
        
        return df
    
    def create_sequences(self, df, seq_length=60):
        """Create sequences."""
        logger.info("\n" + "="*70)
        logger.info("[4/5] CREATING SEQUENCES")
        logger.info("="*70)
        
        feature_cols = ['LAT', 'LON', 'SOG', 'COG', 'hour', 'day_of_week', 
                       'is_weekend', 'speed_change', 'heading_change', 'acceleration',
                       'lat_change', 'lon_change', 'rolling_mean_speed']
        
        X, y = [], []
        
        for mmsi in tqdm(df['MMSI'].unique(), desc="Creating sequences"):
            vessel_data = df[df['MMSI'] == mmsi][feature_cols].values
            
            if len(vessel_data) < seq_length + 1:
                continue
            
            for i in range(len(vessel_data) - seq_length):
                X.append(vessel_data[i:i+seq_length])
                y.append(vessel_data[i+seq_length, :2])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # Normalize
        scaler = MinMaxScaler()
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)
        
        logger.info(f"✓ Created {len(X):,} sequences")
        
        return X_scaled, y, scaler
    
    def train(self, X, y):
        """Train LSTM."""
        logger.info("\n" + "="*70)
        logger.info("[5/5] TRAINING LSTM")
        logger.info("="*70)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Split
        split_idx = int(0.8 * len(X))
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
        
        logger.info(f"✓ Train: {len(X_train):,}, Val: {len(X_val):,}")
        
        # DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Model
        model = LSTMModel(input_size=X.shape[2]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        logger.info(f"🏗️  Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Training loop
        logger.info(f"\n🚀 Training for {self.epochs} epochs...")
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Train
            model.train()
            train_loss = 0
            for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False):
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.output_dir / 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(self.output_dir / 'best_model.pt'))
        logger.info(f"✓ Best model saved")
        
        self.results['training'] = {
            'epochs': epoch + 1,
            'final_train_loss': float(train_loss),
            'final_val_loss': float(best_val_loss)
        }
        
        return model
    
    def verify_trajectories(self, df):
        """Verify trajectories."""
        logger.info("\n" + "="*70)
        logger.info("TRAJECTORY VERIFICATION")
        logger.info("="*70)
        
        verifier = TrajectoryVerifier()
        results = []
        
        for mmsi in tqdm(df['MMSI'].unique(), desc="Verifying"):
            vessel = df[df['MMSI'] == mmsi].sort_values('BaseDateTime')
            if len(vessel) >= 3:
                result = verifier.verify_trajectory(vessel)
                results.append({'smoothness': result['smoothness_score']})
        
        avg_smoothness = np.mean([r['smoothness'] for r in results])
        logger.info(f"✓ Average smoothness: {avg_smoothness:.4f}")
        
        self.results['verification'] = {'avg_smoothness': float(avg_smoothness)}
        
        return results
    
    def save_results(self):
        """Save results."""
        import json
        path = self.output_dir / 'results.json'
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"✓ Results saved to {path}")


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("PYTORCH LSTM WITH CUDA - ULTRA FAST")
    print("="*70 + "\n")
    
    trainer = PyTorchLSTMTrainer(sample_size=100000, batch_size=64, epochs=15)
    
    data_path = r"D:\Maritime_Vessel_monitoring\csv_extracted_data\AIS_2020_01_03\AIS_2020_01_03.csv"
    df = trainer.load_and_sample(data_path)
    df = trainer.preprocess(df)
    df = trainer.engineer_features(df)
    X, y, scaler = trainer.create_sequences(df)
    model = trainer.train(X, y)
    trainer.verify_trajectories(df)
    trainer.save_results()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()

