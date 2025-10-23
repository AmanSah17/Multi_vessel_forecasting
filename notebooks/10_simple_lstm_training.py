"""
Simple LSTM Training - Fast & Reliable

Optimized for:
- Fast training (10-15 minutes)
- CUDA GPU acceleration
- Real-time progress with tqdm
- Minimal dependencies
- Robust error handling
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
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_simple_lstm.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SimpleLSTM(nn.Module):
    """Simple PyTorch LSTM Model."""
    
    def __init__(self, input_size, hidden_size=64, output_size=2):
        """Initialize LSTM."""
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """Forward pass."""
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class SimpleLSTMTrainer:
    """Simple LSTM trainer."""
    
    def __init__(self, sample_size=50000, batch_size=32, epochs=10):
        """Initialize."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_dir = Path('training_logs_simple')
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Sample size: {sample_size:,}")
        logger.info(f"Batch size: {batch_size}, Epochs: {epochs}")
    
    def load_data(self, data_path):
        """Load data."""
        logger.info("\n[1/4] LOADING DATA")
        logger.info("="*70)
        
        logger.info(f"Loading: {data_path}")
        df = pd.read_csv(data_path)
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], format="%Y-%m-%dT%H:%M:%S", errors='coerce')
        
        logger.info(f"Loaded {len(df):,} records, {df['MMSI'].nunique():,} vessels")
        
        # Sample
        logger.info(f"Sampling to {self.sample_size:,}...")
        if len(df) > self.sample_size:
            indices = np.random.choice(len(df), self.sample_size, replace=False)
            df = df.iloc[indices].reset_index(drop=True)
        
        logger.info(f"Sampled to {len(df):,} records")
        self.results['data'] = {'records': len(df), 'vessels': df['MMSI'].nunique()}
        
        return df
    
    def prepare_features(self, df):
        """Prepare features."""
        logger.info("\n[2/4] PREPARING FEATURES")
        logger.info("="*70)
        
        # Sort by MMSI and time
        df = df.sort_values(['MMSI', 'BaseDateTime']).reset_index(drop=True)
        
        # Select key features
        features = ['LAT', 'LON', 'SOG', 'COG']
        
        # Add temporal features
        df['hour'] = df['BaseDateTime'].dt.hour
        df['day_of_week'] = df['BaseDateTime'].dt.dayofweek
        features.extend(['hour', 'day_of_week'])
        
        # Add kinematic features
        df['speed_change'] = df.groupby('MMSI')['SOG'].diff().fillna(0)
        df['heading_change'] = df.groupby('MMSI')['COG'].diff().fillna(0)
        features.extend(['speed_change', 'heading_change'])
        
        logger.info(f"Features: {features}")
        
        return df, features
    
    def create_sequences(self, df, features, seq_length=30):
        """Create sequences."""
        logger.info("\n[3/4] CREATING SEQUENCES")
        logger.info("="*70)
        
        X, y = [], []
        
        logger.info(f"Creating sequences (length={seq_length})...")
        
        for mmsi in tqdm(df['MMSI'].unique(), desc="Processing vessels"):
            vessel_data = df[df['MMSI'] == mmsi][features].values
            
            if len(vessel_data) < seq_length + 1:
                continue
            
            for i in range(len(vessel_data) - seq_length):
                X.append(vessel_data[i:i+seq_length])
                y.append(vessel_data[i+seq_length, :2])  # Next LAT, LON
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        logger.info(f"Created {len(X):,} sequences")
        self.results['sequences'] = {'count': len(X)}
        
        # Normalize
        scaler = MinMaxScaler()
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)
        
        return X_scaled, y, scaler
    
    def train(self, X, y):
        """Train LSTM."""
        logger.info("\n[4/4] TRAINING LSTM")
        logger.info("="*70)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Split
        split_idx = int(0.8 * len(X))
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
        
        logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
        
        # DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Model
        model = SimpleLSTM(input_size=X.shape[2]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training
        logger.info(f"Training for {self.epochs} epochs...")
        best_val_loss = float('inf')
        
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
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), self.output_dir / 'best_model.pt')
        
        logger.info(f"Best model saved")
        
        self.results['training'] = {
            'epochs': self.epochs,
            'final_train_loss': float(train_loss),
            'final_val_loss': float(best_val_loss)
        }
        
        return model
    
    def save_results(self):
        """Save results."""
        path = self.output_dir / 'results.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results saved to {path}")


def main():
    """Main execution."""
    logger.info("\n" + "="*70)
    logger.info("SIMPLE LSTM TRAINING WITH CUDA")
    logger.info("="*70)
    
    trainer = SimpleLSTMTrainer(sample_size=50000, batch_size=32, epochs=10)
    
    # Load data
    data_path = r"D:\Maritime_Vessel_monitoring\csv_extracted_data\AIS_2020_01_03\AIS_2020_01_03.csv"
    df = trainer.load_data(data_path)
    
    # Prepare features
    df, features = trainer.prepare_features(df)
    
    # Create sequences
    X, y, scaler = trainer.create_sequences(df, features)
    
    # Train
    model = trainer.train(X, y)
    
    # Save results
    trainer.save_results()
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Model saved to: {trainer.output_dir}/best_model.pt")
    logger.info(f"Results saved to: {trainer.output_dir}/results.json")


if __name__ == '__main__':
    main()

