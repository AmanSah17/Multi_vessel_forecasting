"""
Full Pipeline: Process All AIS Data (Jan 3-8) with LSTM + MLflow

Features:
- Load all CSV files from Jan 3-8
- Per-vessel 70/20/10 train/val/test split
- LSTM model predicting LAT, LON, COG, Heading
- MLflow logging and model registry
- Visualization of predictions vs actual
"""

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_pipeline_mlflow.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM for predicting LAT, LON, COG, Heading."""
    
    def __init__(self, input_size, hidden_size=128, output_size=4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


def load_all_data(start_date=3, end_date=8):
    """Load all CSV files from Jan 3-8."""
    logger.info(f"\n{'='*70}")
    logger.info(f"[1/6] LOADING DATA (Jan {start_date}-{end_date})")
    logger.info(f"{'='*70}")
    
    base_path = Path(r"D:\Maritime_Vessel_monitoring\csv_extracted_data")
    dfs = []
    
    for day in range(start_date, end_date + 1):
        file_path = base_path / f"AIS_2020_01_{day:02d}" / f"AIS_2020_01_{day:02d}.csv"
        if file_path.exists():
            logger.info(f"Loading {file_path.name}...")
            df = pd.read_csv(file_path)
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], format="%Y-%m-%dT%H:%M:%S", errors='coerce')
            dfs.append(df)
            logger.info(f"  ✓ Loaded {len(df):,} records")
    
    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"\nTotal records: {len(df_all):,}")
    logger.info(f"Total vessels: {df_all['MMSI'].nunique():,}")
    logger.info(f"Date range: {df_all['BaseDateTime'].min()} to {df_all['BaseDateTime'].max()}")
    
    return df_all


def prepare_features(df):
    """Add temporal and kinematic features."""
    logger.info(f"\n{'='*70}")
    logger.info("[2/6] PREPARING FEATURES")
    logger.info(f"{'='*70}")
    
    df = df.sort_values('BaseDateTime').reset_index(drop=True)
    
    # Temporal features
    df['hour'] = df['BaseDateTime'].dt.hour
    df['day_of_week'] = df['BaseDateTime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Kinematic features per vessel
    df['speed_change'] = df.groupby('MMSI')['SOG'].diff().fillna(0)
    df['heading_change'] = df.groupby('MMSI')['COG'].diff().fillna(0)
    
    features = ['LAT', 'LON', 'SOG', 'COG', 'hour', 'day_of_week', 'speed_change', 'heading_change']
    logger.info(f"Features: {features}")
    
    return df, features


def create_sequences_per_vessel(df, features, seq_length=30):
    """Create sequences per vessel with 70/20/10 split."""
    logger.info(f"\n{'='*70}")
    logger.info("[3/6] CREATING SEQUENCES (Per-Vessel Split)")
    logger.info(f"{'='*70}")
    
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    vessels = df['MMSI'].unique()
    logger.info(f"Processing {len(vessels):,} vessels...")
    
    for mmsi in tqdm(vessels, desc="Vessels", unit="vessel"):
        vessel_data = df[df['MMSI'] == mmsi].sort_values('BaseDateTime')[features].values
        
        if len(vessel_data) < seq_length + 1:
            continue
        
        # Create sequences
        X_vessel, y_vessel = [], []
        for i in range(len(vessel_data) - seq_length):
            X_vessel.append(vessel_data[i:i+seq_length])
            y_vessel.append(vessel_data[i+seq_length, :4])  # LAT, LON, SOG, COG
        
        if len(X_vessel) == 0:
            continue
        
        # Per-vessel split: 70/20/10
        n = len(X_vessel)
        train_idx = int(0.7 * n)
        val_idx = int(0.9 * n)
        
        X_train.extend(X_vessel[:train_idx])
        y_train.extend(y_vessel[:train_idx])
        X_val.extend(X_vessel[train_idx:val_idx])
        y_val.extend(y_vessel[train_idx:val_idx])
        X_test.extend(X_vessel[val_idx:])
        y_test.extend(y_vessel[val_idx:])
    
    # Convert to numpy
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    # Normalize
    scaler = MinMaxScaler()
    X_flat = X_train.reshape(-1, X_train.shape[-1])
    X_scaled = scaler.fit_transform(X_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    return X_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler


def train_model_mlflow(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train with MLflow logging."""
    logger.info(f"\n{'='*70}")
    logger.info("[4/6] TRAINING LSTM WITH MLFLOW")
    logger.info(f"{'='*70}")
    
    mlflow.set_experiment("LSTM_AIS_Prediction")
    
    with mlflow.start_run():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {device}")
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)
        
        # DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Model
        model = LSTMModel(input_size=X_train.shape[2]).to(device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # MLflow params
        mlflow.log_params({
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': 0.001,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        })
        
        train_losses, val_losses, train_maes, val_maes = [], [], [], []
        best_val_loss = float('inf')
        
        for epoch in tqdm(range(epochs), desc="Epochs"):
            # Train
            model.train()
            train_loss = 0
            train_preds, train_targets = [], []
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_preds.append(outputs.detach().cpu().numpy())
                train_targets.append(y_batch.detach().cpu().numpy())
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            train_mae = mean_absolute_error(np.vstack(train_targets), np.vstack(train_preds))
            train_maes.append(train_mae)
            
            # Validate
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_mae = mean_absolute_error(y_val_t.cpu().numpy(), val_outputs.cpu().numpy())
            
            val_losses.append(val_loss)
            val_maes.append(val_mae)
            
            # MLflow logging
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mae': train_mae,
                'val_mae': val_mae
            }, step=epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_lstm_model_full.pt')
        
        # Log model
        mlflow.pytorch.log_model(model, "lstm_model")
        
        logger.info(f"✓ Model saved and logged to MLflow")
        
        return model, train_losses, val_losses, train_maes, val_maes


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("FULL PIPELINE: AIS DATA (JAN 3-8) + LSTM + MLFLOW")
    logger.info("="*70)
    
    # Load all data
    df = load_all_data(start_date=3, end_date=8)
    
    # Prepare features
    df, features = prepare_features(df)
    
    # Create sequences
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = create_sequences_per_vessel(df, features)
    
    # Train with MLflow
    model, train_losses, val_losses, train_maes, val_maes = train_model_mlflow(
        X_train, y_train, X_val, y_val, epochs=50, batch_size=32
    )
    
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*70)

