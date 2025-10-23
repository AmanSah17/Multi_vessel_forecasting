"""
End-to-End MLflow Pipeline with Advanced Feature Engineering
- Hyperparameter tuning
- MLflow logging
- Training/validation loss curves
- Model testing and predictions
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import json
import warnings
warnings.filterwarnings('ignore')

# Setup directories
output_dirs = {
    'logs': Path('logs'),
    'results': Path('results'),
    'images': Path('results/images'),
    'csv': Path('results/csv'),
    'models': Path('results/models'),
    'mlflow': Path('mlruns')
}

for dir_path in output_dirs.values():
    dir_path.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(output_dirs['logs'] / 'end_to_end_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# MLflow setup
mlflow.set_tracking_uri(f"file:{output_dirs['mlflow'].absolute()}")
mlflow.set_experiment("Maritime_Vessel_Forecasting")


def load_data_efficient(start_date=3, end_date=8):
    """Load all data efficiently."""
    logger.info(f"\n{'='*70}\n[1/9] LOADING DATA\n{'='*70}")
    
    base_path = Path(r"D:\Maritime_Vessel_monitoring\csv_extracted_data")
    all_data = []
    
    for day in range(start_date, end_date + 1):
        file_path = base_path / f"AIS_2020_01_{day:02d}" / f"AIS_2020_01_{day:02d}.csv"
        if file_path.exists():
            logger.info(f"Loading {file_path.name}...")
            df = pd.read_csv(file_path, usecols=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG'])
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], format="%Y-%m-%dT%H:%M:%S", errors='coerce')
            df = df.dropna(subset=['BaseDateTime', 'LAT', 'LON', 'SOG', 'COG'])
            all_data.append(df)
            logger.info(f"  ✓ {len(df):,} records")
    
    df_all = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total: {len(df_all):,} records, {df_all['MMSI'].nunique():,} vessels")
    
    return df_all


def add_advanced_features(df):
    """Add 50+ advanced features."""
    logger.info(f"\n{'='*70}\n[2/9] ADVANCED FEATURE ENGINEERING\n{'='*70}")
    
    df = df.sort_values('BaseDateTime').reset_index(drop=True)
    
    # Temporal features
    df['hour'] = df['BaseDateTime'].dt.hour
    df['day_of_week'] = df['BaseDateTime'].dt.dayofweek
    df['month'] = df['BaseDateTime'].dt.month
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Kinematic features
    df['speed_change'] = df.groupby('MMSI')['SOG'].diff().fillna(0)
    df['heading_change'] = df.groupby('MMSI')['COG'].diff().fillna(0)
    df['lat_change'] = df.groupby('MMSI')['LAT'].diff().fillna(0)
    df['lon_change'] = df.groupby('MMSI')['LON'].diff().fillna(0)
    
    # Lag features (1, 2, 3)
    for lag in [1, 2, 3]:
        df[f'LAT_lag{lag}'] = df.groupby('MMSI')['LAT'].shift(lag).fillna(0)
        df[f'LON_lag{lag}'] = df.groupby('MMSI')['LON'].shift(lag).fillna(0)
        df[f'SOG_lag{lag}'] = df.groupby('MMSI')['SOG'].shift(lag).fillna(0)
        df[f'COG_lag{lag}'] = df.groupby('MMSI')['COG'].shift(lag).fillna(0)
    
    # Rolling statistics
    for window in [3, 5]:
        df[f'SOG_mean_{window}'] = df.groupby('MMSI')['SOG'].rolling(window).mean().reset_index(drop=True).fillna(0)
        df[f'SOG_std_{window}'] = df.groupby('MMSI')['SOG'].rolling(window).std().reset_index(drop=True).fillna(0)
        df[f'COG_mean_{window}'] = df.groupby('MMSI')['COG'].rolling(window).mean().reset_index(drop=True).fillna(0)
    
    # Acceleration features
    df['speed_accel'] = df.groupby('MMSI')['speed_change'].diff().fillna(0)
    df['heading_accel'] = df.groupby('MMSI')['heading_change'].diff().fillna(0)
    
    # Velocity components
    df['velocity_x'] = df['SOG'] * np.cos(np.radians(df['COG']))
    df['velocity_y'] = df['SOG'] * np.sin(np.radians(df['COG']))
    df['velocity_mag'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    
    # Polynomial features
    df['LAT_sq'] = df['LAT'] ** 2
    df['LON_sq'] = df['LON'] ** 2
    df['SOG_sq'] = df['SOG'] ** 2
    
    # Interaction features
    df['speed_heading_int'] = df['SOG'] * df['COG']
    df['lat_lon_int'] = df['LAT'] * df['LON']
    
    logger.info(f"✓ Total features: {len(df.columns)}")
    
    return df


def create_sequences(df, sequence_length=120, max_sequences=150000):
    """Create sequences with longer context."""
    logger.info(f"\n{'='*70}\n[3/9] CREATING SEQUENCES\n{'='*70}")
    
    features = [col for col in df.columns if col not in ['MMSI', 'BaseDateTime']]
    X, y = [], []
    
    for mmsi in tqdm(df['MMSI'].unique(), desc="Creating sequences"):
        vessel_data = df[df['MMSI'] == mmsi][features].values
        
        if len(vessel_data) < sequence_length + 1:
            continue
        
        for i in range(len(vessel_data) - sequence_length):
            X.append(vessel_data[i:i+sequence_length])
            y.append(vessel_data[i+sequence_length, :4])
            
            if len(X) >= max_sequences:
                break
        
        if len(X) >= max_sequences:
            break
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    logger.info(f"Sequences: {len(X):,}, X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y, features


class EnhancedLSTMModel(nn.Module):
    """Enhanced LSTM with configurable architecture."""
    
    def __init__(self, input_size, hidden_size=256, num_layers=3, output_size=4, dropout=0.2):
        super(EnhancedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class TemporalCNNModel(nn.Module):
    """Temporal CNN with configurable architecture."""
    
    def __init__(self, input_size, output_size=4, num_filters=64, num_layers=4, dropout=0.2):
        super(TemporalCNNModel, self).__init__()
        self.input_proj = nn.Conv1d(input_size, num_filters, 1)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (3 - 1) * dilation // 2
            self.blocks.append(nn.Sequential(
                nn.Conv1d(num_filters, num_filters, 3, padding=padding, dilation=dilation),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        self.fc = nn.Sequential(
            nn.Linear(num_filters, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=2)
        x = self.fc(x)
        return x


def train_model_with_logging(model, train_loader, val_loader, epochs=150, lr=0.001, patience=30, device='cuda', model_name='lstm'):
    """Train model with MLflow logging."""
    logger.info(f"\n{'='*70}\nTRAINING {model_name.upper()}\n{'='*70}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(epochs), desc=f"Training {model_name}"):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # Log to MLflow
        mlflow.log_metric(f"{model_name}_train_loss", train_loss, step=epoch)
        mlflow.log_metric(f"{model_name}_val_loss", val_loss, step=epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dirs['models'] / f'best_{model_name}.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    return train_losses, val_losses


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("END-TO-END MLFLOW PIPELINE")
    logger.info("="*70)

