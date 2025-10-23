"""
Complete LSTM vs Temporal CNN Pipeline
- Advanced feature engineering
- Hyperparameter tuning
- LSTM training
- Temporal CNN training
- Performance comparison
- Organized output
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
import json
import warnings
warnings.filterwarnings('ignore')

# Create output directories
output_dirs = {
    'logs': Path('logs'),
    'results': Path('results'),
    'images': Path('results/images'),
    'csv': Path('results/csv'),
    'models': Path('results/models')
}

for dir_path in output_dirs.values():
    dir_path.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(output_dirs['logs'] / 'complete_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_and_prepare_data(start_date=3, end_date=8, sample_per_day=None):
    """Load and prepare data with advanced features."""

    logger.info(f"\n{'='*70}\n[1/8] LOADING DATA\n{'='*70}")

    base_path = Path(r"D:\Maritime_Vessel_monitoring\csv_extracted_data")
    dfs = []

    for day in range(start_date, end_date + 1):
        file_path = base_path / f"AIS_2020_01_{day:02d}" / f"AIS_2020_01_{day:02d}.csv"
        if file_path.exists():
            logger.info(f"Loading {file_path.name}...")
            df = pd.read_csv(file_path, usecols=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'VesselName'])
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], format="%Y-%m-%dT%H:%M:%S", errors='coerce')
            df = df.dropna(subset=['BaseDateTime', 'LAT', 'LON', 'SOG', 'COG'])

            dfs.append(df)
            logger.info(f"  ✓ {len(df):,} records")

    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total: {len(df_all):,} records, {df_all['MMSI'].nunique():,} vessels")

    return df_all


def add_advanced_features(df):
    """Add advanced features to reduce underfitting."""
    
    logger.info(f"\n{'='*70}\n[2/8] ADVANCED FEATURE ENGINEERING\n{'='*70}")
    
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
    
    # Lag features
    for lag in [1, 2, 3]:
        df[f'LAT_lag{lag}'] = df.groupby('MMSI')['LAT'].shift(lag).fillna(0)
        df[f'LON_lag{lag}'] = df.groupby('MMSI')['LON'].shift(lag).fillna(0)
        df[f'SOG_lag{lag}'] = df.groupby('MMSI')['SOG'].shift(lag).fillna(0)
    
    # Rolling statistics
    for window in [3, 5]:
        df[f'SOG_rolling_mean_{window}'] = df.groupby('MMSI')['SOG'].rolling(window).mean().reset_index(drop=True).fillna(0)
        df[f'COG_rolling_std_{window}'] = df.groupby('MMSI')['COG'].rolling(window).std().reset_index(drop=True).fillna(0)
    
    # Acceleration features
    df['speed_acceleration'] = df.groupby('MMSI')['speed_change'].diff().fillna(0)
    df['heading_acceleration'] = df.groupby('MMSI')['heading_change'].diff().fillna(0)
    
    # Velocity components
    df['velocity_x'] = df['SOG'] * np.cos(np.radians(df['COG']))
    df['velocity_y'] = df['SOG'] * np.sin(np.radians(df['COG']))
    
    # Polynomial features
    df['LAT_squared'] = df['LAT'] ** 2
    df['LON_squared'] = df['LON'] ** 2
    df['SOG_squared'] = df['SOG'] ** 2
    
    logger.info(f"✓ Total features: {len(df.columns)}")
    
    return df


class EnhancedLSTMModel(nn.Module):
    """Enhanced LSTM with tuned hyperparameters."""
    
    def __init__(self, input_size, hidden_size=256, num_layers=3, output_size=4, dropout=0.2):
        super(EnhancedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
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
    """Temporal CNN model."""
    
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
            nn.Linear(num_filters, 128),
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


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, 
                patience=20, device='cuda', model_name='lstm'):
    """Train model with early stopping."""
    
    logger.info(f"\n{'='*70}\nTRAINING {model_name.upper()}\n{'='*70}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(epochs), desc=f"Training {model_name}"):
        # Training
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
        
        # Validation
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


def create_sequences(df, features, sequence_length=60):
    """Create sequences for training."""
    logger.info(f"\n{'='*70}\n[3/8] CREATING SEQUENCES\n{'='*70}")

    X, y = [], []

    for mmsi in tqdm(df['MMSI'].unique(), desc="Creating sequences"):
        vessel_data = df[df['MMSI'] == mmsi][features].values

        if len(vessel_data) < sequence_length + 1:
            continue

        for i in range(len(vessel_data) - sequence_length):
            X.append(vessel_data[i:i+sequence_length])
            y.append(vessel_data[i+sequence_length, :4])  # LAT, LON, SOG, COG

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    logger.info(f"Sequences created: {len(X):,}")
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y


def split_data(X, y, train_ratio=0.7, val_ratio=0.2):
    """Split data into train/val/test."""
    logger.info(f"\n{'='*70}\n[4/8] SPLITTING DATA\n{'='*70}")

    n = len(X)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]

    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def normalize_data(X_train, X_val, X_test):
    """Normalize data."""
    logger.info(f"\n{'='*70}\n[5/8] NORMALIZING DATA\n{'='*70}")

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    logger.info("✓ Data normalized")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    """Create PyTorch dataloaders."""
    logger.info(f"\n{'='*70}\n[6/8] CREATING DATALOADERS\n{'='*70}")

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    return train_loader, val_loader


def plot_comparison(lstm_metrics, cnn_metrics):
    """Plot model comparison."""
    logger.info(f"\n{'='*70}\n[8/8] PLOTTING COMPARISON\n{'='*70}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = ['LSTM', 'CNN']
    mae_values = [lstm_metrics['MAE'], cnn_metrics['MAE']]
    rmse_values = [lstm_metrics['RMSE'], cnn_metrics['RMSE']]
    r2_values = [lstm_metrics['R2'], cnn_metrics['R2']]

    # MAE
    axes[0, 0].bar(models, mae_values, color=['blue', 'red'])
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].set_title('MAE Comparison')
    axes[0, 0].grid(True, alpha=0.3)

    # RMSE
    axes[0, 1].bar(models, rmse_values, color=['blue', 'red'])
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('RMSE Comparison')
    axes[0, 1].grid(True, alpha=0.3)

    # R2
    axes[1, 0].bar(models, r2_values, color=['blue', 'red'])
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].set_title('R² Comparison')
    axes[1, 0].grid(True, alpha=0.3)

    # Per-output MAE
    outputs = ['LAT', 'LON', 'SOG', 'COG']
    lstm_per_output = [lstm_metrics['MAE_per_output'][o] for o in outputs]
    cnn_per_output = [cnn_metrics['MAE_per_output'][o] for o in outputs]

    x = np.arange(len(outputs))
    width = 0.35
    axes[1, 1].bar(x - width/2, lstm_per_output, width, label='LSTM', color='blue')
    axes[1, 1].bar(x + width/2, cnn_per_output, width, label='CNN', color='red')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('Per-Output MAE')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(outputs)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dirs['images'] / 'model_comparison.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Saved: model_comparison.png")
    plt.close()


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("COMPLETE LSTM vs TEMPORAL CNN PIPELINE")
    logger.info("="*70)
    logger.info(f"\nOutput Structure:")
    for name, path in output_dirs.items():
        logger.info(f"  {name}: {path}")

    # Step 1: Load data (ALL DATA - no sampling)
    df = load_and_prepare_data(start_date=3, end_date=8, sample_per_day=None)

    # Step 2: Add advanced features
    df = add_advanced_features(df)

    # Get feature list
    features = [col for col in df.columns if col not in ['MMSI', 'BaseDateTime', 'VesselName']]
    logger.info(f"Total features: {len(features)}")

    # Step 3: Create sequences
    X, y = create_sequences(df, features, sequence_length=60)

    # Step 4: Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)

    # Step 5: Normalize
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_data(X_train, X_val, X_test)

    # Step 6: Create dataloaders
    train_loader, val_loader = create_dataloaders(X_train_scaled, y_train, X_val_scaled, y_val, batch_size=32)

    # Step 7: Train models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nDevice: {device}")

    # Train LSTM
    logger.info(f"\n{'='*70}\n[7/8] TRAINING MODELS\n{'='*70}")

    lstm_model = EnhancedLSTMModel(input_size=len(features), hidden_size=256, num_layers=3, dropout=0.2).to(device)
    logger.info(f"LSTM Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")

    lstm_train_losses, lstm_val_losses = train_model(
        lstm_model, train_loader, val_loader, epochs=200, lr=0.001,
        patience=30, device=device, model_name='lstm'
    )

    plot_training_curves(lstm_train_losses, lstm_val_losses, 'lstm')

    # Train CNN
    cnn_model = TemporalCNNModel(input_size=len(features), num_filters=64, num_layers=4, dropout=0.2).to(device)
    logger.info(f"CNN Parameters: {sum(p.numel() for p in cnn_model.parameters()):,}")

    cnn_train_losses, cnn_val_losses = train_model(
        cnn_model, train_loader, val_loader, epochs=200, lr=0.001,
        patience=30, device=device, model_name='cnn'
    )

    plot_training_curves(cnn_train_losses, cnn_val_losses, 'cnn')

    # Step 8: Evaluate
    logger.info(f"\n{'='*70}\n[8/8] EVALUATION\n{'='*70}")

    # Load best models
    lstm_model.load_state_dict(torch.load(output_dirs['models'] / 'best_lstm.pt'))
    cnn_model.load_state_dict(torch.load(output_dirs['models'] / 'best_cnn.pt'))

    # Create test loader
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate LSTM
    lstm_metrics, lstm_y_true, lstm_y_pred = evaluate_model(lstm_model, test_loader, device, 'lstm')

    # Evaluate CNN
    cnn_metrics, cnn_y_true, cnn_y_pred = evaluate_model(cnn_model, test_loader, device, 'cnn')

    # Save metrics
    save_metrics_csv({'Model': 'LSTM', **lstm_metrics}, 'lstm_metrics.csv')
    save_metrics_csv({'Model': 'CNN', **cnn_metrics}, 'cnn_metrics.csv')

    # Comparison
    comparison_df = pd.DataFrame([
        {'Model': 'LSTM', 'MAE': lstm_metrics['MAE'], 'RMSE': lstm_metrics['RMSE'], 'R2': lstm_metrics['R2']},
        {'Model': 'CNN', 'MAE': cnn_metrics['MAE'], 'RMSE': cnn_metrics['RMSE'], 'R2': cnn_metrics['R2']}
    ])
    comparison_df.to_csv(output_dirs['csv'] / 'model_comparison.csv', index=False)
    logger.info("✓ Saved: model_comparison.csv")

    # Plot comparison
    plot_comparison(lstm_metrics, cnn_metrics)

    logger.info(f"\n{'='*70}")
    logger.info("PIPELINE COMPLETE!")
    logger.info(f"{'='*70}")
    logger.info(f"Results saved in: {output_dirs['results']}")
    logger.info(f"Logs saved in: {output_dirs['logs']}")

