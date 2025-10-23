"""
Memory-Efficient LSTM vs CNN Pipeline
- Processes data efficiently
- Reduced memory footprint
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
import gc
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
        logging.FileHandler(output_dirs['logs'] / 'efficient_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_data_per_vessel(start_date=3, end_date=8):
    """Load data per vessel to reduce memory."""
    logger.info(f"\n{'='*70}\n[1/7] LOADING DATA\n{'='*70}")

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


def add_features_efficient(df):
    """Add features efficiently."""
    logger.info(f"\n{'='*70}\n[2/7] ADDING FEATURES\n{'='*70}")

    df = df.sort_values('BaseDateTime').reset_index(drop=True)

    # Temporal features
    df['hour'] = df['BaseDateTime'].dt.hour
    df['day_of_week'] = df['BaseDateTime'].dt.dayofweek

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Kinematic features
    df['speed_change'] = df.groupby('MMSI')['SOG'].diff().fillna(0)
    df['heading_change'] = df.groupby('MMSI')['COG'].diff().fillna(0)

    # Lag features
    df['LAT_lag1'] = df.groupby('MMSI')['LAT'].shift(1).fillna(0)
    df['LON_lag1'] = df.groupby('MMSI')['LON'].shift(1).fillna(0)
    df['SOG_lag1'] = df.groupby('MMSI')['SOG'].shift(1).fillna(0)

    # Velocity components
    df['velocity_x'] = df['SOG'] * np.cos(np.radians(df['COG']))
    df['velocity_y'] = df['SOG'] * np.sin(np.radians(df['COG']))

    logger.info(f"✓ Features added. Total: {len(df.columns)}")

    return df


def create_sequences_efficient(df, sequence_length=60, max_sequences=100000):
    """Create sequences efficiently."""
    logger.info(f"\n{'='*70}\n[3/7] CREATING SEQUENCES\n{'='*70}")

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

    logger.info(f"Sequences created: {len(X):,}")

    return X, y, features


class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, output_size=4, dropout=0.2):
        super(EnhancedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
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


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=20, device='cuda', model_name='lstm'):
    """Train model."""
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


def evaluate_model(model, test_loader, device='cuda', model_name='lstm'):
    """Evaluate model."""
    logger.info(f"\n{'='*70}\nEVALUATING {model_name.upper()}\n{'='*70}")

    model.eval()
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            y_true_all.append(y_batch.numpy())
            y_pred_all.append(y_pred)

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAE_per_output': {
            'LAT': mean_absolute_error(y_true[:, 0], y_pred[:, 0]),
            'LON': mean_absolute_error(y_true[:, 1], y_pred[:, 1]),
            'SOG': mean_absolute_error(y_true[:, 2], y_pred[:, 2]),
            'COG': mean_absolute_error(y_true[:, 3], y_pred[:, 3])
        }
    }

    logger.info(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.6f}")

    return metrics, y_true, y_pred


def plot_training_curves(lstm_train_losses, lstm_val_losses, cnn_train_losses, cnn_val_losses):
    """Plot training curves."""
    logger.info(f"\n{'='*70}\nGENERATING VISUALIZATIONS\n{'='*70}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # LSTM training curves
    axes[0, 0].plot(lstm_train_losses, label='Train Loss', linewidth=2)
    axes[0, 0].plot(lstm_val_losses, label='Val Loss', linewidth=2)
    axes[0, 0].set_title('LSTM Training Curves', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # CNN training curves
    axes[0, 1].plot(cnn_train_losses, label='Train Loss', linewidth=2)
    axes[0, 1].plot(cnn_val_losses, label='Val Loss', linewidth=2)
    axes[0, 1].set_title('CNN Training Curves', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss (MSE)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Comparison
    axes[1, 0].plot(lstm_val_losses, label='LSTM Val Loss', linewidth=2, marker='o', markersize=4)
    axes[1, 0].plot(cnn_val_losses, label='CNN Val Loss', linewidth=2, marker='s', markersize=4)
    axes[1, 0].set_title('Model Comparison - Validation Loss', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss (MSE)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Training time comparison
    axes[1, 1].text(0.5, 0.7, 'LSTM Training Summary', ha='center', fontsize=12, fontweight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.5, f'Epochs: {len(lstm_train_losses)}\nFinal Val Loss: {lstm_val_losses[-1]:.6f}',
                    ha='center', fontsize=11, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.25, 'CNN Training Summary', ha='center', fontsize=12, fontweight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.05, f'Epochs: {len(cnn_train_losses)}\nFinal Val Loss: {cnn_val_losses[-1]:.6f}',
                    ha='center', fontsize=11, transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_dirs['images'] / 'training_curves_comparison.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Saved: training_curves_comparison.png")
    plt.close()


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("MEMORY-EFFICIENT LSTM vs CNN PIPELINE")
    logger.info("="*70)

    # Load data
    df = load_data_per_vessel(start_date=3, end_date=8)

    # Add features
    df = add_features_efficient(df)

    # Create sequences
    X, y, features = create_sequences_efficient(df, sequence_length=60, max_sequences=100000)

    # Split data
    logger.info(f"\n{'='*70}\n[4/7] SPLITTING DATA\n{'='*70}")
    n = len(X)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.9)

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]

    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    # Normalize
    logger.info(f"\n{'='*70}\n[5/7] NORMALIZING DATA\n{'='*70}")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Create dataloaders
    logger.info(f"\n{'='*70}\n[6/7] CREATING DATALOADERS\n{'='*70}")
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Train models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    lstm_model = EnhancedLSTMModel(input_size=len(features), hidden_size=256, num_layers=3, dropout=0.2).to(device)
    lstm_train_losses, lstm_val_losses = train_model(lstm_model, train_loader, val_loader, epochs=100, device=device, model_name='lstm')

    cnn_model = TemporalCNNModel(input_size=len(features), num_filters=64, num_layers=4, dropout=0.2).to(device)
    cnn_train_losses, cnn_val_losses = train_model(cnn_model, train_loader, val_loader, epochs=100, device=device, model_name='cnn')

    # Evaluate
    logger.info(f"\n{'='*70}\n[7/7] EVALUATION\n{'='*70}")

    lstm_model.load_state_dict(torch.load(output_dirs['models'] / 'best_lstm.pt'))
    cnn_model.load_state_dict(torch.load(output_dirs['models'] / 'best_cnn.pt'))

    lstm_metrics, _, _ = evaluate_model(lstm_model, test_loader, device, 'lstm')
    cnn_metrics, _, _ = evaluate_model(cnn_model, test_loader, device, 'cnn')

    # Save results
    comparison_df = pd.DataFrame([
        {'Model': 'LSTM', 'MAE': lstm_metrics['MAE'], 'RMSE': lstm_metrics['RMSE'], 'R2': lstm_metrics['R2']},
        {'Model': 'CNN', 'MAE': cnn_metrics['MAE'], 'RMSE': cnn_metrics['RMSE'], 'R2': cnn_metrics['R2']}
    ])
    comparison_df.to_csv(output_dirs['csv'] / 'model_comparison.csv', index=False)

    # Generate visualizations
    plot_training_curves(lstm_train_losses, lstm_val_losses, cnn_train_losses, cnn_val_losses)

    logger.info(f"\n{'='*70}\nPIPELINE COMPLETE!\n{'='*70}")
    logger.info(f"Results saved in: {output_dirs['results']}")
    logger.info(f"\nModels:")
    logger.info(f"  - LSTM: MAE={lstm_metrics['MAE']:.6f}, RMSE={lstm_metrics['RMSE']:.6f}, R²={lstm_metrics['R2']:.6f}")
    logger.info(f"  - CNN:  MAE={cnn_metrics['MAE']:.6f}, RMSE={cnn_metrics['RMSE']:.6f}, R²={cnn_metrics['R2']:.6f}")
