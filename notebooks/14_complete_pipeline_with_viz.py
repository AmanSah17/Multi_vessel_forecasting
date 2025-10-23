"""
Complete Pipeline: Load → Train → Evaluate → Visualize with MLflow

Processes all AIS data (Jan 3-8), trains LSTM, evaluates on test set,
and creates prediction visualizations for 30 random vessels.
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
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM for predicting LAT, LON, SOG, COG - Optimized for GPU memory."""

    def __init__(self, input_size, hidden_size=64, output_size=4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def load_all_data(start_date=3, end_date=8, sample_per_day=None):
    """Load all CSV files from Jan 3-8 with optional sampling."""
    logger.info(f"\n{'='*70}\n[1/7] LOADING DATA (Jan {start_date}-{end_date})\n{'='*70}")

    base_path = Path(r"D:\Maritime_Vessel_monitoring\csv_extracted_data")
    dfs = []

    for day in range(start_date, end_date + 1):
        file_path = base_path / f"AIS_2020_01_{day:02d}" / f"AIS_2020_01_{day:02d}.csv"
        if file_path.exists():
            logger.info(f"Loading {file_path.name}...")
            df = pd.read_csv(file_path, usecols=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'VesselName'])
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], format="%Y-%m-%dT%H:%M:%S", errors='coerce')
            df = df.dropna(subset=['BaseDateTime', 'LAT', 'LON', 'SOG', 'COG'])

            if sample_per_day:
                df = df.sample(n=min(sample_per_day, len(df)), random_state=42)

            dfs.append(df)
            logger.info(f"  ✓ {len(df):,} records")

    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"\nTotal: {len(df_all):,} records, {df_all['MMSI'].nunique():,} vessels")
    return df_all


def prepare_features(df):
    """Add temporal and kinematic features."""
    logger.info(f"\n{'='*70}\n[2/7] PREPARING FEATURES\n{'='*70}")
    
    df = df.sort_values('BaseDateTime').reset_index(drop=True)
    df['hour'] = df['BaseDateTime'].dt.hour
    df['day_of_week'] = df['BaseDateTime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['speed_change'] = df.groupby('MMSI')['SOG'].diff().fillna(0)
    df['heading_change'] = df.groupby('MMSI')['COG'].diff().fillna(0)
    
    features = ['LAT', 'LON', 'SOG', 'COG', 'hour', 'day_of_week', 'speed_change', 'heading_change']
    logger.info(f"Features: {features}")
    return df, features


def create_sequences_per_vessel(df, features, seq_length=30):
    """Create sequences with per-vessel 70/20/10 split."""
    logger.info(f"\n{'='*70}\n[3/7] CREATING SEQUENCES (Per-Vessel 70/20/10 Split)\n{'='*70}")
    
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    vessels = df['MMSI'].unique()
    
    for mmsi in tqdm(vessels, desc="Vessels", unit="vessel"):
        vessel_data = df[df['MMSI'] == mmsi].sort_values('BaseDateTime')[features].values
        
        if len(vessel_data) < seq_length + 1:
            continue
        
        X_vessel, y_vessel = [], []
        for i in range(len(vessel_data) - seq_length):
            X_vessel.append(vessel_data[i:i+seq_length])
            y_vessel.append(vessel_data[i+seq_length, :4])
        
        if len(X_vessel) == 0:
            continue
        
        n = len(X_vessel)
        train_idx, val_idx = int(0.7 * n), int(0.9 * n)
        
        X_train.extend(X_vessel[:train_idx])
        y_train.extend(y_vessel[:train_idx])
        X_val.extend(X_vessel[train_idx:val_idx])
        y_val.extend(y_vessel[train_idx:val_idx])
        X_test.extend(X_vessel[val_idx:])
        y_test.extend(y_vessel[val_idx:])
    
    X_train, y_train = np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)
    X_val, y_val = np.array(X_val, dtype=np.float32), np.array(y_val, dtype=np.float32)
    X_test, y_test = np.array(X_test, dtype=np.float32), np.array(y_test, dtype=np.float32)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train LSTM with MLflow logging."""
    logger.info(f"\n{'='*70}\n[4/7] TRAINING LSTM (50 Epochs)\n{'='*70}")
    
    mlflow.set_experiment("LSTM_AIS_Full_Pipeline")
    
    with mlflow.start_run():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {device}")
        
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)
        
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        
        model = LSTMModel(input_size=X_train.shape[2]).to(device)
        logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        mlflow.log_params({
            'epochs': epochs, 'batch_size': batch_size, 'lr': 0.001,
            'hidden_size': 64, 'num_layers': 1, 'dropout': 0.2,
            'train_samples': len(X_train), 'val_samples': len(X_val)
        })
        
        train_losses, val_losses, train_maes, val_maes = [], [], [], []
        best_val_loss = float('inf')
        
        for epoch in tqdm(range(epochs), desc="Epochs"):
            model.train()
            train_loss, train_preds, train_targets = 0, [], []
            
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
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_mae = mean_absolute_error(y_val_t.cpu().numpy(), val_outputs.cpu().numpy())
            
            val_losses.append(val_loss)
            val_maes.append(val_mae)
            
            mlflow.log_metrics({
                'train_loss': train_loss, 'val_loss': val_loss,
                'train_mae': train_mae, 'val_mae': val_mae
            }, step=epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_lstm_model_full.pt')
        
        mlflow.pytorch.log_model(model, "lstm_model")
        logger.info("✓ Model saved to MLflow")
        
        return model, train_losses, val_losses, train_maes, val_maes, device


def evaluate_and_visualize(model, X_test, y_test, df, features, scaler, device):
    """Evaluate on test set and create visualizations."""
    logger.info(f"\n{'='*70}\n[5/7] EVALUATING ON TEST SET\n{'='*70}")
    
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    with torch.no_grad():
        predictions = model(X_test_t).cpu().numpy()
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    logger.info(f"Test MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.6f}")
    
    output_names = ['LAT', 'LON', 'SOG', 'COG']
    for i, name in enumerate(output_names):
        mae_i = mean_absolute_error(y_test[:, i], predictions[:, i])
        logger.info(f"  {name} MAE: {mae_i:.6f}")
    
    # Plot trajectories for 30 random vessels
    logger.info(f"\n{'='*70}\n[6/7] PLOTTING PREDICTIONS (30 Random Vessels)\n{'='*70}")
    
    vessels = df['MMSI'].unique()
    selected = np.random.choice(vessels, min(30, len(vessels)), replace=False)
    
    n_cols, n_rows = 5, 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 24))
    axes = axes.flatten()
    
    for idx, mmsi in enumerate(tqdm(selected, desc="Plotting")):
        vessel_data = df[df['MMSI'] == mmsi].sort_values('BaseDateTime')[features].values
        
        if len(vessel_data) < 31:
            continue
        
        X_v, y_v = [], []
        for i in range(len(vessel_data) - 30):
            X_v.append(vessel_data[i:i+30])
            y_v.append(vessel_data[i+30, :4])
        
        if len(X_v) == 0:
            continue
        
        X_v = np.array(X_v, dtype=np.float32)
        X_v = scaler.transform(X_v.reshape(-1, X_v.shape[-1])).reshape(X_v.shape)
        y_v = np.array(y_v, dtype=np.float32)
        
        X_v_t = torch.FloatTensor(X_v).to(device)
        with torch.no_grad():
            pred_v = model(X_v_t).cpu().numpy()
        
        ax = axes[idx]
        ax.plot(y_v[:, 1], y_v[:, 0], 'b-', linewidth=2, label='Actual', marker='o', markersize=5)
        ax.plot(pred_v[:, 1], pred_v[:, 0], 'r--', linewidth=2, label='Predicted', marker='s', markersize=5)
        ax.set_xlabel('Longitude', fontsize=9)
        ax.set_ylabel('Latitude', fontsize=9)
        ax.set_title(f'Vessel {mmsi}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions_30_vessels.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Saved: predictions_30_vessels.png")
    
    # Time series plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Predictions vs Actual - Time Series (First 500 Samples)', fontsize=14, fontweight='bold')
    
    for i, (ax, name) in enumerate(zip(axes.flatten(), output_names)):
        ax.plot(y_test[:500, i], 'b-', linewidth=2, label='Actual', marker='o', markersize=3)
        ax.plot(predictions[:500, i], 'r--', linewidth=2, label='Predicted', marker='s', markersize=3)
        ax.set_xlabel('Sample', fontsize=11)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(f'{name} - Actual vs Predicted', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('timeseries_predictions.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Saved: timeseries_predictions.png")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("COMPLETE PIPELINE: AIS (JAN 3-8) → LSTM → EVALUATION → VISUALIZATION")
    logger.info("="*70)

    # Load data with sampling to manage memory (50K per day = 300K total)
    df = load_all_data(start_date=3, end_date=8, sample_per_day=50000)
    df, features = prepare_features(df)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = create_sequences_per_vessel(df, features)
    model, train_losses, val_losses, train_maes, val_maes, device = train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=16)
    metrics = evaluate_and_visualize(model, X_test, y_test, df, features, scaler, device)

    logger.info(f"\n{'='*70}\n[7/7] PIPELINE COMPLETE!\n{'='*70}")
    logger.info(f"Final Metrics - MAE: {metrics['mae']:.6f}, RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.6f}")

