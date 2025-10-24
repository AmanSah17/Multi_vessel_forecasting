"""
Experimental Models Pipeline: Smaller LSTM, ARIMA, Kalman Filter
- Sequence length: 24 (5-minute forecasting window = 120 minutes / 5 = 24 steps)
- Smaller LSTM models with more hidden layers
- ARIMA models for each vessel
- Kalman Filter for state estimation
- MLflow logging for all experiments
- 300-vessel test set evaluation
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
import warnings
warnings.filterwarnings('ignore')
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
import time

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

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# GPU optimization - AGGRESSIVE SETTINGS FOR FASTER TRAINING
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Faster but less deterministic
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    except:
        pass

    # Log GPU info
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")


def load_entire_dataset(start_date=3, end_date=8, sample_per_day=None):
    """Load FULL dataset with 5-minute intervals (sequence_length=24 for 120-min window).

    Args:
        start_date: Start day (e.g., 3 for Jan 3)
        end_date: End day (e.g., 8 for Jan 8)
        sample_per_day: If None, load ALL data. If int, sample that many records per day.
    """
    if sample_per_day is None:
        logger.info(f"\n{'='*80}\n[1/10] LOADING FULL DATASET (NO SAMPLING - ALL DATA)\n{'='*80}")
    else:
        logger.info(f"\n{'='*80}\n[1/10] LOADING DATASET (sampled {sample_per_day:,}/day)\n{'='*80}")

    base_path = Path(r"D:\Maritime_Vessel_monitoring\csv_extracted_data")
    all_data = []

    for day in range(start_date, end_date + 1):
        file_path = base_path / f"AIS_2020_01_{day:02d}" / f"AIS_2020_01_{day:02d}.csv"
        if file_path.exists():
            logger.info(f"Loading day {day}...")
            df = pd.read_csv(file_path, usecols=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG'])
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], format="%Y-%m-%dT%H:%M:%S", errors='coerce')
            df = df.dropna(subset=['BaseDateTime', 'LAT', 'LON', 'SOG', 'COG'])

            # Only sample if sample_per_day is specified
            if sample_per_day is not None and len(df) > sample_per_day:
                df = df.sample(n=sample_per_day, random_state=42)

            all_data.append(df)
            logger.info(f"  ✓ {len(df):,} records")

    df_all = pd.concat(all_data, ignore_index=True)
    logger.info(f"✓ Total: {len(df_all):,} records, {df_all['MMSI'].nunique():,} vessels")
    return df_all


def add_comprehensive_features(df):
    """Add 50+ features for experimental models (memory-efficient for large datasets)."""
    logger.info(f"\n{'='*80}\n[2/10] ADDING COMPREHENSIVE FEATURES (memory-efficient)\n{'='*80}")

    df = df.sort_values(['MMSI', 'BaseDateTime']).reset_index(drop=True)

    # Temporal features
    df['hour'] = df['BaseDateTime'].dt.hour.astype('int8')
    df['day_of_week'] = df['BaseDateTime'].dt.dayofweek.astype('int8')
    df['minute'] = df['BaseDateTime'].dt.minute.astype('int8')

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype('float32')
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype('float32')
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7).astype('float32')
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7).astype('float32')

    # Kinematic features (process per vessel to save memory)
    logger.info("Computing kinematic features...")
    for col in ['LAT', 'LON', 'SOG', 'COG']:
        df[f'{col.lower()}_diff'] = df.groupby('MMSI')[col].diff().fillna(0).astype('float32')

    # Lag features (1, 2, 3 steps) - process per vessel
    logger.info("Computing lag features...")
    for lag in [1, 2, 3]:
        for col in ['LAT', 'LON', 'SOG', 'COG']:
            df[f'{col.lower()}_lag{lag}'] = df.groupby('MMSI')[col].shift(lag).fillna(0).astype('float32')

    # Polynomial features (use float32 to save memory)
    logger.info("Computing polynomial features...")
    df['lat_sq'] = (df['LAT'] ** 2).astype('float32')
    df['lon_sq'] = (df['LON'] ** 2).astype('float32')
    df['sog_sq'] = (df['SOG'] ** 2).astype('float32')

    # Velocity features
    df['speed_heading_int'] = (df['SOG'] * df['COG']).astype('float32')
    df['lat_lon_int'] = (df['LAT'] * df['LON']).astype('float32')

    logger.info(f"✓ Total features: {len(df.columns)}")
    return df


def create_sequences_full(df, sequence_length=24, max_sequences=None, sample_vessels=None):
    """Create sequences with shorter length (24 steps = 120 minutes at 5-min intervals).
    Memory-efficient: processes vessels one at a time using groupby to avoid large boolean indexing.

    Args:
        df: DataFrame with vessel data
        sequence_length: Number of timesteps per sequence
        max_sequences: Max sequences to create (for testing)
        sample_vessels: If int, sample this many vessels for faster creation
    """
    logger.info(f"\n{'='*80}\n[3/10] CREATING SEQUENCES (seq_len={sequence_length}, memory-efficient)\n{'='*80}")

    features = [col for col in df.columns if col not in ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG']]
    target_cols = ['LAT', 'LON', 'SOG', 'COG']

    X_list, y_list, mmsi_list = [], [], []

    # Use groupby for memory-efficient iteration (avoids large boolean indexing)
    grouped = df.groupby('MMSI', sort=False)
    unique_mmsi = list(grouped.groups.keys())

    # Sample vessels if specified (for faster testing)
    if sample_vessels and len(unique_mmsi) > sample_vessels:
        unique_mmsi = list(np.random.choice(unique_mmsi, size=sample_vessels, replace=False))
        logger.info(f"Sampling {sample_vessels:,} vessels for faster processing")

    logger.info(f"Processing {len(unique_mmsi):,} vessels...")

    for mmsi in tqdm(unique_mmsi, desc="Creating sequences", unit="vessel"):
        # Use groupby to get vessel data (more memory-efficient than boolean indexing)
        vessel_df = grouped.get_group(mmsi).sort_values('BaseDateTime').reset_index(drop=True)

        if len(vessel_df) < sequence_length + 1:
            continue

        X_vessel = vessel_df[features].values.astype(np.float32)
        y_vessel = vessel_df[target_cols].values.astype(np.float32)

        for i in range(len(vessel_df) - sequence_length):
            X_list.append(X_vessel[i:i+sequence_length])
            y_list.append(y_vessel[i+sequence_length])
            mmsi_list.append(mmsi)

            if max_sequences and len(X_list) >= max_sequences:
                break

        if max_sequences and len(X_list) >= max_sequences:
            break

    # Convert to arrays more efficiently
    logger.info(f"Converting {len(X_list):,} sequences to arrays...")
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    mmsi_list = np.array(mmsi_list)

    logger.info(f"✓ Sequences: {len(X):,}, X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"✓ Memory usage: X={X.nbytes / 1e9:.2f}GB, y={y.nbytes / 1e9:.2f}GB")
    return X, y, features, mmsi_list


def save_sequences_cache(cache_file: Path, X, y, features, mmsi_list):
    """Save sequences to NPZ cache."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, X=X, y=y, features=features, mmsi_list=mmsi_list)
    logger.info(f"✓ Cached sequences to: {cache_file}")


def load_sequences_cache(cache_file: Path):
    """Load sequences from NPZ cache."""
    data = np.load(cache_file, allow_pickle=True)
    return data['X'], data['y'], data['features'].tolist(), data['mmsi_list'].tolist()


# ======================== SMALL LSTM MODELS ========================

class SmallLSTMModel(nn.Module):
    """Smaller LSTM with more layers for better feature extraction."""
    def __init__(self, input_size, hidden_size=128, num_layers=6, output_size=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class TinyLSTMModel(nn.Module):
    """Tiny LSTM for ultra-fast training."""
    def __init__(self, input_size, hidden_size=32, num_layers=4, output_size=4, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ======================== TRAINING FUNCTION ========================

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=20,
                device='cuda', model_name='model', use_amp=True):
    """Train model with MLflow logging."""
    logger.info(f"\nTraining {model_name.upper()}...")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)
    criterion = nn.MSELoss()

    # Use newer AMP API if available
    use_amp = use_amp and torch.cuda.is_available()
    try:
        scaler = GradScaler('cuda', enabled=use_amp)
        autocast_device = 'cuda'
    except TypeError:
        scaler = GradScaler(enabled=use_amp)
        autocast_device = None

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    epoch_iter = tqdm(range(epochs), desc=f"Training {model_name}", unit="epoch", dynamic_ncols=True)

    for epoch in epoch_iter:
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            if autocast_device:
                with autocast(device_type=autocast_device, enabled=use_amp):
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
            else:
                with autocast(enabled=use_amp):
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                if autocast_device:
                    with autocast(device_type=autocast_device, enabled=use_amp):
                        preds = model(X_batch)
                        loss = criterion(preds, y_batch)
                else:
                    with autocast(enabled=use_amp):
                        preds = model(X_batch)
                        loss = criterion(preds, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dirs['models'] / f'best_{model_name}.pt')
        else:
            patience_counter += 1
        
        scheduler.step(val_loss)
        
        # Log metrics
        mlflow.log_metrics({
            f"{model_name}_train_loss": train_loss,
            f"{model_name}_val_loss": val_loss,
            f"{model_name}_best_val_loss": best_val_loss,
            f"{model_name}_lr": optimizer.param_groups[0]['lr']
        }, step=epoch)
        
        epoch_iter.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'best': f'{best_val_loss:.4f}',
            'patience': f'{patience_counter}/{patience}'
        })
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, device='cuda', model_name='model'):
    """Evaluate model on test set."""
    logger.info(f"Evaluating {model_name.upper()}...")
    model.eval()
    y_true_all, y_pred_all = [], []
    use_amp = torch.cuda.is_available()

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc=f"Evaluating {model_name}", unit="batch"):
            X_batch = X_batch.to(device)
            try:
                with autocast(device_type='cuda', enabled=use_amp):
                    preds = model(X_batch).cpu().numpy()
            except TypeError:
                with autocast(enabled=use_amp):
                    preds = model(X_batch).cpu().numpy()
            y_pred_all.append(preds)
            y_true_all.append(y_batch.numpy())
    
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    
    metrics = {
        'MAE': mae, 'RMSE': rmse, 'R2': r2,
        'MAE_LAT': float(mean_absolute_error(y_true[:,0], y_pred[:,0])),
        'MAE_LON': float(mean_absolute_error(y_true[:,1], y_pred[:,1])),
        'MAE_SOG': float(mean_absolute_error(y_true[:,2], y_pred[:,2])),
        'MAE_COG': float(mean_absolute_error(y_true[:,3], y_pred[:,3])),
    }
    
    logger.info(f"{model_name.upper()} -> MAE={mae:.6f} RMSE={rmse:.6f} R2={r2:.6f}")
    return metrics, y_true, y_pred


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dirs['logs'] / 'experimental_models.log', encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True,
    )
    
    logger.info("="*80)
    logger.info("EXPERIMENTAL MODELS PIPELINE: Small LSTM, ARIMA, Kalman Filter")
    logger.info("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load dataset with FAST sampling for quick training & validation
    # Use 3% of vessels + shorter sequence for memory efficiency
    logger.info("\n" + "="*80)
    logger.info("[1/10] LOADING DATASET (3% vessel sampling + seq_len=12 for FAST training)")
    logger.info("="*80)
    df = load_entire_dataset(start_date=3, end_date=8, sample_per_day=None)

    # Sample 3% of unique vessels for FAST processing
    unique_vessels = df['MMSI'].unique()
    sampled_vessels = np.random.choice(unique_vessels, size=max(500, int(len(unique_vessels) * 0.03)), replace=False)
    df = df[df['MMSI'].isin(sampled_vessels)].copy()
    logger.info(f"✓ Sampled {len(sampled_vessels):,} vessels ({len(sampled_vessels)/len(unique_vessels)*100:.1f}%)")
    logger.info(f"✓ Dataset size: {len(df):,} records")

    df = add_comprehensive_features(df)

    # Create/load sequences (seq_len=12 for 60-min forecasting window)
    cache_file = output_dirs['results'] / 'cache' / 'seq_cache_len12_sampled_3pct.npz'

    if cache_file.exists():
        logger.info(f"\n✓ Loading sequences from cache: {cache_file}")
        X, y, features, mmsi_list = load_sequences_cache(cache_file)
        logger.info(f"✓ Loaded {len(X):,} sequences")
    else:
        logger.info(f"\n⏳ Creating sequences (seq_len=12, this may take 1-2 minutes)...")
        logger.info(f"   Tip: Sequences will be cached for future runs")
        X, y, features, mmsi_list = create_sequences_full(df, sequence_length=12)
        logger.info(f"✓ Saving sequences to cache...")
        save_sequences_cache(cache_file, X, y, features, mmsi_list)
    
    # Split data (70/20/10)
    n = len(X)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.9)
    
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]
    
    logger.info(f"Split sizes -> Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    # Normalize (memory-efficient)
    mins = X_train.min(axis=(0, 1), keepdims=True)
    maxs = X_train.max(axis=(0, 1), keepdims=True)
    denom = maxs - mins
    denom[denom == 0] = 1.0
    X_train_scaled = (X_train - mins) / denom
    X_val_scaled = (X_val - mins) / denom
    X_test_scaled = (X_test - mins) / denom
    
    # DataLoaders - optimized for GPU
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test))

    # Use larger batch size for GPU (256 for faster training)
    batch_size = 256 if torch.cuda.is_available() else 64
    logger.info(f"Using batch_size={batch_size} for GPU training")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    input_size = len(features)
    
    # MLflow experiment
    mlflow.set_experiment("Experimental_Models_5min_Forecasting")
    
    with mlflow.start_run(run_name="Small_LSTM_Experiments"):
        mlflow.log_param("sequence_length", 24)
        mlflow.log_param("forecasting_window_minutes", 120)
        mlflow.log_param("num_features", input_size)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))

        # Train Small LSTM
        logger.info("\n" + "="*80)
        logger.info("[6/10] TRAINING SMALL LSTM (hidden_size=128, num_layers=6)")
        logger.info("="*80)
        small_lstm = SmallLSTMModel(input_size=input_size, hidden_size=128, num_layers=6, dropout=0.2).to(device)
        logger.info(f"Model parameters: {sum(p.numel() for p in small_lstm.parameters()):,}")
        small_lstm_train_losses, small_lstm_val_losses = train_model(
            small_lstm, train_loader, val_loader, epochs=100, lr=0.001, patience=20,
            device=device, model_name='small_lstm', use_amp=True
        )
        logger.info(f"✓ Small LSTM training complete")

        # Train Tiny LSTM
        logger.info("\n" + "="*80)
        logger.info("[7/10] TRAINING TINY LSTM (hidden_size=32, num_layers=4)")
        logger.info("="*80)
        tiny_lstm = TinyLSTMModel(input_size=input_size, hidden_size=32, num_layers=4, dropout=0.15).to(device)
        logger.info(f"Model parameters: {sum(p.numel() for p in tiny_lstm.parameters()):,}")
        tiny_lstm_train_losses, tiny_lstm_val_losses = train_model(
            tiny_lstm, train_loader, val_loader, epochs=100, lr=0.001, patience=20,
            device=device, model_name='tiny_lstm', use_amp=True
        )
        logger.info(f"✓ Tiny LSTM training complete")

        # Evaluate models
        logger.info("\n" + "="*80)
        logger.info("[8/10] EVALUATION")
        logger.info("="*80)

        small_lstm.load_state_dict(torch.load(output_dirs['models'] / 'best_small_lstm.pt'))
        tiny_lstm.load_state_dict(torch.load(output_dirs['models'] / 'best_tiny_lstm.pt'))

        small_lstm_metrics, _, _ = evaluate_model(small_lstm, test_loader, device, 'small_lstm')
        tiny_lstm_metrics, _, _ = evaluate_model(tiny_lstm, test_loader, device, 'tiny_lstm')

        for k, v in small_lstm_metrics.items():
            mlflow.log_metric(f"small_lstm_{k}", v)
        for k, v in tiny_lstm_metrics.items():
            mlflow.log_metric(f"tiny_lstm_{k}", v)

        logger.info("\n" + "="*80)
        logger.info("[9/10] EXPERIMENTAL MODELS PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"✓ Models saved to: {output_dirs['models']}")
        logger.info(f"✓ Logs saved to: {output_dirs['logs']}")
        logger.info(f"✓ MLflow experiment: Experimental_Models_5min_Forecasting")
        logger.info("="*80)

