"""
Comprehensive Multi-Model Training Pipeline
- Trains LSTM, CNN, GRU, Transformer on entire dataset
- Generates 300-vessel predictions with detailed metrics
- Creates comprehensive visualizations (LAT, LON, SOG, COG)
- MLflow logging for all models
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
import time
warnings.filterwarnings('ignore')
from torch.cuda.amp import autocast, GradScaler

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
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

import os
import platform


# Enable fast backends when available (may speed up first epochs)
try:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
except Exception as _e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Fast backend toggles not applied: {_e}")


def load_entire_dataset(start_date=3, end_date=8, sample_per_day=100000):
    """Load entire dataset from all available days with sampling - OPTIMIZED FOR SPEED."""
    logger.info(f"\n{'='*80}\n[1/10] LOADING ENTIRE DATASET (sampled - FAST MODE)\n{'='*80}")

    base_path = Path(r"D:\Maritime_Vessel_monitoring\csv_extracted_data")
    all_data = []

    for day in range(start_date, end_date + 1):
        file_path = base_path / f"AIS_2020_01_{day:02d}" / f"AIS_2020_01_{day:02d}.csv"
        if file_path.exists():
            logger.info(f"Loading day {day}...")
            df = pd.read_csv(file_path, usecols=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG'])
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], format="%Y-%m-%dT%H:%M:%S", errors='coerce')
            df = df.dropna(subset=['BaseDateTime', 'LAT', 'LON', 'SOG', 'COG'])

            # Sample to reduce memory - OPTIMIZED: 100K per day (was 200K)
            if len(df) > sample_per_day:
                df = df.sample(n=sample_per_day, random_state=42)

            all_data.append(df)
            logger.info(f"  ✓ {len(df):,} records")

    df_all = pd.concat(all_data, ignore_index=True)
    logger.info(f"✓ Total: {len(df_all):,} records, {df_all['MMSI'].nunique():,} vessels")
    return df_all


def add_comprehensive_features(df):
    """Add 50+ comprehensive features (memory-efficient)."""
    logger.info(f"\n{'='*80}\n[2/10] ADDING COMPREHENSIVE FEATURES (50+)\n{'='*80}")

    df = df.sort_values('BaseDateTime').reset_index(drop=True)

    # Temporal features
    df['hour'] = df['BaseDateTime'].dt.hour
    df['day_of_week'] = df['BaseDateTime'].dt.dayofweek
    df['month'] = df['BaseDateTime'].dt.month
    df['day'] = df['BaseDateTime'].dt.day

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

    # Lag features (1, 2, 3 timesteps)
    for lag in [1, 2, 3]:
        df[f'LAT_lag{lag}'] = df.groupby('MMSI')['LAT'].shift(lag).fillna(0)
        df[f'LON_lag{lag}'] = df.groupby('MMSI')['LON'].shift(lag).fillna(0)
        df[f'SOG_lag{lag}'] = df.groupby('MMSI')['SOG'].shift(lag).fillna(0)
        df[f'COG_lag{lag}'] = df.groupby('MMSI')['COG'].shift(lag).fillna(0)

    # Polynomial features
    df['LAT_squared'] = df['LAT'] ** 2
    df['LON_squared'] = df['LON'] ** 2
    df['SOG_squared'] = df['SOG'] ** 2
    df['COG_squared'] = df['COG'] ** 2

    # Velocity components (compute in float32 to save memory)
    cog_rad = np.radians(df['COG'].values.astype(np.float32))
    sog_vals = df['SOG'].values.astype(np.float32)
    df['velocity_x'] = (sog_vals * np.cos(cog_rad)).astype(np.float32)
    df['velocity_y'] = (sog_vals * np.sin(cog_rad)).astype(np.float32)
    df['velocity_mag'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)

    # Interaction features
    df['speed_heading_interaction'] = df['SOG'] * df['COG']
    df['lat_lon_interaction'] = df['LAT'] * df['LON']

    logger.info(f"✓ Features added. Total: {len(df.columns)} columns")
    return df


def create_sequences_full(df, sequence_length=60, max_sequences=None):
    """Create sequences from entire dataset - OPTIMIZED: shorter sequences for speed."""
    logger.info(f"\n{'='*80}\n[3/10] CREATING SEQUENCES (seq_len={sequence_length} - FAST MODE)\n{'='*80}")

    features = [col for col in df.columns if col not in ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG']]
    target_cols = ['LAT', 'LON', 'SOG', 'COG']

    X_list, y_list, mmsi_list = [], [], []

    for mmsi, group in tqdm(df.groupby('MMSI'), desc="Creating sequences"):
        group = group.sort_values('BaseDateTime').reset_index(drop=True)
        if len(group) < sequence_length + 1:
            continue

        for i in range(len(group) - sequence_length):
            X_seq = group.iloc[i:i+sequence_length][features].values
            y_seq = group.iloc[i+sequence_length][target_cols].values
            X_list.append(X_seq)
            y_list.append(y_seq)
            mmsi_list.append(mmsi)

            if max_sequences and len(X_list) >= max_sequences:
                break
        if max_sequences and len(X_list) >= max_sequences:
            break

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info(f"✓ Sequences created: {len(X):,}")
    return X, y, features, mmsi_list

# Simple NPZ cache for sequences to speed up initialization on subsequent runs
def _seq_cache_path(seq_len: int, num_features: int) -> Path:
    cache_dir = output_dirs['results'] / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"seq_cache_len{seq_len}_feat{num_features}.npz"


def load_sequences_cache(cache_file: Path):
    data = np.load(cache_file, allow_pickle=True)
    X = data['X']
    y = data['y']
    features = data['features'].tolist()
    mmsi_list = data['mmsi_list'].tolist()
    return X, y, features, mmsi_list


def save_sequences_cache(cache_file: Path, X, y, features, mmsi_list):
    np.savez_compressed(
        cache_file,
        X=X,
        y=y,
        features=np.array(features, dtype=object),
        mmsi_list=np.array(mmsi_list, dtype=object)
    )

    return X, y, features, mmsi_list


# Model architectures
class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=4, output_size=4, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class TemporalCNNModel(nn.Module):
    def __init__(self, input_size, output_size=4, num_filters=128, num_layers=5, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(input_size, num_filters, 1)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (3 - 1) * dilation // 2
            self.blocks.append(nn.Sequential(
                nn.Conv1d(num_filters, num_filters, 3, padding=padding, dilation=dilation),
                nn.BatchNorm1d(num_filters), nn.ReLU(), nn.Dropout(dropout)
            ))
        self.fc = nn.Sequential(
            nn.Linear(num_filters, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=2)
        return self.fc(x)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=4, output_size=4, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=20, device='cuda', model_name='lstm', use_amp=True):
    """Train model with MLflow logging (OPTIMIZED - Reduced epochs & patience) and optional AMP."""
    logger.info(f"\n{'='*80}\nTRAINING {model_name.upper()}\n{'='*80}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)

    # AMP setup
    device_type = device.type if isinstance(device, torch.device) else str(device)
    amp_enabled = bool(use_amp and torch.cuda.is_available() and device_type == 'cuda')
    scaler = GradScaler(enabled=amp_enabled)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    # Log hyperparameters to MLflow
    mlflow.log_param(f"{model_name}_epochs", epochs)
    mlflow.log_param(f"{model_name}_learning_rate", lr)
    mlflow.log_param(f"{model_name}_patience", patience)
    mlflow.log_param(f"{model_name}_batch_size", train_loader.batch_size)
    mlflow.log_param(f"{model_name}_amp_enabled", amp_enabled)

    epoch_iter = tqdm(range(epochs), desc=f"Training {model_name}", unit="epoch", dynamic_ncols=True)
    for epoch in epoch_iter:
        epoch_start = time.time()
        model.train()
        train_loss = 0
        batch_count = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
            # AMP-aware backward + step
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_loss += loss.item()
            batch_count += 1

        train_loss /= batch_count
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        val_batch_count = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                with autocast(enabled=amp_enabled):
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
                val_batch_count += 1

        val_loss /= val_batch_count
        current_lr = optimizer.param_groups[0]['lr']

        # Update tqdm and write per-epoch line to terminal
        epoch_iter.set_postfix(train=f"{train_loss:.5f}", val=f"{val_loss:.5f}", best=f"{best_val_loss:.5f}", lr=f"{current_lr:.2e}")
        log_line = f"[{model_name}] epoch {epoch+1}/{epochs} - train={train_loss:.6f} val={val_loss:.6f} best={best_val_loss:.6f} lr={current_lr:.2e} patience={patience_counter}/{patience} time={time.time()-epoch_start:.1f}s"
        tqdm.write(log_line)
        logger.info(log_line)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # Log metrics every epoch
        mlflow.log_metric(f"{model_name}_train_loss", train_loss, step=epoch)
        mlflow.log_metric(f"{model_name}_val_loss", val_loss, step=epoch)
        mlflow.log_metric(f"{model_name}_lr", current_lr, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dirs['models'] / f'best_{model_name}.pt')
            mlflow.log_metric(f"{model_name}_best_val_loss", val_loss, step=epoch)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            early_msg = f"✓ Early stopping at epoch {epoch+1}/{epochs} (best_val={best_val_loss:.6f})"
            tqdm.write(early_msg)
            logger.info(early_msg)
            break

    logger.info(f"✓ {model_name.upper()} training complete. Best val loss: {best_val_loss:.6f}")
    return train_losses, val_losses


def evaluate_model(model, test_loader, device='cuda', model_name='model'):
    """Evaluate model and return metrics + predictions."""
    logger.info(f"\n{'='*80}\nEVALUATING {model_name.upper()}\n{'='*80}")

    model.eval()
    y_true_all, y_pred_all = [], []
    device_type = device.type if isinstance(device, torch.device) else str(device)
    amp_enabled = bool(torch.cuda.is_available() and device_type == 'cuda')
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc=f"Evaluating {model_name}", unit="batch"):
            X_batch = X_batch.to(device)
            with autocast(enabled=amp_enabled):
                preds = model(X_batch).cpu().numpy()
            y_pred_all.append(preds)
            y_true_all.append(y_batch.numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    metrics = {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'MAE_LAT': float(mean_absolute_error(y_true[:, 0], y_pred[:, 0])),
        'MAE_LON': float(mean_absolute_error(y_true[:, 1], y_pred[:, 1])),
        'MAE_SOG': float(mean_absolute_error(y_true[:, 2], y_pred[:, 2])),
        'MAE_COG': float(mean_absolute_error(y_true[:, 3], y_pred[:, 3]))
    }

    logger.info(f"MAE={mae:.6f}, RMSE={rmse:.6f}, R²={r2:.6f}")
    logger.info(f"  LAT MAE: {metrics['MAE_LAT']:.6f}")
    logger.info(f"  LON MAE: {metrics['MAE_LON']:.6f}")
    logger.info(f"  SOG MAE: {metrics['MAE_SOG']:.6f}")
    logger.info(f"  COG MAE: {metrics['MAE_COG']:.6f}")

    return metrics, y_true, y_pred

# ---------- Visualization helpers ----------

def save_vessel_plots(pred_dict, selected_vessels, out_dir: Path):
    """Save per-vessel plots (LAT, LON, SOG, COG) comparing Actual vs LSTM/CNN/GRU.
    Creates up to len(selected_vessels) images, one per vessel.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = {"lstm": "#1f77b4", "cnn": "#ff7f0e", "gru": "#2ca02c"}
    metrics = [
        ("lat", "Latitude"),
        ("lon", "Longitude"),
        ("sog", "SOG"),
        ("cog", "COG"),
    ]

    for mmsi in tqdm(selected_vessels, desc="Plotting vessels", unit="vessel"):
        mmsi_int = int(mmsi)
        if mmsi_int not in pred_dict:
            continue
        d = pred_dict[mmsi_int]
        n = len(d['actual_lat'])
        if n == 0:
            continue
        t = np.arange(n)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
        axes = axes.flatten()

        for ax, (key, title) in zip(axes, metrics):
            # Actual
            ax.plot(t, d[f"actual_{key}"], label="Actual", color="black", linewidth=1.6)
            # Models
            for model_name in ["lstm", "cnn", "gru"]:
                pred_key = f"{model_name}_{key}"
                if pred_key in d:
                    ax.plot(t, d[pred_key], label=model_name.upper(), color=colors[model_name], linewidth=1.2, alpha=0.95)
            ax.set_title(f"{title}")
            ax.grid(True, alpha=0.3)

        axes[0].legend(loc="upper right", ncol=4, fontsize=8)
        fig.suptitle(f"Vessel {mmsi_int} - Actual vs Predictions", fontsize=14)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])

        out_path = out_dir / f"vessel_{mmsi_int}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)



if __name__ == "__main__":
    # Configure logging in the main process only (avoid Windows spawn issues with DataLoader workers)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dirs['logs'] / 'comprehensive_pipeline.log', encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    logger = logging.getLogger(__name__)

    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE MULTI-MODEL TRAINING PIPELINE")
    logger.info("="*80)

    with mlflow.start_run(run_name="Comprehensive_MultiModel_Pipeline"):
        # Load entire dataset (with sampling for memory efficiency)
        df = load_entire_dataset(start_date=3, end_date=8, sample_per_day=200000)

        # Add features
        df = add_comprehensive_features(df)

        # Create or load cached sequences
        seq_len = 120
        logger.info(f"\n{'='*80}\n[3/10] CREATING/LOADING SEQUENCES (seq_len={seq_len})\n{'='*80}")
        features_for_cache = [col for col in df.columns if col not in ['MMSI','BaseDateTime','LAT','LON','SOG','COG']]
        cache_file = _seq_cache_path(seq_len, len(features_for_cache))
        if cache_file.exists():
            logger.info(f"Loading sequences from cache: {cache_file}")
            X, y, features, mmsi_list = load_sequences_cache(cache_file)
        else:
            X, y, features, mmsi_list = create_sequences_full(df, sequence_length=seq_len, max_sequences=None)
            save_sequences_cache(cache_file, X, y, features, mmsi_list)
            logger.info(f"✓ Cached sequences to: {cache_file}")

        mlflow.log_param("total_sequences", len(X))
        mlflow.log_param("num_features", len(features))
        mlflow.log_param("sequence_length", 120)

        # Split data
        logger.info(f"\n{'='*80}\n[4/10] SPLITTING DATA\n{'='*80}")
        n = len(X)
        train_idx = int(n * 0.7)
        val_idx = int(n * 0.9)

        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
        X_test, y_test = X[val_idx:], y[val_idx:]

        logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

        # Normalize
        logger.info(f"\n{'='*80}\n[5/10] NORMALIZING DATA\n{'='*80}")
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        # Create dataloaders
        logger.info(f"\n{'='*80}\n[6/10] CREATING DATALOADERS\n{'='*80}")
        train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test))

        pin_memory = torch.cuda.is_available()
        batch_size = 128 if torch.cuda.is_available() else 64
        is_windows = (platform.system() == "Windows")
        # On Windows, use num_workers=0 to avoid spawn-related issues
        num_workers = 0 if is_windows else max(2, (os.cpu_count() or 4) // 2)
        logger.info(f"DataLoaders: batch_size={batch_size}, pin_memory={pin_memory}, num_workers={num_workers}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {device}")

        # Train models
        logger.info(f"\n{'='*80}\n[7/10] TRAINING MODELS\n{'='*80}")

        models_info = {}

        # LSTM (OPTIMIZED: 100 epochs, patience=20)
        logger.info("\n[7.1/10] Training LSTM...")
        lstm_model = EnhancedLSTMModel(input_size=len(features), hidden_size=256, num_layers=4, dropout=0.1).to(device)
        lstm_train_losses, lstm_val_losses = train_model(lstm_model, train_loader, val_loader, epochs=100, lr=0.001, patience=20, device=device, model_name='lstm')
        models_info['lstm'] = {'model': lstm_model, 'train_losses': lstm_train_losses, 'val_losses': lstm_val_losses}

        # CNN (OPTIMIZED: 100 epochs, patience=20)
        logger.info("\n[7.2/10] Training CNN...")
        cnn_model = TemporalCNNModel(input_size=len(features), num_filters=128, num_layers=5, dropout=0.1).to(device)
        cnn_train_losses, cnn_val_losses = train_model(cnn_model, train_loader, val_loader, epochs=100, lr=0.001, patience=20, device=device, model_name='cnn')
        models_info['cnn'] = {'model': cnn_model, 'train_losses': cnn_train_losses, 'val_losses': cnn_val_losses}

        # GRU (OPTIMIZED: 100 epochs, patience=20)
        logger.info("\n[7.3/10] Training GRU...")
        gru_model = GRUModel(input_size=len(features), hidden_size=512, num_layers=4, dropout=0.1).to(device)
        gru_train_losses, gru_val_losses = train_model(gru_model, train_loader, val_loader, epochs=100, lr=0.001, patience=20, device=device, model_name='gru')
        models_info['gru'] = {'model': gru_model, 'train_losses': gru_train_losses, 'val_losses': gru_val_losses}

        logger.info(f"\n{'='*80}\n[8/10] EVALUATION & PREDICTIONS\n{'='*80}")

        # Generate and save per-vessel plots (up to selected 300)
        logger.info("Generating per-vessel plots (up to 300)...")
        images_dir = output_dirs['images'] / 'vessels_300'
        save_vessel_plots(pred_dict, selected_vessels, images_dir)
        try:
            mlflow.log_artifacts(str(images_dir))
        except Exception as e:
            logger.warning(f"MLflow artifact logging skipped: {e}")
        logger.info(f"\u2713 Saved vessel plots to: {images_dir}")



        # Load best models and evaluate
        all_metrics = {}
        all_predictions = {}

        for model_name in ['lstm', 'cnn', 'gru']:
            model = models_info[model_name]['model']
            model.load_state_dict(torch.load(output_dirs['models'] / f'best_{model_name}.pt'))
            metrics, y_true_test, y_pred = evaluate_model(model, test_loader, device, model_name)
            all_metrics[model_name] = metrics
            all_predictions[model_name] = y_pred
            mlflow.log_metrics({f"{model_name}_{k}": v for k, v in metrics.items()})

        logger.info(f"\n{'='*80}\n[9/10] GENERATING 300-VESSEL PREDICTIONS\n{'='*80}")

        # Select 300 random vessels
        test_mmsi = np.array(mmsi_list)[val_idx:]
        unique_mmsi = np.unique(test_mmsi)
        np.random.seed(42)
        selected_vessels = np.random.choice(unique_mmsi, size=min(300, len(unique_mmsi)), replace=False)

        logger.info(f"Selected {len(selected_vessels)} vessels for detailed analysis")

        # Build predictions dict
        pred_dict = {}
        for mmsi in tqdm(selected_vessels, desc="Generating vessel predictions", unit="vessel"):
            idxs = np.where(test_mmsi == mmsi)[0]
            if len(idxs) == 0:
                continue

            entry = {
                'actual_lat': y_true_test[idxs, 0],
                'actual_lon': y_true_test[idxs, 1],
                'actual_sog': y_true_test[idxs, 2],
                'actual_cog': y_true_test[idxs, 3]
            }

            for model_name, y_pred in all_predictions.items():
                entry[f'{model_name}_lat'] = y_pred[idxs, 0]
                entry[f'{model_name}_lon'] = y_pred[idxs, 1]
                entry[f'{model_name}_sog'] = y_pred[idxs, 2]
                entry[f'{model_name}_cog'] = y_pred[idxs, 3]

                entry[f'{model_name}_mae_lat'] = float(np.mean(np.abs(entry[f'{model_name}_lat'] - entry['actual_lat'])))
                entry[f'{model_name}_mae_lon'] = float(np.mean(np.abs(entry[f'{model_name}_lon'] - entry['actual_lon'])))
                entry[f'{model_name}_mae_sog'] = float(np.mean(np.abs(entry[f'{model_name}_sog'] - entry['actual_sog'])))
                entry[f'{model_name}_mae_cog'] = float(np.mean(np.abs(entry[f'{model_name}_cog'] - entry['actual_cog'])))

            pred_dict[int(mmsi)] = entry

        logger.info(f"✓ Built predictions for {len(pred_dict)} vessels")

        # Generate and save per-vessel plots (up to selected 300)
        logger.info("Generating per-vessel plots (up to 300)...")
        images_dir = output_dirs['images'] / 'vessels_300'
        save_vessel_plots(pred_dict, selected_vessels, images_dir)
        try:
            mlflow.log_artifacts(str(images_dir))
        except Exception as e:
            logger.warning(f"MLflow artifact logging skipped: {e}")
        logger.info(f"\u2713 Saved vessel plots to: {images_dir}")


        logger.info(f"\n{'='*80}\n[10/10] SAVING RESULTS & VISUALIZATIONS\n{'='*80}")

        # Save predictions CSV
        logger.info("Converting predictions to CSV format...")
        rows = []
        for mmsi, d in tqdm(pred_dict.items(), desc="Building CSV rows", unit="vessel"):
            n = len(d['actual_lat'])
            for i in range(n):
                row = {
                    'MMSI': mmsi, 'idx': i,
                    'actual_lat': float(d['actual_lat'][i]),
                    'actual_lon': float(d['actual_lon'][i]),
                    'actual_sog': float(d['actual_sog'][i]),
                    'actual_cog': float(d['actual_cog'][i])
                }
                for model_name in ['lstm', 'cnn', 'gru']:
                    row[f'{model_name}_lat'] = float(d.get(f'{model_name}_lat', [np.nan]*n)[i]) if f'{model_name}_lat' in d else np.nan
                    row[f'{model_name}_lon'] = float(d.get(f'{model_name}_lon', [np.nan]*n)[i]) if f'{model_name}_lon' in d else np.nan
                    row[f'{model_name}_sog'] = float(d.get(f'{model_name}_sog', [np.nan]*n)[i]) if f'{model_name}_sog' in d else np.nan
                    row[f'{model_name}_cog'] = float(d.get(f'{model_name}_cog', [np.nan]*n)[i]) if f'{model_name}_cog' in d else np.nan
                rows.append(row)

        logger.info("Saving predictions to CSV...")
        df_pred = pd.DataFrame(rows)
        df_pred.to_csv(output_dirs['csv'] / 'vessel_predictions_300_detailed.csv', index=False)
        logger.info(f"✓ Saved detailed predictions: {output_dirs['csv'] / 'vessel_predictions_300_detailed.csv'}")

        # Save model comparison
        comparison_df = pd.DataFrame([
            {'Model': 'LSTM', **all_metrics['lstm']},
            {'Model': 'CNN', **all_metrics['cnn']},
            {'Model': 'GRU', **all_metrics['gru']}
        ])
        comparison_df.to_csv(output_dirs['csv'] / 'model_comparison_comprehensive.csv', index=False)
        logger.info(f"✓ Saved model comparison: {output_dirs['csv'] / 'model_comparison_comprehensive.csv'}")

        logger.info(f"\n{'='*80}\nPIPELINE COMPLETE!\n{'='*80}")
        logger.info(f"Results saved in: {output_dirs['results']}")
        logger.info(f"\nModel Performance:")
        for model_name, metrics in all_metrics.items():
            logger.info(f"  {model_name.upper()}: MAE={metrics['MAE']:.6f}, RMSE={metrics['RMSE']:.6f}, R²={metrics['R2']:.6f}")

