"""
Main Execution Pipeline - Complete End-to-End Training and Testing
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
# Optional viz/geospatial libs
try:
    import geopandas as gpd
    from shapely.geometry import LineString, Point
except Exception:
    gpd = None
try:
    import altair as alt
except Exception:
    alt = None


# Setup
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(output_dirs['logs'] / 'main_execution.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

mlflow.set_tracking_uri(f"file:{output_dirs['mlflow'].absolute()}")
mlflow.set_experiment("Maritime_Vessel_Forecasting_Complete")


def load_data(start_date=3, end_date=8, sample_per_day=None):
    """Load data. If sample_per_day is None, load all data."""
    if sample_per_day is None:
        logger.info(f"\n{'='*70}\n[1/10] LOADING DATA (FULL DATASET)\n{'='*70}")
    else:
        logger.info(f"\n{'='*70}\n[1/10] LOADING DATA (SAMPLED: {sample_per_day:,}/day)\n{'='*70}")

    base_path = Path(r"D:\Maritime_Vessel_monitoring\csv_extracted_data")
    all_data = []

    for day in range(start_date, end_date + 1):
        file_path = base_path / f"AIS_2020_01_{day:02d}" / f"AIS_2020_01_{day:02d}.csv"
        if file_path.exists():
            logger.info(f"Loading {file_path.name}...")
            df = pd.read_csv(file_path, usecols=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG'])
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], format="%Y-%m-%dT%H:%M:%S", errors='coerce')
            df = df.dropna(subset=['BaseDateTime', 'LAT', 'LON', 'SOG', 'COG'])

            # Sample for memory efficiency if specified
            if sample_per_day is not None and len(df) > sample_per_day:
                df = df.sample(n=sample_per_day, random_state=42)

            all_data.append(df)
            logger.info(f"  ✓ {len(df):,} records")

    df_all = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total: {len(df_all):,} records, {df_all['MMSI'].nunique():,} vessels")

    return df_all


def add_advanced_features(df):
    """Add 50+ advanced features."""
    logger.info(f"\n{'='*70}\n[2/10] ADVANCED FEATURE ENGINEERING (50+ FEATURES)\n{'='*70}")

    df = df.sort_values('BaseDateTime').reset_index(drop=True)

    # Temporal
    df['hour'] = df['BaseDateTime'].dt.hour
    df['day_of_week'] = df['BaseDateTime'].dt.dayofweek
    df['month'] = df['BaseDateTime'].dt.month

    # Cyclical
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Kinematic
    df['speed_change'] = df.groupby('MMSI')['SOG'].diff().fillna(0)
    df['heading_change'] = df.groupby('MMSI')['COG'].diff().fillna(0)
    df['lat_change'] = df.groupby('MMSI')['LAT'].diff().fillna(0)
    df['lon_change'] = df.groupby('MMSI')['LON'].diff().fillna(0)

    # Lag features
    for lag in [1, 2, 3]:
        df[f'LAT_lag{lag}'] = df.groupby('MMSI')['LAT'].shift(lag).fillna(0)
        df[f'LON_lag{lag}'] = df.groupby('MMSI')['LON'].shift(lag).fillna(0)
        df[f'SOG_lag{lag}'] = df.groupby('MMSI')['SOG'].shift(lag).fillna(0)
        df[f'COG_lag{lag}'] = df.groupby('MMSI')['COG'].shift(lag).fillna(0)

    # Rolling statistics (simplified to avoid memory issues)
    df['SOG_mean_3'] = df.groupby('MMSI')['SOG'].shift(1).fillna(0)
    df['SOG_std_3'] = df.groupby('MMSI')['SOG'].shift(2).fillna(0)
    df['COG_mean_3'] = df.groupby('MMSI')['COG'].shift(1).fillna(0)

    # Acceleration
    df['speed_accel'] = df.groupby('MMSI')['speed_change'].diff().fillna(0)
    df['heading_accel'] = df.groupby('MMSI')['heading_change'].diff().fillna(0)

    # Velocity
    df['velocity_x'] = df['SOG'] * np.cos(np.radians(df['COG']))
    df['velocity_y'] = df['SOG'] * np.sin(np.radians(df['COG']))
    df['velocity_mag'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)

    # Polynomial
    df['LAT_sq'] = df['LAT'] ** 2
    df['LON_sq'] = df['LON'] ** 2
    df['SOG_sq'] = df['SOG'] ** 2

    # Interaction
    df['speed_heading_int'] = df['SOG'] * df['COG']
    df['lat_lon_int'] = df['LAT'] * df['LON']

    logger.info(f"✓ Total features: {len(df.columns)}")

    return df


def create_sequences(df, sequence_length=120, max_sequences=None):
    """Create sequences with 120 timesteps. If max_sequences is None, create all."""
    if max_sequences is None:
        logger.info(f"\n{'='*70}\n[3/10] CREATING SEQUENCES (120 TIMESTEPS, ALL DATA)\n{'='*70}")
    else:
        logger.info(f"\n{'='*70}\n[3/10] CREATING SEQUENCES (120 TIMESTEPS, MAX {max_sequences:,})\n{'='*70}")

    features = [col for col in df.columns if col not in ['MMSI', 'BaseDateTime']]
    X, y, mmsi_list = [], [], []

    for mmsi in tqdm(df['MMSI'].unique(), desc="Creating sequences"):
        vessel_data = df[df['MMSI'] == mmsi][features].values

        if len(vessel_data) < sequence_length + 1:
            continue

        for i in range(len(vessel_data) - sequence_length):
            X.append(vessel_data[i:i+sequence_length])
            y.append(vessel_data[i+sequence_length, :4])
            mmsi_list.append(mmsi)

            if max_sequences is not None and len(X) >= max_sequences:
                break

        if max_sequences is not None and len(X) >= max_sequences:
            break

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    logger.info(f"Sequences: {len(X):,}, X shape: {X.shape}, y shape: {y.shape}")

    return X, y, features, mmsi_list


class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=4, dropout=0.2):
        super(EnhancedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
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
    def __init__(self, input_size, output_size=4, num_filters=64, num_layers=4, dropout=0.1):
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


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=40, device='cuda', model_name='lstm'):
    """Train model with MLflow logging."""
    logger.info(f"\n{'='*70}\nTRAINING {model_name.upper()}\n{'='*70}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=False)

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



def evaluate_model(model, test_loader, device='cuda', model_name='model'):
    """Evaluate model on test_loader and return metrics and predictions."""
    logger.info(f"\n{'='*70}\nEVALUATING {model_name.upper()}\n{'='*70}")
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
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
        'MAE_per_output': {
            'LAT': float(mean_absolute_error(y_true[:, 0], y_pred[:, 0])),
            'LON': float(mean_absolute_error(y_true[:, 1], y_pred[:, 1])),
            'SOG': float(mean_absolute_error(y_true[:, 2], y_pred[:, 2])),
            'COG': float(mean_absolute_error(y_true[:, 3], y_pred[:, 3]))
        }
    }
    logger.info(f"MAE={mae:.6f}, RMSE={rmse:.6f}, R²={r2:.6f}")
    return metrics, y_true, y_pred


def select_random_vessels(mmsi_array, n=300, seed=42):
    """Select up to n unique MMSI values from the given array."""
    unique = np.unique(mmsi_array)
    rng = np.random.default_rng(seed)
    n_select = min(n, len(unique))
    return rng.choice(unique, size=n_select, replace=False)


def build_predictions_dict(selected_mmsi, test_mmsi, y_true_test, preds_by_model):
    """Build vessel-wise predictions dict for plotting.
    preds_by_model: dict like {'lstm': y_pred_lstm, 'cnn': y_pred_cnn}
    """
    pred_dict = {}
    for m in selected_mmsi:
        idxs = np.where(test_mmsi == m)[0]
        if idxs.size == 0:
            continue
        entry = {
            'actual_lat': y_true_test[idxs, 0],
            'actual_lon': y_true_test[idxs, 1]
        }
        # Add per-model predictions and simple per-vessel MAE
        for name, yp in preds_by_model.items():
            entry[f'{name}_lat'] = yp[idxs, 0]
            entry[f'{name}_lon'] = yp[idxs, 1]
            entry[f'{name}_mae_lat'] = float(np.mean(np.abs(entry[f'{name}_lat'] - entry['actual_lat'])))
            entry[f'{name}_mae_lon'] = float(np.mean(np.abs(entry[f'{name}_lon'] - entry['actual_lon'])))
        pred_dict[int(m)] = entry
    return pred_dict


def save_predictions_csv(pred_dict, save_path):
    rows = []
    for mmsi, d in pred_dict.items():
        n = len(d['actual_lat'])
        for i in range(n):
            rows.append({
                'MMSI': mmsi,
                'idx': i,
                'actual_lat': float(d['actual_lat'][i]),
                'actual_lon': float(d['actual_lon'][i]),
                'lstm_lat': float(d.get('lstm_lat', [np.nan]*n)[i]) if 'lstm_lat' in d else np.nan,
                'lstm_lon': float(d.get('lstm_lon', [np.nan]*n)[i]) if 'lstm_lon' in d else np.nan,
                'cnn_lat': float(d.get('cnn_lat', [np.nan]*n)[i]) if 'cnn_lat' in d else np.nan,
                'cnn_lon': float(d.get('cnn_lon', [np.nan]*n)[i]) if 'cnn_lon' in d else np.nan,
            })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    logger.info(f"✓ Saved predictions CSV: {save_path}")


def plot_vessel_predictions_matplotlib(pred_dict, num_vessels=300, save_path=None):
    """Plot lon-lat trajectories for random vessels with different colors per model."""
    vessel_ids = list(pred_dict.keys())[:min(num_vessels, len(pred_dict))]
    n_cols = 5
    n_rows = (len(vessel_ids) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten()
    for idx, mmsi in enumerate(vessel_ids):
        ax = axes[idx]
        d = pred_dict[mmsi]
        ax.plot(d['actual_lon'], d['actual_lat'], color='black', linewidth=1.5, alpha=0.9, label='Actual')
        if 'lstm_lon' in d:
            ax.plot(d['lstm_lon'], d['lstm_lat'], color='#1f77b4', linewidth=1.2, alpha=0.8, label=f"LSTM (LAT MAE {d['lstm_mae_lat']:.3f})")
        if 'cnn_lon' in d:
            ax.plot(d['cnn_lon'], d['cnn_lat'], color='#d62728', linewidth=1.2, alpha=0.8, label=f"CNN (LAT MAE {d['cnn_mae_lat']:.3f})")
        ax.set_title(f"Vessel {mmsi}", fontsize=9)
        ax.set_xlabel('Lon')
        ax.set_ylabel('Lat')
        ax.grid(True, alpha=0.3)
        if idx % n_cols == 0:
            ax.legend(fontsize=7)
    for j in range(len(vessel_ids), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    if save_path is None:
        save_path = output_dirs['images'] / 'vessel_predictions_300_matplotlib.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    logger.info(f"✓ Saved matplotlib vessel predictions: {save_path}")
    plt.close()


def plot_geopandas_map(pred_dict, save_path=None):
    if gpd is None:
        logger.warning("GeoPandas or Shapely not available. Skipping GeoPandas map.")
        return
    records = []
    for mmsi, d in pred_dict.items():
        # Build LineStrings
        actual_coords = list(zip(d['actual_lon'], d['actual_lat']))
        records.append({'MMSI': mmsi, 'model': 'Actual', 'geometry': LineString(actual_coords)})
        if 'lstm_lon' in d:
            lstm_coords = list(zip(d['lstm_lon'], d['lstm_lat']))
            records.append({'MMSI': mmsi, 'model': 'LSTM', 'geometry': LineString(lstm_coords)})
        if 'cnn_lon' in d:
            cnn_coords = list(zip(d['cnn_lon'], d['cnn_lat']))
            records.append({'MMSI': mmsi, 'model': 'CNN', 'geometry': LineString(cnn_coords)})
    gdf = gpd.GeoDataFrame(records, geometry='geometry', crs='EPSG:4326')
    # Plot
    ax = gdf[gdf['model']=='Actual'].plot(color='black', linewidth=1.0, alpha=0.7, figsize=(12, 10))
    if (gdf['model']=='LSTM').any():
        gdf[gdf['model']=='LSTM'].plot(ax=ax, color='#1f77b4', linewidth=0.8, alpha=0.6)
    if (gdf['model']=='CNN').any():
        gdf[gdf['model']=='CNN'].plot(ax=ax, color='#d62728', linewidth=0.8, alpha=0.6)
    ax.set_title('GeoPandas: Actual vs Predicted Trajectories (300 Vessels)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    if save_path is None:
        save_path = output_dirs['images'] / 'vessel_predictions_geopandas.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    logger.info(f"✓ Saved GeoPandas map: {save_path}")
    plt.close()


def save_altair_predictions_html(pred_dict, save_path=None, max_vessels=50):
    if alt is None:
        logger.warning("Altair not available. Skipping Altair visualization.")
        return
    # Flatten limited subset for interactivity
    vessel_ids = list(pred_dict.keys())[:min(max_vessels, len(pred_dict))]
    rows = []
    for mmsi in vessel_ids:
        d = pred_dict[mmsi]
        n = len(d['actual_lat'])
        for i in range(n):
            rows.append({'MMSI': str(mmsi), 'idx': i, 'lon': float(d['actual_lon'][i]), 'lat': float(d['actual_lat'][i]), 'model': 'Actual'})
            if 'lstm_lon' in d:
                rows.append({'MMSI': str(mmsi), 'idx': i, 'lon': float(d['lstm_lon'][i]), 'lat': float(d['lstm_lat'][i]), 'model': 'LSTM'})
            if 'cnn_lon' in d:
                rows.append({'MMSI': str(mmsi), 'idx': i, 'lon': float(d['cnn_lon'][i]), 'lat': float(d['cnn_lat'][i]), 'model': 'CNN'})
    df = pd.DataFrame(rows)
    chart = (alt.Chart(df)
             .mark_line(opacity=0.7)
             .encode(
                 x=alt.X('lon:Q', title='Longitude'),
                 y=alt.Y('lat:Q', title='Latitude'),
                 color=alt.Color('model:N', scale=alt.Scale(domain=['Actual','LSTM','CNN'], range=['black','#1f77b4','#d62728'])),
                 tooltip=['MMSI','model','idx','lon','lat']
             )
             .properties(width=300, height=220)
            )
    facet = chart.facet(column=alt.Column('MMSI:N', title='Vessel'), columns=5)
    if save_path is None:
        save_path = output_dirs['images'] / 'vessel_predictions_altair.html'
    facet.save(str(save_path))
    logger.info(f"\u2713 Saved Altair HTML: {save_path}")



def plot_training_curves_simple(lstm_train, lstm_val, cnn_train, cnn_val, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    # LSTM
    axes[0].plot(lstm_train, label='Train', color='#1f77b4')
    axes[0].plot(lstm_val, label='Val', color='#ff7f0e')
    axes[0].set_title('LSTM Loss per Epoch'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    # CNN
    axes[1].plot(cnn_train, label='Train', color='#1f77b4')
    axes[1].plot(cnn_val, label='Val', color='#ff7f0e')
    axes[1].set_title('Temporal CNN Loss per Epoch'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MSE Loss')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    if save_path is None:
        save_path = output_dirs['images'] / 'training_curves_advanced.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    logger.info(f"\u2713 Saved training curves: {save_path}")
    plt.close()






if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("MAIN EXECUTION PIPELINE - COMPLETE END-TO-END")
    logger.info("="*70)

    with mlflow.start_run(run_name="Complete_Pipeline_Run_FullData"):
        # Step 1: Load data (FULL DATASET - no sampling)
        df = load_data(start_date=3, end_date=8, sample_per_day=None)

        # Step 2: Add features
        df = add_advanced_features(df)

        # Step 3: Create sequences (no limit - use all available)
        X, y, features, mmsi_list = create_sequences(df, sequence_length=120, max_sequences=None)

        # Log dataset info
        mlflow.log_param("total_sequences", len(X))
        mlflow.log_param("num_features", len(features))
        mlflow.log_param("sequence_length", 120)

        logger.info(f"\n{'='*70}\n[4/10] SPLITTING DATA\n{'='*70}")
        n = len(X)
        train_idx = int(n * 0.7)
        val_idx = int(n * 0.9)

        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
        X_test, y_test = X[val_idx:], y[val_idx:]

        logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

        logger.info(f"\n{'='*70}\n[5/10] NORMALIZING DATA\n{'='*70}")
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        logger.info(f"\n{'='*70}\n[6/10] CREATING DATALOADERS\n{'='*70}")
        train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test))

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {device}")

        logger.info(f"\n{'='*70}\n[7/10] TRAINING MODELS\n{'='*70}")

        # Train LSTM
        lstm_model = EnhancedLSTMModel(input_size=len(features), hidden_size=512, num_layers=4, dropout=0.1).to(device)
        lstm_train_losses, lstm_val_losses = train_model(lstm_model, train_loader, val_loader, epochs=200, device=device, model_name='lstm')

        # Train CNN
        cnn_model = TemporalCNNModel(input_size=len(features), num_filters=128, num_layers=5, dropout=0.1).to(device)
        cnn_train_losses, cnn_val_losses = train_model(cnn_model, train_loader, val_loader, epochs=200, device=device, model_name='cnn')

        # Plot training curves
        plot_training_curves_simple(lstm_train_losses, lstm_val_losses, cnn_train_losses, cnn_val_losses,
                                    save_path=output_dirs['images'] / 'training_curves_advanced.png')


        logger.info(f"\n{'='*70}\n[8/10] EVALUATION\n{'='*70}")

        lstm_model.load_state_dict(torch.load(output_dirs['models'] / 'best_lstm.pt'))


        cnn_model.load_state_dict(torch.load(output_dirs['models'] / 'best_cnn.pt'))

        logger.info("Pipeline setup complete. Ready for testing and visualization.")

        # Evaluate and get predictions for each model
        lstm_metrics, y_true_test, y_pred_lstm = evaluate_model(lstm_model, test_loader, device, 'lstm')
        mlflow.log_metric('lstm_MAE', lstm_metrics['MAE'])
        mlflow.log_metric('lstm_RMSE', lstm_metrics['RMSE'])
        mlflow.log_metric('lstm_R2', lstm_metrics['R2'])
        cnn_metrics, _, y_pred_cnn = evaluate_model(cnn_model, test_loader, device, 'cnn')
        mlflow.log_metric('cnn_MAE', cnn_metrics['MAE'])
        mlflow.log_metric('cnn_RMSE', cnn_metrics['RMSE'])
        mlflow.log_metric('cnn_R2', cnn_metrics['R2'])

        # Select 300 random vessels from the test set (by sequence MMSI)
        test_mmsi = np.array(mmsi_list)[val_idx:]
        selected_vessels = select_random_vessels(test_mmsi, n=300)
        preds_by_model = {'lstm': y_pred_lstm, 'cnn': y_pred_cnn}
        predictions_dict = build_predictions_dict(selected_vessels, test_mmsi, y_true_test, preds_by_model)

        # Save CSV and Visualizations (Matplotlib, GeoPandas, Altair)
        save_predictions_csv(predictions_dict, output_dirs['csv'] / 'vessel_predictions_300.csv')
        plot_vessel_predictions_matplotlib(predictions_dict, num_vessels=300, save_path=output_dirs['images'] / 'vessel_predictions_300_matplotlib.png')
        plot_geopandas_map(predictions_dict, save_path=output_dirs['images'] / 'vessel_predictions_geopandas.png')
        save_altair_predictions_html(predictions_dict, save_path=output_dirs['images'] / 'vessel_predictions_altair.html', max_vessels=50)

        logger.info("All predictions and visualizations generated.")
        # Save model comparison CSV
        comparison_df = pd.DataFrame([
            {'Model': 'LSTM', **lstm_metrics},
            {'Model': 'CNN', **cnn_metrics}
        ])
        comparison_df.to_csv(output_dirs['csv'] / 'model_comparison.csv', index=False)
        logger.info(f"\u2713 Saved model comparison CSV: {output_dirs['csv'] / 'model_comparison.csv'}")



