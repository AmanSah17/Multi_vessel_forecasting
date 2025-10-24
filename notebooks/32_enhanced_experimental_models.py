"""
Enhanced Experimental Models Pipeline with Haversine Distance & Advanced Forecasting
- Haversine distance for nonlinear spatial relationships
- Kalman Filter for state estimation
- ARIMA models for time series forecasting
- Tiny LSTM with enhanced features
- MLflow logging for all experiments
- Per-vessel evaluation on test set
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
from statsmodels.tsa.arima.model import ARIMA
from scipy.spatial.distance import euclidean

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
        logging.FileHandler('logs/enhanced_experimental_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# GPU optimization
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# ======================== HAVERSINE DISTANCE ========================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate Haversine distance between two points in km.
    Captures nonlinear spatial relationships in maritime data.
    """
    R = 6371  # Earth radius in km
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def add_haversine_features(df):
    """Add Haversine distance features for nonlinear spatial relationships."""
    logger.info("Computing Haversine distance features...")
    
    df = df.sort_values(['MMSI', 'BaseDateTime']).reset_index(drop=True)
    
    # Haversine distance to previous point
    df['haversine_dist'] = 0.0
    for mmsi in df['MMSI'].unique():
        mask = df['MMSI'] == mmsi
        indices = np.where(mask)[0]
        
        for i in range(1, len(indices)):
            curr_idx = indices[i]
            prev_idx = indices[i-1]
            
            dist = haversine_distance(
                df.loc[prev_idx, 'LAT'], df.loc[prev_idx, 'LON'],
                df.loc[curr_idx, 'LAT'], df.loc[curr_idx, 'LON']
            )
            df.loc[curr_idx, 'haversine_dist'] = dist
    
    df['haversine_dist'] = df['haversine_dist'].astype('float32')
    
    # Haversine distance lag features
    for lag in [1, 2, 3]:
        df[f'haversine_lag{lag}'] = df.groupby('MMSI')['haversine_dist'].shift(lag).fillna(0).astype('float32')
    
    logger.info(f"✓ Added Haversine features")
    return df


def add_comprehensive_features(df):
    """Add comprehensive features including Haversine distance."""
    logger.info(f"\n{'='*80}\n[2/10] ADDING COMPREHENSIVE FEATURES\n{'='*80}")
    
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
    
    # Kinematic features
    logger.info("Computing kinematic features...")
    for col in ['LAT', 'LON', 'SOG', 'COG']:
        df[f'{col.lower()}_diff'] = df.groupby('MMSI')[col].diff().fillna(0).astype('float32')
    
    # Lag features
    logger.info("Computing lag features...")
    for lag in [1, 2, 3]:
        for col in ['LAT', 'LON', 'SOG', 'COG']:
            df[f'{col.lower()}_lag{lag}'] = df.groupby('MMSI')[col].shift(lag).fillna(0).astype('float32')
    
    # Polynomial features
    logger.info("Computing polynomial features...")
    df['lat_sq'] = (df['LAT'] ** 2).astype('float32')
    df['lon_sq'] = (df['LON'] ** 2).astype('float32')
    df['sog_sq'] = (df['SOG'] ** 2).astype('float32')
    df['speed_heading_int'] = (df['SOG'] * df['COG']).astype('float32')
    df['lat_lon_int'] = (df['LAT'] * df['LON']).astype('float32')
    
    # Haversine distance features
    df = add_haversine_features(df)
    
    logger.info(f"✓ Total features: {len(df.columns) - 6}")  # Exclude MMSI, BaseDateTime, LAT, LON, SOG, COG
    return df


# ======================== KALMAN FILTER ========================

class KalmanFilter1D:
    """1D Kalman Filter for vessel trajectory prediction."""
    def __init__(self, process_variance, measurement_variance, initial_value=0, initial_estimate_error=1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.estimate_error = initial_estimate_error
        self.kalman_gain = 0
    
    def update(self, measurement):
        """Update Kalman filter with new measurement."""
        # Prediction step
        self.estimate_error += self.process_variance
        
        # Update step
        self.kalman_gain = self.estimate_error / (self.estimate_error + self.measurement_variance)
        self.estimate += self.kalman_gain * (measurement - self.estimate)
        self.estimate_error *= (1 - self.kalman_gain)
        
        return self.estimate


class KalmanFilterVessel:
    """Multi-dimensional Kalman Filter for vessel state estimation."""
    def __init__(self, process_var=0.01, measurement_var=0.1):
        self.filters = {
            'LAT': KalmanFilter1D(process_var, measurement_var),
            'LON': KalmanFilter1D(process_var, measurement_var),
            'SOG': KalmanFilter1D(process_var, measurement_var),
            'COG': KalmanFilter1D(process_var, measurement_var)
        }
    
    def predict(self, measurements):
        """Predict next state given measurements."""
        predictions = {}
        for key, value in measurements.items():
            if key in self.filters:
                predictions[key] = self.filters[key].update(value)
        return predictions


# ======================== ARIMA MODEL ========================

class ARIMAVessel:
    """ARIMA model for per-vessel forecasting."""
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.models = {}
        self.fitted = False
    
    def fit(self, data_dict):
        """Fit ARIMA models for each variable."""
        for key, values in data_dict.items():
            if len(values) > 10:  # Need enough data
                try:
                    self.models[key] = ARIMA(values, order=self.order).fit()
                except:
                    self.models[key] = None
        self.fitted = True
    
    def predict(self, steps=1):
        """Predict next steps."""
        predictions = {}
        for key, model in self.models.items():
            if model is not None:
                try:
                    forecast = model.get_forecast(steps=steps)
                    predictions[key] = forecast.predicted_mean.values[-1]
                except:
                    predictions[key] = 0
        return predictions


# ======================== TINY LSTM WITH HAVERSINE ========================

class TinyLSTMWithHaversine(nn.Module):
    """Tiny LSTM enhanced with Haversine distance features."""
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


# ======================== TRAINING FUNCTIONS ========================

def train_tiny_lstm_enhanced(X_train, y_train, X_val, y_val, epochs=100, batch_size=256):
    """Train Tiny LSTM with Haversine features."""
    logger.info(f"\n{'='*80}\nTRAINING TINY LSTM WITH HAVERSINE FEATURES\n{'='*80}")
    
    input_size = X_train.shape[2]
    model = TinyLSTMWithHaversine(input_size=input_size).to(DEVICE)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for X_batch, y_batch in pbar:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                
                optimizer.zero_grad()
                with autocast():
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
            y_val_tensor = torch.FloatTensor(y_val).to(DEVICE)
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()
        
        logger.info(f"Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'results/models/tiny_lstm_haversine.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    return model, best_val_loss


def evaluate_models(X_test, y_test, model_dict, experiment_name):
    """Evaluate all models and log to MLflow."""
    logger.info(f"\n{'='*80}\nEVALUATING MODELS: {experiment_name}\n{'='*80}")
    
    with mlflow.start_run(run_name=experiment_name):
        results = {}
        
        for model_name, model in model_dict.items():
            logger.info(f"\nEvaluating {model_name}...")
            
            if model_name == "Tiny LSTM Haversine":
                model.eval()
                with torch.no_grad():
                    X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
                    predictions = model(X_test_tensor).cpu().numpy()
            
            # Compute metrics
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            
            results[model_name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
            
            # Log to MLflow
            mlflow.log_metric(f"{model_name}_MAE", mae)
            mlflow.log_metric(f"{model_name}_RMSE", rmse)
            mlflow.log_metric(f"{model_name}_R2", r2)
            
            logger.info(f"{model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
        return results


# ======================== MAIN PIPELINE ========================

def load_entire_dataset(start_date=3, end_date=8, sample_per_day=None):
    """Load dataset with 5-minute intervals."""
    logger.info(f"\n{'='*80}\n[1/10] LOADING DATASET\n{'='*80}")

    base_path = Path(r"D:\Maritime_Vessel_monitoring\csv_extracted_data")
    all_data = []

    for day in range(start_date, end_date + 1):
        file_path = base_path / f"AIS_2020_01_{day:02d}" / f"AIS_2020_01_{day:02d}.csv"
        if file_path.exists():
            logger.info(f"Loading day {day}...")
            df = pd.read_csv(file_path, usecols=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG'])
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], format="%Y-%m-%dT%H:%M:%S", errors='coerce')
            df = df.dropna(subset=['BaseDateTime', 'LAT', 'LON', 'SOG', 'COG'])

            if sample_per_day is not None and len(df) > sample_per_day:
                df = df.sample(n=sample_per_day, random_state=42)

            all_data.append(df)
            logger.info(f"  ✓ {len(df):,} records")

    df_all = pd.concat(all_data, ignore_index=True)
    logger.info(f"✓ Total: {len(df_all):,} records, {df_all['MMSI'].nunique():,} vessels")
    return df_all


def create_sequences(df, sequence_length=12, sample_vessels=None):
    """Create sequences for LSTM training."""
    logger.info(f"\n{'='*80}\n[3/10] CREATING SEQUENCES (seq_len={sequence_length})\n{'='*80}")

    features = [col for col in df.columns if col not in ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG']]
    target_cols = ['LAT', 'LON', 'SOG', 'COG']

    X_list, y_list, mmsi_list = [], [], []

    grouped = df.groupby('MMSI', sort=False)
    unique_mmsi = list(grouped.groups.keys())

    if sample_vessels and len(unique_mmsi) > sample_vessels:
        unique_mmsi = list(np.random.choice(unique_mmsi, size=sample_vessels, replace=False))
        logger.info(f"Sampling {sample_vessels:,} vessels")

    logger.info(f"Processing {len(unique_mmsi):,} vessels...")

    for mmsi in tqdm(unique_mmsi, desc="Creating sequences", unit="vessel"):
        vessel_df = grouped.get_group(mmsi).sort_values('BaseDateTime').reset_index(drop=True)

        if len(vessel_df) < sequence_length + 1:
            continue

        X_vessel = vessel_df[features].values.astype(np.float32)
        y_vessel = vessel_df[target_cols].values.astype(np.float32)

        for i in range(len(vessel_df) - sequence_length):
            X_list.append(X_vessel[i:i+sequence_length])
            y_list.append(y_vessel[i+sequence_length])
            mmsi_list.append(mmsi)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    mmsi_list = np.array(mmsi_list)

    logger.info(f"✓ Sequences: {len(X):,}, X shape: {X.shape}, y shape: {y.shape}")
    return X, y, features, mmsi_list


def split_data(X, y, mmsi_list, train_ratio=0.7, val_ratio=0.2):
    """Split data into train/val/test."""
    logger.info(f"\n{'='*80}\n[4/10] SPLITTING DATA\n{'='*80}")

    n_samples = len(X)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    logger.info(f"✓ Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    """Main pipeline execution."""
    logger.info("="*80)
    logger.info("ENHANCED EXPERIMENTAL MODELS PIPELINE WITH HAVERSINE & ADVANCED FORECASTING")
    logger.info("="*80)

    # Setup MLflow
    mlflow.set_experiment("Enhanced_Experimental_Models_v2")

    # Load data
    df = load_entire_dataset(start_date=3, end_date=8, sample_per_day=100000)

    # Add features including Haversine
    df = add_comprehensive_features(df)

    # Create sequences
    X, y, features, mmsi_list = create_sequences(df, sequence_length=12, sample_vessels=500)

    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, mmsi_list)

    # Train Tiny LSTM with Haversine
    logger.info(f"\n{'='*80}\n[5/10] TRAINING MODELS\n{'='*80}")

    with mlflow.start_run(run_name="Tiny_LSTM_Haversine"):
        model, best_val_loss = train_tiny_lstm_enhanced(X_train, y_train, X_val, y_val, epochs=100)

        # Log parameters
        mlflow.log_param("model_type", "Tiny LSTM with Haversine")
        mlflow.log_param("input_size", X_train.shape[2])
        mlflow.log_param("sequence_length", 12)
        mlflow.log_param("hidden_size", 32)
        mlflow.log_param("num_layers", 4)
        mlflow.log_param("best_val_loss", best_val_loss)

        # Evaluate
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
            y_pred = model(X_test_tensor).cpu().numpy()

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)

        logger.info(f"✓ Tiny LSTM Haversine: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    logger.info("\n✓ PIPELINE COMPLETE")
    logger.info(f"✓ Results logged to MLflow: mlruns/")


if __name__ == "__main__":
    main()

