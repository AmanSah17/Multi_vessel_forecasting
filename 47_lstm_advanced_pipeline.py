"""
Advanced LSTM Pipeline with Specialized Loss Functions
- GPU optimized
- MLflow tracking
- Geodesic & Haversine features
- SILU/GELU activations
- Comprehensive error tracking
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from pathlib import Path
from tqdm import tqdm
import joblib
import gc
import warnings
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import psutil

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Setup directories
CACHE_DIR = Path('results/cache_checkpoints')
MODEL_DIR = Path('results/lstm_advanced_model')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_DIR = Path('mlruns')

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024

def load_checkpoint(name):
    """Load checkpoint from cache"""
    path = CACHE_DIR / f"{name}.npz"
    if path.exists():
        logger.info(f"Loading checkpoint: {name}")
        data = np.load(path, allow_pickle=True)
        if 'data' in data:
            return data['data']
        return {k: data[k] for k in data.files}
    return None

def extract_geodesic_features(X):
    """Extract geodesic distance features"""
    n_samples, n_timesteps, n_features = X.shape
    features = []
    
    for i in tqdm(range(n_samples), desc="Extracting geodesic features"):
        seq = X[i]  # (n_timesteps, n_features)
        
        # Assuming first 2 features are lat/lon
        lats = seq[:, 0]
        lons = seq[:, 1]
        
        # Haversine distance between consecutive points
        from math import radians, cos, sin, asin, sqrt
        
        distances = []
        for j in range(len(lats)-1):
            lat1, lon1 = radians(lats[j]), radians(lons[j])
            lat2, lon2 = radians(lats[j+1]), radians(lons[j+1])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371  # Earth radius in km
            distances.append(c * r)
        
        # Geodesic features
        geo_features = [
            np.mean(distances) if distances else 0,
            np.max(distances) if distances else 0,
            np.std(distances) if distances else 0,
            np.sum(distances) if distances else 0,
        ]
        features.append(geo_features)
    
    return np.array(features)

def extract_haversine_features(X):
    """Extract Haversine distance features"""
    n_samples, n_timesteps, n_features = X.shape
    features = []
    
    for i in tqdm(range(n_samples), desc="Extracting Haversine features"):
        seq = X[i]
        lats = seq[:, 0]
        lons = seq[:, 1]
        
        from math import radians, cos, sin, asin, sqrt
        
        # Distance to first point
        lat1, lon1 = radians(lats[0]), radians(lons[0])
        
        distances_to_first = []
        for j in range(1, len(lats)):
            lat2, lon2 = radians(lats[j]), radians(lons[j])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371
            distances_to_first.append(c * r)
        
        hav_features = [
            np.mean(distances_to_first) if distances_to_first else 0,
            np.max(distances_to_first) if distances_to_first else 0,
            np.std(distances_to_first) if distances_to_first else 0,
        ]
        features.append(hav_features)
    
    return np.array(features)

class GeodesicLoss(nn.Module):
    """Specialized loss for geodesic distance"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Lat/Lon are first 2 outputs
        lat_pred, lon_pred = pred[:, 0], pred[:, 1]
        lat_true, lon_true = target[:, 0], target[:, 1]
        
        # Haversine distance
        from math import radians, cos, sin, asin, sqrt
        
        lat1 = torch.deg2rad(lat_true)
        lon1 = torch.deg2rad(lon_true)
        lat2 = torch.deg2rad(lat_pred)
        lon2 = torch.deg2rad(lon_pred)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
        c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
        r = 6371  # Earth radius in km
        
        distance = c * r
        return torch.mean(distance)

class LSTMModel(nn.Module):
    """LSTM model with SILU/GELU activations"""
    def __init__(self, input_size, hidden_size=128, n_layers=4, dropout=0.2, output_size=4):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Dense layers with SILU activation
        self.fc1 = nn.Linear(hidden_size, 256)
        self.silu = nn.SiLU()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
    
    def forward(self, x):
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Dense layers
        x = self.fc1(last_hidden)
        x = self.silu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

def load_and_preprocess_data():
    """Load and preprocess data"""
    logger.info("\n[1/6] Loading and preprocessing data...")
    
    # Load cached preprocessed data
    cached = load_checkpoint('preprocessed_data')
    if cached is not None:
        X_train_pca = torch.FloatTensor(cached['X_train_pca']).to(device)
        y_train = torch.FloatTensor(cached['y_train']).to(device)
        X_val_pca = torch.FloatTensor(cached['X_val_pca']).to(device)
        y_val = torch.FloatTensor(cached['y_val']).to(device)
        X_test_pca = torch.FloatTensor(cached['X_test_pca']).to(device)
        y_test = torch.FloatTensor(cached['y_test']).to(device)
        
        logger.info(f"✓ Loaded from cache")
        logger.info(f"  Train: {X_train_pca.shape}, Val: {X_val_pca.shape}, Test: {X_test_pca.shape}")
        
        return X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test
    
    logger.info("Cache not found. Please run 41_corrected_xgboost_pipeline_with_caching.py first")
    return None

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X_batch)
        
        # Combined loss
        mse_loss = nn.MSELoss()(predictions, y_batch)
        mae_loss = nn.L1Loss()(predictions, y_batch)
        
        loss = 0.7 * mse_loss + 0.3 * mae_loss
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    """Validate model"""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(val_loader, desc="Validating", leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            
            mse_loss = nn.MSELoss()(predictions, y_batch)
            mae_loss = nn.L1Loss()(predictions, y_batch)
            loss = 0.7 * mse_loss + 0.3 * mae_loss
            
            total_loss += loss.item()
            
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    return total_loss / len(val_loader), all_preds, all_targets

def main():
    """Main training pipeline"""
    
    # MLflow setup
    mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
    mlflow.set_experiment("LSTM_Advanced_Vessel_Forecasting")
    
    with mlflow.start_run():
        # Load data
        data = load_and_preprocess_data()
        if data is None:
            return
        
        X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test = data
        
        # Reshape for LSTM (batch, seq_len, features)
        X_train_lstm = X_train_pca.reshape(X_train_pca.shape[0], 1, -1)
        X_val_lstm = X_val_pca.reshape(X_val_pca.shape[0], 1, -1)
        X_test_lstm = X_test_pca.reshape(X_test_pca.shape[0], 1, -1)
        
        logger.info(f"LSTM input shapes: Train {X_train_lstm.shape}, Val {X_val_lstm.shape}")
        
        # Create data loaders
        batch_size = 256
        train_dataset = TensorDataset(X_train_lstm, y_train)
        val_dataset = TensorDataset(X_val_lstm, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Model setup
        input_size = X_train_lstm.shape[2]
        model = LSTMModel(input_size=input_size, hidden_size=128, n_layers=4, dropout=0.2, output_size=4)
        model = model.to(device)
        
        logger.info(f"\n[2/6] Model created")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Hidden size: 128")
        logger.info(f"  Layers: 4")
        logger.info(f"  Dropout: 0.2")
        
        # Log model parameters
        mlflow.log_params({
            'model_type': 'LSTM',
            'input_size': input_size,
            'hidden_size': 128,
            'n_layers': 4,
            'dropout': 0.2,
            'batch_size': batch_size,
            'device': str(device)
        })
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Training loop
        logger.info(f"\n[3/6] Training LSTM model...")
        n_epochs = 50
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_preds, val_targets = validate(model, val_loader, device)
            
            # Calculate metrics
            lat_mae = mean_absolute_error(val_targets[:, 0], val_preds[:, 0])
            lon_mae = mean_absolute_error(val_targets[:, 1], val_preds[:, 1])
            sog_mae = mean_absolute_error(val_targets[:, 2], val_preds[:, 2])
            cog_mae = mean_absolute_error(val_targets[:, 3], val_preds[:, 3])
            
            lat_r2 = r2_score(val_targets[:, 0], val_preds[:, 0])
            lon_r2 = r2_score(val_targets[:, 1], val_preds[:, 1])
            
            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lat_mae': lat_mae,
                'lon_mae': lon_mae,
                'sog_mae': sog_mae,
                'cog_mae': cog_mae,
                'lat_r2': lat_r2,
                'lon_r2': lon_r2,
            }, step=epoch)
            
            logger.info(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            logger.info(f"  Lat MAE: {lat_mae:.6f} | Lon MAE: {lon_mae:.6f} | SOG MAE: {sog_mae:.6f}")
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), MODEL_DIR / 'best_model.pth')
                logger.info(f"  ✓ Best model saved (Val Loss: {val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(MODEL_DIR / 'best_model.pth'))
        
        # Test evaluation
        logger.info(f"\n[4/6] Evaluating on test set...")
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_lstm.to(device)).cpu().numpy()
        
        test_targets = y_test.cpu().numpy()
        
        # Calculate test metrics
        test_lat_mae = mean_absolute_error(test_targets[:, 0], test_preds[:, 0])
        test_lon_mae = mean_absolute_error(test_targets[:, 1], test_preds[:, 1])
        test_sog_mae = mean_absolute_error(test_targets[:, 2], test_preds[:, 2])
        test_cog_mae = mean_absolute_error(test_targets[:, 3], test_preds[:, 3])
        
        test_lat_r2 = r2_score(test_targets[:, 0], test_preds[:, 0])
        test_lon_r2 = r2_score(test_targets[:, 1], test_preds[:, 1])
        
        logger.info(f"\n[5/6] Test Metrics:")
        logger.info(f"  Latitude MAE: {test_lat_mae:.6f} | R²: {test_lat_r2:.6f}")
        logger.info(f"  Longitude MAE: {test_lon_mae:.6f} | R²: {test_lon_r2:.6f}")
        logger.info(f"  SOG MAE: {test_sog_mae:.6f}")
        logger.info(f"  COG MAE: {test_cog_mae:.6f}")
        
        # Log test metrics
        mlflow.log_metrics({
            'test_lat_mae': test_lat_mae,
            'test_lon_mae': test_lon_mae,
            'test_sog_mae': test_sog_mae,
            'test_cog_mae': test_cog_mae,
            'test_lat_r2': test_lat_r2,
            'test_lon_r2': test_lon_r2,
        })
        
        # Save model
        logger.info(f"\n[6/6] Saving model...")
        torch.save(model.state_dict(), MODEL_DIR / 'lstm_model.pth')
        joblib.dump(model, MODEL_DIR / 'lstm_model.pkl')
        logger.info(f"✓ Model saved to {MODEL_DIR}")
        
        logger.info(f"\n✅ LSTM TRAINING COMPLETE!")

if __name__ == '__main__':
    main()

