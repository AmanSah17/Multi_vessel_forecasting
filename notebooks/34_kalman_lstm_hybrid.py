"""
Kalman-LSTM Hybrid Model for Vessel Trajectory Prediction
- Integrates Kalman Filter with LSTM for robust predictions
- Per-vessel time series forecasting
- Learning rate scheduler for adaptive training
- Per-vessel visualization with 5-minute intervals
- MLflow logging for experiment tracking
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/kalman_lstm_hybrid.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# ======================== KALMAN FILTER ========================

class KalmanFilter1D:
    """1D Kalman Filter for smoothing LSTM predictions."""
    def __init__(self, process_variance=0.01, measurement_variance=0.1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0
        self.estimate_error = 1.0
        self.kalman_gain = 0
    
    def update(self, measurement):
        """Update filter with measurement."""
        self.estimate_error += self.process_variance
        self.kalman_gain = self.estimate_error / (self.estimate_error + self.measurement_variance)
        self.estimate += self.kalman_gain * (measurement - self.estimate)
        self.estimate_error *= (1 - self.kalman_gain)
        return self.estimate


class KalmanFilterVessel:
    """Multi-dimensional Kalman Filter for vessel state."""
    def __init__(self, process_var=0.01, measurement_var=0.1):
        self.filters = {
            'LAT': KalmanFilter1D(process_var, measurement_var),
            'LON': KalmanFilter1D(process_var, measurement_var),
            'SOG': KalmanFilter1D(process_var, measurement_var),
            'COG': KalmanFilter1D(process_var, measurement_var)
        }
    
    def smooth(self, predictions):
        """Smooth LSTM predictions with Kalman Filter."""
        smoothed = {}
        for key, value in predictions.items():
            if key in self.filters:
                smoothed[key] = self.filters[key].update(value)
        return smoothed


# ======================== KALMAN-LSTM HYBRID MODEL ========================

class KalmanLSTMHybrid(nn.Module):
    """Hybrid model combining Kalman Filter with LSTM."""
    def __init__(self, input_size, hidden_size=64, num_layers=4, output_size=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ======================== TRAINING FUNCTIONS ========================

def train_kalman_lstm_hybrid(X_train, y_train, X_val, y_val, epochs=50, batch_size=64, lr=0.0001):
    """Train Kalman-LSTM Hybrid model with learning rate scheduler."""
    logger.info(f"\n{'='*80}\nTRAINING KALMAN-LSTM HYBRID MODEL\n{'='*80}")
    logger.info(f"Config: input_size={X_train.shape[2]}, hidden_size=32, num_layers=2, batch_size={batch_size}, lr={lr}, epochs={epochs}")
    
    input_size = X_train.shape[2]
    model = KalmanLSTMHybrid(input_size=input_size, hidden_size=32, num_layers=2).to(DEVICE)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for X_batch, y_batch in pbar:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                
                optimizer.zero_grad()
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation with smaller batch size to avoid OOM
        model.eval()
        val_loss = 0
        val_batch_size = 32
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            for i in range(0, len(X_val_tensor), val_batch_size):
                X_val_batch = X_val_tensor[i:i+val_batch_size].to(DEVICE)
                y_val_batch = y_val_tensor[i:i+val_batch_size].to(DEVICE)
                val_pred_batch = model(X_val_batch)
                val_loss += criterion(val_pred_batch, y_val_batch).item() * len(X_val_batch)
            val_loss /= len(X_val_tensor)
        
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'results/models/kalman_lstm_hybrid.pt')
            logger.info(f"  -> Best model saved (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    return model, best_val_loss, train_losses, val_losses


def apply_kalman_smoothing(predictions, process_var=0.01, measurement_var=0.1):
    """Apply Kalman Filter smoothing to LSTM predictions."""
    logger.info("Applying Kalman Filter smoothing to predictions...")
    
    smoothed_predictions = []
    
    for pred in tqdm(predictions, desc="Kalman Smoothing", unit="sample"):
        kf = KalmanFilterVessel(process_var=process_var, measurement_var=measurement_var)
        
        pred_dict = {
            'LAT': pred[0],
            'LON': pred[1],
            'SOG': pred[2],
            'COG': pred[3]
        }
        
        smoothed = kf.smooth(pred_dict)
        smoothed_predictions.append([smoothed['LAT'], smoothed['LON'], smoothed['SOG'], smoothed['COG']])
    
    return np.array(smoothed_predictions)


def evaluate_hybrid_model(X_test, y_test, model, apply_kalman=True):
    """Evaluate Kalman-LSTM Hybrid model."""
    logger.info(f"\n{'='*80}\nEVALUATING KALMAN-LSTM HYBRID MODEL\n{'='*80}")

    model.eval()
    batch_size = 32
    lstm_predictions = []

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        for i in range(0, len(X_test_tensor), batch_size):
            X_batch = X_test_tensor[i:i+batch_size].to(DEVICE)
            batch_pred = model(X_batch).cpu().numpy()
            lstm_predictions.append(batch_pred)

    lstm_predictions = np.vstack(lstm_predictions)

    # Apply Kalman smoothing
    if apply_kalman:
        predictions = apply_kalman_smoothing(lstm_predictions)
    else:
        predictions = lstm_predictions

    # Compute metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    logger.info(f"Kalman-LSTM Hybrid: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    return predictions, {'MAE': mae, 'RMSE': rmse, 'R2': r2}


# ======================== VISUALIZATION FUNCTIONS ========================

def plot_per_vessel_predictions(X_test, y_test, predictions, mmsi_test, top_n=20):
    """Plot per-vessel predictions with 5-minute intervals."""
    logger.info(f"\n{'='*80}\nCREATING PER-VESSEL VISUALIZATIONS\n{'='*80}")
    
    output_dir = Path('results/kalman_lstm_predictions')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    unique_mmsi = np.unique(mmsi_test)[:top_n]
    
    for mmsi in tqdm(unique_mmsi, desc="Creating plots", unit="vessel"):
        mask = mmsi_test == mmsi
        indices = np.where(mask)[0]
        
        if len(indices) < 2:
            continue
        
        # Get data for this vessel
        vessel_y = y_test[indices]
        vessel_pred = predictions[indices]
        
        # Create timestamps (5-minute intervals)
        timestamps = np.arange(len(vessel_y)) * 5  # minutes
        
        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Vessel {mmsi} - Kalman-LSTM Hybrid Predictions', fontsize=16, fontweight='bold')
        
        # LAT
        axes[0, 0].plot(timestamps, vessel_y[:, 0], 'b-', label='Actual', linewidth=2, alpha=0.7)
        axes[0, 0].plot(timestamps, vessel_pred[:, 0], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('Latitude (degrees)')
        axes[0, 0].set_title('Latitude Prediction')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # LON
        axes[0, 1].plot(timestamps, vessel_y[:, 1], 'b-', label='Actual', linewidth=2, alpha=0.7)
        axes[0, 1].plot(timestamps, vessel_pred[:, 1], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('Longitude (degrees)')
        axes[0, 1].set_title('Longitude Prediction')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # SOG
        axes[1, 0].plot(timestamps, vessel_y[:, 2], 'b-', label='Actual', linewidth=2, alpha=0.7)
        axes[1, 0].plot(timestamps, vessel_pred[:, 2], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('Speed Over Ground (knots)')
        axes[1, 0].set_title('SOG Prediction')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # COG
        axes[1, 1].plot(timestamps, vessel_y[:, 3], 'b-', label='Actual', linewidth=2, alpha=0.7)
        axes[1, 1].plot(timestamps, vessel_pred[:, 3], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        axes[1, 1].set_xlabel('Time (minutes)')
        axes[1, 1].set_ylabel('Course Over Ground (degrees)')
        axes[1, 1].set_title('COG Prediction')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'vessel_{mmsi}_kalman_lstm.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Saved {len(unique_mmsi)} vessel plots to {output_dir}")


# ======================== MAIN PIPELINE ========================

def main():
    """Main pipeline execution."""
    logger.info("="*80)
    logger.info("KALMAN-LSTM HYBRID MODEL FOR VESSEL TRAJECTORY PREDICTION")
    logger.info("="*80)

    # Setup MLflow
    mlflow.set_experiment("Kalman_LSTM_Hybrid_v1")

    # Load cached test data
    logger.info("\n[1/5] Loading test data...")
    cache_file = Path('results/cache/seq_cache_len12_sampled_3pct.npz')

    if not cache_file.exists():
        logger.error(f"Cache file not found: {cache_file}")
        return

    data = np.load(cache_file, allow_pickle=True)
    X = data['X']
    y = data['y']
    mmsi_list = data['mmsi_list']

    # Split data
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.2)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    mmsi_test = mmsi_list[train_size+val_size:]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train model
    logger.info("\n[2/5] Training Kalman-LSTM Hybrid model...")
    with mlflow.start_run(run_name="Kalman_LSTM_Training"):
        model, best_val_loss, train_losses, val_losses = train_kalman_lstm_hybrid(
            X_train, y_train, X_val, y_val,
            epochs=50, batch_size=64, lr=0.0001
        )

        # Log parameters
        mlflow.log_params({
            "model_type": "Kalman-LSTM Hybrid",
            "input_size": X_train.shape[2],
            "hidden_size": 64,
            "num_layers": 4,
            "learning_rate": 0.0001,
            "epochs": 50,
            "batch_size": 256,
            "scheduler": "ReduceLROnPlateau",
            "best_val_loss": best_val_loss
        })

        # Log metrics
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("final_train_loss", train_losses[-1])
        mlflow.log_metric("final_val_loss", val_losses[-1])

    # Evaluate model
    logger.info("\n[3/5] Evaluating model...")
    with mlflow.start_run(run_name="Kalman_LSTM_Evaluation"):
        predictions, metrics = evaluate_hybrid_model(X_test, y_test, model, apply_kalman=True)

        mlflow.log_metrics({
            "test_mae": metrics['MAE'],
            "test_rmse": metrics['RMSE'],
            "test_r2": metrics['R2']
        })

        logger.info(f"Test Results: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")

    # Create visualizations
    logger.info("\n[4/5] Creating per-vessel visualizations...")
    plot_per_vessel_predictions(X_test, y_test, predictions, mmsi_test, top_n=20)

    # Save results
    logger.info("\n[5/5] Saving results...")
    results_df = pd.DataFrame({
        'MMSI': mmsi_test,
        'pred_LAT': predictions[:, 0],
        'pred_LON': predictions[:, 1],
        'pred_SOG': predictions[:, 2],
        'pred_COG': predictions[:, 3],
        'actual_LAT': y_test[:, 0],
        'actual_LON': y_test[:, 1],
        'actual_SOG': y_test[:, 2],
        'actual_COG': y_test[:, 3]
    })

    results_df.to_csv('results/kalman_lstm_predictions.csv', index=False)
    logger.info("Results saved to results/kalman_lstm_predictions.csv")

    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Test MAE: {metrics['MAE']:.4f}")
    logger.info(f"Test RMSE: {metrics['RMSE']:.4f}")
    logger.info(f"Test R2: {metrics['R2']:.4f}")
    logger.info("Results logged to MLflow: mlruns/")


if __name__ == "__main__":
    main()

