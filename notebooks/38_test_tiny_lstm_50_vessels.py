"""
Test Predictions on 50 Random Vessels using Tiny LSTM Model
- Load pre-trained Tiny LSTM model (best performer)
- Make predictions on test set (NO TRAINING)
- Plot trajectory predictions vs actual values for 50 random vessels
- 5-minute interval time series visualization
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_tiny_lstm_50_vessels.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")


class TinyLSTMModel(nn.Module):
    """Tiny LSTM for ultra-fast training."""
    def __init__(self, input_size=28, hidden_size=32, num_layers=4, output_size=4, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def load_test_data():
    """Load test data from cache."""
    logger.info("\n[1/4] Loading test data...")
    
    cache_file = 'results/cache/seq_cache_len12_sampled_3pct.npz'
    data = np.load(cache_file)
    X = data['X']
    y = data['y']
    mmsi_list = data['mmsi_list']
    
    # Split to get test set (70% train, 20% val, 10% test)
    n_train = int(0.7 * len(X))
    n_val = int(0.2 * len(X))
    
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]
    mmsi_test = mmsi_list[n_train+n_val:]
    
    logger.info(f"Test set loaded: X={X_test.shape}, y={y_test.shape}")
    logger.info(f"Unique vessels in test set: {len(np.unique(mmsi_test))}")
    
    return X_test, y_test, mmsi_test


def load_tiny_lstm_model():
    """Load pre-trained Tiny LSTM model."""
    logger.info("\n[2/4] Loading Tiny LSTM model...")
    
    model_path = 'results/models/best_tiny_lstm.pt'
    
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        return None
    
    model = TinyLSTMModel(input_size=28, hidden_size=32, num_layers=4, output_size=4, dropout=0.15).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    logger.info("[OK] Tiny LSTM model loaded successfully")
    return model


def make_predictions(X_test, model):
    """Make predictions using Tiny LSTM model."""
    logger.info("\n[3/4] Making predictions on test set...")
    
    predictions = []
    batch_size = 256
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X_test), batch_size), desc="Predicting", unit="batch"):
            batch_X = X_test[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch_X).to(DEVICE)
            batch_pred = model(batch_tensor).cpu().numpy()
            predictions.append(batch_pred)
    
    predictions = np.vstack(predictions)
    logger.info(f"Predictions shape: {predictions.shape}")
    
    return predictions


def plot_vessel_trajectory(y_test, predictions, mmsi_test, vessel_mmsi, output_dir):
    """Plot trajectory predictions for a single vessel."""
    mask = mmsi_test == vessel_mmsi
    indices = np.where(mask)[0]
    
    if len(indices) < 2:
        return False
    
    vessel_y = y_test[indices]
    vessel_pred = predictions[indices]
    timestamps = np.arange(len(vessel_y)) * 5  # 5-minute intervals in minutes
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Vessel {vessel_mmsi} - Tiny LSTM Trajectory Predictions vs Actual Values\n(5-minute intervals)', 
                 fontsize=14, fontweight='bold')
    
    variables = ['Latitude', 'Longitude', 'SOG (knots)', 'COG (degrees)']
    
    for idx, (ax, var) in enumerate(zip(axes.flat, variables)):
        # Actual values
        ax.plot(timestamps, vessel_y[:, idx], 'b-', label='Actual', linewidth=2.5, 
                alpha=0.9, marker='o', markersize=5, markerfacecolor='lightblue', markeredgewidth=1)
        
        # Predicted values
        ax.plot(timestamps, vessel_pred[:, idx], 'r--', label='Tiny LSTM Prediction', 
               linewidth=2, alpha=0.8, color='#FF6B6B', marker='s', markersize=4)
        
        ax.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
        ax.set_ylabel(var, fontsize=11, fontweight='bold')
        ax.set_title(var, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'vessel_{vessel_mmsi}_trajectory.png', dpi=120, bbox_inches='tight')
    plt.close()
    
    return True


def calculate_metrics(y_test, predictions):
    """Calculate prediction metrics."""
    logger.info("\nCalculating prediction metrics...")
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    
    logger.info(f"Tiny LSTM: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    
    return metrics


def main():
    """Main pipeline - NO TRAINING, ONLY TESTING."""
    logger.info("\n" + "="*80)
    logger.info("TEST PREDICTIONS ON 50 RANDOM VESSELS")
    logger.info("Using Pre-trained Tiny LSTM Model (Best Performer)")
    logger.info("="*80)
    
    # Load test data
    X_test, y_test, mmsi_test = load_test_data()
    
    # Load trained model
    model = load_tiny_lstm_model()
    if model is None:
        logger.error("Failed to load model. Exiting.")
        return
    
    # Make predictions
    predictions = make_predictions(X_test, model)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, predictions)
    
    # Create output directory
    output_dir = Path('results/test_tiny_lstm_50_vessels')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations for 50 random vessels
    logger.info("\n[4/4] Generating trajectory plots for 50 random vessels...")
    
    unique_mmsi = np.unique(mmsi_test)
    selected_mmsi = np.random.choice(unique_mmsi, size=min(50, len(unique_mmsi)), replace=False)
    
    logger.info(f"Selected {len(selected_mmsi)} random vessels for visualization")
    
    success_count = 0
    for mmsi in tqdm(selected_mmsi, desc="Creating trajectory plots", unit="vessel"):
        if plot_vessel_trajectory(y_test, predictions, mmsi_test, mmsi, output_dir):
            success_count += 1
    
    # Save metrics summary
    metrics_df = pd.DataFrame([metrics], index=['Tiny LSTM'])
    metrics_df.to_csv(output_dir / 'model_metrics.csv')
    logger.info(f"\nMetrics saved to {output_dir / 'model_metrics.csv'}")
    
    # Save detailed predictions
    results_df = pd.DataFrame({
        'MMSI': mmsi_test,
        'actual_LAT': y_test[:, 0],
        'actual_LON': y_test[:, 1],
        'actual_SOG': y_test[:, 2],
        'actual_COG': y_test[:, 3],
        'pred_LAT': predictions[:, 0],
        'pred_LON': predictions[:, 1],
        'pred_SOG': predictions[:, 2],
        'pred_COG': predictions[:, 3],
    })
    
    results_df.to_csv(output_dir / 'all_predictions.csv', index=False)
    logger.info(f"Predictions saved to {output_dir / 'all_predictions.csv'}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("[COMPLETE] TEST PREDICTIONS COMPLETE")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Trajectory plots created: {success_count}/50 vessels")
    logger.info(f"\nModel Performance:")
    logger.info(f"\n{metrics_df.to_string()}")
    logger.info("\nAll trajectory plots saved to: " + str(output_dir))


if __name__ == "__main__":
    main()

