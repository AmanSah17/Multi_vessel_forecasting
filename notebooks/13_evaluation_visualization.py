"""
Evaluation & Visualization: Test Set Performance + Prediction Plots

Features:
- Evaluate on test set
- Plot actual vs predicted for 30 random vessels
- Multiple timestamps visualization
- MLflow logging of plots
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation_mlflow.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM for predicting LAT, LON, COG, Heading."""
    
    def __init__(self, input_size, hidden_size=128, output_size=4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


def evaluate_on_test_set(model, X_test, y_test, device):
    """Evaluate model on test set."""
    logger.info(f"\n{'='*70}")
    logger.info("[5/6] EVALUATING ON TEST SET")
    logger.info(f"{'='*70}")
    
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    with torch.no_grad():
        predictions = model(X_test_t).cpu().numpy()
    
    # Metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    logger.info(f"Test MAE:  {mae:.6f}")
    logger.info(f"Test RMSE: {rmse:.6f}")
    logger.info(f"Test R²:   {r2:.6f}")
    
    # Per-output metrics
    output_names = ['LAT', 'LON', 'SOG', 'COG']
    for i, name in enumerate(output_names):
        mae_i = mean_absolute_error(y_test[:, i], predictions[:, i])
        logger.info(f"  {name} MAE: {mae_i:.6f}")
    
    return predictions, {'mae': mae, 'rmse': rmse, 'r2': r2}


def plot_predictions_vs_actual(df, model, scaler, features, seq_length=30, n_vessels=30):
    """Plot predictions vs actual for random vessels."""
    logger.info(f"\n{'='*70}")
    logger.info("[6/6] PLOTTING PREDICTIONS VS ACTUAL")
    logger.info(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Select random vessels
    vessels = df['MMSI'].unique()
    selected_vessels = np.random.choice(vessels, min(n_vessels, len(vessels)), replace=False)
    
    logger.info(f"Selected {len(selected_vessels)} random vessels for visualization")
    
    # Create figure grid
    n_cols = 3
    n_rows = (len(selected_vessels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten()
    
    for idx, mmsi in enumerate(tqdm(selected_vessels, desc="Plotting vessels")):
        vessel_data = df[df['MMSI'] == mmsi].sort_values('BaseDateTime')[features].values
        
        if len(vessel_data) < seq_length + 1:
            continue
        
        # Create sequences
        X_vessel, y_vessel = [], []
        for i in range(len(vessel_data) - seq_length):
            X_vessel.append(vessel_data[i:i+seq_length])
            y_vessel.append(vessel_data[i+seq_length, :4])
        
        if len(X_vessel) == 0:
            continue
        
        X_vessel = np.array(X_vessel, dtype=np.float32)
        y_vessel = np.array(y_vessel, dtype=np.float32)
        
        # Normalize
        X_flat = X_vessel.reshape(-1, X_vessel.shape[-1])
        X_scaled = scaler.transform(X_flat).reshape(X_vessel.shape)
        
        # Predict
        X_t = torch.FloatTensor(X_scaled).to(device)
        with torch.no_grad():
            predictions = model(X_t).cpu().numpy()
        
        # Plot LAT vs LON (trajectory)
        ax = axes[idx]
        ax.plot(y_vessel[:, 1], y_vessel[:, 0], 'b-', linewidth=2, label='Actual', marker='o', markersize=4)
        ax.plot(predictions[:, 1], predictions[:, 0], 'r--', linewidth=2, label='Predicted', marker='s', markersize=4)
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.set_title(f'Vessel {mmsi} - Trajectory', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions_vs_actual_trajectories.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Trajectory plots saved")
    plt.show()
    
    # Plot time series for each output
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Predictions vs Actual - Time Series (First 500 Test Samples)', fontsize=14, fontweight='bold')
    
    output_names = ['LAT', 'LON', 'SOG', 'COG']
    for i, (ax, name) in enumerate(zip(axes.flatten(), output_names)):
        ax.plot(y_test[:500, i], 'b-', linewidth=2, label='Actual', marker='o', markersize=3)
        ax.plot(predictions[:500, i], 'r--', linewidth=2, label='Predicted', marker='s', markersize=3)
        ax.set_xlabel('Sample', fontsize=11)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(f'{name} - Actual vs Predicted', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions_vs_actual_timeseries.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Time series plots saved")
    plt.show()


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("EVALUATION & VISUALIZATION")
    logger.info("="*70)
    
    # This script should be run after 12_full_pipeline_mlflow.py
    # It will load the trained model and evaluate on test set
    
    logger.info("Note: Run 12_full_pipeline_mlflow.py first to train the model")
    logger.info("Then run this script to evaluate and visualize predictions")

