"""
Tiny LSTM Model Predictions on 300 Random Vessels
Generates predictions and visualizations for vessel trajectories
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define TinyLSTM model (matching training architecture)
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

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = Path('results/models/best_tiny_lstm.pt')
CACHE_FILE = Path('results/cache/seq_cache_len12_sampled_3pct.npz')
OUTPUT_DIR = Path('results/predictions_300_vessels')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Device: {DEVICE}")
logger.info(f"Output directory: {OUTPUT_DIR}")


def load_model_and_data():
    """Load trained model and test data"""
    logger.info("\n" + "="*80)
    logger.info("[1/4] LOADING MODEL AND DATA")
    logger.info("="*80)
    
    # Load cached sequences
    logger.info("Loading cached sequences...")
    data = np.load(CACHE_FILE, allow_pickle=True)
    X = data['X']
    y = data['y']
    features_obj = data['features']
    if isinstance(features_obj, np.ndarray):
        features = features_obj.tolist() if features_obj.ndim > 0 else features_obj.item()
    else:
        features = features_obj
    mmsi_list = data['mmsi_list']
    
    logger.info(f"✓ Loaded {len(X):,} sequences")
    logger.info(f"✓ Features: {len(features)}")
    
    # Split data (same as training)
    n_train = int(0.7 * len(X))
    n_val = int(0.2 * len(X))
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    mmsi_test = mmsi_list[n_train + n_val:]
    
    logger.info(f"✓ Test set: {len(X_test):,} sequences")
    
    # Load model
    logger.info("Loading Tiny LSTM model...")
    model = TinyLSTMModel(input_size=len(features), hidden_size=32, num_layers=4, output_size=4, dropout=0.15)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    logger.info(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, X_test, y_test, mmsi_test, features


def get_300_random_vessels(mmsi_test, X_test, y_test):
    """Select 300 random unique vessels from test set"""
    logger.info("\n" + "="*80)
    logger.info("[2/4] SELECTING 300 RANDOM VESSELS")
    logger.info("="*80)
    
    unique_mmsi = np.unique(mmsi_test)
    n_vessels = min(300, len(unique_mmsi))
    selected_mmsi = np.random.choice(unique_mmsi, size=n_vessels, replace=False)
    
    logger.info(f"✓ Selected {n_vessels} unique vessels")
    
    # Get indices for selected vessels
    mask = np.isin(mmsi_test, selected_mmsi)
    X_selected = X_test[mask]
    y_selected = y_test[mask]
    mmsi_selected = mmsi_test[mask]
    
    logger.info(f"✓ Selected {len(X_selected):,} sequences from {n_vessels} vessels")
    
    return X_selected, y_selected, mmsi_selected, selected_mmsi


def generate_predictions(model, X_selected, y_selected, mmsi_selected):
    """Generate predictions for selected vessels"""
    logger.info("\n" + "="*80)
    logger.info("[3/4] GENERATING PREDICTIONS")
    logger.info("="*80)
    
    predictions = []
    actuals = []
    mmsi_pred = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X_selected), 256), desc="Predicting", unit="batch"):
            batch_X = X_selected[i:i+256]
            batch_y = y_selected[i:i+256]
            batch_mmsi = mmsi_selected[i:i+256]
            
            X_batch = torch.from_numpy(batch_X).float().to(DEVICE)
            y_pred = model(X_batch).cpu().numpy()
            
            predictions.append(y_pred)
            actuals.append(batch_y)
            mmsi_pred.append(batch_mmsi)
    
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    mmsi_pred = np.hstack(mmsi_pred)
    
    logger.info(f"✓ Generated {len(predictions):,} predictions")
    logger.info(f"✓ Predictions shape: {predictions.shape}")
    logger.info(f"✓ Actuals shape: {actuals.shape}")
    
    return predictions, actuals, mmsi_pred


def create_visualizations(predictions, actuals, mmsi_pred):
    """Create comprehensive visualizations"""
    logger.info("\n" + "="*80)
    logger.info("[4/4] CREATING VISUALIZATIONS")
    logger.info("="*80)
    
    # Target columns: LAT, LON, SOG, COG
    target_cols = ['LAT', 'LON', 'SOG', 'COG']
    
    # 1. Prediction vs Actual scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Tiny LSTM: Predicted vs Actual Values (300 Vessels)', fontsize=16, fontweight='bold')
    
    for idx, col in enumerate(target_cols):
        ax = axes[idx // 2, idx % 2]
        ax.scatter(actuals[:, idx], predictions[:, idx], alpha=0.3, s=10)
        
        # Add diagonal line
        min_val = min(actuals[:, idx].min(), predictions[:, idx].min())
        max_val = max(actuals[:, idx].max(), predictions[:, idx].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel(f'Actual {col}', fontsize=11)
        ax.set_ylabel(f'Predicted {col}', fontsize=11)
        ax.set_title(f'{col} Prediction Accuracy', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: predictions_vs_actual.png")
    plt.close()
    
    # 2. Error distribution
    errors = predictions - actuals
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Tiny LSTM: Prediction Error Distribution (300 Vessels)', fontsize=16, fontweight='bold')
    
    for idx, col in enumerate(target_cols):
        ax = axes[idx // 2, idx % 2]
        ax.hist(errors[:, idx], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(errors[:, idx].mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {errors[:, idx].mean():.3f}')
        ax.set_xlabel(f'Error ({col})', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{col} Error Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'error_distribution.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: error_distribution.png")
    plt.close()
    
    # 3. Metrics by target variable
    mae = np.mean(np.abs(errors), axis=0)
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Tiny LSTM: Performance Metrics by Variable (300 Vessels)', fontsize=16, fontweight='bold')
    
    x_pos = np.arange(len(target_cols))
    
    axes[0].bar(x_pos, mae, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_ylabel('MAE', fontsize=12)
    axes[0].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(target_cols)
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mae):
        axes[0].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
    
    axes[1].bar(x_pos, rmse, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(target_cols)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(rmse):
        axes[1].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_by_variable.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: metrics_by_variable.png")
    plt.close()
    
    # 4. Sample vessel trajectories (top 9 vessels)
    unique_mmsi_pred = np.unique(mmsi_pred)[:9]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('Tiny LSTM: Sample Vessel Trajectory Predictions (9 Vessels)', fontsize=16, fontweight='bold')
    
    for vessel_idx, mmsi in enumerate(unique_mmsi_pred):
        ax = axes[vessel_idx // 3, vessel_idx % 3]
        mask = mmsi_pred == mmsi
        
        actual_lons = actuals[mask, 1]
        actual_lats = actuals[mask, 0]
        pred_lons = predictions[mask, 1]
        pred_lats = predictions[mask, 0]
        
        ax.plot(actual_lons, actual_lats, 'b-', linewidth=2, label='Actual', alpha=0.7)
        ax.plot(pred_lons, pred_lats, 'r--', linewidth=2, label='Predicted', alpha=0.7)
        ax.scatter(actual_lons[0], actual_lats[0], color='blue', s=100, marker='o', label='Start (Actual)', zorder=5)
        ax.scatter(pred_lons[0], pred_lats[0], color='red', s=100, marker='s', label='Start (Pred)', zorder=5)
        
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.set_title(f'Vessel MMSI: {mmsi}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sample_trajectories.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: sample_trajectories.png")
    plt.close()
    
    logger.info(f"\n✓ All visualizations saved to: {OUTPUT_DIR}")


def main():
    """Main execution"""
    logger.info("\n" + "="*80)
    logger.info("TINY LSTM PREDICTIONS ON 300 RANDOM VESSELS")
    logger.info("="*80)
    
    # Load model and data
    model, X_test, y_test, mmsi_test, features = load_model_and_data()
    
    # Select 300 random vessels
    X_selected, y_selected, mmsi_selected, selected_mmsi = get_300_random_vessels(mmsi_test, X_test, y_test)
    
    # Generate predictions
    predictions, actuals, mmsi_pred = generate_predictions(model, X_selected, y_selected, mmsi_selected)
    
    # Create visualizations
    create_visualizations(predictions, actuals, mmsi_pred)
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'MMSI': mmsi_pred,
        'Actual_LAT': actuals[:, 0],
        'Predicted_LAT': predictions[:, 0],
        'Actual_LON': actuals[:, 1],
        'Predicted_LON': predictions[:, 1],
        'Actual_SOG': actuals[:, 2],
        'Predicted_SOG': predictions[:, 2],
        'Actual_COG': actuals[:, 3],
        'Predicted_COG': predictions[:, 3],
    })
    
    csv_path = OUTPUT_DIR / 'predictions_300_vessels.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved predictions to: {csv_path}")
    
    # Print summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    logger.info(f"Total predictions: {len(predictions):,}")
    logger.info(f"Unique vessels: {len(selected_mmsi)}")
    logger.info(f"\nPrediction Errors:")
    errors = predictions - actuals
    for i, col in enumerate(['LAT', 'LON', 'SOG', 'COG']):
        mae = np.mean(np.abs(errors[:, i]))
        rmse = np.sqrt(np.mean(errors[:, i]**2))
        logger.info(f"  {col}: MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    logger.info("\n✓ PREDICTION COMPLETE")


if __name__ == '__main__':
    main()

