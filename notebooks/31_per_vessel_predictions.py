"""
Per-Vessel Predictions from Tiny LSTM Model
Generates individual vessel trajectory predictions and visualizations
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
from torch.utils.data import DataLoader, TensorDataset
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
OUTPUT_DIR = Path('results/per_vessel_predictions')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Device: {DEVICE}")
logger.info(f"Output directory: {OUTPUT_DIR}")


def load_model_and_data():
    """Load trained model and test data"""
    logger.info("\n" + "="*80)
    logger.info("[1/3] LOADING MODEL AND DATA")
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
    logger.info(f"✓ Feature names: {features}")
    
    # Split data (same as training: 70% train, 20% val, 10% test)
    n_train = int(0.7 * len(X))
    n_val = int(0.2 * len(X))
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    mmsi_test = mmsi_list[n_train + n_val:]
    
    logger.info(f"✓ Test set: {len(X_test):,} sequences from {len(np.unique(mmsi_test)):,} unique vessels")
    
    # Load model
    logger.info("Loading Tiny LSTM model...")
    model = TinyLSTMModel(input_size=len(features), hidden_size=32, num_layers=4, output_size=4, dropout=0.15)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    logger.info(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create test dataloader
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    logger.info(f"✓ Test DataLoader created with batch_size=256")
    
    return model, test_loader, X_test, y_test, mmsi_test, features


def generate_per_vessel_predictions(model, test_loader, X_test, y_test, mmsi_test):
    """Generate predictions for each vessel"""
    logger.info("\n" + "="*80)
    logger.info("[2/3] GENERATING PER-VESSEL PREDICTIONS")
    logger.info("="*80)
    
    all_predictions = []
    all_actuals = []
    all_mmsi = []
    
    logger.info("Running inference on test set...")
    with torch.no_grad():
        for batch_X, batch_y in tqdm(test_loader, desc="Inference", unit="batch"):
            batch_X = batch_X.to(DEVICE)
            batch_pred = model(batch_X).cpu().numpy()
            
            all_predictions.append(batch_pred)
            all_actuals.append(batch_y.numpy())
    
    predictions = np.vstack(all_predictions)
    actuals = np.vstack(all_actuals)
    
    logger.info(f"✓ Generated {len(predictions):,} predictions")
    logger.info(f"✓ Predictions shape: {predictions.shape}")
    logger.info(f"✓ Actuals shape: {actuals.shape}")
    
    # Group by vessel
    unique_mmsi = np.unique(mmsi_test)
    logger.info(f"✓ Found {len(unique_mmsi):,} unique vessels in test set")
    
    vessel_data = {}
    for mmsi in unique_mmsi:
        mask = mmsi_test == mmsi
        vessel_data[mmsi] = {
            'predictions': predictions[mask],
            'actuals': actuals[mask],
            'indices': np.where(mask)[0]
        }
    
    return vessel_data, unique_mmsi


def create_per_vessel_visualizations(vessel_data, unique_mmsi, n_vessels=20):
    """Create visualizations for top N vessels"""
    logger.info("\n" + "="*80)
    logger.info("[3/3] CREATING PER-VESSEL VISUALIZATIONS")
    logger.info("="*80)
    
    target_cols = ['LAT', 'LON', 'SOG', 'COG']
    
    # Select top N vessels by number of sequences
    vessel_sizes = [(mmsi, len(vessel_data[mmsi]['predictions'])) for mmsi in unique_mmsi]
    vessel_sizes.sort(key=lambda x: x[1], reverse=True)
    top_vessels = [mmsi for mmsi, _ in vessel_sizes[:n_vessels]]
    
    logger.info(f"Creating visualizations for top {n_vessels} vessels...")
    
    # Create individual vessel plots
    for vessel_idx, mmsi in enumerate(tqdm(top_vessels, desc="Creating plots", unit="vessel")):
        pred = vessel_data[mmsi]['predictions']
        actual = vessel_data[mmsi]['actuals']
        n_seq = len(pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Vessel MMSI: {mmsi} ({n_seq} sequences)', fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(target_cols):
            ax = axes[idx // 2, idx % 2]
            
            # Plot time series
            x = np.arange(n_seq)
            ax.plot(x, actual[:, idx], 'b-', linewidth=2, label='Actual', alpha=0.7)
            ax.plot(x, pred[:, idx], 'r--', linewidth=2, label='Predicted', alpha=0.7)
            
            # Calculate metrics
            mae = np.mean(np.abs(pred[:, idx] - actual[:, idx]))
            rmse = np.sqrt(np.mean((pred[:, idx] - actual[:, idx])**2))
            
            ax.set_xlabel('Sequence Index', fontsize=11)
            ax.set_ylabel(col, fontsize=11)
            ax.set_title(f'{col} (MAE={mae:.3f}, RMSE={rmse:.3f})', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = OUTPUT_DIR / f'vessel_{mmsi:09d}_predictions.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info(f"✓ Created {n_vessels} individual vessel plots")
    
    # Create trajectory plots (LAT vs LON)
    logger.info("Creating trajectory plots...")
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle('Vessel Trajectories: Predicted vs Actual (Top 20 Vessels)', fontsize=18, fontweight='bold')
    
    for vessel_idx, mmsi in enumerate(top_vessels):
        ax = axes[vessel_idx // 5, vessel_idx % 5]
        
        pred = vessel_data[mmsi]['predictions']
        actual = vessel_data[mmsi]['actuals']
        
        # Plot trajectory
        ax.plot(actual[:, 1], actual[:, 0], 'b-', linewidth=2, label='Actual', alpha=0.7)
        ax.plot(pred[:, 1], pred[:, 0], 'r--', linewidth=2, label='Predicted', alpha=0.7)
        
        # Mark start and end
        ax.scatter(actual[0, 1], actual[0, 0], color='blue', s=100, marker='o', label='Start (Actual)', zorder=5)
        ax.scatter(pred[0, 1], pred[0, 0], color='red', s=100, marker='s', label='Start (Pred)', zorder=5)
        ax.scatter(actual[-1, 1], actual[-1, 0], color='blue', s=100, marker='*', label='End (Actual)', zorder=5)
        ax.scatter(pred[-1, 1], pred[-1, 0], color='red', s=100, marker='^', label='End (Pred)', zorder=5)
        
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.set_title(f'MMSI: {mmsi}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if vessel_idx == 0:
            ax.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'all_vessel_trajectories.png', dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved: all_vessel_trajectories.png")
    plt.close()
    
    # Create summary statistics table
    logger.info("Creating summary statistics...")
    summary_data = []
    for mmsi in unique_mmsi:
        pred = vessel_data[mmsi]['predictions']
        actual = vessel_data[mmsi]['actuals']
        errors = pred - actual
        
        summary_data.append({
            'MMSI': mmsi,
            'Sequences': len(pred),
            'LAT_MAE': np.mean(np.abs(errors[:, 0])),
            'LON_MAE': np.mean(np.abs(errors[:, 1])),
            'SOG_MAE': np.mean(np.abs(errors[:, 2])),
            'COG_MAE': np.mean(np.abs(errors[:, 3])),
            'Overall_MAE': np.mean(np.abs(errors))
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Overall_MAE')
    summary_df.to_csv(OUTPUT_DIR / 'per_vessel_metrics.csv', index=False)
    logger.info(f"✓ Saved: per_vessel_metrics.csv")
    
    # Print top and worst performing vessels
    logger.info("\n" + "="*80)
    logger.info("TOP 10 BEST PERFORMING VESSELS (Lowest MAE)")
    logger.info("="*80)
    print(summary_df.head(10).to_string(index=False))
    
    logger.info("\n" + "="*80)
    logger.info("TOP 10 WORST PERFORMING VESSELS (Highest MAE)")
    logger.info("="*80)
    print(summary_df.tail(10).to_string(index=False))
    
    return summary_df


def main():
    """Main execution"""
    logger.info("\n" + "="*80)
    logger.info("PER-VESSEL PREDICTIONS FROM TINY LSTM MODEL")
    logger.info("="*80)
    
    # Load model and data
    model, test_loader, X_test, y_test, mmsi_test, features = load_model_and_data()
    
    # Generate per-vessel predictions
    vessel_data, unique_mmsi = generate_per_vessel_predictions(model, test_loader, X_test, y_test, mmsi_test)
    
    # Create visualizations
    summary_df = create_per_vessel_visualizations(vessel_data, unique_mmsi, n_vessels=20)
    
    logger.info("\n" + "="*80)
    logger.info("✓ PER-VESSEL PREDICTION COMPLETE")
    logger.info("="*80)
    logger.info(f"✓ Results saved to: {OUTPUT_DIR}")
    logger.info(f"✓ Total vessels analyzed: {len(unique_mmsi)}")
    logger.info(f"✓ Individual plots created: 20")
    logger.info(f"✓ Summary metrics: per_vessel_metrics.csv")


if __name__ == '__main__':
    main()

