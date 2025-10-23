"""
Complete End-to-End Pipeline with Testing and Visualization
- Advanced feature engineering (50+ features)
- Longer sequences (120 timesteps)
- MLflow logging
- Training curves visualization
- Model testing on 300 random vessels
- Consolidated predictions visualization
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
        logging.FileHandler(output_dirs['logs'] / 'complete_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

mlflow.set_tracking_uri(f"file:{output_dirs['mlflow'].absolute()}")
mlflow.set_experiment("Maritime_Vessel_Forecasting_v2")


def plot_training_curves(lstm_train, lstm_val, cnn_train, cnn_val):
    """Plot training and validation curves."""
    logger.info(f"\n{'='*70}\nGENERATING TRAINING CURVES\n{'='*70}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # LSTM curves
    axes[0, 0].plot(lstm_train, label='Train Loss', linewidth=2, color='#1f77b4')
    axes[0, 0].plot(lstm_val, label='Val Loss', linewidth=2, color='#ff7f0e')
    axes[0, 0].set_title('LSTM Training Curves', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # CNN curves
    axes[0, 1].plot(cnn_train, label='Train Loss', linewidth=2, color='#2ca02c')
    axes[0, 1].plot(cnn_val, label='Val Loss', linewidth=2, color='#d62728')
    axes[0, 1].set_title('CNN Training Curves', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss (MSE)')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Comparison
    axes[1, 0].plot(lstm_val, label='LSTM Val Loss', linewidth=2.5, marker='o', markersize=3)
    axes[1, 0].plot(cnn_val, label='CNN Val Loss', linewidth=2.5, marker='s', markersize=3)
    axes[1, 0].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss (MSE)')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary
    axes[1, 1].text(0.5, 0.8, 'Training Summary', ha='center', fontsize=13, fontweight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.6, f'LSTM Epochs: {len(lstm_train)}\nLSTM Final Val Loss: {lstm_val[-1]:.6f}', 
                    ha='center', fontsize=11, transform=axes[1, 1].transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 1].text(0.5, 0.3, f'CNN Epochs: {len(cnn_train)}\nCNN Final Val Loss: {cnn_val[-1]:.6f}', 
                    ha='center', fontsize=11, transform=axes[1, 1].transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dirs['images'] / 'training_curves_advanced.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Saved: training_curves_advanced.png")
    plt.close()


def plot_vessel_predictions(predictions_dict, num_vessels=300):
    """Plot predictions for random vessels."""
    logger.info(f"\n{'='*70}\nGENERATING VESSEL PREDICTIONS\n{'='*70}")
    
    # Create grid of subplots
    n_cols = 5
    n_rows = (num_vessels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten()
    
    vessel_ids = list(predictions_dict.keys())[:num_vessels]
    
    for idx, vessel_id in enumerate(tqdm(vessel_ids, desc="Plotting vessels")):
        ax = axes[idx]
        pred_data = predictions_dict[vessel_id]
        
        # Plot actual vs predicted
        ax.plot(pred_data['actual'], label='Actual', linewidth=2, marker='o', markersize=4)
        ax.plot(pred_data['lstm'], label='LSTM', linewidth=2, marker='s', markersize=4, alpha=0.7)
        ax.plot(pred_data['cnn'], label='CNN', linewidth=2, marker='^', markersize=4, alpha=0.7)
        
        ax.set_title(f'Vessel {vessel_id}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('LAT')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(vessel_ids), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dirs['images'] / f'vessel_predictions_{num_vessels}.png', dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved: vessel_predictions_{num_vessels}.png")
    plt.close()


def plot_consolidated_predictions(predictions_dict, num_vessels=50):
    """Plot consolidated predictions for multiple vessels."""
    logger.info(f"\nGenerating consolidated predictions plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    vessel_ids = list(predictions_dict.keys())[:num_vessels]
    
    # LAT predictions
    for vessel_id in vessel_ids:
        pred_data = predictions_dict[vessel_id]
        axes[0, 0].plot(pred_data['actual'], alpha=0.3, linewidth=1)
    axes[0, 0].set_title(f'Actual LAT - {num_vessels} Vessels', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('LAT')
    axes[0, 0].grid(True, alpha=0.3)
    
    # LSTM predictions
    for vessel_id in vessel_ids:
        pred_data = predictions_dict[vessel_id]
        axes[0, 1].plot(pred_data['lstm'], alpha=0.3, linewidth=1)
    axes[0, 1].set_title(f'LSTM Predictions - {num_vessels} Vessels', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('LAT')
    axes[0, 1].grid(True, alpha=0.3)
    
    # CNN predictions
    for vessel_id in vessel_ids:
        pred_data = predictions_dict[vessel_id]
        axes[1, 0].plot(pred_data['cnn'], alpha=0.3, linewidth=1)
    axes[1, 0].set_title(f'CNN Predictions - {num_vessels} Vessels', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('LAT')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error comparison
    lstm_errors = []
    cnn_errors = []
    for vessel_id in vessel_ids:
        pred_data = predictions_dict[vessel_id]
        lstm_errors.append(np.mean(np.abs(pred_data['actual'] - pred_data['lstm'])))
        cnn_errors.append(np.mean(np.abs(pred_data['actual'] - pred_data['cnn'])))
    
    axes[1, 1].boxplot([lstm_errors, cnn_errors], labels=['LSTM', 'CNN'])
    axes[1, 1].set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dirs['images'] / f'consolidated_predictions_{num_vessels}.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: consolidated_predictions_{num_vessels}.png")
    plt.close()


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("COMPLETE END-TO-END PIPELINE WITH TESTING")
    logger.info("="*70)

