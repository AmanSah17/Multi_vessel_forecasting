"""
Predictions on 50 Random Vessels using Trained Models
- Load trained models (XGBoost, Neural Network, Random Forest)
- Make predictions on test set
- Plot results for 50 random vessels
- Compare model predictions with actual values
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
import pickle
import warnings
warnings.filterwarnings('ignore')

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/predictions_50_vessels.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NeuralNetworkRegressor(nn.Module):
    """Neural Network for regression."""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=4, dropout=0.2):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_data_and_models():
    """Load test data and trained models."""
    logger.info("\n[1/4] Loading data and models...")
    
    # Load test data
    cache_file = 'results/cache/seq_cache_len12_sampled_3pct.npz'
    data = np.load(cache_file)
    X = data['X']
    y = data['y']
    mmsi_list = data['mmsi_list']
    
    # Split to get test set
    n_train = int(0.7 * len(X))
    n_val = int(0.2 * len(X))
    
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]
    mmsi_test = mmsi_list[n_train+n_val:]
    
    logger.info(f"Test set: X={X_test.shape}, y={y_test.shape}")
    
    # Load PCA-transformed test data
    pca_file = 'results/advanced_pca_models/X_test_pca.npy'
    X_test_pca = np.load(pca_file)
    logger.info(f"PCA-transformed test data: {X_test_pca.shape}")
    
    # Load models
    models = {}
    
    # Load XGBoost
    xgb_file = 'results/advanced_pca_models/xgboost_model.pkl'
    with open(xgb_file, 'rb') as f:
        models['XGBoost'] = pickle.load(f)
    logger.info("Loaded XGBoost model")
    
    # Load Random Forest
    rf_file = 'results/advanced_pca_models/random_forest_model.pkl'
    with open(rf_file, 'rb') as f:
        models['Random Forest'] = pickle.load(f)
    logger.info("Loaded Random Forest model")
    
    # Load Neural Network
    nn_file = 'results/advanced_pca_models/neural_network_model.pt'
    nn_model = NeuralNetworkRegressor(input_size=6).to(DEVICE)
    nn_model.load_state_dict(torch.load(nn_file))
    nn_model.eval()
    models['Neural Network'] = nn_model
    logger.info("Loaded Neural Network model")
    
    return X_test_pca, y_test, mmsi_test, models


def make_predictions(X_test_pca, models):
    """Make predictions using all models."""
    logger.info("\n[2/4] Making predictions...")
    
    predictions = {}
    
    # XGBoost predictions
    predictions['XGBoost'] = models['XGBoost'].predict(X_test_pca)
    logger.info(f"XGBoost predictions: {predictions['XGBoost'].shape}")
    
    # Random Forest predictions
    predictions['Random Forest'] = models['Random Forest'].predict(X_test_pca)
    logger.info(f"Random Forest predictions: {predictions['Random Forest'].shape}")
    
    # Neural Network predictions
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_pca).to(DEVICE)
        nn_pred = models['Neural Network'](X_test_tensor).cpu().numpy()
    predictions['Neural Network'] = nn_pred
    logger.info(f"Neural Network predictions: {predictions['Neural Network'].shape}")
    
    return predictions


def plot_vessel_predictions(y_test, predictions, mmsi_test, vessel_mmsi, output_dir):
    """Plot predictions for a single vessel."""
    mask = mmsi_test == vessel_mmsi
    indices = np.where(mask)[0]
    
    if len(indices) < 2:
        return False
    
    vessel_y = y_test[indices]
    timestamps = np.arange(len(vessel_y)) * 5  # 5-minute intervals
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Vessel {vessel_mmsi} - Model Predictions Comparison', fontsize=14, fontweight='bold')
    
    variables = ['Latitude', 'Longitude', 'SOG (knots)', 'COG (degrees)']
    colors = {'XGBoost': '#FF6B6B', 'Random Forest': '#4ECDC4', 'Neural Network': '#45B7D1'}
    
    for idx, (ax, var) in enumerate(zip(axes.flat, variables)):
        # Actual values
        ax.plot(timestamps, vessel_y[:, idx], 'b-', label='Actual', linewidth=2.5, alpha=0.8, marker='o', markersize=4)
        
        # Model predictions
        for model_name, color in colors.items():
            vessel_pred = predictions[model_name][indices]
            ax.plot(timestamps, vessel_pred[:, idx], '--', label=model_name, linewidth=2, alpha=0.7, color=color)
        
        ax.set_xlabel('Time (minutes)', fontsize=10)
        ax.set_ylabel(var, fontsize=10)
        ax.set_title(var, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'vessel_{vessel_mmsi}_predictions.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    return True


def calculate_metrics(y_test, predictions, mmsi_test):
    """Calculate metrics for each model."""
    logger.info("\n[3/4] Calculating metrics...")
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    metrics_summary = {}
    
    for model_name, pred in predictions.items():
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        
        metrics_summary[model_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        logger.info(f"{model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    
    return metrics_summary


def main():
    """Main pipeline."""
    logger.info("\n" + "="*80)
    logger.info("PREDICTIONS ON 50 RANDOM VESSELS")
    logger.info("="*80)
    
    # Load data and models
    X_test_pca, y_test, mmsi_test, models = load_data_and_models()
    
    # Make predictions
    predictions = make_predictions(X_test_pca, models)
    
    # Calculate metrics
    metrics_summary = calculate_metrics(y_test, predictions, mmsi_test)
    
    # Create output directory
    output_dir = Path('results/predictions_50_vessels')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations for 50 random vessels
    logger.info("\n[4/4] Generating visualizations for 50 random vessels...")
    
    unique_mmsi = np.unique(mmsi_test)
    selected_mmsi = np.random.choice(unique_mmsi, size=min(50, len(unique_mmsi)), replace=False)
    
    success_count = 0
    for mmsi in tqdm(selected_mmsi, desc="Creating vessel plots", unit="vessel"):
        if plot_vessel_predictions(y_test, predictions, mmsi_test, mmsi, output_dir):
            success_count += 1
    
    # Save metrics summary
    metrics_df = pd.DataFrame(metrics_summary).T
    metrics_df.to_csv(output_dir / 'model_metrics.csv')
    logger.info(f"Metrics saved to {output_dir / 'model_metrics.csv'}")
    
    # Save predictions
    results_df = pd.DataFrame({
        'MMSI': mmsi_test,
        'actual_LAT': y_test[:, 0],
        'actual_LON': y_test[:, 1],
        'actual_SOG': y_test[:, 2],
        'actual_COG': y_test[:, 3],
        'xgb_LAT': predictions['XGBoost'][:, 0],
        'xgb_LON': predictions['XGBoost'][:, 1],
        'xgb_SOG': predictions['XGBoost'][:, 2],
        'xgb_COG': predictions['XGBoost'][:, 3],
        'rf_LAT': predictions['Random Forest'][:, 0],
        'rf_LON': predictions['Random Forest'][:, 1],
        'rf_SOG': predictions['Random Forest'][:, 2],
        'rf_COG': predictions['Random Forest'][:, 3],
        'nn_LAT': predictions['Neural Network'][:, 0],
        'nn_LON': predictions['Neural Network'][:, 1],
        'nn_SOG': predictions['Neural Network'][:, 2],
        'nn_COG': predictions['Neural Network'][:, 3],
    })
    results_df.to_csv(output_dir / 'all_predictions.csv', index=False)
    logger.info(f"Predictions saved to {output_dir / 'all_predictions.csv'}")
    
    logger.info("\n" + "="*80)
    logger.info("[COMPLETE] PREDICTIONS PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Visualizations created: {success_count}/50 vessels")
    logger.info(f"\nModel Performance Summary:")
    logger.info(f"\n{metrics_df.to_string()}")


if __name__ == "__main__":
    main()

