"""
Test Predictions on 50 Random Vessels using Last Trained Model Weights
- Load pre-trained models (XGBoost, Neural Network, Random Forest)
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
import pickle
import warnings
warnings.filterwarnings('ignore')

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_50_vessels.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")


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


def extract_time_series_features_vectorized(X):
    """Extract time-series features from sequences (vectorized)."""
    n_samples, n_timesteps, n_features = X.shape

    # Calculate features for each dimension
    features_list = []

    for dim in tqdm(range(n_features), desc="Extracting features", unit="dim"):
        X_dim = X[:, :, dim]  # (n_samples, n_timesteps)

        # Statistical features
        mean = np.mean(X_dim, axis=1)
        std = np.std(X_dim, axis=1)
        min_val = np.min(X_dim, axis=1)
        max_val = np.max(X_dim, axis=1)
        median = np.median(X_dim, axis=1)

        # Percentiles
        p25 = np.percentile(X_dim, 25, axis=1)
        p75 = np.percentile(X_dim, 75, axis=1)

        # Trend features
        diff = np.diff(X_dim, axis=1)
        trend_mean = np.mean(diff, axis=1)
        trend_std = np.std(diff, axis=1)

        # Combine features
        dim_features = np.column_stack([mean, std, min_val, max_val, median, p25, p75, trend_mean, trend_std])
        features_list.append(dim_features)

    # Concatenate all features
    X_features = np.hstack(features_list)
    return X_features


def load_pca_data(X_test):
    """Extract features and apply PCA transformation."""
    logger.info("Extracting features and applying PCA...")

    # Extract features
    X_test_features = extract_time_series_features_vectorized(X_test)
    logger.info(f"Features extracted: {X_test_features.shape}")

    # Apply PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=6)
    X_test_pca = pca.fit_transform(X_test_features)
    logger.info(f"PCA applied: {X_test_pca.shape}, explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    return X_test_pca


def load_trained_models():
    """Load all trained models."""
    logger.info("\n[2/4] Loading trained models...")
    
    models = {}
    model_dir = Path('results/advanced_pca_models')
    
    # Load XGBoost
    try:
        with open(model_dir / 'xgboost_model.pkl', 'rb') as f:
            models['XGBoost'] = pickle.load(f)
        logger.info("[OK] XGBoost model loaded")
    except Exception as e:
        logger.warning(f"Could not load XGBoost: {e}")
    
    # Load Random Forest
    try:
        with open(model_dir / 'random_forest_model.pkl', 'rb') as f:
            models['Random Forest'] = pickle.load(f)
        logger.info("[OK] Random Forest model loaded")
    except Exception as e:
        logger.warning(f"Could not load Random Forest: {e}")
    
    # Load Neural Network
    try:
        nn_model = NeuralNetworkRegressor(input_size=6).to(DEVICE)
        nn_model.load_state_dict(torch.load(model_dir / 'neural_network_model.pt', map_location=DEVICE))
        nn_model.eval()
        models['Neural Network'] = nn_model
        logger.info("[OK] Neural Network model loaded")
    except Exception as e:
        logger.warning(f"Could not load Neural Network: {e}")
    
    if not models:
        logger.error("No models loaded! Check if model files exist.")
        return None
    
    logger.info(f"Total models loaded: {len(models)}")
    return models


def make_predictions(X_test_pca, models):
    """Make predictions using all loaded models."""
    logger.info("\n[3/4] Making predictions on test set...")
    
    predictions = {}
    
    # XGBoost predictions
    if 'XGBoost' in models:
        predictions['XGBoost'] = models['XGBoost'].predict(X_test_pca)
        logger.info(f"XGBoost predictions: {predictions['XGBoost'].shape}")
    
    # Random Forest predictions
    if 'Random Forest' in models:
        predictions['Random Forest'] = models['Random Forest'].predict(X_test_pca)
        logger.info(f"Random Forest predictions: {predictions['Random Forest'].shape}")
    
    # Neural Network predictions
    if 'Neural Network' in models:
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_pca).to(DEVICE)
            nn_pred = models['Neural Network'](X_test_tensor).cpu().numpy()
        predictions['Neural Network'] = nn_pred
        logger.info(f"Neural Network predictions: {predictions['Neural Network'].shape}")
    
    return predictions


def plot_vessel_trajectory(y_test, predictions, mmsi_test, vessel_mmsi, output_dir):
    """Plot trajectory predictions for a single vessel."""
    mask = mmsi_test == vessel_mmsi
    indices = np.where(mask)[0]
    
    if len(indices) < 2:
        return False
    
    vessel_y = y_test[indices]
    timestamps = np.arange(len(vessel_y)) * 5  # 5-minute intervals in minutes
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Vessel {vessel_mmsi} - Trajectory Predictions vs Actual Values\n(5-minute intervals)', 
                 fontsize=14, fontweight='bold')
    
    variables = ['Latitude', 'Longitude', 'SOG (knots)', 'COG (degrees)']
    colors = {'XGBoost': '#FF6B6B', 'Random Forest': '#4ECDC4', 'Neural Network': '#45B7D1'}
    
    for idx, (ax, var) in enumerate(zip(axes.flat, variables)):
        # Actual values
        ax.plot(timestamps, vessel_y[:, idx], 'b-', label='Actual', linewidth=2.5, 
                alpha=0.9, marker='o', markersize=5, markerfacecolor='lightblue', markeredgewidth=1)
        
        # Model predictions
        for model_name, color in colors.items():
            if model_name in predictions:
                vessel_pred = predictions[model_name][indices]
                ax.plot(timestamps, vessel_pred[:, idx], '--', label=model_name, 
                       linewidth=2, alpha=0.7, color=color, marker='s', markersize=3)
        
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
    """Main pipeline - NO TRAINING, ONLY TESTING."""
    logger.info("\n" + "="*80)
    logger.info("TEST PREDICTIONS ON 50 RANDOM VESSELS")
    logger.info("Using Pre-trained Model Weights (NO TRAINING)")
    logger.info("="*80)
    
    # Load test data
    X_test, y_test, mmsi_test = load_test_data()

    # Extract features and apply PCA
    X_test_pca = load_pca_data(X_test)
    
    # Load trained models
    models = load_trained_models()
    if not models:
        logger.error("Failed to load models. Exiting.")
        return
    
    # Make predictions
    predictions = make_predictions(X_test_pca, models)
    
    # Calculate metrics
    metrics_summary = calculate_metrics(y_test, predictions)
    
    # Create output directory
    output_dir = Path('results/test_50_vessels_predictions')
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
    metrics_df = pd.DataFrame(metrics_summary).T
    metrics_df.to_csv(output_dir / 'model_metrics.csv')
    logger.info(f"\nMetrics saved to {output_dir / 'model_metrics.csv'}")
    
    # Save detailed predictions
    results_df = pd.DataFrame({
        'MMSI': mmsi_test,
        'actual_LAT': y_test[:, 0],
        'actual_LON': y_test[:, 1],
        'actual_SOG': y_test[:, 2],
        'actual_COG': y_test[:, 3],
    })
    
    for model_name in predictions.keys():
        results_df[f'{model_name}_LAT'] = predictions[model_name][:, 0]
        results_df[f'{model_name}_LON'] = predictions[model_name][:, 1]
        results_df[f'{model_name}_SOG'] = predictions[model_name][:, 2]
        results_df[f'{model_name}_COG'] = predictions[model_name][:, 3]
    
    results_df.to_csv(output_dir / 'all_predictions.csv', index=False)
    logger.info(f"Predictions saved to {output_dir / 'all_predictions.csv'}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("[COMPLETE] TEST PREDICTIONS COMPLETE")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Trajectory plots created: {success_count}/50 vessels")
    logger.info(f"\nModel Performance Summary:")
    logger.info(f"\n{metrics_df.to_string()}")
    logger.info("\nAll trajectory plots saved to: " + str(output_dir))


if __name__ == "__main__":
    main()

