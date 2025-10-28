"""
Advanced XGBoost Pipeline for Vessel Trajectory Forecasting
- Advanced feature extraction (EDA-based
)
- Haversine distance for spatial nonlinearity
- PCA + Standardization
- Extensive hyperparameter tuning
- Focus on LAT/LON precision for accurate location tracking
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import optuna
from optuna.pruners import MedianPruner

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/xgboost_advanced_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between two points in km."""
    R = 6371
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def extract_advanced_features(X):
    """Extract advanced time-series features from sequences."""
    logger.info("Extracting advanced features...")
    n_samples, n_timesteps, n_features = X.shape
    features_list = []
    
    for dim in tqdm(range(n_features), desc="Feature extraction", unit="dim", leave=False):
        X_dim = X[:, :, dim]
        
        # Statistical features
        features_dict = {
            'mean': np.mean(X_dim, axis=1),
            'std': np.std(X_dim, axis=1),
            'min': np.min(X_dim, axis=1),
            'max': np.max(X_dim, axis=1),
            'median': np.median(X_dim, axis=1),
            'p25': np.percentile(X_dim, 25, axis=1),
            'p75': np.percentile(X_dim, 75, axis=1),
            'range': np.max(X_dim, axis=1) - np.min(X_dim, axis=1),
            'skew': np.array([pd.Series(row).skew() for row in X_dim]),
            'kurtosis': np.array([pd.Series(row).kurtosis() for row in X_dim]),
        }
        
        # Trend features
        diff = np.diff(X_dim, axis=1)
        features_dict['trend_mean'] = np.mean(diff, axis=1)
        features_dict['trend_std'] = np.std(diff, axis=1)
        features_dict['trend_max'] = np.max(diff, axis=1)
        features_dict['trend_min'] = np.min(diff, axis=1)
        
        # Autocorrelation-like features
        features_dict['first_last_diff'] = X_dim[:, -1] - X_dim[:, 0]
        features_dict['first_last_ratio'] = np.divide(X_dim[:, -1], X_dim[:, 0] + 1e-6)
        
        # Volatility
        features_dict['volatility'] = np.std(diff, axis=1)
        
        dim_features = np.column_stack(list(features_dict.values()))
        features_list.append(dim_features)
    
    X_features = np.hstack(features_list)
    logger.info(f"Features extracted: {X_features.shape}")
    return X_features


def add_haversine_features(X, y):
    """Add Haversine distance features for spatial nonlinearity."""
    logger.info("Adding Haversine distance features...")
    
    n_samples = X.shape[0]
    haversine_features = []
    
    for i in tqdm(range(n_samples), desc="Haversine calculation", unit="sample", leave=False):
        seq = X[i]  # (n_timesteps, n_features)
        
        # Extract LAT (feature 0) and LON (feature 1)
        lats = seq[:, 0]
        lons = seq[:, 1]
        
        # Distance to first point
        dist_to_first = haversine_distance(lats[0], lons[0], lats, lons)
        
        # Distance between consecutive points
        consecutive_dists = [0.0]
        for j in range(1, len(lats)):
            dist = haversine_distance(lats[j-1], lons[j-1], lats[j], lons[j])
            consecutive_dists.append(dist)
        
        # Total distance traveled
        total_dist = np.sum(consecutive_dists)
        
        # Average distance per step
        avg_dist = np.mean(consecutive_dists[1:]) if len(consecutive_dists) > 1 else 0
        
        haversine_features.append([
            np.mean(dist_to_first),
            np.max(dist_to_first),
            np.std(dist_to_first),
            total_dist,
            avg_dist,
            np.max(consecutive_dists),
            np.std(consecutive_dists)
        ])
    
    haversine_array = np.array(haversine_features)
    logger.info(f"Haversine features added: {haversine_array.shape}")
    return haversine_array


def load_and_prepare_data():
    """Load test data and prepare features."""
    logger.info("\n[1/6] Loading and preparing data...")
    
    cache_file = 'results/cache/seq_cache_len12_sampled_3pct.npz'
    data = np.load(cache_file)
    X = data['X']
    y = data['y']
    mmsi_list = data['mmsi_list']
    
    n_train = int(0.7 * len(X))
    n_val = int(0.2 * len(X))
    
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]
    mmsi_test = mmsi_list[n_train+n_val:]
    
    logger.info(f"Test set: X={X_test.shape}, y={y_test.shape}")
    
    # Extract advanced features
    X_test_features = extract_advanced_features(X_test)
    
    # Add Haversine features
    X_haversine = add_haversine_features(X_test, y_test)
    X_test_features = np.hstack([X_test_features, X_haversine])
    
    logger.info(f"Total features before PCA: {X_test_features.shape}")
    
    # Standardization
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test_features)
    
    # PCA
    logger.info("Applying PCA...")
    pca = PCA(n_components=0.95)  # Keep 95% variance
    X_test_pca = pca.fit_transform(X_test_scaled)
    logger.info(f"PCA applied: {X_test_pca.shape}, explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_test_pca, y_test, mmsi_test, scaler, pca


def objective(trial, X_train, y_train, X_val, y_val):
    """Objective function for Optuna hyperparameter tuning."""
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
    }
    
    model = MultiOutputRegressor(XGBRegressor(**params, verbosity=0, random_state=42))
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    
    return mae


def hyperparameter_tuning(X_train, y_train, X_val, y_val, n_trials=100):
    """Perform extensive hyperparameter tuning."""
    logger.info("\n[2/6] Hyperparameter tuning with Optuna...")
    
    study = optuna.create_study(
        direction='minimize',
        pruner=MedianPruner(),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    logger.info(f"Best MAE: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    return study.best_params


def train_xgboost_optimized(X_train, y_train, X_val, y_val, best_params):
    """Train XGBoost with optimized parameters."""
    logger.info("\n[3/6] Training XGBoost with optimized parameters...")
    
    model = MultiOutputRegressor(XGBRegressor(**best_params, verbosity=0, random_state=42))
    model.fit(X_train, y_train)
    
    logger.info("[OK] XGBoost trained")
    return model


def make_predictions(X_test_pca, model):
    """Make predictions."""
    logger.info("\n[4/6] Making predictions...")
    
    predictions = model.predict(X_test_pca)
    logger.info(f"Predictions shape: {predictions.shape}")
    
    return predictions


def calculate_metrics(y_test, predictions):
    """Calculate metrics with focus on LAT/LON precision."""
    logger.info("\nCalculating metrics...")
    
    mae_overall = mean_absolute_error(y_test, predictions)
    rmse_overall = np.sqrt(mean_squared_error(y_test, predictions))
    r2_overall = r2_score(y_test, predictions)
    
    # Per-variable metrics
    mae_lat = mean_absolute_error(y_test[:, 0], predictions[:, 0])
    mae_lon = mean_absolute_error(y_test[:, 1], predictions[:, 1])
    mae_sog = mean_absolute_error(y_test[:, 2], predictions[:, 2])
    mae_cog = mean_absolute_error(y_test[:, 3], predictions[:, 3])
    
    rmse_lat = np.sqrt(mean_squared_error(y_test[:, 0], predictions[:, 0]))
    rmse_lon = np.sqrt(mean_squared_error(y_test[:, 1], predictions[:, 1]))
    
    r2_lat = r2_score(y_test[:, 0], predictions[:, 0])
    r2_lon = r2_score(y_test[:, 1], predictions[:, 1])
    
    metrics = {
        'Overall': {'MAE': mae_overall, 'RMSE': rmse_overall, 'R2': r2_overall},
        'LAT': {'MAE': mae_lat, 'RMSE': rmse_lat, 'R2': r2_lat},
        'LON': {'MAE': mae_lon, 'RMSE': rmse_lon, 'R2': r2_lon},
        'SOG': {'MAE': mae_sog},
        'COG': {'MAE': mae_cog}
    }
    
    logger.info(f"Overall - MAE: {mae_overall:.4f}, RMSE: {rmse_overall:.4f}, R2: {r2_overall:.4f}")
    logger.info(f"LAT - MAE: {mae_lat:.6f}, RMSE: {rmse_lat:.6f}, R2: {r2_lat:.4f}")
    logger.info(f"LON - MAE: {mae_lon:.6f}, RMSE: {rmse_lon:.6f}, R2: {r2_lon:.4f}")
    logger.info(f"SOG - MAE: {mae_sog:.4f}")
    logger.info(f"COG - MAE: {mae_cog:.4f}")
    
    return metrics


def plot_vessel_predictions(y_test, predictions, mmsi_test, vessel_mmsi, output_dir):
    """Plot predictions for a single vessel."""
    mask = mmsi_test == vessel_mmsi
    indices = np.where(mask)[0]
    
    if len(indices) < 2:
        return False
    
    vessel_y = y_test[indices]
    vessel_pred = predictions[indices]
    timestamps = np.arange(len(vessel_y)) * 5
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Vessel {vessel_mmsi} - XGBoost Advanced Predictions\n(Advanced Features + Haversine + PCA)', 
                 fontsize=14, fontweight='bold')
    
    variables = ['Latitude', 'Longitude', 'SOG (knots)', 'COG (degrees)']
    
    for idx, (ax, var) in enumerate(zip(axes.flat, variables)):
        ax.plot(timestamps, vessel_y[:, idx], 'b-', label='Actual', linewidth=2.5, 
                alpha=0.9, marker='o', markersize=5, markerfacecolor='lightblue', markeredgewidth=1)
        ax.plot(timestamps, vessel_pred[:, idx], 'r--', label='XGBoost Prediction', 
               linewidth=2, alpha=0.8, color='#FF6B6B', marker='s', markersize=4)
        
        ax.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
        ax.set_ylabel(var, fontsize=11, fontweight='bold')
        ax.set_title(var, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'vessel_{vessel_mmsi}_xgboost.png', dpi=120, bbox_inches='tight')
    plt.close()
    
    return True


def main():
    """Main pipeline."""
    logger.info("\n" + "="*80)
    logger.info("ADVANCED XGBOOST PIPELINE FOR VESSEL TRAJECTORY FORECASTING")
    logger.info("="*80)
    
    # Load and prepare data
    X_test_pca, y_test, mmsi_test, scaler, pca = load_and_prepare_data()
    
    # Split for tuning (use 80% for training, 20% for validation during tuning)
    n_tune = int(0.8 * len(X_test_pca))
    X_train_tune = X_test_pca[:n_tune]
    y_train_tune = y_test[:n_tune]
    X_val_tune = X_test_pca[n_tune:]
    y_val_tune = y_test[n_tune:]
    
    # Hyperparameter tuning
    best_params = hyperparameter_tuning(X_train_tune, y_train_tune, X_val_tune, y_val_tune, n_trials=100)
    
    # Train final model on all test data
    model = train_xgboost_optimized(X_test_pca, y_test, X_val_tune, y_val_tune, best_params)
    
    # Make predictions
    predictions = make_predictions(X_test_pca, model)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, predictions)
    
    # Create output directory
    output_dir = Path('results/xgboost_advanced_50_vessels')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and preprocessing objects
    logger.info("\n[5/6] Saving models and preprocessing objects...")
    with open(output_dir / 'xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(output_dir / 'pca.pkl', 'wb') as f:
        pickle.dump(pca, f)
    logger.info("Models saved")
    
    # Generate visualizations
    logger.info("\n[6/6] Generating predictions for 50 random vessels...")
    
    unique_mmsi = np.unique(mmsi_test)
    selected_mmsi = np.random.choice(unique_mmsi, size=min(50, len(unique_mmsi)), replace=False)
    
    success_count = 0
    for mmsi in tqdm(selected_mmsi, desc="Creating plots", unit="vessel", leave=False):
        if plot_vessel_predictions(y_test, predictions, mmsi_test, mmsi, output_dir):
            success_count += 1
    
    # Save metrics and predictions
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(output_dir / 'model_metrics.csv')
    
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
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("[COMPLETE] ADVANCED XGBOOST PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Plots created: {success_count}/50 vessels")
    logger.info(f"\nBest Hyperparameters: {best_params}")
    logger.info(f"\nMetrics Summary:\n{metrics_df.to_string()}")


if __name__ == "__main__":
    main()

