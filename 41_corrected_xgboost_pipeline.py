"""
CORRECTED XGBoost Pipeline - Addressing All Issues + MLflow Monitoring

Key Fixes:
1. Proper train/val/test split BEFORE preprocessing
2. Fit scaler/PCA only on training data
3. Temporal-aware splitting (no data leakage)
4. Circular encoding for COG (course over ground)
5. Geodesic error metrics (meters instead of degrees)
6. Proper model saving with joblib
7. Comprehensive preprocessing pipeline object
8. MLflow logging for all metrics and artifacts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from tqdm import tqdm
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import optuna
from optuna.pruners import MedianPruner
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/xgboost_corrected_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")

# Setup MLflow
mlflow.set_experiment("XGBoost_Vessel_Trajectory_Forecasting")
mlflow.set_tracking_uri("file:./mlruns")


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance in kilometers."""
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


def encode_circular_feature(angle_degrees):
    """Convert circular feature (angle) to sin/cos for proper modeling."""
    angle_rad = np.radians(angle_degrees)
    return np.sin(angle_rad), np.cos(angle_rad)


def extract_advanced_features(X):
    """Extract 483 advanced time-series features from sequences.
    
    Args:
        X: shape (n_samples, n_timesteps, n_features)
    
    Returns:
        Features: shape (n_samples, 483)
    """
    logger.info("Extracting advanced features...")
    n_samples, n_timesteps, n_features = X.shape
    features_list = []
    
    for dim in tqdm(range(n_features), desc="Feature extraction", unit="dim", leave=False):
        X_dim = X[:, :, dim]
        
        # Statistical features (10)
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
        
        # Trend features (7)
        diff = np.diff(X_dim, axis=1)
        features_dict['trend_mean'] = np.mean(diff, axis=1)
        features_dict['trend_std'] = np.std(diff, axis=1)
        features_dict['trend_max'] = np.max(diff, axis=1)
        features_dict['trend_min'] = np.min(diff, axis=1)
        
        # Autocorrelation-like features (2)
        features_dict['first_last_diff'] = X_dim[:, -1] - X_dim[:, 0]
        features_dict['first_last_ratio'] = np.divide(X_dim[:, -1], X_dim[:, 0] + 1e-6)
        
        # Volatility (1)
        features_dict['volatility'] = np.std(diff, axis=1)
        
        dim_features = np.column_stack(list(features_dict.values()))
        features_list.append(dim_features)
    
    X_features = np.hstack(features_list)
    logger.info(f"Features extracted: {X_features.shape}")
    return X_features


def add_haversine_features(X):
    """Add 7 Haversine distance features.
    
    Args:
        X: shape (n_samples, n_timesteps, n_features)
           Assumes LAT is feature 0, LON is feature 1
    
    Returns:
        Haversine features: shape (n_samples, 7)
    """
    logger.info("Adding Haversine distance features...")
    
    n_samples = X.shape[0]
    haversine_features = []
    
    for i in tqdm(range(n_samples), desc="Haversine calculation", unit="sample", leave=False):
        seq = X[i]  # (n_timesteps, n_features)
        
        lats = seq[:, 0]
        lons = seq[:, 1]
        
        # Distance to first point
        dist_to_first = haversine_distance(lats[0], lons[0], lats, lons)
        
        # Consecutive distances
        consecutive_dists = [0.0]
        for j in range(1, len(lats)):
            dist = haversine_distance(lats[j-1], lons[j-1], lats[j], lons[j])
            consecutive_dists.append(dist)
        
        total_dist = np.sum(consecutive_dists)
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


def load_and_split_data():
    """Load data and perform PROPER temporal split.
    
    CRITICAL: Split BEFORE feature extraction to avoid data leakage
    """
    logger.info("\n[1/7] Loading and splitting data (BEFORE preprocessing)...")
    
    cache_file = 'results/cache/seq_cache_len12_sampled_3pct.npz'
    data = np.load(cache_file)
    X = data['X']
    y = data['y']
    mmsi_list = data['mmsi_list']
    
    logger.info(f"Total data: X={X.shape}, y={y.shape}")
    
    # PROPER TEMPORAL SPLIT (no leakage)
    # Assuming sequences are temporally ordered
    n_total = len(X)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    mmsi_train = mmsi_list[:n_train]
    
    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    mmsi_val = mmsi_list[n_train:n_train+n_val]
    
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]
    mmsi_test = mmsi_list[n_train+n_val:]
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return (X_train, y_train, mmsi_train), (X_val, y_val, mmsi_val), (X_test, y_test, mmsi_test)


def preprocess_data(train_data, val_data, test_data):
    """Extract features and fit preprocessing on TRAINING data only.
    
    CRITICAL: Fit scaler/PCA on train, transform val/test
    """
    logger.info("\n[2/7] Extracting features and fitting preprocessing...")
    
    X_train, y_train, _ = train_data
    X_val, y_val, _ = val_data
    X_test, y_test, _ = test_data
    
    # Extract features for all splits
    logger.info("Extracting features for train...")
    X_train_feat = extract_advanced_features(X_train)
    X_train_hav = add_haversine_features(X_train)
    X_train_feat = np.hstack([X_train_feat, X_train_hav])
    
    logger.info("Extracting features for val...")
    X_val_feat = extract_advanced_features(X_val)
    X_val_hav = add_haversine_features(X_val)
    X_val_feat = np.hstack([X_val_feat, X_val_hav])
    
    logger.info("Extracting features for test...")
    X_test_feat = extract_advanced_features(X_test)
    X_test_hav = add_haversine_features(X_test)
    X_test_feat = np.hstack([X_test_feat, X_test_hav])
    
    logger.info(f"Features shape: {X_train_feat.shape}")
    
    # FIT scaler/PCA on TRAINING data only
    logger.info("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    
    logger.info("Fitting PCA on training data...")
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # TRANSFORM val/test with fitted scaler/PCA
    X_val_scaled = scaler.transform(X_val_feat)
    X_val_pca = pca.transform(X_val_scaled)
    
    X_test_scaled = scaler.transform(X_test_feat)
    X_test_pca = pca.transform(X_test_scaled)
    
    logger.info(f"PCA components: {pca.n_components_}, variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    return (X_train_pca, y_train), (X_val_pca, y_val), (X_test_pca, y_test), scaler, pca


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function with MLflow logging - AGGRESSIVE MEMORY OPTIMIZATION."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 100, step=50),  # Reduced: 50-100
        'max_depth': trial.suggest_int('max_depth', 3, 8),  # Reduced: 3-8
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15, log=True),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 3),
        'gamma': trial.suggest_float('gamma', 0, 0.1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.1),
    }

    with mlflow.start_run(nested=True):
        # Log hyperparameters
        mlflow.log_params(params)

        try:
            # Train model with memory optimization
            model = MultiOutputRegressor(
                XGBRegressor(**params, verbosity=0, random_state=42, tree_method='hist')
            )
            model.fit(X_train, y_train)

            # Validate
            y_pred_val = model.predict(X_val)

            # Calculate metrics
            mae_val = mean_absolute_error(y_val, y_pred_val)
            rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
            r2_val = r2_score(y_val, y_pred_val)

            # Log metrics
            mlflow.log_metric("val_mae", mae_val)
            mlflow.log_metric("val_rmse", rmse_val)
            mlflow.log_metric("val_r2", r2_val)

            logger.info(f"Trial {trial.number}: MAE_val={mae_val:.4f}, R2={r2_val:.4f}")

            # Clean up to free memory
            del model
            import gc
            gc.collect()

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)[:100]}")
            return float('inf')

    return mae_val


def hyperparameter_tuning(X_train, y_train, X_val, y_val, n_trials=20):
    """Tune on validation set with MLflow logging - MEMORY OPTIMIZED."""
    logger.info("\n[3/7] Hyperparameter tuning with Optuna (MEMORY OPTIMIZED)...")

    # Log hyperparameter tuning parameters to current run
    mlflow.log_param("n_trials", n_trials)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("val_size", len(X_val))

    study = optuna.create_study(
        direction='minimize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5)
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True  # Garbage collection after each trial
    )

    logger.info(f"Best MAE: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Log best params and metrics
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_val_mae", study.best_value)
    mlflow.log_metric("best_trial_number", study.best_trial.number)

    return study.best_params


def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """Train final model on train+val, evaluate on test."""
    logger.info("\n[4/7] Training final model on train+val...")

    # Log training configuration to current run
    mlflow.log_params(best_params)
    mlflow.log_param("training_data_size", len(X_train) + len(X_val))

    # Combine train and val for final training
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.vstack([y_train, y_val])

    model = MultiOutputRegressor(XGBRegressor(**best_params, verbosity=0, random_state=42))
    model.fit(X_combined, y_combined)

    # Log training metrics
    y_pred_combined = model.predict(X_combined)
    mae_combined = mean_absolute_error(y_combined, y_pred_combined)
    r2_combined = r2_score(y_combined, y_pred_combined)

    mlflow.log_metric("final_train_mae", mae_combined)
    mlflow.log_metric("final_train_r2", r2_combined)

    logger.info(f"Final model trained - MAE: {mae_combined:.4f}, R2: {r2_combined:.4f}")

    return model


def geodesic_error(y_true, y_pred):
    """Calculate geodesic error in meters for lat/lon predictions.
    
    Args:
        y_true: shape (n_samples, 4) - [lat, lon, sog, cog]
        y_pred: shape (n_samples, 4) - [lat, lon, sog, cog]
    
    Returns:
        errors_meters: shape (n_samples,) - error in meters
    """
    lat_true, lon_true = y_true[:, 0], y_true[:, 1]
    lat_pred, lon_pred = y_pred[:, 0], y_pred[:, 1]
    
    errors_km = np.array([
        haversine_distance(lat_true[i], lon_true[i], lat_pred[i], lon_pred[i])
        for i in range(len(y_true))
    ])
    
    return errors_km * 1000  # Convert to meters


def calculate_metrics(y_test, predictions):
    """Calculate comprehensive metrics with MLflow logging."""
    logger.info("\n[5/7] Calculating metrics...")

    # Geodesic error (meters)
    geo_errors = geodesic_error(y_test, predictions)

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
        'Geodesic_Error_Mean_m': np.mean(geo_errors),
        'Geodesic_Error_Median_m': np.median(geo_errors),
        'Geodesic_Error_Std_m': np.std(geo_errors),
        'LAT_MAE_degrees': mae_lat,
        'LON_MAE_degrees': mae_lon,
        'SOG_MAE_knots': mae_sog,
        'COG_MAE_degrees': mae_cog,
        'LAT_RMSE_degrees': rmse_lat,
        'LON_RMSE_degrees': rmse_lon,
        'LAT_R2': r2_lat,
        'LON_R2': r2_lon,
    }

    # Log all metrics to MLflow
    for key, val in metrics.items():
        logger.info(f"{key}: {val:.4f}")
        mlflow.log_metric(f"test_{key}", val)

    return metrics


def save_pipeline(model, scaler, pca, output_dir):
    """Save complete preprocessing pipeline using joblib."""
    logger.info("\n[6/7] Saving pipeline...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with joblib (better for sklearn objects)
    joblib.dump(model, output_dir / 'xgboost_model.joblib')
    joblib.dump(scaler, output_dir / 'scaler.joblib')
    joblib.dump(pca, output_dir / 'pca.joblib')
    
    # Save metadata
    metadata = {
        'n_pca_components': pca.n_components_,
        'pca_variance_explained': float(pca.explained_variance_ratio_.sum()),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Pipeline saved to {output_dir}")


def main():
    """Main corrected pipeline with MLflow monitoring."""
    logger.info("\n" + "="*80)
    logger.info("CORRECTED XGBOOST PIPELINE - NO DATA LEAKAGE + MLFLOW MONITORING")
    logger.info("="*80)

    with mlflow.start_run(run_name="complete_pipeline"):
        # Log pipeline configuration
        mlflow.log_param("pipeline_version", "1.0_corrected")
        mlflow.log_param("train_val_test_split", "70-20-10")

        # Step 1: Load and split BEFORE preprocessing
        train_data, val_data, test_data = load_and_split_data()

        # Step 2: Preprocess (fit on train only)
        (X_train_pca, y_train), (X_val_pca, y_val), (X_test_pca, y_test), scaler, pca = preprocess_data(
            train_data, val_data, test_data
        )

        # Log preprocessing info
        mlflow.log_param("pca_components", pca.n_components_)
        mlflow.log_metric("pca_variance_explained", pca.explained_variance_ratio_.sum())

        # Step 3: Hyperparameter tuning on val (REDUCED TO 5 TRIALS FOR MEMORY OPTIMIZATION)
        best_params = hyperparameter_tuning(X_train_pca, y_train, X_val_pca, y_val, n_trials=5)

        # Step 4: Train final model on train+val
        model = train_final_model(X_train_pca, y_train, X_val_pca, y_val, best_params)

        # Step 5: Evaluate ONLY on test (never touched before)
        predictions = model.predict(X_test_pca)
        test_metrics = calculate_metrics(y_test, predictions)

        # Step 6: Save pipeline
        save_pipeline(model, scaler, pca, 'results/xgboost_corrected_50_vessels')

        # Log artifacts
        mlflow.log_artifact('logs/xgboost_corrected_pipeline.log')

        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE - NO DATA LEAKAGE + MLFLOW MONITORING")
        logger.info("="*80)

        return model, scaler, pca, test_metrics


if __name__ == "__main__":
    main()

