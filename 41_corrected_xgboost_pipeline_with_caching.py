"""
CORRECTED XGBoost Pipeline with CACHING - Addressing All Issues + MLflow Monitoring

Key Features:
1. Checkpoint/caching after each processing step
2. Resume from cache if available
3. Memory-efficient processing
4. Proper train/val/test split BEFORE preprocessing
5. Fit scaler/PCA only on training data
6. Temporal-aware splitting (no data leakage)
7. Comprehensive MLflow logging
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
import joblib
import warnings
import gc
import psutil
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import optuna
from optuna.pruners import MedianPruner
import mlflow
import mlflow.sklearn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup cache directory
CACHE_DIR = Path('results/cache_checkpoints')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024

def save_checkpoint(data, name):
    """Save checkpoint to cache"""
    path = CACHE_DIR / f"{name}.npz"
    logger.info(f"Saving checkpoint: {name} (Memory: {get_memory_usage():.2f}GB)")
    if isinstance(data, dict):
        np.savez_compressed(path, **data)
    else:
        np.savez_compressed(path, data=data)
    logger.info(f"✓ Checkpoint saved: {path}")
    gc.collect()

def load_checkpoint(name):
    """Load checkpoint from cache"""
    path = CACHE_DIR / f"{name}.npz"
    if path.exists():
        logger.info(f"Loading checkpoint: {name}")
        data = np.load(path, allow_pickle=True)
        if 'data' in data:
            return data['data']
        return {k: data[k] for k in data.files}
    return None

def load_and_split_data():
    """Load data and split into train/val/test BEFORE preprocessing"""
    logger.info("\n[1/7] Loading and splitting data (BEFORE preprocessing)...")
    
    # Check cache
    cached = load_checkpoint('data_split')
    if cached is not None:
        return cached['train_data'], cached['val_data'], cached['test_data']
    
    # Load from original cache
    cache_file = 'results/cache/seq_cache_len12_sampled_3pct.npz'
    data = np.load(cache_file)
    X, y = data['X'], data['y']
    
    logger.info(f"Total data: X={X.shape}, y={y.shape}")
    
    # Split: 70% train, 20% val, 10% test
    n_train = int(0.7 * len(X))
    n_val = int(0.2 * len(X))
    
    train_data = (X[:n_train], y[:n_train])
    val_data = (X[n_train:n_train+n_val], y[n_train:n_train+n_val])
    test_data = (X[n_train+n_val:], y[n_train+n_val:])
    
    logger.info(f"Train: {train_data[0].shape}, Val: {val_data[0].shape}, Test: {test_data[0].shape}")
    
    # Save checkpoint
    save_checkpoint({
        'train_data': train_data[0],
        'train_labels': train_data[1],
        'val_data': val_data[0],
        'val_labels': val_data[1],
        'test_data': test_data[0],
        'test_labels': test_data[1]
    }, 'data_split')
    
    return train_data, val_data, test_data

def preprocess_data(train_data, val_data, test_data):
    """Preprocess data with caching"""
    logger.info("\n[2/7] Extracting features and fitting preprocessing...")
    
    # Check cache
    cached = load_checkpoint('preprocessed_data')
    if cached is not None:
        logger.info("✓ Using cached preprocessed data")
        scaler = joblib.load(CACHE_DIR / 'scaler.pkl')
        pca = joblib.load(CACHE_DIR / 'pca.pkl')
        return (
            (cached['X_train_pca'], cached['y_train']),
            (cached['X_val_pca'], cached['y_val']),
            (cached['X_test_pca'], cached['y_test']),
            scaler, pca
        )
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # Flatten sequences for feature extraction
    logger.info("Flattening sequences...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Fit scaler on training data only
    logger.info("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Fit PCA on training data only
    logger.info("Fitting PCA on training data...")
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Transform in batches to avoid memory issues
    logger.info("Transforming validation data with PCA (in batches)...")
    batch_size = 10000
    X_val_pca_list = []
    for i in range(0, len(X_val_scaled), batch_size):
        X_val_pca_list.append(pca.transform(X_val_scaled[i:i+batch_size]))
    X_val_pca = np.vstack(X_val_pca_list)

    logger.info("Transforming test data with PCA (in batches)...")
    X_test_pca_list = []
    for i in range(0, len(X_test_scaled), batch_size):
        X_test_pca_list.append(pca.transform(X_test_scaled[i:i+batch_size]))
    X_test_pca = np.vstack(X_test_pca_list)
    
    logger.info(f"PCA components: {pca.n_components_}, variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Save preprocessing objects
    joblib.dump(scaler, CACHE_DIR / 'scaler.pkl')
    joblib.dump(pca, CACHE_DIR / 'pca.pkl')
    
    # Save checkpoint
    save_checkpoint({
        'X_train_pca': X_train_pca,
        'y_train': y_train,
        'X_val_pca': X_val_pca,
        'y_val': y_val,
        'X_test_pca': X_test_pca,
        'y_test': y_test
    }, 'preprocessed_data')
    
    return (X_train_pca, y_train), (X_val_pca, y_val), (X_test_pca, y_test), scaler, pca

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function - AGGRESSIVE MEMORY OPTIMIZATION"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 100, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15, log=True),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 3),
        'gamma': trial.suggest_float('gamma', 0, 0.1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.1),
    }

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)

        try:
            model = MultiOutputRegressor(
                XGBRegressor(**params, verbosity=0, random_state=42, tree_method='hist')
            )
            model.fit(X_train, y_train)

            y_pred_val = model.predict(X_val)
            mae_val = mean_absolute_error(y_val, y_pred_val)
            rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
            r2_val = r2_score(y_val, y_pred_val)

            mlflow.log_metric("val_mae", mae_val)
            mlflow.log_metric("val_rmse", rmse_val)
            mlflow.log_metric("val_r2", r2_val)

            logger.info(f"Trial {trial.number}: MAE_val={mae_val:.4f}, R2={r2_val:.4f}")
            
            del model
            gc.collect()

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)[:100]}")
            return float('inf')

    return mae_val

def hyperparameter_tuning(X_train, y_train, X_val, y_val, n_trials=5):
    """Tune hyperparameters with Optuna"""
    logger.info(f"\n[3/7] Hyperparameter tuning with Optuna ({n_trials} trials)...")

    mlflow.log_param("n_trials", n_trials)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("val_size", len(X_val))

    study = optuna.create_study(
        direction='minimize',
        pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=1),
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=2)
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True
    )

    best_params = study.best_params
    logger.info(f"Best params: {best_params}")
    mlflow.log_params(best_params)
    
    return best_params

def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """Train final model on train+val"""
    logger.info("\n[4/7] Training final model on train+val...")

    mlflow.log_params(best_params)
    mlflow.log_param("training_data_size", len(X_train) + len(X_val))

    X_combined = np.vstack([X_train, X_val])
    y_combined = np.vstack([y_train, y_val])

    model = MultiOutputRegressor(XGBRegressor(**best_params, verbosity=0, random_state=42, tree_method='hist'))
    model.fit(X_combined, y_combined)

    y_pred_combined = model.predict(X_combined)
    mae_combined = mean_absolute_error(y_combined, y_pred_combined)
    r2_combined = r2_score(y_combined, y_pred_combined)

    mlflow.log_metric("final_train_mae", mae_combined)
    mlflow.log_metric("final_train_r2", r2_combined)

    return model

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    logger.info("\n[5/7] Calculating test metrics...")
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'test_mae': mae,
        'test_rmse': rmse,
        'test_r2': r2
    }
    
    for key, val in metrics.items():
        mlflow.log_metric(key, val)
        logger.info(f"{key}: {val:.4f}")
    
    return metrics

def save_pipeline(model, scaler, pca, output_dir):
    """Save complete pipeline"""
    logger.info(f"\n[6/7] Saving pipeline to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, f"{output_dir}/model.pkl")
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    joblib.dump(pca, f"{output_dir}/pca.pkl")
    
    logger.info("✓ Pipeline saved successfully")

def main():
    """Main pipeline with caching"""
    with mlflow.start_run(run_name="xgboost_with_caching"):
        mlflow.log_param("pipeline_version", "2.0_with_caching")
        
        # Load and split
        train_data, val_data, test_data = load_and_split_data()
        
        # Preprocess with caching
        (X_train_pca, y_train), (X_val_pca, y_val), (X_test_pca, y_test), scaler, pca = preprocess_data(
            train_data, val_data, test_data
        )
        
        # Hyperparameter tuning
        best_params = hyperparameter_tuning(X_train_pca, y_train, X_val_pca, y_val, n_trials=5)
        
        # Train final model
        model = train_final_model(X_train_pca, y_train, X_val_pca, y_val, best_params)
        
        # Evaluate
        predictions = model.predict(X_test_pca)
        test_metrics = calculate_metrics(y_test, predictions)
        
        # Save
        save_pipeline(model, scaler, pca, 'results/xgboost_corrected_50_vessels')
        
        logger.info("\n[7/7] ✓ TRAINING COMPLETE!")
        logger.info(f"Test Metrics: {test_metrics}")

if __name__ == "__main__":
    main()

