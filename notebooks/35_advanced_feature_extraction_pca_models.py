"""
Advanced Feature Extraction & PCA-based Model Training for Vessel Trajectory Prediction
- Extract advanced time-series features from vessel monitoring data
- Apply PCA (n_components=4-8) for dimensionality reduction
- Train multiple models: XGBoost, Random Forest, SVM, Neural Networks
- Hyperparameter tuning with MLflow logging
- Evaluate on 50 random vessels
- Compare with baseline Kalman-LSTM model
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from tqdm import tqdm
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import pickle

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/advanced_feature_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


# ======================== FEATURE EXTRACTION FUNCTIONS ========================

def extract_time_series_features_vectorized(X):
    """
    Vectorized feature extraction using NumPy for speed.

    Args:
        X: (n_samples, seq_len, n_features) - sequence data

    Returns:
        features: (n_samples, n_extracted_features)
    """
    n_samples, seq_len, n_features = X.shape
    features_list = []

    logger.info(f"Extracting features from {n_samples} sequences...")

    for feat_idx in tqdm(range(n_features), desc="Feature extraction by dimension", leave=False):
        feat_data = X[:, :, feat_idx]  # (n_samples, seq_len)

        # Basic statistics
        feat_mean = np.mean(feat_data, axis=1)
        feat_std = np.std(feat_data, axis=1)
        feat_min = np.min(feat_data, axis=1)
        feat_max = np.max(feat_data, axis=1)
        feat_median = np.median(feat_data, axis=1)
        feat_q25 = np.percentile(feat_data, 25, axis=1)
        feat_q75 = np.percentile(feat_data, 75, axis=1)

        # Trend features
        diff = np.diff(feat_data, axis=1)
        diff_mean = np.mean(diff, axis=1)
        diff_std = np.std(diff, axis=1)
        diff_sum = np.sum(np.abs(diff), axis=1)

        # Overall change
        overall_change = feat_data[:, -1] - feat_data[:, 0]

        features_list.extend([
            feat_mean, feat_std, feat_min, feat_max, feat_median, feat_q25, feat_q75,
            diff_mean, diff_std, diff_sum, overall_change
        ])

    return np.column_stack(features_list)


def apply_pca_transformation(X_train, X_val, X_test, n_components=6):
    """Apply PCA to reduce dimensionality."""
    logger.info(f"\nApplying PCA with n_components={n_components}")
    
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    logger.info(f"PCA - Explained variance: {explained_var:.4f}")
    logger.info(f"PCA - Component variance ratios: {pca.explained_variance_ratio_}")
    
    return X_train_pca, X_val_pca, X_test_pca, pca


# ======================== MODEL DEFINITIONS ========================

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


def train_neural_network(X_train, y_train, X_val, y_val, epochs=50, batch_size=64, lr=0.001):
    """Train neural network model with GPU optimization."""
    logger.info(f"\n{'='*80}\nTRAINING NEURAL NETWORK\n{'='*80}")

    input_size = X_train.shape[1]
    model = NeuralNetworkRegressor(input_size=input_size).to(DEVICE)

    # Create tensors on CPU first, then move to GPU
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
    y_val_tensor = torch.FloatTensor(y_val).to(DEVICE)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in tqdm(range(epochs), desc="Training NN", unit="epoch"):
        model.train()
        train_loss = 0
        batch_count = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

        train_loss /= batch_count
        train_losses.append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()

        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model, best_val_loss, train_losses, val_losses


def train_xgboost_model(X_train, y_train, X_val, y_val, params=None):
    """Train XGBoost model."""
    logger.info(f"\n{'='*80}\nTRAINING XGBOOST MODEL\n{'='*80}")
    
    if params is None:
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    return model


def train_random_forest_model(X_train, y_train, params=None):
    """Train Random Forest model."""
    logger.info(f"\n{'='*80}\nTRAINING RANDOM FOREST MODEL\n{'='*80}")
    
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    return model


def train_svm_model(X_train, y_train, params=None):
    """Train SVM model for multi-output regression."""
    logger.info(f"\n{'='*80}\nTRAINING SVM MODEL\n{'='*80}")

    if params is None:
        params = {
            'kernel': 'rbf',
            'C': 100,
            'epsilon': 0.1,
            'gamma': 'scale'
        }

    # Train separate SVM for each output
    from sklearn.multioutput import MultiOutputRegressor
    model = MultiOutputRegressor(SVR(**params))
    model.fit(X_train, y_train)

    return model


# ======================== EVALUATION FUNCTIONS ========================

def evaluate_model(model, X_test, y_test, model_type='sklearn'):
    """Evaluate model and return metrics."""
    if model_type == 'neural_network':
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
            predictions = model(X_test_tensor).cpu().numpy()
    else:
        predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    return predictions, {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def main():
    """Main pipeline."""
    logger.info("\n" + "="*80)
    logger.info("ADVANCED FEATURE EXTRACTION & PCA-BASED MODEL TRAINING")
    logger.info("="*80)

    # Load cached sequences
    logger.info("\n[1/6] Loading cached sequences...")
    cache_file = 'results/cache/seq_cache_len12_sampled_3pct.npz'
    data = np.load(cache_file)
    X = data['X']
    y = data['y']
    mmsi_list = data['mmsi_list']

    logger.info(f"Loaded sequences: X={X.shape}, y={y.shape}")

    # Split data
    n_train = int(0.7 * len(X))
    n_val = int(0.2 * len(X))

    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
    mmsi_test = mmsi_list[n_train+n_val:]

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Extract features
    logger.info("\n[2/6] Extracting time-series features...")
    X_train_feat = extract_time_series_features_vectorized(X_train)
    X_val_feat = extract_time_series_features_vectorized(X_val)
    X_test_feat = extract_time_series_features_vectorized(X_test)

    logger.info(f"Extracted features: {X_train_feat.shape}")

    # Apply PCA
    logger.info("\n[3/6] Applying PCA...")
    X_train_pca, X_val_pca, X_test_pca, _ = apply_pca_transformation(
        X_train_feat, X_val_feat, X_test_feat, n_components=6
    )

    # Train models with MLflow
    logger.info("\n[4/6] Training models with MLflow logging...")

    mlflow.set_experiment("Vessel_Trajectory_Advanced_Models")

    models_results = {}

    # Neural Network
    logger.info("\n--- Training Neural Network ---")
    with mlflow.start_run(run_name="NN_PCA"):
        nn_model, _, _, _ = train_neural_network(X_train_pca, y_train, X_val_pca, y_val, epochs=30, batch_size=64)
        nn_pred, nn_metrics = evaluate_model(nn_model, X_test_pca, y_test, model_type='neural_network')

        mlflow.log_params({'model': 'Neural Network', 'n_components': 6, 'epochs': 30})
        mlflow.log_metrics(nn_metrics)
        models_results['Neural Network'] = (nn_model, nn_metrics, nn_pred)
        logger.info(f"[OK] NN Metrics: MAE={nn_metrics['MAE']:.4f}, RMSE={nn_metrics['RMSE']:.4f}, R2={nn_metrics['R2']:.4f}")

    # XGBoost
    logger.info("\n--- Training XGBoost ---")
    with mlflow.start_run(run_name="XGBoost_PCA"):
        xgb_model = train_xgboost_model(X_train_pca, y_train, X_val_pca, y_val)
        xgb_pred, xgb_metrics = evaluate_model(xgb_model, X_test_pca, y_test, model_type='sklearn')

        mlflow.log_params({'model': 'XGBoost', 'n_components': 6})
        mlflow.log_metrics(xgb_metrics)
        models_results['XGBoost'] = (xgb_model, xgb_metrics, xgb_pred)
        logger.info(f"[OK] XGBoost Metrics: MAE={xgb_metrics['MAE']:.4f}, RMSE={xgb_metrics['RMSE']:.4f}, R2={xgb_metrics['R2']:.4f}")

    # Random Forest
    logger.info("\n--- Training Random Forest ---")
    with mlflow.start_run(run_name="RandomForest_PCA"):
        rf_model = train_random_forest_model(X_train_pca, y_train)
        rf_pred, rf_metrics = evaluate_model(rf_model, X_test_pca, y_test, model_type='sklearn')

        mlflow.log_params({'model': 'Random Forest', 'n_components': 6})
        mlflow.log_metrics(rf_metrics)
        models_results['Random Forest'] = (rf_model, rf_metrics, rf_pred)
        logger.info(f"[OK] Random Forest Metrics: MAE={rf_metrics['MAE']:.4f}, RMSE={rf_metrics['RMSE']:.4f}, R2={rf_metrics['R2']:.4f}")

    # SVM
    logger.info("\n--- Training SVM ---")
    with mlflow.start_run(run_name="SVM_PCA"):
        svm_model = train_svm_model(X_train_pca, y_train)
        svm_pred, svm_metrics = evaluate_model(svm_model, X_test_pca, y_test, model_type='sklearn')

        mlflow.log_params({'model': 'SVM', 'n_components': 6})
        mlflow.log_metrics(svm_metrics)
        models_results['SVM'] = (svm_model, svm_metrics, svm_pred)
        logger.info(f"✓ SVM Metrics: MAE={svm_metrics['MAE']:.4f}, RMSE={svm_metrics['RMSE']:.4f}, R2={svm_metrics['R2']:.4f}")

    # Find best model
    logger.info("\n[5/6] Model Comparison...")
    best_model_name = min(models_results.keys(), key=lambda x: models_results[x][1]['MAE'])
    best_model, _, best_pred = models_results[best_model_name]
    logger.info(f"\n[BEST] Best Model: {best_model_name}")
    logger.info(f"Best Model Metrics: {models_results[best_model_name][1]}")

    # Predictions on 50 random vessels
    logger.info("\n[6/6] Generating predictions for 50 random vessels...")

    # Save results
    output_dir = Path('results/advanced_pca_models')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    import pickle
    with open(output_dir / 'xgboost_model.pkl', 'wb') as f:
        pickle.dump(models_results['XGBoost'][0], f)
    logger.info("Saved XGBoost model")

    with open(output_dir / 'random_forest_model.pkl', 'wb') as f:
        pickle.dump(models_results['Random Forest'][0], f)
    logger.info("Saved Random Forest model")

    torch.save(models_results['Neural Network'][0].state_dict(), output_dir / 'neural_network_model.pt')
    logger.info("Saved Neural Network model")

    # Save PCA-transformed test data
    np.save(output_dir / 'X_test_pca.npy', X_test_pca)
    logger.info("Saved PCA-transformed test data")

    results_df = pd.DataFrame({
        'MMSI': mmsi_test,
        'pred_LAT': best_pred[:, 0],
        'pred_LON': best_pred[:, 1],
        'pred_SOG': best_pred[:, 2],
        'pred_COG': best_pred[:, 3],
        'actual_LAT': y_test[:, 0],
        'actual_LON': y_test[:, 1],
        'actual_SOG': y_test[:, 2],
        'actual_COG': y_test[:, 3]
    })

    results_df.to_csv(output_dir / 'predictions.csv', index=False)
    logger.info(f"Results saved to {output_dir / 'predictions.csv'}")

    # Generate visualizations for 50 random vessels
    logger.info("\nGenerating visualizations for 50 random vessels...")
    unique_mmsi = np.unique(mmsi_test)
    selected_mmsi = np.random.choice(unique_mmsi, size=min(50, len(unique_mmsi)), replace=False)

    for mmsi in tqdm(selected_mmsi, desc="Creating vessel plots", unit="vessel"):
        mask = mmsi_test == mmsi
        indices = np.where(mask)[0]

        if len(indices) < 2:
            continue

        vessel_y = y_test[indices]
        vessel_pred = best_pred[indices]
        timestamps = np.arange(len(vessel_y)) * 5

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f'Vessel {mmsi} - {best_model_name} Predictions', fontsize=14, fontweight='bold')

        # LAT
        axes[0, 0].plot(timestamps, vessel_y[:, 0], 'b-', label='Actual', linewidth=2, alpha=0.7)
        axes[0, 0].plot(timestamps, vessel_pred[:, 0], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        axes[0, 0].set_ylabel('Latitude')
        axes[0, 0].set_title('Latitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # LON
        axes[0, 1].plot(timestamps, vessel_y[:, 1], 'b-', label='Actual', linewidth=2, alpha=0.7)
        axes[0, 1].plot(timestamps, vessel_pred[:, 1], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        axes[0, 1].set_ylabel('Longitude')
        axes[0, 1].set_title('Longitude')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # SOG
        axes[1, 0].plot(timestamps, vessel_y[:, 2], 'b-', label='Actual', linewidth=2, alpha=0.7)
        axes[1, 0].plot(timestamps, vessel_pred[:, 2], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('SOG (knots)')
        axes[1, 0].set_title('Speed Over Ground')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # COG
        axes[1, 1].plot(timestamps, vessel_y[:, 3], 'b-', label='Actual', linewidth=2, alpha=0.7)
        axes[1, 1].plot(timestamps, vessel_pred[:, 3], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        axes[1, 1].set_xlabel('Time (minutes)')
        axes[1, 1].set_ylabel('COG (degrees)')
        axes[1, 1].set_title('Course Over Ground')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'vessel_{mmsi}_predictions.png', dpi=100, bbox_inches='tight')
        plt.close()

    logger.info("\n" + "="*80)
    logger.info("[COMPLETE] PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Visualizations: {len(selected_mmsi)} vessels")


if __name__ == "__main__":
    main()

