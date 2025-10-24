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

def extract_time_series_features(X, window_size=12):
    """
    Extract advanced time-series features from sequences.
    
    Args:
        X: (n_samples, seq_len, n_features) - sequence data
        window_size: lookback window for feature extraction
    
    Returns:
        features: (n_samples, n_extracted_features)
    """
    n_samples, seq_len, n_features = X.shape
    extracted_features = []
    
    for i in range(n_samples):
        seq = X[i]  # (seq_len, n_features)
        features = []
        
        # Statistical features for each feature dimension
        for feat_idx in range(n_features):
            feat_seq = seq[:, feat_idx]
            
            # Basic statistics
            features.extend([
                np.mean(feat_seq),
                np.std(feat_seq),
                np.min(feat_seq),
                np.max(feat_seq),
                np.median(feat_seq),
                np.percentile(feat_seq, 25),
                np.percentile(feat_seq, 75),
            ])
            
            # Trend features
            if len(feat_seq) > 1:
                diff = np.diff(feat_seq)
                features.extend([
                    np.mean(diff),
                    np.std(diff),
                    np.sum(np.abs(diff)),  # Total variation
                ])
            
            # Autocorrelation-like features
            if len(feat_seq) > 2:
                features.append(feat_seq[-1] - feat_seq[0])  # Overall change
        
        extracted_features.append(features)
    
    return np.array(extracted_features)


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


def train_neural_network(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.001):
    """Train neural network model."""
    logger.info(f"\n{'='*80}\nTRAINING NEURAL NETWORK\n{'='*80}")
    
    input_size = X_train.shape[1]
    model = NeuralNetworkRegressor(input_size=input_size).to(DEVICE)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
            y_val_tensor = torch.FloatTensor(y_val).to(DEVICE)
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()
        
        val_losses.append(val_loss)
        
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
    """Train SVM model."""
    logger.info(f"\n{'='*80}\nTRAINING SVM MODEL\n{'='*80}")
    
    if params is None:
        params = {
            'kernel': 'rbf',
            'C': 100,
            'epsilon': 0.1,
            'gamma': 'scale'
        }
    
    model = SVR(**params)
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
    X_train_feat = extract_time_series_features(X_train)
    X_val_feat = extract_time_series_features(X_val)
    X_test_feat = extract_time_series_features(X_test)
    
    logger.info(f"Extracted features: {X_train_feat.shape}")
    
    # Apply PCA
    logger.info("\n[3/6] Applying PCA...")
    X_train_pca, X_val_pca, X_test_pca, pca = apply_pca_transformation(
        X_train_feat, X_val_feat, X_test_feat, n_components=6
    )
    
    # Train models with MLflow
    logger.info("\n[4/6] Training models with MLflow logging...")
    
    mlflow.set_experiment("Vessel_Trajectory_Advanced_Models")
    
    models_results = {}
    
    # Neural Network
    with mlflow.start_run(run_name="NN_PCA"):
        nn_model, _, _, _ = train_neural_network(X_train_pca, y_train, X_val_pca, y_val, epochs=50)
        nn_pred, nn_metrics = evaluate_model(nn_model, X_test_pca, y_test, model_type='neural_network')
        
        mlflow.log_params({'model': 'Neural Network', 'n_components': 6})
        mlflow.log_metrics(nn_metrics)
        models_results['Neural Network'] = (nn_model, nn_metrics, nn_pred)
        logger.info(f"NN Metrics: {nn_metrics}")
    
    # XGBoost
    with mlflow.start_run(run_name="XGBoost_PCA"):
        xgb_model = train_xgboost_model(X_train_pca, y_train, X_val_pca, y_val)
        xgb_pred, xgb_metrics = evaluate_model(xgb_model, X_test_pca, y_test, model_type='sklearn')
        
        mlflow.log_params({'model': 'XGBoost', 'n_components': 6})
        mlflow.log_metrics(xgb_metrics)
        models_results['XGBoost'] = (xgb_model, xgb_metrics, xgb_pred)
        logger.info(f"XGBoost Metrics: {xgb_metrics}")
    
    # Random Forest
    with mlflow.start_run(run_name="RandomForest_PCA"):
        rf_model = train_random_forest_model(X_train_pca, y_train)
        rf_pred, rf_metrics = evaluate_model(rf_model, X_test_pca, y_test, model_type='sklearn')
        
        mlflow.log_params({'model': 'Random Forest', 'n_components': 6})
        mlflow.log_metrics(rf_metrics)
        models_results['Random Forest'] = (rf_model, rf_metrics, rf_pred)
        logger.info(f"Random Forest Metrics: {rf_metrics}")
    
    # SVM
    with mlflow.start_run(run_name="SVM_PCA"):
        svm_model = train_svm_model(X_train_pca, y_train)
        svm_pred, svm_metrics = evaluate_model(svm_model, X_test_pca, y_test, model_type='sklearn')
        
        mlflow.log_params({'model': 'SVM', 'n_components': 6})
        mlflow.log_metrics(svm_metrics)
        models_results['SVM'] = (svm_model, svm_metrics, svm_pred)
        logger.info(f"SVM Metrics: {svm_metrics}")
    
    # Find best model
    logger.info("\n[5/6] Model Comparison...")
    best_model_name = min(models_results.keys(), key=lambda x: models_results[x][1]['MAE'])
    logger.info(f"\nBest Model: {best_model_name}")
    logger.info(f"Best Model Metrics: {models_results[best_model_name][1]}")
    
    # Predictions on 50 random vessels
    logger.info("\n[6/6] Generating predictions for 50 random vessels...")
    best_model, best_metrics, best_pred = models_results[best_model_name]
    
    # Save results
    output_dir = Path('results/advanced_pca_models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()

