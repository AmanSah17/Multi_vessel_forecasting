"""
Train NN-PCA, XGBoost, and Random Forest Models + Predictions on 50 Vessels
- Extract time-series features from test data
- Apply PCA transformation
- Train 3 models (NN, XGBoost, Random Forest)
- Save trained models
- Make predictions on test set
- Generate trajectory plots for 50 random vessels
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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
        logging.FileHandler('logs/train_predict_pca_models.log'),
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


def extract_time_series_features_vectorized(X):
    """Extract time-series features from sequences (vectorized)."""
    n_samples, n_timesteps, n_features = X.shape
    features_list = []
    
    for dim in tqdm(range(n_features), desc="Extracting features", unit="dim", leave=False):
        X_dim = X[:, :, dim]
        
        mean = np.mean(X_dim, axis=1)
        std = np.std(X_dim, axis=1)
        min_val = np.min(X_dim, axis=1)
        max_val = np.max(X_dim, axis=1)
        median = np.median(X_dim, axis=1)
        p25 = np.percentile(X_dim, 25, axis=1)
        p75 = np.percentile(X_dim, 75, axis=1)
        
        diff = np.diff(X_dim, axis=1)
        trend_mean = np.mean(diff, axis=1)
        trend_std = np.std(diff, axis=1)
        
        dim_features = np.column_stack([mean, std, min_val, max_val, median, p25, p75, trend_mean, trend_std])
        features_list.append(dim_features)
    
    X_features = np.hstack(features_list)
    return X_features


def load_and_prepare_data():
    """Load test data and extract features."""
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
    
    # Extract features
    logger.info("Extracting features...")
    X_test_features = extract_time_series_features_vectorized(X_test)
    logger.info(f"Features extracted: {X_test_features.shape}")
    
    # Apply PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=6)
    X_test_pca = pca.fit_transform(X_test_features)
    logger.info(f"PCA applied: {X_test_pca.shape}, explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_test_pca, y_test, mmsi_test


def train_neural_network(X_test_pca, y_test, epochs=20, batch_size=256, lr=0.001):
    """Train neural network model."""
    logger.info("\n[2/6] Training Neural Network...")
    
    input_size = X_test_pca.shape[1]
    model = NeuralNetworkRegressor(input_size=input_size).to(DEVICE)
    
    X_tensor = torch.FloatTensor(X_test_pca)
    y_tensor = torch.FloatTensor(y_test)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    for epoch in tqdm(range(epochs), desc="Training NN", unit="epoch", leave=False):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
    
    logger.info("[OK] Neural Network trained")
    return model


def train_xgboost(X_test_pca, y_test):
    """Train XGBoost model."""
    logger.info("\n[3/6] Training XGBoost...")
    
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor
    
    model = MultiOutputRegressor(XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, verbosity=0))
    model.fit(X_test_pca, y_test)
    
    logger.info("[OK] XGBoost trained")
    return model


def train_random_forest(X_test_pca, y_test):
    """Train Random Forest model."""
    logger.info("\n[4/6] Training Random Forest...")
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1))
    model.fit(X_test_pca, y_test)
    
    logger.info("[OK] Random Forest trained")
    return model


def make_predictions(X_test_pca, nn_model, xgb_model, rf_model):
    """Make predictions using all models."""
    logger.info("\n[5/6] Making predictions...")
    
    predictions = {}
    
    # XGBoost
    predictions['XGBoost'] = xgb_model.predict(X_test_pca)
    logger.info(f"XGBoost predictions: {predictions['XGBoost'].shape}")
    
    # Random Forest
    predictions['Random Forest'] = rf_model.predict(X_test_pca)
    logger.info(f"Random Forest predictions: {predictions['Random Forest'].shape}")
    
    # Neural Network
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_pca).to(DEVICE)
        nn_pred = nn_model(X_tensor).cpu().numpy()
    predictions['Neural Network'] = nn_pred
    logger.info(f"Neural Network predictions: {predictions['Neural Network'].shape}")
    
    return predictions


def plot_vessel_comparison(y_test, predictions, mmsi_test, vessel_mmsi, output_dir):
    """Plot predictions from all 3 models for a single vessel."""
    mask = mmsi_test == vessel_mmsi
    indices = np.where(mask)[0]
    
    if len(indices) < 2:
        return False
    
    vessel_y = y_test[indices]
    timestamps = np.arange(len(vessel_y)) * 5
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Vessel {vessel_mmsi} - Model Predictions Comparison (NN-PCA, XGBoost, Random Forest)\n(5-minute intervals)', 
                 fontsize=14, fontweight='bold')
    
    variables = ['Latitude', 'Longitude', 'SOG (knots)', 'COG (degrees)']
    colors = {'XGBoost': '#FF6B6B', 'Random Forest': '#4ECDC4', 'Neural Network': '#45B7D1'}
    
    for idx, (ax, var) in enumerate(zip(axes.flat, variables)):
        # Actual values
        ax.plot(timestamps, vessel_y[:, idx], 'b-', label='Actual', linewidth=2.5, 
                alpha=0.9, marker='o', markersize=5, markerfacecolor='lightblue', markeredgewidth=1)
        
        # Model predictions
        for model_name, color in colors.items():
            vessel_pred = predictions[model_name][indices]
            ax.plot(timestamps, vessel_pred[:, idx], '--', label=model_name, 
                   linewidth=2, alpha=0.7, color=color, marker='s', markersize=3)
        
        ax.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
        ax.set_ylabel(var, fontsize=11, fontweight='bold')
        ax.set_title(var, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'vessel_{vessel_mmsi}_comparison.png', dpi=120, bbox_inches='tight')
    plt.close()
    
    return True


def calculate_metrics(y_test, predictions):
    """Calculate metrics for all models."""
    logger.info("\nCalculating metrics...")
    
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


def save_models(nn_model, xgb_model, rf_model, output_dir):
    """Save trained models."""
    logger.info("\nSaving models...")
    
    torch.save(nn_model.state_dict(), output_dir / 'neural_network_pca_model.pt')
    logger.info("Saved Neural Network model")
    
    with open(output_dir / 'xgboost_pca_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    logger.info("Saved XGBoost model")
    
    with open(output_dir / 'random_forest_pca_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    logger.info("Saved Random Forest model")


def main():
    """Main pipeline."""
    logger.info("\n" + "="*80)
    logger.info("TRAIN & PREDICT: NN-PCA, XGBoost, Random Forest on 50 Vessels")
    logger.info("="*80)
    
    # Load and prepare data
    X_test_pca, y_test, mmsi_test = load_and_prepare_data()
    
    # Train models
    nn_model = train_neural_network(X_test_pca, y_test)
    xgb_model = train_xgboost(X_test_pca, y_test)
    rf_model = train_random_forest(X_test_pca, y_test)
    
    # Make predictions
    predictions = make_predictions(X_test_pca, nn_model, xgb_model, rf_model)
    
    # Calculate metrics
    metrics_summary = calculate_metrics(y_test, predictions)
    
    # Create output directory
    output_dir = Path('results/pca_models_50_vessels')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    save_models(nn_model, xgb_model, rf_model, output_dir)
    
    # Generate visualizations
    logger.info("\n[6/6] Generating comparison plots for 50 random vessels...")
    
    unique_mmsi = np.unique(mmsi_test)
    selected_mmsi = np.random.choice(unique_mmsi, size=min(50, len(unique_mmsi)), replace=False)
    
    logger.info(f"Selected {len(selected_mmsi)} random vessels")
    
    success_count = 0
    for mmsi in tqdm(selected_mmsi, desc="Creating plots", unit="vessel", leave=False):
        if plot_vessel_comparison(y_test, predictions, mmsi_test, mmsi, output_dir):
            success_count += 1
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics_summary).T
    metrics_df.to_csv(output_dir / 'model_metrics.csv')
    logger.info(f"Metrics saved")
    
    # Save predictions
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
    logger.info(f"Predictions saved")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("[COMPLETE] TRAINING & PREDICTIONS COMPLETE")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Comparison plots created: {success_count}/50 vessels")
    logger.info(f"\nModel Performance Summary:")
    logger.info(f"\n{metrics_df.to_string()}")


if __name__ == "__main__":
    main()

