"""
Part 2: Clustering, PCA, Sequence Creation, and Training Functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.pytorch
import optuna

logger = logging.getLogger(__name__)


def apply_clustering_and_pca(df, features, n_clusters=5, n_components=10, output_dir='results/enhanced_lstm_haversine'):
    """Apply clustering and PCA for feature engineering."""
    logger.info(f"\n{'='*70}\n[4/10] CLUSTERING & PCA FOR FEATURE ENGINEERING\n{'='*70}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for clustering
    X_cluster = df[['LAT', 'LON', 'SOG', 'COG']].values
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
    
    # K-Means Clustering
    logger.info(f"Applying K-Means clustering (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_cluster_scaled)
    logger.info(f"✓ Clusters assigned: {df['cluster'].value_counts().to_dict()}")
    
    # PCA
    logger.info(f"Applying PCA (n_components={n_components})...")
    pca = PCA(n_components=min(n_components, len(features)))
    X_pca = pca.fit_transform(df[features].values)
    explained_var = pca.explained_variance_ratio_.sum()
    logger.info(f"✓ PCA explained variance: {explained_var:.4f}")
    
    # Plot PCA variance
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.cumsum(pca.explained_variance_ratio_), 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax.set_title('PCA - Cumulative Explained Variance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '03_pca_variance.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_dir / '03_pca_variance.png'}")
    plt.close()
    
    # Plot clusters
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(df['LON'], df['LAT'], c=df['cluster'], cmap='viridis', s=20, alpha=0.6)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Vessel Clusters (K-Means)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    plt.savefig(output_dir / '04_clusters_map.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_dir / '04_clusters_map.png'}")
    plt.close()
    
    return df, X_pca, pca, kmeans


def create_sequences_per_vessel(df, features, seq_length=30):
    """Create sequences with per-vessel 70/20/10 split."""
    logger.info(f"\n{'='*70}\n[5/10] CREATING SEQUENCES (Per-Vessel 70/20/10 Split)\n{'='*70}")
    
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    vessels = df['MMSI'].unique()
    
    for mmsi in tqdm(vessels, desc="Vessels", unit="vessel"):
        vessel_data = df[df['MMSI'] == mmsi].sort_values('BaseDateTime')[features].values
        
        if len(vessel_data) < seq_length + 1:
            continue
        
        X_vessel, y_vessel = [], []
        for i in range(len(vessel_data) - seq_length):
            X_vessel.append(vessel_data[i:i+seq_length])
            y_vessel.append(vessel_data[i+seq_length, :4])  # Only LAT, LON, SOG, COG
        
        if len(X_vessel) == 0:
            continue
        
        n = len(X_vessel)
        train_idx, val_idx = int(0.7 * n), int(0.9 * n)
        
        X_train.extend(X_vessel[:train_idx])
        y_train.extend(y_vessel[:train_idx])
        X_val.extend(X_vessel[train_idx:val_idx])
        y_val.extend(y_vessel[train_idx:val_idx])
        X_test.extend(X_vessel[val_idx:])
        y_test.extend(y_vessel[val_idx:])
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    # Scale features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def train_model_with_config(X_train, y_train, X_val, y_val, config, device):
    """Train model with given configuration."""
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    
    # Import model class
    from enhanced_lstm_haversine_tuning import EnhancedLSTMModel, haversine_distance, calculate_haversine_errors
    
    model = EnhancedLSTMModel(
        input_size=X_train.shape[2],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        bidirectional=config.get('bidirectional', False)
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )
    
    train_losses, val_losses, val_haversine_errors = [], [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            
            # Calculate haversine error
            val_pred_np = val_outputs.cpu().numpy()
            val_true_np = y_val_t.cpu().numpy()
            hav_errors = calculate_haversine_errors(val_true_np, val_pred_np)
            val_haversine_errors.append(hav_errors['haversine_mean_m'])
        
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, val_haversine_errors, best_val_loss


def objective(trial, X_train, y_train, X_val, y_val, device):
    """Optuna objective function for hyperparameter tuning."""
    
    config = {
        'hidden_size': trial.suggest_int('hidden_size', 64, 256, step=64),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'bidirectional': trial.suggest_categorical('bidirectional', [False, True]),
        'epochs': 50,  # Reduced for tuning
        'patience': 10
    }
    
    try:
        model, train_losses, val_losses, val_hav_errors, best_val_loss = train_model_with_config(
            X_train, y_train, X_val, y_val, config, device
        )
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(config)
            mlflow.log_metric("val_loss", best_val_loss)
            mlflow.log_metric("val_haversine_mean_m", val_hav_errors[-1] if val_hav_errors else float('inf'))
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return best_val_loss
        
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        return float('inf')


def hyperparameter_tuning(X_train, y_train, X_val, y_val, n_trials=20):
    """Perform hyperparameter tuning with Optuna."""
    logger.info(f"\n{'='*70}\n[6/10] HYPERPARAMETER TUNING WITH OPTUNA\n{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    study = optuna.create_study(
        direction='minimize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, device),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    logger.info(f"\n✓ Best trial:")
    logger.info(f"  Value (Val Loss): {study.best_value:.6f}")
    logger.info(f"  Params: {study.best_params}")
    
    return study.best_params

