"""
MLflow Hyperparameter Tuning for LSTM and CNN Models
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
import json
from itertools import product

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
        logging.FileHandler(output_dirs['logs'] / 'hyperparameter_tuning.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Set MLflow tracking
mlflow.set_tracking_uri(f"file:{output_dirs['mlflow'].absolute()}")


class HyperparameterTuner:
    """Hyperparameter tuning for models."""
    
    def __init__(self, model_class, train_loader, val_loader, device='cuda'):
        self.model_class = model_class
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.results = []
    
    def grid_search(self, param_grid, input_size, epochs=50, patience=15):
        """Perform grid search over hyperparameters."""
        logger.info(f"\n{'='*70}\nGRID SEARCH HYPERPARAMETER TUNING\n{'='*70}")
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        logger.info(f"Total combinations: {len(combinations)}")
        
        for idx, combo in enumerate(tqdm(combinations, desc="Grid Search")):
            params = dict(zip(param_names, combo))
            
            with mlflow.start_run(run_name=f"GridSearch_{idx}"):
                # Log parameters
                mlflow.log_params(params)
                
                # Train model
                model = self.model_class(input_size=input_size, **params).to(self.device)
                val_loss = self._train_model(model, epochs, patience)
                
                # Log metrics
                mlflow.log_metric("best_val_loss", val_loss)
                
                # Store results
                result = {**params, 'val_loss': val_loss, 'run_id': mlflow.active_run().info.run_id}
                self.results.append(result)
        
        # Find best
        best_result = min(self.results, key=lambda x: x['val_loss'])
        logger.info(f"\nBest hyperparameters: {best_result}")
        
        return best_result
    
    def _train_model(self, model, epochs, patience):
        """Train model and return best validation loss."""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(self.val_loader)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        return best_val_loss
    
    def save_results(self):
        """Save tuning results."""
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('val_loss')
        results_df.to_csv(output_dirs['csv'] / 'hyperparameter_tuning_results.csv', index=False)
        logger.info(f"✓ Saved tuning results")
        
        return results_df


# Hyperparameter grids
LSTM_PARAM_GRID = {
    'hidden_size': [128, 256, 512],
    'num_layers': [2, 3, 4],
    'dropout': [0.1, 0.2, 0.3]
}

CNN_PARAM_GRID = {
    'num_filters': [32, 64, 128],
    'num_layers': [3, 4, 5],
    'dropout': [0.1, 0.2, 0.3]
}

if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("HYPERPARAMETER TUNING WITH MLFLOW")
    logger.info("="*70)

