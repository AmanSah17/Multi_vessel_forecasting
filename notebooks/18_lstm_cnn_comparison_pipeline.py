"""
LSTM vs Temporal CNN Comparison Pipeline
- Advanced feature engineering
- Hyperparameter tuning
- LSTM model training
- Temporal CNN model training
- Performance comparison
- Organized output structure
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
import json
import warnings
warnings.filterwarnings('ignore')

# Create output directories
output_dirs = {
    'logs': Path('logs'),
    'results': Path('results'),
    'images': Path('results/images'),
    'csv': Path('results/csv'),
    'models': Path('results/models')
}

for dir_path in output_dirs.values():
    dir_path.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(output_dirs['logs'] / 'training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class EnhancedLSTMModel(nn.Module):
    """Enhanced LSTM with tunable hyperparameters."""
    
    def __init__(self, input_size, hidden_size=256, num_layers=3, output_size=4, dropout=0.2):
        super(EnhancedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout, bidirectional=False
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class TemporalCNNModel(nn.Module):
    """Temporal CNN for comparison."""
    
    def __init__(self, input_size, output_size=4, num_filters=64, num_layers=4, dropout=0.2):
        super(TemporalCNNModel, self).__init__()
        
        self.input_proj = nn.Conv1d(input_size, num_filters, 1)
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (3 - 1) * dilation // 2
            
            self.blocks.append(nn.Sequential(
                nn.Conv1d(num_filters, num_filters, 3, padding=padding, dilation=dilation),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        self.fc = nn.Sequential(
            nn.Linear(num_filters, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, features, time)
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = x.mean(dim=2)  # Global average pooling
        x = self.fc(x)
        
        return x


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, 
                patience=20, device='cuda', model_name='lstm'):
    """Train model with early stopping."""
    
    logger.info(f"\n{'='*70}\nTRAINING {model_name.upper()}\n{'='*70}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(epochs), desc=f"Training {model_name}"):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dirs['models'] / f'best_{model_name}.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"✓ Training complete. Best validation loss: {best_val_loss:.6f}")
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, device='cuda', model_name='lstm'):
    """Evaluate model on test set."""
    
    logger.info(f"\n{'='*70}\nEVALUATING {model_name.upper()}\n{'='*70}")
    
    model.eval()
    y_true_all = []
    y_pred_all = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            y_true_all.append(y_batch.numpy())
            y_pred_all.append(y_pred)
    
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAE_per_output': {
            'LAT': mean_absolute_error(y_true[:, 0], y_pred[:, 0]),
            'LON': mean_absolute_error(y_true[:, 1], y_pred[:, 1]),
            'SOG': mean_absolute_error(y_true[:, 2], y_pred[:, 2]),
            'COG': mean_absolute_error(y_true[:, 3], y_pred[:, 3])
        }
    }
    
    logger.info(f"MAE: {mae:.6f}")
    logger.info(f"RMSE: {rmse:.6f}")
    logger.info(f"R²: {r2:.6f}")
    logger.info(f"Per-output MAE:")
    for output, value in metrics['MAE_per_output'].items():
        logger.info(f"  {output}: {value:.6f}")
    
    return metrics, y_true, y_pred


def plot_training_curves(train_losses, val_losses, model_name='lstm'):
    """Plot training curves."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title(f'{model_name.upper()} - Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dirs['images'] / f'{model_name}_training_curves.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {model_name}_training_curves.png")
    plt.close()


def save_metrics_csv(metrics_dict, filename='model_comparison.csv'):
    """Save metrics to CSV."""
    
    df = pd.DataFrame([metrics_dict])
    df.to_csv(output_dirs['csv'] / filename, index=False)
    logger.info(f"✓ Saved: {filename}")


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("LSTM vs TEMPORAL CNN COMPARISON PIPELINE")
    logger.info("="*70)
    logger.info(f"Output directories created:")
    for name, path in output_dirs.items():
        logger.info(f"  {name}: {path}")

