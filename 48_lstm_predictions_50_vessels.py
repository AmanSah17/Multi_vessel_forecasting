"""
LSTM Model Predictions on 50 Random Vessels
- Load trained LSTM model
- Make predictions on test set
- Generate per-vessel visualizations
- Calculate comprehensive metrics
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = Path('results/lstm_advanced_model')
CACHE_DIR = Path('results/cache_checkpoints')
OUTPUT_DIR = Path('results/lstm_predictions_50_vessels')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Using device: {device}")

# Define LSTM Model class (needed for loading)
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, n_layers=4, dropout=0.2, output_size=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, 256)
        self.silu = nn.SiLU()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        x = self.fc1(last_hidden)
        x = self.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Load model
logger.info("Loading LSTM model...")
model = LSTMModel(input_size=48, hidden_size=128, n_layers=4, dropout=0.2, output_size=4)
model.load_state_dict(torch.load(MODEL_DIR / 'best_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Load test data
logger.info("Loading test data...")
cached_data = np.load(CACHE_DIR / 'preprocessed_data.npz', allow_pickle=True)
X_test_pca = torch.FloatTensor(cached_data['X_test_pca']).to(device)
y_test = torch.FloatTensor(cached_data['y_test']).to(device)

# Reshape for LSTM
X_test_lstm = X_test_pca.reshape(X_test_pca.shape[0], 1, -1)

logger.info(f"Test data shape: {X_test_lstm.shape}")

# Load vessel IDs
try:
    cache_file = 'results/cache/seq_cache_len12_sampled_3pct.npz'
    cache_data = np.load(cache_file, allow_pickle=True)
    all_vessel_ids = cache_data['vessel_ids']
    
    n_total = len(all_vessel_ids)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    test_vessel_ids = all_vessel_ids[n_train+n_val:]
    
    logger.info(f"Loaded {len(np.unique(test_vessel_ids))} unique vessels")
except:
    logger.warning("Could not load vessel IDs, creating synthetic ones...")
    test_vessel_ids = np.arange(len(y_test))

# Make predictions
logger.info("Making predictions on test set...")
with torch.no_grad():
    y_pred_all = model(X_test_lstm).cpu().numpy()

y_test_np = y_test.cpu().numpy()

# Group by vessel
logger.info("Grouping predictions by vessel...")
vessel_data = defaultdict(lambda: {'actual': [], 'predicted': [], 'indices': []})

for idx, vessel_id in enumerate(test_vessel_ids):
    vessel_data[vessel_id]['actual'].append(y_test_np[idx])
    vessel_data[vessel_id]['predicted'].append(y_pred_all[idx])
    vessel_data[vessel_id]['indices'].append(idx)

# Convert to arrays
for vessel_id in vessel_data:
    vessel_data[vessel_id]['actual'] = np.array(vessel_data[vessel_id]['actual'])
    vessel_data[vessel_id]['predicted'] = np.array(vessel_data[vessel_id]['predicted'])

# Select 50 random vessels
np.random.seed(42)
all_vessel_ids_list = list(vessel_data.keys())
selected_vessels = np.random.choice(all_vessel_ids_list, size=min(50, len(all_vessel_ids_list)), replace=False)

logger.info(f"Selected {len(selected_vessels)} random vessels for analysis")

# Calculate metrics for selected vessels
output_names = ['Latitude', 'Longitude', 'SOG (knots)', 'COG (degrees)']
vessel_metrics = []

for vessel_id in tqdm(selected_vessels, desc="Calculating metrics"):
    data = vessel_data[vessel_id]
    actual = data['actual']
    predicted = data['predicted']
    n_samples = len(actual)
    
    metrics_row = {'Vessel_ID': vessel_id, 'N_Samples': n_samples}
    
    for idx, name in enumerate(output_names):
        actual_out = actual[:, idx]
        predicted_out = predicted[:, idx]
        
        mae = mean_absolute_error(actual_out, predicted_out)
        rmse = np.sqrt(mean_squared_error(actual_out, predicted_out))
        r2 = r2_score(actual_out, predicted_out)
        
        metrics_row[f'{name}_MAE'] = mae
        metrics_row[f'{name}_RMSE'] = rmse
        metrics_row[f'{name}_R2'] = r2
    
    vessel_metrics.append(metrics_row)

metrics_df = pd.DataFrame(vessel_metrics)
metrics_df = metrics_df.sort_values('N_Samples', ascending=False)
metrics_df.to_csv(OUTPUT_DIR / 'lstm_50_vessels_metrics.csv', index=False)
logger.info(f"✓ Saved: lstm_50_vessels_metrics.csv")

# Display top vessels
logger.info("\n" + "="*100)
logger.info("TOP 10 VESSELS BY SAMPLE COUNT")
logger.info("="*100)
print(metrics_df.head(10).to_string(index=False))

# Create performance plots for top 10 vessels
top_vessels = metrics_df.head(10)['Vessel_ID'].values

logger.info(f"\nGenerating performance plots for top 10 vessels...")

for vessel_idx, vessel_id in enumerate(tqdm(top_vessels, desc="Creating plots"), 1):
    data = vessel_data[vessel_id]
    actual = data['actual']
    predicted = data['predicted']
    n_samples = len(actual)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'LSTM - Vessel {vessel_id} Performance ({n_samples} samples)', 
                 fontsize=14, fontweight='bold')
    
    for idx, (ax, name) in enumerate(zip(axes.flat, output_names)):
        actual_out = actual[:, idx]
        predicted_out = predicted[:, idx]
        
        # Scatter plot
        ax.scatter(actual_out, predicted_out, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(actual_out.min(), predicted_out.min())
        max_val = max(actual_out.max(), predicted_out.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        # Metrics
        mae = mean_absolute_error(actual_out, predicted_out)
        rmse = np.sqrt(mean_squared_error(actual_out, predicted_out))
        r2 = r2_score(actual_out, predicted_out)
        
        ax.set_xlabel('Actual', fontsize=10, fontweight='bold')
        ax.set_ylabel('Predicted', fontsize=10, fontweight='bold')
        ax.set_title(f'{name}\nMAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'lstm_vessel_{vessel_id}_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

logger.info(f"✓ Saved 10 vessel performance plots")

# Create overall comparison plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('LSTM Model: Per-Vessel Performance Comparison (50 Random Vessels)', 
             fontsize=16, fontweight='bold')

for idx, name in enumerate(output_names):
    ax = axes.flat[idx]
    
    r2_values = metrics_df[f'{name}_R2'].values
    vessel_ids = metrics_df['Vessel_ID'].values
    
    # Show top 15 vessels
    top_15_idx = np.argsort(metrics_df['N_Samples'].values)[-15:]
    
    colors = ['green' if r2 > 0.9 else 'orange' if r2 > 0.7 else 'red' for r2 in r2_values[top_15_idx]]
    ax.barh(range(len(top_15_idx)), r2_values[top_15_idx], color=colors, edgecolor='black', linewidth=1)
    ax.set_yticks(range(len(top_15_idx)))
    ax.set_yticklabels([f'V{vid}' for vid in vessel_ids[top_15_idx]], fontsize=9)
    ax.set_xlabel('R² Score', fontweight='bold')
    ax.set_title(f'{name} - R² Scores (Top 15 Vessels)', fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'lstm_all_vessels_r2_comparison.png', dpi=300, bbox_inches='tight')
logger.info(f"✓ Saved: lstm_all_vessels_r2_comparison.png")
plt.close()

# Summary statistics
logger.info("\n" + "="*100)
logger.info("OVERALL STATISTICS - 50 RANDOM VESSELS")
logger.info("="*100)

for name in output_names:
    r2_col = f'{name}_R2'
    mae_col = f'{name}_MAE'
    
    logger.info(f"\n{name}:")
    logger.info(f"  R² - Mean: {metrics_df[r2_col].mean():.4f}, Std: {metrics_df[r2_col].std():.4f}")
    logger.info(f"  R² - Min: {metrics_df[r2_col].min():.4f}, Max: {metrics_df[r2_col].max():.4f}")
    logger.info(f"  MAE - Mean: {metrics_df[mae_col].mean():.4f}, Std: {metrics_df[mae_col].std():.4f}")
    logger.info(f"  MAE - Min: {metrics_df[mae_col].min():.4f}, Max: {metrics_df[mae_col].max():.4f}")

logger.info("\n" + "="*100)
logger.info("✓ LSTM PREDICTIONS COMPLETE!")
logger.info("="*100)
logger.info(f"\nOutput files saved to: {OUTPUT_DIR}")
logger.info(f"  - lstm_50_vessels_metrics.csv")
logger.info(f"  - lstm_vessel_[ID]_performance.png (top 10 vessels)")
logger.info(f"  - lstm_all_vessels_r2_comparison.png")

