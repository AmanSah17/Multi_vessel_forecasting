"""
Per-vessel predictions and performance analysis
Load trained model and make predictions for each vessel in test set
Generate per-vessel performance plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup
MODEL_DIR = Path('results/xgboost_corrected_50_vessels')
CACHE_DIR = Path('results/cache_checkpoints')
DATA_DIR = Path('data')
OUTPUT_DIR = Path('results/per_vessel_predictions')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model and preprocessing objects
logger.info("Loading trained model and preprocessing objects...")
model = joblib.load(MODEL_DIR / 'model.pkl')
scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
pca = joblib.load(MODEL_DIR / 'pca.pkl')

# Load preprocessed test data from cache
logger.info("Loading test data from cache...")
cached_data = np.load(CACHE_DIR / 'preprocessed_data.npz', allow_pickle=True)
X_test_pca = cached_data['X_test_pca']
y_test = cached_data['y_test']

# Load original cache to get vessel IDs
logger.info("Loading original cache to extract vessel information...")
try:
    cache_file = 'results/cache/seq_cache_len12_sampled_3pct.npz'
    cache_data = np.load(cache_file, allow_pickle=True)
    all_vessel_ids = cache_data['vessel_ids']

    # Get test set vessel IDs (last 10% of data)
    n_total = len(all_vessel_ids)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    test_vessel_ids = all_vessel_ids[n_train+n_val:]

    logger.info(f"Test data shape: X={X_test_pca.shape}, y={y_test.shape}")
    logger.info(f"Unique vessels in test set: {len(np.unique(test_vessel_ids))}")
except Exception as e:
    logger.warning(f"Could not load vessel IDs from cache: {e}")
    logger.info("Creating synthetic vessel IDs based on sequence patterns...")
    # Create synthetic vessel IDs based on position changes
    test_vessel_ids = np.zeros(len(y_test), dtype=int)
    vessel_id = 0
    threshold = 5.0  # degrees

    for i in range(1, len(y_test)):
        lat_diff = abs(y_test[i, 0] - y_test[i-1, 0])
        lon_diff = abs(y_test[i, 1] - y_test[i-1, 1])

        if lat_diff > threshold or lon_diff > threshold:
            vessel_id += 1
        test_vessel_ids[i] = vessel_id

    logger.info(f"Created {len(np.unique(test_vessel_ids))} synthetic vessel groups")

# Make predictions for all test samples
logger.info("Making predictions on all test samples...")
y_pred_all = model.predict(X_test_pca)

# Group by vessel
logger.info("Grouping predictions by vessel...")
vessel_data = defaultdict(lambda: {'actual': [], 'predicted': [], 'indices': []})

for idx, vessel_id in enumerate(test_vessel_ids):
    vessel_data[vessel_id]['actual'].append(y_test[idx])
    vessel_data[vessel_id]['predicted'].append(y_pred_all[idx])
    vessel_data[vessel_id]['indices'].append(idx)

# Convert to arrays
for vessel_id in vessel_data:
    vessel_data[vessel_id]['actual'] = np.array(vessel_data[vessel_id]['actual'])
    vessel_data[vessel_id]['predicted'] = np.array(vessel_data[vessel_id]['predicted'])
    vessel_data[vessel_id]['indices'] = np.array(vessel_data[vessel_id]['indices'])

logger.info(f"Grouped into {len(vessel_data)} vessels")

# Calculate metrics per vessel
output_names = ['Latitude', 'Longitude', 'SOG (knots)', 'COG (degrees)']
vessel_metrics = []

for vessel_id, data in sorted(vessel_data.items()):
    actual = data['actual']
    predicted = data['predicted']
    n_samples = len(actual)
    
    metrics_row = {'Vessel_ID': vessel_id, 'N_Samples': n_samples}
    
    for idx, name in enumerate(output_names):
        actual_out = actual[:, idx]
        predicted_out = predicted[:, idx]
        
        mae = np.mean(np.abs(actual_out - predicted_out))
        rmse = np.sqrt(np.mean((actual_out - predicted_out) ** 2))
        r2 = 1 - (np.sum((actual_out - predicted_out) ** 2) / np.sum((actual_out - actual_out.mean()) ** 2))
        
        metrics_row[f'{name}_MAE'] = mae
        metrics_row[f'{name}_RMSE'] = rmse
        metrics_row[f'{name}_R2'] = r2
    
    vessel_metrics.append(metrics_row)

metrics_df = pd.DataFrame(vessel_metrics)
metrics_df = metrics_df.sort_values('N_Samples', ascending=False)
metrics_df.to_csv(OUTPUT_DIR / 'per_vessel_metrics.csv', index=False)
logger.info(f"✓ Saved: per_vessel_metrics.csv")

# Display top 10 vessels by sample count
logger.info("\n" + "="*100)
logger.info("TOP 10 VESSELS BY SAMPLE COUNT IN TEST SET")
logger.info("="*100)
print(metrics_df.head(10).to_string(index=False))

# Create per-vessel performance plots for top 10 vessels
top_vessels = metrics_df.head(10)['Vessel_ID'].values

logger.info(f"\nGenerating performance plots for top 10 vessels...")

for vessel_idx, vessel_id in enumerate(top_vessels, 1):
    data = vessel_data[vessel_id]
    actual = data['actual']
    predicted = data['predicted']
    n_samples = len(actual)
    
    # Create 2x2 subplot for this vessel
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Vessel {vessel_id} - Performance Analysis ({n_samples} samples)', 
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
        mae = np.mean(np.abs(actual_out - predicted_out))
        rmse = np.sqrt(np.mean((actual_out - predicted_out) ** 2))
        r2 = 1 - (np.sum((actual_out - predicted_out) ** 2) / np.sum((actual_out - actual_out.mean()) ** 2))
        
        ax.set_xlabel('Actual', fontsize=10, fontweight='bold')
        ax.set_ylabel('Predicted', fontsize=10, fontweight='bold')
        ax.set_title(f'{name}\nMAE={mae:.4f}, R²={r2:.4f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'vessel_{vessel_id}_performance.png', dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved: vessel_{vessel_id}_performance.png")
    plt.close()

# Create overall performance comparison across all vessels
logger.info("\nGenerating overall vessel comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Per-Vessel Model Performance Comparison (All Vessels in Test Set)', 
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
plt.savefig(OUTPUT_DIR / 'all_vessels_r2_comparison.png', dpi=300, bbox_inches='tight')
logger.info(f"✓ Saved: all_vessels_r2_comparison.png")
plt.close()

# Create MAE comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Per-Vessel MAE Comparison (All Vessels in Test Set)', 
             fontsize=16, fontweight='bold')

for idx, name in enumerate(output_names):
    ax = axes.flat[idx]
    
    mae_values = metrics_df[f'{name}_MAE'].values
    vessel_ids = metrics_df['Vessel_ID'].values
    
    # Show top 15 vessels
    top_15_idx = np.argsort(metrics_df['N_Samples'].values)[-15:]
    
    ax.barh(range(len(top_15_idx)), mae_values[top_15_idx], color='steelblue', edgecolor='black', linewidth=1)
    ax.set_yticks(range(len(top_15_idx)))
    ax.set_yticklabels([f'V{vid}' for vid in vessel_ids[top_15_idx]], fontsize=9)
    ax.set_xlabel('MAE', fontweight='bold')
    ax.set_title(f'{name} - MAE (Top 15 Vessels)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'all_vessels_mae_comparison.png', dpi=300, bbox_inches='tight')
logger.info(f"✓ Saved: all_vessels_mae_comparison.png")
plt.close()

# Summary statistics
logger.info("\n" + "="*100)
logger.info("OVERALL STATISTICS ACROSS ALL VESSELS")
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
logger.info("✓ PER-VESSEL PREDICTION AND ANALYSIS COMPLETE!")
logger.info("="*100)
logger.info(f"\nOutput files saved to: {OUTPUT_DIR}")
logger.info(f"  - per_vessel_metrics.csv ({len(metrics_df)} vessels)")
logger.info(f"  - vessel_[ID]_performance.png (top 10 vessels)")
logger.info(f"  - all_vessels_r2_comparison.png")
logger.info(f"  - all_vessels_mae_comparison.png")

