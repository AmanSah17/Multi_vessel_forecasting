"""
Load trained XGBoost model and make predictions on 50 random test samples
Visualize predictions vs actual values
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup
MODEL_DIR = Path('results/xgboost_corrected_50_vessels')
CACHE_DIR = Path('results/cache_checkpoints')
OUTPUT_DIR = Path('results/predictions_visualization_2000_samples')
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

logger.info(f"Test data shape: X={X_test_pca.shape}, y={y_test.shape}")

# Select 5000 random samples
np.random.seed(42)
random_indices = np.random.choice(len(X_test_pca), size=2000, replace=False)
X_test_sample = X_test_pca[random_indices]
y_test_sample = y_test[random_indices]

# Make predictions
logger.info("Making predictions on 5000 random test samples...")
y_pred_sample = model.predict(X_test_sample)

logger.info(f"Predictions shape: {y_pred_sample.shape}")
logger.info(f"Actual shape: {y_test_sample.shape}")

# Output names
output_names = ['Latitude', 'Longitude', 'SOG (knots)', 'COG (degrees)']

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('XGBoost Model: Predictions vs Actual (50 Random Test Samples)', fontsize=16, fontweight='bold')

for idx, (ax, name) in enumerate(zip(axes.flat, output_names)):
    actual = y_test_sample[:, idx]
    predicted = y_pred_sample[:, idx]
    
    # Scatter plot
    ax.scatter(actual, predicted, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Metrics
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2))
    
    ax.set_xlabel('Actual', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_title(f'{name}\nMAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'predictions_vs_actual_scatter.png', dpi=300, bbox_inches='tight')
logger.info(f"✓ Saved: predictions_vs_actual_scatter.png")
plt.close()

# Create time series comparison plots
fig, axes = plt.subplots(2, 2, figsize=(28, 18))
fig.suptitle('XGBoost Model: Time Series Comparison (5000 Random Test Samples)', fontsize=16, fontweight='bold')

for idx, (ax, name) in enumerate(zip(axes.flat, output_names)):
    actual = y_test_sample[:, idx]
    predicted = y_pred_sample[:, idx]
    
    x_pos = np.arange(len(actual))
    
    ax.plot(x_pos, actual, 'o-', label='Actual', linewidth=2, markersize=6, alpha=0.7)
    ax.plot(x_pos, predicted, 's--', label='Predicted', linewidth=2, markersize=6, alpha=0.7)
    
    ax.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'predictions_vs_actual_timeseries.png', dpi=300, bbox_inches='tight')
logger.info(f"✓ Saved: predictions_vs_actual_timeseries.png")
plt.close()

# Create error distribution plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('XGBoost Model: Prediction Error Distribution (50 Random Test Samples)', fontsize=16, fontweight='bold')

for idx, (ax, name) in enumerate(zip(axes.flat, output_names)):
    actual = y_test_sample[:, idx]
    predicted = y_pred_sample[:, idx]
    errors = actual - predicted
    
    ax.hist(errors, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean Error: {errors.mean():.4f}')
    ax.axvline(0, color='green', linestyle='-', linewidth=2, label='Zero Error')
    
    ax.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'{name}\nStd Dev: {errors.std():.4f}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'prediction_error_distribution.png', dpi=500, bbox_inches='tight')
logger.info(f"✓ Saved: prediction_error_distribution.png")
plt.close()

# Create detailed metrics table
logger.info("\n" + "="*80)
logger.info("DETAILED METRICS FOR 50 RANDOM TEST SAMPLES")
logger.info("="*80)

metrics_data = []
for idx, name in enumerate(output_names):
    actual = y_test_sample[:, idx]
    predicted = y_pred_sample[:, idx]
    
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2))
    mape = np.mean(np.abs((actual - predicted) / (np.abs(actual) + 1e-8))) * 100
    
    metrics_data.append({
        'Output': name,
        'MAE': f'{mae:.6f}',
        'RMSE': f'{rmse:.6f}',
        'R²': f'{r2:.6f}',
        'MAPE (%)': f'{mape:.2f}',
        'Actual Mean': f'{actual.mean():.6f}',
        'Predicted Mean': f'{predicted.mean():.6f}'
    })
    
    logger.info(f"\n{name}:")
    logger.info(f"  MAE:            {mae:.6f}")
    logger.info(f"  RMSE:           {rmse:.6f}")
    logger.info(f"  R²:             {r2:.6f}")
    logger.info(f"  MAPE:           {mape:.2f}%")
    logger.info(f"  Actual Mean:    {actual.mean():.6f}")
    logger.info(f"  Predicted Mean: {predicted.mean():.6f}")

# Save metrics to CSV
import pandas as pd
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv(OUTPUT_DIR / 'prediction_metrics.csv', index=False)
logger.info(f"\n✓ Saved: prediction_metrics.csv")

# Save predictions to CSV
predictions_df = pd.DataFrame({
    'Latitude_Actual': y_test_sample[:, 0],
    'Latitude_Predicted': y_pred_sample[:, 0],
    'Longitude_Actual': y_test_sample[:, 1],
    'Longitude_Predicted': y_pred_sample[:, 1],
    'SOG_Actual': y_test_sample[:, 2],
    'SOG_Predicted': y_pred_sample[:, 2],
    'COG_Actual': y_test_sample[:, 3],
    'COG_Predicted': y_pred_sample[:, 3],
})
predictions_df.to_csv(OUTPUT_DIR / 'predictions_50_samples.csv', index=False)
logger.info(f"✓ Saved: predictions_50_samples.csv")

logger.info("\n" + "="*80)
logger.info("✓ PREDICTION AND VISUALIZATION COMPLETE!")
logger.info("="*80)
logger.info(f"\nOutput files saved to: {OUTPUT_DIR}")
logger.info("  - predictions_vs_actual_scatter.png")
logger.info("  - predictions_vs_actual_timeseries.png")
logger.info("  - prediction_error_distribution.png")
logger.info("  - prediction_metrics.csv")
logger.info("  - predictions_50_samples.csv")

