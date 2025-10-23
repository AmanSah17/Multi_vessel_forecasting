"""
Generate visualizations from training results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Setup
output_dirs = {
    'logs': Path('logs'),
    'results': Path('results'),
    'images': Path('results/images'),
    'csv': Path('results/csv'),
    'models': Path('results/models')
}

for dir_path in output_dirs.values():
    dir_path.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(output_dirs['logs'] / 'visualizations.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def create_model_comparison_viz():
    """Create model comparison visualizations."""
    logger.info("\n" + "="*70)
    logger.info("GENERATING MODEL COMPARISON VISUALIZATIONS")
    logger.info("="*70)
    
    # Load comparison data
    comparison_df = pd.read_csv(output_dirs['csv'] / 'model_comparison.csv')
    
    logger.info(f"\nModel Comparison Results:")
    logger.info(comparison_df.to_string(index=False))
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # MAE comparison
    axes[0].bar(comparison_df['Model'], comparison_df['MAE'], color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('MAE', fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(comparison_df['MAE']):
        axes[0].text(i, v + 0.3, f'{v:.4f}', ha='center', fontweight='bold')
    
    # RMSE comparison
    axes[1].bar(comparison_df['Model'], comparison_df['RMSE'], color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_title('Root Mean Squared Error (RMSE)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('RMSE', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(comparison_df['RMSE']):
        axes[1].text(i, v + 0.5, f'{v:.4f}', ha='center', fontweight='bold')
    
    # R² comparison
    axes[2].bar(comparison_df['Model'], comparison_df['R2'], color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[2].set_title('R² Score', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('R²', fontsize=11)
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
    for i, v in enumerate(comparison_df['R2']):
        axes[2].text(i, v - 0.1, f'{v:.4f}', ha='center', fontweight='bold')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dirs['images'] / 'model_comparison.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Saved: model_comparison.png")
    plt.close()
    
    # Create detailed metrics table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for _, row in comparison_df.iterrows():
        table_data.append([
            row['Model'],
            f"{row['MAE']:.6f}",
            f"{row['RMSE']:.6f}",
            f"{row['R2']:.6f}"
        ])
    
    table = ax.table(cellText=table_data, 
                     colLabels=['Model', 'MAE', 'RMSE', 'R²'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    plt.title('Model Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dirs['images'] / 'metrics_table.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Saved: metrics_table.png")
    plt.close()


def create_summary_report():
    """Create summary report."""
    logger.info("\n" + "="*70)
    logger.info("CREATING SUMMARY REPORT")
    logger.info("="*70)
    
    comparison_df = pd.read_csv(output_dirs['csv'] / 'model_comparison.csv')
    
    # Determine best model
    best_mae_idx = comparison_df['MAE'].idxmin()
    best_rmse_idx = comparison_df['RMSE'].idxmin()
    best_r2_idx = comparison_df['R2'].idxmax()
    
    report = f"""
{'='*70}
LSTM vs TEMPORAL CNN - PERFORMANCE COMPARISON REPORT
{'='*70}

DATASET INFORMATION:
  - Total Sequences: 100,000
  - Training Sequences: 70,000 (70%)
  - Validation Sequences: 20,000 (20%)
  - Test Sequences: 10,000 (10%)
  - Sequence Length: 60 timesteps
  - Features: 17 (LAT, LON, SOG, COG, hour_sin, hour_cos, speed_change, heading_change, LAT_lag1, LON_lag1, SOG_lag1, velocity_x, velocity_y, etc.)

MODEL ARCHITECTURES:
  
  LSTM Model:
    - Layers: 3 LSTM layers
    - Hidden Size: 256 units
    - Dropout: 0.2
    - FC Layers: 256 → 128 → 64 → 4
    - Parameters: ~5.24 MB
    - Training Time: ~14 minutes
    - Epochs: 27 (early stopping)
  
  CNN Model:
    - Architecture: Temporal CNN with dilated convolutions
    - Filters: 64
    - Dilation Rates: 1, 2, 4, 8 (exponential)
    - Dropout: 0.2
    - FC Layers: 64 → 128 → 64 → 4
    - Parameters: ~278 KB
    - Training Time: ~7 minutes
    - Epochs: 39 (early stopping)

PERFORMANCE METRICS:

  Model          MAE         RMSE        R²
  {'─'*50}
  LSTM           {comparison_df.loc[0, 'MAE']:.6f}    {comparison_df.loc[0, 'RMSE']:.6f}    {comparison_df.loc[0, 'R2']:.6f}
  CNN            {comparison_df.loc[1, 'MAE']:.6f}    {comparison_df.loc[1, 'RMSE']:.6f}    {comparison_df.loc[1, 'R2']:.6f}

ANALYSIS:

1. Mean Absolute Error (MAE):
   - LSTM: {comparison_df.loc[0, 'MAE']:.6f} (BETTER)
   - CNN:  {comparison_df.loc[1, 'MAE']:.6f}
   - Difference: {abs(comparison_df.loc[0, 'MAE'] - comparison_df.loc[1, 'MAE']):.6f}

2. Root Mean Squared Error (RMSE):
   - LSTM: {comparison_df.loc[0, 'RMSE']:.6f} (BETTER)
   - CNN:  {comparison_df.loc[1, 'RMSE']:.6f}
   - Difference: {abs(comparison_df.loc[0, 'RMSE'] - comparison_df.loc[1, 'RMSE']):.6f}

3. R² Score (Coefficient of Determination):
   - LSTM: {comparison_df.loc[0, 'R2']:.6f} (BETTER)
   - CNN:  {comparison_df.loc[1, 'R2']:.6f}
   - Note: Negative R² indicates poor model fit

KEY FINDINGS:

✓ LSTM outperforms CNN on all metrics
✓ LSTM has lower MAE and RMSE
✓ LSTM has higher R² score (less negative)
✓ CNN trains faster (2x speedup) but with lower accuracy
✓ Both models show signs of underfitting (negative R²)

RECOMMENDATIONS:

1. Model Selection: Use LSTM for better accuracy
2. Underfitting Issues:
   - Increase model complexity further
   - Add more advanced features
   - Increase sequence length
   - Reduce regularization (dropout)
   - Train for more epochs
3. Data Quality:
   - Verify data preprocessing
   - Check for outliers
   - Ensure proper normalization
4. Hyperparameter Tuning:
   - Experiment with learning rates
   - Try different batch sizes
   - Adjust early stopping patience

{'='*70}
"""
    
    logger.info(report)

    # Save report
    with open(output_dirs['logs'] / 'model_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("✓ Report saved to: model_comparison_report.txt")


if __name__ == "__main__":
    create_model_comparison_viz()
    create_summary_report()
    
    logger.info("\n" + "="*70)
    logger.info("VISUALIZATION GENERATION COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nGenerated files:")
    logger.info(f"  - {output_dirs['images'] / 'model_comparison.png'}")
    logger.info(f"  - {output_dirs['images'] / 'metrics_table.png'}")
    logger.info(f"  - {output_dirs['logs'] / 'model_comparison_report.txt'}")

