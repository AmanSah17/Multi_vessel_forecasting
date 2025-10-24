"""
Comprehensive Visualization Script
- Generates detailed plots for 300 vessels
- Shows LAT, LON, SOG, COG predictions vs actual
- Creates model comparison visualizations
- Generates training curves for all models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from tqdm import tqdm

# Setup
output_dirs = {
    'logs': Path('logs'),
    'results': Path('results'),
    'images': Path('results/images'),
    'csv': Path('results/csv'),
    'models': Path('results/models')
}

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
plt.rcParams['figure.figsize'] = (18, 12)


def plot_vessel_trajectories_detailed(csv_path, output_path, num_vessels=50):
    """Plot vessel trajectories with LAT/LON predictions from all models."""
    logger.info(f"\nGenerating trajectory plots for {num_vessels} vessels...")
    
    df = pd.read_csv(csv_path)
    unique_mmsi = df['MMSI'].unique()[:num_vessels]
    
    n_cols = 5
    n_rows = (len(unique_mmsi) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 4*n_rows))
    axes = axes.flatten()
    
    for idx, mmsi in enumerate(tqdm(unique_mmsi, desc="Plotting trajectories")):
        ax = axes[idx]
        vessel_data = df[df['MMSI'] == mmsi]
        
        # Plot actual trajectory
        ax.plot(vessel_data['actual_lon'], vessel_data['actual_lat'], 
                color='black', linewidth=2, alpha=0.9, label='Actual', marker='o', markersize=3)
        
        # Plot model predictions
        colors = {'lstm': '#1f77b4', 'cnn': '#d62728', 'gru': '#2ca02c'}
        for model_name, color in colors.items():
            if f'{model_name}_lon' in vessel_data.columns:
                ax.plot(vessel_data[f'{model_name}_lon'], vessel_data[f'{model_name}_lat'],
                        color=color, linewidth=1.5, alpha=0.7, label=model_name.upper(), marker='s', markersize=2)
        
        ax.set_title(f'Vessel {mmsi}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
        if idx % n_cols == 0:
            ax.legend(fontsize=8, loc='best')
    
    for j in range(len(unique_mmsi), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    plt.close()


def plot_metric_comparisons(csv_path, output_path):
    """Plot LAT, LON, SOG, COG predictions vs actual for all models."""
    logger.info("\nGenerating metric comparison plots...")

    df = pd.read_csv(csv_path)
    unique_mmsi = df['MMSI'].unique()[:20]  # Top 20 vessels

    fig, axes = plt.subplots(4, 5, figsize=(25, 16))

    metrics = ['lat', 'lon', 'sog', 'cog']
    models = ['lstm', 'cnn', 'gru']

    for row, metric in tqdm(enumerate(metrics), total=len(metrics), desc="Plotting metrics"):
        for col, mmsi in tqdm(enumerate(unique_mmsi), total=len(unique_mmsi), desc=f"  {metric.upper()}", leave=False):
            ax = axes[row, col]
            vessel_data = df[df['MMSI'] == mmsi]

            x = np.arange(len(vessel_data))
            actual_col = f'actual_{metric}'

            ax.plot(x, vessel_data[actual_col], color='black', linewidth=2, label='Actual', marker='o', markersize=4)

            colors = {'lstm': '#1f77b4', 'cnn': '#d62728', 'gru': '#2ca02c'}
            for model_name, color in colors.items():
                pred_col = f'{model_name}_{metric}'
                if pred_col in vessel_data.columns:
                    ax.plot(x, vessel_data[pred_col], color=color, linewidth=1.5, alpha=0.7,
                           label=model_name.upper(), marker='s', markersize=3)

            ax.set_title(f'Vessel {mmsi} - {metric.upper()}', fontsize=9, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel(metric.upper())
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(fontsize=7, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    plt.close()


def plot_mae_distribution(csv_path, output_path):
    """Plot MAE distribution across all vessels for each model and metric."""
    logger.info("\nGenerating MAE distribution plots...")

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = ['lat', 'lon', 'sog', 'cog']
    models = ['lstm', 'cnn', 'gru']

    for idx, metric in tqdm(enumerate(metrics), total=len(metrics), desc="Processing MAE metrics"):
        ax = axes[idx // 2, idx % 2]

        mae_data = []
        labels = []

        for model_name in tqdm(models, desc=f"  Computing {metric} MAE", leave=False):
            mae_col = f'{model_name}_mae_{metric}'
            if mae_col in df.columns:
                mae_values = df.groupby('MMSI')[mae_col].mean().values
                mae_data.append(mae_values)
                labels.append(model_name.upper())

        bp = ax.boxplot(mae_data, labels=labels, patch_artist=True)
        colors = ['#1f77b4', '#d62728', '#2ca02c']
        for patch, color in zip(bp['boxes'], colors[:len(labels)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f'MAE Distribution - {metric.upper()}', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAE')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    plt.close()


def plot_model_performance_summary(csv_comparison_path, output_path):
    """Plot overall model performance comparison."""
    logger.info("\nGenerating model performance summary...")
    
    df = pd.read_csv(csv_comparison_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['MAE', 'RMSE', 'R2', 'MAE_LAT']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        if metric in df.columns:
            bars = ax.bar(df['Model'], df[metric], color=['#1f77b4', '#d62728', '#2ca02c'], alpha=0.7, edgecolor='black', linewidth=2)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    plt.close()


def plot_absolute_errors_heatmap(csv_path, output_path):
    """Plot absolute errors as heatmap for top vessels."""
    logger.info("\nGenerating absolute errors heatmap...")

    df = pd.read_csv(csv_path)

    # Calculate absolute errors
    metrics = ['lat', 'lon', 'sog', 'cog']
    models = ['lstm', 'cnn', 'gru']

    error_data = []
    vessel_labels = []

    for mmsi in tqdm(df['MMSI'].unique()[:30], desc="Computing vessel errors", unit="vessel"):  # Top 30 vessels
        vessel_data = df[df['MMSI'] == mmsi]
        vessel_labels.append(f"V{mmsi}")

        vessel_errors = []
        for model_name in models:
            for metric in metrics:
                mae_col = f'{model_name}_mae_{metric}'
                if mae_col in vessel_data.columns:
                    vessel_errors.append(vessel_data[mae_col].mean())

        error_data.append(vessel_errors)

    error_matrix = np.array(error_data)

    logger.info("Rendering heatmap...")
    fig, ax = plt.subplots(figsize=(14, 10))

    im = ax.imshow(error_matrix, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(np.arange(len(models) * len(metrics)))
    ax.set_yticks(np.arange(len(vessel_labels)))

    labels = [f"{m}_{mt}" for m in models for mt in metrics]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(vessel_labels, fontsize=8)

    ax.set_title('Absolute Errors Heatmap (MAE) - Top 30 Vessels', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MAE', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE VISUALIZATION GENERATION")
    logger.info("="*80)
    
    csv_path = output_dirs['csv'] / 'vessel_predictions_300_detailed.csv'
    csv_comparison_path = output_dirs['csv'] / 'model_comparison_comprehensive.csv'
    
    if csv_path.exists():
        plot_vessel_trajectories_detailed(csv_path, output_dirs['images'] / 'vessel_trajectories_50.png', num_vessels=50)
        plot_metric_comparisons(csv_path, output_dirs['images'] / 'metric_comparisons_20vessels.png')
        plot_mae_distribution(csv_path, output_dirs['images'] / 'mae_distribution.png')
        plot_absolute_errors_heatmap(csv_path, output_dirs['images'] / 'absolute_errors_heatmap.png')
    
    if csv_comparison_path.exists():
        plot_model_performance_summary(csv_comparison_path, output_dirs['images'] / 'model_performance_summary.png')
    
    logger.info(f"\n{'='*80}\nVISUALIZATION COMPLETE!\n{'='*80}")
    logger.info(f"All plots saved to: {output_dirs['images']}")

