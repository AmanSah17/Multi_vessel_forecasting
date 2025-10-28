"""
MLflow Monitoring Dashboard for XGBoost Vessel Trajectory Model

Visualizes:
1. Training and validation metrics over trials
2. Hyperparameter importance
3. Model performance comparison
4. Error distribution analysis
5. Real-time experiment tracking
"""

import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")

# Setup MLflow
mlflow.set_tracking_uri("file:./mlruns")


class MLflowMonitor:
    """Monitor and visualize MLflow experiments."""
    
    def __init__(self, experiment_name="XGBoost_Vessel_Trajectory_Forecasting"):
        """Initialize monitor."""
        self.experiment_name = experiment_name
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if self.experiment is None:
            logger.warning(f"Experiment '{experiment_name}' not found")
            return
        
        self.experiment_id = self.experiment.experiment_id
        logger.info(f"Monitoring experiment: {experiment_name} (ID: {self.experiment_id})")
    
    def get_runs_data(self):
        """Fetch all runs from experiment."""
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_ids=[self.experiment_id])
        
        data = []
        for run in runs:
            run_data = {
                'run_id': run.info.run_id,
                'run_name': run.info.run_name,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
            }
            
            # Add metrics
            for key, metric in run.data.metrics.items():
                run_data[key] = metric
            
            # Add params
            for key, param in run.data.params.items():
                run_data[f'param_{key}'] = param
            
            data.append(run_data)
        
        return pd.DataFrame(data)
    
    def plot_training_metrics(self):
        """Plot training metrics over trials."""
        df = self.get_runs_data()
        
        # Filter hyperparameter tuning runs
        tuning_runs = df[df['run_name'].str.contains('hyperparameter_tuning', na=False)]
        
        if len(tuning_runs) == 0:
            logger.warning("No hyperparameter tuning runs found")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Metrics Over Trials', fontsize=14, fontweight='bold')
        
        # Plot 1: Train vs Val MAE
        if 'train_mae' in tuning_runs.columns and 'val_mae' in tuning_runs.columns:
            axes[0, 0].plot(tuning_runs['train_mae'], 'b-o', label='Train MAE', linewidth=2)
            axes[0, 0].plot(tuning_runs['val_mae'], 'r-s', label='Val MAE', linewidth=2)
            axes[0, 0].set_xlabel('Trial')
            axes[0, 0].set_ylabel('MAE')
            axes[0, 0].set_title('Training vs Validation MAE')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Validation RMSE
        if 'val_rmse' in tuning_runs.columns:
            axes[0, 1].plot(tuning_runs['val_rmse'], 'g-^', linewidth=2)
            axes[0, 1].set_xlabel('Trial')
            axes[0, 1].set_ylabel('RMSE')
            axes[0, 1].set_title('Validation RMSE')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Validation R2
        if 'val_r2' in tuning_runs.columns:
            axes[1, 0].plot(tuning_runs['val_r2'], 'purple', marker='D', linewidth=2)
            axes[1, 0].set_xlabel('Trial')
            axes[1, 0].set_ylabel('R² Score')
            axes[1, 0].set_title('Validation R² Score')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Best metrics summary
        axes[1, 1].axis('off')
        summary_text = f"""
        BEST TRIAL SUMMARY
        ─────────────────────
        Total Trials: {len(tuning_runs)}
        
        Best Val MAE: {tuning_runs['val_mae'].min():.4f}
        Best Val RMSE: {tuning_runs['val_rmse'].min():.4f}
        Best Val R²: {tuning_runs['val_r2'].max():.4f}
        
        Avg Train MAE: {tuning_runs['train_mae'].mean():.4f}
        Avg Val MAE: {tuning_runs['val_mae'].mean():.4f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_test_metrics(self):
        """Plot test evaluation metrics."""
        df = self.get_runs_data()
        
        # Filter final model runs
        final_runs = df[df['run_name'].str.contains('complete_pipeline', na=False)]
        
        if len(final_runs) == 0:
            logger.warning("No complete pipeline runs found")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Test Set Evaluation Metrics', fontsize=14, fontweight='bold')
        
        # Get latest run
        latest_run = final_runs.iloc[-1]
        
        # Plot 1: Position errors (LAT, LON)
        metrics_pos = {
            'LAT MAE': latest_run.get('test_LAT_MAE_degrees', 0),
            'LON MAE': latest_run.get('test_LON_MAE_degrees', 0),
            'LAT RMSE': latest_run.get('test_LAT_RMSE_degrees', 0),
            'LON RMSE': latest_run.get('test_LON_RMSE_degrees', 0),
        }
        
        axes[0, 0].bar(metrics_pos.keys(), metrics_pos.values(), color=['blue', 'red', 'lightblue', 'lightcoral'])
        axes[0, 0].set_ylabel('Error (degrees)')
        axes[0, 0].set_title('Position Errors')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Speed and Course errors
        metrics_nav = {
            'SOG MAE': latest_run.get('test_SOG_MAE_knots', 0),
            'COG MAE': latest_run.get('test_COG_MAE_degrees', 0),
        }
        
        axes[0, 1].bar(metrics_nav.keys(), metrics_nav.values(), color=['green', 'orange'])
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].set_title('Navigation Errors')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: R² Scores
        r2_scores = {
            'LAT R²': latest_run.get('test_LAT_R2', 0),
            'LON R²': latest_run.get('test_LON_R2', 0),
        }
        
        axes[1, 0].bar(r2_scores.keys(), r2_scores.values(), color=['darkblue', 'darkred'])
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('Model R² Scores')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Geodesic errors
        geo_metrics = {
            'Mean (m)': latest_run.get('test_Geodesic_Error_Mean_m', 0),
            'Median (m)': latest_run.get('test_Geodesic_Error_Median_m', 0),
            'Std (m)': latest_run.get('test_Geodesic_Error_Std_m', 0),
        }
        
        axes[1, 1].bar(geo_metrics.keys(), geo_metrics.values(), color=['purple', 'violet', 'plum'])
        axes[1, 1].set_ylabel('Error (meters)')
        axes[1, 1].set_title('Geodesic Distance Errors')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_hyperparameter_importance(self):
        """Analyze hyperparameter importance."""
        df = self.get_runs_data()
        
        # Filter tuning runs
        tuning_runs = df[df['run_name'].str.contains('hyperparameter_tuning', na=False)]
        
        if len(tuning_runs) == 0:
            logger.warning("No hyperparameter tuning runs found")
            return None
        
        # Extract hyperparameters
        param_cols = [col for col in tuning_runs.columns if col.startswith('param_')]
        
        if len(param_cols) == 0:
            logger.warning("No hyperparameters found")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Hyperparameter Analysis', fontsize=14, fontweight='bold')
        
        # Plot correlations with val_mae
        if 'val_mae' in tuning_runs.columns:
            correlations = {}
            for param_col in param_cols:
                try:
                    # Convert to numeric if possible
                    param_vals = pd.to_numeric(tuning_runs[param_col], errors='coerce')
                    if param_vals.notna().sum() > 0:
                        corr = param_vals.corr(tuning_runs['val_mae'])
                        correlations[param_col.replace('param_', '')] = corr
                except:
                    pass
            
            if correlations:
                sorted_corr = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
                axes[0, 0].barh(list(sorted_corr.keys()), list(sorted_corr.values()))
                axes[0, 0].set_xlabel('Correlation with Val MAE')
                axes[0, 0].set_title('Hyperparameter Importance')
                axes[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        
        # Plot learning rate vs val_mae
        if 'param_learning_rate' in tuning_runs.columns and 'val_mae' in tuning_runs.columns:
            lr_vals = pd.to_numeric(tuning_runs['param_learning_rate'], errors='coerce')
            axes[0, 1].scatter(lr_vals, tuning_runs['val_mae'], alpha=0.6, s=50)
            axes[0, 1].set_xlabel('Learning Rate')
            axes[0, 1].set_ylabel('Val MAE')
            axes[0, 1].set_title('Learning Rate vs Validation MAE')
            axes[0, 1].set_xscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot max_depth vs val_mae
        if 'param_max_depth' in tuning_runs.columns and 'val_mae' in tuning_runs.columns:
            depth_vals = pd.to_numeric(tuning_runs['param_max_depth'], errors='coerce')
            axes[1, 0].scatter(depth_vals, tuning_runs['val_mae'], alpha=0.6, s=50, color='green')
            axes[1, 0].set_xlabel('Max Depth')
            axes[1, 0].set_ylabel('Val MAE')
            axes[1, 0].set_title('Max Depth vs Validation MAE')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot n_estimators vs val_mae
        if 'param_n_estimators' in tuning_runs.columns and 'val_mae' in tuning_runs.columns:
            n_est_vals = pd.to_numeric(tuning_runs['param_n_estimators'], errors='coerce')
            axes[1, 1].scatter(n_est_vals, tuning_runs['val_mae'], alpha=0.6, s=50, color='red')
            axes[1, 1].set_xlabel('Number of Estimators')
            axes[1, 1].set_ylabel('Val MAE')
            axes[1, 1].set_title('N Estimators vs Validation MAE')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, output_dir='mlflow_reports'):
        """Generate comprehensive monitoring report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Generating MLflow monitoring report to {output_dir}...")
        
        # Plot 1: Training metrics
        fig1 = self.plot_training_metrics()
        if fig1:
            fig1.savefig(output_dir / 'training_metrics.png', dpi=120, bbox_inches='tight')
            logger.info("✓ Saved training_metrics.png")
        
        # Plot 2: Test metrics
        fig2 = self.plot_test_metrics()
        if fig2:
            fig2.savefig(output_dir / 'test_metrics.png', dpi=120, bbox_inches='tight')
            logger.info("✓ Saved test_metrics.png")
        
        # Plot 3: Hyperparameter importance
        fig3 = self.plot_hyperparameter_importance()
        if fig3:
            fig3.savefig(output_dir / 'hyperparameter_importance.png', dpi=120, bbox_inches='tight')
            logger.info("✓ Saved hyperparameter_importance.png")
        
        # Generate summary report
        df = self.get_runs_data()
        summary = f"""
# MLflow Monitoring Report
## Experiment: {self.experiment_name}

### Summary Statistics
- Total Runs: {len(df)}
- Completed Runs: {len(df[df['status'] == 'FINISHED'])}
- Failed Runs: {len(df[df['status'] == 'FAILED'])}

### Runs Overview
{df.to_string()}

### Best Metrics
{df.describe().to_string()}
"""
        
        with open(output_dir / 'report.md', 'w') as f:
            f.write(summary)
        
        logger.info("✓ Saved report.md")
        logger.info(f"Report generated successfully in {output_dir}")


def main():
    """Generate monitoring dashboard."""
    logger.info("\n" + "="*80)
    logger.info("MLFLOW MONITORING DASHBOARD")
    logger.info("="*80)
    
    monitor = MLflowMonitor()
    
    if monitor.experiment is None:
        logger.error("Experiment not found. Run training first.")
        return
    
    # Generate report
    monitor.generate_report()
    
    logger.info("\n" + "="*80)
    logger.info("MONITORING COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()

