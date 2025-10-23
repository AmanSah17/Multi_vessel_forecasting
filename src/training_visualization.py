"""
Training Visualization Module

Provides comprehensive visualization for training and validation:
- Training progress
- Model performance comparison
- Prediction accuracy
- Anomaly detection metrics
- Consistency verification
- Data distribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """Visualizes training and validation metrics."""
    
    def __init__(self, figsize: Tuple[int, int] = (16, 12)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = figsize
    
    def plot_data_split(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                       test_df: pd.DataFrame, output_path: str = None):
        """
        Visualize train/val/test split.
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            output_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Data size distribution
        sizes = [len(train_df), len(val_df), len(test_df)]
        labels = ['Train', 'Validation', 'Test']
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        axes[0, 0].set_title('Data Split Distribution', fontsize=14, fontweight='bold')
        
        # Records per split
        axes[0, 1].bar(labels, sizes, color=colors)
        axes[0, 1].set_ylabel('Number of Records')
        axes[0, 1].set_title('Records per Split', fontsize=14, fontweight='bold')
        for i, v in enumerate(sizes):
            axes[0, 1].text(i, v + 100, str(v), ha='center', fontweight='bold')
        
        # Temporal distribution
        train_df_copy = train_df.copy()
        val_df_copy = val_df.copy()
        test_df_copy = test_df.copy()
        
        train_df_copy['split'] = 'Train'
        val_df_copy['split'] = 'Validation'
        test_df_copy['split'] = 'Test'
        
        combined = pd.concat([train_df_copy, val_df_copy, test_df_copy])
        combined['date'] = combined['BaseDateTime'].dt.date
        
        daily_counts = combined.groupby(['date', 'split']).size().unstack(fill_value=0)
        daily_counts.plot(ax=axes[1, 0], color=colors)
        axes[1, 0].set_title('Records per Day by Split', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Record Count')
        axes[1, 0].legend(title='Split')
        
        # Vessel distribution
        vessel_counts = pd.DataFrame({
            'Train': train_df['MMSI'].value_counts(),
            'Validation': val_df['MMSI'].value_counts(),
            'Test': test_df['MMSI'].value_counts(),
        }).fillna(0)
        
        vessel_counts.plot(kind='bar', ax=axes[1, 1], color=colors)
        axes[1, 1].set_title('Vessels per Split', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Vessel MMSI')
        axes[1, 1].set_ylabel('Record Count')
        axes[1, 1].legend(title='Split')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved data split visualization to {output_path}")
        
        return fig
    
    def plot_prediction_performance(self, predictions: Dict[str, np.ndarray],
                                   actual: np.ndarray, 
                                   output_path: str = None):
        """
        Visualize prediction performance.
        
        Args:
            predictions: Dictionary of model predictions
            actual: Actual values
            output_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Calculate metrics
        metrics = {}
        for model_name, pred in predictions.items():
            mae = np.mean(np.abs(pred - actual))
            rmse = np.sqrt(np.mean((pred - actual) ** 2))
            mape = np.mean(np.abs((actual - pred) / (np.abs(actual) + 1e-6))) * 100
            metrics[model_name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        
        # MAE comparison
        mae_values = [metrics[m]['MAE'] for m in predictions.keys()]
        axes[0, 0].bar(predictions.keys(), mae_values, color='#3498db')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].set_title('Mean Absolute Error by Model', fontsize=14, fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        rmse_values = [metrics[m]['RMSE'] for m in predictions.keys()]
        axes[0, 1].bar(predictions.keys(), rmse_values, color='#e74c3c')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Root Mean Squared Error by Model', fontsize=14, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        mape_values = [metrics[m]['MAPE'] for m in predictions.keys()]
        axes[1, 0].bar(predictions.keys(), mape_values, color='#2ecc71')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].set_title('Mean Absolute Percentage Error by Model', fontsize=14, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Prediction vs Actual (first model)
        first_model = list(predictions.keys())[0]
        axes[1, 1].scatter(actual, predictions[first_model], alpha=0.5)
        axes[1, 1].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual')
        axes[1, 1].set_ylabel('Predicted')
        axes[1, 1].set_title(f'{first_model} - Actual vs Predicted', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved prediction performance to {output_path}")
        
        return fig, metrics
    
    def plot_anomaly_detection_metrics(self, metrics: Dict, output_path: str = None):
        """
        Visualize anomaly detection metrics.
        
        Args:
            metrics: Dictionary with precision, recall, f1, auc
            output_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        detectors = list(metrics.keys())
        
        # Precision
        precision = [metrics[d].get('precision', 0) for d in detectors]
        axes[0, 0].bar(detectors, precision, color='#3498db')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_title('Precision by Detector', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Recall
        recall = [metrics[d].get('recall', 0) for d in detectors]
        axes[0, 1].bar(detectors, recall, color='#e74c3c')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_title('Recall by Detector', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1-Score
        f1 = [metrics[d].get('f1', 0) for d in detectors]
        axes[1, 0].bar(detectors, f1, color='#2ecc71')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_title('F1-Score by Detector', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # ROC-AUC
        auc = [metrics[d].get('auc', 0) for d in detectors]
        axes[1, 1].bar(detectors, auc, color='#f39c12')
        axes[1, 1].set_ylabel('ROC-AUC')
        axes[1, 1].set_title('ROC-AUC by Detector', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved anomaly detection metrics to {output_path}")
        
        return fig
    
    def plot_consistency_scores(self, consistency_scores: Dict[str, float],
                               output_path: str = None):
        """
        Visualize consistency scores.
        
        Args:
            consistency_scores: Dictionary of MMSI -> consistency score
            output_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        mmsis = list(consistency_scores.keys())
        scores = list(consistency_scores.values())
        
        # Bar chart
        colors = ['#2ecc71' if s > 0.8 else '#f39c12' if s > 0.6 else '#e74c3c' for s in scores]
        axes[0].barh(range(len(mmsis)), scores, color=colors)
        axes[0].set_yticks(range(len(mmsis)))
        axes[0].set_yticklabels(mmsis)
        axes[0].set_xlabel('Consistency Score')
        axes[0].set_title('Trajectory Consistency Scores', fontsize=14, fontweight='bold')
        axes[0].set_xlim([0, 1])
        
        # Distribution
        axes[1].hist(scores, bins=20, color='#3498db', edgecolor='black')
        axes[1].axvline(np.mean(scores), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.3f}')
        axes[1].axvline(np.median(scores), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.3f}')
        axes[1].set_xlabel('Consistency Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Consistency Scores', fontsize=14, fontweight='bold')
        axes[1].legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved consistency scores to {output_path}")
        
        return fig
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importance_scores: np.ndarray,
                               output_path: str = None):
        """
        Visualize feature importance.
        
        Args:
            feature_names: List of feature names
            importance_scores: Array of importance scores
            output_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_scores = importance_scores[sorted_idx]
        
        # Plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
        ax.barh(range(len(sorted_features)), sorted_scores, color=colors)
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance to {output_path}")
        
        return fig
    
    def plot_training_summary(self, train_metrics: Dict, val_metrics: Dict,
                             output_path: str = None):
        """
        Visualize training summary.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            output_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Training vs Validation Loss
        epochs = range(1, len(train_metrics.get('loss', [])) + 1)
        if 'loss' in train_metrics and 'loss' in val_metrics:
            axes[0, 0].plot(epochs, train_metrics['loss'], 'b-', label='Train')
            axes[0, 0].plot(epochs, val_metrics['loss'], 'r-', label='Validation')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
            axes[0, 0].legend()
        
        # Accuracy
        if 'accuracy' in train_metrics and 'accuracy' in val_metrics:
            axes[0, 1].plot(epochs, train_metrics['accuracy'], 'b-', label='Train')
            axes[0, 1].plot(epochs, val_metrics['accuracy'], 'r-', label='Validation')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
            axes[0, 1].legend()
        
        # Metrics summary
        axes[1, 0].axis('off')
        summary_text = "Training Summary\n" + "=" * 40 + "\n"
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                summary_text += f"{key}: {value:.4f}\n"
        axes[1, 0].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                       verticalalignment='center')
        
        # Validation summary
        axes[1, 1].axis('off')
        summary_text = "Validation Summary\n" + "=" * 40 + "\n"
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                summary_text += f"{key}: {value:.4f}\n"
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training summary to {output_path}")
        
        return fig

