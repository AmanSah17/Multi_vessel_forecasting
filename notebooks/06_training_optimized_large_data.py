"""
Optimized Training for Large Datasets

Handles large AIS datasets (7M+ records) with memory-efficient processing.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training_pipeline import TrainingPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptimizedTrainingLogger:
    """Optimized logger for large datasets."""
    
    def __init__(self, output_dir='training_logs_optimized'):
        """Initialize logger."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        logger.info(f"Optimized training logger initialized: {output_dir}")
    
    def log_section(self, title):
        """Log section header."""
        logger.info(f"\n{'='*70}")
        logger.info(f"{title}")
        logger.info(f"{'='*70}")
    
    def log_data_info(self, df, stage='raw'):
        """Log data information efficiently."""
        self.log_section(f"DATA INFO - {stage.upper()}")
        
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info(f"Missing values: {df.isnull().sum().sum():,}")
        logger.info(f"Duplicates: {df.duplicated().sum():,}")
        
        if 'MMSI' in df.columns:
            logger.info(f"Vessels: {df['MMSI'].nunique():,}")
        if 'BaseDateTime' in df.columns:
            logger.info(f"Time range: {df['BaseDateTime'].min()} to {df['BaseDateTime'].max()}")
        
        self.results[f'data_info_{stage}'] = {
            'shape': df.shape,
            'memory_mb': float(df.memory_usage(deep=True).sum() / 1024**2),
            'missing_values': int(df.isnull().sum().sum()),
            'duplicates': int(df.duplicated().sum()),
            'vessels': int(df['MMSI'].nunique()) if 'MMSI' in df.columns else 0
        }
    
    def log_split_info(self, train_df, val_df, test_df):
        """Log split information."""
        self.log_section("TRAIN/VAL/TEST SPLIT")
        
        total = len(train_df) + len(val_df) + len(test_df)
        logger.info(f"Training:   {len(train_df):,} ({len(train_df)/total*100:.1f}%)")
        logger.info(f"Validation: {len(val_df):,} ({len(val_df)/total*100:.1f}%)")
        logger.info(f"Test:       {len(test_df):,} ({len(test_df)/total*100:.1f}%)")
        logger.info(f"Total:      {total:,}")
        
        self.results['split_info'] = {
            'train': len(train_df),
            'validation': len(val_df),
            'test': len(test_df),
            'total': total
        }
    
    def log_training_step(self, step_num, step_name, status, details=None):
        """Log training step."""
        logger.info(f"\n[{step_num}/8] {step_name}")
        logger.info(f"Status: {status}")
        if details:
            for key, value in details.items():
                logger.info(f"  {key}: {value}")
    
    def log_metrics(self, metrics):
        """Log evaluation metrics."""
        self.log_section("EVALUATION METRICS")
        
        for metric_type, values in metrics.items():
            logger.info(f"\n{metric_type}:")
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {key}: {value:.4f}")
                    else:
                        logger.info(f"  {key}: {value}")
        
        self.results['evaluation_metrics'] = metrics
    
    def save_results(self):
        """Save results to JSON."""
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {results_path}")


def run_optimized_training(df, sample_size=None, output_dir='training_logs_optimized'):
    """Run optimized training for large datasets."""
    
    logger.info("\n" + "="*70)
    logger.info("MARITIME VESSEL FORECASTING - OPTIMIZED TRAINING")
    logger.info("="*70)
    
    train_logger = OptimizedTrainingLogger(output_dir)
    pipeline = TrainingPipeline(output_dir='models')
    
    try:
        # Step 1: Log raw data
        train_logger.log_training_step(1, "Logging raw data", "IN PROGRESS")
        train_logger.log_data_info(df, 'raw')
        
        # Step 2: Sample data if needed (for memory efficiency)
        if sample_size and len(df) > sample_size:
            train_logger.log_training_step(2, "Sampling data", "IN PROGRESS", {
                'original_size': len(df),
                'sample_size': sample_size,
                'sampling_ratio': f"{sample_size/len(df)*100:.1f}%"
            })
            
            # Sample stratified by MMSI to maintain vessel distribution
            df_sampled = df.groupby('MMSI', group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, int(len(x) * sample_size / len(df)))))
            ).reset_index(drop=True)
            
            logger.info(f"✓ Sampled to {len(df_sampled):,} records")
            df = df_sampled
        else:
            train_logger.log_training_step(2, "Data sampling", "SKIPPED", {
                'reason': 'Data size within limits'
            })
        
        # Step 3: Load and preprocess
        train_logger.log_training_step(3, "Loading and preprocessing", "IN PROGRESS")
        df_processed = pipeline.load_data(df)
        train_logger.log_data_info(df_processed, 'processed')
        
        # Step 4: Feature engineering
        train_logger.log_training_step(4, "Feature engineering", "IN PROGRESS")
        df_features = pipeline.engineer_features(df_processed)
        logger.info(f"✓ Features: {len(df_features.columns)} columns")
        
        # Step 5: Create train/val/test split
        train_logger.log_training_step(5, "Creating train/val/test split", "IN PROGRESS")
        train_df, val_df, test_df = pipeline.create_train_val_test_split(df_features)
        train_logger.log_split_info(train_df, val_df, test_df)
        
        # Step 6: Train prediction models
        train_logger.log_training_step(6, "Training prediction models", "IN PROGRESS", {
            'models': 'Kalman Filter, ARIMA, Ensemble',
            'training_records': len(train_df),
            'validation_records': len(val_df)
        })
        pipeline.train_prediction_models(train_df, val_df)
        logger.info("✓ Prediction models trained")
        
        # Step 7: Train anomaly detectors
        train_logger.log_training_step(7, "Training anomaly detectors", "IN PROGRESS", {
            'detectors': 'Isolation Forest, Rule-based, Ensemble',
            'training_records': len(train_df)
        })
        pipeline.train_anomaly_detectors(train_df)
        logger.info("✓ Anomaly detectors trained")
        
        # Step 8: Evaluate models
        train_logger.log_training_step(8, "Evaluating models", "IN PROGRESS", {
            'test_records': len(test_df)
        })
        metrics = pipeline.evaluate(test_df)
        train_logger.log_metrics(metrics)
        logger.info("✓ Models evaluated")
        
        # Save models
        logger.info("\nSaving trained models...")
        pipeline.save_models()
        logger.info("✓ All models saved")
        
        # Save results
        train_logger.save_results()
        
        logger.info("\n" + "="*70)
        logger.info("✓ TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"\nResults saved to: {output_dir}/")
        logger.info(f"Log file: training_optimized.log")
        
        return pipeline, metrics, train_logger
    
    except Exception as e:
        logger.error(f"✗ Training failed: {e}", exc_info=True)
        raise


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("MARITIME VESSEL FORECASTING - OPTIMIZED TRAINING FOR LARGE DATA")
    print("="*70 + "\n")
    
    # Load AIS data
    try:
        data_path = r"D:\Maritime_Vessel_monitoring\csv_extracted_data\AIS_2020_01_03\AIS_2020_01_03.csv"
        logger.info(f"Loading data from: {data_path}")
        
        df = pd.read_csv(data_path)
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], format="%Y-%m-%dT%H:%M:%S", errors='coerce')
        logger.info(f"✓ Loaded {len(df):,} records")
    
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        return
    
    # Run optimized training with sampling
    # Sample to 500K records to avoid memory issues
    pipeline, metrics, train_logger = run_optimized_training(
        df,
        sample_size=500000,  # Sample to 500K records
        output_dir='training_logs_optimized'
    )
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print("\nResults saved to: training_logs_optimized/")
    print("  - training_results.json")
    print("  - training_optimized.log")


if __name__ == '__main__':
    main()

