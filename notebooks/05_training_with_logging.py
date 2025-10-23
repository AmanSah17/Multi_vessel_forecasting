"""
Training Pipeline with Comprehensive Logging

Trains the maritime vessel forecasting pipeline with detailed logging
of training, validation, and test results.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training_pipeline import TrainingPipeline
from src.training_visualization import TrainingVisualizer

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_with_logging.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingLogger:
    """Comprehensive training logger."""
    
    def __init__(self, output_dir='training_logs'):
        """Initialize logger."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        logger.info(f"Training logger initialized: {output_dir}")
    
    def log_data_info(self, df, stage='raw'):
        """Log data information."""
        logger.info(f"\n{'='*70}")
        logger.info(f"DATA INFO - {stage.upper()}")
        logger.info(f"{'='*70}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")
        logger.info(f"Duplicates: {df.duplicated().sum()}")
        
        if 'MMSI' in df.columns:
            logger.info(f"Vessels: {df['MMSI'].nunique()}")
        if 'BaseDateTime' in df.columns:
            logger.info(f"Time range: {df['BaseDateTime'].min()} to {df['BaseDateTime'].max()}")
        
        self.results[f'data_info_{stage}'] = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': int(df.isnull().sum().sum()),
            'duplicates': int(df.duplicated().sum())
        }
    
    def log_split_info(self, train_df, val_df, test_df):
        """Log train/val/test split information."""
        total = len(train_df) + len(val_df) + len(test_df)
        
        logger.info(f"\n{'='*70}")
        logger.info("TRAIN/VAL/TEST SPLIT")
        logger.info(f"{'='*70}")
        logger.info(f"Training:   {len(train_df):,} ({len(train_df)/total*100:.1f}%)")
        logger.info(f"Validation: {len(val_df):,} ({len(val_df)/total*100:.1f}%)")
        logger.info(f"Test:       {len(test_df):,} ({len(test_df)/total*100:.1f}%)")
        logger.info(f"Total:      {total:,}")
        
        self.results['split_info'] = {
            'train': len(train_df),
            'validation': len(val_df),
            'test': len(test_df),
            'total': total,
            'train_pct': len(train_df)/total*100,
            'val_pct': len(val_df)/total*100,
            'test_pct': len(test_df)/total*100
        }
    
    def log_model_training(self, model_name, status, details=None):
        """Log model training."""
        logger.info(f"\n{'='*70}")
        logger.info(f"MODEL TRAINING: {model_name}")
        logger.info(f"{'='*70}")
        logger.info(f"Status: {status}")
        if details:
            for key, value in details.items():
                logger.info(f"  {key}: {value}")
        
        self.results[f'model_{model_name}'] = {
            'status': status,
            'details': details or {}
        }
    
    def log_evaluation(self, metrics):
        """Log evaluation metrics."""
        logger.info(f"\n{'='*70}")
        logger.info("EVALUATION METRICS")
        logger.info(f"{'='*70}")
        
        for metric_type, values in metrics.items():
            logger.info(f"\n{metric_type}:")
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {key}: {value:.4f}")
                    else:
                        logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {values}")
        
        self.results['evaluation_metrics'] = metrics
    
    def save_results(self):
        """Save results to JSON."""
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {results_path}")
    
    def generate_report(self):
        """Generate comprehensive report."""
        report = f"""
MARITIME VESSEL FORECASTING - TRAINING REPORT
{'='*70}

EXECUTION TIME: {datetime.now().isoformat()}

DATA STATISTICS:
{json.dumps(self.results.get('data_info_raw', {}), indent=2)}

PREPROCESSED DATA:
{json.dumps(self.results.get('data_info_processed', {}), indent=2)}

TRAIN/VAL/TEST SPLIT:
{json.dumps(self.results.get('split_info', {}), indent=2)}

EVALUATION METRICS:
{json.dumps(self.results.get('evaluation_metrics', {}), indent=2, default=str)}

{'='*70}
"""
        report_path = self.output_dir / 'training_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_path}")


def run_training_with_logging(df, output_dir='training_logs'):
    """Run training with comprehensive logging."""
    
    logger.info("\n" + "="*70)
    logger.info("MARITIME VESSEL FORECASTING - TRAINING WITH LOGGING")
    logger.info("="*70)
    
    # Initialize logger and pipeline
    train_logger = TrainingLogger(output_dir)
    pipeline = TrainingPipeline(output_dir='models')
    
    try:
        # Step 1: Log raw data
        logger.info("\n[1/8] Logging raw data...")
        train_logger.log_data_info(df, 'raw')
        
        # Step 2: Load and preprocess
        logger.info("\n[2/8] Loading and preprocessing data...")
        df_processed = pipeline.load_data(df)
        train_logger.log_data_info(df_processed, 'processed')
        
        # Step 3: Feature engineering
        logger.info("\n[3/8] Engineering features...")
        df_features = pipeline.engineer_features(df_processed)
        logger.info(f"Features engineered: {len(df_features.columns)} columns")
        logger.info(f"Feature columns: {df_features.columns.tolist()}")
        
        # Step 4: Create train/val/test split
        logger.info("\n[4/8] Creating train/val/test split...")
        train_df, val_df, test_df = pipeline.create_train_val_test_split(df_features)
        train_logger.log_split_info(train_df, val_df, test_df)
        
        # Step 5: Train prediction models
        logger.info("\n[5/8] Training prediction models...")
        pipeline.train_prediction_models(train_df, val_df)
        train_logger.log_model_training('prediction_ensemble', 'TRAINED', {
            'models': ['Kalman Filter', 'ARIMA', 'Ensemble'],
            'training_records': len(train_df),
            'validation_records': len(val_df)
        })
        
        # Step 6: Train anomaly detectors
        logger.info("\n[6/8] Training anomaly detectors...")
        pipeline.train_anomaly_detectors(train_df)
        train_logger.log_model_training('anomaly_ensemble', 'TRAINED', {
            'detectors': ['Isolation Forest', 'Rule-based', 'Ensemble'],
            'training_records': len(train_df)
        })
        
        # Step 7: Evaluate models
        logger.info("\n[7/8] Evaluating models on test set...")
        metrics = pipeline.evaluate(test_df)
        train_logger.log_evaluation(metrics)
        
        # Step 8: Save models
        logger.info("\n[8/8] Saving trained models...")
        pipeline.save_models()
        logger.info("✓ All models saved successfully")
        
        # Save results
        train_logger.save_results()
        train_logger.generate_report()
        
        logger.info("\n" + "="*70)
        logger.info("✓ TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"\nResults saved to: {output_dir}/")
        logger.info(f"Log file: training_with_logging.log")
        
        return pipeline, metrics, train_logger
    
    except Exception as e:
        logger.error(f"✗ Training failed: {e}", exc_info=True)
        raise


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("MARITIME VESSEL FORECASTING - TRAINING WITH LOGGING")
    print("="*70 + "\n")
    
    # Try to load AIS data
    try:
        # Try loading from the known path
        data_path = r"D:\Maritime_Vessel_monitoring\csv_extracted_data\AIS_2020_01_03\AIS_2020_01_03.csv"
        logger.info(f"Attempting to load data from: {data_path}")
        
        df = pd.read_csv(data_path)
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], format="%Y-%m-%dT%H:%M:%S", errors='coerce')
        logger.info(f"✓ Loaded {len(df)} records from {data_path}")
    
    except FileNotFoundError:
        logger.warning(f"Data file not found at {data_path}")
        logger.info("Generating sample data instead...")
        
        # Generate sample data
        np.random.seed(42)
        dates = pd.date_range('2020-01-03', periods=50000, freq='1min')
        
        data = []
        for i in range(50000):
            mmsi = 200000000 + (i % 10)
            lat = 40 + np.sin(i/1000) * 0.5 + np.random.randn() * 0.01
            lon = -74 + np.cos(i/1000) * 0.5 + np.random.randn() * 0.01
            sog = np.abs(np.sin(i/500) * 10 + 12 + np.random.randn() * 0.5)
            cog = (i * 0.1) % 360
            
            data.append({
                'MMSI': mmsi,
                'BaseDateTime': dates[i],
                'LAT': lat,
                'LON': lon,
                'SOG': sog,
                'COG': cog,
                'VesselName': f'Vessel_{mmsi}',
                'IMO': 1000000 + mmsi % 1000000,
            })
        
        df = pd.DataFrame(data)
        logger.info(f"✓ Generated {len(df)} sample records")
    
    # Run training with logging
    pipeline, metrics, train_logger = run_training_with_logging(df, output_dir='training_logs')
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print("\nResults saved to: training_logs/")
    print("  - training_results.json")
    print("  - training_report.txt")
    print("  - training_with_logging.log")


if __name__ == '__main__':
    main()

