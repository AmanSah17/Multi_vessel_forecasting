"""
Quick Training Demo - Fast Results

Trains on a smaller sample (50K records) for quick demonstration.
Completes in ~5-10 minutes instead of 60-90 minutes.
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
        logging.FileHandler('training_quick_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_quick_training(df, sample_size=50000, output_dir='training_logs_quick'):
    """Run quick training demo."""
    
    logger.info("\n" + "="*70)
    logger.info("MARITIME VESSEL FORECASTING - QUICK TRAINING DEMO")
    logger.info("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    pipeline = TrainingPipeline(output_dir='models')
    results = {}
    
    try:
        # Step 1: Log raw data
        logger.info(f"\n[1/8] Raw data: {len(df):,} records, {df['MMSI'].nunique():,} vessels")
        results['raw_data'] = {
            'records': len(df),
            'vessels': df['MMSI'].nunique(),
            'memory_mb': float(df.memory_usage(deep=True).sum() / 1024**2)
        }
        
        # Step 2: Sample data
        logger.info(f"\n[2/8] Sampling to {sample_size:,} records...")
        df_sampled = df.groupby('MMSI', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(len(x) * sample_size / len(df))))),
            include_groups=False
        ).reset_index(drop=True)
        logger.info(f"✓ Sampled to {len(df_sampled):,} records")
        
        # Step 3: Load and preprocess
        logger.info(f"\n[3/8] Preprocessing...")
        df_processed = pipeline.load_data(df_sampled)
        logger.info(f"✓ Preprocessed to {len(df_processed):,} records")
        results['processed_data'] = {
            'records': len(df_processed),
            'vessels': df_processed['MMSI'].nunique()
        }
        
        # Step 4: Feature engineering
        logger.info(f"\n[4/8] Engineering features...")
        df_features = pipeline.engineer_features(df_processed)
        logger.info(f"✓ Features: {len(df_features.columns)} columns")
        
        # Step 5: Create train/val/test split
        logger.info(f"\n[5/8] Creating train/val/test split...")
        train_df, val_df, test_df = pipeline.create_train_val_test_split(df_features)
        logger.info(f"✓ Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
        results['split'] = {
            'train': len(train_df),
            'validation': len(val_df),
            'test': len(test_df)
        }
        
        # Step 6: Train prediction models
        logger.info(f"\n[6/8] Training prediction models...")
        pipeline.train_prediction_models(train_df, val_df)
        logger.info(f"✓ Prediction models trained")
        
        # Step 7: Train anomaly detectors
        logger.info(f"\n[7/8] Training anomaly detectors...")
        pipeline.train_anomaly_detectors(train_df)
        logger.info(f"✓ Anomaly detectors trained")
        
        # Step 8: Evaluate models
        logger.info(f"\n[8/8] Evaluating models...")
        metrics = pipeline.evaluate(test_df)
        logger.info(f"✓ Models evaluated")
        results['metrics'] = metrics
        
        # Save models
        logger.info(f"\nSaving models...")
        pipeline.save_models()
        logger.info(f"✓ Models saved")
        
        # Save results
        results_path = output_path / 'training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"✓ Results saved to {results_path}")
        
        logger.info("\n" + "="*70)
        logger.info("✓ QUICK TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
        return pipeline, metrics
    
    except Exception as e:
        logger.error(f"✗ Training failed: {e}", exc_info=True)
        raise


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("MARITIME VESSEL FORECASTING - QUICK TRAINING DEMO")
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
    
    # Run quick training with 50K sample
    pipeline, metrics = run_quick_training(
        df,
        sample_size=50000,  # Quick demo with 50K records
        output_dir='training_logs_quick'
    )
    
    print("\n" + "="*70)
    print("✓ QUICK TRAINING COMPLETE!")
    print("="*70)
    print("\nResults saved to: training_logs_quick/")
    print("  - training_results.json")
    print("  - training_quick_demo.log")
    print("\nModels saved to: models/")
    print("  - prediction_kalman.pkl")
    print("  - prediction_arima.pkl")
    print("  - prediction_ensemble.pkl")
    print("  - anomaly_isolation_forest.pkl")
    print("  - anomaly_rule_based.pkl")
    print("  - anomaly_ensemble.pkl")


if __name__ == '__main__':
    main()

