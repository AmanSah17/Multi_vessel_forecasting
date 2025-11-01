"""
Main Execution Script: Enhanced LSTM Pipeline with Haversine Distance & Hyperparameter Tuning

Complete pipeline that:
1. Loads and preprocesses AIS data
2. Performs EDA
3. Applies clustering and PCA
4. Creates sequences
5. Performs hyperparameter tuning with Optuna
6. Trains final model with best parameters
7. Evaluates with haversine distance metrics
8. Saves all artifacts and visualizations

Usage:
    python 49_main_enhanced_lstm_pipeline.py
"""

import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import torch
import mlflow
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_lstm_haversine_main.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory
Path('logs').mkdir(exist_ok=True)

# Import all functions from the three parts
sys.path.append(str(Path(__file__).parent))

from enhanced_lstm_haversine_tuning import (
    load_all_data,
    perform_eda,
    prepare_features,
    haversine_distance,
    calculate_haversine_errors,
    EnhancedLSTMModel
)

from enhanced_lstm_haversine_tuning_part2 import (
    apply_clustering_and_pca,
    create_sequences_per_vessel,
    hyperparameter_tuning
)

from enhanced_lstm_haversine_tuning_part3 import (
    train_final_model,
    plot_training_curves,
    evaluate_with_haversine,
    save_model_and_artifacts
)


def main():
    """Main pipeline execution."""
    
    logger.info("\n" + "="*80)
    logger.info("ENHANCED LSTM PIPELINE WITH HAVERSINE DISTANCE & HYPERPARAMETER TUNING")
    logger.info("="*80)
    
    # Configuration
    CONFIG = {
        'start_date': 3,
        'end_date': 8,
        'sample_per_day': 50000,  # Limit samples per day for faster processing
        'n_clusters': 5,
        'n_components': 10,
        'seq_length': 30,
        'n_trials': 20,  # Number of Optuna trials
        'final_epochs': 200,
        'output_dir': 'results/enhanced_lstm_haversine'
    }
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start MLflow run
    with mlflow.start_run(run_name="enhanced_lstm_haversine_pipeline"):
        
        # Log configuration
        mlflow.log_params(CONFIG)
        
        # Step 1: Load data
        df = load_all_data(
            start_date=CONFIG['start_date'],
            end_date=CONFIG['end_date'],
            sample_per_day=CONFIG['sample_per_day']
        )
        
        # Step 2: EDA
        perform_eda(df, output_dir=CONFIG['output_dir'])
        
        # Step 3: Prepare features
        df, features = prepare_features(df)
        
        # Step 4: Clustering & PCA
        df, X_pca, pca, kmeans = apply_clustering_and_pca(
            df, 
            features, 
            n_clusters=CONFIG['n_clusters'],
            n_components=CONFIG['n_components'],
            output_dir=CONFIG['output_dir']
        )
        
        # Step 5: Create sequences
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = create_sequences_per_vessel(
            df, 
            features, 
            seq_length=CONFIG['seq_length']
        )
        
        logger.info(f"\nDataset sizes:")
        logger.info(f"  Train: {len(X_train):,} sequences")
        logger.info(f"  Val:   {len(X_val):,} sequences")
        logger.info(f"  Test:  {len(X_test):,} sequences")
        
        # Step 6: Hyperparameter tuning
        best_params = hyperparameter_tuning(
            X_train, 
            y_train, 
            X_val, 
            y_val, 
            n_trials=CONFIG['n_trials']
        )
        
        # Log best params
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        
        # Step 7: Train final model with best parameters
        model, train_losses, val_losses, train_maes, val_maes, val_haversine_means, device = train_final_model(
            X_train,
            y_train,
            X_val,
            y_val,
            best_params,
            epochs=CONFIG['final_epochs'],
            output_dir=CONFIG['output_dir']
        )
        
        # Step 8: Plot training curves
        plot_training_curves(
            train_losses,
            val_losses,
            train_maes,
            val_maes,
            val_haversine_means,
            output_dir=CONFIG['output_dir']
        )
        
        # Step 9: Evaluate on test set with haversine metrics
        metrics = evaluate_with_haversine(
            model,
            X_test,
            y_test,
            device,
            output_dir=CONFIG['output_dir']
        )
        
        # Step 10: Save model and artifacts
        save_model_and_artifacts(
            model,
            scaler,
            best_params,
            metrics,
            output_dir=CONFIG['output_dir']
        )
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info(f"\n📊 Final Test Metrics:")
        logger.info(f"  MAE:  {metrics['mae']:.6f}")
        logger.info(f"  RMSE: {metrics['rmse']:.6f}")
        logger.info(f"  R²:   {metrics['r2']:.6f}")
        logger.info(f"\n🌍 Haversine Distance Errors:")
        logger.info(f"  Mean:   {metrics['haversine_mean_m']:.2f} meters")
        logger.info(f"  Median: {metrics['haversine_median_m']:.2f} meters")
        logger.info(f"  P95:    {metrics['haversine_p95_m']:.2f} meters")
        logger.info(f"  P99:    {metrics['haversine_p99_m']:.2f} meters")
        logger.info(f"\n✓ All outputs saved to: {CONFIG['output_dir']}")
        logger.info(f"✓ MLflow tracking URI: file:./mlruns")
        logger.info(f"✓ View results: mlflow ui")
        
        return model, metrics


if __name__ == "__main__":
    try:
        model, metrics = main()
        logger.info("\n✅ Pipeline executed successfully!")
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed with error: {e}", exc_info=True)
        raise

