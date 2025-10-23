"""
Training and Validation Visualization Example

This script demonstrates how to:
1. Run the complete training pipeline
2. Visualize data splits
3. Visualize prediction performance
4. Visualize anomaly detection metrics
5. Visualize consistency scores
6. Generate comprehensive training reports
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training_pipeline import TrainingPipeline
from training_visualization import TrainingVisualizer
from data_preprocessing import VesselDataPreprocessor


def generate_sample_data(n_records: int = 10000, n_vessels: int = 5) -> pd.DataFrame:
    """
    Generate sample AIS data for demonstration.
    
    Args:
        n_records: Number of records to generate
        n_vessels: Number of vessels
        
    Returns:
        DataFrame with sample AIS data
    """
    print(f"Generating {n_records} sample records for {n_vessels} vessels...")
    
    np.random.seed(42)
    
    # Generate timestamps
    dates = pd.date_range('2024-01-01', periods=n_records, freq='1min')
    
    # Generate vessel data
    data = []
    for i in range(n_records):
        mmsi = 200000000 + np.random.randint(0, n_vessels)
        lat = 40 + np.random.randn() * 0.1
        lon = -74 + np.random.randn() * 0.1
        sog = np.abs(np.random.randn() * 5 + 10)
        cog = np.random.uniform(0, 360)
        
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
    print(f"Generated {len(df)} records")
    return df


def run_training_with_visualization(data_filepath: str = None, 
                                   output_dir: str = 'training_results'):
    """
    Run complete training pipeline with visualization.
    
    Args:
        data_filepath: Path to AIS data (if None, generates sample data)
        output_dir: Directory to save results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n" + "=" * 60)
    print("MARITIME VESSEL FORECASTING - TRAINING & VISUALIZATION")
    print("=" * 60 + "\n")
    
    # Step 1: Load or generate data
    print("STEP 1: Loading Data")
    print("-" * 60)
    
    if data_filepath and os.path.exists(data_filepath):
        print(f"Loading data from {data_filepath}")
        df = pd.read_csv(data_filepath)
    else:
        print("Generating sample data...")
        df = generate_sample_data(n_records=10000, n_vessels=5)
        sample_path = output_path / 'sample_data.csv'
        df.to_csv(sample_path, index=False)
        print(f"Sample data saved to {sample_path}")
    
    print(f"Loaded {len(df)} records for {df['MMSI'].nunique()} vessels\n")
    
    # Step 2: Initialize pipeline and visualizer
    print("STEP 2: Initializing Pipeline and Visualizer")
    print("-" * 60)
    
    pipeline = TrainingPipeline(output_dir='models')
    visualizer = TrainingVisualizer(figsize=(16, 12))
    
    print("Pipeline and visualizer initialized\n")
    
    # Step 3: Preprocess data
    print("STEP 3: Preprocessing Data")
    print("-" * 60)
    
    preprocessor = VesselDataPreprocessor()
    df = preprocessor.preprocess(df)
    print(f"Preprocessed data: {len(df)} records\n")
    
    # Step 4: Feature engineering
    print("STEP 4: Feature Engineering")
    print("-" * 60)
    
    df = pipeline.engineer_features(df)
    print(f"Engineered features: {df.columns.tolist()}\n")
    
    # Step 5: Create train/val/test split
    print("STEP 5: Creating Train/Val/Test Split")
    print("-" * 60)
    
    train_df, val_df, test_df = pipeline.create_train_val_test_split(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}\n")
    
    # Step 6: Visualize data split
    print("STEP 6: Visualizing Data Split")
    print("-" * 60)
    
    split_fig = visualizer.plot_data_split(
        train_df, val_df, test_df,
        output_path=output_path / 'data_split.png'
    )
    print("Data split visualization saved\n")
    
    # Step 7: Train models
    print("STEP 7: Training Models")
    print("-" * 60)
    
    pipeline.train_prediction_models(train_df, val_df)
    pipeline.train_anomaly_detectors(train_df)
    print("Models trained\n")
    
    # Step 8: Generate predictions
    print("STEP 8: Generating Predictions")
    print("-" * 60)

    X_test = test_df[['LAT', 'LON', 'SOG', 'COG']].values

    predictions = {}
    for model_name, model in pipeline.prediction_models.items():
        try:
            if model_name == 'kalman':
                pred = np.array([model.predict(X_test[i:i+1]) for i in range(len(X_test))])
                predictions[model_name] = pred.flatten()[:len(X_test)]
            elif model_name == 'arima':
                # Skip ARIMA for now due to shape issues
                continue
            elif model_name == 'ensemble':
                pred = np.array([model.predict(X_test[i:i+1]) for i in range(len(X_test))])
                predictions[model_name] = pred.flatten()[:len(X_test)]
        except Exception as e:
            print(f"Warning: Could not generate predictions for {model_name}: {e}")

    print(f"Generated predictions for {len(predictions)} models\n")

    # Step 9: Visualize prediction performance
    print("STEP 9: Visualizing Prediction Performance")
    print("-" * 60)

    if predictions:
        actual = X_test[:, 0]  # Use latitude as actual
        # Ensure all predictions have same length as actual
        predictions_aligned = {}
        for model_name, pred in predictions.items():
            if len(pred) == len(actual):
                predictions_aligned[model_name] = pred

        if predictions_aligned:
            perf_fig, metrics = visualizer.plot_prediction_performance(
                predictions_aligned, actual,
                output_path=output_path / 'prediction_performance.png'
            )
        else:
            print("Warning: No predictions with matching length")
            metrics = {}
        
        if metrics:
            print("Prediction Performance Metrics:")
            for model_name, model_metrics in metrics.items():
                print(f"  {model_name}:")
                for metric_name, value in model_metrics.items():
                    print(f"    {metric_name}: {value:.4f}")
        print()
    
    # Step 10: Evaluate models
    print("STEP 10: Evaluating Models")
    print("-" * 60)
    
    metrics = pipeline.evaluate(test_df)
    print("Models evaluated\n")
    
    # Step 11: Visualize consistency scores
    print("STEP 11: Visualizing Consistency Scores")
    print("-" * 60)
    
    if metrics['trajectory_verification']:
        consistency_fig = visualizer.plot_consistency_scores(
            metrics['trajectory_verification'],
            output_path=output_path / 'consistency_scores.png'
        )
        
        avg_consistency = np.mean(list(metrics['trajectory_verification'].values()))
        print(f"Average consistency score: {avg_consistency:.4f}\n")
    
    # Step 12: Save models
    print("STEP 12: Saving Models")
    print("-" * 60)
    
    pipeline.save_models()
    print("Models saved\n")
    
    # Step 13: Generate summary report
    print("STEP 13: Generating Summary Report")
    print("-" * 60)
    
    summary_report = f"""
TRAINING SUMMARY REPORT
{'=' * 60}

Data Statistics:
  Total Records: {len(df)}
  Total Vessels: {df['MMSI'].nunique()}
  Date Range: {df['BaseDateTime'].min()} to {df['BaseDateTime'].max()}

Data Split:
  Training Records: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)
  Validation Records: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)
  Test Records: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)

Models Trained:
  Prediction Models: {list(pipeline.prediction_models.keys())}
  Anomaly Detectors: {list(pipeline.anomaly_detectors.keys())}

Prediction Performance:
"""
    
    if predictions and metrics:
        for model_name, model_metrics in metrics.get('prediction', {}).items():
            summary_report += f"  {model_name}: {model_metrics}\n"
    
    summary_report += f"""
Trajectory Verification:
  Average Consistency Score: {np.mean(list(metrics['trajectory_verification'].values())):.4f}
  Vessels Analyzed: {len(metrics['trajectory_verification'])}

Output Files:
  - data_split.png: Data split visualization
  - prediction_performance.png: Prediction metrics
  - consistency_scores.png: Consistency scores
  - Models saved in: models/

{'=' * 60}
"""
    
    print(summary_report)
    
    # Save report
    report_path = output_path / 'training_report.txt'
    with open(report_path, 'w') as f:
        f.write(summary_report)
    print(f"Report saved to {report_path}\n")
    
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")
    print("\nGenerated files:")
    for file in sorted(output_path.glob('*')):
        print(f"  - {file.name}")
    
    return pipeline, visualizer, metrics


if __name__ == '__main__':
    # Run training with visualization
    pipeline, visualizer, metrics = run_training_with_visualization(
        output_dir='training_results'
    )
    
    # Display plots
    plt.show()

