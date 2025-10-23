"""
Example: End-to-End ML Pipeline for Maritime Vessel Forecasting

This notebook demonstrates:
1. Data loading and preprocessing
2. MMSI analysis and visualization
3. Feature engineering
4. Model training
5. Evaluation and anomaly detection
"""

import sys
sys.path.insert(0, '../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data_preprocessing import VesselDataPreprocessor, load_and_preprocess
from mmsi_analysis import MMSIAnalyzer
from trajectory_verification import TrajectoryVerifier
from training_pipeline import TrainingPipeline


def main():
    """Run complete pipeline example."""
    
    print("=" * 60)
    print("Maritime Vessel Forecasting - End-to-End Pipeline")
    print("=" * 60)
    
    # ============================================================
    # 1. DATA LOADING AND PREPROCESSING
    # ============================================================
    print("\n[1] Loading and Preprocessing Data...")
    
    # Example: Load your AIS data
    # df = load_and_preprocess('path/to/ais_data.csv')
    
    # For demonstration, create sample data
    df = create_sample_data()
    print(f"Loaded {len(df)} records for {df['MMSI'].nunique()} vessels")
    print(f"Date range: {df['BaseDateTime'].min()} to {df['BaseDateTime'].max()}")
    
    # ============================================================
    # 2. MMSI ANALYSIS
    # ============================================================
    print("\n[2] Analyzing MMSI Distribution...")
    
    analyzer = MMSIAnalyzer()
    mmsi_results = analyzer.analyze(df)
    
    print(f"Total unique MMSI: {mmsi_results['total_mmsi']}")
    print(f"Mean records per MMSI: {mmsi_results['mmsi_distribution']['mean_records_per_mmsi']:.1f}")
    print(f"Formatting issues: {mmsi_results['formatting_issues']}")
    print(f"Suspicious patterns: {mmsi_results['suspicious_patterns']}")
    
    # Visualize
    fig = analyzer.visualize_distribution(df, top_n=10)
    plt.savefig('mmsi_distribution.png', dpi=100, bbox_inches='tight')
    print("Saved MMSI distribution plot to mmsi_distribution.png")
    
    # ============================================================
    # 3. TRAJECTORY VERIFICATION
    # ============================================================
    print("\n[3] Verifying Trajectory Consistency...")
    
    verifier = TrajectoryVerifier()
    
    # Check a sample vessel
    sample_mmsi = df['MMSI'].iloc[0]
    sample_trajectory = df[df['MMSI'] == sample_mmsi].sort_values('BaseDateTime')
    
    if len(sample_trajectory) > 3:
        results = verifier.verify_trajectory(sample_trajectory)
        consistency_score = verifier.get_consistency_score(sample_trajectory)
        
        print(f"\nVessel {sample_mmsi}:")
        print(f"  Smoothness Score: {results['smoothness_score']:.3f}")
        print(f"  Consistency Score: {consistency_score:.3f}")
        print(f"  Speed Violations: {results['speed_consistency']['exceeds_max_speed']}")
        print(f"  Turn Rate Violations: {results['turn_rate_check'].get('unrealistic_turns', 0)}")
        
        anomalies = results['anomalies']
        if anomalies:
            print(f"  Detected Anomalies: {len(anomalies)}")
            for anomaly in anomalies:
                print(f"    - {anomaly['type']}: {anomaly['description']}")
    
    # ============================================================
    # 4. TRAINING PIPELINE
    # ============================================================
    print("\n[4] Running Training Pipeline...")
    
    pipeline = TrainingPipeline(output_dir='models')
    
    # Feature engineering
    df = pipeline.engineer_features(df)
    
    # Train/val/test split
    train_df, val_df, test_df = pipeline.create_train_val_test_split(df)
    
    # Train models
    pipeline.train_prediction_models(train_df, val_df)
    pipeline.train_anomaly_detectors(train_df)
    
    # Evaluate
    metrics = pipeline.evaluate(test_df)
    
    print(f"Average Trajectory Consistency: {np.mean(list(metrics['trajectory_verification'].values())):.3f}")
    
    # Save models
    pipeline.save_models()
    print("Models saved to 'models' directory")
    
    # ============================================================
    # 5. INFERENCE EXAMPLE
    # ============================================================
    print("\n[5] Running Inference on Test Data...")
    
    # Get a test vessel
    test_vessel = test_df[test_df['MMSI'] == test_df['MMSI'].iloc[0]].sort_values('BaseDateTime')
    
    if len(test_vessel) > 10:
        # Prepare features
        features = ['LAT', 'LON', 'SOG', 'COG', 'speed_change', 'heading_change']
        X_test = test_vessel[features].fillna(0).values
        
        # Predict with ensemble
        if 'ensemble' in pipeline.prediction_models:
            predictions = pipeline.prediction_models['ensemble'].predict(X_test[-1:])
            print(f"Prediction results: {predictions}")
        
        # Detect anomalies
        if 'ensemble' in pipeline.anomaly_detectors:
            anomaly_scores = pipeline.anomaly_detectors['ensemble'].get_anomaly_scores(X_test)
            anomaly_count = (anomaly_scores > 0.5).sum()
            print(f"Anomalies detected: {anomaly_count}/{len(X_test)}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


def create_sample_data(n_records: int = 1000, n_vessels: int = 5) -> pd.DataFrame:
    """Create sample AIS data for demonstration."""
    
    np.random.seed(42)
    
    data = []
    base_time = pd.Timestamp('2024-01-01')
    
    for vessel_id in range(n_vessels):
        mmsi = str(200000000 + vessel_id)
        
        # Generate trajectory
        lat = 40.0 + np.random.randn() * 0.1
        lon = -74.0 + np.random.randn() * 0.1
        
        for i in range(n_records // n_vessels):
            # Add random walk
            lat += np.random.randn() * 0.001
            lon += np.random.randn() * 0.001
            
            data.append({
                'MMSI': mmsi,
                'BaseDateTime': base_time + pd.Timedelta(minutes=i),
                'LAT': lat,
                'LON': lon,
                'SOG': max(0, 15 + np.random.randn() * 3),
                'COG': np.random.uniform(0, 360),
                'Heading': np.random.uniform(0, 360),
                'VesselName': f'Vessel_{vessel_id}' if np.random.rand() > 0.1 else None,
                'IMO': str(1000000 + vessel_id),
                'CallSign': f'CALL{vessel_id}',
                'VesselType': 70,  # Cargo
                'Status': 0,  # Under way
            })
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    main()

