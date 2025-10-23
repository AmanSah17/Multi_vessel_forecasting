"""
Training Pipeline Module

Orchestrates the complete training workflow:
- Data loading and preprocessing
- Feature engineering
- Model training
- Validation and evaluation
- Model persistence
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging
import pickle
from pathlib import Path

from .data_preprocessing import VesselDataPreprocessor
from .trajectory_prediction import (
    KalmanFilterPredictor, ARIMAPredictor, LSTMPredictor, EnsemblePredictor
)
from .anomaly_detection import (
    IsolationForestDetector, RuleBasedDetector, EnsembleAnomalyDetector,
    create_default_rule_detector
)
from .trajectory_verification import TrajectoryVerifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates the complete ML training pipeline."""
    
    def __init__(self, output_dir: str = 'models'):
        """
        Initialize training pipeline.
        
        Args:
            output_dir: Directory to save trained models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.preprocessor = VesselDataPreprocessor()
        self.verifier = TrajectoryVerifier()
        
        self.prediction_models = {}
        self.anomaly_detectors = {}
        self.metrics = {}
    
    def load_data(self, data_input) -> pd.DataFrame:
        """
        Load and preprocess raw data.

        Args:
            data_input: Either a file path (str) or a DataFrame

        Returns:
            Preprocessed dataframe
        """
        # Handle both DataFrame and file path inputs
        if isinstance(data_input, pd.DataFrame):
            logger.info(f"Using provided DataFrame with {len(data_input)} records")
            df = data_input.copy()
        else:
            logger.info(f"Loading data from {data_input}")
            df = pd.read_csv(data_input)

        df = self.preprocessor.preprocess(df)
        logger.info(f"Loaded {len(df)} records for {df['MMSI'].nunique()} vessels")
        return df
    
    def create_train_val_test_split(self, df: pd.DataFrame,
                                   train_ratio: float = 0.6,
                                   val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/val/test split.
        
        Args:
            df: Full dataset
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            
        Returns:
            (train_df, val_df, test_df)
        """
        df = df.sort_values('BaseDateTime')
        n = len(df)
        
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_idx]
        val_df = df.iloc[train_idx:val_idx]
        test_df = df.iloc[val_idx:]
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for modeling.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            Dataframe with engineered features
        """
        df = df.copy()
        
        # Temporal features
        df['hour'] = df['BaseDateTime'].dt.hour
        df['day_of_week'] = df['BaseDateTime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Kinematic features (per vessel)
        for mmsi in df['MMSI'].unique():
            mask = df['MMSI'] == mmsi
            vessel_df = df[mask].sort_values('BaseDateTime')
            
            # Speed changes
            if 'SOG' in df.columns:
                df.loc[mask, 'speed_change'] = vessel_df['SOG'].diff().fillna(0)
            
            # Heading changes
            if 'COG' in df.columns:
                cog_diff = vessel_df['COG'].diff().fillna(0)
                cog_diff = np.minimum(np.abs(cog_diff), 360 - np.abs(cog_diff))
                df.loc[mask, 'heading_change'] = cog_diff
        
        # Fill NaN values
        df = df.fillna(0)
        
        logger.info(f"Engineered features: {df.columns.tolist()}")
        return df
    
    def train_prediction_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Train trajectory prediction models.
        
        Args:
            train_df: Training data
            val_df: Validation data
        """
        logger.info("Training prediction models...")
        
        # Prepare data
        X_train = train_df[['LAT', 'LON', 'SOG', 'COG']].values
        X_val = val_df[['LAT', 'LON', 'SOG', 'COG']].values
        
        # Train individual models
        kf = KalmanFilterPredictor()
        kf.fit(X_train[:-1], X_train[1:])
        self.prediction_models['kalman'] = kf
        
        arima = ARIMAPredictor()
        arima.fit(X_train[:, 0])  # Fit on latitude
        self.prediction_models['arima'] = arima
        
        # Create ensemble
        ensemble = EnsemblePredictor({
            'kalman': kf,
            'arima': arima,
        })
        self.prediction_models['ensemble'] = ensemble
        
        logger.info("Prediction models trained")
    
    def train_anomaly_detectors(self, train_df: pd.DataFrame):
        """
        Train anomaly detection models.
        
        Args:
            train_df: Training data (assumed to be normal)
        """
        logger.info("Training anomaly detectors...")
        
        # Prepare features
        features = ['LAT', 'LON', 'SOG', 'COG', 'speed_change', 'heading_change']
        X_train = train_df[features].fillna(0).values
        
        # Train Isolation Forest
        iso_forest = IsolationForestDetector(contamination=0.05)
        iso_forest.fit(X_train)
        self.anomaly_detectors['isolation_forest'] = iso_forest
        
        # Train rule-based detector
        rule_detector = create_default_rule_detector()
        rule_detector.fit(X_train)
        self.anomaly_detectors['rule_based'] = rule_detector
        
        # Create ensemble
        ensemble = EnsembleAnomalyDetector(self.anomaly_detectors)
        self.anomaly_detectors['ensemble'] = ensemble
        
        logger.info("Anomaly detectors trained")
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """
        Evaluate models on test set.
        
        Args:
            test_df: Test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating models...")
        
        metrics = {
            'prediction': {},
            'anomaly_detection': {},
            'trajectory_verification': {},
        }
        
        # Evaluate trajectory verification
        for mmsi in test_df['MMSI'].unique()[:10]:  # Sample 10 vessels
            vessel_traj = test_df[test_df['MMSI'] == mmsi].sort_values('BaseDateTime')
            if len(vessel_traj) > 3:
                consistency_score = self.verifier.get_consistency_score(vessel_traj)
                metrics['trajectory_verification'][mmsi] = consistency_score
        
        avg_consistency = np.mean(list(metrics['trajectory_verification'].values()))
        logger.info(f"Average trajectory consistency: {avg_consistency:.3f}")
        
        self.metrics = metrics
        return metrics
    
    def save_models(self):
        """Save trained models to disk."""
        logger.info(f"Saving models to {self.output_dir}")
        
        for name, model in self.prediction_models.items():
            path = self.output_dir / f"prediction_{name}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {name} to {path}")
        
        for name, model in self.anomaly_detectors.items():
            path = self.output_dir / f"anomaly_{name}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {name} to {path}")
    
    def load_models(self):
        """Load trained models from disk."""
        logger.info(f"Loading models from {self.output_dir}")
        
        for path in self.output_dir.glob("prediction_*.pkl"):
            with open(path, 'rb') as f:
                model = pickle.load(f)
                name = path.stem.replace("prediction_", "")
                self.prediction_models[name] = model
        
        for path in self.output_dir.glob("anomaly_*.pkl"):
            with open(path, 'rb') as f:
                model = pickle.load(f)
                name = path.stem.replace("anomaly_", "")
                self.anomaly_detectors[name] = model
        
        logger.info(f"Loaded {len(self.prediction_models)} prediction models")
        logger.info(f"Loaded {len(self.anomaly_detectors)} anomaly detectors")
    
    def run_full_pipeline(self, data_input):
        """
        Run complete training pipeline.

        Args:
            data_input: Either a file path (str) or a DataFrame with AIS data
        """
        logger.info("=" * 50)
        logger.info("Starting full training pipeline")
        logger.info("=" * 50)

        # Load and preprocess
        df = self.load_data(data_input)
        
        # Feature engineering
        df = self.engineer_features(df)
        
        # Train/val/test split
        train_df, val_df, test_df = self.create_train_val_test_split(df)
        
        # Train models
        self.train_prediction_models(train_df, val_df)
        self.train_anomaly_detectors(train_df)
        
        # Evaluate
        metrics = self.evaluate(test_df)
        
        # Save models
        self.save_models()
        
        logger.info("=" * 50)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 50)
        
        return metrics

