"""
XGBoost Integration Module for Maritime Vessel Trajectory Prediction
Loads pre-trained XGBoost model and preprocessing objects for real-time predictions
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """
    Loads and manages XGBoost model with preprocessing pipeline
    Handles feature extraction, scaling, PCA transformation, and predictions
    """
    
    def __init__(self, model_dir: str = "results/xgboost_advanced_50_vessels"):
        """
        Initialize XGBoost predictor with pre-trained model and preprocessing objects
        
        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.pca = None
        self.is_loaded = False
        
        self._load_model_artifacts()
    
    def _load_model_artifacts(self):
        """Load model, scaler, and PCA transformer from disk"""
        try:
            # Load model
            model_path = self.model_dir / "xgboost_model.pkl"
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✅ Loaded XGBoost model from {model_path}")
            
            # Load scaler
            scaler_path = self.model_dir / "scaler.pkl"
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"✅ Loaded StandardScaler from {scaler_path}")
            
            # Load PCA
            pca_path = self.model_dir / "pca.pkl"
            with open(pca_path, 'rb') as f:
                self.pca = pickle.load(f)
            logger.info(f"✅ Loaded PCA transformer from {pca_path}")
            
            self.is_loaded = True
            logger.info("✅ All model artifacts loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading model artifacts: {e}")
            raise
    
    def extract_advanced_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract 483 advanced time-series features from sequences
        
        Args:
            X: Input array of shape (n_samples, n_timesteps, n_features)
        
        Returns:
            Features array of shape (n_samples, 483)
        """
        n_samples, n_timesteps, n_features = X.shape
        features_list = []
        
        for dim in range(n_features):
            X_dim = X[:, :, dim]
            
            # Statistical features (14 per dimension)
            features_dict = {
                'mean': np.mean(X_dim, axis=1),
                'std': np.std(X_dim, axis=1),
                'min': np.min(X_dim, axis=1),
                'max': np.max(X_dim, axis=1),
                'median': np.median(X_dim, axis=1),
                'p25': np.percentile(X_dim, 25, axis=1),
                'p75': np.percentile(X_dim, 75, axis=1),
                'range': np.max(X_dim, axis=1) - np.min(X_dim, axis=1),
                'skew': np.array([pd.Series(row).skew() for row in X_dim]),
                'kurtosis': np.array([pd.Series(row).kurtosis() for row in X_dim]),
            }
            
            # Trend features
            diff = np.diff(X_dim, axis=1)
            features_dict['trend_mean'] = np.mean(diff, axis=1)
            features_dict['trend_std'] = np.std(diff, axis=1)
            features_dict['trend_max'] = np.max(diff, axis=1)
            features_dict['trend_min'] = np.min(diff, axis=1)
            
            # Autocorrelation-like features
            features_dict['first_last_diff'] = X_dim[:, -1] - X_dim[:, 0]
            features_dict['first_last_ratio'] = np.divide(X_dim[:, -1], X_dim[:, 0] + 1e-6)
            
            # Volatility
            features_dict['volatility'] = np.std(diff, axis=1)
            
            dim_features = np.column_stack(list(features_dict.values()))
            features_list.append(dim_features)
        
        X_features = np.hstack(features_list)
        return X_features
    
    def add_haversine_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Add 7 Haversine distance features for spatial nonlinearity
        
        Args:
            X: Input sequences (n_samples, n_timesteps, n_features)
            y: Target values (n_samples, 4) - LAT, LON, SOG, COG
        
        Returns:
            Haversine features array of shape (n_samples, 7)
        """
        n_samples = X.shape[0]
        haversine_features = np.zeros((n_samples, 7))
        
        # Extract LAT/LON from sequences (assuming indices 0, 1)
        lats = X[:, :, 0]  # (n_samples, n_timesteps)
        lons = X[:, :, 1]
        
        for i in range(n_samples):
            lat_seq = lats[i]
            lon_seq = lons[i]
            
            # Distance to first point
            distances = self._haversine_distance(lat_seq[0], lon_seq[0], lat_seq, lon_seq)
            haversine_features[i, 0] = np.mean(distances)  # mean distance
            haversine_features[i, 1] = np.max(distances)   # max distance
            haversine_features[i, 2] = np.std(distances)   # std distance
            
            # Total distance traveled
            consecutive_dists = self._haversine_distance(lat_seq[:-1], lon_seq[:-1], 
                                                         lat_seq[1:], lon_seq[1:])
            haversine_features[i, 3] = np.sum(consecutive_dists)  # total distance
            haversine_features[i, 4] = np.mean(consecutive_dists)  # avg distance per step
            haversine_features[i, 5] = np.max(consecutive_dists)   # max consecutive distance
            haversine_features[i, 6] = np.std(consecutive_dists)   # std of consecutive distances
        
        return haversine_features
    
    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate Haversine distance in km"""
        R = 6371
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def preprocess_and_predict(self, X: np.ndarray, y_dummy: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Full preprocessing pipeline and prediction
        
        Args:
            X: Input sequences (n_samples, n_timesteps, n_features)
            y_dummy: Dummy targets for Haversine calculation (can be zeros)
        
        Returns:
            Predictions array of shape (n_samples, 4) - LAT, LON, SOG, COG
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call _load_model_artifacts() first.")
        
        # Extract features
        X_features = self.extract_advanced_features(X)
        
        # Add Haversine features
        if y_dummy is None:
            y_dummy = np.zeros((X.shape[0], 4))
        X_haversine = self.add_haversine_features(X, y_dummy)
        
        # Combine features
        X_combined = np.hstack([X_features, X_haversine])
        
        # Scale
        X_scaled = self.scaler.transform(X_combined)
        
        # PCA
        X_pca = self.pca.transform(X_scaled)
        
        # Predict
        predictions = self.model.predict(X_pca)
        
        return predictions
    
    def predict_single_vessel(self, vessel_df: pd.DataFrame, 
                             sequence_length: int = 12) -> Dict:
        """
        Predict next position for a single vessel
        
        Args:
            vessel_df: DataFrame with vessel data (LAT, LON, SOG, COG, etc.)
            sequence_length: Number of timesteps to use
        
        Returns:
            Dictionary with predictions and metadata
        """
        if len(vessel_df) < sequence_length:
            return {"error": f"Insufficient data. Need {sequence_length} points, got {len(vessel_df)}"}
        
        # Prepare sequence (last sequence_length rows)
        last_seq = vessel_df.tail(sequence_length)
        
        # Extract features (assuming columns: LAT, LON, SOG, COG, ... 28 total)
        feature_cols = [col for col in vessel_df.columns if col not in ['VesselName', 'MMSI', 'BaseDateTime', 'CallSign']]
        X_seq = last_seq[feature_cols].values.reshape(1, sequence_length, -1)
        
        # Predict
        predictions = self.preprocess_and_predict(X_seq)
        pred = predictions[0]
        
        return {
            "predicted_lat": float(pred[0]),
            "predicted_lon": float(pred[1]),
            "predicted_sog": float(pred[2]),
            "predicted_cog": float(pred[3]),
            "last_known_lat": float(last_seq.iloc[-1]['LAT']),
            "last_known_lon": float(last_seq.iloc[-1]['LON']),
            "last_known_sog": float(last_seq.iloc[-1]['SOG']),
            "last_known_cog": float(last_seq.iloc[-1]['COG']),
            "last_timestamp": str(last_seq.iloc[-1]['BaseDateTime']),
            "vessel_name": vessel_df.iloc[-1].get('VesselName', 'Unknown'),
            "mmsi": int(vessel_df.iloc[-1].get('MMSI', 0)) if 'MMSI' in vessel_df.columns else None
        }

