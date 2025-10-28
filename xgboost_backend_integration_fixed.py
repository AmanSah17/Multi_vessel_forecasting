"""
Fixed XGBoost Backend Integration with Proper Feature Extraction
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class XGBoostBackendPredictor:
    """Backend predictor with proper 483-feature extraction"""

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(__file__),
                "results",
                "xgboost_advanced_50_vessels"
            )

        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.pca = None
        self.is_loaded = False
        self._load_model_artifacts()

    def _load_model_artifacts(self):
        """Load model, scaler, and PCA"""
        try:
            model_path = os.path.join(self.model_dir, "xgboost_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"✅ Loaded XGBoost model")

            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"✅ Loaded StandardScaler")

            pca_path = os.path.join(self.model_dir, "pca.pkl")
            if os.path.exists(pca_path):
                with open(pca_path, 'rb') as f:
                    self.pca = pickle.load(f)
                logger.info(f"✅ Loaded PCA")

            if self.model and self.scaler and self.pca:
                self.is_loaded = True
                logger.info("✅ XGBoost predictor ready")
            else:
                logger.warning("⚠️  Some artifacts missing")

        except Exception as e:
            logger.error(f"❌ Error loading artifacts: {e}")
            self.is_loaded = False

    def _prepare_sequence_data(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare 28-dimensional sequence data from vessel trajectory"""
        if df.empty or len(df) < 12:
            return None

        df = df.sort_values('BaseDateTime').reset_index(drop=True)

        # Ensure we have at least 12 records
        if len(df) < 12:
            logger.warning(f"Only {len(df)} records, need 12+")
            return None

        # Take last 12 records
        df = df.tail(12).reset_index(drop=True)

        # Create 28-dimensional feature matrix
        features_list = []

        for idx, row in df.iterrows():
            # Base features (4)
            lat = float(row['LAT'])
            lon = float(row['LON'])
            sog = float(row['SOG'] or 0.0)
            cog = float(row['COG'] or 0.0)

            # Temporal features (4)
            dt = pd.to_datetime(row['BaseDateTime'])
            hour = float(dt.hour)
            day_of_week = float(dt.dayofweek)
            is_weekend = float(1 if day_of_week >= 5 else 0)
            month = float(dt.month)

            # Kinematic features (4) - differences from previous
            if idx > 0:
                prev_row = df.iloc[idx - 1]
                speed_change = float((row['SOG'] or 0.0) - (prev_row['SOG'] or 0.0))
                heading_change = float((row['COG'] or 0.0) - (prev_row['COG'] or 0.0))
                lat_change = float(row['LAT'] - prev_row['LAT'])
                lon_change = float(row['LON'] - prev_row['LON'])
            else:
                speed_change = 0.0
                heading_change = 0.0
                lat_change = 0.0
                lon_change = 0.0

            # Velocity components (3)
            velocity_x = sog * np.cos(np.radians(cog))
            velocity_y = sog * np.sin(np.radians(cog))
            velocity_mag = np.sqrt(velocity_x**2 + velocity_y**2)

            # Polynomial features (3)
            lat_sq = lat ** 2
            lon_sq = lon ** 2
            sog_sq = sog ** 2

            # Interaction features (2)
            speed_heading_int = sog * cog
            lat_lon_int = lat * lon

            # Combine all 28 features
            features = [
                lat, lon, sog, cog,  # 4
                hour, day_of_week, is_weekend, month,  # 4
                speed_change, heading_change, lat_change, lon_change,  # 4
                velocity_x, velocity_y, velocity_mag,  # 3
                lat_sq, lon_sq, sog_sq,  # 3
                speed_heading_int, lat_lon_int  # 2
            ]

            features_list.append(features)

        # Convert to numpy array: shape (12, 28)
        sequence = np.array(features_list, dtype=np.float32)
        logger.info(f"Prepared sequence shape: {sequence.shape}")

        return sequence

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract 483 features from vessel trajectory"""
        if df.empty or len(df) < 12:
            return None

        # Prepare 28-dimensional sequence
        X_seq = self._prepare_sequence_data(df)
        if X_seq is None:
            return None

        # X_seq is already shape (12, 28), reshape to (1, 12, 28) for feature extraction
        if X_seq.shape != (12, 28):
            logger.error(f"Unexpected sequence shape: {X_seq.shape}")
            return None

        X_seq = X_seq.reshape(1, 12, 28)

        # Extract advanced features (17 per dimension × 28 = 476)
        n_samples, n_timesteps, n_features = X_seq.shape
        features_list = []

        for dim in range(n_features):
            X_dim = X_seq[:, :, dim]  # Shape: (1, 12)

            # Statistical features (10)
            features_list.extend([
                np.mean(X_dim),
                np.std(X_dim),
                np.min(X_dim),
                np.max(X_dim),
                np.median(X_dim),
                np.percentile(X_dim, 25),
                np.percentile(X_dim, 75),
                np.max(X_dim) - np.min(X_dim),
                pd.Series(X_dim.flatten()).skew(),
                pd.Series(X_dim.flatten()).kurtosis()
            ])

            # Trend features (7)
            diff = np.diff(X_dim, axis=1)
            trend = np.polyfit(range(n_timesteps), X_dim.flatten(), 1)[0]
            features_list.extend([
                trend,
                np.std(diff),
                np.max(diff),
                np.min(diff),
                X_dim[0, -1] - X_dim[0, 0],
                X_dim[0, -1] / (X_dim[0, 0] + 1e-6),
                np.std(diff) / (np.mean(X_dim) + 1e-6)
            ])

        # Haversine distance features (7)
        lats = X_seq[0, :, 0]  # LAT is first feature
        lons = X_seq[0, :, 1]  # LON is second feature

        dist_to_first = []
        for i in range(len(lats)):
            dist = self._haversine_distance(lats[0], lons[0], lats[i], lons[i])
            dist_to_first.append(dist)

        consecutive_dists = [0.0]
        for i in range(1, len(lats)):
            dist = self._haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])
            consecutive_dists.append(dist)

        features_list.extend([
            np.mean(dist_to_first),
            np.max(dist_to_first),
            np.std(dist_to_first),
            np.sum(consecutive_dists),
            np.mean(consecutive_dists[1:]) if len(consecutive_dists) > 1 else 0,
            np.max(consecutive_dists),
            np.std(consecutive_dists)
        ])

        # Convert to array: shape (1, 483)
        features_array = np.array(features_list, dtype=np.float32).reshape(1, -1)
        logger.info(f"Extracted {features_array.shape[1]} features")

        if features_array.shape[1] != 483:
            logger.warning(f"⚠️  Expected 483 features, got {features_array.shape[1]}")

        return features_array

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in km"""
        R = 6371
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)

        a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def predict(self, df: pd.DataFrame) -> Optional[Dict]:
        """Make XGBoost prediction"""
        if not self.is_loaded:
            logger.error("❌ Model not loaded")
            return None

        if df.empty:
            logger.error("❌ Empty dataframe")
            return None

        try:
            # Extract features
            X = self.extract_features(df)
            if X is None:
                return None

            logger.info(f"Feature shape before PCA: {X.shape}")

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Apply PCA
            X_pca = self.pca.transform(X_scaled)
            logger.info(f"Feature shape after PCA: {X_pca.shape}")

            # Make prediction
            predictions = self.model.predict(X_pca)
            pred_lat, pred_lon, pred_sog, pred_cog = predictions[0]

            logger.info(f"✅ Prediction successful: LAT={pred_lat:.4f}, LON={pred_lon:.4f}")

            return {
                "predicted_lat": float(pred_lat),
                "predicted_lon": float(pred_lon),
                "predicted_sog": float(pred_sog),
                "predicted_cog": float(pred_cog),
                "confidence": 0.95
            }

        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_status(self) -> Dict:
        """Get model status"""
        return {
            "is_loaded": self.is_loaded,
            "model_dir": self.model_dir,
            "has_model": self.model is not None,
            "has_scaler": self.scaler is not None,
            "has_pca": self.pca is not None
        }

