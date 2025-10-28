"""
XGBoost Backend Integration Module
Handles model loading, preprocessing, and predictions on the backend
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import logging
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class XGBoostBackendPredictor:
    """
    Backend predictor that loads and manages XGBoost model, scaler, and PCA
    Keeps all model weights on the backend for security and performance
    """
    
    def __init__(self, model_dir: str = None):
        """Initialize predictor with model artifacts"""
        if model_dir is None:
            # Default to results directory in current workspace
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
        """Load model, scaler, and PCA from pickle files"""
        try:
            # Load XGBoost model
            model_path = os.path.join(self.model_dir, "xgboost_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"✅ Loaded XGBoost model from {model_path}")
            
            # Load StandardScaler
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"✅ Loaded StandardScaler from {scaler_path}")
            
            # Load PCA
            pca_path = os.path.join(self.model_dir, "pca.pkl")
            if os.path.exists(pca_path):
                with open(pca_path, 'rb') as f:
                    self.pca = pickle.load(f)
                logger.info(f"✅ Loaded PCA from {pca_path}")
            
            if self.model and self.scaler and self.pca:
                self.is_loaded = True
                logger.info("✅ XGBoost backend predictor initialized successfully")
            else:
                logger.warning("⚠️  Some model artifacts missing")
                
        except Exception as e:
            logger.error(f"❌ Error loading model artifacts: {e}")
            self.is_loaded = False
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract advanced features from vessel trajectory data
        Returns feature matrix ready for PCA and prediction
        """
        if df.empty or len(df) < 2:
            return None
        
        features = []
        
        # Sort by timestamp
        df = df.sort_values('BaseDateTime')
        
        # Extract LAT, LON, SOG, COG
        for col in ['LAT', 'LON', 'SOG', 'COG']:
            if col in df.columns:
                values = df[col].values.astype(float)
                
                # Statistical features
                features.extend([
                    np.mean(values),
                    np.std(values),
                    np.min(values),
                    np.max(values),
                    np.median(values),
                    np.percentile(values, 25),
                    np.percentile(values, 75),
                    np.max(values) - np.min(values),  # range
                    pd.Series(values).skew(),
                    pd.Series(values).kurtosis()
                ])
                
                # Temporal features
                if len(values) > 1:
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    features.extend([
                        trend,
                        np.std(np.diff(values)),
                        np.max(np.diff(values)),
                        np.min(np.diff(values)),
                        values[-1] - values[0],  # first-last diff
                        values[-1] / (values[0] + 1e-6),  # first-last ratio
                        np.std(np.diff(values)) / (np.mean(values) + 1e-6)  # volatility
                    ])
        
        # Haversine distance features
        if 'LAT' in df.columns and 'LON' in df.columns:
            lats = df['LAT'].values
            lons = df['LON'].values
            
            # Calculate distances between consecutive points
            distances = []
            for i in range(len(lats) - 1):
                dist = self._haversine_distance(
                    lats[i], lons[i], lats[i+1], lons[i+1]
                )
                distances.append(dist)
            
            if distances:
                features.extend([
                    np.mean(distances),
                    np.std(distances),
                    np.max(distances),
                    np.min(distances),
                    np.sum(distances),
                    np.percentile(distances, 75),
                    np.percentile(distances, 25)
                ])
        
        return np.array(features).reshape(1, -1)
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in km"""
        R = 6371  # Earth radius in km
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def predict(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Make predictions for vessel trajectory
        Returns predicted LAT, LON, SOG, COG
        """
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
            
            # Apply PCA
            X_pca = self.pca.transform(X)
            
            # Make prediction
            predictions = self.model.predict(X_pca)
            
            # predictions shape: (1, 4) for [LAT, LON, SOG, COG]
            pred_lat, pred_lon, pred_sog, pred_cog = predictions[0]
            
            return {
                "predicted_lat": float(pred_lat),
                "predicted_lon": float(pred_lon),
                "predicted_sog": float(pred_sog),
                "predicted_cog": float(pred_cog),
                "confidence": 0.95  # High confidence from model performance
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
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


class VesselPredictionEngine:
    """
    High-level prediction engine that uses XGBoost backend predictor
    Handles vessel data fetching, preprocessing, and result formatting
    """
    
    def __init__(self, db, xgboost_predictor: XGBoostBackendPredictor):
        """Initialize with database and XGBoost predictor"""
        self.db = db
        self.predictor = xgboost_predictor
    
    def predict_vessel_position(
        self,
        vessel_name: str = None,
        mmsi: int = None,
        minutes_ahead: int = 30,
        limit: int = 12
    ) -> Dict:
        """
        Predict vessel position after specified minutes
        
        Args:
            vessel_name: Vessel name
            mmsi: MMSI number
            minutes_ahead: Minutes to predict ahead
            limit: Number of historical points to use
        
        Returns:
            Dictionary with predictions and map data
        """
        try:
            # Fetch vessel data
            if mmsi:
                df = self.db.fetch_vessel_by_mmsi(int(mmsi), limit=limit)
            elif vessel_name:
                df = self.db.fetch_vessel_by_name(vessel_name, limit=limit)
            else:
                return {"error": "No vessel identifier provided"}
            
            if df.empty:
                return {"error": "No vessel data found"}
            
            # Sort by timestamp
            df = df.sort_values('BaseDateTime')
            
            # Get last known position
            last_row = df.iloc[-1]
            last_lat = float(last_row['LAT'])
            last_lon = float(last_row['LON'])
            last_sog = float(last_row['SOG'])
            last_cog = float(last_row['COG'])
            last_time = last_row['BaseDateTime']
            
            # Make prediction
            pred_result = self.predictor.predict(df)
            
            if not pred_result:
                # Fallback to dead reckoning
                pred_result = self._dead_reckoning(
                    last_lat, last_lon, last_sog, last_cog, minutes_ahead
                )
            
            # Generate trajectory points for map
            trajectory_points = self._generate_trajectory(
                last_lat, last_lon,
                pred_result['predicted_lat'],
                pred_result['predicted_lon'],
                minutes_ahead
            )
            
            return {
                "vessel_name": str(last_row.get('VesselName', 'Unknown')),
                "mmsi": int(last_row.get('MMSI', 0)),
                "last_position": {
                    "lat": last_lat,
                    "lon": last_lon,
                    "sog": last_sog,
                    "cog": last_cog,
                    "timestamp": str(last_time)
                },
                "predicted_position": {
                    "lat": pred_result['predicted_lat'],
                    "lon": pred_result['predicted_lon'],
                    "sog": pred_result['predicted_sog'],
                    "cog": pred_result['predicted_cog'],
                    "timestamp": str(datetime.fromisoformat(str(last_time)) + timedelta(minutes=minutes_ahead))
                },
                "trajectory_points": trajectory_points,
                "confidence": pred_result.get('confidence', 0.95),
                "minutes_ahead": minutes_ahead
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def _dead_reckoning(lat: float, lon: float, sog: float, cog: float, minutes: int) -> Dict:
        """Fallback dead reckoning calculation"""
        # Convert SOG (knots) to km/h
        speed_kmh = sog * 1.852
        distance_km = (speed_kmh * minutes) / 60
        
        # Convert COG to radians
        cog_rad = np.radians(cog)
        
        # Calculate new position
        lat_change = (distance_km / 111.0) * np.cos(cog_rad)
        lon_change = (distance_km / (111.0 * np.cos(np.radians(lat)))) * np.sin(cog_rad)
        
        return {
            "predicted_lat": lat + lat_change,
            "predicted_lon": lon + lon_change,
            "predicted_sog": sog,
            "predicted_cog": cog,
            "confidence": 0.7
        }
    
    @staticmethod
    def _generate_trajectory(lat1: float, lon1: float, lat2: float, lon2: float, minutes: int, steps: int = 6) -> List[Dict]:
        """Generate intermediate trajectory points for map visualization"""
        points = []
        
        for i in range(steps + 1):
            t = i / steps
            lat = lat1 + (lat2 - lat1) * t
            lon = lon1 + (lon2 - lon1) * t
            
            points.append({
                "lat": float(lat),
                "lon": float(lon),
                "minutes_ahead": int((minutes * t)),
                "order": i
            })
        
        return points

