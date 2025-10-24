"""
Vessel Prediction Service
Handles PREDICT and VERIFY intents using XGBoost model with Maritime NLU database
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VesselPredictionService:
    """
    Service for vessel trajectory prediction and verification
    Integrates XGBoost model with Maritime NLU database
    """
    
    def __init__(self, db_handler, xgboost_predictor):
        """
        Initialize prediction service
        
        Args:
            db_handler: MaritimeDB instance for data retrieval
            xgboost_predictor: XGBoostPredictor instance for predictions
        """
        self.db = db_handler
        self.predictor = xgboost_predictor
        self.sequence_length = 12  # 12 timesteps = 60 minutes at 5-min intervals
    
    def predict_vessel_position(self, vessel_name: Optional[str] = None, 
                               mmsi: Optional[int] = None,
                               minutes_ahead: int = 30) -> Dict:
        """
        Predict vessel position after X minutes using XGBoost model
        
        Args:
            vessel_name: Name of vessel
            mmsi: MMSI number of vessel
            minutes_ahead: Minutes to predict ahead (default 30)
        
        Returns:
            Dictionary with predictions and metadata
        """
        try:
            # Fetch vessel data
            if mmsi:
                vessel_df = self.db.fetch_vessel_by_mmsi(int(mmsi), limit=self.sequence_length + 10)
            elif vessel_name:
                vessel_df = self.db.fetch_vessel_by_name(vessel_name, limit=self.sequence_length + 10)
            else:
                return {"error": "Must provide vessel_name or mmsi"}
            
            if vessel_df.empty:
                return {"error": f"No data found for vessel"}
            
            # Get XGBoost prediction
            xgb_pred = self.predictor.predict_single_vessel(vessel_df, self.sequence_length)
            
            if "error" in xgb_pred:
                return xgb_pred
            
            # Get last 5 points for verification
            last_5 = vessel_df.tail(5).to_dict(orient='records')
            
            # Calculate confidence based on recent movement consistency
            confidence = self._calculate_confidence(vessel_df.tail(10))
            
            return {
                "status": "success",
                "vessel_name": xgb_pred["vessel_name"],
                "mmsi": xgb_pred["mmsi"],
                "prediction_method": "XGBoost Advanced",
                "minutes_ahead": minutes_ahead,
                "predicted_position": {
                    "latitude": xgb_pred["predicted_lat"],
                    "longitude": xgb_pred["predicted_lon"],
                    "sog": xgb_pred["predicted_sog"],
                    "cog": xgb_pred["predicted_cog"]
                },
                "last_known_position": {
                    "latitude": xgb_pred["last_known_lat"],
                    "longitude": xgb_pred["last_known_lon"],
                    "sog": xgb_pred["last_known_sog"],
                    "cog": xgb_pred["last_known_cog"],
                    "timestamp": xgb_pred["last_timestamp"]
                },
                "last_5_points": last_5,
                "confidence_score": confidence,
                "distance_traveled_nm": self._calculate_distance(
                    xgb_pred["last_known_lat"], xgb_pred["last_known_lon"],
                    xgb_pred["predicted_lat"], xgb_pred["predicted_lon"]
                )
            }
        
        except Exception as e:
            logger.error(f"Error predicting vessel position: {e}")
            return {"error": str(e), "status": "failed"}
    
    def verify_vessel_course(self, vessel_name: Optional[str] = None,
                            mmsi: Optional[int] = None) -> Dict:
        """
        Verify vessel course consistency using last 5 points
        Plot course and estimated 30-minute trajectory
        
        Args:
            vessel_name: Name of vessel
            mmsi: MMSI number of vessel
        
        Returns:
            Dictionary with verification results
        """
        try:
            # Fetch vessel data
            if mmsi:
                vessel_df = self.db.fetch_vessel_by_mmsi(int(mmsi), limit=self.sequence_length + 10)
            elif vessel_name:
                vessel_df = self.db.fetch_vessel_by_name(vessel_name, limit=self.sequence_length + 10)
            else:
                return {"error": "Must provide vessel_name or mmsi"}
            
            if vessel_df.empty:
                return {"error": f"No data found for vessel"}
            
            # Get last 5 points
            last_5 = vessel_df.tail(5)
            
            # Calculate course consistency
            course_consistency = self._check_course_consistency(last_5)
            
            # Get 30-minute prediction
            pred_result = self.predict_vessel_position(vessel_name, mmsi, minutes_ahead=30)
            
            if "error" in pred_result:
                return pred_result
            
            # Build verification report
            return {
                "status": "success",
                "vessel_name": last_5.iloc[-1].get('VesselName', 'Unknown'),
                "mmsi": int(last_5.iloc[-1].get('MMSI', 0)) if 'MMSI' in last_5.columns else None,
                "verification": {
                    "course_consistency": course_consistency["consistency"],
                    "course_change_rate": course_consistency["change_rate"],
                    "speed_consistency": course_consistency["speed_consistency"],
                    "anomaly_detected": course_consistency["anomaly"],
                    "anomaly_reason": course_consistency.get("reason", "")
                },
                "last_5_points": last_5.to_dict(orient='records'),
                "predicted_30min": pred_result["predicted_position"],
                "trajectory_points": self._generate_trajectory_points(
                    last_5.iloc[-1], pred_result["predicted_position"], steps=6
                ),
                "confidence_score": pred_result.get("confidence_score", 0.0)
            }
        
        except Exception as e:
            logger.error(f"Error verifying vessel course: {e}")
            return {"error": str(e), "status": "failed"}
    
    def _check_course_consistency(self, last_5_df: pd.DataFrame) -> Dict:
        """Check if vessel course is consistent"""
        if len(last_5_df) < 2:
            return {"consistency": "insufficient_data", "change_rate": 0, "speed_consistency": 0, "anomaly": False}
        
        # Calculate course changes
        cogs = last_5_df['COG'].values
        sogs = last_5_df['SOG'].values
        
        # Course change rate
        cog_diffs = np.abs(np.diff(cogs))
        cog_diffs = np.where(cog_diffs > 180, 360 - cog_diffs, cog_diffs)  # Handle wrap-around
        avg_cog_change = np.mean(cog_diffs)
        
        # Speed consistency
        speed_std = np.std(sogs)
        speed_mean = np.mean(sogs)
        speed_cv = speed_std / (speed_mean + 1e-6)  # Coefficient of variation
        
        # Detect anomalies
        anomaly = False
        reason = ""
        
        if avg_cog_change > 45:  # Large course changes
            anomaly = True
            reason = f"Large course changes detected ({avg_cog_change:.1f}°)"
        
        if speed_cv > 0.5:  # High speed variation
            anomaly = True
            reason = f"High speed variation detected (CV={speed_cv:.2f})"
        
        consistency = "stable" if not anomaly else "unstable"
        
        return {
            "consistency": consistency,
            "change_rate": float(avg_cog_change),
            "speed_consistency": float(speed_cv),
            "anomaly": anomaly,
            "reason": reason
        }
    
    def _calculate_confidence(self, recent_df: pd.DataFrame) -> float:
        """Calculate prediction confidence based on data quality"""
        if len(recent_df) < 5:
            return 0.5
        
        # Check for missing values
        missing_ratio = recent_df[['LAT', 'LON', 'SOG', 'COG']].isnull().sum().sum() / (len(recent_df) * 4)
        
        # Check for outliers in SOG
        sog_mean = recent_df['SOG'].mean()
        sog_std = recent_df['SOG'].std()
        outliers = ((recent_df['SOG'] - sog_mean).abs() > 3 * sog_std).sum()
        outlier_ratio = outliers / len(recent_df)
        
        # Confidence = 1 - (missing_ratio + outlier_ratio)
        confidence = max(0.0, min(1.0, 1.0 - missing_ratio - outlier_ratio))
        
        return float(confidence)
    
    @staticmethod
    def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Haversine distance in nautical miles"""
        R = 3440.065  # Earth radius in nautical miles
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return float(R * c)
    
    @staticmethod
    def _generate_trajectory_points(last_point: pd.Series, predicted_point: Dict, steps: int = 6) -> List[Dict]:
        """Generate intermediate trajectory points for visualization"""
        trajectory = []
        
        lat1 = float(last_point['LAT'])
        lon1 = float(last_point['LON'])
        lat2 = predicted_point['latitude']
        lon2 = predicted_point['longitude']
        
        for i in range(steps + 1):
            t = i / steps
            lat = lat1 + (lat2 - lat1) * t
            lon = lon1 + (lon2 - lon1) * t
            
            trajectory.append({
                "step": i,
                "latitude": float(lat),
                "longitude": float(lon),
                "time_minutes": int(30 * t)
            })
        
        return trajectory

