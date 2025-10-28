"""
Fixed Intent Executor with Proper Prediction Integration
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import XGBoost integration
try:
    from xgboost_backend_integration import XGBoostBackendPredictor, VesselPredictionEngine
    XGBOOST_AVAILABLE = True
except Exception as e:
    logger.warning(f"XGBoost not available: {e}")
    XGBOOST_AVAILABLE = False


class IntentExecutorFixed:
    """Fixed intent executor with proper prediction handling"""
    
    def __init__(self, db, time_tolerance_minutes: int = 30):
        self.db = db
        self.time_tolerance_minutes = time_tolerance_minutes
        self.xgboost_enabled = False
        self.predictor = None
        self.prediction_engine = None
        
        # Initialize XGBoost if available
        if XGBOOST_AVAILABLE:
            self._initialize_xgboost()
    
    def _initialize_xgboost(self):
        """Initialize XGBoost model"""
        try:
            logger.info("🚀 Initializing XGBoost Backend...")
            
            # Try multiple paths
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", "results", "xgboost_advanced_50_vessels"),
                os.path.join(os.path.dirname(__file__), "results", "xgboost_advanced_50_vessels"),
                "results/xgboost_advanced_50_vessels",
            ]
            
            model_dir = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_dir = path
                    break
            
            if model_dir:
                self.predictor = XGBoostBackendPredictor(model_dir)
                if self.predictor.is_loaded:
                    self.prediction_engine = VesselPredictionEngine(self.db, self.predictor)
                    self.xgboost_enabled = True
                    logger.info("✅ XGBoost backend integration enabled")
                else:
                    logger.warning("⚠️  XGBoost model failed to load")
            else:
                logger.warning("⚠️  Model directory not found")
        
        except Exception as e:
            logger.warning(f"⚠️  XGBoost initialization error: {e}")
    
    def handle(self, parsed: Dict) -> Dict:
        """Handle parsed intent"""
        intent = parsed.get("intent", "").upper()
        
        if intent == "SHOW":
            return self._handle_show(parsed)
        elif intent == "VERIFY":
            return self._handle_verify(parsed)
        elif intent == "PREDICT":
            return self._handle_predict(parsed)
        else:
            return {"message": f"Unknown intent: {intent}"}
    
    def _handle_show(self, parsed: Dict) -> Dict:
        """Handle SHOW intent"""
        vessel_name = parsed.get("vessel_name")
        mmsi = parsed.get("identifiers", {}).get("mmsi")
        
        if not vessel_name and not mmsi:
            return {"message": "Please specify a vessel name or MMSI"}
        
        try:
            # Fetch from database
            if vessel_name:
                df = self.db.fetch_vessel_by_name(vessel_name, limit=12)
            else:
                df = self.db.fetch_vessel_by_mmsi(mmsi, limit=12)
            
            if df.empty:
                return {"message": f"No data found for vessel"}
            
            # Get latest position
            latest = df.iloc[-1]
            
            # Build response
            response = {
                "VesselName": latest.get("VesselName", "Unknown"),
                "MMSI": latest.get("MMSI"),
                "LAT": float(latest.get("LAT", 0)),
                "LON": float(latest.get("LON", 0)),
                "SOG": float(latest.get("SOG", 0)),
                "COG": float(latest.get("COG", 0)),
                "BaseDateTime": str(latest.get("BaseDateTime", "")),
                "track": df.to_dict("records")
            }
            
            return response
        
        except Exception as e:
            logger.error(f"Error in SHOW: {e}")
            return {"message": f"Error: {str(e)}"}
    
    def _handle_verify(self, parsed: Dict) -> Dict:
        """Handle VERIFY intent"""
        vessel_name = parsed.get("vessel_name")
        mmsi = parsed.get("identifiers", {}).get("mmsi")
        
        if not vessel_name and not mmsi:
            return {"message": "Please specify a vessel name or MMSI"}
        
        try:
            # Fetch recent data
            if vessel_name:
                df = self.db.fetch_vessel_by_name(vessel_name, limit=5)
            else:
                df = self.db.fetch_vessel_by_mmsi(mmsi, limit=5)
            
            if df.empty or len(df) < 2:
                return {"message": "Not enough data to verify course"}
            
            # Calculate variance
            cog_values = pd.to_numeric(df["COG"], errors="coerce").dropna()
            sog_values = pd.to_numeric(df["SOG"], errors="coerce").dropna()
            
            cog_variance = float(cog_values.std()) if len(cog_values) > 1 else 0.0
            sog_variance = float(sog_values.std()) if len(sog_values) > 1 else 0.0
            
            # Determine consistency
            is_consistent = cog_variance < 10 and sog_variance < 5
            
            response = {
                "VesselName": df.iloc[-1].get("VesselName", "Unknown"),
                "is_consistent": is_consistent,
                "cog_variance": cog_variance,
                "sog_variance": sog_variance,
                "last_3_points": df.tail(3).to_dict("records")
            }
            
            return response
        
        except Exception as e:
            logger.error(f"Error in VERIFY: {e}")
            return {"message": f"Error: {str(e)}"}
    
    def _handle_predict(self, parsed: Dict) -> Dict:
        """Handle PREDICT intent"""
        vessel_name = parsed.get("vessel_name")
        mmsi = parsed.get("identifiers", {}).get("mmsi")
        minutes = parsed.get("duration_minutes", 30)
        
        if not vessel_name and not mmsi:
            return {"message": "Please specify a vessel name or MMSI"}
        
        try:
            # Fetch historical data
            if vessel_name:
                df = self.db.fetch_vessel_by_name(vessel_name, limit=12)
            else:
                df = self.db.fetch_vessel_by_mmsi(mmsi, limit=12)
            
            if df.empty:
                return {"message": "No historical data found"}
            
            latest = df.iloc[-1]
            
            # Try XGBoost prediction
            if self.xgboost_enabled and self.prediction_engine:
                try:
                    result = self.prediction_engine.predict_vessel_position(
                        vessel_name=vessel_name or str(mmsi),
                        mmsi=mmsi,
                        minutes_ahead=minutes,
                        limit=12
                    )
                    
                    if "error" not in result:
                        return result
                except Exception as e:
                    logger.warning(f"XGBoost prediction failed: {e}")
            
            # Fallback to dead reckoning
            logger.info("Using dead reckoning fallback")
            
            lat = float(latest.get("LAT", 0))
            lon = float(latest.get("LON", 0))
            sog = float(latest.get("SOG", 0))
            cog = float(latest.get("COG", 0))
            
            # Simple dead reckoning
            from math import radians, cos, sin, sqrt, atan2, degrees
            
            # Convert to radians
            lat_rad = radians(lat)
            lon_rad = radians(lon)
            cog_rad = radians(cog)
            
            # Earth radius in nautical miles
            R = 3440.065
            
            # Distance traveled
            distance = (sog * minutes) / 60
            
            # Calculate new position
            new_lat = asin(sin(lat_rad) * cos(distance/R) + cos(lat_rad) * sin(distance/R) * cos(cog_rad))
            new_lon = lon_rad + atan2(sin(cog_rad) * sin(distance/R) * cos(lat_rad), cos(distance/R) - sin(lat_rad) * sin(new_lat))
            
            pred_lat = degrees(new_lat)
            pred_lon = degrees(new_lon)
            
            # Generate trajectory
            trajectory = []
            for i in range(0, minutes + 1, 5):
                dist = (sog * i) / 60
                t_lat = asin(sin(lat_rad) * cos(dist/R) + cos(lat_rad) * sin(dist/R) * cos(cog_rad))
                t_lon = lon_rad + atan2(sin(cog_rad) * sin(dist/R) * cos(lat_rad), cos(dist/R) - sin(lat_rad) * sin(t_lat))
                trajectory.append({
                    "lat": degrees(t_lat),
                    "lon": degrees(t_lon),
                    "minutes": i
                })
            
            response = {
                "vessel_name": vessel_name or str(mmsi),
                "mmsi": mmsi,
                "last_position": {
                    "lat": lat,
                    "lon": lon,
                    "sog": sog,
                    "cog": cog,
                    "timestamp": str(latest.get("BaseDateTime", ""))
                },
                "predicted_position": {
                    "lat": pred_lat,
                    "lon": pred_lon,
                    "sog": sog,
                    "cog": cog,
                    "timestamp": f"+{minutes} minutes"
                },
                "trajectory_points": trajectory,
                "confidence": 0.7,
                "method": "dead_reckoning"
            }
            
            return response
        
        except Exception as e:
            logger.error(f"Error in PREDICT: {e}")
            import traceback
            traceback.print_exc()
            return {"message": f"Error: {str(e)}"}


# Import math functions
from math import asin, cos, sin, sqrt, atan2, degrees, radians

