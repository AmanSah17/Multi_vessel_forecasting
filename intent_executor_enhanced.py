"""
Enhanced Intent Executor with XGBoost Backend Integration
Handles SHOW, PREDICT, and VERIFY intents with ML-based predictions
"""

try:
    from .db_handler import MaritimeDB
except ImportError:
    from db_handler import MaritimeDB

from typing import Dict, Optional
import re
import pandas as pd
import math
from datetime import datetime, timedelta
import difflib
import logging
import os

logger = logging.getLogger(__name__)

# Import XGBoost backend integration
try:
    from xgboost_backend_integration import XGBoostBackendPredictor, VesselPredictionEngine
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("⚠️  XGBoost backend integration not available")


class IntentExecutor:
    """Enhanced intent executor with XGBoost predictions"""
    
    def __init__(self, db: MaritimeDB, time_tolerance_minutes: int = 30):
        self.db = db
        self.time_tolerance_minutes = time_tolerance_minutes
        
        # Initialize XGBoost backend if available
        self.xgboost_enabled = False
        self.predictor = None
        self.prediction_engine = None
        
        if XGBOOST_AVAILABLE:
            self._initialize_xgboost()
    
    def _initialize_xgboost(self):
        """Initialize XGBoost backend predictor"""
        try:
            # Try to find model directory
            model_dir = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "..",
                "results",
                "xgboost_advanced_50_vessels"
            )
            
            # Also try relative to current working directory
            if not os.path.exists(model_dir):
                model_dir = os.path.join(
                    os.getcwd(),
                    "results",
                    "xgboost_advanced_50_vessels"
                )
            
            if os.path.exists(model_dir):
                self.predictor = XGBoostBackendPredictor(model_dir)
                if self.predictor.is_loaded:
                    self.prediction_engine = VesselPredictionEngine(self.db, self.predictor)
                    self.xgboost_enabled = True
                    logger.info("✅ XGBoost backend integration enabled")
                    logger.info(f"📊 Model status: {self.predictor.get_status()}")
            else:
                logger.warning(f"⚠️  Model directory not found: {model_dir}")
                
        except Exception as e:
            logger.warning(f"⚠️  Error initializing XGBoost: {e}")
    
    def handle(self, parsed: Dict):
        """Handle parsed intent"""
        intent = parsed.get("intent")
        vessel_name = parsed.get("vessel_name")
        identifiers = parsed.get("identifiers", {})
        requested_dt_str = parsed.get("datetime")
        
        mmsi = identifiers.get("mmsi")
        
        # If vessel_name looks numeric, treat it as MMSI
        if vessel_name and vessel_name.isdigit() and not mmsi:
            mmsi = vessel_name
        
        if intent == "SHOW":
            return self._handle_show(vessel_name, mmsi, requested_dt_str)
        
        elif intent == "VERIFY":
            return self._handle_verify(vessel_name, mmsi)
        
        elif intent == "PREDICT":
            time_horizon = parsed.get("time_horizon")
            minutes = self._parse_minutes(time_horizon) if time_horizon else 30
            return self._handle_predict(vessel_name, mmsi, minutes)
        
        return {"message": "Unknown intent"}
    
    def _handle_show(self, vessel_name: str, mmsi: int, requested_dt_str: str) -> Dict:
        """Handle SHOW intent"""
        df = pd.DataFrame()
        
        if mmsi:
            df = self.db.fetch_vessel_by_mmsi(int(mmsi), limit=1000)
        elif vessel_name:
            df = self.db.fetch_vessel_by_name(vessel_name, limit=1000)
            
            if df.empty:
                like_df = self.db.fetch_vessel_by_name_like(f"%{vessel_name}%", limit=1000)
                if not like_df.empty:
                    df = like_df
            
            if df.empty:
                try:
                    from rapidfuzz import process
                    vessel_candidates = self.db.get_all_vessel_names()
                    best = process.extractOne(vessel_name, vessel_candidates)
                    if best and best[1] > 80:
                        df = self.db.fetch_vessel_by_name(best[0], limit=1000)
                        if not df.empty:
                            df['matched_name'] = best[0]
                            df['match_score'] = best[1]
                except Exception:
                    try:
                        vessel_candidates = self.db.get_all_vessel_names()
                        cm = difflib.get_close_matches(vessel_name, vessel_candidates, n=1, cutoff=0.7)
                        if cm:
                            df = self.db.fetch_vessel_by_name(cm[0], limit=1000)
                            if not df.empty:
                                df['matched_name'] = cm[0]
                    except Exception:
                        pass
        
        if not df.empty:
            last_row = df.iloc[-1]
            track = df.tail(10).to_dict(orient='records')
            
            return {
                "VesselName": last_row.VesselName,
                "LAT": float(last_row.LAT),
                "LON": float(last_row.LON),
                "SOG": float(last_row.SOG),
                "COG": float(last_row.COG),
                "BaseDateTime": last_row.BaseDateTime,
                "track": track[::-1],
                "message": f"Last known position for {last_row.VesselName} at {last_row.BaseDateTime}: {last_row.LAT}, {last_row.LON} (MMSI {int(last_row.MMSI)})"
            }
        
        return {"message": "No data found"}
    
    def _handle_verify(self, vessel_name: str, mmsi: int) -> Dict:
        """Handle VERIFY intent"""
        df = pd.DataFrame()
        
        if mmsi:
            df = self.db.fetch_vessel_by_mmsi(int(mmsi), limit=5)
        elif vessel_name:
            df = self.db.fetch_vessel_by_name(vessel_name, limit=5)
        
        if df.empty:
            return {"message": "No data found"}
        
        df = df.sort_values('BaseDateTime')
        
        # Check course consistency
        if len(df) >= 3:
            last_3 = df.tail(3)
            cogs = last_3['COG'].values
            sogs = last_3['SOG'].values
            
            cog_variance = float(pd.Series(cogs).std())
            sog_variance = float(pd.Series(sogs).std())
            
            is_consistent = cog_variance < 30 and sog_variance < 5
            
            return {
                "vessel_name": str(last_3.iloc[-1]['VesselName']),
                "is_consistent": is_consistent,
                "cog_variance": cog_variance,
                "sog_variance": sog_variance,
                "last_3_points": last_3[['BaseDateTime', 'LAT', 'LON', 'SOG', 'COG']].to_dict(orient='records'),
                "message": "Course is consistent" if is_consistent else "Anomaly detected in course"
            }
        
        return {"message": "Insufficient data for verification"}
    
    def _handle_predict(self, vessel_name: str, mmsi: int, minutes: int) -> Dict:
        """Handle PREDICT intent with XGBoost backend"""
        
        # Try XGBoost prediction first
        if self.xgboost_enabled and self.prediction_engine:
            try:
                result = self.prediction_engine.predict_vessel_position(
                    vessel_name=vessel_name,
                    mmsi=mmsi,
                    minutes_ahead=minutes,
                    limit=12
                )
                
                if "error" not in result:
                    logger.info(f"✅ XGBoost prediction successful for {vessel_name or mmsi}")
                    return result
                else:
                    logger.warning(f"⚠️  XGBoost prediction failed: {result['error']}")
            
            except Exception as e:
                logger.warning(f"⚠️  XGBoost prediction error: {e}")
        
        # Fallback to dead reckoning
        logger.info(f"📍 Using dead reckoning for {vessel_name or mmsi}")
        
        df = pd.DataFrame()
        if mmsi:
            df = self.db.fetch_vessel_by_mmsi(int(mmsi), limit=2)
        elif vessel_name:
            df = self.db.fetch_vessel_by_name(vessel_name, limit=2)
        
        if df.empty:
            return {"message": "No vessel data found"}
        
        df = df.sort_values('BaseDateTime')
        last_row = df.iloc[-1]
        
        # Dead reckoning
        lat = float(last_row['LAT'])
        lon = float(last_row['LON'])
        sog = float(last_row['SOG'])
        cog = float(last_row['COG'])
        
        # Simple dead reckoning
        speed_kmh = sog * 1.852
        distance_km = (speed_kmh * minutes) / 60
        
        cog_rad = math.radians(cog)
        lat_change = (distance_km / 111.0) * math.cos(cog_rad)
        lon_change = (distance_km / (111.0 * math.cos(math.radians(lat)))) * math.sin(cog_rad)
        
        pred_lat = lat + lat_change
        pred_lon = lon + lon_change
        
        return {
            "vessel_name": str(last_row['VesselName']),
            "mmsi": int(last_row['MMSI']),
            "last_position": {
                "lat": lat,
                "lon": lon,
                "sog": sog,
                "cog": cog,
                "timestamp": str(last_row['BaseDateTime'])
            },
            "predicted_position": {
                "lat": pred_lat,
                "lon": pred_lon,
                "sog": sog,
                "cog": cog,
                "timestamp": str(datetime.fromisoformat(str(last_row['BaseDateTime'])) + timedelta(minutes=minutes))
            },
            "confidence": 0.7,
            "method": "dead_reckoning",
            "minutes_ahead": minutes
        }
    
    @staticmethod
    def _parse_minutes(time_horizon: Optional[str]) -> Optional[int]:
        """Parse minutes from time horizon string"""
        if not time_horizon:
            return None
        m = re.search(r"(\d+)", time_horizon)
        if m:
            return int(m.group(1))
        return None

