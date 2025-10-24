"""
Enhanced Intent Executor with XGBoost Integration
Replaces simple dead-reckoning with advanced ML predictions
"""

try:
    from .db_handler import MaritimeDB
except ImportError:
    from db_handler import MaritimeDB

from typing import Dict, Optional
import re
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import difflib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# maximum tolerance when matching a requested datetime (minutes)
TIME_TOLERANCE_MINUTES = 30


class IntentExecutor:
    def __init__(self, db: MaritimeDB, time_tolerance_minutes: int = 30, enable_xgboost: bool = True):
        self.db = db
        self.time_tolerance_minutes = time_tolerance_minutes
        self.xgboost_enabled = False
        self.predictor = None
        self.service = None
        
        # Try to load XGBoost components
        if enable_xgboost:
            self._initialize_xgboost()
    
    def _initialize_xgboost(self):
        """Initialize XGBoost predictor and service"""
        try:
            # Import XGBoost components
            from xgboost_integration import XGBoostPredictor
            from vessel_prediction_service import VesselPredictionService
            
            # Find model directory
            current_dir = os.path.dirname(__file__)
            model_dir = os.path.join(current_dir, "..", "..", "xgboost_advanced_50_vessels")
            
            # Fallback to absolute path if relative path doesn't work
            if not os.path.exists(model_dir):
                model_dir = "results/xgboost_advanced_50_vessels"
            
            if os.path.exists(model_dir):
                self.predictor = XGBoostPredictor(model_dir)
                self.service = VesselPredictionService(self.db, self.predictor)
                self.xgboost_enabled = True
                logger.info("✅ XGBoost integration enabled")
            else:
                logger.warning(f"⚠️  Model directory not found: {model_dir}")
        
        except ImportError as e:
            logger.warning(f"⚠️  XGBoost not available: {e}")
        except Exception as e:
            logger.warning(f"⚠️  Error initializing XGBoost: {e}")

    def handle(self, parsed: Dict):
        intent = parsed.get("intent")
        vessel_name = parsed.get("vessel_name")
        identifiers = parsed.get("identifiers", {})
        requested_dt_str = parsed.get("datetime")

        # Prioritize MMSI if provided
        mmsi = identifiers.get("mmsi")

        df = pd.DataFrame()

        # If vessel_name looks numeric, treat it as MMSI
        if vessel_name and vessel_name.isdigit() and not mmsi:
            mmsi = vessel_name

        if intent == "SHOW":
            if mmsi:
                df = self.db.fetch_vessel_by_mmsi(int(mmsi), limit=1000)
            elif vessel_name:
                df = self.db.fetch_vessel_by_name(vessel_name, limit=1000)

                # if no exact matches, try a case-insensitive LIKE (wildcard on both sides)
                if df.empty:
                    like_df = self.db.fetch_vessel_by_name_like(f"%{vessel_name}%", limit=1000)
                    if not like_df.empty:
                        df = like_df

                # if still empty, try fuzzy match on vessel list
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
                                candidate = cm[0]
                                df = self.db.fetch_vessel_by_name(candidate, limit=1000)
                                if not df.empty:
                                    df['matched_name'] = candidate
                                    df['match_score'] = None
                        except Exception:
                            pass

            # If a datetime was requested, try to find the record closest to that time
            if requested_dt_str:
                try:
                    if mmsi:
                        row_df = self.db.fetch_vessel_by_mmsi_at_or_before(int(mmsi), str(requested_dt_str))
                    else:
                        row_df = self.db.fetch_vessel_by_name_at_or_before(vessel_name, str(requested_dt_str))       

                    if not row_df.empty:
                        sel_row = row_df.iloc[0]
                        track_df = self.db.fetch_track_ending_at(
                            vessel_name=vessel_name if not mmsi else None, 
                            mmsi=int(mmsi) if mmsi else None, 
                            end_dt=sel_row.BaseDateTime, 
                            limit=10
                        )
                        track = track_df.to_dict(orient='records') if not track_df.empty else []
                        return {
                            "VesselName": sel_row.VesselName,
                            "LAT": float(sel_row.LAT),
                            "LON": float(sel_row.LON),
                            "SOG": float(sel_row.SOG),
                            "COG": float(sel_row.COG),
                            "BaseDateTime": sel_row.BaseDateTime,
                            "track": track[::-1],
                            "message": f"Last known position for {sel_row.VesselName} at {sel_row.BaseDateTime}: {sel_row.LAT}, {sel_row.LON} (MMSI {int(sel_row.MMSI)})"
                        }
                except Exception:
                    pass

        elif intent == "VERIFY":
            # Enhanced VERIFY with XGBoost
            if self.xgboost_enabled and self.service:
                try:
                    result = self.service.verify_vessel_course(vessel_name=vessel_name, mmsi=mmsi)
                    if "error" not in result:
                        return result
                except Exception as e:
                    logger.warning(f"XGBoost verification failed: {e}")
            
            # Fallback to simple verification
            df = pd.DataFrame()
            if mmsi:
                df = self.db.fetch_vessel_by_mmsi(int(mmsi), limit=5)
            elif vessel_name:
                df = self.db.fetch_vessel_by_name(vessel_name, limit=5)

            return self._verify_movement(df)

        elif intent == "PREDICT":
            # Enhanced PREDICT with XGBoost
            time_horizon = parsed.get("time_horizon")
            minutes = self._parse_minutes(time_horizon) if time_horizon else 30
            
            if self.xgboost_enabled and self.service:
                try:
                    result = self.service.predict_vessel_position(
                        vessel_name=vessel_name,
                        mmsi=mmsi,
                        minutes_ahead=minutes
                    )
                    if "error" not in result:
                        return result
                except Exception as e:
                    logger.warning(f"XGBoost prediction failed: {e}")
            
            # Fallback to simple dead-reckoning
            if mmsi:
                df = self.db.fetch_vessel_by_mmsi(int(mmsi), limit=12)
            elif vessel_name:
                df = self.db.fetch_vessel_by_name(vessel_name, limit=12)

            return self._predict_position(df, minutes)

        # Return last known position
        if not df.empty:
            last_row = df.iloc[-1]
            track = df.tail(10).to_dict(orient='records')
            msg = f"Last known position for {last_row.VesselName} at {last_row.BaseDateTime}: {last_row.LAT}, {last_row.LON} (MMSI {int(last_row.MMSI)})"
            return {
                "VesselName": last_row.VesselName,
                "LAT": float(last_row.LAT),
                "LON": float(last_row.LON),
                "SOG": float(last_row.SOG),
                "COG": float(last_row.COG),
                "BaseDateTime": last_row.BaseDateTime,
                "track": track[::-1],
                "message": msg
            }

        return {"message": "No data found"}

    def _parse_minutes(self, time_horizon: Optional[str]) -> Optional[int]:
        if not time_horizon:
            return None
        m = re.search(r"(\d+)", time_horizon)
        if m:
            return int(m.group(1))
        return None

    def _predict_position(self, df: pd.DataFrame, minutes: int):
        """Fallback: Simple dead-reckoning using last known SOG and COG"""
        if df.empty or len(df) < 1:
            return {"message": "No data to predict"}

        last = df.iloc[-1]
        try:
            sog = float(last.SOG or 0.0)
            cog = float(last.COG or 0.0)
            lat = float(last.LAT)
            lon = float(last.LON)
        except Exception:
            return {"message": "Insufficient numeric data for prediction"}

        distance_nm = sog * (minutes / 60.0)
        delta_deg = distance_nm / 60.0

        rad = math.radians(90 - cog)
        dlat = delta_deg * math.sin(rad)
        dlon = delta_deg * math.cos(rad) / max(math.cos(math.radians(lat)), 0.0001)

        pred_lat = lat + dlat
        pred_lon = lon + dlon

        return {
            "VesselName": last.VesselName,
            "Predicted_LAT": pred_lat,
            "Predicted_LON": pred_lon,
            "MinutesAhead": minutes,
            "BaseDateTime": last.BaseDateTime,
            "method": "dead_reckoning"
        }

    def _verify_movement(self, df: pd.DataFrame):
        """Check last 5 points for sudden jumps or unrealistic turns"""
        if df.empty or len(df) < 2:
            return {"message": "Not enough data to verify"}

        def haversine_nm(lat1, lon1, lat2, lon2):
            R_km = 6371.0
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2  
            c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
            dist_km = R_km * c
            return dist_km / 1.852

        points = []
        for _, row in df.iterrows():
            points.append((float(row.LAT), float(row.LON), float(row.SOG or 0.0), float(row.COG or 0.0)))

        verdict = "consistent"
        reasons = []
        for i in range(1, len(points)):
            lat1, lon1, sog1, cog1 = points[i-1]
            lat2, lon2, sog2, cog2 = points[i]
            dist = haversine_nm(lat1, lon1, lat2, lon2)
            
            if dist > 5.0:
                verdict = "suspicious"
                reasons.append(f"Large jump of {dist:.1f} nm between points {i-1} and {i}")

            if abs(cog2 - cog1) > 90:
                verdict = "suspicious"
                reasons.append(f"Large course change of {abs(cog2-cog1):.1f}° between points {i-1} and {i}")  

        return {
            "verdict": verdict, 
            "reasons": reasons, 
            "points": df.to_dict(orient='records'),
            "method": "simple_verification"
        }

