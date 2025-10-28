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

# maximum tolerance when matching a requested datetime (minutes)
TIME_TOLERANCE_MINUTES = 30

logger = logging.getLogger(__name__)

# Try to import XGBoost backend
try:
    from xgboost_backend_integration import XGBoostBackendPredictor
    XGBOOST_AVAILABLE = True
    logger.info("✅ XGBoost backend available")
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("⚠️  XGBoost backend not available - using dead reckoning only")

class IntentExecutor:
    def __init__(self, db: MaritimeDB, time_tolerance_minutes: int = 30):
        self.db = db
        self.time_tolerance_minutes = time_tolerance_minutes
        self.xgboost_predictor = None
        if XGBOOST_AVAILABLE:
            try:
                self.xgboost_predictor = XGBoostBackendPredictor()
                logger.info("✅ XGBoost predictor initialized")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize XGBoost predictor: {e}")

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
                # Normalize vessel name to uppercase for database queries
                vessel_name_upper = vessel_name.upper()
                df = self.db.fetch_vessel_by_name(vessel_name_upper, limit=1000)

                # if no exact matches, try a case-insensitive LIKE (wildcard on both sides)
                if df.empty:
                    like_df = self.db.fetch_vessel_by_name_like(f"%{vessel_name_upper}%", limit=1000)
                    if not like_df.empty:
                        df = like_df

                # if still empty, try fuzzy match on vessel list (if rapidfuzz available)
                if df.empty:
                    try:
                        from rapidfuzz import process
                        vessel_candidates = self.db.get_all_vessel_names()
                        best = process.extractOne(vessel_name, vessel_candidates)
                        if best and best[1] > 80:
                            df = self.db.fetch_vessel_by_name(best[0], limit=1000)
                            # annotate that we matched a similar name
                            if not df.empty:
                                df['matched_name'] = best[0]
                                df['match_score'] = best[1]
                    except Exception:
                        # RapidFuzz not available; try Python stdlib difflib fallback
                        try:
                            vessel_candidates = self.db.get_all_vessel_names()
                            # get_close_matches returns list of close matches; cutoff 0.7
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
                # Try DB-level fast lookup first
                try:
                    # If vessel is numeric -> MMSI
                    if mmsi:
                        row_df = self.db.fetch_vessel_by_mmsi_at_or_before(int(mmsi), str(requested_dt_str))
                    else:
                        row_df = self.db.fetch_vessel_by_name_at_or_before(vessel_name, str(requested_dt_str))

                    if not row_df.empty:
                        sel_row = row_df.iloc[0]
                        # fetch track ending at this timestamp
                        track_df = self.db.fetch_track_ending_at(vessel_name=vessel_name if not mmsi else None, mmsi=int(mmsi) if mmsi else None, end_dt=sel_row.BaseDateTime, limit=10)
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
                    else:
                        # No row <= requested_dt; try nearest within tolerance using a small SQL window
                        # We'll try a symmetric window of +/- time_tolerance_minutes around the requested time and pick nearest
                        try:
                            # build window
                            try:
                                target_dt = pd.to_datetime(requested_dt_str)
                            except Exception:
                                target_dt = None
                            if target_dt is None:
                                # last-resort: fall back to pandas logic
                                raise RuntimeError("cannot parse target_dt")

                            start_window = (target_dt - pd.Timedelta(minutes=self.time_tolerance_minutes)).strftime('%Y-%m-%d %H:%M:%S')
                            end_window = (target_dt + pd.Timedelta(minutes=self.time_tolerance_minutes)).strftime('%Y-%m-%d %H:%M:%S')
                            if mmsi:
                                window_q = self.db.fetch_by_time_range(start_window, end_window, limit=1000)
                                window_q = window_q[window_q['MMSI'] == int(mmsi)]
                            else:
                                window_q = self.db.fetch_by_time_range(start_window, end_window, limit=1000)
                                window_q = window_q[window_q['VesselName'] == vessel_name]

                            if not window_q.empty:
                                window_q['BaseDateTime_dt'] = pd.to_datetime(window_q['BaseDateTime'], errors='coerce')
                                window_q['absdiff'] = (window_q['BaseDateTime_dt'] - target_dt).abs()
                                nearest = window_q.loc[window_q['absdiff'].idxmin()]
                                nearest_dt = nearest.BaseDateTime
                                # fetch track ending at nearest_dt
                                track_df = self.db.fetch_track_ending_at(vessel_name=vessel_name if not mmsi else None, mmsi=int(mmsi) if mmsi else None, end_dt=nearest_dt, limit=10)
                                track = track_df.to_dict(orient='records') if not track_df.empty else []
                                return {
                                    "VesselName": nearest.VesselName,
                                    "LAT": float(nearest.LAT),
                                    "LON": float(nearest.LON),
                                    "SOG": float(nearest.SOG),
                                    "COG": float(nearest.COG),
                                    "BaseDateTime": nearest.BaseDateTime,
                                    "track": track[::-1]
                                }
                        except Exception:
                            # fall back to previous pandas logic if SQL path fails
                            pass
                except Exception:
                    # outer SQL-path failure; proceed to pandas-based fallback below
                    pass

        elif intent == "VERIFY":
            # Verify consistency across last 3 points
            df = pd.DataFrame()
            if mmsi:
                df = self.db.fetch_vessel_by_mmsi(int(mmsi), limit=3)
            elif vessel_name:
                # Normalize vessel name to uppercase for database queries
                vessel_name_upper = vessel_name.upper()
                df = self.db.fetch_vessel_by_name(vessel_name_upper, limit=3)

            return self._verify_movement(df)


        elif intent == "PREDICT":
            # Predict position after given time horizon (e.g., 'after 30 minutes')
            # Use duration_minutes from NLP parser first (more reliable)
            minutes = parsed.get("duration_minutes")

            # Fallback: extract time horizon from parsed (assumed to be like 'after 30 minutes')
            if minutes is None:
                time_horizon = parsed.get("time_horizon")
                minutes = self._parse_minutes(time_horizon) if time_horizon else None

            if mmsi:
                df = self.db.fetch_vessel_by_mmsi(int(mmsi), limit=12)
            elif vessel_name:
                # Normalize vessel name to uppercase for database queries
                vessel_name_upper = vessel_name.upper()
                df = self.db.fetch_vessel_by_mmsi(int(mmsi), limit=12) if mmsi else self.db.fetch_vessel_by_name(vessel_name_upper, limit=12)

            if minutes is None:
                # default 30 minutes
                minutes = 30

            return self._predict_position(df, minutes, vessel_name)

        # Return last known position
        if not df.empty:
            # Return last known plus last 10 track points (most recent first)
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
                "track": track[::-1],  # return newest first
                "message": msg
            }

        return {"message": "No data found"}

    def _parse_minutes(self, time_horizon: Optional[str]) -> Optional[int]:
        if not time_horizon:
            return None
        import re
        m = re.search(r"(\d+)", time_horizon)
        if m:
            return int(m.group(1))
        return None

    def _predict_position(self, df: pd.DataFrame, minutes: int, vessel_name: Optional[str] = None):
        """Predict vessel position using XGBoost model or dead-reckoning fallback.

        Args:
            df: DataFrame with vessel trajectory data
            minutes: Minutes ahead to predict
            vessel_name: Vessel name for logging

        Returns:
            Dictionary with prediction results
        """
        if df.empty or len(df) < 1:
            return {"message": "No data to predict"}

        last = df.iloc[-1]
        try:
            sog = float(last.SOG or 0.0)  # knots
            cog = float(last.COG or 0.0)  # degrees
            lat = float(last.LAT)
            lon = float(last.LON)
        except Exception:
            return {"message": "Insufficient numeric data for prediction"}

        # Try XGBoost prediction first
        if self.xgboost_predictor is not None:
            try:
                logger.info(f"Attempting XGBoost prediction for {vessel_name or 'unknown'}")
                xgb_result = self.xgboost_predictor.predict(df)

                if xgb_result and "predicted_lat" in xgb_result:
                    logger.info(f"✅ XGBoost prediction successful")
                    return {
                        "vessel_name": last.VesselName,
                        "last_position": {
                            "lat": lat,
                            "lon": lon,
                            "sog": sog,
                            "cog": cog,
                            "datetime": last.BaseDateTime
                        },
                        "predicted_position": {
                            "lat": xgb_result.get("predicted_lat"),
                            "lon": xgb_result.get("predicted_lon"),
                            "sog": xgb_result.get("predicted_sog", sog),
                            "cog": xgb_result.get("predicted_cog", cog)
                        },
                        "minutes_ahead": minutes,
                        "confidence": xgb_result.get("confidence", 0.95),
                        "method": "xgboost"
                    }
            except Exception as e:
                logger.warning(f"⚠️  XGBoost prediction failed: {e}, falling back to dead reckoning")

        # Fallback to dead-reckoning
        logger.info(f"Using dead reckoning for {vessel_name or 'unknown'}")

        # distance in nautical miles = sog * (minutes / 60)
        distance_nm = sog * (minutes / 60.0)

        # convert nm to degrees (approx): 1 degree lat ~ 60 nm
        delta_deg = distance_nm / 60.0

        # Convert bearing to delta lat/lon (simplified):
        rad = math.radians(90 - cog)
        dlat = delta_deg * math.sin(rad)
        dlon = delta_deg * math.cos(rad) / max(math.cos(math.radians(lat)), 0.0001)

        pred_lat = lat + dlat
        pred_lon = lon + dlon

        return {
            "vessel_name": last.VesselName,
            "last_position": {
                "lat": lat,
                "lon": lon,
                "sog": sog,
                "cog": cog,
                "datetime": last.BaseDateTime
            },
            "predicted_position": {
                "lat": pred_lat,
                "lon": pred_lon,
                "sog": sog,
                "cog": cog
            },
            "minutes_ahead": minutes,
            "confidence": 0.70,
            "method": "dead_reckoning"
        }

    def _verify_movement(self, df: pd.DataFrame):
        """Check last 3 points for sudden jumps or unrealistic turns.
        Return a short verdict and the last points.
        """
        if df.empty or len(df) < 2:
            return {"message": "Not enough data to verify"}

        # compute pairwise distances and bearing changes
        def haversine_nm(lat1, lon1, lat2, lon2):
            # returns distance in nautical miles
            R_km = 6371.0
            import math
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
            c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
            dist_km = R_km * c
            return dist_km / 1.852  # km -> nautical miles

        points = []
        for _, row in df.iterrows():
            points.append((float(row.LAT), float(row.LON), float(row.SOG or 0.0), float(row.COG or 0.0)))

        # analyze distances and speed consistency
        verdict = "consistent"
        reasons = []
        for i in range(1, len(points)):
            lat1, lon1, sog1, cog1 = points[i-1]
            lat2, lon2, sog2, cog2 = points[i]
            dist = haversine_nm(lat1, lon1, lat2, lon2)
            # if a large jump (> 5 nm within short time) mark suspicious
            if dist > 5.0:
                verdict = "suspicious"
                reasons.append(f"Large jump of {dist:.1f} nm between points {i-1} and {i}")

            # large change in heading
            if abs(cog2 - cog1) > 90:
                verdict = "suspicious"
                reasons.append(f"Large course change of {abs(cog2-cog1):.1f} degrees between points {i-1} and {i}")

        return {"verdict": verdict, "reasons": reasons, "points": df.to_dict(orient='records')}
