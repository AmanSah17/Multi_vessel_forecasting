# 🔍 API Issues Found and Fixes

**Date**: 2025-10-25  
**Status**: Issues Identified and Solutions Ready

---

## 📋 Issues Summary

### Issue 1: PREDICT Intent Returns Empty Response ❌
**Symptom**: `{"message": "No data to predict"}`

**Root Cause**: 
- In `intent_executor.py`, the PREDICT handler fetches data with `limit=2`
- The database query returns 0 rows because the vessel name matching fails
- The vessel name is normalized to "Champagne Cher" but database has "CHAMPAGNE CHER"

**Location**: `f:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app\intent_executor.py` (lines 130-145)

**Current Code**:
```python
elif intent == "PREDICT":
    time_horizon = parsed.get("time_horizon")
    minutes = self._parse_minutes(time_horizon) if time_horizon else None

    if mmsi:
        df = self.db.fetch_vessel_by_mmsi(int(mmsi), limit=2)
    elif vessel_name:
        df = self.db.fetch_vessel_by_name(vessel_name, limit=2)  # ❌ FAILS HERE

    if minutes is None:
        minutes = 30

    return self._predict_position(df, minutes)
```

**Problem**: 
- `vessel_name` is "Champagne Cher" (title case from NLP)
- Database has "CHAMPAGNE CHER" (uppercase)
- `fetch_vessel_by_name()` does exact match, so it fails

---

### Issue 2: NLP Parser Extracts `duration_minutes` But Intent Executor Ignores It ⚠️
**Symptom**: `duration_minutes` is None in parsed output even though NLP extracts it

**Root Cause**:
- NLP parser correctly extracts `duration_minutes` in `_compute_end_dt()`
- But `intent_executor.py` doesn't use it - it only uses `time_horizon` string
- Then it tries to parse `time_horizon` again with regex

**Location**: `f:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app\intent_executor.py` (line 133)

**Current Code**:
```python
time_horizon = parsed.get("time_horizon")
minutes = self._parse_minutes(time_horizon) if time_horizon else None  # ❌ REDUNDANT
```

**Better Approach**:
```python
# Use duration_minutes from NLP parser first
minutes = parsed.get("duration_minutes")
if minutes is None:
    # Fallback: parse time_horizon if duration_minutes not available
    time_horizon = parsed.get("time_horizon")
    minutes = self._parse_minutes(time_horizon) if time_horizon else None
```

---

### Issue 3: Vessel Name Case Sensitivity ❌
**Symptom**: Vessel search works but data fetching fails for PREDICT

**Root Cause**:
- NLP parser returns vessel names in title case: "Champagne Cher"
- Database stores them in uppercase: "CHAMPAGNE CHER"
- `fetch_vessel_by_name()` does case-sensitive exact match

**Location**: `f:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app\db_handler.py`

**Solution**: 
- Normalize vessel names to uppercase before database queries
- Or use case-insensitive SQL queries

---

### Issue 4: PREDICT Response Structure Mismatch ❌
**Symptom**: Frontend expects `predicted_position` object but gets flat keys

**Root Cause**:
- `_predict_position()` returns flat structure:
  ```python
  {
    "VesselName": "...",
    "Predicted_LAT": 32.5,
    "Predicted_LON": -77.0,
    "MinutesAhead": 30,
    "BaseDateTime": "..."
  }
  ```
- Frontend expects nested structure:
  ```python
  {
    "vessel_name": "...",
    "last_position": {"lat": 32.7, "lon": -77.0},
    "predicted_position": {"lat": 32.5, "lon": -77.0},
    "confidence": 0.95,
    "method": "xgboost"
  }
  ```

**Location**: `f:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app\intent_executor.py` (lines 200-215)

---

### Issue 5: XGBoost Integration Not Called ❌
**Symptom**: Predictions use only dead reckoning, not ML model

**Root Cause**:
- `intent_executor.py` doesn't import or use XGBoost backend
- `_predict_position()` only does simple dead reckoning calculation
- XGBoost model is loaded on backend but never called

**Location**: `f:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app\intent_executor.py`

**Missing**: 
```python
from xgboost_backend_integration import XGBoostBackendPredictor
```

---

## ✅ Fixes Required

### Fix 1: Normalize Vessel Names to Uppercase
**File**: `f:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app\intent_executor.py`

**Change**:
```python
# Before
elif vessel_name:
    df = self.db.fetch_vessel_by_name(vessel_name, limit=2)

# After
elif vessel_name:
    df = self.db.fetch_vessel_by_name(vessel_name.upper(), limit=2)
```

---

### Fix 2: Use `duration_minutes` from NLP Parser
**File**: `f:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app\intent_executor.py`

**Change**:
```python
# Before
time_horizon = parsed.get("time_horizon")
minutes = self._parse_minutes(time_horizon) if time_horizon else None

# After
minutes = parsed.get("duration_minutes")
if minutes is None:
    time_horizon = parsed.get("time_horizon")
    minutes = self._parse_minutes(time_horizon) if time_horizon else None
```

---

### Fix 3: Integrate XGBoost Model
**File**: `f:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app\intent_executor.py`

**Add Import**:
```python
try:
    from xgboost_backend_integration import XGBoostBackendPredictor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
```

**Update `_predict_position()`**:
```python
def _predict_position(self, df: pd.DataFrame, minutes: int):
    if df.empty or len(df) < 1:
        return {"message": "No data to predict"}
    
    # Try XGBoost first
    if XGBOOST_AVAILABLE:
        try:
            predictor = XGBoostBackendPredictor()
            result = predictor.predict(df)
            if result:
                return {
                    "vessel_name": df.iloc[-1].VesselName,
                    "last_position": {
                        "lat": float(df.iloc[-1].LAT),
                        "lon": float(df.iloc[-1].LON)
                    },
                    "predicted_position": {
                        "lat": result.get("predicted_lat"),
                        "lon": result.get("predicted_lon")
                    },
                    "confidence": result.get("confidence", 0.95),
                    "method": "xgboost"
                }
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
    
    # Fallback to dead reckoning
    return self._dead_reckoning(df, minutes)
```

---

### Fix 4: Standardize Response Format
**File**: `f:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app\intent_executor.py`

**Update Response Structure**:
```python
# All PREDICT responses should follow this format:
{
    "vessel_name": "CHAMPAGNE CHER",
    "last_position": {
        "lat": 32.7315,
        "lon": -77.00767,
        "sog": 19.4,
        "cog": 27.0,
        "datetime": "2024-01-28 03:00:00"
    },
    "predicted_position": {
        "lat": 32.5,
        "lon": -77.2,
        "sog": 19.4,
        "cog": 27.0
    },
    "minutes_ahead": 30,
    "confidence": 0.95,
    "method": "xgboost"  # or "dead_reckoning"
}
```

---

## 🔧 Implementation Order

1. **Fix vessel name normalization** (5 min)
2. **Use duration_minutes from NLP** (5 min)
3. **Integrate XGBoost model** (15 min)
4. **Standardize response format** (10 min)
5. **Test all intents** (10 min)

**Total Time**: ~45 minutes

---

## 🧪 Testing After Fixes

```bash
# Test PREDICT intent
python test_backend_direct.py

# Expected output:
# ✅ Prediction request successful
# ✅ Prediction data found!
# ✅ Vessel: CHAMPAGNE CHER
# ✅ Last Position: (32.7315, -77.00767)
# ✅ Predicted Position: (32.5, -77.2)
# ✅ Confidence: 0.95
```

---

## 📊 Impact

| Component | Status | Impact |
|-----------|--------|--------|
| Vessel Search | ✅ Working | No changes needed |
| NLP Parsing | ✅ Working | Minor optimization |
| Data Fetching | ⚠️ Partial | Fix case sensitivity |
| PREDICT Intent | ❌ Broken | Major fixes needed |
| XGBoost Integration | ❌ Not Used | Critical fix needed |

---

## 📝 Summary

**Root Cause**: Intent executor doesn't properly handle vessel name case sensitivity and doesn't integrate XGBoost model

**Solution**: Normalize vessel names, use NLP-extracted duration, integrate XGBoost, standardize response format

**Effort**: ~45 minutes

**Priority**: 🔴 CRITICAL - Blocks all predictions

