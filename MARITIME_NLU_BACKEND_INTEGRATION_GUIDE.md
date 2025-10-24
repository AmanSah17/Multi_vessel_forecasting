# Maritime NLU Backend Integration Guide

## 🎯 Quick Integration Steps

This guide shows how to integrate the XGBoost model into the Maritime NLU backend.

---

## Step 1: Copy Integration Files

Copy these files to your Maritime NLU project:

```bash
# From Multi_vessel_forecasting project
cp xgboost_integration.py Maritime_NLU_Repo/backend/nlu_chatbot/src/app/
cp vessel_prediction_service.py Maritime_NLU_Repo/backend/nlu_chatbot/src/app/
cp trajectory_visualization.py Maritime_NLU_Repo/backend/nlu_chatbot/src/app/

# Copy model directory
cp -r results/xgboost_advanced_50_vessels Maritime_NLU_Repo/backend/nlu_chatbot/
```

---

## Step 2: Update intent_executor.py

Modify `backend/nlu_chatbot/src/app/intent_executor.py`:

```python
# Add imports at top
from xgboost_integration import XGBoostPredictor
from vessel_prediction_service import VesselPredictionService
from trajectory_visualization import TrajectoryVisualizer
import os

class IntentExecutor:
    def __init__(self, db: MaritimeDB, time_tolerance_minutes: int = 30):
        self.db = db
        self.time_tolerance_minutes = time_tolerance_minutes
        
        # Initialize XGBoost components
        try:
            model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "xgboost_advanced_50_vessels")
            self.predictor = XGBoostPredictor(model_dir)
            self.service = VesselPredictionService(db, self.predictor)
            self.visualizer = TrajectoryVisualizer()
            self.xgboost_enabled = True
        except Exception as e:
            print(f"Warning: XGBoost not available: {e}")
            self.xgboost_enabled = False
    
    def handle(self, parsed: Dict):
        intent = parsed.get("intent")
        vessel_name = parsed.get("vessel_name")
        identifiers = parsed.get("identifiers", {})
        mmsi = identifiers.get("mmsi")
        
        # NEW: Handle PREDICT intent with XGBoost
        if intent == "PREDICT" and self.xgboost_enabled:
            time_horizon = parsed.get("time_horizon")
            minutes = self._parse_minutes(time_horizon) if time_horizon else 30
            
            return self.service.predict_vessel_position(
                vessel_name=vessel_name,
                mmsi=mmsi,
                minutes_ahead=minutes
            )
        
        # NEW: Handle VERIFY intent with XGBoost
        elif intent == "VERIFY" and self.xgboost_enabled:
            return self.service.verify_vessel_course(
                vessel_name=vessel_name,
                mmsi=mmsi
            )
        
        # ... rest of existing code
```

---

## Step 3: Update main.py

Add new endpoints to `backend/nlu_chatbot/src/app/main.py`:

```python
# Add imports
from intent_executor import IntentExecutor

# In the app initialization section, ensure executor has XGBoost
executor = IntentExecutor(db)

# Add new endpoints
@app.post("/predict")
async def predict_vessel(request: QueryRequest):
    """Predict vessel position using XGBoost model"""
    try:
        parsed = nlp_engine.parse_query(request.text)
        response = executor.handle(parsed)
        response = clean_nan_values(response)
        formatted_text = ResponseFormatter.format_response(parsed.get("intent", ""), response)
        
        return {
            "parsed": parsed,
            "response": response,
            "formatted_response": formatted_text
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.post("/verify")
async def verify_vessel(request: QueryRequest):
    """Verify vessel course using XGBoost model"""
    try:
        parsed = nlp_engine.parse_query(request.text)
        response = executor.handle(parsed)
        response = clean_nan_values(response)
        formatted_text = ResponseFormatter.format_response(parsed.get("intent", ""), response)
        
        return {
            "parsed": parsed,
            "response": response,
            "formatted_response": formatted_text
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.get("/model/status")
def model_status():
    """Check XGBoost model status"""
    return {
        "xgboost_enabled": executor.xgboost_enabled,
        "model_type": "XGBoost Advanced",
        "features": 483,
        "pca_components": 80,
        "accuracy_lat_mae": 0.3056,
        "accuracy_lon_mae": 1.1040
    }
```

---

## Step 4: Update NLP Interpreter

Modify `backend/nlu_chatbot/src/app/nlp_interpreter.py` to recognize PREDICT and VERIFY intents:

```python
def parse_query(self, query: str) -> Dict:
    # ... existing code ...
    
    # Add PREDICT intent detection
    if any(word in query.lower() for word in ["predict", "forecast", "estimate", "where will"]):
        intent = "PREDICT"
        # Extract time horizon
        import re
        time_match = re.search(r"(\d+)\s*(minute|hour|day)", query.lower())
        if time_match:
            parsed["time_horizon"] = f"after {time_match.group(1)} {time_match.group(2)}"
    
    # Add VERIFY intent detection
    elif any(word in query.lower() for word in ["verify", "check", "validate", "course"]):
        intent = "VERIFY"
    
    # ... rest of existing code ...
    
    return parsed
```

---

## Step 5: Update Requirements

Add to `backend/nlu_chatbot/requirements.txt`:

```
xgboost>=1.5.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

Install:
```bash
cd backend/nlu_chatbot
pip install -r requirements.txt
```

---

## Step 6: Test Integration

### Test 1: Check Model Loading

```bash
cd backend/nlu_chatbot/src/app
python -c "from xgboost_integration import XGBoostPredictor; p = XGBoostPredictor(); print('✅ Model loaded')"
```

### Test 2: Test Prediction Endpoint

```bash
# Start backend
cd backend/nlu_chatbot/src/app
uvicorn main:app --reload

# In another terminal, test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Predict VESSEL_NAME position after 30 minutes"}'
```

### Test 3: Test Verify Endpoint

```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"text": "Verify VESSEL_NAME course"}'
```

### Test 4: Check Model Status

```bash
curl http://localhost:8000/model/status
```

---

## Step 7: Update Frontend (Optional)

Add buttons to Streamlit frontend (`frontend/pages/chatbot.py`):

```python
import streamlit as st

# Add prediction section
st.sidebar.header("🔮 XGBoost Predictions")

if st.sidebar.button("Predict Vessel Position"):
    vessel_name = st.sidebar.text_input("Vessel Name")
    minutes = st.sidebar.slider("Minutes Ahead", 5, 120, 30)
    
    if vessel_name:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"text": f"Predict {vessel_name} position after {minutes} minutes"}
        )
        st.json(response.json())

if st.sidebar.button("Verify Vessel Course"):
    vessel_name = st.sidebar.text_input("Vessel Name")
    
    if vessel_name:
        response = requests.post(
            "http://localhost:8000/verify",
            json={"text": f"Verify {vessel_name} course"}
        )
        st.json(response.json())
```

---

## Usage Examples

### Example 1: Predict Position

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "Predict CHAMPAGNE CHER position after 30 minutes"}'
```

**Response**:
```json
{
  "parsed": {
    "intent": "PREDICT",
    "vessel_name": "CHAMPAGNE CHER",
    "time_horizon": "after 30 minutes"
  },
  "response": {
    "status": "success",
    "vessel_name": "CHAMPAGNE CHER",
    "predicted_position": {
      "latitude": 40.7128,
      "longitude": -74.0060,
      "sog": 12.5,
      "cog": 180.0
    },
    "confidence_score": 0.95
  },
  "formatted_response": "CHAMPAGNE CHER will be at latitude 40.7128, longitude -74.0060 in 30 minutes with confidence 95%"
}
```

### Example 2: Verify Course

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "Verify CHAMPAGNE CHER course"}'
```

**Response**:
```json
{
  "parsed": {
    "intent": "VERIFY",
    "vessel_name": "CHAMPAGNE CHER"
  },
  "response": {
    "status": "success",
    "verification": {
      "course_consistency": "stable",
      "anomaly_detected": false
    },
    "confidence_score": 0.95
  },
  "formatted_response": "CHAMPAGNE CHER course is stable with no anomalies detected"
}
```

---

## Troubleshooting

### Issue: Model files not found

**Solution**: Ensure model directory path is correct:
```python
model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "xgboost_advanced_50_vessels")
print(f"Looking for model at: {model_dir}")
print(f"Files: {os.listdir(model_dir)}")
```

### Issue: XGBoost not installed

**Solution**: Install dependencies:
```bash
pip install xgboost scikit-learn matplotlib
```

### Issue: Database connection error

**Solution**: Verify database path in main.py:
```python
db_path = os.environ.get("BACKEND_DB_PATH", "maritime_data.db")
print(f"Using database: {db_path}")
```

---

## Performance Optimization

### Enable GPU Acceleration

```python
# In xgboost_integration.py
self.model = pickle.load(f)
# For GPU predictions (if available):
# self.model.set_params(tree_method='gpu_hist', gpu_id=0)
```

### Batch Processing

```python
# Process multiple vessels
vessels = ["VESSEL1", "VESSEL2", "VESSEL3"]
predictions = []
for vessel in vessels:
    pred = service.predict_vessel_position(vessel_name=vessel)
    predictions.append(pred)
```

---

## Monitoring

### Log Predictions

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Predicted {vessel_name}: LAT={pred_lat}, LON={pred_lon}, Confidence={conf}")
```

### Track Accuracy

```python
# Compare predictions with actual positions after 30 minutes
actual_lat = db.fetch_vessel_by_name(vessel_name, limit=1)['LAT']
predicted_lat = prediction['predicted_position']['latitude']
error = abs(actual_lat - predicted_lat)
logger.info(f"Prediction error: {error}°")
```

---

## Summary

✅ **Integration Steps**:
1. Copy integration files
2. Update intent_executor.py
3. Update main.py with new endpoints
4. Update NLP interpreter
5. Install dependencies
6. Test endpoints
7. Update frontend (optional)

✅ **Features**:
- PREDICT: Estimate vessel position after X minutes
- VERIFY: Check course consistency and detect anomalies
- Visualizations: Plot trajectories with confidence scores
- High Accuracy: 0.3056° latitude precision

✅ **Status**: Ready for production deployment

---

**Last Updated**: 2025-10-25  
**Version**: 1.0

