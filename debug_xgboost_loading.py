"""
Debug XGBoost Model Loading
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np

# Add backend path
sys.path.insert(0, r"F:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app")

print("=" * 80)
print("DEBUGGING XGBOOST MODEL LOADING")
print("=" * 80)

# Check if model files exist
model_dir = r"f:\PyTorch_GPU\maritime_vessel_forecasting\Multi_vessel_forecasting\results\xgboost_advanced_50_vessels"

print(f"\n[1] Checking model directory: {model_dir}")
print(f"    Exists: {os.path.exists(model_dir)}")

files_to_check = ["xgboost_model.pkl", "scaler.pkl", "pca.pkl"]
for file in files_to_check:
    path = os.path.join(model_dir, file)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f"    {file}: {exists} ({size} bytes)")

# Try to load the model
print(f"\n[2] Loading XGBoost model...")
try:
    model_path = os.path.join(model_dir, "xgboost_model.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"    ✅ Model loaded successfully")
    print(f"    Type: {type(model)}")
except Exception as e:
    print(f"    ❌ Failed to load model: {e}")
    model = None

# Try to load scaler
print(f"\n[3] Loading StandardScaler...")
try:
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"    ✅ Scaler loaded successfully")
    print(f"    Type: {type(scaler)}")
except Exception as e:
    print(f"    ❌ Failed to load scaler: {e}")
    scaler = None

# Try to load PCA
print(f"\n[4] Loading PCA...")
try:
    pca_path = os.path.join(model_dir, "pca.pkl")
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    print(f"    ✅ PCA loaded successfully")
    print(f"    Type: {type(pca)}")
    print(f"    Components: {pca.n_components_}")
except Exception as e:
    print(f"    ❌ Failed to load PCA: {e}")
    pca = None

# Try to initialize XGBoostBackendPredictor
print(f"\n[5] Initializing XGBoostBackendPredictor...")
try:
    from xgboost_backend_integration import XGBoostBackendPredictor
    predictor = XGBoostBackendPredictor(model_dir=model_dir)
    print(f"    ✅ Predictor initialized")
    print(f"    Is loaded: {predictor.is_loaded}")
    print(f"    Status: {predictor.get_status()}")
except Exception as e:
    print(f"    ❌ Failed to initialize predictor: {e}")
    import traceback
    traceback.print_exc()

# Try to make a test prediction
print(f"\n[6] Testing prediction with sample data...")
try:
    from db_handler import MaritimeDB
    db = MaritimeDB(r"f:\Maritime_NLU_Repo\backend\nlu_chatbot\maritime_sample_0104.db")
    
    # Fetch sample data
    df = db.fetch_vessel_by_name("CHAMPAGNE CHER", limit=20)
    print(f"    Fetched {len(df)} records")
    
    if not df.empty and predictor.is_loaded:
        result = predictor.predict(df)
        print(f"    ✅ Prediction successful")
        print(f"    Result: {result}")
    else:
        print(f"    ❌ Cannot predict: df empty={df.empty}, predictor loaded={predictor.is_loaded}")
        
except Exception as e:
    print(f"    ❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)

