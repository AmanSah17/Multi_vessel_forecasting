"""
Simple XGBoost Test
"""

import sys
sys.path.insert(0, r"F:\Maritime_NLU_Repo\backend\nlu_chatbot\src\app")

from xgboost_backend_integration import XGBoostBackendPredictor
from db_handler import MaritimeDB

print("=" * 80)
print("TESTING XGBOOST PREDICTION")
print("=" * 80)

# Initialize predictor
print("\n[1] Initializing XGBoost predictor...")
predictor = XGBoostBackendPredictor()
print(f"    Status: {predictor.get_status()}")

# Load database
print("\n[2] Loading database...")
db = MaritimeDB(r"f:\Maritime_NLU_Repo\backend\nlu_chatbot\maritime_sample_0104.db")

# Fetch sample data
print("\n[3] Fetching vessel data...")
df = db.fetch_vessel_by_name("CHAMPAGNE CHER", limit=20)
print(f"    Records: {len(df)}")

# Test prediction
print("\n[4] Testing XGBoost prediction...")
result = predictor.predict(df)

if result:
    print(f"    SUCCESS!")
    print(f"    Predicted LAT: {result['predicted_lat']:.4f}")
    print(f"    Predicted LON: {result['predicted_lon']:.4f}")
    print(f"    Predicted SOG: {result['predicted_sog']:.2f}")
    print(f"    Predicted COG: {result['predicted_cog']:.2f}")
    print(f"    Confidence: {result['confidence']:.2f}")
else:
    print(f"    FAILED - No result returned")

print("\n" + "=" * 80)

