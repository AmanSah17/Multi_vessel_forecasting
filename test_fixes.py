"""
Test Script for API Fixes
Tests: Vessel Search, NLP Parsing, Data Fetching, Predictions
"""

import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKEND_URL = "http://127.0.0.1:8000"
XGBOOST_URL = "http://127.0.0.1:8001"

print("\n" + "="*80)
print("TESTING API FIXES - MARITIME NLU SYSTEM")
print("="*80 + "\n")

# ============================================================================
# TEST 1: VESSEL SEARCH
# ============================================================================
print("\n[TEST 1] VESSEL SEARCH")
print("-" * 80)

try:
    logger.info("1a. Fetching all vessels...")
    r = requests.get(f"{BACKEND_URL}/vessels", timeout=5)
    logger.info(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        vessels = r.json().get("vessels", [])
        logger.info(f"[OK] Retrieved {len(vessels)} vessels")
        logger.info(f"Vessels: {vessels[:3]}...")
    else:
        logger.error(f"[FAIL] Failed to get vessels: {r.text}")
    
    logger.info("\n1b. Searching for 'CHAMPAGNE'...")
    r = requests.get(f"{BACKEND_URL}/vessels/search?q=CHAMPAGNE", timeout=5)
    logger.info(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        results = r.json().get("vessels", [])
        logger.info(f"[OK] Search returned {len(results)} results")
        logger.info(f"Results: {results}")
    else:
        logger.error(f"[FAIL] Search failed: {r.text}")
        
except Exception as e:
    logger.error(f"[FAIL] Vessel search error: {e}")

# ============================================================================
# TEST 2: NLP PARSING
# ============================================================================
print("\n\n[TEST 2] NLP PARSING")
print("-" * 80)

test_queries = [
    "Show CHAMPAGNE CHER position",
    "Verify MAERSK SEALAND course",
    "Predict EVER GIVEN position after 30 minutes",
]

for query in test_queries:
    try:
        logger.info(f"\nQuery: '{query}'")
        r = requests.post(
            f"{BACKEND_URL}/query",
            json={"text": query},
            timeout=10
        )
        logger.info(f"Status: {r.status_code}")
        
        if r.status_code == 200:
            data = r.json()
            parsed = data.get("parsed", {})
            logger.info(f"[OK] Parsed successfully")
            logger.info(f"  Intent: {parsed.get('intent')}")
            logger.info(f"  Vessel: {parsed.get('vessel_name')}")
            logger.info(f"  Duration Minutes: {parsed.get('duration_minutes')}")
        else:
            logger.error(f"[FAIL] Parse failed: {r.text}")
            
    except Exception as e:
        logger.error(f"[FAIL] NLP parsing error: {e}")

# ============================================================================
# TEST 3: DATA FETCHING
# ============================================================================
print("\n\n[TEST 3] DATA FETCHING")
print("-" * 80)

try:
    logger.info("Fetching data for CHAMPAGNE CHER...")
    r = requests.post(
        f"{BACKEND_URL}/query",
        json={"text": "Show CHAMPAGNE CHER position"},
        timeout=10
    )
    
    if r.status_code == 200:
        data = r.json()
        response = data.get("response", {})
        
        logger.info(f"[OK] Data fetched successfully")
        logger.info(f"  Vessel Name: {response.get('VesselName')}")
        logger.info(f"  Position: ({response.get('LAT')}, {response.get('LON')})")
        logger.info(f"  SOG: {response.get('SOG')} knots")
        logger.info(f"  COG: {response.get('COG')} degrees")
        logger.info(f"  Track Points: {len(response.get('track', []))}")
    else:
        logger.error(f"[FAIL] Data fetch failed: {r.text}")
        
except Exception as e:
    logger.error(f"[FAIL] Data fetching error: {e}")

# ============================================================================
# TEST 4: PREDICTION API (CRITICAL TEST)
# ============================================================================
print("\n\n[TEST 4] PREDICTION API - CRITICAL TEST")
print("-" * 80)

try:
    logger.info("Testing PREDICT intent...")
    query = "Predict CHAMPAGNE CHER position after 30 minutes"
    
    r = requests.post(
        f"{BACKEND_URL}/query",
        json={"text": query},
        timeout=15
    )
    
    logger.info(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        parsed = data.get("parsed", {})
        response = data.get("response", {})
        
        logger.info(f"[OK] Prediction request successful")
        logger.info(f"\nParsed Intent:")
        logger.info(f"  Intent: {parsed.get('intent')}")
        logger.info(f"  Vessel: {parsed.get('vessel_name')}")
        logger.info(f"  Duration Minutes: {parsed.get('duration_minutes')}")
        
        logger.info(f"\nResponse Data:")
        logger.info(f"  Keys in response: {list(response.keys())}")
        
        # Check for prediction data
        if "predicted_position" in response:
            logger.info(f"[OK] Prediction data found!")
            logger.info(f"  Vessel: {response.get('vessel_name')}")
            logger.info(f"  Last Position: ({response['last_position']['lat']}, {response['last_position']['lon']})")
            logger.info(f"  Predicted Position: ({response['predicted_position']['lat']}, {response['predicted_position']['lon']})")
            logger.info(f"  Minutes Ahead: {response.get('minutes_ahead')}")
            logger.info(f"  Confidence: {response.get('confidence')}")
            logger.info(f"  Method: {response.get('method')}")
        else:
            logger.warning(f"[FAIL] No prediction data in response")
            logger.warning(f"  Response: {response}")
    else:
        logger.error(f"[FAIL] Prediction failed: {r.text}")
        
except Exception as e:
    logger.error(f"[FAIL] Prediction error: {e}")

# ============================================================================
# TEST 5: XGBOOST SERVER
# ============================================================================
print("\n\n[TEST 5] XGBOOST SERVER")
print("-" * 80)

try:
    logger.info("Checking XGBoost server health...")
    r = requests.get(f"{XGBOOST_URL}/health", timeout=5)
    logger.info(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        logger.info(f"[OK] XGBoost server is healthy")
        logger.info(f"Response: {r.json()}")
    else:
        logger.error(f"[FAIL] XGBoost server unhealthy: {r.text}")
    
    logger.info("\nChecking XGBoost model status...")
    r = requests.get(f"{XGBOOST_URL}/model/status", timeout=5)
    logger.info(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        status = r.json()
        logger.info(f"[OK] Model status retrieved")
        logger.info(f"  Is Loaded: {status.get('is_loaded')}")
        logger.info(f"  Has Model: {status.get('has_model')}")
        logger.info(f"  Has Scaler: {status.get('has_scaler')}")
        logger.info(f"  Has PCA: {status.get('has_pca')}")
    else:
        logger.error(f"[FAIL] Model status check failed: {r.text}")
        
except Exception as e:
    logger.error(f"[FAIL] XGBoost server error: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\nSUMMARY:")
print("  1. Vessel Search: Check if working")
print("  2. NLP Parsing: Check if duration_minutes extracted")
print("  3. Data Fetching: Check if vessel data retrieved")
print("  4. PREDICTION API: Check if predictions returned with correct structure")
print("  5. XGBoost Server: Check if model loaded")
print("\n")

