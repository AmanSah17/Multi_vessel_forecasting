"""
Comprehensive Debug Script for API Issues
Tests: Vessel Search, NLP Parsing, Data Fetching, Predictions
"""

import requests
import json
import logging
from datetime import datetime
import sys
import io

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BACKEND_URL = "http://127.0.0.1:8000"
XGBOOST_URL = "http://127.0.0.1:8001"

print("\n" + "="*80)
print("[DEBUG] COMPREHENSIVE API DEBUG - MARITIME NLU SYSTEM")
print("="*80 + "\n")

# ============================================================================
# TEST 1: VESSEL SEARCH
# ============================================================================
print("\n[TEST 1] VESSEL SEARCH")
print("-" * 80)

try:
    # Test 1a: Get all vessels
    logger.info("1a. Fetching all vessels...")
    r = requests.get(f"{BACKEND_URL}/vessels", timeout=5)
    logger.info(f"Status: {r.status_code}")

    if r.status_code == 200:
        vessels = r.json().get("vessels", [])
        logger.info(f"[OK] Retrieved {len(vessels)} vessels")
        logger.info(f"Vessels: {vessels[:3]}...")
    else:
        logger.error(f"[FAIL] Failed to get vessels: {r.text}")

    # Test 1b: Search vessels
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
print("\n\n📝 TEST 2: NLP PARSING")
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
            logger.info(f"✅ Parsed successfully")
            logger.info(f"  Intent: {parsed.get('intent')}")
            logger.info(f"  Vessel: {parsed.get('vessel_name')}")
            logger.info(f"  Time Horizon: {parsed.get('time_horizon')}")
            logger.info(f"  Duration Minutes: {parsed.get('duration_minutes')}")
        else:
            logger.error(f"❌ Parse failed: {r.text}")
            
    except Exception as e:
        logger.error(f"❌ NLP parsing error: {e}")

# ============================================================================
# TEST 3: DATA FETCHING
# ============================================================================
print("\n\n📊 TEST 3: DATA FETCHING")
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
        
        logger.info(f"✅ Data fetched successfully")
        logger.info(f"  Vessel Name: {response.get('VesselName')}")
        logger.info(f"  Position: ({response.get('LAT')}, {response.get('LON')})")
        logger.info(f"  SOG: {response.get('SOG')} knots")
        logger.info(f"  COG: {response.get('COG')} degrees")
        logger.info(f"  DateTime: {response.get('BaseDateTime')}")
        logger.info(f"  Track Points: {len(response.get('track', []))}")
        
        if response.get('track'):
            logger.info(f"  First track point: {response['track'][0]}")
    else:
        logger.error(f"❌ Data fetch failed: {r.text}")
        
except Exception as e:
    logger.error(f"❌ Data fetching error: {e}")

# ============================================================================
# TEST 4: PREDICTION API
# ============================================================================
print("\n\n🎯 TEST 4: PREDICTION API")
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
        
        logger.info(f"✅ Prediction request successful")
        logger.info(f"\nParsed Intent:")
        logger.info(f"  Intent: {parsed.get('intent')}")
        logger.info(f"  Vessel: {parsed.get('vessel_name')}")
        logger.info(f"  Time Horizon: {parsed.get('time_horizon')}")
        logger.info(f"  Duration Minutes: {parsed.get('duration_minutes')}")
        
        logger.info(f"\nResponse Data:")
        logger.info(f"  Keys in response: {list(response.keys())}")
        
        # Check for prediction data
        if "Predicted_LAT" in response:
            logger.info(f"✅ Prediction data found!")
            logger.info(f"  Vessel: {response.get('VesselName')}")
            logger.info(f"  Last Position: ({response.get('LAT')}, {response.get('LON')})")
            logger.info(f"  Predicted Position: ({response.get('Predicted_LAT')}, {response.get('Predicted_LON')})")
            logger.info(f"  Minutes Ahead: {response.get('MinutesAhead')}")
        elif "predicted_position" in response:
            logger.info(f"✅ Prediction data found (nested)!")
            logger.info(f"  {response}")
        else:
            logger.warning(f"⚠️  No prediction data in response")
            logger.warning(f"  Response: {response}")
    else:
        logger.error(f"❌ Prediction failed: {r.text}")
        
except Exception as e:
    logger.error(f"❌ Prediction error: {e}")

# ============================================================================
# TEST 5: XGBOOST SERVER
# ============================================================================
print("\n\n🤖 TEST 5: XGBOOST SERVER")
print("-" * 80)

try:
    logger.info("Checking XGBoost server health...")
    r = requests.get(f"{XGBOOST_URL}/health", timeout=5)
    logger.info(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        logger.info(f"✅ XGBoost server is healthy")
        logger.info(f"Response: {r.json()}")
    else:
        logger.error(f"❌ XGBoost server unhealthy: {r.text}")
    
    logger.info("\nChecking XGBoost model status...")
    r = requests.get(f"{XGBOOST_URL}/model/status", timeout=5)
    logger.info(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        status = r.json()
        logger.info(f"✅ Model status retrieved")
        logger.info(f"  Is Loaded: {status.get('is_loaded')}")
        logger.info(f"  Model Info: {status}")
    else:
        logger.error(f"❌ Model status check failed: {r.text}")
        
except Exception as e:
    logger.error(f"❌ XGBoost server error: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "="*80)
print("✅ DEBUG COMPLETE")
print("="*80)
print("\n📌 ISSUES FOUND:")
print("  1. Check if PREDICT intent returns correct response structure")
print("  2. Verify NLP parser extracts duration_minutes correctly")
print("  3. Check if XGBoost predictions are being called")
print("  4. Verify response formatting for frontend compatibility")
print("\n")

