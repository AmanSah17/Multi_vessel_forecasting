"""
Test Backend Directly
"""

import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKEND_URL = "http://127.0.0.1:8000"

logger.info("🚀 Testing Backend Directly")

# Test 1: Health check
logger.info("\n1️⃣  Health Check")
r = requests.get(f"{BACKEND_URL}/health")
logger.info(f"Status: {r.status_code}")
logger.info(f"Response: {r.json()}")

# Test 2: Get vessels
logger.info("\n2️⃣  Get Vessels")
r = requests.get(f"{BACKEND_URL}/vessels")
logger.info(f"Status: {r.status_code}")
vessels = r.json().get("vessels", [])
logger.info(f"Vessels: {len(vessels)}")
logger.info(f"Response: {vessels}")

# Test 3: Search vessels
logger.info("\n3️⃣  Search Vessels")
r = requests.get(f"{BACKEND_URL}/vessels/search?q=CHAMPAGNE")
logger.info(f"Status: {r.status_code}")
logger.info(f"Response: {r.json()}")

# Test 4: Admin unique vessels
logger.info("\n4️⃣  Admin Unique Vessels")
r = requests.get(f"{BACKEND_URL}/admin/unique_vessels_df")
logger.info(f"Status: {r.status_code}")
data = r.json()
logger.info(f"Columns: {data.get('columns', [])}")
logger.info(f"Records: {len(data.get('records', []))}")
if data.get('records'):
    logger.info(f"Sample: {data['records'][0]}")

# Test 5: Query with SHOW intent
logger.info("\n5️⃣  Query with SHOW Intent")
query = "Show CHAMPAGNE CHER position"
r = requests.post(
    f"{BACKEND_URL}/query",
    json={"text": query},
    timeout=10
)
logger.info(f"Status: {r.status_code}")
response = r.json()
logger.info(f"Parsed: {response.get('parsed', {})}")
logger.info(f"Response: {response.get('response', {})}")

# Test 6: Query with PREDICT intent
logger.info("\n6️⃣  Query with PREDICT Intent")
query = "Predict CHAMPAGNE CHER position after 30 minutes"
r = requests.post(
    f"{BACKEND_URL}/query",
    json={"text": query},
    timeout=15
)
logger.info(f"Status: {r.status_code}")
response = r.json()
logger.info(f"Parsed: {response.get('parsed', {})}")
result = response.get('response', {})
logger.info(f"Vessel: {result.get('vessel_name', 'N/A')}")
logger.info(f"Last Position: {result.get('last_position', {})}")
logger.info(f"Predicted Position: {result.get('predicted_position', {})}")
logger.info(f"Confidence: {result.get('confidence', 'N/A')}")
logger.info(f"Method: {result.get('method', 'N/A')}")

logger.info("\n✅ All tests completed")

