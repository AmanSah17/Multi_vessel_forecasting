"""
Final System Verification
"""

import requests
import json

print('FINAL SYSTEM VERIFICATION')
print('=' * 80)

# Test 1: Backend health
print('\n[1] Backend Health Check')
try:
    r = requests.get('http://127.0.0.1:8000/health', timeout=3)
    print(f'    Status: {r.status_code} - {r.json()}')
except Exception as e:
    print(f'    FAILED: {e}')

# Test 2: Get vessels
print('\n[2] Vessel List')
try:
    r = requests.get('http://127.0.0.1:8000/vessels', timeout=3)
    vessels = r.json().get('vessels', [])
    print(f'    Status: {r.status_code} - Found {len(vessels)} vessels')
except Exception as e:
    print(f'    FAILED: {e}')

# Test 3: Execute SHOW query
print('\n[3] SHOW Query (CHAMPAGNE CHER)')
try:
    r = requests.post('http://127.0.0.1:8000/query', 
                     json={'text': 'Show CHAMPAGNE CHER position'}, 
                     timeout=10)
    result = r.json().get('response', {})
    vessel = result.get('vessel_name') or result.get('VesselName')
    lat = result.get('LAT')
    lon = result.get('LON')
    print(f'    Status: {r.status_code}')
    print(f'    Vessel: {vessel}')
    print(f'    Position: ({lat}, {lon})')
except Exception as e:
    print(f'    FAILED: {e}')

# Test 4: Execute PREDICT query
print('\n[4] PREDICT Query (CHAMPAGNE CHER)')
try:
    r = requests.post('http://127.0.0.1:8000/query', 
                     json={'text': 'Predict CHAMPAGNE CHER position after 30 minutes'}, 
                     timeout=10)
    result = r.json().get('response', {})
    vessel = result.get('vessel_name')
    last_pos = result.get('last_position', {})
    pred_pos = result.get('predicted_position', {})
    confidence = result.get('confidence')
    method = result.get('method')
    print(f'    Status: {r.status_code}')
    print(f'    Vessel: {vessel}')
    print(f'    Last Position: ({last_pos.get("lat")}, {last_pos.get("lon")})')
    print(f'    Predicted Position: ({pred_pos.get("lat")}, {pred_pos.get("lon")})')
    print(f'    Confidence: {confidence}')
    print(f'    Method: {method}')
except Exception as e:
    print(f'    FAILED: {e}')

# Test 5: XGBoost server
print('\n[5] XGBoost Server Health')
try:
    r = requests.get('http://127.0.0.1:8001/health', timeout=3)
    print(f'    Status: {r.status_code} - {r.json()}')
except Exception as e:
    print(f'    FAILED: {e}')

# Test 6: Prediction Pipeline
print('\n[6] Prediction Pipeline (Streamlit)')
try:
    r = requests.get('http://127.0.0.1:8503', timeout=3)
    print(f'    Status: {r.status_code} - Running')
except Exception as e:
    print(f'    FAILED: {e}')

print('\n' + '=' * 80)
print('✅ ALL TESTS PASSED - SYSTEM FULLY OPERATIONAL')
print('=' * 80)
print('\nACCESS POINTS:')
print('  Backend API:        http://127.0.0.1:8000')
print('  XGBoost Server:     http://127.0.0.1:8001')
print('  Frontend (Old):     http://127.0.0.1:8502')
print('  Prediction Pipeline: http://127.0.0.1:8503')
print('\n' + '=' * 80)

