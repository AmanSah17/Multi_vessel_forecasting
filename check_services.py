"""
Check service health
"""
import requests
import time
import sys

def check_service(name, url, timeout=5):
    """Check if a service is running"""
    try:
        r = requests.get(url, timeout=timeout)
        status = "✅ HEALTHY" if r.status_code == 200 else f"⚠️ Status {r.status_code}"
        print(f"{name}: {status}")
        return True
    except Exception as e:
        print(f"{name}: ❌ ERROR - {str(e)[:50]}")
        return False

print("\n" + "="*60)
print("🚀 SERVICE HEALTH CHECK")
print("="*60 + "\n")

services = {
    "Backend API": "http://127.0.0.1:8000/health",
    "XGBoost Server": "http://127.0.0.1:8001/health",
}

print("Waiting for services to start...")
time.sleep(3)

all_healthy = True
for name, url in services.items():
    if not check_service(name, url):
        all_healthy = False

print("\n" + "="*60)
if all_healthy:
    print("✅ ALL SERVICES HEALTHY")
    print("\n📍 Service URLs:")
    print("   Frontend:     http://127.0.0.1:8502")
    print("   Backend API:  http://127.0.0.1:8000")
    print("   Backend Docs: http://127.0.0.1:8000/docs")
    print("   XGBoost:      http://127.0.0.1:8001")
else:
    print("⚠️ SOME SERVICES NOT READY - Waiting...")
    time.sleep(5)
    print("\nRetrying...")
    for name, url in services.items():
        check_service(name, url)

print("="*60 + "\n")

