"""
Comprehensive Backend Integration Test
Tests the complete pipeline with real database queries
"""

import requests
import json
import logging
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Service URLs
BACKEND_URL = "http://127.0.0.1:8000"
XGBOOST_URL = "http://127.0.0.1:8001"
FRONTEND_URL = "http://127.0.0.1:8502"

class BackendIntegrationTester:
    """Test backend integration with XGBoost"""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def test_service_health(self):
        """Test all services are running"""
        logger.info("\n" + "="*80)
        logger.info("1. TESTING SERVICE HEALTH")
        logger.info("="*80)
        
        services = {
            "Backend API": f"{BACKEND_URL}/health",
            "XGBoost Server": f"{XGBOOST_URL}/health",
            "Frontend": FRONTEND_URL
        }
        
        for name, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ {name}: HEALTHY (Status: {response.status_code})")
                    self.passed += 1
                else:
                    logger.error(f"❌ {name}: UNHEALTHY (Status: {response.status_code})")
                    self.failed += 1
            except Exception as e:
                logger.error(f"❌ {name}: ERROR - {e}")
                self.failed += 1
    
    def test_get_vessels(self):
        """Get list of vessels from database"""
        logger.info("\n" + "="*80)
        logger.info("2. FETCHING VESSELS FROM DATABASE")
        logger.info("="*80)
        
        try:
            response = requests.get(f"{BACKEND_URL}/vessels", timeout=10)
            if response.status_code == 200:
                vessels = response.json().get("vessels", [])
                logger.info(f"✅ Retrieved {len(vessels)} unique vessels")
                
                # Show first 10 vessels
                logger.info("\n📋 Sample Vessels:")
                for i, vessel in enumerate(vessels[:10], 1):
                    logger.info(f"   {i}. {vessel}")
                
                self.passed += 1
                return vessels[:10]  # Return first 10 for testing
            else:
                logger.error(f"❌ Failed to get vessels: {response.status_code}")
                self.failed += 1
                return []
        except Exception as e:
            logger.error(f"❌ Error fetching vessels: {e}")
            self.failed += 1
            return []
    
    def test_show_vessel(self, vessel_name):
        """Test SHOW intent"""
        logger.info(f"\n📍 Testing SHOW intent for: {vessel_name}")
        
        try:
            query = f"Show {vessel_name} position"
            response = requests.post(
                f"{BACKEND_URL}/query",
                json={"text": query},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("response", {})
                
                if "error" not in result:
                    logger.info(f"   ✅ Vessel: {result.get('VesselName', 'N/A')}")
                    logger.info(f"   📍 Position: ({result.get('LAT', 'N/A')}, {result.get('LON', 'N/A')})")
                    logger.info(f"   ⚓ SOG: {result.get('SOG', 'N/A')} kts, COG: {result.get('COG', 'N/A')}°")
                    logger.info(f"   🕐 Time: {result.get('BaseDateTime', 'N/A')}")
                    
                    # Show track points
                    track = result.get('track', [])
                    logger.info(f"   📊 Track points: {len(track)}")
                    
                    self.passed += 1
                    return result
                else:
                    logger.warning(f"   ⚠️  {result.get('error', 'Unknown error')}")
                    self.failed += 1
                    return None
            else:
                logger.error(f"   ❌ Request failed: {response.status_code}")
                self.failed += 1
                return None
                
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
            self.failed += 1
            return None
    
    def test_predict_position(self, vessel_name, minutes=30):
        """Test PREDICT intent with XGBoost"""
        logger.info(f"\n🔮 Testing PREDICT intent for: {vessel_name} (+{minutes} min)")
        
        try:
            query = f"Predict {vessel_name} position after {minutes} minutes"
            response = requests.post(
                f"{BACKEND_URL}/query",
                json={"text": query},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("response", {})
                
                if "error" not in result:
                    logger.info(f"   ✅ Vessel: {result.get('vessel_name', 'N/A')}")
                    
                    last_pos = result.get('last_position', {})
                    pred_pos = result.get('predicted_position', {})
                    
                    logger.info(f"   📍 Last Position: ({last_pos.get('lat', 'N/A')}, {last_pos.get('lon', 'N/A')})")
                    logger.info(f"   🎯 Predicted Position: ({pred_pos.get('lat', 'N/A')}, {pred_pos.get('lon', 'N/A')})")
                    logger.info(f"   📊 Confidence: {result.get('confidence', 'N/A')*100:.1f}%")
                    logger.info(f"   🔧 Method: {result.get('method', 'N/A')}")
                    
                    # Show trajectory points
                    trajectory = result.get('trajectory_points', [])
                    logger.info(f"   📈 Trajectory points: {len(trajectory)}")
                    
                    self.passed += 1
                    return result
                else:
                    logger.warning(f"   ⚠️  {result.get('error', 'Unknown error')}")
                    self.failed += 1
                    return None
            else:
                logger.error(f"   ❌ Request failed: {response.status_code}")
                self.failed += 1
                return None
                
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
            self.failed += 1
            return None
    
    def test_verify_course(self, vessel_name):
        """Test VERIFY intent"""
        logger.info(f"\n✓ Testing VERIFY intent for: {vessel_name}")
        
        try:
            query = f"Verify {vessel_name} course"
            response = requests.post(
                f"{BACKEND_URL}/query",
                json={"text": query},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("response", {})
                
                if "error" not in result:
                    is_consistent = result.get('is_consistent', False)
                    status = "✅ CONSISTENT" if is_consistent else "⚠️  ANOMALY DETECTED"
                    
                    logger.info(f"   {status}")
                    logger.info(f"   COG Variance: {result.get('cog_variance', 'N/A'):.2f}°")
                    logger.info(f"   SOG Variance: {result.get('sog_variance', 'N/A'):.2f} kts")
                    
                    last_3 = result.get('last_3_points', [])
                    logger.info(f"   📊 Last 3 points analyzed: {len(last_3)}")
                    
                    self.passed += 1
                    return result
                else:
                    logger.warning(f"   ⚠️  {result.get('error', 'Unknown error')}")
                    self.failed += 1
                    return None
            else:
                logger.error(f"   ❌ Request failed: {response.status_code}")
                self.failed += 1
                return None
                
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
            self.failed += 1
            return None
    
    def test_xgboost_model_status(self):
        """Test XGBoost model status"""
        logger.info("\n" + "="*80)
        logger.info("3. TESTING XGBOOST MODEL STATUS")
        logger.info("="*80)
        
        try:
            response = requests.get(f"{XGBOOST_URL}/model/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                logger.info(f"✅ Model Status:")
                logger.info(f"   Loaded: {status.get('is_loaded', False)}")
                logger.info(f"   Has Model: {status.get('has_model', False)}")
                logger.info(f"   Has Scaler: {status.get('has_scaler', False)}")
                logger.info(f"   Has PCA: {status.get('has_pca', False)}")
                self.passed += 1
            else:
                logger.error(f"❌ Failed to get model status: {response.status_code}")
                self.failed += 1
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            self.failed += 1
    
    def run_all_tests(self):
        """Run all tests"""
        logger.info("\n" + "="*80)
        logger.info("🚀 STARTING COMPREHENSIVE BACKEND INTEGRATION TESTS")
        logger.info("="*80)
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test service health
        self.test_service_health()
        time.sleep(1)
        
        # Test XGBoost model status
        self.test_xgboost_model_status()
        time.sleep(1)
        
        # Get vessels
        vessels = self.test_get_vessels()
        time.sleep(1)
        
        if vessels:
            logger.info("\n" + "="*80)
            logger.info("4. TESTING INTENTS WITH SAMPLE VESSELS")
            logger.info("="*80)
            
            # Test each vessel with all intents
            for vessel in vessels[:3]:  # Test first 3 vessels
                logger.info(f"\n{'='*80}")
                logger.info(f"Testing Vessel: {vessel}")
                logger.info(f"{'='*80}")
                
                # SHOW intent
                show_result = self.test_show_vessel(vessel)
                time.sleep(1)
                
                # VERIFY intent
                verify_result = self.test_verify_course(vessel)
                time.sleep(1)
                
                # PREDICT intent
                predict_result = self.test_predict_position(vessel, minutes=30)
                time.sleep(1)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "="*80)
        logger.info("📊 TEST SUMMARY")
        logger.info("="*80)
        logger.info(f"✅ Passed: {self.passed}")
        logger.info(f"❌ Failed: {self.failed}")
        logger.info(f"📈 Success Rate: {(self.passed/(self.passed+self.failed)*100):.1f}%" if (self.passed+self.failed) > 0 else "N/A")
        logger.info("="*80)


if __name__ == "__main__":
    tester = BackendIntegrationTester()
    tester.run_all_tests()

