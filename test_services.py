"""
Comprehensive Test Suite for Maritime NLU + XGBoost Services
Tests all endpoints and functionality
"""

import requests
import json
import time
from datetime import datetime
import sys
from typing import Dict, List


class ServiceTester:
    """Test all services"""
    
    def __init__(self):
        self.backend_url = "http://127.0.0.1:8000"
        self.xgboost_url = "http://127.0.0.1:8001"
        self.frontend_url = "http://127.0.0.1:8502"
        
        self.results = {
            "passed": 0,
            "failed": 0,
            "errors": []
        }
    
    def print_header(self, text: str):
        """Print test header"""
        print("\n" + "="*80)
        print(f"🧪 {text}")
        print("="*80)
    
    def print_test(self, name: str, passed: bool, message: str = ""):
        """Print test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if message:
            print(f"     {message}")
        
        if passed:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
            self.results["errors"].append(f"{name}: {message}")
    
    # ========================================================================
    # HEALTH CHECKS
    # ========================================================================
    
    def test_backend_health(self):
        """Test backend health endpoint"""
        self.print_header("Backend Health Check")
        
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            passed = response.status_code == 200
            self.print_test(
                "Backend Health",
                passed,
                f"Status: {response.status_code}"
            )
            return passed
        except Exception as e:
            self.print_test("Backend Health", False, str(e))
            return False
    
    def test_xgboost_health(self):
        """Test XGBoost health endpoint"""
        self.print_header("XGBoost Health Check")
        
        try:
            response = requests.get(f"{self.xgboost_url}/health", timeout=5)
            passed = response.status_code == 200
            self.print_test(
                "XGBoost Health",
                passed,
                f"Status: {response.status_code}"
            )
            return passed
        except Exception as e:
            self.print_test("XGBoost Health", False, str(e))
            return False
    
    def test_frontend_health(self):
        """Test frontend health"""
        self.print_header("Frontend Health Check")
        
        try:
            response = requests.get(self.frontend_url, timeout=5)
            passed = response.status_code == 200
            self.print_test(
                "Frontend Health",
                passed,
                f"Status: {response.status_code}"
            )
            return passed
        except Exception as e:
            self.print_test("Frontend Health", False, str(e))
            return False
    
    # ========================================================================
    # MODEL STATUS
    # ========================================================================
    
    def test_model_status(self):
        """Test model status endpoint"""
        self.print_header("Model Status")
        
        try:
            response = requests.get(f"{self.backend_url}/model/status", timeout=5)
            passed = response.status_code == 200
            
            if passed:
                data = response.json()
                print(f"     XGBoost Enabled: {data.get('xgboost_enabled')}")
                print(f"     Model Type: {data.get('model_type')}")
                print(f"     Features: {data.get('features')}")
                print(f"     PCA Components: {data.get('pca_components')}")
            
            self.print_test("Model Status", passed)
            return passed
        except Exception as e:
            self.print_test("Model Status", False, str(e))
            return False
    
    # ========================================================================
    # VESSEL QUERIES
    # ========================================================================
    
    def test_show_vessels(self):
        """Test SHOW intent"""
        self.print_header("Vessel Query Tests")
        
        test_queries = [
            "Show CHAMPAGNE CHER",
            "Show vessel CHAMPAGNE CHER",
            "What is the position of CHAMPAGNE CHER"
        ]
        
        for query in test_queries:
            try:
                response = requests.post(
                    f"{self.backend_url}/query",
                    json={"text": query},
                    timeout=5
                )
                passed = response.status_code == 200
                self.print_test(f"Query: {query}", passed)
            except Exception as e:
                self.print_test(f"Query: {query}", False, str(e))
    
    # ========================================================================
    # PREDICTIONS
    # ========================================================================
    
    def test_predict_position(self):
        """Test PREDICT intent"""
        self.print_header("Prediction Tests")
        
        test_queries = [
            "Predict CHAMPAGNE CHER position after 30 minutes",
            "Where will CHAMPAGNE CHER be in 60 minutes",
            "Forecast CHAMPAGNE CHER location after 15 minutes"
        ]
        
        for query in test_queries:
            try:
                response = requests.post(
                    f"{self.backend_url}/query",
                    json={"text": query},
                    timeout=10
                )
                passed = response.status_code == 200
                
                if passed:
                    data = response.json()
                    has_prediction = "predicted_position" in data.get("response", {})
                    self.print_test(
                        f"Predict: {query[:40]}...",
                        has_prediction,
                        f"Response time: {response.elapsed.total_seconds():.2f}s"
                    )
                else:
                    self.print_test(f"Predict: {query[:40]}...", False)
            except Exception as e:
                self.print_test(f"Predict: {query[:40]}...", False, str(e))
    
    # ========================================================================
    # VERIFICATION
    # ========================================================================
    
    def test_verify_course(self):
        """Test VERIFY intent"""
        self.print_header("Verification Tests")
        
        test_queries = [
            "Verify CHAMPAGNE CHER course",
            "Check CHAMPAGNE CHER movement",
            "Is CHAMPAGNE CHER course consistent"
        ]
        
        for query in test_queries:
            try:
                response = requests.post(
                    f"{self.backend_url}/query",
                    json={"text": query},
                    timeout=10
                )
                passed = response.status_code == 200
                self.print_test(f"Verify: {query[:40]}...", passed)
            except Exception as e:
                self.print_test(f"Verify: {query[:40]}...", False, str(e))
    
    # ========================================================================
    # PERFORMANCE TESTS
    # ========================================================================
    
    def test_response_times(self):
        """Test response times"""
        self.print_header("Performance Tests")
        
        # Health check response time
        try:
            start = time.time()
            requests.get(f"{self.backend_url}/health", timeout=5)
            elapsed = time.time() - start
            passed = elapsed < 0.5
            self.print_test(
                "Health Check Response Time",
                passed,
                f"{elapsed*1000:.1f}ms (target: <500ms)"
            )
        except Exception as e:
            self.print_test("Health Check Response Time", False, str(e))
        
        # Prediction response time
        try:
            start = time.time()
            requests.post(
                f"{self.backend_url}/query",
                json={"text": "Predict CHAMPAGNE CHER position after 30 minutes"},
                timeout=10
            )
            elapsed = time.time() - start
            passed = elapsed < 5.0
            self.print_test(
                "Prediction Response Time",
                passed,
                f"{elapsed*1000:.1f}ms (target: <5000ms)"
            )
        except Exception as e:
            self.print_test("Prediction Response Time", False, str(e))
    
    # ========================================================================
    # LOAD TESTS
    # ========================================================================
    
    def test_concurrent_requests(self, num_requests: int = 5):
        """Test concurrent requests"""
        self.print_header(f"Load Test ({num_requests} concurrent requests)")
        
        import concurrent.futures
        
        def make_request():
            try:
                response = requests.post(
                    f"{self.backend_url}/query",
                    json={"text": "Show CHAMPAGNE CHER"},
                    timeout=10
                )
                return response.status_code == 200
            except:
                return False
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
                results = list(executor.map(make_request, range(num_requests)))
            
            passed_count = sum(results)
            passed = passed_count == num_requests
            self.print_test(
                f"Concurrent Requests",
                passed,
                f"{passed_count}/{num_requests} successful"
            )
        except Exception as e:
            self.print_test("Concurrent Requests", False, str(e))
    
    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================
    
    def run_all(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("🚀 MARITIME NLU + XGBOOST SERVICES - COMPREHENSIVE TEST SUITE")
        print("="*80)
        
        # Health checks
        self.test_backend_health()
        self.test_xgboost_health()
        self.test_frontend_health()
        
        # Model status
        self.test_model_status()
        
        # Queries
        self.test_show_vessels()
        self.test_predict_position()
        self.test_verify_course()
        
        # Performance
        self.test_response_times()
        self.test_concurrent_requests(5)
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        
        total = self.results["passed"] + self.results["failed"]
        pass_rate = (self.results["passed"] / total * 100) if total > 0 else 0
        
        print(f"\n✅ Passed: {self.results['passed']}")
        print(f"❌ Failed: {self.results['failed']}")
        print(f"📊 Total: {total}")
        print(f"📈 Pass Rate: {pass_rate:.1f}%")
        
        if self.results["errors"]:
            print("\n⚠️  Errors:")
            for error in self.results["errors"]:
                print(f"   • {error}")
        
        print("\n" + "="*80 + "\n")
        
        return self.results["failed"] == 0


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Maritime NLU + XGBoost services")
    parser.add_argument("--health-only", action="store_true", help="Only run health checks")
    parser.add_argument("--load-test", type=int, default=5, help="Number of concurrent requests for load test")
    
    args = parser.parse_args()
    
    tester = ServiceTester()
    
    if args.health_only:
        tester.test_backend_health()
        tester.test_xgboost_health()
        tester.test_frontend_health()
    else:
        success = tester.run_all()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

