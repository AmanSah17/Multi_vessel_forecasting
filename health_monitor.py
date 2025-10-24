"""
Real-Time Health Monitor for Maritime NLU + XGBoost Services
Monitors all services and provides real-time status updates
"""

import requests
import time
import sys
from datetime import datetime
from typing import Dict, List
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceMonitor:
    """Monitor health of all services"""
    
    def __init__(self):
        self.services = {
            "XGBoost Model": {
                "url": "http://127.0.0.1:8001/health",
                "port": 8001,
                "timeout": 2
            },
            "Maritime NLU Backend": {
                "url": "http://127.0.0.1:8000/health",
                "port": 8000,
                "timeout": 2
            },
            "Maritime NLU Frontend": {
                "url": "http://127.0.0.1:8502",
                "port": 8502,
                "timeout": 2
            }
        }
        
        self.history = {name: [] for name in self.services.keys()}
        self.start_time = datetime.now()
    
    def check_service(self, name: str, config: Dict) -> Dict:
        """Check if a service is healthy"""
        try:
            response = requests.get(config["url"], timeout=config["timeout"])
            return {
                "status": "✅ HEALTHY",
                "code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "timestamp": datetime.now()
            }
        except requests.exceptions.Timeout:
            return {
                "status": "⏳ TIMEOUT",
                "code": None,
                "response_time": config["timeout"],
                "timestamp": datetime.now()
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "❌ DOWN",
                "code": None,
                "response_time": None,
                "timestamp": datetime.now()
            }
        except Exception as e:
            return {
                "status": f"⚠️  ERROR: {str(e)[:30]}",
                "code": None,
                "response_time": None,
                "timestamp": datetime.now()
            }
    
    def get_model_status(self) -> Dict:
        """Get detailed model status"""
        try:
            response = requests.get("http://127.0.0.1:8000/model/status", timeout=2)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def print_header(self):
        """Print monitor header"""
        print("\n" + "="*80)
        print("🔄 MARITIME NLU + XGBOOST SERVICES - REAL-TIME HEALTH MONITOR")
        print("="*80)
    
    def print_status(self):
        """Print current status of all services"""
        self.print_header()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        uptime = datetime.now() - self.start_time
        uptime_str = f"{uptime.seconds // 3600}h {(uptime.seconds % 3600) // 60}m {uptime.seconds % 60}s"
        
        print(f"\n📊 Timestamp: {timestamp}")
        print(f"⏱️  Uptime: {uptime_str}")
        print("\n" + "-"*80)
        print(f"{'Service':<30} {'Status':<20} {'Response Time':<15} {'Port':<10}")
        print("-"*80)
        
        all_healthy = True
        for name, config in self.services.items():
            result = self.check_service(name, config)
            self.history[name].append(result)
            
            status = result["status"]
            response_time = f"{result['response_time']*1000:.1f}ms" if result['response_time'] else "N/A"
            port = config["port"]
            
            print(f"{name:<30} {status:<20} {response_time:<15} {port:<10}")
            
            if "DOWN" in status or "ERROR" in status:
                all_healthy = False
        
        print("-"*80)
        
        # Model status
        print("\n📈 Model Status:")
        model_status = self.get_model_status()
        if model_status:
            print(f"  • XGBoost Enabled: {model_status.get('xgboost_enabled', False)}")
            print(f"  • Model Type: {model_status.get('model_type', 'N/A')}")
            print(f"  • Features: {model_status.get('features', 'N/A')}")
            print(f"  • PCA Components: {model_status.get('pca_components', 'N/A')}")
            print(f"  • Latitude MAE: {model_status.get('accuracy_lat_mae', 'N/A')}°")
            print(f"  • Longitude MAE: {model_status.get('accuracy_lon_mae', 'N/A')}°")
        else:
            print("  ⚠️  Unable to retrieve model status")
        
        # Summary
        print("\n" + "-"*80)
        if all_healthy:
            print("✅ All services operational")
        else:
            print("⚠️  Some services may need attention")
        
        # Statistics
        print("\n📊 Statistics:")
        for name, history in self.history.items():
            if history:
                healthy_count = sum(1 for h in history if "HEALTHY" in h["status"])
                total_count = len(history)
                uptime_pct = (healthy_count / total_count) * 100
                print(f"  • {name}: {uptime_pct:.1f}% uptime ({healthy_count}/{total_count})")
        
        print("\n" + "="*80 + "\n")
    
    def run(self, interval: int = 5, duration: int = None):
        """Run continuous monitoring"""
        print("\n🚀 Starting health monitor...")
        print(f"📋 Check interval: {interval} seconds")
        if duration:
            print(f"⏱️  Duration: {duration} seconds")
        print("Press Ctrl+C to stop\n")
        
        start_time = time.time()
        
        try:
            while True:
                self.print_status()
                
                if duration and (time.time() - start_time) > duration:
                    print("⏱️  Duration reached. Stopping monitor.")
                    break
                
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Monitor stopped by user")
            self.print_summary()
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*80)
        print("📋 MONITORING SUMMARY")
        print("="*80 + "\n")
        
        for name, history in self.history.items():
            if not history:
                continue
            
            healthy_count = sum(1 for h in history if "HEALTHY" in h["status"])
            down_count = sum(1 for h in history if "DOWN" in h["status"])
            timeout_count = sum(1 for h in history if "TIMEOUT" in h["status"])
            total_count = len(history)
            
            uptime_pct = (healthy_count / total_count) * 100 if total_count > 0 else 0
            
            print(f"📊 {name}:")
            print(f"   • Total Checks: {total_count}")
            print(f"   • Healthy: {healthy_count} ({uptime_pct:.1f}%)")
            print(f"   • Down: {down_count}")
            print(f"   • Timeout: {timeout_count}")
            
            # Average response time
            response_times = [h['response_time'] for h in history if h['response_time']]
            if response_times:
                avg_time = sum(response_times) / len(response_times)
                print(f"   • Avg Response Time: {avg_time*1000:.1f}ms")
            
            print()
        
        print("="*80 + "\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor Maritime NLU + XGBoost services")
    parser.add_argument("--interval", type=int, default=5, help="Check interval in seconds (default: 5)")
    parser.add_argument("--duration", type=int, default=None, help="Monitor duration in seconds (default: infinite)")
    parser.add_argument("--once", action="store_true", help="Check once and exit")
    
    args = parser.parse_args()
    
    monitor = ServiceMonitor()
    
    if args.once:
        monitor.print_status()
    else:
        monitor.run(interval=args.interval, duration=args.duration)


if __name__ == "__main__":
    main()

