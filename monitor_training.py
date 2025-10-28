"""
Simple training progress monitor
"""

import os
import time
from pathlib import Path
from datetime import datetime

def monitor_training():
    """Monitor training progress."""
    log_file = Path("training_output.log")
    
    if not log_file.exists():
        print("❌ training_output.log not found")
        return
    
    print("\n" + "="*80)
    print("TRAINING PROGRESS MONITOR")
    print("="*80)
    
    while True:
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Get last 50 lines
            recent_lines = lines[-50:]
            
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("\n" + "="*80)
            print(f"TRAINING PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            # Print recent output
            for line in recent_lines:
                print(line.rstrip())
            
            # Check if training is complete
            if any("PIPELINE COMPLETE" in line for line in lines):
                print("\n✅ TRAINING COMPLETE!")
                break
            
            # Check for errors
            if any("ERROR" in line or "Exception" in line for line in recent_lines):
                print("\n❌ ERROR DETECTED!")
                break
            
            print("\n" + "="*80)
            print("Waiting 30 seconds before next update...")
            print("="*80)
            
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    monitor_training()

