"""
Quick status check for training progress
"""

import os
from pathlib import Path
from datetime import datetime

def check_training_status():
    """Check current training status."""
    log_file = Path("training_output.log")
    
    print("\n" + "="*80)
    print("TRAINING STATUS CHECK")
    print("="*80)
    
    if not log_file.exists():
        print("❌ training_output.log not found - training may not have started")
        return
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Get last 30 lines
    recent_lines = lines[-30:]
    
    print("\nLast 30 lines of training output:")
    print("-" * 80)
    for line in recent_lines:
        print(line.rstrip())
    
    print("\n" + "="*80)
    
    # Check for key milestones
    all_text = ''.join(lines)
    
    if "PIPELINE COMPLETE" in all_text:
        print("✅ TRAINING COMPLETE!")
    elif "Hyperparameter tuning" in all_text:
        print("⏳ Currently in hyperparameter tuning phase")
    elif "Extracting features for test" in all_text:
        print("⏳ Currently extracting test features")
    elif "Extracting features for train" in all_text:
        print("⏳ Currently extracting training features")
    elif "Loading and splitting data" in all_text:
        print("⏳ Currently loading and splitting data")
    else:
        print("❓ Training status unclear")
    
    # Check for errors
    if "ERROR" in all_text or "Exception" in all_text or "Traceback" in all_text:
        print("\n⚠️  ERRORS DETECTED - Check logs for details")
    else:
        print("\n✅ No errors detected so far")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    check_training_status()

