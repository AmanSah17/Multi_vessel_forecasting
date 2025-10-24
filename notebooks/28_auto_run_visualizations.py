"""
Auto-run visualizations after training completes
Monitors the training log and runs visualizations when training is done
"""

import time
import subprocess
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

log_file = Path('logs/comprehensive_pipeline.log')
viz_script = Path('notebooks/27_comprehensive_visualizations.py')


def check_training_complete():
    """Check if training has completed by looking for completion markers in log."""
    if not log_file.exists():
        return False
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Check for completion markers
    completion_markers = [
        'PIPELINE COMPLETE!',
        'Model Performance:',
        'All predictions and visualizations generated'
    ]
    
    return any(marker in content for marker in completion_markers)


def wait_for_training():
    """Wait for training to complete."""
    logger.info("Waiting for training to complete...")
    logger.info(f"Monitoring: {log_file}")
    
    check_interval = 60  # Check every 60 seconds
    max_wait_time = 14400  # 4 hours max
    elapsed_time = 0
    
    while elapsed_time < max_wait_time:
        if check_training_complete():
            logger.info("✅ Training completed!")
            return True
        
        logger.info(f"Training in progress... ({elapsed_time//60} minutes elapsed)")
        time.sleep(check_interval)
        elapsed_time += check_interval
    
    logger.warning(f"⚠️ Timeout: Training did not complete within {max_wait_time//3600} hours")
    return False


def run_visualizations():
    """Run the visualization script."""
    logger.info("\n" + "="*80)
    logger.info("STARTING VISUALIZATION GENERATION")
    logger.info("="*80)
    
    if not viz_script.exists():
        logger.error(f"Visualization script not found: {viz_script}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(viz_script)],
            capture_output=False,
            text=True,
            timeout=3600  # 1 hour timeout for visualizations
        )
        
        if result.returncode == 0:
            logger.info("✅ Visualizations generated successfully!")
            return True
        else:
            logger.error(f"❌ Visualization script failed with return code {result.returncode}")
            return False
    
    except subprocess.TimeoutExpired:
        logger.error("❌ Visualization generation timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Error running visualizations: {e}")
        return False


def main():
    """Main function."""
    logger.info("\n" + "="*80)
    logger.info("AUTO-RUN VISUALIZATION MONITOR")
    logger.info("="*80)
    
    # Wait for training
    if wait_for_training():
        # Run visualizations
        if run_visualizations():
            logger.info("\n" + "="*80)
            logger.info("✅ ALL TASKS COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info("\nResults saved to:")
            logger.info("  - results/csv/vessel_predictions_300_detailed.csv")
            logger.info("  - results/csv/model_comparison_comprehensive.csv")
            logger.info("  - results/images/")
            logger.info("\nOpen results/images/ to view all visualizations")
            return 0
        else:
            logger.error("Visualization generation failed")
            return 1
    else:
        logger.error("Training did not complete")
        return 1


if __name__ == "__main__":
    sys.exit(main())

