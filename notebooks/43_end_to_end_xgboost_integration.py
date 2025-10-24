"""
End-to-End XGBoost Integration Pipeline
Demonstrates full workflow: fetch vessel data, predict, verify, and visualize
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from xgboost_integration import XGBoostPredictor
from vessel_prediction_service import VesselPredictionService
from trajectory_visualization import TrajectoryVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EndToEndPipeline:
    """Complete pipeline for vessel prediction and verification"""
    
    def __init__(self, db_path: str = None, model_dir: str = "results/xgboost_advanced_50_vessels"):
        """
        Initialize pipeline
        
        Args:
            db_path: Path to Maritime database
            model_dir: Path to XGBoost model directory
        """
        logger.info("🚀 Initializing End-to-End XGBoost Pipeline...")
        
        # Import Maritime NLU components
        try:
            sys.path.insert(0, "f:\\Maritime_NLU_Repo\\backend\\nlu_chatbot\\src\\app")
            from db_handler import MaritimeDB
            
            # Use provided db_path or find it
            if db_path is None:
                db_path = "f:\\Maritime_NLU_Repo\\backend\\nlu_chatbot\\maritime_sample_0104.db"
            
            self.db = MaritimeDB(db_path)
            logger.info(f"✅ Connected to Maritime database: {db_path}")
        except Exception as e:
            logger.error(f"❌ Error loading Maritime NLU: {e}")
            raise
        
        # Initialize XGBoost predictor
        try:
            self.predictor = XGBoostPredictor(model_dir)
            logger.info("✅ XGBoost model loaded")
        except Exception as e:
            logger.error(f"❌ Error loading XGBoost model: {e}")
            raise
        
        # Initialize prediction service
        self.service = VesselPredictionService(self.db, self.predictor)
        logger.info("✅ Prediction service initialized")
        
        # Initialize visualizer
        self.visualizer = TrajectoryVisualizer("results/xgboost_predictions")
        logger.info("✅ Visualizer initialized")
        
        self.results = []
    
    def predict_random_vessels(self, n_vessels: int = 5, minutes_ahead: int = 30):
        """
        Predict positions for random vessels
        
        Args:
            n_vessels: Number of random vessels to predict
            minutes_ahead: Minutes to predict ahead
        """
        logger.info(f"\n📊 Predicting positions for {n_vessels} random vessels...")
        
        try:
            # Get all vessel names
            all_vessels = self.db.get_all_vessel_names()
            logger.info(f"Total vessels in database: {len(all_vessels)}")
            
            # Sample random vessels
            selected_vessels = np.random.choice(all_vessels, min(n_vessels, len(all_vessels)), replace=False)
            logger.info(f"Selected {len(selected_vessels)} random vessels for prediction")
            
            predictions = []
            
            for vessel_name in tqdm(selected_vessels, desc="Predicting vessels", unit="vessel"):
                try:
                    # Get prediction
                    pred = self.service.predict_vessel_position(
                        vessel_name=vessel_name,
                        minutes_ahead=minutes_ahead
                    )
                    
                    if "error" not in pred:
                        predictions.append(pred)
                        logger.info(f"✅ Predicted: {vessel_name}")
                    else:
                        logger.warning(f"⚠️  {vessel_name}: {pred['error']}")
                
                except Exception as e:
                    logger.error(f"❌ Error predicting {vessel_name}: {e}")
            
            self.results.extend(predictions)
            logger.info(f"✅ Successfully predicted {len(predictions)}/{len(selected_vessels)} vessels")
            
            return predictions
        
        except Exception as e:
            logger.error(f"❌ Error in predict_random_vessels: {e}")
            return []
    
    def verify_vessel_courses(self, vessel_names: list = None):
        """
        Verify courses for specific vessels
        
        Args:
            vessel_names: List of vessel names to verify (if None, use predicted vessels)
        """
        logger.info(f"\n🔍 Verifying vessel courses...")
        
        if vessel_names is None:
            vessel_names = [r['vessel_name'] for r in self.results if 'vessel_name' in r]
        
        verifications = []
        
        for vessel_name in tqdm(vessel_names, desc="Verifying courses", unit="vessel"):
            try:
                verify = self.service.verify_vessel_course(vessel_name=vessel_name)
                
                if "error" not in verify:
                    verifications.append(verify)
                    
                    # Log anomalies
                    if verify['verification']['anomaly_detected']:
                        logger.warning(f"⚠️  ANOMALY in {vessel_name}: {verify['verification']['anomaly_reason']}")
                    else:
                        logger.info(f"✅ Verified: {vessel_name} - Course {verify['verification']['consistency']}")
                else:
                    logger.warning(f"⚠️  {vessel_name}: {verify['error']}")
            
            except Exception as e:
                logger.error(f"❌ Error verifying {vessel_name}: {e}")
        
        logger.info(f"✅ Verified {len(verifications)} vessels")
        return verifications
    
    def visualize_predictions(self, predictions: list = None):
        """
        Create visualizations for predictions
        
        Args:
            predictions: List of predictions to visualize (if None, use all results)
        """
        logger.info(f"\n📈 Creating visualizations...")
        
        if predictions is None:
            predictions = self.results
        
        if not predictions:
            logger.warning("No predictions to visualize")
            return []
        
        saved_files = self.visualizer.plot_batch_predictions(predictions)
        logger.info(f"✅ Created {len(saved_files)} visualizations")
        
        return saved_files
    
    def generate_report(self, output_file: str = "results/xgboost_predictions/integration_report.json"):
        """
        Generate comprehensive report
        
        Args:
            output_file: Path to save report
        """
        logger.info(f"\n📋 Generating report...")
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "pipeline": "XGBoost Advanced Integration",
            "total_predictions": len(self.results),
            "predictions": self.results,
            "summary": {
                "avg_confidence": np.mean([r.get('confidence_score', 0) for r in self.results]) if self.results else 0,
                "total_vessels_processed": len(self.results),
                "model_info": {
                    "name": "XGBoost Advanced",
                    "features": 483,
                    "pca_components": 80,
                    "variance_retained": "95.10%"
                }
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Report saved to {output_file}")
        return report
    
    def run_full_pipeline(self, n_vessels: int = 5, minutes_ahead: int = 30):
        """
        Run complete pipeline
        
        Args:
            n_vessels: Number of vessels to process
            minutes_ahead: Minutes to predict ahead
        """
        logger.info("\n" + "="*60)
        logger.info("🚀 STARTING END-TO-END XGBOOST INTEGRATION PIPELINE")
        logger.info("="*60)
        
        try:
            # Step 1: Predict
            predictions = self.predict_random_vessels(n_vessels, minutes_ahead)
            
            # Step 2: Verify
            verifications = self.verify_vessel_courses([p['vessel_name'] for p in predictions])
            
            # Step 3: Visualize
            visualizations = self.visualize_predictions(predictions)
            
            # Step 4: Report
            report = self.generate_report()
            
            logger.info("\n" + "="*60)
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Predictions: {len(predictions)}")
            logger.info(f"Verifications: {len(verifications)}")
            logger.info(f"Visualizations: {len(visualizations)}")
            logger.info(f"Report: results/xgboost_predictions/integration_report.json")
            
            return {
                "predictions": predictions,
                "verifications": verifications,
                "visualizations": visualizations,
                "report": report
            }
        
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise


if __name__ == "__main__":
    # Run pipeline
    pipeline = EndToEndPipeline()
    results = pipeline.run_full_pipeline(n_vessels=5, minutes_ahead=30)
    
    logger.info("\n✅ All tasks completed!")

