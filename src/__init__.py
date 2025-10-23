"""
Maritime Vessel Forecasting - ML Pipeline

A comprehensive end-to-end machine learning pipeline for:
- Vessel trajectory prediction
- Trajectory consistency verification
- Anomaly detection
- MMSI analysis
"""

__version__ = "1.0.0"
__author__ = "Maritime Analytics Team"

from .data_preprocessing import VesselDataPreprocessor, load_and_preprocess
from .mmsi_analysis import MMSIAnalyzer
from .trajectory_prediction import (
    KalmanFilterPredictor,
    ARIMAPredictor,
    LSTMPredictor,
    EnsemblePredictor,
)
from .trajectory_verification import TrajectoryVerifier
from .anomaly_detection import (
    IsolationForestDetector,
    AutoencoderDetector,
    RuleBasedDetector,
    EnsembleAnomalyDetector,
    create_default_rule_detector,
)
from .training_pipeline import TrainingPipeline
from .training_visualization import TrainingVisualizer

__all__ = [
    "VesselDataPreprocessor",
    "load_and_preprocess",
    "MMSIAnalyzer",
    "KalmanFilterPredictor",
    "ARIMAPredictor",
    "LSTMPredictor",
    "EnsemblePredictor",
    "TrajectoryVerifier",
    "IsolationForestDetector",
    "AutoencoderDetector",
    "RuleBasedDetector",
    "EnsembleAnomalyDetector",
    "create_default_rule_detector",
    "TrainingPipeline",
    "TrainingVisualizer",
]

