"""
Anomaly Detection Module

Implements multiple anomaly detection approaches:
- Isolation Forest (statistical outliers)
- Autoencoder (learned patterns)
- Rule-based detection (domain-specific)
- Ensemble approach
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors."""
    
    @abstractmethod
    def fit(self, X: np.ndarray):
        """Fit detector to normal data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 = anomaly, 0 = normal)."""
        pass


class IsolationForestDetector(AnomalyDetector):
    """
    Isolation Forest for anomaly detection.
    
    Best for: Statistical outliers, high-dimensional data
    Advantages: Fast, handles mixed data types
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies
        """
        self.contamination = contamination
        self.model = None
    
    def fit(self, X: np.ndarray):
        """
        Fit Isolation Forest.
        
        Args:
            X: Training data (normal trajectories)
        """
        try:
            from sklearn.ensemble import IsolationForest
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            self.model.fit(X)
            logger.info("Isolation Forest fitted")
        except ImportError:
            logger.error("scikit-learn not available")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            X: Data to predict
            
        Returns:
            1 for anomaly, 0 for normal
        """
        if self.model is None:
            logger.error("Model not fitted")
            return np.zeros(len(X))
        
        predictions = self.model.predict(X)
        # Convert -1 (anomaly) to 1, 1 (normal) to 0
        return (predictions == -1).astype(int)


class AutoencoderDetector(AnomalyDetector):
    """
    Autoencoder for anomaly detection.
    
    Best for: Learning complex normal patterns
    Advantages: Unsupervised, captures non-linear patterns
    """
    
    def __init__(self, encoding_dim: int = 8, threshold: float = 0.95):
        """
        Initialize Autoencoder detector.
        
        Args:
            encoding_dim: Dimension of encoded representation
            threshold: Reconstruction error threshold (percentile)
        """
        self.encoding_dim = encoding_dim
        self.threshold = threshold
        self.model = None
        self.reconstruction_threshold = None
    
    def fit(self, X: np.ndarray):
        """
        Fit Autoencoder.
        
        Args:
            X: Training data (normal trajectories)
        """
        try:
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense
            from sklearn.preprocessing import StandardScaler
            
            # Normalize data
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Build autoencoder
            input_dim = X.shape[1]
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(self.encoding_dim, activation='relu')(input_layer)
            decoded = Dense(input_dim, activation='linear')(encoded)
            
            self.model = Model(input_layer, decoded)
            self.model.compile(optimizer='adam', loss='mse')
            
            # Train
            self.model.fit(X_scaled, X_scaled, epochs=50, batch_size=32, 
                          verbose=0)
            
            # Calculate reconstruction error threshold
            train_predictions = self.model.predict(X_scaled, verbose=0)
            train_mse = np.mean(np.power(X_scaled - train_predictions, 2), axis=1)
            self.reconstruction_threshold = np.percentile(train_mse, self.threshold * 100)
            
            logger.info(f"Autoencoder fitted, threshold: {self.reconstruction_threshold:.4f}")
            
        except ImportError:
            logger.error("TensorFlow not available")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies based on reconstruction error.
        
        Args:
            X: Data to predict
            
        Returns:
            1 for anomaly, 0 for normal
        """
        if self.model is None or self.reconstruction_threshold is None:
            logger.error("Model not fitted")
            return np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - predictions, 2), axis=1)
        
        return (mse > self.reconstruction_threshold).astype(int)


class RuleBasedDetector(AnomalyDetector):
    """
    Rule-based anomaly detection using domain knowledge.
    
    Best for: Known anomaly patterns
    Advantages: Interpretable, fast, no training required
    """
    
    def __init__(self):
        """Initialize rule-based detector."""
        self.rules = []
    
    def fit(self, X: np.ndarray):
        """
        Fit rule-based detector (no-op, rules are predefined).
        
        Args:
            X: Training data (unused)
        """
        logger.info("Rule-based detector initialized")
    
    def add_rule(self, rule_func, name: str):
        """
        Add a detection rule.
        
        Args:
            rule_func: Function that returns True for anomalies
            name: Rule name
        """
        self.rules.append((name, rule_func))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using rules.
        
        Args:
            X: Data to predict (trajectory features)
            
        Returns:
            1 for anomaly, 0 for normal
        """
        anomalies = np.zeros(len(X))
        
        for name, rule_func in self.rules:
            try:
                rule_predictions = rule_func(X)
                anomalies = np.logical_or(anomalies, rule_predictions).astype(int)
            except Exception as e:
                logger.warning(f"Rule '{name}' failed: {e}")
        
        return anomalies


class EnsembleAnomalyDetector:
    """Ensemble of multiple anomaly detectors."""
    
    def __init__(self, detectors: Dict[str, AnomalyDetector]):
        """
        Initialize ensemble.
        
        Args:
            detectors: Dictionary of detector name -> detector instance
        """
        self.detectors = detectors
    
    def fit(self, X: np.ndarray):
        """Fit all detectors."""
        for name, detector in self.detectors.items():
            logger.info(f"Fitting {name}...")
            detector.fit(X)
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from all detectors.
        
        Args:
            X: Data to predict
            
        Returns:
            Dictionary of predictions from each detector
        """
        predictions = {}
        for name, detector in self.detectors.items():
            predictions[name] = detector.predict(X)
        return predictions
    
    def predict_ensemble(self, X: np.ndarray, 
                        method: str = 'majority') -> np.ndarray:
        """
        Combine predictions from all detectors.
        
        Args:
            X: Data to predict
            method: 'majority', 'any', or 'all'
            
        Returns:
            Ensemble prediction
        """
        predictions = self.predict(X)
        pred_array = np.array(list(predictions.values()))
        
        if method == 'majority':
            # Majority voting
            return (np.sum(pred_array, axis=0) > len(self.detectors) / 2).astype(int)
        elif method == 'any':
            # Any detector flags as anomaly
            return np.max(pred_array, axis=0)
        elif method == 'all':
            # All detectors must flag as anomaly
            return np.min(pred_array, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (0-1) from ensemble.
        
        Args:
            X: Data to predict
            
        Returns:
            Anomaly scores
        """
        predictions = self.predict(X)
        pred_array = np.array(list(predictions.values()))
        
        # Score = proportion of detectors flagging as anomaly
        return np.mean(pred_array, axis=0)


# Module-level rule functions (for pickling compatibility)
def _speed_rule(X):
    """Rule: Speed anomaly (>50 knots)."""
    if X.shape[1] > 2:  # Assuming SOG is column 2
        return X[:, 2] > 50
    return np.zeros(len(X), dtype=bool)


def _turn_rate_rule(X):
    """Rule: Impossible turn rate (>45°/min)."""
    if X.shape[1] > 3:  # Assuming turn_rate is column 3
        return X[:, 3] > 45
    return np.zeros(len(X), dtype=bool)


def _acceleration_rule(X):
    """Rule: Impossible acceleration (>2 knots/min)."""
    if X.shape[1] > 4:  # Assuming acceleration is column 4
        return X[:, 4] > 2
    return np.zeros(len(X), dtype=bool)


def create_default_rule_detector() -> RuleBasedDetector:
    """Create rule-based detector with domain-specific rules."""
    detector = RuleBasedDetector()
    detector.add_rule(_speed_rule, "speed_anomaly")
    detector.add_rule(_turn_rate_rule, "turn_rate_anomaly")
    detector.add_rule(_acceleration_rule, "acceleration_anomaly")
    return detector

