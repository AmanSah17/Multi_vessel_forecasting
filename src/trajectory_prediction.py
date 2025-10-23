"""
Trajectory Prediction Models

Implements multiple prediction approaches:
- Kalman Filter (real-time, low-latency)
- ARIMA (statistical baseline)
- LSTM (deep learning)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrajectoryPredictor(ABC):
    """Abstract base class for trajectory predictors."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next position."""
        pass


class KalmanFilterPredictor(TrajectoryPredictor):
    """
    Kalman Filter for vessel trajectory prediction.
    
    Best for: Real-time prediction, low-latency applications
    Advantages: Handles missing data, computationally efficient
    """
    
    def __init__(self, process_variance: float = 0.01, 
                 measurement_variance: float = 0.1):
        """
        Initialize Kalman Filter.
        
        Args:
            process_variance: Process noise (Q)
            measurement_variance: Measurement noise (R)
        """
        self.Q = process_variance  # Process noise
        self.R = measurement_variance  # Measurement noise
        self.P = 1.0  # Estimate error
        self.x = 0.0  # State estimate
        self.K = 0.0  # Kalman gain
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Estimate Kalman Filter parameters from historical data.
        
        Args:
            X: Historical positions
            y: Actual positions
        """
        # Estimate process and measurement noise from residuals
        residuals = y - X
        self.Q = np.var(np.diff(residuals))
        self.R = np.var(residuals)
        logger.info(f"Kalman Filter - Q: {self.Q:.6f}, R: {self.R:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict next position using Kalman Filter.
        
        Args:
            X: Current position measurement
            
        Returns:
            Predicted next position
        """
        # Prediction step
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # Update step
        self.K = P_pred / (P_pred + self.R)
        self.x = x_pred + self.K * (X - x_pred)
        self.P = (1 - self.K) * P_pred
        
        return self.x
    
    def predict_trajectory(self, positions: np.ndarray, 
                          steps: int = 10) -> np.ndarray:
        """
        Predict trajectory for multiple steps.
        
        Args:
            positions: Historical positions (lat, lon)
            steps: Number of steps to predict
            
        Returns:
            Predicted positions
        """
        predictions = []
        current = positions[-1]
        
        for _ in range(steps):
            current = self.predict(current)
            predictions.append(current)
        
        return np.array(predictions)


class ARIMAPredictor(TrajectoryPredictor):
    """
    ARIMA for vessel trajectory prediction.
    
    Best for: Statistical baseline, interpretable results
    Advantages: Handles seasonality, good for short-term forecasts
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Initialize ARIMA predictor.
        
        Args:
            order: (p, d, q) parameters
        """
        self.order = order
        self.p, self.d, self.q = order
        self.coefficients = None
        self.mean = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit ARIMA model.
        
        Args:
            X: Time series data
            y: Unused (for compatibility)
        """
        # Differencing
        diff_data = X
        for _ in range(self.d):
            diff_data = np.diff(diff_data)
        
        self.mean = np.mean(diff_data)
        
        # Simplified AR coefficient estimation
        if self.p > 0:
            self.coefficients = np.corrcoef(diff_data[:-1], diff_data[1:])[0, 1]
        
        logger.info(f"ARIMA{self.order} fitted")
    
    def predict(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Predict next values.
        
        Args:
            X: Historical data
            steps: Number of steps to predict
            
        Returns:
            Predicted values
        """
        predictions = []
        current = X[-1]
        
        for _ in range(steps):
            if self.coefficients is not None:
                current = self.mean + self.coefficients * (current - self.mean)
            else:
                current = self.mean
            predictions.append(current)
        
        return np.array(predictions)


class LSTMPredictor(TrajectoryPredictor):
    """
    LSTM for vessel trajectory prediction.
    
    Best for: Long-term patterns, complex non-linear relationships
    Advantages: Captures temporal dependencies, handles variable-length sequences
    """
    
    def __init__(self, sequence_length: int = 60, 
                 prediction_horizon: int = 10):
        """
        Initialize LSTM predictor.
        
        Args:
            sequence_length: Input sequence length (timesteps)
            prediction_horizon: Number of steps to predict
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit LSTM model.
        
        Args:
            X: Training sequences (N, sequence_length, features)
            y: Target values (N, prediction_horizon, features)
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from sklearn.preprocessing import MinMaxScaler
            
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            self.model = Sequential([
                LSTM(64, activation='relu', input_shape=(self.sequence_length, X.shape[-1]),
                     return_sequences=True),
                Dropout(0.2),
                LSTM(32, activation='relu'),
                Dropout(0.2),
                Dense(self.prediction_horizon * X.shape[-1]),
            ])
            
            self.model.compile(optimizer='adam', loss='mse')
            logger.info("LSTM model created")
            
        except ImportError:
            logger.warning("TensorFlow not available. LSTM predictor will not work.")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict trajectory.
        
        Args:
            X: Input sequence (sequence_length, features)
            
        Returns:
            Predicted trajectory
        """
        if self.model is None:
            logger.error("Model not fitted")
            return None
        
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(1, *X.shape)
        predictions = self.model.predict(X_scaled)
        
        return predictions.reshape(self.prediction_horizon, -1)


class EnsemblePredictor:
    """Ensemble of multiple predictors for robust predictions."""
    
    def __init__(self, predictors: Dict[str, TrajectoryPredictor]):
        """
        Initialize ensemble.
        
        Args:
            predictors: Dictionary of predictor name -> predictor instance
        """
        self.predictors = predictors
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all predictors."""
        for name, predictor in self.predictors.items():
            logger.info(f"Fitting {name}...")
            predictor.fit(X, y)
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from all models.
        
        Args:
            X: Input data
            
        Returns:
            Dictionary of predictions from each model
        """
        predictions = {}
        for name, predictor in self.predictors.items():
            predictions[name] = predictor.predict(X)
        return predictions
    
    def predict_ensemble(self, X: np.ndarray, 
                        method: str = 'mean') -> np.ndarray:
        """
        Combine predictions from all models.
        
        Args:
            X: Input data
            method: 'mean', 'median', or 'weighted'
            
        Returns:
            Ensemble prediction
        """
        predictions = self.predict(X)
        pred_array = np.array(list(predictions.values()))
        
        if method == 'mean':
            return np.mean(pred_array, axis=0)
        elif method == 'median':
            return np.median(pred_array, axis=0)
        elif method == 'weighted':
            # Weighted by model performance (to be implemented)
            return np.mean(pred_array, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

