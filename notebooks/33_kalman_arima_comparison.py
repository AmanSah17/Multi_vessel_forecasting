"""
Kalman Filter & ARIMA Models Comparison Pipeline
- Kalman Filter for real-time state estimation
- ARIMA for time series forecasting
- Per-vessel evaluation
- MLflow logging for experiment tracking
- Comparison with Tiny LSTM Haversine baseline
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from pathlib import Path
from tqdm import tqdm
import mlflow
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima.model import ARIMA
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/kalman_arima_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======================== KALMAN FILTER ========================

class KalmanFilter1D:
    """1D Kalman Filter for single variable prediction."""
    def __init__(self, process_variance=0.01, measurement_variance=0.1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0
        self.estimate_error = 1.0
        self.kalman_gain = 0
    
    def update(self, measurement):
        """Update filter with new measurement."""
        self.estimate_error += self.process_variance
        self.kalman_gain = self.estimate_error / (self.estimate_error + self.measurement_variance)
        self.estimate += self.kalman_gain * (measurement - self.estimate)
        self.estimate_error *= (1 - self.kalman_gain)
        return self.estimate
    
    def predict(self, steps=1):
        """Predict next value."""
        return self.estimate


class KalmanFilterVessel:
    """Multi-dimensional Kalman Filter for vessel state."""
    def __init__(self, process_var=0.01, measurement_var=0.1):
        self.filters = {
            'LAT': KalmanFilter1D(process_var, measurement_var),
            'LON': KalmanFilter1D(process_var, measurement_var),
            'SOG': KalmanFilter1D(process_var, measurement_var),
            'COG': KalmanFilter1D(process_var, measurement_var)
        }
        self.history = {key: [] for key in self.filters.keys()}
    
    def update(self, measurements):
        """Update all filters with measurements."""
        predictions = {}
        for key, value in measurements.items():
            if key in self.filters:
                pred = self.filters[key].update(value)
                predictions[key] = pred
                self.history[key].append(pred)
        return predictions
    
    def predict_next(self):
        """Predict next state."""
        return {key: f.predict() for key, f in self.filters.items()}


# ======================== ARIMA MODEL ========================

class ARIMAVessel:
    """ARIMA model for per-vessel forecasting."""
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.models = {}
        self.history = {key: [] for key in ['LAT', 'LON', 'SOG', 'COG']}
    
    def fit(self, data_dict, min_samples=20):
        """Fit ARIMA models for each variable."""
        for key, values in data_dict.items():
            if len(values) >= min_samples:
                try:
                    self.models[key] = ARIMA(values, order=self.order).fit()
                    self.history[key] = list(values)
                except Exception as e:
                    logger.warning(f"ARIMA fit failed for {key}: {e}")
                    self.models[key] = None
            else:
                self.models[key] = None
    
    def predict(self, steps=1):
        """Predict next steps."""
        predictions = {}
        for key, model in self.models.items():
            if model is not None:
                try:
                    forecast = model.get_forecast(steps=steps)
                    predictions[key] = forecast.predicted_mean.values[-1]
                except:
                    predictions[key] = self.history[key][-1] if self.history[key] else 0
            else:
                predictions[key] = self.history[key][-1] if self.history[key] else 0
        return predictions


# ======================== EVALUATION FUNCTIONS ========================

def evaluate_kalman_filter(X_test, y_test, sequence_length=12):
    """Evaluate Kalman Filter on test set."""
    logger.info("\n" + "="*80)
    logger.info("EVALUATING KALMAN FILTER")
    logger.info("="*80)
    
    predictions = []
    
    for i in tqdm(range(len(X_test)), desc="Kalman Filter Inference", unit="sample"):
        kf = KalmanFilterVessel(process_var=0.01, measurement_var=0.1)
        
        # Train on sequence
        for t in range(sequence_length):
            measurements = {
                'LAT': X_test[i, t, 0],
                'LON': X_test[i, t, 1],
                'SOG': X_test[i, t, 2],
                'COG': X_test[i, t, 3]
            }
            kf.update(measurements)
        
        # Predict next
        pred = kf.predict_next()
        predictions.append([pred['LAT'], pred['LON'], pred['SOG'], pred['COG']])
    
    predictions = np.array(predictions)
    
    # Compute metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    logger.info(f"Kalman Filter: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    
    return predictions, {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def evaluate_arima(X_test, y_test, sequence_length=12):
    """Evaluate ARIMA on test set."""
    logger.info("\n" + "="*80)
    logger.info("EVALUATING ARIMA")
    logger.info("="*80)
    
    predictions = []
    
    for i in tqdm(range(len(X_test)), desc="ARIMA Inference", unit="sample"):
        arima = ARIMAVessel(order=(1, 1, 1))
        
        # Prepare data for ARIMA
        data_dict = {
            'LAT': X_test[i, :sequence_length, 0],
            'LON': X_test[i, :sequence_length, 1],
            'SOG': X_test[i, :sequence_length, 2],
            'COG': X_test[i, :sequence_length, 3]
        }
        
        # Fit and predict
        arima.fit(data_dict)
        pred = arima.predict(steps=1)
        predictions.append([pred['LAT'], pred['LON'], pred['SOG'], pred['COG']])
    
    predictions = np.array(predictions)
    
    # Compute metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    logger.info(f"ARIMA: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    
    return predictions, {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def evaluate_ensemble(lstm_pred, kalman_pred, arima_pred, y_test, weights=None):
    """Evaluate ensemble of all models."""
    logger.info("\n" + "="*80)
    logger.info("EVALUATING ENSEMBLE (LSTM + Kalman + ARIMA)")
    logger.info("="*80)
    
    if weights is None:
        weights = [0.5, 0.25, 0.25]  # LSTM, Kalman, ARIMA
    
    ensemble_pred = (weights[0] * lstm_pred + 
                     weights[1] * kalman_pred + 
                     weights[2] * arima_pred)
    
    mae = mean_absolute_error(y_test, ensemble_pred)
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    r2 = r2_score(y_test, ensemble_pred)
    
    logger.info(f"Ensemble: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    
    return ensemble_pred, {'MAE': mae, 'RMSE': rmse, 'R2': r2}


# ======================== MAIN PIPELINE ========================

def main():
    """Main comparison pipeline - Kalman Filter & ARIMA only (no LSTM training)."""
    logger.info("="*80)
    logger.info("KALMAN FILTER & ARIMA COMPARISON PIPELINE (NO LSTM TRAINING)")
    logger.info("="*80)

    # Setup MLflow
    mlflow.set_experiment("Kalman_ARIMA_Comparison_v2")

    # Load cached test data
    logger.info("\nLoading test data...")
    cache_file = Path('results/cache/seq_cache_len12_sampled_3pct.npz')

    if not cache_file.exists():
        logger.error(f"Cache file not found: {cache_file}")
        return

    data = np.load(cache_file, allow_pickle=True)
    X = data['X']
    y = data['y']

    # Use test set (last 10%)
    test_start = int(len(X) * 0.9)
    X_test = X[test_start:]
    y_test = y[test_start:]

    logger.info(f"✓ Test set: {len(X_test)} samples")
    logger.info(f"✓ Test targets shape: {y_test.shape}")

    # Evaluate models
    with mlflow.start_run(run_name="Kalman_ARIMA_Only"):
        logger.info("\n" + "="*80)
        logger.info("[1/2] EVALUATING KALMAN FILTER")
        logger.info("="*80)
        kalman_pred, kalman_metrics = evaluate_kalman_filter(X_test, y_test)

        logger.info("\n" + "="*80)
        logger.info("[2/2] EVALUATING ARIMA")
        logger.info("="*80)
        arima_pred, arima_metrics = evaluate_arima(X_test, y_test)

        # Log all results to MLflow
        logger.info("\n" + "="*80)
        logger.info("LOGGING RESULTS TO MLFLOW")
        logger.info("="*80)

        # Kalman Filter
        mlflow.log_metrics({
            "Kalman_MAE": kalman_metrics['MAE'],
            "Kalman_RMSE": kalman_metrics['RMSE'],
            "Kalman_R2": kalman_metrics['R2']
        })

        # ARIMA
        mlflow.log_metrics({
            "ARIMA_MAE": arima_metrics['MAE'],
            "ARIMA_RMSE": arima_metrics['RMSE'],
            "ARIMA_R2": arima_metrics['R2']
        })

        # Log parameters
        mlflow.log_params({
            "test_samples": len(X_test),
            "sequence_length": 12,
            "kalman_process_var": 0.01,
            "kalman_measurement_var": 0.1,
            "arima_order": "(1,1,1)",
            "models_compared": "Kalman Filter, ARIMA"
        })

        # Summary
        logger.info("\n" + "="*80)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*80)
        logger.info(f"Kalman Filter:  MAE={kalman_metrics['MAE']:.4f}, RMSE={kalman_metrics['RMSE']:.4f}, R2={kalman_metrics['R2']:.4f}")
        logger.info(f"ARIMA:          MAE={arima_metrics['MAE']:.4f}, RMSE={arima_metrics['RMSE']:.4f}, R2={arima_metrics['R2']:.4f}")

        # Determine best model
        if kalman_metrics['MAE'] < arima_metrics['MAE']:
            logger.info(f"\n🏆 BEST MODEL: Kalman Filter (MAE improvement: {arima_metrics['MAE'] - kalman_metrics['MAE']:.4f})")
        else:
            logger.info(f"\n🏆 BEST MODEL: ARIMA (MAE improvement: {kalman_metrics['MAE'] - arima_metrics['MAE']:.4f})")

        logger.info("\n✓ COMPARISON COMPLETE")
        logger.info("✓ Results logged to MLflow: mlruns/")


if __name__ == "__main__":
    main()

