"""
How to Use Saved XGBoost Model for Vessel Trajectory Predictions

This notebook shows:
1. Load saved model and preprocessing pipeline
2. Prepare vessel data from database
3. Make single-step predictions
4. Make multi-step recursive predictions (sliding window)
5. Visualize predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


class VesselTrajectoryPredictor:
    """Complete pipeline for vessel trajectory prediction."""
    
    def __init__(self, model_dir='results/xgboost_corrected_50_vessels'):
        """Load saved model and preprocessing objects."""
        logger.info(f"Loading model from {model_dir}...")
        
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / 'xgboost_model.joblib')
        self.scaler = joblib.load(model_dir / 'scaler.joblib')
        self.pca = joblib.load(model_dir / 'pca.joblib')
        
        logger.info("Model loaded successfully")
    
    def extract_advanced_features(self, X):
        """Extract 483 advanced features from sequences.
        
        Args:
            X: shape (n_samples, n_timesteps, n_features)
        
        Returns:
            Features: shape (n_samples, 483)
        """
        n_samples, n_timesteps, n_features = X.shape
        features_list = []
        
        for dim in range(n_features):
            X_dim = X[:, :, dim]
            
            # Statistical features (10)
            features_dict = {
                'mean': np.mean(X_dim, axis=1),
                'std': np.std(X_dim, axis=1),
                'min': np.min(X_dim, axis=1),
                'max': np.max(X_dim, axis=1),
                'median': np.median(X_dim, axis=1),
                'p25': np.percentile(X_dim, 25, axis=1),
                'p75': np.percentile(X_dim, 75, axis=1),
                'range': np.max(X_dim, axis=1) - np.min(X_dim, axis=1),
                'skew': np.array([pd.Series(row).skew() for row in X_dim]),
                'kurtosis': np.array([pd.Series(row).kurtosis() for row in X_dim]),
            }
            
            # Trend features (7)
            diff = np.diff(X_dim, axis=1)
            features_dict['trend_mean'] = np.mean(diff, axis=1)
            features_dict['trend_std'] = np.std(diff, axis=1)
            features_dict['trend_max'] = np.max(diff, axis=1)
            features_dict['trend_min'] = np.min(diff, axis=1)
            
            # Autocorrelation (2)
            features_dict['first_last_diff'] = X_dim[:, -1] - X_dim[:, 0]
            features_dict['first_last_ratio'] = np.divide(X_dim[:, -1], X_dim[:, 0] + 1e-6)
            
            # Volatility (1)
            features_dict['volatility'] = np.std(diff, axis=1)
            
            dim_features = np.column_stack(list(features_dict.values()))
            features_list.append(dim_features)
        
        X_features = np.hstack(features_list)
        return X_features
    
    def add_haversine_features(self, X):
        """Add 7 Haversine distance features."""
        n_samples = X.shape[0]
        haversine_features = []
        
        for i in range(n_samples):
            seq = X[i]
            lats = seq[:, 0]
            lons = seq[:, 1]
            
            # Distance to first point
            dist_to_first = self._haversine_distance(lats[0], lons[0], lats, lons)
            
            # Consecutive distances
            consecutive_dists = [0.0]
            for j in range(1, len(lats)):
                dist = self._haversine_distance(lats[j-1], lons[j-1], lats[j], lons[j])
                consecutive_dists.append(dist)
            
            total_dist = np.sum(consecutive_dists)
            avg_dist = np.mean(consecutive_dists[1:]) if len(consecutive_dists) > 1 else 0
            
            haversine_features.append([
                np.mean(dist_to_first),
                np.max(dist_to_first),
                np.std(dist_to_first),
                total_dist,
                avg_dist,
                np.max(consecutive_dists),
                np.std(consecutive_dists)
            ])
        
        return np.array(haversine_features)
    
    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate Haversine distance in km."""
        R = 6371
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def preprocess_features(self, X_raw):
        """Extract features and apply preprocessing.
        
        Args:
            X_raw: shape (n_samples, 12, 4) - raw vessel data
        
        Returns:
            X_pca: shape (n_samples, n_components) - ready for model
        """
        # Extract features
        X_feat = self.extract_advanced_features(X_raw)
        X_hav = self.add_haversine_features(X_raw)
        X_combined = np.hstack([X_feat, X_hav])
        
        # Apply preprocessing (TRANSFORM, not fit)
        X_scaled = self.scaler.transform(X_combined)
        X_pca = self.pca.transform(X_scaled)
        
        return X_pca
    
    def predict_single_step(self, vessel_data):
        """Make single-step prediction.
        
        Args:
            vessel_data: DataFrame with last 12 records
                        Columns: LAT, LON, SOG, COG
        
        Returns:
            prediction: dict with predicted LAT, LON, SOG, COG
        """
        # Prepare data
        X = vessel_data[['LAT', 'LON', 'SOG', 'COG']].values
        X = X.reshape(1, 12, 4).astype(np.float32)
        
        # Preprocess
        X_pca = self.preprocess_features(X)
        
        # Predict
        pred = self.model.predict(X_pca)[0]
        
        return {
            'predicted_lat': float(pred[0]),
            'predicted_lon': float(pred[1]),
            'predicted_sog': float(pred[2]),
            'predicted_cog': float(pred[3]),
        }
    
    def predict_multi_step(self, vessel_data, steps=12, step_interval_minutes=5):
        """Make multi-step recursive predictions (sliding window).
        
        Args:
            vessel_data: DataFrame with last 12 records
            steps: number of future steps to predict
            step_interval_minutes: minutes between steps
        
        Returns:
            predictions: list of dicts with predictions
        """
        predictions = []
        current_data = vessel_data.copy()
        
        for step in range(steps):
            # Get last 12 records
            X = current_data[['LAT', 'LON', 'SOG', 'COG']].tail(12).values
            X = X.reshape(1, 12, 4).astype(np.float32)
            
            # Predict next step
            X_pca = self.preprocess_features(X)
            pred = self.model.predict(X_pca)[0]
            
            # Store prediction
            pred_dict = {
                'step': step + 1,
                'minutes_ahead': (step + 1) * step_interval_minutes,
                'predicted_lat': float(pred[0]),
                'predicted_lon': float(pred[1]),
                'predicted_sog': float(pred[2]),
                'predicted_cog': float(pred[3]),
            }
            predictions.append(pred_dict)
            
            # Append prediction to data for next iteration
            new_row = pd.DataFrame([{
                'LAT': pred[0],
                'LON': pred[1],
                'SOG': pred[2],
                'COG': pred[3],
            }])
            current_data = pd.concat([current_data, new_row], ignore_index=True)
        
        return predictions
    
    def plot_predictions(self, vessel_data, predictions_multi, vessel_name="Vessel"):
        """Plot actual vs predicted trajectory.
        
        Args:
            vessel_data: DataFrame with actual data
            predictions_multi: list of multi-step predictions
            vessel_name: name for plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{vessel_name} - Trajectory Predictions', fontsize=14, fontweight='bold')
        
        # Extract actual data
        actual_lat = vessel_data['LAT'].values
        actual_lon = vessel_data['LON'].values
        actual_sog = vessel_data['SOG'].values
        actual_cog = vessel_data['COG'].values
        
        # Extract predictions
        pred_lat = [p['predicted_lat'] for p in predictions_multi]
        pred_lon = [p['predicted_lon'] for p in predictions_multi]
        pred_sog = [p['predicted_sog'] for p in predictions_multi]
        pred_cog = [p['predicted_cog'] for p in predictions_multi]
        
        # Time axis
        actual_time = np.arange(len(actual_lat)) * 5
        pred_time = np.array([p['minutes_ahead'] for p in predictions_multi])
        
        # Plot LAT
        axes[0, 0].plot(actual_time, actual_lat, 'b-o', label='Actual', linewidth=2)
        axes[0, 0].plot(pred_time, pred_lat, 'r--s', label='Predicted', linewidth=2)
        axes[0, 0].set_ylabel('Latitude (°)', fontweight='bold')
        axes[0, 0].set_title('Latitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot LON
        axes[0, 1].plot(actual_time, actual_lon, 'b-o', label='Actual', linewidth=2)
        axes[0, 1].plot(pred_time, pred_lon, 'r--s', label='Predicted', linewidth=2)
        axes[0, 1].set_ylabel('Longitude (°)', fontweight='bold')
        axes[0, 1].set_title('Longitude')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot SOG
        axes[1, 0].plot(actual_time, actual_sog, 'b-o', label='Actual', linewidth=2)
        axes[1, 0].plot(pred_time, pred_sog, 'r--s', label='Predicted', linewidth=2)
        axes[1, 0].set_xlabel('Time (minutes)', fontweight='bold')
        axes[1, 0].set_ylabel('SOG (knots)', fontweight='bold')
        axes[1, 0].set_title('Speed Over Ground')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot COG
        axes[1, 1].plot(actual_time, actual_cog, 'b-o', label='Actual', linewidth=2)
        axes[1, 1].plot(pred_time, pred_cog, 'r--s', label='Predicted', linewidth=2)
        axes[1, 1].set_xlabel('Time (minutes)', fontweight='bold')
        axes[1, 1].set_ylabel('COG (°)', fontweight='bold')
        axes[1, 1].set_title('Course Over Ground')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example: Load model and make predictions for a vessel."""
    
    # 1. Initialize predictor
    predictor = VesselTrajectoryPredictor()
    
    # 2. Load vessel data from database (example)
    # In real usage, fetch from your database
    vessel_data = pd.DataFrame({
        'LAT': np.linspace(32.7, 32.78, 12),
        'LON': np.linspace(-77.0, -76.98, 12),
        'SOG': np.random.normal(15.2, 0.5, 12),
        'COG': np.random.normal(45.0, 2.0, 12),
    })
    
    logger.info("Vessel Data (last 12 records):")
    logger.info(vessel_data.tail())
    
    # 3. Single-step prediction
    logger.info("\n=== SINGLE-STEP PREDICTION ===")
    pred_single = predictor.predict_single_step(vessel_data)
    logger.info(f"Next position: ({pred_single['predicted_lat']:.4f}, {pred_single['predicted_lon']:.4f})")
    logger.info(f"Next speed: {pred_single['predicted_sog']:.2f} knots")
    logger.info(f"Next course: {pred_single['predicted_cog']:.2f}°")
    
    # 4. Multi-step predictions (12 steps = 60 minutes)
    logger.info("\n=== MULTI-STEP PREDICTIONS (60 minutes ahead) ===")
    predictions_multi = predictor.predict_multi_step(vessel_data, steps=12)
    
    for pred in predictions_multi[:3]:  # Show first 3
        logger.info(f"Step {pred['step']} ({pred['minutes_ahead']} min): "
                   f"LAT={pred['predicted_lat']:.4f}, LON={pred['predicted_lon']:.4f}")
    
    # 5. Plot
    logger.info("\n=== PLOTTING ===")
    fig = predictor.plot_predictions(vessel_data, predictions_multi, "CHAMPAGNE CHER")
    plt.savefig('vessel_predictions.png', dpi=120, bbox_inches='tight')
    logger.info("Plot saved to vessel_predictions.png")
    
    return predictor, vessel_data, predictions_multi


if __name__ == "__main__":
    predictor, vessel_data, predictions = example_usage()

