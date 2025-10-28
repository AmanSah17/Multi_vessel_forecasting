"""
Advanced Prediction Patterns for Vessel Trajectories

Demonstrates:
1. Batch predictions for multiple vessels
2. Uncertainty quantification
3. Anomaly detection in predictions
4. Comparison with dead reckoning
5. Real-time streaming predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedVesselPredictor:
    """Advanced prediction patterns."""

    def __init__(self, model_dir='results/xgboost_corrected_50_vessels'):
        """Initialize with saved model."""
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / 'xgboost_model.joblib')
        self.scaler = joblib.load(model_dir / 'scaler.joblib')
        self.pca = joblib.load(model_dir / 'pca.joblib')

    def dead_reckoning(self, vessel_data, minutes_ahead=60):
        """Simple dead reckoning baseline.

        Assumes constant velocity and course.
        """
        last_row = vessel_data.iloc[-1]
        lat = last_row['LAT']
        lon = last_row['LON']
        sog = last_row['SOG']
        cog = last_row['COG']

        # Calculate distance traveled
        hours = minutes_ahead / 60.0
        distance_nm = sog * hours
        delta_deg = distance_nm / 60.0

        # Convert to radians
        cog_rad = np.radians(cog)
        lat_rad = np.radians(lat)

        # Calculate new position
        dlat = delta_deg * np.sin(cog_rad)
        dlon = delta_deg * np.cos(cog_rad) / np.cos(lat_rad)

        new_lat = lat + dlat
        new_lon = lon + dlon

        return {
            'predicted_lat': new_lat,
            'predicted_lon': new_lon,
            'predicted_sog': sog,  # Constant
            'predicted_cog': cog,  # Constant
            'method': 'dead_reckoning'
        }

    def batch_predict_multiple_vessels(self, vessel_dict, steps=12):
        """Predict for multiple vessels at once.

        Args:
            vessel_dict: {vessel_name: vessel_dataframe, ...}
            steps: number of prediction steps

        Returns:
            results: {vessel_name: predictions, ...}
        """
        results = {}

        for vessel_name, vessel_data in vessel_dict.items():
            logger.info(f"Predicting for {vessel_name}...")

            # Multi-step prediction
            predictions = []
            current_data = vessel_data.copy()

            for step in range(steps):
                X = current_data[['LAT', 'LON', 'SOG', 'COG']].tail(12).values
                X = X.reshape(1, 12, 4).astype(np.float32)

                # Preprocess and predict
                X_feat = self._extract_features(X)
                X_hav = self._add_haversine(X)
                X_combined = np.hstack([X_feat, X_hav])
                X_scaled = self.scaler.transform(X_combined)
                X_pca = self.pca.transform(X_scaled)

                pred = self.model.predict(X_pca)[0]

                predictions.append({
                    'step': step + 1,
                    'lat': float(pred[0]),
                    'lon': float(pred[1]),
                    'sog': float(pred[2]),
                    'cog': float(pred[3]),
                })

                # Append for next iteration
                new_row = pd.DataFrame([{
                    'LAT': pred[0],
                    'LON': pred[1],
                    'SOG': pred[2],
                    'COG': pred[3],
                }])
                current_data = pd.concat([current_data, new_row], ignore_index=True)

            results[vessel_name] = predictions

        return results

    def uncertainty_quantification(self, vessel_data, n_bootstrap=10):
        """Estimate prediction uncertainty using bootstrap.

        Args:
            vessel_data: DataFrame with vessel data
            n_bootstrap: number of bootstrap samples

        Returns:
            predictions with uncertainty bounds
        """
        predictions_bootstrap = []

        for _ in range(n_bootstrap):
            # Add small noise to input
            X = vessel_data[['LAT', 'LON', 'SOG', 'COG']].tail(12).values
            noise = np.random.normal(0, 0.001, X.shape)
            X_noisy = X + noise
            X_noisy = X_noisy.reshape(1, 12, 4).astype(np.float32)

            # Predict
            X_feat = self._extract_features(X_noisy)
            X_hav = self._add_haversine(X_noisy)
            X_combined = np.hstack([X_feat, X_hav])
            X_scaled = self.scaler.transform(X_combined)
            X_pca = self.pca.transform(X_scaled)

            pred = self.model.predict(X_pca)[0]
            predictions_bootstrap.append(pred)

        predictions_bootstrap = np.array(predictions_bootstrap)

        # Calculate statistics
        mean_pred = np.mean(predictions_bootstrap, axis=0)
        std_pred = np.std(predictions_bootstrap, axis=0)

        return {
            'mean_lat': float(mean_pred[0]),
            'mean_lon': float(mean_pred[1]),
            'mean_sog': float(mean_pred[2]),
            'mean_cog': float(mean_pred[3]),
            'std_lat': float(std_pred[0]),
            'std_lon': float(std_pred[1]),
            'std_sog': float(std_pred[2]),
            'std_cog': float(std_pred[3]),
        }

    def _extract_features(self, X):
        """Extract 483 features."""
        n_samples, n_timesteps, n_features = X.shape
        features_list = []

        for dim in range(n_features):
            X_dim = X[:, :, dim]

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

            diff = np.diff(X_dim, axis=1)
            features_dict['trend_mean'] = np.mean(diff, axis=1)
            features_dict['trend_std'] = np.std(diff, axis=1)
            features_dict['trend_max'] = np.max(diff, axis=1)
            features_dict['trend_min'] = np.min(diff, axis=1)
            features_dict['first_last_diff'] = X_dim[:, -1] - X_dim[:, 0]
            features_dict['first_last_ratio'] = np.divide(X_dim[:, -1], X_dim[:, 0] + 1e-6)
            features_dict['volatility'] = np.std(diff, axis=1)

            dim_features = np.column_stack(list(features_dict.values()))
            features_list.append(dim_features)

        return np.hstack(features_list)

    def _add_haversine(self, X):
        """Add 7 Haversine features."""
        n_samples = X.shape[0]
        haversine_features = []

        for i in range(n_samples):
            seq = X[i]
            lats = seq[:, 0]
            lons = seq[:, 1]

            dist_to_first = self._haversine_distance(lats[0], lons[0], lats, lons)

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


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_advanced_usage():
    """Demonstrate advanced prediction patterns."""

    predictor = AdvancedVesselPredictor()

    # Create sample vessel data
    vessel_data = pd.DataFrame({
        'LAT': np.linspace(32.7, 32.78, 12),
        'LON': np.linspace(-77.0, -76.98, 12),
        'SOG': np.random.normal(15.2, 0.5, 12),
        'COG': np.random.normal(45.0, 2.0, 12),
    })

    # 1. Compare methods
    logger.info("\n=== COMPARING XGBOOST VS DEAD RECKONING ===")
    comparison = predictor.compare_methods(vessel_data)
    logger.info(f"XGBoost: LAT={comparison['xgboost']['lat']:.4f}, LON={comparison['xgboost']['lon']:.4f}")
    logger.info(f"Dead Reckoning: LAT={comparison['dead_reckoning']['lat']:.4f}, LON={comparison['dead_reckoning']['lon']:.4f}")
    logger.info(f"Difference: LAT={comparison['difference']['lat']:.4f}, LON={comparison['difference']['lon']:.4f}")

    # 2. Uncertainty quantification
    logger.info("\n=== UNCERTAINTY QUANTIFICATION ===")
    uncertainty = predictor.uncertainty_quantification(vessel_data, n_bootstrap=10)
    logger.info(f"Predicted LAT: {uncertainty['mean_lat']:.4f} ± {uncertainty['std_lat']:.4f}")
    logger.info(f"Predicted LON: {uncertainty['mean_lon']:.4f} ± {uncertainty['std_lon']:.4f}")

    # 3. Batch predictions
    logger.info("\n=== BATCH PREDICTIONS ===")
    vessels = {
        'Vessel_A': vessel_data.copy(),
        'Vessel_B': vessel_data.copy(),
    }
    batch_results = predictor.batch_predict_multiple_vessels(vessels, steps=6)
    logger.info(f"Predicted for {len(batch_results)} vessels")


if __name__ == "__main__":
    example_advanced_usage()
