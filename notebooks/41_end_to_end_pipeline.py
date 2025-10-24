"""
End-to-End Vessel Trajectory Pipeline
- Load pre-trained XGBoost model
- Predict: Estimate vessel's next position after X minutes
- Verify: Plot course with last 5 points and 30-minute forecast
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/end_to_end_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


def haversine_distance(lat1, lon1, lat2, lon2):
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


def extract_advanced_features(X):
    """Extract 483 features from sequences."""
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


def add_haversine_features(X):
    """Add 7 Haversine distance features."""
    n_samples = X.shape[0]
    haversine_features = []
    
    for i in range(n_samples):
        seq = X[i]
        lats = seq[:, 0]
        lons = seq[:, 1]
        
        dist_to_first = haversine_distance(lats[0], lons[0], lats, lons)
        
        consecutive_dists = [0.0]
        for j in range(1, len(lats)):
            dist = haversine_distance(lats[j-1], lons[j-1], lats[j], lons[j])
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


def load_model_and_preprocessing():
    """Load pre-trained model and preprocessing objects."""
    logger.info("Loading pre-trained model and preprocessing objects...")
    
    model_path = 'results/xgboost_advanced_50_vessels/xgboost_model.pkl'
    scaler_path = 'results/xgboost_advanced_50_vessels/scaler.pkl'
    pca_path = 'results/xgboost_advanced_50_vessels/pca.pkl'
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    
    logger.info("[OK] Model and preprocessing objects loaded")
    return model, scaler, pca


def predict_next_position(X_sequence, model, scaler, pca, minutes_ahead=30):
    """
    Predict vessel's next position after X minutes.
    
    Args:
        X_sequence: Single sequence (12, 28)
        model: Trained XGBoost model
        scaler: StandardScaler
        pca: PCA transformer
        minutes_ahead: Minutes to predict ahead (default 30)
    
    Returns:
        dict with predicted LAT, LON, SOG, COG
    """
    # Reshape for batch processing
    X_batch = X_sequence.reshape(1, 12, 28)
    
    # Extract features
    X_features = extract_advanced_features(X_batch)
    X_haversine = add_haversine_features(X_batch)
    X_combined = np.hstack([X_features, X_haversine])
    
    # Preprocess
    X_scaled = scaler.transform(X_combined)
    X_pca = pca.transform(X_scaled)
    
    # Predict
    prediction = model.predict(X_pca)[0]
    
    # Get current position (last point in sequence)
    current_lat = X_sequence[-1, 0]
    current_lon = X_sequence[-1, 1]
    current_sog = X_sequence[-1, 2]
    current_cog = X_sequence[-1, 3]
    
    # Predicted values
    pred_lat = prediction[0]
    pred_lon = prediction[1]
    pred_sog = prediction[2]
    pred_cog = prediction[3]
    
    return {
        'current_lat': current_lat,
        'current_lon': current_lon,
        'current_sog': current_sog,
        'current_cog': current_cog,
        'pred_lat': pred_lat,
        'pred_lon': pred_lon,
        'pred_sog': pred_sog,
        'pred_cog': pred_cog,
        'minutes_ahead': minutes_ahead
    }


def extrapolate_trajectory(current_lat, current_lon, current_sog, current_cog, 
                          minutes_ahead=30, interval_minutes=5):
    """
    Extrapolate vessel trajectory assuming constant speed and course.
    
    Args:
        current_lat, current_lon: Current position
        current_sog: Speed Over Ground (knots)
        current_cog: Course Over Ground (degrees)
        minutes_ahead: Total minutes to extrapolate
        interval_minutes: Interval between points (default 5 minutes)
    
    Returns:
        DataFrame with extrapolated trajectory
    """
    # Convert speed from knots to km/minute
    speed_km_per_min = current_sog * 1.852 / 60
    
    # Calculate number of steps
    n_steps = int(minutes_ahead / interval_minutes) + 1
    
    trajectory = []
    lat, lon = current_lat, current_lon
    
    for step in range(n_steps):
        trajectory.append({
            'time_minutes': step * interval_minutes,
            'lat': lat,
            'lon': lon,
            'type': 'extrapolated'
        })
        
        # Calculate next position
        if step < n_steps - 1:
            # Distance traveled in this interval (km)
            distance_km = speed_km_per_min * interval_minutes
            
            # Convert to degrees (approximate)
            lat_change = distance_km / 111.0 * np.cos(np.radians(current_cog))
            lon_change = distance_km / 111.0 * np.sin(np.radians(current_cog))
            
            lat += lat_change
            lon += lon_change
    
    return pd.DataFrame(trajectory)


def plot_verification(X_sequence, prediction, output_dir, vessel_mmsi):
    """
    Plot verification: last 5 points + 30-minute forecast.
    
    Args:
        X_sequence: Full sequence (12, 28)
        prediction: Prediction dict from predict_next_position
        output_dir: Output directory
        vessel_mmsi: Vessel MMSI for filename
    """
    # Extract last 5 points
    last_5_lats = X_sequence[-5:, 0]
    last_5_lons = X_sequence[-5:, 1]
    
    # Extrapolate trajectory
    trajectory = extrapolate_trajectory(
        prediction['current_lat'],
        prediction['current_lon'],
        prediction['current_sog'],
        prediction['current_cog'],
        minutes_ahead=30,
        interval_minutes=5
    )
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Vessel {vessel_mmsi} - End-to-End Verification\n(Last 5 Points + 30-Min Forecast)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Course with last 5 points and forecast
    ax1.plot(last_5_lons, last_5_lats, 'b-o', linewidth=2.5, markersize=8, 
            label='Last 5 Points', alpha=0.8)
    ax1.plot(trajectory['lon'], trajectory['lat'], 'r--s', linewidth=2, markersize=6,
            label='30-Min Forecast (Constant Speed/Course)', alpha=0.7)
    
    # Mark current position
    ax1.plot(prediction['current_lon'], prediction['current_lat'], 'go', markersize=12,
            label='Current Position', zorder=5)
    
    # Mark predicted position
    ax1.plot(prediction['pred_lon'], prediction['pred_lat'], 'r*', markersize=15,
            label='Model Prediction', zorder=5)
    
    ax1.set_xlabel('Longitude', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Latitude', fontsize=11, fontweight='bold')
    ax1.set_title('Vessel Course & Trajectory', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Timeline with predictions
    times = trajectory['time_minutes'].values
    lats = trajectory['lat'].values
    lons = trajectory['lon'].values
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(times, lats, 'b-o', linewidth=2, markersize=6, label='Latitude')
    line2 = ax2_twin.plot(times, lons, 'r-s', linewidth=2, markersize=6, label='Longitude')
    
    ax2.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Latitude (degrees)', fontsize=11, fontweight='bold', color='b')
    ax2_twin.set_ylabel('Longitude (degrees)', fontsize=11, fontweight='bold', color='r')
    ax2.set_title('Position Over Time (30-Min Forecast)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='best', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'vessel_{vessel_mmsi}_verification.png', dpi=120, bbox_inches='tight')
    plt.close()
    
    return trajectory


def main():
    """Main end-to-end pipeline."""
    logger.info("\n" + "="*80)
    logger.info("END-TO-END VESSEL TRAJECTORY PIPELINE")
    logger.info("="*80)
    
    # Load model and preprocessing
    model, scaler, pca = load_model_and_preprocessing()
    
    # Load test data
    logger.info("\nLoading test data...")
    cache_file = 'results/cache/seq_cache_len12_sampled_3pct.npz'
    data = np.load(cache_file)
    X = data['X']
    y = data['y']
    mmsi_list = data['mmsi_list']
    
    n_train = int(0.7 * len(X))
    n_val = int(0.2 * len(X))
    
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]
    mmsi_test = mmsi_list[n_train+n_val:]
    
    logger.info(f"Test set: {X_test.shape[0]} sequences from {len(np.unique(mmsi_test))} vessels")
    
    # Create output directory
    output_dir = Path('results/end_to_end_pipeline')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select random vessels for demonstration
    unique_mmsi = np.unique(mmsi_test)
    selected_mmsi = np.random.choice(unique_mmsi, size=min(10, len(unique_mmsi)), replace=False)
    
    logger.info(f"\nProcessing {len(selected_mmsi)} random vessels...")
    
    results = []
    
    for mmsi in tqdm(selected_mmsi, desc="Processing vessels", unit="vessel"):
        mask = mmsi_test == mmsi
        indices = np.where(mask)[0]
        
        if len(indices) < 2:
            continue
        
        # Use last sequence for this vessel
        last_idx = indices[-1]
        X_sequence = X_test[last_idx]
        
        # Predict next position
        prediction = predict_next_position(X_sequence, model, scaler, pca, minutes_ahead=30)
        
        # Extrapolate trajectory
        trajectory = plot_verification(X_sequence, prediction, output_dir, mmsi)
        
        # Store results
        results.append({
            'MMSI': mmsi,
            'current_lat': prediction['current_lat'],
            'current_lon': prediction['current_lon'],
            'current_sog': prediction['current_sog'],
            'current_cog': prediction['current_cog'],
            'pred_lat': prediction['pred_lat'],
            'pred_lon': prediction['pred_lon'],
            'pred_sog': prediction['pred_sog'],
            'pred_cog': prediction['pred_cog'],
            'forecast_30min_lat': trajectory['lat'].iloc[-1],
            'forecast_30min_lon': trajectory['lon'].iloc[-1],
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'predictions_and_forecasts.csv', index=False)
    
    logger.info("\n" + "="*80)
    logger.info("[COMPLETE] END-TO-END PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Vessels processed: {len(results)}")
    logger.info(f"Verification plots: {len(results)}")
    logger.info(f"\nResults saved to: predictions_and_forecasts.csv")
    logger.info(f"\nSample Results:")
    logger.info(f"\n{results_df.head().to_string()}")


if __name__ == "__main__":
    main()

