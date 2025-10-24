"""
Interactive Prediction & Verification Tool
- User-friendly CLI for vessel trajectory prediction
- Commands: predict, verify, list_vessels, exit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


class VesselPredictionTool:
    """Interactive vessel trajectory prediction tool."""
    
    def __init__(self):
        """Initialize the tool."""
        self.model = None
        self.scaler = None
        self.pca = None
        self.X_test = None
        self.y_test = None
        self.mmsi_test = None
        self.unique_vessels = None
        self.output_dir = Path('results/interactive_predictions')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_resources()
    
    def _load_resources(self):
        """Load model and data."""
        print("\n" + "="*70)
        print("VESSEL TRAJECTORY PREDICTION & VERIFICATION TOOL")
        print("="*70)
        print("\n[Loading] Model and preprocessing objects...")
        
        try:
            with open('results/xgboost_advanced_50_vessels/xgboost_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('results/xgboost_advanced_50_vessels/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open('results/xgboost_advanced_50_vessels/pca.pkl', 'rb') as f:
                self.pca = pickle.load(f)
            print("[OK] Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False
        
        print("[Loading] Test data...")
        try:
            cache_file = 'results/cache/seq_cache_len12_sampled_3pct.npz'
            data = np.load(cache_file)
            X = data['X']
            y = data['y']
            mmsi_list = data['mmsi_list']
            
            n_train = int(0.7 * len(X))
            n_val = int(0.2 * len(X))
            
            self.X_test = X[n_train+n_val:]
            self.y_test = y[n_train+n_val:]
            self.mmsi_test = mmsi_list[n_train+n_val:]
            self.unique_vessels = np.unique(self.mmsi_test)
            
            print(f"[OK] Test data loaded: {len(self.X_test)} sequences, {len(self.unique_vessels)} vessels")
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            return False
        
        return True
    
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
    
    def _add_haversine_features(self, X):
        """Add 7 Haversine features."""
        n_samples = X.shape[0]
        haversine_features = []
        
        for i in range(n_samples):
            seq = X[i]
            lats = seq[:, 0]
            lons = seq[:, 1]
            
            R = 6371
            lat1_rad = np.radians(lats[0])
            lon1_rad = np.radians(lons[0])
            lats_rad = np.radians(lats)
            lons_rad = np.radians(lons)
            
            dlat = lats_rad - lat1_rad
            dlon = lons_rad - lon1_rad
            
            a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lats_rad) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            dist_to_first = R * c
            
            consecutive_dists = [0.0]
            for j in range(1, len(lats)):
                lat1, lon1 = lats_rad[j-1], lons_rad[j-1]
                lat2, lon2 = lats_rad[j], lons_rad[j]
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                consecutive_dists.append(R * c)
            
            haversine_features.append([
                np.mean(dist_to_first),
                np.max(dist_to_first),
                np.std(dist_to_first),
                np.sum(consecutive_dists),
                np.mean(consecutive_dists[1:]) if len(consecutive_dists) > 1 else 0,
                np.max(consecutive_dists),
                np.std(consecutive_dists)
            ])
        
        return np.array(haversine_features)
    
    def predict(self, mmsi, minutes_ahead=30):
        """Predict vessel position."""
        mask = self.mmsi_test == mmsi
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            print(f"\n[ERROR] Vessel {mmsi} not found in test set")
            return None
        
        last_idx = indices[-1]
        X_sequence = self.X_test[last_idx]
        
        # Extract features
        X_batch = X_sequence.reshape(1, 12, 28)
        X_features = self._extract_features(X_batch)
        X_haversine = self._add_haversine_features(X_batch)
        X_combined = np.hstack([X_features, X_haversine])
        
        # Preprocess
        X_scaled = self.scaler.transform(X_combined)
        X_pca = self.pca.transform(X_scaled)
        
        # Predict
        prediction = self.model.predict(X_pca)[0]
        
        current_lat = X_sequence[-1, 0]
        current_lon = X_sequence[-1, 1]
        current_sog = X_sequence[-1, 2]
        current_cog = X_sequence[-1, 3]
        
        return {
            'mmsi': mmsi,
            'current_lat': current_lat,
            'current_lon': current_lon,
            'current_sog': current_sog,
            'current_cog': current_cog,
            'pred_lat': prediction[0],
            'pred_lon': prediction[1],
            'pred_sog': prediction[2],
            'pred_cog': prediction[3],
            'X_sequence': X_sequence,
            'minutes_ahead': minutes_ahead
        }
    
    def verify(self, prediction):
        """Create verification plot."""
        X_sequence = prediction['X_sequence']
        last_5_lats = X_sequence[-5:, 0]
        last_5_lons = X_sequence[-5:, 1]
        
        # Extrapolate trajectory
        speed_km_per_min = prediction['current_sog'] * 1.852 / 60
        n_steps = int(prediction['minutes_ahead'] / 5) + 1
        
        trajectory_lats = [prediction['current_lat']]
        trajectory_lons = [prediction['current_lon']]
        
        lat, lon = prediction['current_lat'], prediction['current_lon']
        for step in range(1, n_steps):
            distance_km = speed_km_per_min * 5
            lat_change = distance_km / 111.0 * np.cos(np.radians(prediction['current_cog']))
            lon_change = distance_km / 111.0 * np.sin(np.radians(prediction['current_cog']))
            lat += lat_change
            lon += lon_change
            trajectory_lats.append(lat)
            trajectory_lons.append(lon)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Vessel {prediction["mmsi"]} - Prediction & Verification\n(Last 5 Points + {prediction["minutes_ahead"]}-Min Forecast)', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Course
        ax1.plot(last_5_lons, last_5_lats, 'b-o', linewidth=2.5, markersize=8, label='Last 5 Points', alpha=0.8)
        ax1.plot(trajectory_lons, trajectory_lats, 'r--s', linewidth=2, markersize=6, label=f'{prediction["minutes_ahead"]}-Min Forecast', alpha=0.7)
        ax1.plot(prediction['current_lon'], prediction['current_lat'], 'go', markersize=12, label='Current Position', zorder=5)
        ax1.plot(prediction['pred_lon'], prediction['pred_lat'], 'r*', markersize=15, label='Model Prediction', zorder=5)
        
        ax1.set_xlabel('Longitude', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Latitude', fontsize=11, fontweight='bold')
        ax1.set_title('Vessel Course & Trajectory', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Timeline
        times = np.arange(0, len(trajectory_lats)) * 5
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(times, trajectory_lats, 'b-o', linewidth=2, markersize=6, label='Latitude')
        line2 = ax2_twin.plot(times, trajectory_lons, 'r-s', linewidth=2, markersize=6, label='Longitude')
        
        ax2.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Latitude (degrees)', fontsize=11, fontweight='bold', color='b')
        ax2_twin.set_ylabel('Longitude (degrees)', fontsize=11, fontweight='bold', color='r')
        ax2.set_title('Position Over Time', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        ax2.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='best', fontsize=10)
        
        plt.tight_layout()
        filename = self.output_dir / f'vessel_{prediction["mmsi"]}_verification.png'
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def list_vessels(self):
        """List available vessels."""
        print(f"\n[Available Vessels] Total: {len(self.unique_vessels)}")
        print("─" * 50)
        for i, mmsi in enumerate(self.unique_vessels[:20], 1):
            count = np.sum(self.mmsi_test == mmsi)
            print(f"  {i:2d}. MMSI: {mmsi:10d} | Sequences: {count:6d}")
        if len(self.unique_vessels) > 20:
            print(f"  ... and {len(self.unique_vessels) - 20} more vessels")
    
    def run(self):
        """Run interactive CLI."""
        print("\n" + "="*70)
        print("COMMANDS:")
        print("  predict <mmsi> [minutes]  - Predict vessel position")
        print("  verify <mmsi> [minutes]   - Predict and plot verification")
        print("  list                      - List available vessels")
        print("  help                      - Show this help")
        print("  exit                      - Exit program")
        print("="*70 + "\n")
        
        while True:
            try:
                cmd = input("\n>>> ").strip().lower()
                
                if not cmd:
                    continue
                
                parts = cmd.split()
                action = parts[0]
                
                if action == 'exit':
                    print("\n[Goodbye] Thank you for using the Vessel Prediction Tool!")
                    break
                
                elif action == 'help':
                    print("\n" + "="*70)
                    print("COMMANDS:")
                    print("  predict <mmsi> [minutes]  - Predict vessel position")
                    print("  verify <mmsi> [minutes]   - Predict and plot verification")
                    print("  list                      - List available vessels")
                    print("  help                      - Show this help")
                    print("  exit                      - Exit program")
                    print("="*70)
                
                elif action == 'list':
                    self.list_vessels()
                
                elif action == 'predict':
                    if len(parts) < 2:
                        print("[ERROR] Usage: predict <mmsi> [minutes]")
                        continue
                    
                    mmsi = int(parts[1])
                    minutes = int(parts[2]) if len(parts) > 2 else 30
                    
                    print(f"\n[Predicting] Vessel {mmsi} position in {minutes} minutes...")
                    pred = self.predict(mmsi, minutes)
                    
                    if pred:
                        print(f"\n[Results]")
                        print(f"  Current Position:  LAT={pred['current_lat']:.4f}°, LON={pred['current_lon']:.4f}°")
                        print(f"  Current Speed:     {pred['current_sog']:.2f} knots")
                        print(f"  Current Course:    {pred['current_cog']:.2f}°")
                        print(f"  Predicted Position: LAT={pred['pred_lat']:.4f}°, LON={pred['pred_lon']:.4f}°")
                        print(f"  Predicted Speed:   {pred['pred_sog']:.2f} knots")
                        print(f"  Predicted Course:  {pred['pred_cog']:.2f}°")
                
                elif action == 'verify':
                    if len(parts) < 2:
                        print("[ERROR] Usage: verify <mmsi> [minutes]")
                        continue
                    
                    mmsi = int(parts[1])
                    minutes = int(parts[2]) if len(parts) > 2 else 30
                    
                    print(f"\n[Predicting] Vessel {mmsi}...")
                    pred = self.predict(mmsi, minutes)
                    
                    if pred:
                        print(f"[Plotting] Verification plot...")
                        filename = self.verify(pred)
                        print(f"[OK] Plot saved: {filename}")
                
                else:
                    print(f"[ERROR] Unknown command: {action}")
            
            except ValueError as e:
                print(f"[ERROR] Invalid input: {e}")
            except Exception as e:
                print(f"[ERROR] {e}")


if __name__ == "__main__":
    tool = VesselPredictionTool()
    tool.run()

