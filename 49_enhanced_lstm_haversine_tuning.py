"""
Enhanced LSTM Pipeline with Haversine Distance & Hyperparameter Tuning

Features:
1. Comprehensive EDA with feature analysis
2. Clustering and PCA for feature engineering
3. Haversine distance metrics for evaluation (meters)
4. Hyperparameter tuning with Optuna
5. Early stopping with patience
6. Training curves plotted per epoch
7. Fine-tuning capabilities
8. MLflow tracking integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.pytorch
import optuna
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_lstm_haversine.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Setup MLflow
mlflow.set_experiment("Enhanced_LSTM_Haversine_Tuning")
mlflow.set_tracking_uri("file:./mlruns")


# ============================================================================
# HAVERSINE DISTANCE UTILITIES
# ============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate Haversine distance between two points.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in kilometers
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def calculate_haversine_errors(y_true, y_pred):
    """
    Calculate haversine distance errors between predicted and actual positions.
    
    Args:
        y_true: Actual values [LAT, LON, SOG, COG]
        y_pred: Predicted values [LAT, LON, SOG, COG]
    
    Returns:
        Dictionary with haversine error metrics in meters
    """
    # Extract lat/lon
    lat_true = y_true[:, 0]
    lon_true = y_true[:, 1]
    lat_pred = y_pred[:, 0]
    lon_pred = y_pred[:, 1]
    
    # Calculate haversine distances
    distances_km = haversine_distance(lat_true, lon_true, lat_pred, lon_pred)
    distances_m = distances_km * 1000  # Convert to meters
    
    return {
        'haversine_mean_m': np.mean(distances_m),
        'haversine_median_m': np.median(distances_m),
        'haversine_std_m': np.std(distances_m),
        'haversine_min_m': np.min(distances_m),
        'haversine_max_m': np.max(distances_m),
        'haversine_p95_m': np.percentile(distances_m, 95),
        'haversine_p99_m': np.percentile(distances_m, 99),
    }


# ============================================================================
# ENHANCED LSTM MODEL
# ============================================================================

class EnhancedLSTMModel(nn.Module):
    """Enhanced LSTM with configurable architecture."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=4, dropout=0.3, bidirectional=False):
        super(EnhancedLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_all_data(start_date=3, end_date=8, sample_per_day=None):
    """Load all CSV files from specified date range."""
    logger.info(f"\n{'='*70}\n[1/10] LOADING DATA (Jan {start_date}-{end_date})\n{'='*70}")
    
    base_path = Path(r"D:\Maritime_Vessel_monitoring\csv_extracted_data")
    dfs = []
    
    for day in range(start_date, end_date + 1):
        file_path = base_path / f"AIS_2020_01_{day:02d}" / f"AIS_2020_01_{day:02d}.csv"
        if file_path.exists():
            logger.info(f"Loading {file_path.name}...")
            df = pd.read_csv(file_path, usecols=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'VesselName'])
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], format="%Y-%m-%dT%H:%M:%S", errors='coerce')
            df = df.dropna(subset=['BaseDateTime', 'LAT', 'LON', 'SOG', 'COG'])
            
            if sample_per_day:
                df = df.sample(n=min(sample_per_day, len(df)), random_state=42)
            
            dfs.append(df)
            logger.info(f"  ✓ {len(df):,} records")
    
    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"\nTotal: {len(df_all):,} records, {df_all['MMSI'].nunique():,} vessels")
    return df_all


def perform_eda(df, output_dir='results/enhanced_lstm_haversine'):
    """Perform comprehensive EDA."""
    logger.info(f"\n{'='*70}\n[2/10] EXPLORATORY DATA ANALYSIS (EDA)\n{'='*70}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Basic statistics
    logger.info("\n📊 Data Statistics:")
    logger.info(f"  LAT - Mean: {df['LAT'].mean():.4f}, Std: {df['LAT'].std():.4f}")
    logger.info(f"  LON - Mean: {df['LON'].mean():.4f}, Std: {df['LON'].std():.4f}")
    logger.info(f"  SOG - Mean: {df['SOG'].mean():.4f}, Std: {df['SOG'].std():.4f}")
    logger.info(f"  COG - Mean: {df['COG'].mean():.4f}, Std: {df['COG'].std():.4f}")
    
    # Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('EDA - Feature Distributions', fontsize=16, fontweight='bold')
    
    axes[0, 0].hist(df['LAT'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Latitude Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('LAT')
    
    axes[0, 1].hist(df['LON'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Longitude Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('LON')
    
    axes[1, 0].hist(df['SOG'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Speed Over Ground Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('SOG (knots)')
    
    axes[1, 1].hist(df['COG'], bins=50, color='gold', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Course Over Ground Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('COG (degrees)')
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_eda_distributions.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_dir / '01_eda_distributions.png'}")
    plt.close()
    
    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_data = df[['LAT', 'LON', 'SOG', 'COG']].corr()
    sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '02_eda_correlation.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_dir / '02_eda_correlation.png'}")
    plt.close()


def prepare_features(df):
    """Add temporal and kinematic features."""
    logger.info(f"\n{'='*70}\n[3/10] PREPARING FEATURES\n{'='*70}")
    
    df = df.sort_values('BaseDateTime').reset_index(drop=True)
    df['hour'] = df['BaseDateTime'].dt.hour
    df['day_of_week'] = df['BaseDateTime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['BaseDateTime'].dt.month
    df['speed_change'] = df.groupby('MMSI')['SOG'].diff().fillna(0)
    df['heading_change'] = df.groupby('MMSI')['COG'].diff().fillna(0)
    df['lat_change'] = df.groupby('MMSI')['LAT'].diff().fillna(0)
    df['lon_change'] = df.groupby('MMSI')['LON'].diff().fillna(0)
    
    features = ['LAT', 'LON', 'SOG', 'COG', 'hour', 'day_of_week', 'is_weekend', 'month',
                'speed_change', 'heading_change', 'lat_change', 'lon_change']
    logger.info(f"Features ({len(features)}): {features}")
    return df, features

