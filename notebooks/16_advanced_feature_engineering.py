"""
Advanced Feature Engineering for LSTM Model
- Lag features (temporal dependencies)
- Rolling statistics (trend & volatility)
- Acceleration features (higher-order derivatives)
- Cyclical encoding (circular features)
- Polynomial features (non-linear relationships)
- Velocity components (movement representation)
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_engineering.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory
Path('logs').mkdir(exist_ok=True)


def add_lag_features(df, columns=['LAT', 'LON', 'SOG', 'COG'], lags=[1, 2, 3]):
    """Add lag features for temporal dependencies."""
    logger.info(f"\n{'='*70}\n[1] ADDING LAG FEATURES\n{'='*70}")
    
    for col in columns:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df.groupby('MMSI')[col].shift(lag).fillna(0)
            logger.info(f"  ✓ Added {col}_lag{lag}")
    
    logger.info(f"Total lag features added: {len(columns) * len(lags)}")
    return df


def add_rolling_statistics(df, columns=['LAT', 'LON', 'SOG', 'COG'], windows=[3, 5, 10]):
    """Add rolling statistics for trend and volatility."""
    logger.info(f"\n{'='*70}\n[2] ADDING ROLLING STATISTICS\n{'='*70}")
    
    for col in columns:
        for window in windows:
            # Rolling mean
            df[f'{col}_rolling_mean_{window}'] = df.groupby('MMSI')[col].rolling(window).mean().reset_index(drop=True).fillna(0)
            
            # Rolling std
            df[f'{col}_rolling_std_{window}'] = df.groupby('MMSI')[col].rolling(window).std().reset_index(drop=True).fillna(0)
            
            # Rolling max
            df[f'{col}_rolling_max_{window}'] = df.groupby('MMSI')[col].rolling(window).max().reset_index(drop=True).fillna(0)
            
            logger.info(f"  ✓ Added rolling stats for {col} (window={window})")
    
    logger.info(f"Total rolling features added: {len(columns) * len(windows) * 3}")
    return df


def add_acceleration_features(df):
    """Add acceleration features (higher-order derivatives)."""
    logger.info(f"\n{'='*70}\n[3] ADDING ACCELERATION FEATURES\n{'='*70}")
    
    # Speed acceleration
    df['speed_acceleration'] = df.groupby('MMSI')['speed_change'].diff().fillna(0)
    
    # Heading acceleration
    df['heading_acceleration'] = df.groupby('MMSI')['heading_change'].diff().fillna(0)
    
    # Lat/Lon acceleration
    df['lat_acceleration'] = df.groupby('MMSI')['lat_change'].diff().fillna(0)
    df['lon_acceleration'] = df.groupby('MMSI')['lon_change'].diff().fillna(0)
    
    logger.info("  ✓ Added speed_acceleration")
    logger.info("  ✓ Added heading_acceleration")
    logger.info("  ✓ Added lat_acceleration")
    logger.info("  ✓ Added lon_acceleration")
    
    return df


def add_cyclical_encoding(df):
    """Add cyclical encoding for circular features."""
    logger.info(f"\n{'='*70}\n[4] ADDING CYCLICAL ENCODING\n{'='*70}")
    
    # Hour cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week cyclical encoding
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Month cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    logger.info("  ✓ Added hour_sin, hour_cos")
    logger.info("  ✓ Added dow_sin, dow_cos")
    logger.info("  ✓ Added month_sin, month_cos")
    
    return df


def add_polynomial_features(df, columns=['LAT', 'LON', 'SOG', 'COG']):
    """Add polynomial features for non-linear relationships."""
    logger.info(f"\n{'='*70}\n[5] ADDING POLYNOMIAL FEATURES\n{'='*70}")
    
    for col in columns:
        df[f'{col}_squared'] = df[col] ** 2
        logger.info(f"  ✓ Added {col}_squared")
    
    return df


def add_velocity_components(df):
    """Add velocity components for better movement representation."""
    logger.info(f"\n{'='*70}\n[6] ADDING VELOCITY COMPONENTS\n{'='*70}")
    
    # Velocity components
    df['velocity_x'] = df['SOG'] * np.cos(np.radians(df['COG']))
    df['velocity_y'] = df['SOG'] * np.sin(np.radians(df['COG']))
    
    # Velocity magnitude (already SOG, but explicit)
    df['velocity_magnitude'] = df['SOG']
    
    # Velocity direction (already COG, but explicit)
    df['velocity_direction'] = df['COG']
    
    logger.info("  ✓ Added velocity_x, velocity_y")
    logger.info("  ✓ Added velocity_magnitude, velocity_direction")
    
    return df


def add_interaction_features(df):
    """Add interaction features for non-linear relationships."""
    logger.info(f"\n{'='*70}\n[7] ADDING INTERACTION FEATURES\n{'='*70}")
    
    # Speed × Heading interaction
    df['speed_heading_interaction'] = df['SOG'] * np.cos(np.radians(df['COG']))
    
    # Distance from origin
    df['distance_from_origin'] = np.sqrt(df['LAT']**2 + df['LON']**2)
    
    # Speed × Distance interaction
    df['speed_distance_interaction'] = df['SOG'] * df['distance_from_origin']
    
    logger.info("  ✓ Added speed_heading_interaction")
    logger.info("  ✓ Added distance_from_origin")
    logger.info("  ✓ Added speed_distance_interaction")
    
    return df


def apply_all_feature_engineering(df):
    """Apply all feature engineering techniques."""
    logger.info("\n" + "="*70)
    logger.info("ADVANCED FEATURE ENGINEERING")
    logger.info("="*70)
    
    # Apply all techniques
    df = add_lag_features(df, columns=['LAT', 'LON', 'SOG', 'COG'], lags=[1, 2, 3])
    df = add_rolling_statistics(df, columns=['LAT', 'LON', 'SOG', 'COG'], windows=[3, 5])
    df = add_acceleration_features(df)
    df = add_cyclical_encoding(df)
    df = add_polynomial_features(df, columns=['LAT', 'LON', 'SOG', 'COG'])
    df = add_velocity_components(df)
    df = add_interaction_features(df)
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"FEATURE ENGINEERING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Original features: 12")
    logger.info(f"New features: {len(df.columns) - 12}")
    logger.info(f"Total features: {len(df.columns)}")
    
    return df


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("ADVANCED FEATURE ENGINEERING MODULE")
    logger.info("="*70)
    logger.info("\nThis module provides advanced feature engineering techniques:")
    logger.info("  1. Lag features (temporal dependencies)")
    logger.info("  2. Rolling statistics (trend & volatility)")
    logger.info("  3. Acceleration features (higher-order derivatives)")
    logger.info("  4. Cyclical encoding (circular features)")
    logger.info("  5. Polynomial features (non-linear relationships)")
    logger.info("  6. Velocity components (movement representation)")
    logger.info("  7. Interaction features (feature combinations)")
    logger.info("\nUse apply_all_feature_engineering(df) to apply all techniques")

