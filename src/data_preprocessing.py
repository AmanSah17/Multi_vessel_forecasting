"""
Data Preprocessing Module for Maritime Vessel Forecasting

Handles:
- Missing vessel names
- Time series resampling to 1-minute intervals
- MMSI validation and formatting
- Outlier detection and removal
- Missing value imputation
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VesselDataPreprocessor:
    """Preprocesses raw AIS data for ML pipeline."""
    
    def __init__(self, resample_interval: str = '1min'):
        """
        Initialize preprocessor.
        
        Args:
            resample_interval: Resampling interval (default: '1min')
        """
        self.resample_interval = resample_interval
        self.mmsi_pattern = r'^\d{9}$'  # 9-digit MMSI
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main preprocessing pipeline.
        
        Args:
            df: Raw AIS dataframe
            
        Returns:
            Cleaned and preprocessed dataframe
        """
        logger.info("Starting data preprocessing...")
        
        # Step 1: Parse datetime
        df = self._parse_datetime(df)
        
        # Step 2: Handle missing vessel names
        df = self._handle_missing_vessel_names(df)
        
        # Step 3: Validate and clean MMSI
        df = self._validate_mmsi(df)
        
        # Step 4: Remove duplicates
        df = self._remove_duplicates(df)
        
        # Step 5: Resample to uniform intervals
        df = self._resample_timeseries(df)
        
        # Step 6: Handle missing values
        df = self._handle_missing_values(df)
        
        # Step 7: Remove outliers
        df = self._remove_outliers(df)
        
        logger.info("Preprocessing completed!")
        return df
    
    def _parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse BaseDateTime column."""
        if 'BaseDateTime' in df.columns:
            df['BaseDateTime'] = pd.to_datetime(
                df['BaseDateTime'],
                format='%Y-%m-%dT%H:%M:%S',
                errors='coerce'
            )
            df = df.dropna(subset=['BaseDateTime'])
        return df
    
    def _handle_missing_vessel_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace missing vessel names with 'Unidentified Vessel'."""
        if 'VesselName' in df.columns:
            df['VesselName'] = df['VesselName'].fillna('Unidentified Vessel')
            df['VesselName'] = df['VesselName'].replace('', 'Unidentified Vessel')
            logger.info(f"Unidentified vessels: {(df['VesselName'] == 'Unidentified Vessel').sum()}")
        return df
    
    def _validate_mmsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate MMSI format (9 digits)."""
        if 'MMSI' in df.columns:
            # Convert to string and check format
            df['MMSI'] = df['MMSI'].astype(str).str.strip()
            
            # Flag invalid MMSIs
            invalid_mask = ~df['MMSI'].str.match(self.mmsi_pattern)
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                logger.warning(f"Invalid MMSI format: {invalid_count} records")
                df = df[~invalid_mask]
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records."""
        initial_len = len(df)
        df = df.drop_duplicates(subset=['MMSI', 'BaseDateTime'], keep='first')
        removed = initial_len - len(df)
        logger.info(f"Removed {removed} duplicate records")
        return df
    
    def _resample_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample time series to uniform 1-minute intervals per vessel."""
        resampled_dfs = []
        
        for mmsi, group in df.groupby('MMSI'):
            group = group.set_index('BaseDateTime').sort_index()
            
            # Resample numeric columns
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            resampled = group[numeric_cols].resample(self.resample_interval).mean()
            
            # Forward fill for categorical columns
            for col in ['VesselName', 'IMO', 'CallSign', 'VesselType', 'Status']:
                if col in group.columns:
                    resampled[col] = group[col].resample(self.resample_interval).first()
            
            resampled['MMSI'] = mmsi
            resampled_dfs.append(resampled)
        
        result = pd.concat(resampled_dfs).reset_index()
        logger.info(f"Resampled to {self.resample_interval} intervals")
        return result
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with interpolation."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                # Linear interpolation for position/speed
                if col in ['LAT', 'LON', 'SOG', 'COG', 'Heading']:
                    df[col] = df.groupby('MMSI')[col].transform(
                        lambda x: x.interpolate(method='linear', limit_direction='both')
                    )
                else:
                    # Forward fill for other numeric columns
                    df[col] = df.groupby('MMSI')[col].transform(
                        lambda x: x.fillna(method='ffill').fillna(method='bfill')
                    )
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove unrealistic values."""
        initial_len = len(df)
        
        # Speed outliers (max ~50 knots for most vessels)
        if 'SOG' in df.columns:
            df = df[df['SOG'] <= 50]
        
        # Latitude/Longitude bounds
        if 'LAT' in df.columns:
            df = df[(df['LAT'] >= -90) & (df['LAT'] <= 90)]
        if 'LON' in df.columns:
            df = df[(df['LON'] >= -180) & (df['LON'] <= 180)]
        
        removed = initial_len - len(df)
        logger.info(f"Removed {removed} outlier records")
        return df


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """Convenience function to load and preprocess data."""
    df = pd.read_csv(filepath)
    preprocessor = VesselDataPreprocessor()
    return preprocessor.preprocess(df)

