"""
Trajectory Consistency Verification Module

Verifies whether vessel movements are physically realistic and consistent.
Implements smoothness checks and anomaly detection based on kinematics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrajectoryVerifier:
    """Verifies trajectory consistency and detects anomalies."""
    
    # Physical constraints for vessels
    MAX_SPEED_KNOTS = 50  # Maximum realistic speed
    MAX_TURN_RATE = 45  # Maximum turn rate (degrees per minute)
    MAX_ACCELERATION = 2  # Maximum acceleration (knots per minute)
    
    def __init__(self, max_speed: float = MAX_SPEED_KNOTS,
                 max_turn_rate: float = MAX_TURN_RATE,
                 max_acceleration: float = MAX_ACCELERATION):
        """
        Initialize trajectory verifier.
        
        Args:
            max_speed: Maximum realistic speed (knots)
            max_turn_rate: Maximum turn rate (degrees/minute)
            max_acceleration: Maximum acceleration (knots/minute)
        """
        self.max_speed = max_speed
        self.max_turn_rate = max_turn_rate
        self.max_acceleration = max_acceleration
    
    def verify_trajectory(self, trajectory: pd.DataFrame) -> Dict:
        """
        Perform comprehensive trajectory verification.
        
        Args:
            trajectory: DataFrame with columns [LAT, LON, SOG, COG, BaseDateTime]
            
        Returns:
            Dictionary with verification results
        """
        results = {
            'smoothness_score': self._check_smoothness(trajectory),
            'speed_consistency': self._check_speed_consistency(trajectory),
            'heading_consistency': self._check_heading_consistency(trajectory),
            'acceleration_check': self._check_acceleration(trajectory),
            'turn_rate_check': self._check_turn_rate(trajectory),
            'anomalies': self._detect_anomalies(trajectory),
        }
        return results
    
    def _check_smoothness(self, trajectory: pd.DataFrame) -> float:
        """
        Check smoothness of trajectory using last 3 points.
        
        Calculates curvature and checks if movement is smooth.
        Returns score between 0 (not smooth) and 1 (very smooth).
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            Smoothness score
        """
        if len(trajectory) < 3:
            return 1.0  # Not enough points
        
        # Get last 3 points
        points = trajectory[['LAT', 'LON']].tail(3).values
        
        # Calculate vectors
        v1 = points[1] - points[0]
        v2 = points[2] - points[1]
        
        # Calculate angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        # Smoothness: 0° (straight) = 1.0, 180° (sharp turn) = 0.0
        smoothness = 1 - (angle / np.pi)
        return float(smoothness)
    
    def _check_speed_consistency(self, trajectory: pd.DataFrame) -> Dict:
        """
        Check if speed values are realistic and consistent.
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            Dictionary with speed checks
        """
        sog = trajectory['SOG'].values
        
        violations = {
            'exceeds_max_speed': (sog > self.max_speed).sum(),
            'negative_speed': (sog < 0).sum(),
            'speed_std': float(np.std(sog)),
            'speed_mean': float(np.mean(sog)),
        }
        
        return violations
    
    def _check_heading_consistency(self, trajectory: pd.DataFrame) -> Dict:
        """
        Check if heading/COG changes are realistic.
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            Dictionary with heading checks
        """
        if 'COG' not in trajectory.columns or len(trajectory) < 2:
            return {}
        
        cog = trajectory['COG'].values
        cog_diff = np.abs(np.diff(cog))
        
        # Handle 360° wraparound
        cog_diff = np.minimum(cog_diff, 360 - cog_diff)
        
        violations = {
            'max_heading_change': float(np.max(cog_diff)),
            'mean_heading_change': float(np.mean(cog_diff)),
            'sudden_turns': (cog_diff > self.max_turn_rate).sum(),
        }
        
        return violations
    
    def _check_acceleration(self, trajectory: pd.DataFrame) -> Dict:
        """
        Check if speed changes are physically realistic.
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            Dictionary with acceleration checks
        """
        if len(trajectory) < 2:
            return {}
        
        sog = trajectory['SOG'].values
        time_diff = np.diff(trajectory['BaseDateTime'].values) / np.timedelta64(1, 'm')
        
        # Avoid division by zero
        time_diff = np.maximum(time_diff, 1)
        
        acceleration = np.abs(np.diff(sog)) / time_diff
        
        violations = {
            'max_acceleration': float(np.max(acceleration)),
            'mean_acceleration': float(np.mean(acceleration)),
            'unrealistic_acceleration': (acceleration > self.max_acceleration).sum(),
        }
        
        return violations
    
    def _check_turn_rate(self, trajectory: pd.DataFrame) -> Dict:
        """
        Check if turn rates are physically realistic.
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            Dictionary with turn rate checks
        """
        if 'COG' not in trajectory.columns or len(trajectory) < 2:
            return {}
        
        cog = trajectory['COG'].values
        time_diff = np.diff(trajectory['BaseDateTime'].values) / np.timedelta64(1, 'm')
        
        # Avoid division by zero
        time_diff = np.maximum(time_diff, 1)
        
        cog_diff = np.abs(np.diff(cog))
        cog_diff = np.minimum(cog_diff, 360 - cog_diff)
        
        turn_rate = cog_diff / time_diff
        
        violations = {
            'max_turn_rate': float(np.max(turn_rate)),
            'mean_turn_rate': float(np.mean(turn_rate)),
            'unrealistic_turns': (turn_rate > self.max_turn_rate).sum(),
        }
        
        return violations
    
    def _detect_anomalies(self, trajectory: pd.DataFrame) -> List[Dict]:
        """
        Detect specific anomalies in trajectory.
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Check for speed anomalies
        speed_violations = self._check_speed_consistency(trajectory)
        if speed_violations['exceeds_max_speed'] > 0:
            anomalies.append({
                'type': 'speed_violation',
                'severity': 'high',
                'count': speed_violations['exceeds_max_speed'],
                'description': f"Speed exceeds {self.max_speed} knots"
            })
        
        # Check for turn rate anomalies
        turn_violations = self._check_turn_rate(trajectory)
        if turn_violations.get('unrealistic_turns', 0) > 0:
            anomalies.append({
                'type': 'turn_rate_violation',
                'severity': 'high',
                'count': turn_violations['unrealistic_turns'],
                'description': f"Turn rate exceeds {self.max_turn_rate}°/min"
            })
        
        # Check for acceleration anomalies
        accel_violations = self._check_acceleration(trajectory)
        if accel_violations.get('unrealistic_acceleration', 0) > 0:
            anomalies.append({
                'type': 'acceleration_violation',
                'severity': 'medium',
                'count': accel_violations['unrealistic_acceleration'],
                'description': f"Acceleration exceeds {self.max_acceleration} knots/min"
            })
        
        # Check for low smoothness
        smoothness = self._check_smoothness(trajectory)
        if smoothness < 0.3:
            anomalies.append({
                'type': 'low_smoothness',
                'severity': 'medium',
                'smoothness_score': smoothness,
                'description': "Trajectory shows sudden direction changes"
            })
        
        return anomalies
    
    def is_trajectory_valid(self, trajectory: pd.DataFrame) -> bool:
        """
        Determine if trajectory is valid (no critical anomalies).
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            True if trajectory is valid, False otherwise
        """
        anomalies = self._detect_anomalies(trajectory)
        
        # Check for high-severity anomalies
        high_severity = [a for a in anomalies if a.get('severity') == 'high']
        
        return len(high_severity) == 0
    
    def get_consistency_score(self, trajectory: pd.DataFrame) -> float:
        """
        Calculate overall consistency score (0-1).
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            Consistency score
        """
        results = self.verify_trajectory(trajectory)
        
        # Smoothness contributes 40%
        smoothness_score = results['smoothness_score'] * 0.4
        
        # Speed consistency contributes 30%
        speed_violations = results['speed_consistency']['exceeds_max_speed']
        speed_score = max(0, 1 - speed_violations / max(len(trajectory), 1)) * 0.3
        
        # Turn rate consistency contributes 30%
        turn_violations = results['turn_rate_check'].get('unrealistic_turns', 0)
        turn_score = max(0, 1 - turn_violations / max(len(trajectory), 1)) * 0.3
        
        total_score = smoothness_score + speed_score + turn_score
        return float(np.clip(total_score, 0, 1))

