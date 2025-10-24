"""
Trajectory Visualization Module
Plots predicted trajectories vs actual last 5 points with 30-minute intervals
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """
    Visualizes vessel trajectories with predictions and verification
    """
    
    def __init__(self, output_dir: str = "results/xgboost_predictions"):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color scheme
        self.colors = {
            'actual': '#00CC44',      # Green
            'predicted': '#FF9900',   # Orange
            'trajectory': '#00D9FF',  # Cyan
            'background': '#001F3F',  # Navy
            'grid': '#2C3E50'         # Steel gray
        }
    
    def plot_prediction_with_verification(self, 
                                         vessel_name: str,
                                         last_5_points: List[Dict],
                                         predicted_point: Dict,
                                         trajectory_points: List[Dict],
                                         confidence: float = 0.0,
                                         mmsi: Optional[int] = None) -> str:
        """
        Plot vessel trajectory with last 5 actual points and 30-minute prediction
        
        Args:
            vessel_name: Name of vessel
            last_5_points: List of last 5 actual points
            predicted_point: Predicted position
            trajectory_points: Intermediate trajectory points
            confidence: Confidence score (0-1)
            mmsi: MMSI number
        
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.patch.set_facecolor(self.colors['background'])
        
        # Extract data
        lats = [p['LAT'] for p in last_5_points]
        lons = [p['LON'] for p in last_5_points]
        sogs = [p['SOG'] for p in last_5_points]
        cogs = [p['COG'] for p in last_5_points]
        
        # Plot 1: Map view (LAT vs LON)
        ax = axes[0, 0]
        self._plot_map_view(ax, lats, lons, predicted_point, trajectory_points)
        ax.set_title(f"Vessel Trajectory Map\n{vessel_name} (MMSI: {mmsi})", 
                    fontsize=12, fontweight='bold', color='white')
        
        # Plot 2: Speed Over Ground (SOG)
        ax = axes[0, 1]
        self._plot_sog(ax, sogs, predicted_point.get('sog', 0))
        ax.set_title("Speed Over Ground (SOG)", fontsize=12, fontweight='bold', color='white')
        
        # Plot 3: Course Over Ground (COG)
        ax = axes[1, 0]
        self._plot_cog(ax, cogs, predicted_point.get('cog', 0))
        ax.set_title("Course Over Ground (COG)", fontsize=12, fontweight='bold', color='white')
        
        # Plot 4: Prediction Confidence & Info
        ax = axes[1, 1]
        self._plot_info_panel(ax, vessel_name, mmsi, confidence, last_5_points[-1], predicted_point)
        ax.set_title("Prediction Confidence & Info", fontsize=12, fontweight='bold', color='white')
        
        # Style all axes
        for ax in axes.flat:
            ax.set_facecolor(self.colors['grid'])
            ax.grid(True, alpha=0.3, color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
        
        plt.tight_layout()
        
        # Save figure
        filename = f"vessel_{vessel_name.replace(' ', '_')}_{mmsi}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, facecolor=self.colors['background'], edgecolor='white')
        logger.info(f"✅ Saved visualization to {filepath}")
        plt.close()
        
        return str(filepath)
    
    def _plot_map_view(self, ax, lats: List[float], lons: List[float], 
                      predicted_point: Dict, trajectory_points: List[Dict]):
        """Plot map view of trajectory"""
        # Plot actual points
        ax.plot(lons, lats, 'o-', color=self.colors['actual'], linewidth=2, 
               markersize=8, label='Actual Track', zorder=3)
        
        # Plot trajectory points
        traj_lons = [p['longitude'] for p in trajectory_points]
        traj_lats = [p['latitude'] for p in trajectory_points]
        ax.plot(traj_lons, traj_lats, '--', color=self.colors['trajectory'], 
               linewidth=2, alpha=0.7, label='Predicted Trajectory', zorder=2)
        
        # Plot predicted point
        ax.plot(predicted_point['longitude'], predicted_point['latitude'], 
               'X', color=self.colors['predicted'], markersize=15, 
               label='Predicted Position (30 min)', zorder=4)
        
        # Plot last point with arrow
        if len(lons) > 1:
            arrow = FancyArrowPatch((lons[-2], lats[-2]), (lons[-1], lats[-1]),
                                   arrowstyle='->', mutation_scale=20, 
                                   color=self.colors['actual'], linewidth=2, zorder=3)
            ax.add_patch(arrow)
        
        ax.set_xlabel('Longitude', color='white', fontsize=10)
        ax.set_ylabel('Latitude', color='white', fontsize=10)
        ax.legend(loc='best', facecolor=self.colors['grid'], edgecolor='white', 
                 labelcolor='white', fontsize=9)
    
    def _plot_sog(self, ax, sogs: List[float], predicted_sog: float):
        """Plot Speed Over Ground"""
        x = np.arange(len(sogs))
        ax.bar(x, sogs, color=self.colors['actual'], alpha=0.7, label='Actual SOG', width=0.6)
        
        # Add predicted SOG
        ax.axhline(y=predicted_sog, color=self.colors['predicted'], linestyle='--', 
                  linewidth=2, label=f'Predicted SOG: {predicted_sog:.2f} knots')
        
        ax.set_xlabel('Time Point', color='white', fontsize=10)
        ax.set_ylabel('SOG (knots)', color='white', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'T-{len(sogs)-i-1}' for i in range(len(sogs))], color='white')
        ax.legend(loc='best', facecolor=self.colors['grid'], edgecolor='white', 
                 labelcolor='white', fontsize=9)
    
    def _plot_cog(self, ax, cogs: List[float], predicted_cog: float):
        """Plot Course Over Ground"""
        x = np.arange(len(cogs))
        ax.plot(x, cogs, 'o-', color=self.colors['actual'], linewidth=2, 
               markersize=8, label='Actual COG')
        
        # Add predicted COG
        ax.axhline(y=predicted_cog, color=self.colors['predicted'], linestyle='--', 
                  linewidth=2, label=f'Predicted COG: {predicted_cog:.1f}°')
        
        ax.set_xlabel('Time Point', color='white', fontsize=10)
        ax.set_ylabel('COG (degrees)', color='white', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'T-{len(cogs)-i-1}' for i in range(len(cogs))], color='white')
        ax.set_ylim([0, 360])
        ax.legend(loc='best', facecolor=self.colors['grid'], edgecolor='white', 
                 labelcolor='white', fontsize=9)
    
    def _plot_info_panel(self, ax, vessel_name: str, mmsi: Optional[int], 
                        confidence: float, last_point: Dict, predicted_point: Dict):
        """Plot information panel"""
        ax.axis('off')
        
        # Prepare info text
        info_text = f"""
VESSEL INFORMATION
{'─' * 40}
Name: {vessel_name}
MMSI: {mmsi if mmsi else 'N/A'}

LAST KNOWN POSITION
{'─' * 40}
Latitude:  {last_point.get('LAT', 'N/A'):.6f}°
Longitude: {last_point.get('LON', 'N/A'):.6f}°
SOG: {last_point.get('SOG', 'N/A'):.2f} knots
COG: {last_point.get('COG', 'N/A'):.1f}°
Time: {str(last_point.get('BaseDateTime', 'N/A'))[:19]}

PREDICTED POSITION (30 min ahead)
{'─' * 40}
Latitude:  {predicted_point['latitude']:.6f}°
Longitude: {predicted_point['longitude']:.6f}°
SOG: {predicted_point['sog']:.2f} knots
COG: {predicted_point['cog']:.1f}°

PREDICTION QUALITY
{'─' * 40}
Confidence: {confidence*100:.1f}%
Model: XGBoost Advanced
Features: 483 engineered
PCA: 80 components
        """
        
        # Color confidence bar
        conf_color = self.colors['actual'] if confidence > 0.7 else \
                    self.colors['predicted'] if confidence > 0.5 else '#FF4444'
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               color='white', bbox=dict(boxstyle='round', facecolor=self.colors['grid'], 
                                       edgecolor='white', alpha=0.8))
        
        # Add confidence bar
        conf_rect = mpatches.Rectangle((0.05, 0.02), confidence * 0.9, 0.03, 
                                      transform=ax.transAxes, facecolor=conf_color, 
                                      edgecolor='white', linewidth=2)
        ax.add_patch(conf_rect)
    
    def plot_batch_predictions(self, predictions_list: List[Dict], 
                              output_prefix: str = "batch") -> List[str]:
        """
        Plot multiple vessel predictions
        
        Args:
            predictions_list: List of prediction dictionaries
            output_prefix: Prefix for output files
        
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        for i, pred in enumerate(predictions_list):
            try:
                filepath = self.plot_prediction_with_verification(
                    vessel_name=pred['vessel_name'],
                    last_5_points=pred['last_5_points'],
                    predicted_point=pred['predicted_position'],
                    trajectory_points=pred['trajectory_points'],
                    confidence=pred.get('confidence_score', 0.0),
                    mmsi=pred.get('mmsi')
                )
                saved_files.append(filepath)
                logger.info(f"✅ Plotted {i+1}/{len(predictions_list)}: {pred['vessel_name']}")
            except Exception as e:
                logger.error(f"❌ Error plotting {pred.get('vessel_name', 'Unknown')}: {e}")
        
        return saved_files

