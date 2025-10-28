"""
Map Prediction Visualizer
Generates interactive maps with vessel predictions and trajectories
"""

import folium
from folium import plugins
import json
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MapPredictionVisualizer:
    """Generate interactive maps for vessel predictions"""
    
    @staticmethod
    def create_prediction_map(
        prediction_data: Dict,
        output_path: str = None,
        zoom_start: int = 10
    ) -> Optional[str]:
        """
        Create interactive map with prediction and trajectory
        
        Args:
            prediction_data: Dictionary with prediction results
            output_path: Path to save HTML map
            zoom_start: Initial zoom level
        
        Returns:
            HTML string or path to saved file
        """
        try:
            # Extract data
            last_pos = prediction_data.get('last_position', {})
            pred_pos = prediction_data.get('predicted_position', {})
            trajectory = prediction_data.get('trajectory_points', [])
            vessel_name = prediction_data.get('vessel_name', 'Unknown')
            confidence = prediction_data.get('confidence', 0.95)
            
            # Calculate center
            center_lat = (last_pos.get('lat', 0) + pred_pos.get('lat', 0)) / 2
            center_lon = (last_pos.get('lon', 0) + pred_pos.get('lon', 0)) / 2
            
            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=zoom_start,
                tiles='OpenStreetMap'
            )
            
            # Add last known position
            folium.CircleMarker(
                location=[last_pos.get('lat', 0), last_pos.get('lon', 0)],
                radius=8,
                popup=f"<b>{vessel_name}</b><br>Last Position<br>SOG: {last_pos.get('sog', 0):.2f} kts<br>COG: {last_pos.get('cog', 0):.2f}°",
                color='green',
                fill=True,
                fillColor='green',
                fillOpacity=0.8,
                weight=2
            ).add_to(m)
            
            # Add predicted position
            folium.CircleMarker(
                location=[pred_pos.get('lat', 0), pred_pos.get('lon', 0)],
                radius=8,
                popup=f"<b>{vessel_name}</b><br>Predicted Position<br>Confidence: {confidence*100:.1f}%<br>SOG: {pred_pos.get('sog', 0):.2f} kts<br>COG: {pred_pos.get('cog', 0):.2f}°",
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.8,
                weight=2
            ).add_to(m)
            
            # Add trajectory line
            if trajectory:
                trajectory_coords = [
                    [point['lat'], point['lon']]
                    for point in sorted(trajectory, key=lambda x: x['order'])
                ]
                
                folium.PolyLine(
                    trajectory_coords,
                    color='blue',
                    weight=2,
                    opacity=0.7,
                    popup=f"Predicted trajectory for {vessel_name}"
                ).add_to(m)
                
                # Add intermediate points
                for point in trajectory:
                    if point['order'] > 0 and point['order'] < len(trajectory) - 1:
                        folium.CircleMarker(
                            location=[point['lat'], point['lon']],
                            radius=4,
                            popup=f"T+{point['minutes_ahead']} min",
                            color='blue',
                            fill=True,
                            fillColor='lightblue',
                            fillOpacity=0.6,
                            weight=1
                        ).add_to(m)
            
            # Add arrow from last to predicted
            folium.plugins.AntPath(
                locations=[
                    [last_pos.get('lat', 0), last_pos.get('lon', 0)],
                    [pred_pos.get('lat', 0), pred_pos.get('lon', 0)]
                ],
                color='orange',
                weight=3,
                opacity=0.8,
                popup=f"Prediction vector"
            ).add_to(m)
            
            # Add legend
            legend_html = f"""
            <div style="position: fixed; 
                     bottom: 50px; right: 50px; width: 250px; height: auto; 
                     background-color: white; border:2px solid grey; z-index:9999; 
                     font-size:14px; padding: 10px">
                <b>{vessel_name}</b><br>
                <hr>
                <i style="background:green"></i> Last Known Position<br>
                <i style="background:red"></i> Predicted Position<br>
                <i style="background:blue"></i> Trajectory<br>
                <i style="background:orange"></i> Prediction Vector<br>
                <hr>
                <b>Confidence:</b> {confidence*100:.1f}%<br>
                <b>Method:</b> {prediction_data.get('method', 'XGBoost')}<br>
                <b>Minutes Ahead:</b> {prediction_data.get('minutes_ahead', 30)}
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Save or return
            if output_path:
                m.save(output_path)
                logger.info(f"✅ Map saved to {output_path}")
                return output_path
            else:
                return m._repr_html_()
            
        except Exception as e:
            logger.error(f"❌ Error creating map: {e}")
            return None
    
    @staticmethod
    def create_multi_vessel_map(
        predictions_list: List[Dict],
        output_path: str = None,
        zoom_start: int = 8
    ) -> Optional[str]:
        """
        Create map with multiple vessel predictions
        
        Args:
            predictions_list: List of prediction dictionaries
            output_path: Path to save HTML map
            zoom_start: Initial zoom level
        
        Returns:
            HTML string or path to saved file
        """
        try:
            if not predictions_list:
                return None
            
            # Calculate center
            all_lats = []
            all_lons = []
            
            for pred in predictions_list:
                last_pos = pred.get('last_position', {})
                pred_pos = pred.get('predicted_position', {})
                all_lats.extend([last_pos.get('lat', 0), pred_pos.get('lat', 0)])
                all_lons.extend([last_pos.get('lon', 0), pred_pos.get('lon', 0)])
            
            center_lat = sum(all_lats) / len(all_lats) if all_lats else 0
            center_lon = sum(all_lons) / len(all_lons) if all_lons else 0
            
            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=zoom_start,
                tiles='OpenStreetMap'
            )
            
            # Add each vessel
            colors = ['green', 'blue', 'red', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen']
            
            for idx, pred in enumerate(predictions_list):
                color = colors[idx % len(colors)]
                vessel_name = pred.get('vessel_name', f'Vessel {idx}')
                
                last_pos = pred.get('last_position', {})
                pred_pos = pred.get('predicted_position', {})
                
                # Last position
                folium.CircleMarker(
                    location=[last_pos.get('lat', 0), last_pos.get('lon', 0)],
                    radius=6,
                    popup=f"<b>{vessel_name}</b><br>Last Position",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.8,
                    weight=2
                ).add_to(m)
                
                # Predicted position
                folium.CircleMarker(
                    location=[pred_pos.get('lat', 0), pred_pos.get('lon', 0)],
                    radius=6,
                    popup=f"<b>{vessel_name}</b><br>Predicted Position",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.4,
                    weight=2
                ).add_to(m)
                
                # Connection line
                folium.PolyLine(
                    [
                        [last_pos.get('lat', 0), last_pos.get('lon', 0)],
                        [pred_pos.get('lat', 0), pred_pos.get('lon', 0)]
                    ],
                    color=color,
                    weight=2,
                    opacity=0.6
                ).add_to(m)
            
            # Save or return
            if output_path:
                m.save(output_path)
                logger.info(f"✅ Multi-vessel map saved to {output_path}")
                return output_path
            else:
                return m._repr_html_()
            
        except Exception as e:
            logger.error(f"❌ Error creating multi-vessel map: {e}")
            return None
    
    @staticmethod
    def create_trajectory_comparison_map(
        vessel_name: str,
        historical_track: List[Dict],
        predicted_trajectory: List[Dict],
        output_path: str = None,
        zoom_start: int = 10
    ) -> Optional[str]:
        """
        Create map comparing historical track with predicted trajectory
        
        Args:
            vessel_name: Vessel name
            historical_track: List of historical positions
            predicted_trajectory: List of predicted positions
            output_path: Path to save HTML map
            zoom_start: Initial zoom level
        
        Returns:
            HTML string or path to saved file
        """
        try:
            if not historical_track or not predicted_trajectory:
                return None
            
            # Calculate center
            all_lats = [p.get('LAT', p.get('lat', 0)) for p in historical_track + predicted_trajectory]
            all_lons = [p.get('LON', p.get('lon', 0)) for p in historical_track + predicted_trajectory]
            
            center_lat = sum(all_lats) / len(all_lats) if all_lats else 0
            center_lon = sum(all_lons) / len(all_lons) if all_lons else 0
            
            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=zoom_start,
                tiles='OpenStreetMap'
            )
            
            # Add historical track
            hist_coords = [
                [p.get('LAT', p.get('lat', 0)), p.get('LON', p.get('lon', 0))]
                for p in historical_track
            ]
            
            folium.PolyLine(
                hist_coords,
                color='green',
                weight=2,
                opacity=0.7,
                popup=f"Historical track for {vessel_name}"
            ).add_to(m)
            
            # Add predicted trajectory
            pred_coords = [
                [p.get('lat', 0), p.get('lon', 0)]
                for p in predicted_trajectory
            ]
            
            folium.PolyLine(
                pred_coords,
                color='red',
                weight=2,
                opacity=0.7,
                popup=f"Predicted trajectory for {vessel_name}"
            ).add_to(m)
            
            # Save or return
            if output_path:
                m.save(output_path)
                logger.info(f"✅ Trajectory comparison map saved to {output_path}")
                return output_path
            else:
                return m._repr_html_()
            
        except Exception as e:
            logger.error(f"❌ Error creating trajectory comparison map: {e}")
            return None

