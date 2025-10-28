"""
Maritime Vessel Prediction Pipeline - Streamlit App
Dedicated page for vessel trajectory predictions with XGBoost integration
"""

import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
import json
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Vessel Prediction Pipeline",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚢 Maritime Vessel Prediction Pipeline")
st.markdown("---")

# Backend URL
BACKEND_URL = "http://127.0.0.1:8000"

# Initialize session state
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = None

# ============================================================================
# SIDEBAR: Vessel Selection and Query Configuration
# ============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Fetch available vessels
    @st.cache_data(ttl=60)
    def fetch_all_vessels():
        try:
            r = requests.get(f"{BACKEND_URL}/vessels", timeout=5)
            return r.json().get("vessels", [])
        except Exception as e:
            st.error(f"Failed to fetch vessels: {e}")
            return []
    
    vessels = fetch_all_vessels()
    
    if not vessels:
        st.error("❌ No vessels available. Check backend connection.")
        st.stop()
    
    # Vessel selection
    selected_vessel = st.selectbox(
        "Select Vessel",
        options=vessels,
        help="Choose a vessel to predict"
    )
    
    st.markdown("---")
    
    # Prediction parameters
    st.markdown("### Prediction Parameters")
    
    prediction_minutes = st.slider(
        "Prediction Horizon (minutes)",
        min_value=5,
        max_value=240,
        value=30,
        step=5,
        help="How many minutes ahead to predict"
    )
    
    query_type = st.radio(
        "Query Type",
        options=["SHOW", "VERIFY", "PREDICT"],
        help="Type of query to execute"
    )
    
    st.markdown("---")
    
    # Advanced options
    with st.expander("Advanced Options"):
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_method = st.checkbox("Show Prediction Method", value=True)
        show_track = st.checkbox("Show Historical Track", value=True)

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Analysis", "🗺️ Map", "📈 History"])

# ============================================================================
# TAB 1: PREDICTION
# ============================================================================
with tab1:
    st.markdown("### Execute Prediction Query")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if query_type == "SHOW":
            query_text = f"Show {selected_vessel} position"
        elif query_type == "VERIFY":
            query_text = f"Verify {selected_vessel} course consistency"
        else:  # PREDICT
            query_text = f"Predict {selected_vessel} position after {prediction_minutes} minutes"
        
        st.text_input("Query", value=query_text, disabled=True, key="query_display")
    
    with col2:
        execute_button = st.button("🚀 Execute", use_container_width=True)
    
    st.markdown("---")
    
    # Execute query
    if execute_button:
        with st.spinner(f"Executing {query_type} query..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"text": query_text},
                    timeout=20
                )
                
                if response.status_code == 200:
                    data = response.json()
                    parsed = data.get("parsed", {})
                    result = data.get("response", {})
                    
                    st.session_state.current_prediction = {
                        "query": query_text,
                        "parsed": parsed,
                        "result": result,
                        "timestamp": datetime.now()
                    }
                    
                    st.success("✅ Query executed successfully!")
                else:
                    st.error(f"❌ Backend error: {response.status_code}")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Display results
    if st.session_state.current_prediction:
        pred = st.session_state.current_prediction
        result = pred["result"]
        
        st.markdown("### Results")
        
        # Check if result has data
        if isinstance(result, dict):
            if result.get("message") and "No data" in result.get("message", ""):
                st.warning("⚠️ " + result.get("message"))
            else:
                # Display vessel information
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Vessel Name",
                        result.get("vessel_name") or result.get("VesselName", "N/A")
                    )
                
                with col2:
                    st.metric(
                        "MMSI",
                        result.get("MMSI", "N/A")
                    )
                
                with col3:
                    st.metric(
                        "IMO",
                        result.get("IMO", "N/A")
                    )
                
                st.markdown("---")
                
                # Last position
                if result.get("last_position") or (result.get("LAT") and result.get("LON")):
                    st.markdown("#### 📍 Last Known Position")
                    
                    last_pos = result.get("last_position", {})
                    if not last_pos:
                        last_pos = {
                            "lat": result.get("LAT"),
                            "lon": result.get("LON"),
                            "sog": result.get("SOG"),
                            "cog": result.get("COG"),
                            "datetime": result.get("BaseDateTime")
                        }
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Latitude", f"{last_pos.get('lat', 'N/A'):.4f}")
                    
                    with col2:
                        st.metric("Longitude", f"{last_pos.get('lon', 'N/A'):.4f}")
                    
                    with col3:
                        st.metric("SOG (knots)", f"{last_pos.get('sog', 'N/A')}")
                    
                    with col4:
                        st.metric("COG (°)", f"{last_pos.get('cog', 'N/A')}")
                    
                    st.caption(f"Time: {last_pos.get('datetime', 'N/A')}")
                
                # Predicted position (for PREDICT queries)
                if result.get("predicted_position"):
                    st.markdown("---")
                    st.markdown("#### 🎯 Predicted Position")
                    
                    pred_pos = result.get("predicted_position", {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Latitude", f"{pred_pos.get('lat', 'N/A'):.4f}")
                    
                    with col2:
                        st.metric("Longitude", f"{pred_pos.get('lon', 'N/A'):.4f}")
                    
                    with col3:
                        st.metric("SOG (knots)", f"{pred_pos.get('sog', 'N/A')}")
                    
                    with col4:
                        st.metric("COG (°)", f"{pred_pos.get('cog', 'N/A')}")
                    
                    # Confidence and method
                    if show_confidence or show_method:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if show_confidence:
                                confidence = result.get("confidence", 0.7)
                                st.metric("Confidence", f"{confidence:.1%}")
                        
                        with col2:
                            if show_method:
                                method = result.get("method", "unknown")
                                st.metric("Method", method.replace("_", " ").title())
                    
                    st.caption(f"Prediction: {result.get('minutes_ahead', 'N/A')} minutes ahead")
        
        else:
            st.info("No prediction data available yet.")

# ============================================================================
# TAB 2: ANALYSIS
# ============================================================================
with tab2:
    st.markdown("### Prediction Analysis")
    
    if st.session_state.current_prediction:
        pred = st.session_state.current_prediction
        result = pred["result"]
        
        if result.get("last_position") and result.get("predicted_position"):
            last_pos = result.get("last_position", {})
            pred_pos = result.get("predicted_position", {})
            
            # Calculate distance
            import math
            
            lat1 = last_pos.get("lat", 0)
            lon1 = last_pos.get("lon", 0)
            lat2 = pred_pos.get("lat", 0)
            lon2 = pred_pos.get("lon", 0)
            
            # Haversine distance
            R = 3440.065  # Earth radius in nautical miles
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            distance_nm = R * c
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Distance Traveled", f"{distance_nm:.2f} NM")
            
            with col2:
                sog = last_pos.get("sog", 0)
                if sog > 0:
                    time_hours = distance_nm / sog
                    st.metric("Expected Time", f"{time_hours:.2f} hours")
            
            with col3:
                minutes = result.get("minutes_ahead", 30)
                st.metric("Prediction Window", f"{minutes} minutes")
            
            st.markdown("---")
            
            # Bearing change
            cog_last = last_pos.get("cog", 0)
            cog_pred = pred_pos.get("cog", 0)
            bearing_change = (cog_pred - cog_last) % 360
            
            st.metric("Course Change", f"{bearing_change:.1f}°")
            
            # Speed change
            sog_last = last_pos.get("sog", 0)
            sog_pred = pred_pos.get("sog", 0)
            speed_change = sog_pred - sog_last
            
            st.metric("Speed Change", f"{speed_change:+.1f} knots")
    
    else:
        st.info("Execute a prediction query first to see analysis.")

# ============================================================================
# TAB 3: MAP
# ============================================================================
with tab3:
    st.markdown("### Interactive Map")
    
    if st.session_state.current_prediction:
        pred = st.session_state.current_prediction
        result = pred["result"]
        
        last_pos = result.get("last_position") or {
            "lat": result.get("LAT"),
            "lon": result.get("LON")
        }
        
        if last_pos.get("lat") and last_pos.get("lon"):
            # Create map
            m = folium.Map(
                location=[last_pos.get("lat"), last_pos.get("lon")],
                zoom_start=10,
                tiles="OpenStreetMap"
            )
            
            # Add last position marker
            folium.Marker(
                location=[last_pos.get("lat"), last_pos.get("lon")],
                popup=f"Last Position: {selected_vessel}",
                icon=folium.Icon(color="blue", icon="info-sign"),
                tooltip="Last Known Position"
            ).add_to(m)
            
            # Add predicted position marker
            if result.get("predicted_position"):
                pred_pos = result.get("predicted_position", {})
                folium.Marker(
                    location=[pred_pos.get("lat"), pred_pos.get("lon")],
                    popup=f"Predicted Position: {selected_vessel}",
                    icon=folium.Icon(color="green", icon="arrow-right"),
                    tooltip="Predicted Position"
                ).add_to(m)
                
                # Draw line between positions
                folium.PolyLine(
                    locations=[
                        [last_pos.get("lat"), last_pos.get("lon")],
                        [pred_pos.get("lat"), pred_pos.get("lon")]
                    ],
                    color="red",
                    weight=2,
                    opacity=0.7,
                    popup="Predicted Track"
                ).add_to(m)
            
            # Add historical track if available
            if show_track and result.get("track"):
                track_points = result.get("track", [])
                if track_points:
                    track_coords = [
                        [p.get("LAT"), p.get("LON")] 
                        for p in track_points 
                        if p.get("LAT") and p.get("LON")
                    ]
                    
                    if track_coords:
                        folium.PolyLine(
                            locations=track_coords,
                            color="blue",
                            weight=1,
                            opacity=0.5,
                            popup="Historical Track"
                        ).add_to(m)
            
            st_folium(m, width=1200, height=600)
        
        else:
            st.warning("No position data available for map.")
    
    else:
        st.info("Execute a prediction query first to see the map.")

# ============================================================================
# TAB 4: HISTORY
# ============================================================================
with tab4:
    st.markdown("### Prediction History")
    
    if st.session_state.prediction_history:
        for i, pred in enumerate(st.session_state.prediction_history):
            with st.expander(f"Prediction {i+1}: {pred.get('query', 'N/A')}"):
                st.json(pred)
    
    else:
        st.info("No prediction history yet. Execute queries to build history.")

st.markdown("---")
st.caption("Maritime Vessel Prediction Pipeline | Backend: http://127.0.0.1:8000")

