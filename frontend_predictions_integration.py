"""
Enhanced Frontend Integration for Predictions
Integrates XGBoost predictions with Streamlit frontend
"""

import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Maritime Vessel Predictions",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚓ Maritime Vessel Prediction Dashboard")
st.markdown("**Real-time vessel position tracking and XGBoost-powered predictions**")

# Backend URLs
BACKEND_URL = "http://127.0.0.1:8000"
XGBOOST_URL = "http://127.0.0.1:8001"

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Configuration")
    
    # Get vessels from database
    @st.cache_data(ttl=60)
    def get_vessels():
        try:
            r = requests.get(f"{BACKEND_URL}/vessels", timeout=10)
            return r.json().get("vessels", [])
        except Exception as e:
            st.error(f"Error fetching vessels: {e}")
            return []
    
    vessels = get_vessels()
    
    if vessels:
        selected_vessel = st.selectbox(
            "Select Vessel",
            options=vessels,
            help="Choose a vessel from the database"
        )
        
        st.markdown("### 📊 Query Options")
        query_type = st.radio(
            "Query Type",
            options=["Show Position", "Verify Course", "Predict Position"],
            help="Select the type of query to execute"
        )
        
        if query_type == "Predict Position":
            minutes_ahead = st.slider(
                "Predict position after (minutes)",
                min_value=5,
                max_value=120,
                value=30,
                step=5
            )
        else:
            minutes_ahead = None
        
        st.markdown("---")
        st.markdown("### 📈 Model Info")
        st.info("""
        **XGBoost Model**
        - Latitude MAE: 0.3056°
        - Longitude MAE: 1.1040°
        - Confidence: 95%
        - Features: 483 → 80 (PCA)
        """)
    else:
        st.warning("No vessels found in database")
        selected_vessel = None

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🗺️ Vessel Map")
    map_placeholder = st.empty()

with col2:
    st.markdown("### 📋 Details")
    details_placeholder = st.empty()

# Query execution
if selected_vessel:
    st.markdown("---")
    st.markdown("### 🔍 Execute Query")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Execute Query", key="execute_btn", use_container_width=True):
            # Build query
            if query_type == "Show Position":
                query_text = f"Show {selected_vessel} position"
            elif query_type == "Verify Course":
                query_text = f"Verify {selected_vessel} course"
            else:  # Predict Position
                query_text = f"Predict {selected_vessel} position after {minutes_ahead} minutes"
            
            st.info(f"📤 Sending query: **{query_text}**")
            
            try:
                # Send query to backend
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"text": query_text},
                    timeout=20
                )
                
                if response.status_code == 200:
                    data = response.json()
                    parsed = data.get("parsed", {})
                    result = data.get("response", {})
                    
                    st.session_state.last_response = result
                    st.session_state.prediction_data = {
                        "query": query_text,
                        "intent": parsed.get("intent"),
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    st.success("✅ Query executed successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Backend error: {response.status_code}")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Display results
if st.session_state.last_response:
    result = st.session_state.last_response
    
    with details_placeholder.container():
        st.markdown("#### Vessel Information")
        
        if "error" not in result:
            # Display vessel info
            vessel_name = result.get("VesselName", "Unknown")
            mmsi = result.get("MMSI", "N/A")
            
            st.write(f"**Vessel**: {vessel_name}")
            st.write(f"**MMSI**: {mmsi}")
            
            # Display position
            if "LAT" in result and "LON" in result:
                lat = result.get("LAT")
                lon = result.get("LON")
                sog = result.get("SOG", "N/A")
                cog = result.get("COG", "N/A")
                
                st.write(f"**Position**: {lat}°, {lon}°")
                st.write(f"**Speed**: {sog} kts")
                st.write(f"**Course**: {cog}°")
            
            # Display prediction if available
            if "predicted_position" in result:
                pred = result.get("predicted_position", {})
                pred_lat = pred.get("lat", "N/A")
                pred_lon = pred.get("lon", "N/A")
                confidence = result.get("confidence", "N/A")
                method = result.get("method", "N/A")
                
                st.markdown("#### Prediction")
                st.write(f"**Predicted Position**: {pred_lat}°, {pred_lon}°")
                st.write(f"**Confidence**: {confidence*100:.1f}%" if isinstance(confidence, (int, float)) else f"**Confidence**: {confidence}")
                st.write(f"**Method**: {method}")
            
            # Display track
            if "track" in result and result["track"]:
                track = result["track"]
                st.markdown(f"#### Track History ({len(track)} points)")
                
                # Create dataframe
                track_df = pd.DataFrame(track)
                st.dataframe(track_df[["LAT", "LON", "SOG", "COG"]], use_container_width=True)
        
        else:
            st.error(f"Error: {result.get('error', 'Unknown error')}")
    
    # Display map
    with map_placeholder.container():
        try:
            # Create map
            if "LAT" in result and "LON" in result:
                lat = result.get("LAT")
                lon = result.get("LON")
                
                m = folium.Map(
                    location=[lat, lon],
                    zoom_start=10,
                    tiles="OpenStreetMap"
                )
                
                # Add current position
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    color="green",
                    fill=True,
                    fillColor="green",
                    popup=f"<b>{result.get('VesselName', 'Vessel')}</b><br>Current Position",
                    tooltip="Current Position"
                ).add_to(m)
                
                # Add predicted position if available
                if "predicted_position" in result:
                    pred = result.get("predicted_position", {})
                    pred_lat = pred.get("lat")
                    pred_lon = pred.get("lon")
                    
                    if pred_lat and pred_lon:
                        folium.CircleMarker(
                            location=[pred_lat, pred_lon],
                            radius=8,
                            color="red",
                            fill=True,
                            fillColor="red",
                            popup=f"<b>{result.get('VesselName', 'Vessel')}</b><br>Predicted Position",
                            tooltip="Predicted Position"
                        ).add_to(m)
                        
                        # Add line between current and predicted
                        folium.PolyLine(
                            locations=[[lat, lon], [pred_lat, pred_lon]],
                            color="orange",
                            weight=2,
                            opacity=0.7
                        ).add_to(m)
                
                # Add track
                if "track" in result and result["track"]:
                    track = result["track"]
                    track_coords = [[p.get("LAT"), p.get("LON")] for p in track if "LAT" in p and "LON" in p]
                    
                    if track_coords:
                        folium.PolyLine(
                            locations=track_coords,
                            color="blue",
                            weight=2,
                            opacity=0.5
                        ).add_to(m)
                
                # Display map
                st_folium(m, width=700, height=500)
        
        except Exception as e:
            st.error(f"Error rendering map: {e}")

# Chat history
st.markdown("---")
st.markdown("### 💬 Query History")

if st.session_state.chat_history:
    for i, (role, message) in enumerate(st.session_state.chat_history):
        if role == "User":
            st.write(f"👤 **User**: {message}")
        else:
            st.write(f"🤖 **Bot**: {message}")
else:
    st.info("No queries yet. Execute a query above to see history.")

