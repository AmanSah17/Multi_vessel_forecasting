"""
XGBoost FastAPI Server
Provides REST API for XGBoost model predictions
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from xgboost_backend_integration import XGBoostBackendPredictor, VesselPredictionEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="XGBoost Vessel Prediction Server",
    description="ML-based vessel position prediction service",
    version="1.0.0"
)

# Global predictor instance
predictor = None
prediction_engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global predictor, prediction_engine
    
    try:
        logger.info("🚀 Initializing XGBoost Model Server...")
        
        # Initialize predictor
        model_dir = os.path.join(
            os.path.dirname(__file__),
            "results",
            "xgboost_advanced_50_vessels"
        )
        
        predictor = XGBoostBackendPredictor(model_dir)
        
        if predictor.is_loaded:
            logger.info("✅ XGBoost model loaded successfully")
            logger.info(f"📊 Model status: {predictor.get_status()}")
        else:
            logger.warning("⚠️  XGBoost model failed to load")
            
    except Exception as e:
        logger.error(f"❌ Error initializing model: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor.is_loaded if predictor else False
    }


@app.get("/model/status")
async def model_status():
    """Get model status"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    return predictor.get_status()


class PredictionRequest(BaseModel):
    """Prediction request model"""
    features: List[float]
    vessel_name: Optional[str] = None
    mmsi: Optional[int] = None


class PredictionResponse(BaseModel):
    """Prediction response model"""
    predicted_lat: float
    predicted_lon: float
    predicted_sog: float
    predicted_cog: float
    confidence: float
    vessel_name: Optional[str] = None
    mmsi: Optional[int] = None


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction
    
    Args:
        request: PredictionRequest with features
    
    Returns:
        PredictionResponse with predictions
    """
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import numpy as np
        
        # Convert features to numpy array
        X = np.array(request.features).reshape(1, -1)
        
        # Apply PCA
        X_pca = predictor.pca.transform(X)
        
        # Make prediction
        predictions = predictor.model.predict(X_pca)
        pred_lat, pred_lon, pred_sog, pred_cog = predictions[0]
        
        return PredictionResponse(
            predicted_lat=float(pred_lat),
            predicted_lon=float(pred_lon),
            predicted_sog=float(pred_sog),
            predicted_cog=float(pred_cog),
            confidence=0.95,
            vessel_name=request.vessel_name,
            mmsi=request.mmsi
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
async def info():
    """Get server information"""
    return {
        "name": "XGBoost Vessel Prediction Server",
        "version": "1.0.0",
        "model": "XGBoost MultiOutputRegressor",
        "features": 80,  # After PCA
        "outputs": 4,  # LAT, LON, SOG, COG
        "confidence": 0.95,
        "endpoints": {
            "/health": "Health check",
            "/model/status": "Model status",
            "/predict": "Make prediction",
            "/info": "Server information",
            "/docs": "API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting XGBoost FastAPI Server...")
    logger.info("📡 Server will be available at http://127.0.0.1:8001")
    logger.info("📚 API docs available at http://127.0.0.1:8001/docs")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )

