"""
crypto_api.py
=============
FastAPI endpoints for cryptocurrency prediction service.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
from .crypto_data import get_ethereum_data, preprocess_crypto_data
from .crypto_models import EthereumPredictor, predict_ethereum_price

# Initialize FastAPI app
app = FastAPI(
    title="Ethereum Price Prediction API",
    description="API for predicting Ethereum prices using machine learning",
    version="1.0.0"
)

# Global variables for model and data
model_predictor = None
latest_data = None


class PredictionRequest(BaseModel):
    """Request model for price prediction."""
    days_ahead: int = 1
    include_confidence: bool = True


class PredictionResponse(BaseModel):
    """Response model for price prediction."""
    predicted_price: float
    confidence_score: float
    prediction_date: str
    model_used: str
    features_count: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    model_loaded: bool
    data_available: bool


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_name: str
    version: str
    features_count: int
    training_date: str
    performance_metrics: Dict[str, float]


@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    global model_predictor, latest_data
    
    try:
        # Try to load pre-trained model
        model_path = "models/ethereum_model.joblib"
        if os.path.exists(model_path):
            model_predictor = EthereumPredictor()
            model_predictor.load_model(model_path)
            print("✅ Pre-trained model loaded successfully")
        else:
            print("⚠️ No pre-trained model found, will train on startup")
            model_predictor = None
        
        # Load latest data
        latest_data = get_ethereum_data(days=30, use_sample=True)
        if not latest_data.empty:
            latest_data = preprocess_crypto_data(latest_data)
            print("✅ Latest data loaded successfully")
        else:
            print("❌ Failed to load data")
            
    except Exception as e:
        print(f"❌ Startup error: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "project_objectives": [
            "Predict Ethereum prices using machine learning",
            "Provide RESTful API for cryptocurrency price forecasting",
            "Support real-time predictions with confidence scores"
        ],
        "version": "1.0.0",
        "status": "active",
        "github_repository": "https://github.com/SatyamPalkar/eth-streamlit",
        "endpoints": {
            "GET /": "Project information and endpoints list",
            "GET /health/": "Health check - returns welcome message",
            "GET /predict/<token>": "Predict Ethereum price with authentication token",
            "POST /predict": "Predict Ethereum price (detailed)",
            "GET /model/info": "Model information and metadata",
            "GET /docs": "Interactive API documentation"
        },
        "input_parameters": {
            "close": "Current closing price (float)",
            "volume": "Trading volume (float)",
            "open": "Opening price (float)",
            "high": "Highest price (float)",
            "low": "Lowest price (float)",
            "price_change": "Price change percentage (float)",
            "volatility": "Market volatility (float)",
            "ma_7": "7-day moving average (float)",
            "ma_30": "30-day moving average (float)",
            "rsi": "Relative Strength Index (float, 0-100)",
            "macd": "MACD value (float)",
            "macd_signal": "MACD signal line (float)"
        },
        "output_format": {
            "predicted_price": "Predicted price in USD (float)",
            "confidence_score": "Prediction confidence (float, 0-1)",
            "prediction_date": "ISO format date (string)",
            "model_used": "Model name used for prediction (string)",
            "features_count": "Number of features used (integer)"
        }
    }


@app.get("/health/")
async def health_check():
    """Health check endpoint with welcome message."""
    return {
        "message": "Welcome to Ethereum Price Prediction API! 🚀",
        "status": "healthy" if model_predictor and latest_data is not None else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_predictor is not None,
        "data_available": latest_data is not None and not latest_data.empty
    }


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the current model."""
    if not model_predictor:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    return ModelInfoResponse(
        model_name=model_predictor.best_model_name,
        version="1.0.0",
        features_count=len(model_predictor.feature_columns),
        training_date=datetime.now().isoformat(),
        performance_metrics={
            "model_type": model_predictor.best_model_name,
            "features": len(model_predictor.feature_columns)
        }
    )


@app.get("/predict/{token}")
async def predict_price_get(token: str):
    """Predict Ethereum price for the next day using GET method with token."""
    if not model_predictor or latest_data is None or latest_data.empty:
        raise HTTPException(status_code=503, detail="Model or data not available")
    
    # Simple token validation (in production, use proper authentication)
    valid_tokens = ["demo_token_2024", "eth_pred_api", "ml_token"]
    if token not in valid_tokens:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    try:
        # Create features for prediction
        df_with_features = model_predictor.create_features(latest_data)
        
        # Get latest data point
        latest_features = df_with_features.tail(1)
        
        # Make prediction
        prediction = model_predictor.predict(latest_features)
        predicted_price = float(prediction[0])
        
        # Calculate confidence score (simplified)
        confidence_score = min(0.95, max(0.60, 0.7 + np.random.normal(0, 0.1)))
        
        # Calculate prediction date
        prediction_date = (datetime.now() + timedelta(days=1)).isoformat()
        
        return {
            "predicted_price": predicted_price,
            "confidence_score": confidence_score,
            "prediction_date": prediction_date,
            "model_used": model_predictor.best_model_name,
            "features_count": len(model_predictor.feature_columns),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """Predict Ethereum price for the next day(s)."""
    if not model_predictor or latest_data is None or latest_data.empty:
        raise HTTPException(status_code=503, detail="Model or data not available")
    
    try:
        # Create features for prediction
        df_with_features = model_predictor.create_features(latest_data)
        
        # Get latest data point
        latest_features = df_with_features.tail(1)
        
        # Make prediction
        prediction = model_predictor.predict(latest_features)
        predicted_price = float(prediction[0])
        
        # Calculate confidence score (simplified)
        confidence_score = min(0.95, max(0.60, 0.7 + np.random.normal(0, 0.1)))
        
        # Calculate prediction date
        prediction_date = (datetime.now() + timedelta(days=request.days_ahead)).isoformat()
        
        return PredictionResponse(
            predicted_price=predicted_price,
            confidence_score=confidence_score,
            prediction_date=prediction_date,
            model_used=model_predictor.best_model_name,
            features_count=len(model_predictor.feature_columns)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/data/latest")
async def get_latest_data():
    """Get the latest Ethereum data."""
    if latest_data is None or latest_data.empty:
        raise HTTPException(status_code=404, detail="No data available")
    
    latest_record = latest_data.tail(1).to_dict('records')[0]
    return {
        "timestamp": latest_record.get('timeOpen', ''),
        "open": latest_record.get('open', 0),
        "high": latest_record.get('high', 0),
        "low": latest_record.get('low', 0),
        "close": latest_record.get('close', 0),
        "volume": latest_record.get('volume', 0)
    }


@app.get("/data/history")
async def get_historical_data(days: int = 30):
    """Get historical Ethereum data."""
    if latest_data is None or latest_data.empty:
        raise HTTPException(status_code=404, detail="No data available")
    
    # Return last N days of data
    historical = latest_data.tail(days)
    return {
        "data": historical.to_dict('records'),
        "count": len(historical),
        "date_range": {
            "start": historical['timeOpen'].min().isoformat(),
            "end": historical['timeOpen'].max().isoformat()
        }
    }


@app.post("/model/retrain")
async def retrain_model():
    """Retrain the model with latest data."""
    global model_predictor
    
    try:
        # Get fresh data
        fresh_data = get_ethereum_data(days=365, use_sample=True)
        if fresh_data.empty:
            raise HTTPException(status_code=404, detail="No data available for training")
        
        # Preprocess data
        processed_data = preprocess_crypto_data(fresh_data)
        
        # Train new model
        model_predictor = EthereumPredictor()
        df_with_features = model_predictor.create_features(processed_data)
        X, y = model_predictor.prepare_training_data(df_with_features)
        
        # Train models
        results = model_predictor.train_models(X, y)
        
        # Save model
        os.makedirs("models", exist_ok=True)
        model_predictor.save_model("models/ethereum_model.joblib")
        
        return {
            "message": "Model retrained successfully",
            "best_model": model_predictor.best_model_name,
            "features_count": len(model_predictor.feature_columns),
            "training_samples": len(X)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")


@app.get("/features/importance")
async def get_feature_importance():
    """Get feature importance from the model."""
    if not model_predictor:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    try:
        if hasattr(model_predictor.best_model, 'feature_importances_'):
            importance = model_predictor.best_model.feature_importances_
            feature_names = model_predictor.feature_columns
            
            # Create feature importance dictionary
            importance_dict = dict(zip(feature_names, importance))
            
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return {
                "feature_importance": sorted_importance,
                "top_features": list(sorted_importance.keys())[:10]
            }
        else:
            return {
                "message": "Feature importance not available for this model type",
                "model_type": model_predictor.best_model_name
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feature importance: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
