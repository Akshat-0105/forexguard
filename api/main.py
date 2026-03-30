"""
ForexGuard Anomaly Detection API

FastAPI application for real-time anomaly detection in forex trading events.
Combines Isolation Forest and LSTM Autoencoder with explainability.

Endpoints:
  GET  /health                 - Health check
  POST /predict                - Predict anomaly for single event
  POST /predict_batch          - Predict anomalies for multiple events
"""

import os
import sys
import pickle
import logging
from typing import List, Optional, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_engineering import FeatureEngineer
from models.isolation_forest import IsolationForestAnomalyDetector
from models.lstm_encoder import LSTMAutoencoderAnomalyDetector
from models.ensemble_detector import EnsembleAnomalyDetector
from models.explainable_detector import ExplainableAnomalyDetector
from models.anomaly_explainer import AnomalyExplainer

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'saved_models')
IF_MODEL_PATH = os.path.join(MODEL_DIR, 'isolation_forest_model.pkl')
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model.pt')
IF_SCALER_PATH = os.path.join(MODEL_DIR, 'if_scaler.pkl')
LSTM_SCALER_PATH = os.path.join(MODEL_DIR, 'lstm_scaler.pkl')

# Ensemble weights
DEFAULT_IF_WEIGHT = 0.6
DEFAULT_LSTM_WEIGHT = 0.4



# Input field names
INPUT_FIELDS = [
    'user_id',
    'timestamp',
    'event_type',
    'ip_address',
    'device_id',
    'amount',
    'trade_volume',
    'instrument',
    'session_id'
]


# ============================================================================
# PYDANTIC REQUEST/RESPONSE SCHEMAS
# ============================================================================

class AnomalyEventInput(BaseModel):
    """Request schema for single anomaly detection prediction."""
    
    user_id: str = Field(
        ...,
        description="Unique user identifier",
        example="user_12345"
    )
    timestamp: str = Field(
        ...,
        description="Event timestamp (ISO 8601 format)",
        example="2024-03-30T14:30:00"
    )
    event_type: str = Field(
        ...,
        description="Type of event (e.g., 'withdrawal', 'trade', 'deposit')",
        example="withdrawal"
    )
    ip_address: str = Field(
        ...,
        description="Source IP address",
        example="192.168.1.100"
    )
    device_id: str = Field(
        ...,
        description="Device identifier",
        example="device_abc123"
    )
    amount: float = Field(
        ...,
        ge=0,
        description="Transaction amount",
        example=5000.0
    )
    trade_volume: float = Field(
        ...,
        ge=0,
        description="Trading volume for this event",
        example=125000.0
    )
    instrument: str = Field(
        ...,
        description="Trading instrument (e.g., 'EUR/USD')",
        example="EUR/USD"
    )
    session_id: str = Field(
        ...,
        description="Session identifier",
        example="session_xyz789"
    )
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp format."""
        try:
            pd.to_datetime(v)
        except:
            raise ValueError('Invalid timestamp format. Use ISO 8601 format.')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_12345",
                "timestamp": "2024-03-30T14:30:00",
                "event_type": "withdrawal",
                "ip_address": "192.168.1.100",
                "device_id": "device_abc123",
                "amount": 5000.0,
                "trade_volume": 125000.0,
                "instrument": "EUR/USD",
                "session_id": "session_xyz789"
            }
        }


class AnomalyReason(BaseModel):
    """Individual anomaly reason with severity."""
    
    text: str = Field(
        ...,
        description="Human-readable reason text"
    )
    severity: float = Field(
        ...,
        ge=0.0,
        description="Severity score [0, 2.0+]"
    )
    feature: str = Field(
        ...,
        description="Feature name that triggered anomaly"
    )


class AnomalyPredictionResponse(BaseModel):
    """Response schema for anomaly detection prediction."""
    
    prediction: int = Field(
        ...,
        description="Binary prediction (0=normal, 1=anomalous)"
    )
    anomaly_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Anomaly score [0, 1]"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence"
    )
    reasons: List[str] = Field(
        default_factory=list,
        description="Top 3 reasons for anomaly (only for anomalies)"
    )
    num_reasons: int = Field(
        ...,
        ge=0,
        le=3,
        description="Number of reasons provided"
    )
    timestamp: str = Field(
        ...,
        description="Server processing timestamp"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "anomaly_score": 0.92,
                "confidence": 0.95,
                "reasons": [
                    {
                        "text": "High action frequency (24.2 actions/min)",
                        "severity": 2.0,
                        "feature": "actions_per_minute"
                    },
                    {
                        "text": "Sudden withdrawal spike (zscore=3.6)",
                        "severity": 2.0,
                        "feature": "withdrawal_zscore"
                    }
                ],
                "num_reasons": 2,
                "timestamp": "2024-03-30T14:30:05.123456"
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(
        ...,
        description="Service status"
    )
    timestamp: str = Field(
        ...,
        description="Check timestamp"
    )
    version: str = Field(
        default="1.0.0",
        description="API version"
    )


# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

app = FastAPI(
    title="ForexGuard Anomaly Detection API",
    description="Real-time anomaly detection for forex trading events",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# GLOBAL STATE - MODELS AND UTILITIES
# ============================================================================

class ModelManager:
    """Manages loading and inference with all models."""
    
    def __init__(self):
        """Initialize model manager."""
        self.feature_engineer = None
        self.if_detector = None
        self.lstm_detector = None
        self.ensemble_detector = None
        self.explainer = None
        self.explainable_detector = None
        self.models_loaded = False
        self.error_message = None
        self.if_weight = DEFAULT_IF_WEIGHT
        self.lstm_weight = DEFAULT_LSTM_WEIGHT
        
    def load_models(self):
        """Load all pre-trained models and initializers."""
        try:
            logger.info("Loading models...")
            
            # Initialize feature engineer
            self.feature_engineer = FeatureEngineer(verbose=False)
            
            # Load IF model
            try:
                if os.path.exists(IF_MODEL_PATH) and os.path.exists(IF_SCALER_PATH):
                    with open(IF_MODEL_PATH, 'rb') as f:
                        self.if_detector = pickle.load(f)
                    with open(IF_SCALER_PATH, 'rb') as f:
                        if_scaler = pickle.load(f)
                    self.if_detector.scaler = if_scaler
                else:
                    self.if_detector = None
            except Exception as e:
                logger.warning(f"Failed to load IF model: {e}")
                self.if_detector = None
            
            # Load LSTM model
            try:
                if os.path.exists(LSTM_MODEL_PATH):
                    self.lstm_detector = torch.load(LSTM_MODEL_PATH, map_location='cpu')
                else:
                    self.lstm_detector = None
            except Exception as e:
                logger.warning(f"Failed to load LSTM model: {e}")
                self.lstm_detector = None
            
            # Initialize explainer
            self.explainer = AnomalyExplainer(verbose=False, use_fallback=True)
            
            # Create ensemble detector
            self.ensemble_detector = EnsembleAnomalyDetector(
                if_weight=DEFAULT_IF_WEIGHT,
                lstm_weight=DEFAULT_LSTM_WEIGHT,
                verbose=False
            )
            
            # Create explainable detector wrapper
            self.explainable_detector = ExplainableAnomalyDetector(
                explainer=self.explainer
            )
            
            self.models_loaded = True
            logger.info("Models loaded successfully")
            
        except Exception as e:
            self.models_loaded = False
            self.error_message = str(e)
            logger.error(f"Failed to load models: {e}")
            raise
    
    def predict(self, features_df: pd.DataFrame) -> Dict:
        """Generate predictions for features DataFrame."""
        if not self.models_loaded:
            raise RuntimeError("Models not loaded")
        
        # Fallback if no models loaded
        if self.if_detector is None and self.lstm_detector is None:
            anomaly_score = 0.5
            prediction = 0
            confidence = 0.5
        
        else:
            try:
                # Get feature columns dynamically
                feature_cols = self.feature_engineer.get_feature_columns()
                
                # Get Isolation Forest score
                if self.if_detector is not None:
                    if_scores, _ = self.if_detector.predict(features_df[feature_cols])
                    if_score = if_scores[0]
                else:
                    if_score = 0.5
                
                # Get LSTM score
                if self.lstm_detector is not None:
                    lstm_scores, _ = self.lstm_detector.predict(features_df[feature_cols])
                    lstm_score = lstm_scores[0]
                else:
                    lstm_score = 0.5
                
                # Use ensemble detector
                ensemble_input = pd.DataFrame({
                    'if_score': [if_score],
                    'lstm_score': [lstm_score]
                })
                anomaly_score, prediction = self.ensemble_detector.detect(ensemble_input)
                anomaly_score = anomaly_score[0]
                prediction = prediction[0]
                
                # Confidence: distance from 0.5 threshold
                confidence = min(abs(anomaly_score - 0.5) / (0.25 + 1e-8), 1.0)
                confidence = 0.5 + confidence * 0.5
                
            except Exception as e:
                logger.error(f"Error during inference: {e}")
                anomaly_score = 0.5
                prediction = 0
                confidence = 0.5
        
        # Generate explanations
        reasons = []
        if prediction == 1:
            try:
                # Compute feature stats if not already done
                if self.explainer.feature_stats is None:
                    logger.info("Computing feature stats for explanations...")
                    # Use empirical stats from single sample for now
                    self.explainer.compute_feature_stats(features_df)
                
                # Get explanation for this event
                explanation = self.explainer.explain_event(
                    row=features_df.iloc[0],
                    anomaly_score=anomaly_score,
                    is_anomaly=True
                )
                
                # Convert explanation reasons to response format
                for reason_dict in explanation.get('reasons', []):
                    reasons.append(AnomalyReason(
                        text=reason_dict['reason'],
                        severity=reason_dict['severity'],
                        feature=reason_dict['feature']
                    ))
            
            except Exception as e:
                logger.error(f"Error generating explanations: {e}")
                # Fallback: generic reason
                reasons.append(AnomalyReason(
                    text="Anomalous pattern detected",
                    severity=1.5,
                    feature="ensemble"
                ))
        
        return {
            'anomaly_score': float(anomaly_score),
            'prediction': int(prediction),
            'confidence': float(confidence),
            'reasons': reasons,
            'num_reasons': len(reasons)
        }


# Initialize global model manager
model_manager = ModelManager()


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on application startup."""
    try:
        model_manager.load_models()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down ForexGuard API")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )


@app.post("/predict", response_model=AnomalyPredictionResponse, tags=["Prediction"])
async def predict_anomaly(event: AnomalyEventInput):
    """Predict anomaly for a single event."""
    try:
        logger.info(f"Processing request for user: {event.user_id}")
        
        # Convert input to DataFrame
        event_dict = event.dict()
        df = pd.DataFrame([event_dict])
        
        # Apply feature engineering
        try:
            engineered_df = model_manager.feature_engineer.engineer_features(df)
        except Exception as e:
            logger.warning(f"Feature engineering failed: {e}")
            engineered_df = df
        
        # Generate prediction
        result = model_manager.predict(engineered_df)
        
        response = AnomalyPredictionResponse(
            prediction=result['prediction'],
            anomaly_score=result['anomaly_score'],
            confidence=result['confidence'],
            reasons=result['reasons'],
            num_reasons=result['num_reasons'],
            timestamp=datetime.utcnow().isoformat()
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict_batch", tags=["Prediction"])
async def predict_batch(events: List[AnomalyEventInput]):
    """Predict anomalies for multiple events."""
    try:
        logger.info(f"Processing batch: {len(events)} events")
        
        results = []
        for i, event in enumerate(events):
            try:
                event_dict = event.dict()
                df = pd.DataFrame([event_dict])
                
                try:
                    engineered_df = model_manager.feature_engineer.engineer_features(df)
                except:
                    engineered_df = df
                
                result = model_manager.predict(engineered_df)
                
                response = AnomalyPredictionResponse(
                    prediction=result['prediction'],
                    anomaly_score=result['anomaly_score'],
                    confidence=result['confidence'],
                    reasons=result['reasons'],
                    num_reasons=result['num_reasons'],
                    timestamp=datetime.utcnow().isoformat()
                )
                
                results.append(response)
                
            except Exception as e:
                logger.error(f"Error on event {i}: {e}")
                results.append({
                    "error": str(e),
                    "event_index": i
                })
        
        logger.info(f"Batch complete: {len(results)} results")
        return results
    
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API root endpoint."""
    return {
        "name": "ForexGuard Anomaly Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "predict_batch": "POST /predict_batch",
            "docs": "GET /docs",
            "openapi": "GET /openapi.json"
        },
        "documentation": "Visit /docs for interactive API documentation"
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    logger.error(f"Validation error: {exc}")
    return {
        "error": "Validation Error",
        "detail": str(exc)
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting ForexGuard API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


# For deployment: gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.main:app
