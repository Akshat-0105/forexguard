"""
ForexGuard Anomaly Detection API

FastAPI application for real-time anomaly detection in forex trading events.
Combines Isolation Forest and LSTM Autoencoder with explainability.
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
from pydantic import BaseModel, Field, field_validator
from fastapi.responses import JSONResponse

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
    
    user_id: str = Field(..., description="Unique user identifier")
    timestamp: str = Field(..., description="Event timestamp (ISO 8601 format)")
    event_type: str = Field("unknown", description="Type of event")
    ip_address: str = Field("0.0.0.0", description="Source IP address")
    device_id: str = Field("unknown", description="Device identifier")
    amount: float = Field(0.0, ge=0, description="Transaction amount")
    trade_volume: float = Field(0.0, ge=0, description="Trading volume")
    instrument: str = Field("unknown", description="Trading instrument")
    session_id: str = Field("unknown", description="Session identifier")
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        """Validate timestamp format."""
        try:
            pd.to_datetime(v)
        except:
            raise ValueError('Invalid timestamp format. Use ISO 8601 format.')
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class AnomalyReason(BaseModel):
    """Individual anomaly reason with severity."""
    
    text: str = Field(..., description="Human-readable reason text")
    severity: float = Field(..., ge=0.0, description="Severity score [0, 2.0+]")
    feature: str = Field(..., description="Feature name that triggered anomaly")


class AnomalyPredictionResponse(BaseModel):
    """Response schema for anomaly detection prediction."""
    
    prediction: int = Field(..., description="Binary prediction (0=normal, 1=anomalous)")
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Anomaly score [0, 1]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    reasons: List[AnomalyReason] = Field(default_factory=list, description="Top 3 reasons")
    num_reasons: int = Field(..., ge=0, le=3, description="Number of reasons provided")
    timestamp: str = Field(..., description="Server processing timestamp")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction": 1,
                    "anomaly_score": 0.92,
                    "confidence": 0.95,
                    "reasons": [
                        {
                            "text": "High action frequency",
                            "severity": 2.0,
                            "feature": "actions_per_minute"
                        }
                    ],
                    "num_reasons": 1,
                    "timestamp": "2024-03-30T14:30:05.123456"
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Check timestamp")
    version: str = Field(default="1.0.0", description="API version")


# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

app = FastAPI(
    title="ForexGuard Anomaly Detection API",
    description="Real-time anomaly detection for forex trading events",
    version="1.0.0"
)

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
        try:
            logger.info("Loading models...")
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
            
            self.explainer = AnomalyExplainer(verbose=False, use_fallback=True)
            self.ensemble_detector = EnsembleAnomalyDetector(
                if_weight=DEFAULT_IF_WEIGHT,
                lstm_weight=DEFAULT_LSTM_WEIGHT,
                verbose=False
            )
            self.explainable_detector = ExplainableAnomalyDetector(explainer=self.explainer)
            
            self.models_loaded = True
            logger.info("Models loaded successfully")
        except Exception as e:
            self.models_loaded = False
            self.error_message = str(e)
            logger.error(f"Failed to load models: {e}")
            raise
    
    def predict(self, features_df: pd.DataFrame) -> Dict:
        if not self.models_loaded:
            raise RuntimeError("Models not loaded")
        
        if self.if_detector is None and self.lstm_detector is None:
            anomaly_score, prediction, confidence = 0.5, 0, 0.5
        else:
            try:
                feature_cols = self.feature_engineer.get_feature_columns()
                if self.if_detector:
                    if_scores, _ = self.if_detector.predict(features_df[feature_cols])
                    if_score = if_scores[0]
                else:
                    if_score = 0.5
                
                if self.lstm_detector:
                    lstm_scores, _ = self.lstm_detector.predict(features_df[feature_cols])
                    lstm_score = lstm_scores[0]
                else:
                    lstm_score = 0.5
                
                ensemble_input = pd.DataFrame({'if_score': [if_score], 'lstm_score': [lstm_score]})
                scores, preds = self.ensemble_detector.detect(ensemble_input)
                anomaly_score, prediction = scores[0], preds[0]
                
                confidence = min(abs(anomaly_score - 0.5) / (0.25 + 1e-8), 1.0)
                confidence = 0.5 + confidence * 0.5
            except Exception as e:
                logger.error(f"Error during inference: {e}")
                anomaly_score, prediction, confidence = 0.5, 0, 0.5
        
        reasons = []
        if prediction == 1:
            try:
                if self.explainer.feature_stats is None:
                    self.explainer.compute_feature_stats(features_df)
                
                explanation = self.explainer.explain_event(
                    row=features_df.iloc[0],
                    anomaly_score=anomaly_score,
                    is_anomaly=True
                )
                
                for r in explanation.get('reasons', []):
                    reasons.append(AnomalyReason(text=r['reason'], severity=r['severity'], feature=r['feature']))
            except Exception as e:
                logger.error(f"Error generating explanations: {e}")
                reasons.append(AnomalyReason(text="Anomalous pattern detected", severity=1.5, feature="ensemble"))
        
        def safe_float(x):
            return float(x) if not (pd.isna(x) or np.isinf(x)) else 0.0

        return {
            'anomaly_score': safe_float(anomaly_score),
            'prediction': int(prediction),
            'confidence': safe_float(confidence),
            'reasons': reasons,
            'num_reasons': len(reasons)
        }


model_manager = ModelManager()


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    model_manager.load_models()

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(status="ok", timestamp=datetime.utcnow().isoformat())

@app.post("/predict", response_model=AnomalyPredictionResponse, tags=["Prediction"])
async def predict_anomaly(event: AnomalyEventInput):
    try:
        df = pd.DataFrame([event.model_dump()])
        try:
            engineered_df = model_manager.feature_engineer.engineer_features(df)
        except:
            engineered_df = df
        
        result = model_manager.predict(engineered_df)
        
        response_dict = {
            "prediction": int(result['prediction']),
            "anomaly_score": float(result['anomaly_score']),
            "confidence": float(result['confidence']),
            "reasons": [r.model_dump() for r in result['reasons']],
            "num_reasons": int(result['num_reasons']),
            "timestamp": datetime.utcnow().isoformat()
        }
        return JSONResponse(content=response_dict)
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch", tags=["Prediction"])
async def predict_batch(events: List[AnomalyEventInput]):
    results = []
    for i, event in enumerate(events):
        try:
            df = pd.DataFrame([event.model_dump()])
            try:
                engineered_df = model_manager.feature_engineer.engineer_features(df)
            except:
                engineered_df = df
            result = model_manager.predict(engineered_df)
            results.append(AnomalyPredictionResponse(
                prediction=result['prediction'],
                anomaly_score=result['anomaly_score'],
                confidence=result['confidence'],
                reasons=result['reasons'],
                num_reasons=result['num_reasons'],
                timestamp=datetime.utcnow().isoformat()
            ))
        except Exception as e:
            results.append({"error": str(e), "event_index": i})
    return results

@app.get("/", tags=["Info"])
async def root():
    return {"name": "ForexGuard API", "version": "1.0.0", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
