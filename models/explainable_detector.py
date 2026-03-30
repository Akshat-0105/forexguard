"""
Explainable Anomaly Detection Interface

Combines LSTM, Isolation Forest, and explainability layer into a unified interface
for interpretable anomaly detection with human-readable reasons.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ExplainableAnomalyDetector:
    """Unified interface for anomaly detection with explanations."""
    
    def __init__(self, explainer=None, percentile: float = 95):
        """Initialize detector with optional explainer."""
        self.explainer = explainer
        self.models = {}
        self.ensemble_weights = {}
        self.percentile = percentile
        
    def add_model(self, name: str, model, weight: float = 1.0):
        """Add a model to the detector."""
        self.models[name] = model
        self.ensemble_weights[name] = weight
        logger.info(f"Added model: {name} (weight={weight})")
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range using min-max scaling."""
        min_val = scores.min()
        max_val = scores.max()
        
        if max_val - min_val < 1e-10:
            return np.zeros_like(scores)
        
        return (scores - min_val) / (max_val - min_val)
    
    def predict_single_model(self, df: pd.DataFrame, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from a single model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        model = self.models[model_name]
        scores, predictions = model.predict(df)
        return scores, predictions
    
    def predict_ensemble(self, df: pd.DataFrame, method: str = 'average') -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from ensemble of models."""
        if not self.models:
            raise ValueError("No models registered for ensemble")
        
        all_scores = []
        all_weights = []
        
        for model_name in self.models:
            scores, _ = self.predict_single_model(df, model_name)
            normalized_scores = self._normalize_scores(scores)
            weight = self.ensemble_weights.get(model_name, 1.0)
            
            all_scores.append(normalized_scores)
            all_weights.append(weight)
        
        all_scores = np.array(all_scores)
        all_weights = np.array(all_weights)
        
        if method == 'average':
            weighted_scores = all_scores * all_weights[:, np.newaxis]
            ensemble_scores = weighted_scores.sum(axis=0) / all_weights.sum()
        elif method == 'voting':
            all_preds = (all_scores > np.median(all_scores, axis=1, keepdims=True)).astype(float)
            ensemble_scores = (all_preds * all_weights[:, np.newaxis]).sum(axis=0) / all_weights.sum()
        elif method == 'max':
            ensemble_scores = all_scores.max(axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        threshold = np.percentile(ensemble_scores, self.percentile)
        ensemble_predictions = (ensemble_scores > threshold).astype(int)
        
        return ensemble_scores, ensemble_predictions
    
    def detect(self, df: pd.DataFrame, model_name: Optional[str] = None,
              ensemble_method: str = 'average') -> pd.DataFrame:
        """Detect anomalies with explanations using adaptive thresholds."""
        # Input validation
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        logger.info("Running anomaly detection...")
        
        # Ensure explainer has computed adaptive thresholds
        if self.explainer:
            if self.explainer.thresholds is None:
                logger.info("Computing adaptive thresholds from feature distribution...")
                self.explainer.compute_feature_stats(df)
        
        # Get predictions
        if model_name:
            logger.info(f"Using model: {model_name}")
            scores, predictions = self.predict_single_model(df, model_name)
        else:
            if len(self.models) == 0:
                raise ValueError("No models registered")
            elif len(self.models) == 1:
                model_name = list(self.models.keys())[0]
                logger.info(f"Using single model: {model_name}")
                scores, predictions = self.predict_single_model(df, model_name)
            else:
                logger.info(f"Using ensemble ({ensemble_method}) with {len(self.models)} models")
                scores, predictions = self.predict_ensemble(df, method=ensemble_method)
        
        # Generate explanations if explainer available
        if self.explainer:
            results = self.explainer.create_explainable_results(df, scores, predictions)
        else:
            logger.warning("No explainer configured, returning scores and predictions only")
            results = pd.DataFrame({
                'anomaly_score': scores,
                'predicted_label': predictions,
                'reasons': ['No explainer configured'] * len(scores)
            })
        
        logger.info(f"Detection complete: {predictions.sum()} anomalies detected")
        
        return results
    
    def explain_event(self, row: pd.Series, anomaly_score: float,
                     is_anomaly: bool = True) -> Dict:
        """Get detailed explanation for a single event."""
        if not self.explainer:
            return {
                'anomaly_score': float(anomaly_score),
                'predicted_label': 1 if is_anomaly else 0,
                'reasons': []
            }
        
        return self.explainer.explain_event(row, anomaly_score, is_anomaly)
    
    def get_summary(self) -> Dict:
        """Get summary of detector configuration."""
        return {
            'models': list(self.models.keys()),
            'weights': self.ensemble_weights,
            'explainer_available': self.explainer is not None,
            'num_models': len(self.models),
            'percentile': self.percentile
        }
