"""
Ensemble Anomaly Detection - Combines Isolation Forest and LSTM Autoencoder

Combines scores from two complementary models for improved anomaly detection:
- Isolation Forest (traditional ML, fast)
- LSTM Autoencoder (deep learning, captures temporal patterns)

Ensemble strategy:
- Weighted average of normalized scores (default: 0.5 each)
- Threshold optimization via F1-score or percentile
- Superior performance through model combination
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    precision_recall_curve,
    f1_score as compute_f1
)
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EnsembleAnomalyDetector:
    """Ensemble detector combining Isolation Forest and LSTM anomaly scores."""
    
    def __init__(self, if_weight: float = 0.5, lstm_weight: float = 0.5, 
                 verbose: bool = True):
        """Initialize ensemble detector with specified weights."""
        total_weight = if_weight + lstm_weight
        if total_weight == 0:
            raise ValueError("Weights cannot both be zero")
        self.if_weight = if_weight / total_weight
        self.lstm_weight = lstm_weight / total_weight
        
        self.verbose = verbose
        self.threshold = None
        self.threshold_method = None
        self.evaluation_results = {}
    
    def _safe_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Safely normalize scores to [0, 1] range without fitting."""
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < 1e-10:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)
    
    def _normalize_scores(self, if_scores: np.ndarray, 
                         lstm_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize individual model scores to [0, 1] range."""
        if self.verbose:
            logger.info("Normalizing individual model scores...")
        
        if_scores_norm = self._safe_normalize(if_scores)
        lstm_scores_norm = self._safe_normalize(lstm_scores)
        
        if self.verbose:
            logger.info(f"IF scores normalized: [{if_scores_norm.min():.4f}, {if_scores_norm.max():.4f}]")
            logger.info(f"LSTM scores normalized: [{lstm_scores_norm.min():.4f}, {lstm_scores_norm.max():.4f}]")
        
        return if_scores_norm, lstm_scores_norm
    
    def _compute_ensemble_score(self, if_scores_norm: np.ndarray, 
                               lstm_scores_norm: np.ndarray) -> np.ndarray:
        """Compute weighted ensemble score."""
        if self.verbose:
            logger.info("Computing ensemble score...")
        
        ensemble_scores = (
            self.if_weight * if_scores_norm + 
            self.lstm_weight * lstm_scores_norm
        )
        
        if self.verbose:
            logger.info(f"Ensemble scores computed: [{ensemble_scores.min():.4f}, {ensemble_scores.max():.4f}]")
        
        return ensemble_scores
    
    def _set_threshold_percentile(self, ensemble_scores: np.ndarray, 
                                 percentile: float = 95) -> float:
        """Set threshold using percentile of scores."""
        self.threshold = np.percentile(ensemble_scores, percentile)
        self.threshold_method = 'percentile'
        
        if self.verbose:
            logger.info(f"Threshold set via {percentile}th percentile: {self.threshold:.4f}")
        
        return self.threshold
    
    def _set_threshold_f1_optimization(self, ensemble_scores: np.ndarray, 
                                      y_true: np.ndarray) -> Tuple[float, float]:
        """Optimize threshold to maximize F1-score."""
        if self.verbose:
            logger.info("Optimizing threshold for maximum F1-score...")
        
        thresholds = np.percentile(ensemble_scores, np.linspace(0, 100, 101))
        best_f1 = 0
        if best_threshold == 0:
            best_threshold = np.percentile(ensemble_scores, 95)
        
        for threshold in thresholds:
            y_pred = (ensemble_scores > threshold).astype(int)
            if y_pred.sum() == 0:
                continue
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.threshold = best_threshold
        self.threshold_method = 'f1_optimization'
        
        if self.verbose:
            logger.info(f"Optimal threshold: {self.threshold:.4f} (F1: {best_f1:.4f})")
        
        return self.threshold, best_f1
    
    def detect(self, df: pd.DataFrame, if_score_col: str = 'if_score',
               lstm_score_col: str = 'lstm_score', y_true_col: str = 'is_anomaly',
               threshold_method: str = 'percentile', percentile: float = 95,
               if_weight: Optional[float] = None,
               lstm_weight: Optional[float] = None) -> Dict:
        """Detect anomalies using ensemble of IF and LSTM scores."""
        # Input validation
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        required_cols = [if_score_col, lstm_score_col, y_true_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if self.verbose:
            logger.info("Detecting anomalies with ensemble method...")
        
        # Override weights if provided
        if if_weight is not None and lstm_weight is not None:
            total = if_weight + lstm_weight
            self.if_weight = if_weight / total
            self.lstm_weight = lstm_weight / total
        
        # Extract scores
        if_scores = df[if_score_col].values
        lstm_scores = df[lstm_score_col].values
        y_true = df[y_true_col].values
        
        # Normalize scores
        if_scores_norm, lstm_scores_norm = self._normalize_scores(if_scores, lstm_scores)
        
        # Compute ensemble score
        ensemble_scores = self._compute_ensemble_score(if_scores_norm, lstm_scores_norm)
        
        # Set threshold
        if threshold_method == 'f1_optimization':
            threshold, max_f1 = self._set_threshold_f1_optimization(ensemble_scores, y_true)
        else:
            threshold = self._set_threshold_percentile(ensemble_scores, percentile)
        
        # Generate predictions
        y_pred = (ensemble_scores > threshold).astype(int)
        
        n_predicted = y_pred.sum()
        anomaly_pct = (100 * n_predicted / len(y_pred)) if len(y_pred) > 0 else 0
        if self.verbose:
            logger.info(f"Predictions generated: {n_predicted} anomalies ({anomaly_pct:.1f}%)")
        
        # Evaluate
        evaluation = self._evaluate(y_true, y_pred, ensemble_scores)
        self.evaluation_results = evaluation
        
        return {
            'ensemble_scores': ensemble_scores,
            'predictions': y_pred,
            'threshold': self.threshold,
            'threshold_method': self.threshold_method,
            'weights': {'if': self.if_weight, 'lstm': self.lstm_weight},
            'evaluation': evaluation,
            'if_scores_norm': if_scores_norm,
            'lstm_scores_norm': lstm_scores_norm
        }
    
    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                  ensemble_scores: np.ndarray) -> Dict:
        """Evaluate ensemble predictions against ground truth."""
        # Classification metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # ROC-AUC
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, ensemble_scores)
        else:
            roc_auc = None
        
        # Accuracy
        accuracy = (y_pred == y_true).mean()
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        if self.verbose:
            logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
            if roc_auc is not None:
                logger.info(f"ROC-AUC: {roc_auc:.4f}, Specificity: {specificity:.4f}")
            logger.info(f"Confusion Matrix - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'specificity': float(specificity),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        }
    
    def get_results_dataframe(self, df: pd.DataFrame, ensemble_scores: np.ndarray,
                             predictions: np.ndarray) -> pd.DataFrame:
        """Create output DataFrame with ensemble scores and predictions."""
        result_df = df.copy()
        result_df['ensemble_score'] = ensemble_scores
        result_df['predicted_label'] = predictions
        
        return result_df
    
    def compare_with_individuals(self, if_preds: np.ndarray, lstm_preds: np.ndarray,
                                ensemble_preds: np.ndarray, y_true: np.ndarray) -> Dict:
        """Compare ensemble predictions with individual model predictions."""
        metrics = {}
        
        for name, preds in [
            ('Isolation Forest', if_preds),
            ('LSTM', lstm_preds),
            ('Ensemble', ensemble_preds)
        ]:
            precision = precision_score(y_true, preds, zero_division=0)
            recall = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            
            metrics[name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            logger.info(f"{name}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
        
        ensemble_f1 = metrics['Ensemble']['f1_score']
        if_f1 = metrics['Isolation Forest']['f1_score']
        lstm_f1 = metrics['LSTM']['f1_score']
        
        if ensemble_f1 > max(if_f1, lstm_f1):
            improvement = ((ensemble_f1 - max(if_f1, lstm_f1)) / max(if_f1, lstm_f1) * 100)
            logger.info(f"Ensemble outperforms by {improvement:.1f}%")
        
        return metrics


def create_ensemble_from_models(if_results: Dict, lstm_results: Dict, 
                               data_df: pd.DataFrame,
                               if_weight: float = 0.5,
                               lstm_weight: float = 0.5) -> Dict:
    """Create ensemble from trained model results."""
    # Extract scores with validation
    if_scores = if_results.get('test_scores')
    if if_scores is None:
        if_scores = if_results.get('evaluation', {}).get('scores')
    if if_scores is None:
        raise ValueError("Cannot find IF scores in if_results")
    
    lstm_scores = lstm_results.get('test_errors_event')
    if lstm_scores is None:
        lstm_scores = lstm_results.get('evaluation', {}).get('scores')
    if lstm_scores is None:
        raise ValueError("Cannot find LSTM scores in lstm_results")
    
    # Prepare ensemble input
    ensemble_df = data_df.copy()
    ensemble_df['if_score'] = if_scores
    ensemble_df['lstm_score'] = lstm_scores
    
    # Run ensemble detection
    detector = EnsembleAnomalyDetector(
        if_weight=if_weight,
        lstm_weight=lstm_weight,
        verbose=True
    )
    
    results = detector.detect(
        ensemble_df,
        threshold_method='f1_optimization'
    )
    
    return results
