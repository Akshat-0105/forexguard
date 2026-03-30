"""
Isolation Forest Anomaly Detection Model

Training pipeline for detecting anomalies in forex trading events using Isolation Forest.
- Trains on normal data only (is_anomaly == 0)
- Tests on full dataset
- Uses engineered features with StandardScaler normalization
- Generates anomaly scores and predictions
- Evaluates using ground truth labels
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from typing import Tuple, Dict, Optional
import logging
import joblib
import os

logger = logging.getLogger(__name__)


class IsolationForestAnomalyDetector:
    """Isolation Forest-based anomaly detection model for forex trading events."""
    
    IF_N_ESTIMATORS = 100
    IF_CONTAMINATION = 0.05
    IF_RANDOM_STATE = 42
    
    def __init__(self, verbose: bool = True):
        """Initialize the anomaly detector."""
        self.verbose = verbose
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.threshold = None
        self.training_stats = {}
        self.evaluation_results = {}
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range with robust handling."""
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < 1e-10:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)
    
    def train(self, df: pd.DataFrame, test_size: Optional[float] = None) -> Dict:
        """Train Isolation Forest on normal data and evaluate on full dataset."""
        from features.feature_engineering import FeatureEngineer
        
        logger.info("Starting Isolation Forest training...")
        
        # Get feature columns from FeatureEngineer
        fe = FeatureEngineer()
        self.feature_columns = fe.get_feature_columns()
        
        df = df.copy()
        
        missing_features = [f for f in self.feature_columns if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        
        if 'is_anomaly' not in df.columns:
            raise ValueError("Ground truth 'is_anomaly' column not found")
        
        X = df[self.feature_columns].copy()
        y = df['is_anomaly'].copy()
        
        n_samples = len(X)
        n_features = len(self.feature_columns)
        n_anomalies = y.sum()
        n_normal = len(y) - n_anomalies
        
        logger.info(f"Data: {n_samples:,} samples, {n_features} features")
        logger.info(f"Class distribution: {n_normal:,} normal, {n_anomalies:,} anomalous")
        
        # Train/Test Split (train on normal only)
        X_train = X[y == 0].copy()
        
        # Testing: Full dataset (including anomalies)
        X_test = X.copy()
        y_test = y.copy()
        
        logger.info(f"Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
        
        if len(X_train) == 0:
            raise ValueError("No normal samples found for training")
        
        # Feature Scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        scaling_stats = {
            'mean': self.scaler.mean_.tolist(),
            'std': self.scaler.scale_.tolist(),
            'feature_names': self.feature_columns
        }
        logger.info(f"Scaler fitted ({len(self.feature_columns)} features)")
        
        # Train Isolation Forest
        logger.info(f"Training model (n_estimators={self.IF_N_ESTIMATORS})")
        
        self.model = IsolationForest(
            n_estimators=self.IF_N_ESTIMATORS,
            contamination=self.IF_CONTAMINATION,
            random_state=self.IF_RANDOM_STATE,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled)
        logger.info(f"Model trained on {len(X_train_scaled):,} samples")
        
        # Generate anomaly scores
        # Get raw decision function scores (lower = more anomalous by default in IF)
        raw_scores = self.model.decision_function(X_test_scaled)
        
        # Convert to anomaly scores: negation + scaling so higher = more anomalous
        anomaly_scores = -raw_scores
        anomaly_scores_normalized = self._normalize(anomaly_scores)
        
        # Check for constant scores (degenerate case)
        if np.all(anomaly_scores_normalized == 0):
            logger.warning("All anomaly scores are zero")
        
        # Compute threshold: percentile based on contamination using normalized scores
        self.threshold = np.percentile(anomaly_scores_normalized, 100 * (1 - self.IF_CONTAMINATION))
        self.threshold = float(np.clip(self.threshold, 0.0, 1.0))
        
        if self.threshold <= 0:
            self.threshold = 0.01
        
        logger.info(f"Anomaly scores: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}], threshold: {self.threshold:.4f}")
        
        # Generate predictions
        y_pred = (anomaly_scores_normalized > self.threshold).astype(int)
        
        n_predicted_anomalies = y_pred.sum()
        logger.info(f"Predictions: {n_predicted_anomalies:,} anomalies ({100*n_predicted_anomalies/len(y_pred):.1f}%)")
        
        # Evaluate against ground truth
        logger.info("Evaluating...")
        
        evaluation = self._evaluate(y_test, y_pred, anomaly_scores_normalized)
        
        # Store results
        self.evaluation_results = evaluation
        self.training_stats = {
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': n_features,
            'scaling_stats': scaling_stats,
            'threshold': float(self.threshold),
            'anomaly_scores_stats': {
                'min': float(anomaly_scores.min()),
                'max': float(anomaly_scores.max()),
                'mean': float(anomaly_scores.mean()),
                'std': float(anomaly_scores.std())
            }
        }
        
        logger.info("Training complete.")
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'evaluation': evaluation,
            'training_stats': self.training_stats
        }
    
    def predict(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions on new data."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model must be trained first")
        
        if len(X_test) == 0:
            raise ValueError("Empty input dataframe")
        
        if self.feature_columns is None:
            raise RuntimeError("Feature columns not set")
        
        # Preserve original index
        orig_index = X_test.index
        
        # Auto-engineer features if needed
        if not all(f in X_test.columns for f in self.feature_columns):
            from features.feature_engineering import FeatureEngineer
            fe = FeatureEngineer()
            X_test = fe.transform(X_test)
        
        # Restore index after engineering
        if len(X_test) == len(orig_index):
            X_test.index = orig_index
        
        # Order features consistently before scaling
        X_ordered = X_test[self.feature_columns].copy()
        X_scaled = self.scaler.transform(X_ordered)
        
        raw_scores = self.model.decision_function(X_scaled)
        anomaly_scores = -raw_scores
        anomaly_scores_normalized = self._normalize(anomaly_scores)
        
        # Predictions use normalized scores with normalized threshold (consistent with training)
        predictions = (anomaly_scores_normalized > self.threshold).astype(int)
        
        return anomaly_scores_normalized, predictions
    
    def _evaluate(self, y_true: pd.Series, y_pred: np.ndarray, 
                  anomaly_scores: np.ndarray) -> Dict:
        """Evaluate predictions against ground truth and return metrics."""
        # Classification metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # ROC-AUC (if we have both classes)
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, anomaly_scores)
        else:
            roc_auc = None
        
        # Accuracy
        accuracy = (y_pred == y_true).mean()
        
        # Log metrics
        logger.info(f"\nClassification Metrics:")
        logger.info(f"  Precision:  {precision:.4f}")
        logger.info(f"  Recall:     {recall:.4f}")
        logger.info(f"  F1-Score:   {f1:.4f}")
        logger.info(f"  Accuracy:   {accuracy:.4f}")
        if roc_auc is not None:
            logger.info(f"  ROC-AUC:    {roc_auc:.4f}")
        
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  True Negatives:  {tn:,} (correct normal)")
        logger.info(f"  False Positives: {fp:,} (false alarms)")
        logger.info(f"  False Negatives: {fn:,} (missed anomalies)")
        logger.info(f"  True Positives:  {tp:,} (detected anomalies)")
        
        # Detection rate
        if (tp + fn) > 0:
            detection_rate = tp / (tp + fn)
            logger.info(f"\nDetection Rate:  {detection_rate:.2%} ({tp}/{tp+fn})")
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            },
            'classification_report': classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
        }
    
    def get_results_dataframe(self, df: pd.DataFrame, 
                             anomaly_scores: np.ndarray,
                             predictions: np.ndarray) -> pd.DataFrame:
        """Add anomaly scores and predictions to DataFrame."""
        result_df = df.copy()
        result_df['anomaly_score'] = anomaly_scores
        result_df['predicted_label'] = predictions
        
        return result_df
    
    def save_model(self, model_path: str = 'models/if_model.pkl',
                   scaler_path: str = 'models/if_scaler.pkl') -> None:
        """Save model, scaler, and metadata to disk."""
        if self.model is None:
            raise RuntimeError("No trained model to save")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump({
            'threshold': self.threshold,
            'feature_columns': self.feature_columns
        }, model_path.replace('.pkl', '_metadata.pkl'))
        
        logger.info(f"✓ Model saved to {model_path}")
        logger.info(f"✓ Scaler saved to {scaler_path}")
        logger.info(f"✓ Metadata saved")
    
    def load_model(self, model_path: str = 'models/if_model.pkl',
                   scaler_path: str = 'models/if_scaler.pkl') -> None:
        """Load model, scaler, and metadata from disk."""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        metadata_path = model_path.replace('.pkl', '_metadata.pkl')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.threshold = metadata.get('threshold')
            self.feature_columns = metadata.get('feature_columns')
        
        logger.info(f"✓ Model loaded from {model_path}")
        logger.info(f"✓ Scaler loaded from {scaler_path}")
        if self.threshold is not None:
            logger.info(f"✓ Metadata loaded (threshold: {self.threshold:.4f})")


def train_isolation_forest(df: pd.DataFrame, verbose: bool = True) -> Tuple[Dict, pd.DataFrame]:
    """Train Isolation Forest and return results."""
    detector = IsolationForestAnomalyDetector(verbose=verbose)
    results = detector.train(df)
    
    # Use predict() to generate scores and predictions
    anomaly_scores_norm, predictions = detector.predict(df)
    
    if predictions.sum() == 0:
        logger.warning("Model predicted no anomalies")
    
    results_df = detector.get_results_dataframe(df, anomaly_scores_norm, predictions)
    
    return results, results_df
