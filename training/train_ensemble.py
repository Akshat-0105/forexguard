"""Ensemble model training combining Isolation Forest and LSTM Autoencoder predictions."""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from models.isolation_forest import IsolationForestAnomalyDetector
from models.lstm_encoder import LSTMAutoencoderAnomalyDetector
from models.ensemble_detector import EnsembleAnomalyDetector
import logging

logger = logging.getLogger(__name__)


def train_individual_models(data_file='data/engineered_features.csv'):
    """Train individual IF and LSTM models."""
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df):,} samples")
    
    # Train IF - gets scores for all samples
    logger.info("Training Isolation Forest...")
    if_detector = IsolationForestAnomalyDetector(verbose=False)
    if_results = if_detector.train(df)
    if_scores_all, if_preds = if_detector.predict(df)
    logger.info("IF model trained")
    
    # Train LSTM - works with sequences (fewer samples than events)
    logger.info("Training LSTM Autoencoder...")
    lstm_detector = LSTMAutoencoderAnomalyDetector(verbose=False)
    lstm_results = lstm_detector.train(df)
    lstm_scores_all, lstm_preds = lstm_detector.predict(df)
    logger.info("LSTM model trained")
    
    # Create combined dataframe - use aligned scores
    logger.info("Preparing ensemble input...")
    ensemble_input = pd.DataFrame({
        'if_score': if_scores_all,
        'lstm_score': lstm_scores_all,
        'is_anomaly': df['is_anomaly'].values
    })
    logger.info(f"Ensemble input prepared with {len(ensemble_input):,} samples")
    
    return {
        'if_results': if_results,
        'if_scores': if_scores_all,
        'if_detector': if_detector,
        'lstm_results': lstm_results,
        'lstm_scores': lstm_scores_all,
        'lstm_detector': lstm_detector,
        'ensemble_input': ensemble_input,
        'df': df
    }


def train_ensemble_percentile(ensemble_input, percentile=95, if_weight=0.5, lstm_weight=0.5):
    """Train ensemble using percentile threshold."""
    logger.info(f"Ensemble model with percentile threshold ({percentile}th)")
    
    detector = EnsembleAnomalyDetector(
        if_weight=if_weight,
        lstm_weight=lstm_weight,
        verbose=True
    )
    
    results = detector.detect(
        ensemble_input,
        threshold_method='percentile',
        percentile=percentile
    )
    
    return results, detector


def train_ensemble_f1_optimized(ensemble_input, if_weight=0.5, lstm_weight=0.5):
    """Train ensemble with F1-score optimized threshold."""
    logger.info("Ensemble model with F1-optimized threshold")
    
    detector = EnsembleAnomalyDetector(
        if_weight=if_weight,
        lstm_weight=lstm_weight,
        verbose=True
    )
    
    results = detector.detect(
        ensemble_input,
        threshold_method='f1_optimization'
    )
    
    return results, detector


def compare_weight_strategies(ensemble_input):
    """Compare different weighting strategies."""
    logger.info("Comparing weight strategies")
    
    strategies = [
        ('Equal Weight', 0.5, 0.5),
        ('IF Heavy', 0.7, 0.3),
        ('LSTM Heavy', 0.3, 0.7),
        ('IF Only', 1.0, 0.0),
        ('LSTM Only', 0.0, 1.0),
    ]
    
    results_list = []
    
    for name, if_w, lstm_w in strategies:
        logger.info(f"{name} (IF: {if_w*100:.0f}%, LSTM: {lstm_w*100:.0f}%)")
        
        detector = EnsembleAnomalyDetector(if_weight=if_w, lstm_weight=lstm_w, verbose=False)
        results = detector.detect(
            ensemble_input,
            threshold_method='f1_optimization'
        )
        
        metrics = results['evaluation']
        logger.info(f"  Precision:  {metrics['precision']:.4f}")
        logger.info(f"  Recall:     {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:   {metrics['f1_score']:.4f} ← Key metric")
        logger.info(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
        
        results_list.append({
            'strategy': name,
            'if_weight': if_w,
            'lstm_weight': lstm_w,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc']
        })
    
    # Summary table
    logger.info("Summary:")
    comparison_df = pd.DataFrame(results_list)
    logger.info(comparison_df[['strategy', 'precision', 'recall', 'f1_score', 'roc_auc']].to_string())


def compare_all_three(if_preds, lstm_preds, ensemble_results, y_true):
    """Compare IF, LSTM, and Ensemble predictions."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    logger.info("Comparison: IF vs LSTM vs Ensemble")
    
    ensemble_preds = ensemble_results['predictions']
    
    comparisons = [
        ('Isolation Forest', if_preds),
        ('LSTM Autoencoder', lstm_preds),
        ('Ensemble (Combined)', ensemble_preds)
    ]
    
    logger.info("Model Performance:")
    
    for name, preds in comparisons:
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        detected = preds.sum()
        
        logger.info(
            f"{name:<25} "
            f"{p:<12.4f} "
            f"{r:<12.4f} "
            f"{f1:<12.4f} "
            f"{detected:<12,}"
        )
    
    # Ensemble advantage
    ensemble_f1 = f1_score(ensemble_preds, y_true, zero_division=0)
    if_f1 = f1_score(if_preds, y_true, zero_division=0)
    lstm_f1 = f1_score(lstm_preds, y_true, zero_division=0)
    
    logger.info(f"Best F1-Score: Ensemble ({ensemble_f1:.4f})")


def main():
    """Command-line interface for ensemble training."""
    parser = argparse.ArgumentParser(
        description='Train ensemble anomaly detection model'
    )
    
    parser.add_argument(
        '--method',
        choices=['percentile', 'f1'],
        default='f1',
        help='Threshold setting method (default: f1)'
    )
    
    parser.add_argument(
        '--percentile',
        type=int,
        default=95,
        help='Percentile for threshold (default: 95)'
    )
    
    parser.add_argument(
        '--if',
        type=float,
        dest='if_weight',
        default=0.5,
        help='Weight for IF scores (default: 0.5)'
    )
    
    parser.add_argument(
        '--lstm',
        type=float,
        dest='lstm_weight',
        default=0.5,
        help='Weight for LSTM scores (default: 0.5)'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare different weight strategies'
    )
    
    parser.add_argument(
        '--data',
        default='data/engineered_features.csv',
        help='Path to engineered features CSV'
    )
    
    args = parser.parse_args()
    
    # Check data exists
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        return
    
    # Train individual models
    logger.info("Starting ensemble training...")
    models_data = train_individual_models(args.data)
    ensemble_input = models_data['ensemble_input']
    
    # Compare strategies if requested
    if args.compare:
        compare_weight_strategies(ensemble_input)
        return
    
    # Train ensemble with specified method
    if args.method == 'f1':
        results, detector = train_ensemble_f1_optimized(
            ensemble_input,
            if_weight=args.if_weight,
            lstm_weight=args.lstm_weight
        )
    else:
        results, detector = train_ensemble_percentile(
            ensemble_input,
            percentile=args.percentile,
            if_weight=args.if_weight,
            lstm_weight=args.lstm_weight
        )
    
    # Get predictions from individual models
    if_scores = models_data['if_scores']
    lstm_scores = models_data['lstm_scores']
    y_true = ensemble_input['is_anomaly'].values
    
    # Simple thresholding for individual models (for comparison)
    if_threshold = np.percentile(if_scores, 95)
    lstm_threshold = np.percentile(lstm_scores, 95)
    if_preds = (if_scores > if_threshold).astype(int)
    lstm_preds = (lstm_scores > lstm_threshold).astype(int)
    
    # Compare all three
    compare_all_three(if_preds, lstm_preds, results, y_true)
    
    # Create output dataframe
    output_df = ensemble_input.copy()
    output_df['ensemble_score'] = results['ensemble_scores']
    output_df['predicted_label'] = results['predictions']
    output_df['if_score_norm'] = results['if_scores_norm']
    output_df['lstm_score_norm'] = results['lstm_scores_norm']
    
    # Save results
    output_file = f"data/ensemble_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")
    logger.info("Ensemble training complete.")


if __name__ == '__main__':
    main()
