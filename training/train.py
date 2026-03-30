"""
Training Pipeline Configuration and Runner

Allows easy training of Isolation Forest with custom hyperparameters.
Supports multiple model configurations for hyperparameter tuning.
"""

import argparse
import json
import pandas as pd
from models.isolation_forest import IsolationForestAnomalyDetector
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)


# Default configurations for Isolation Forest
DEFAULT_CONFIGS = {
    'baseline': {
        'n_estimators': 100,
        'contamination': 0.05,
        'random_state': 42,
        'description': 'Baseline configuration (5% contamination)'
    },
    'conservative': {
        'n_estimators': 100,
        'contamination': 0.02,
        'random_state': 42,
        'description': 'Conservative model (2% contamination, fewer false alarms)'
    },
    'aggressive': {
        'n_estimators': 100,
        'contamination': 0.10,
        'random_state': 42,
        'description': 'Aggressive model (10% contamination, higher recall)'
    },
    'deep_forest': {
        'n_estimators': 200,
        'contamination': 0.05,
        'random_state': 42,
        'description': 'Deeper forest (200 estimators, higher complexity)'
    }
}


def train_with_config(config_name: str = 'baseline', 
                      data_path: str = 'data/engineered_features.csv',
                      save_results: bool = True) -> dict:
    """Train Isolation Forest with specified configuration."""
    if config_name not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Choose from: {list(DEFAULT_CONFIGS.keys())}")
    
    config = DEFAULT_CONFIGS[config_name]
    logger.info(f"Training {config_name}: {config['description']}")
    
    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset: {len(df):,} samples, {df.shape[1]} columns")
    
    # Create custom detector with config parameters
    detector = IsolationForestAnomalyDetector(verbose=True)
    
    # Override default parameters
    detector.IF_N_ESTIMATORS = config['n_estimators']
    detector.IF_CONTAMINATION = config['contamination']
    detector.IF_RANDOM_STATE = config['random_state']
    
    # Train
    results = detector.train(df)
    
    # Generate predictions
    anomaly_scores_norm, predictions = detector.predict(df)
    
    # Create results dataframe
    results_df = detector.get_results_dataframe(df, anomaly_scores_norm, predictions)
    
    # Save if requested
    if save_results:
        # Save results CSV
        results_csv = f'data/if_results_{config_name}.csv'
        results_df.to_csv(results_csv, index=False)
        logger.info(f"Results saved: {results_csv}")
        
        # Save model config
        config_json = f'models/if_config_{config_name}.json'
        os.makedirs(os.path.dirname(config_json), exist_ok=True)
        
        config_data = {
            'config_name': config_name,
            'parameters': config,
            'training_stats': results['training_stats'],
            'evaluation': results['evaluation'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(config_json, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Config saved: {config_json}")
        
        # Save model
        model_path = f'models/if_model_{config_name}.pkl'
        scaler_path = f'models/if_scaler_{config_name}.pkl'
        detector.save_model(model_path, scaler_path)
    
    return {
        'config': config,
        'results': results,
        'results_df': results_df,
        'detector': detector
    }


def compare_configurations():
    """Train multiple configurations and compare performance."""
    logger.info("Comparing all configurations")
    
    comparison_results = {}
    
    # Train each config
    for config_name in ['baseline', 'conservative', 'aggressive', 'deep_forest']:
        logger.info(f"Training {config_name.upper()}...")
        
        try:
            result = train_with_config(config_name, save_results=True)
            metrics = result['results']['evaluation']
            
            comparison_results[config_name] = {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1_score'],
                'accuracy': metrics['accuracy'],
                'roc_auc': metrics['roc_auc'],
                'tp': metrics['confusion_matrix']['tp'],
                'fp': metrics['confusion_matrix']['fp'],
                'fn': metrics['confusion_matrix']['fn']
            }
        except Exception as e:
            logger.error(f"Error training {config_name}: {e}")
    
    # Summary comparison
    comparison_df = pd.DataFrame(comparison_results).T
    
    logger.info(f"Metrics: {comparison_df[['precision', 'recall', 'f1', 'accuracy']].to_string()}")
    logger.info(f"Confusion Matrix: {comparison_df[['tp', 'fp', 'fn']].to_string()}")
    
    # Recommendations
    best_f1 = comparison_df['f1'].idxmax()
    logger.info(f"Best F1-Score: {best_f1} (F1={comparison_df.loc[best_f1, 'f1']:.4f})")
    
    return comparison_results


def main():
    """Command-line interface for model training."""
    parser = argparse.ArgumentParser(
        description='Train Isolation Forest anomaly detection model'
    )
    
    parser.add_argument(
        '--config',
        choices=['baseline', 'conservative', 'aggressive', 'deep_forest'],
        default='baseline',
        help='Model configuration to train (default: baseline)'
    )
    
    parser.add_argument(
        '--data',
        default='data/engineered_features.csv',
        help='Path to engineered features CSV (default: data/engineered_features.csv)'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all configurations'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save model and results'
    )
    
    args = parser.parse_args()
    
    # Run training
    if args.compare:
        compare_configurations()
    else:
        train_with_config(
            config_name=args.config,
            data_path=args.data,
            save_results=not args.no_save
        )


if __name__ == '__main__':
    main()
