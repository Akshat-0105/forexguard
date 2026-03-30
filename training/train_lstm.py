"""LSTM autoencoder training CLI."""

import os
import json
import argparse
import pandas as pd
from datetime import datetime
from models.lstm_encoder import LSTMAutoencoderAnomalyDetector
import logging

logger = logging.getLogger(__name__)


# Pre-configured LSTM hyperparameter sets
LSTM_CONFIGS = {
    'default': {
        'sequence_length': 10,
        'hidden_size': 32,
        'batch_size': 64,
        'epochs': 15,
        'learning_rate': 0.001,
        'description': 'Default configuration (balanced)'
    },
    'fast': {
        'sequence_length': 5,
        'hidden_size': 16,
        'batch_size': 128,
        'epochs': 10,
        'learning_rate': 0.001,
        'description': 'Fast training (shorter sequences, smaller hidden size)'
    },
    'deep': {
        'sequence_length': 15,
        'hidden_size': 64,
        'batch_size': 32,
        'epochs': 20,
        'learning_rate': 0.0005,
        'description': 'Deep learning (longer sequences, larger bottleneck)'
    },
    'aggressive': {
        'sequence_length': 10,
        'hidden_size': 48,
        'batch_size': 32,
        'epochs': 25,
        'learning_rate': 0.0001,
        'description': 'Aggressive training (small learning rate, more epochs)'
    }
}


def train_single_config(config_name, config_params, data_file='data/engineered_features.csv'):
    """Train a single LSTM configuration."""
    if os.path.getsize(data_file) == 0:
        raise ValueError("Input data file is empty")
    
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return None
    
    logger.info(f"Training {config_name}...")
    logger.info(f"Sequence: {config_params['sequence_length']}, Hidden: {config_params['hidden_size']}, "
                f"Batch: {config_params['batch_size']}, Epochs: {config_params['epochs']}")
    
    # Load data
    df = pd.read_csv(data_file)
    logger.info(f"Loading {len(df):,} events from {data_file}...")
    
    # Train
    try:
        logger.info("Initializing LSTM autoencoder...")
        detector = LSTMAutoencoderAnomalyDetector(
            sequence_length=config_params['sequence_length'],
            hidden_size=config_params['hidden_size'],
            batch_size=config_params['batch_size'],
            epochs=config_params['epochs'],
            learning_rate=config_params['learning_rate'],
            verbose=True
        )
        
        logger.info("Training model...")
        results = detector.train(df)
        
        # Extract metrics
        evaluation = results['evaluation']
        
        logger.info(f"Completed {config_name}")
        cm = evaluation['confusion_matrix']
        logger.info(f"TP: {cm['tp']}, TN: {cm['tn']}, FP: {cm['fp']}, FN: {cm['fn']}")
        logger.info(f"Precision: {evaluation['precision']:.4f}, Recall: {evaluation['recall']:.4f}, "
                    f"F1: {evaluation['f1_score']:.4f}, ROC-AUC: {evaluation['roc_auc']:.4f}")
        
        # Save results
        model_dir = 'models/lstm_models'
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_file = os.path.join(model_dir, f'lstm_{config_name}_{timestamp}_config.json')
        results_file = os.path.join(model_dir, f'lstm_{config_name}_{timestamp}_results.json')
        config_to_save = {
            **config_params,
            'timestamp': timestamp,
            'config_name': config_name
        }
        with open(config_file, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        logger.info(f"Config saved to {config_file}")
        
        # Save results
        results_to_save = {
            'config_name': config_name,
            'timestamp': timestamp,
            'hyperparameters': config_params,
            'metrics': evaluation,
            'training_stats': results['training_stats'],
            'threshold': float(results['threshold'])
        }
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        logger.info(f"Results saved to {results_file}")
        
        return {
            'config_name': config_name,
            'metrics': evaluation,
            'timestamp': timestamp,
            'config_file': config_file,
            'results_file': results_file
        }
        
    except Exception as e:
        logger.exception(f"Training failed for {config_name}")
        return None


def compare_configurations(configs_to_train):
    """Train and compare multiple LSTM configurations."""
    
    logger.info(f"Comparing configurations: {', '.join(configs_to_train)}")
    
    results_list = []
    
    for config_name in configs_to_train:
        if config_name not in LSTM_CONFIGS:
            logger.warning(f"Unknown configuration: {config_name}")
            continue
        
        config_params = LSTM_CONFIGS[config_name]
        result = train_single_config(config_name, config_params)
        
        if result:
            results_list.append(result)
    
    # Comparison table
    if results_list:
        logger.info("Comparison Summary:")
        
        for result in results_list:
            metrics = result['metrics']
            logger.info(
                f"{result['config_name']}: Precision {metrics['precision']:.4f}, "
                f"Recall {metrics['recall']:.4f}, F1 {metrics['f1_score']:.4f}, "
                f"ROC-AUC {metrics['roc_auc']:.4f}"
            )
        
        # Find best configuration
        best_result = max(results_list, key=lambda x: x['metrics']['f1_score'])
        logger.info(f"Best F1-Score: {best_result['config_name']} ({best_result['metrics']['f1_score']:.4f})")
    
    return results_list


def list_configurations():
    """List all available LSTM configurations."""
    logger.info("Available LSTM Configurations:")
    
    for config_name, config_params in LSTM_CONFIGS.items():
        logger.info(f"{config_name.upper()}: Seq {config_params['sequence_length']}, "
                    f"Hidden {config_params['hidden_size']}, Epochs {config_params['epochs']}, "
                    f"Batch {config_params['batch_size']}, LR {config_params['learning_rate']}")


def main():
    """Command-line interface for LSTM training."""
    parser = argparse.ArgumentParser(
        description='Train LSTM autoencoder anomaly detection models'
    )
    
    parser.add_argument(
        '--config',
        choices=list(LSTM_CONFIGS.keys()),
        help='Configuration preset to use'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple configurations'
    )
    
    parser.add_argument(
        '--sequence',
        type=int,
        help='Custom sequence length'
    )
    
    parser.add_argument(
        '--hidden',
        type=int,
        help='Custom hidden size'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Custom number of epochs'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        help='Custom learning rate'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        help='Custom batch size'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available configurations'
    )
    
    parser.add_argument(
        '--data',
        default='data/engineered_features.csv',
        help='Path to engineered features CSV'
    )
    
    args = parser.parse_args()
    
    # List configurations
    if args.list:
        list_configurations()
        return
    
    # Compare multiple configurations
    if args.compare:
        configs_to_compare = list(LSTM_CONFIGS.keys())
        compare_configurations(configs_to_compare)
        return
    
    # Custom configuration
    if args.sequence or args.hidden or args.epochs or args.lr or args.batch:
        config_name = 'custom'
        base_config = LSTM_CONFIGS['default'].copy()
        
        if args.sequence:
            base_config['sequence_length'] = args.sequence
        if args.hidden:
            base_config['hidden_size'] = args.hidden
        if args.epochs:
            base_config['epochs'] = args.epochs
        if args.lr:
            base_config['learning_rate'] = args.lr
        if args.batch:
            base_config['batch_size'] = args.batch
        
        base_config['description'] = 'Custom configuration'
        
        train_single_config(config_name, base_config, args.data)
        return
    
    # Default: train with specified config or default
    config_name = args.config or 'default'
    config_params = LSTM_CONFIGS[config_name]
    
    train_single_config(config_name, config_params, args.data)


if __name__ == '__main__':
    main()
