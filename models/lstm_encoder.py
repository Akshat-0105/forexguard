"""
LSTM Autoencoder for Anomaly Detection

Time-series anomaly detection using LSTM autoencoders.
- Learns normal user behavior patterns from normal sequences
- Detects anomalies based on reconstruction error
- Processes sequences of engineered features per user
- Trains exclusively on normal data (is_anomaly == 0)
- Uses PyTorch for deep learning implementation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from typing import Tuple, Dict, Optional, List
import logging
import os

logger = logging.getLogger(__name__)

# Check device availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for sequence reconstruction.
    
    Architecture:
    - Encoder: LSTM layer(s) mapping input_size → hidden_size
    - Decoder: LSTM layer(s) reconstructing hidden_size → input_size
    - Output: Reconstructed sequence of same shape as input
    """
    
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1):
        """
        Initialize LSTM autoencoder.
        
        Args:
            input_size: Number of features per timestep
            hidden_size: Size of LSTM hidden state (bottleneck)
            num_layers: Number of LSTM layers (default: 1 for simplicity)
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder: compress sequence into hidden representation
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Decoder: reconstruct sequence from hidden representation
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Linear layer to map hidden state back to input dimension
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode sequence and reconstruct it.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Reconstructed sequence of same shape as input
        """
        # Encode: compress sequence
        _, (h_n, c_n) = self.encoder(x)
        
        # Get sequence length from input
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        
        # Decode: use hidden state to reconstruct
        # Initialize decoder input: start token repeated seq_len times
        # Use zeros with hidden_size input dimension
        decoder_input = torch.zeros(
            batch_size, seq_len, self.hidden_size,
            device=x.device, dtype=x.dtype
        )
        
        decoder_output, _ = self.decoder(decoder_input, (h_n, c_n))
        
        # Map hidden states to reconstructed features
        reconstructed = self.fc(decoder_output)
        
        return reconstructed


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for time-series sequences.
    
    Creates sequences of fixed length from user timelines.
    """
    
    def __init__(self, X: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Array of sequences (num_sequences, seq_len, num_features)
        """
        self.X = torch.FloatTensor(X)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]


class LSTMAutoencoderAnomalyDetector:
    """LSTM Autoencoder-based anomaly detection for user behavior sequences."""
    
    SEQUENCE_LENGTH = 10
    HIDDEN_SIZE = 32
    NUM_LAYERS = 1
    BATCH_SIZE = 64
    EPOCHS = 15
    LEARNING_RATE = 0.001
    PERCENTILE_THRESHOLD = 95
    
    def __init__(self, verbose: bool = True):
        """Initialize LSTM autoencoder detector."""
        self.verbose = verbose
        self.feature_columns = None
        self.model = None
        self.scaler = None
        self.threshold = None
        self.training_stats = {}
        self.evaluation_results = {}
        
        from features.feature_engineering import FeatureEngineer
        self.feature_columns = FeatureEngineer().get_feature_columns()
    
    def _create_sequences(self, data: np.ndarray, 
                         seq_length: int,
                         global_offset: int = 0) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Create rolling sequences from time-series data.
        
        Args:
            data: Input array (num_samples, num_features)
            seq_length: Length of sequences
            global_offset: Offset for global indices (for multi-user case)
            
        Returns:
            Tuple of (sequences array, metadata with GLOBAL indices)
        """
        sequences = []
        metadata = []  # Store GLOBAL indices for event alignment
        
        if len(data) < seq_length:
            return np.array([]), []
        
        for i in range(len(data) - seq_length + 1):
            sequence = data[i:i + seq_length]
            sequences.append(sequence)
            # Store GLOBAL indices: sequence spans [start, end)
            global_start = global_offset + i
            global_end = global_offset + i + seq_length
            metadata.append((global_start, global_end))
        
        return np.array(sequences), metadata
    
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Preprocess: sort by user, fit scaler on normal data only."""
        logger.info("Preprocessing data...")
        
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        X = df[self.feature_columns].copy()
        
        # Fit scaler only on normal data to prevent leakage
        normal_mask = df['is_anomaly'] == 0
        self.scaler = StandardScaler()
        self.scaler.fit(X[normal_mask])
        X_scaled = self.scaler.transform(X)
        
        logger.info(f"Data preprocessed: {X_scaled.shape}")
        return X_scaled, {'mean': self.scaler.mean_, 'std': self.scaler.scale_}
    
    def _create_user_sequences(self, X_scaled: np.ndarray, 
                              user_ids: np.ndarray,
                              is_anomaly: np.ndarray) -> Dict:
        """
        Create sequences grouped by user with GLOBAL index tracking for event alignment.
        
        Args:
            X_scaled: Scaled feature array
            user_ids: User ID for each sample
            is_anomaly: Anomaly label for each sample
            
        Returns:
            Dictionary with train/test sequences and metadata
        """
        logger.info("Creating user-grouped sequences with event-level tracking...")
        
        train_sequences = []
        test_sequences = []
        test_labels = []
        all_metadata = []  # Track GLOBAL indices for all sequences
        
        unique_users = np.unique(user_ids)
        global_offset = 0  # Track position in full sorted dataset
        
        for user_id in unique_users:
            user_mask = user_ids == user_id
            user_data = X_scaled[user_mask]
            user_labels = is_anomaly[user_mask]
            
            # Create sequences for this user with GLOBAL offset
            sequences, metadata = self._create_sequences(
                user_data, 
                self.SEQUENCE_LENGTH,
                global_offset=global_offset
            )
            
            if len(sequences) == 0:
                # Advance offset by user's data length
                global_offset += len(user_data)
                continue
            
            # Determine if sequences are anomalous
            # A sequence is anomalous if ANY sample in it is anomalous
            seq_labels = []
            for start_idx, end_idx in metadata:
                # Get labels for this sequence within the user's data
                user_start = start_idx - global_offset
                user_end = end_idx - global_offset
                labels_slice = user_labels[user_start:user_end]
                is_seq_anomalous = np.any(labels_slice)
                seq_labels.append(is_seq_anomalous)
            
            seq_labels = np.array(seq_labels)
            
            # Split into train (normal only) and test
            normal_mask = ~seq_labels
            train_sequences.extend(sequences[normal_mask])
            
            # All sequences go to test
            test_sequences.extend(sequences)
            test_labels.extend(seq_labels)
            all_metadata.extend(metadata)  # Store GLOBAL metadata
            
            # Advance offset for next user
            global_offset += len(user_data)
        
        train_sequences = np.array(train_sequences)
        test_sequences = np.array(test_sequences)
        test_labels = np.array(test_labels)
        all_metadata = np.array(all_metadata)
        
        logger.info(f"✓ Sequences created")
        logger.info(f"  Training sequences (normal only): {len(train_sequences)}")
        logger.info(f"  Test sequences: {len(test_sequences)}")
        logger.info(f"  Test anomalies: {test_labels.sum()} ({100*test_labels.sum()/len(test_labels):.1f}%)")
        
        return {
            'train': train_sequences,
            'test': test_sequences,
            'test_labels': test_labels,
            'metadata': all_metadata,  # GLOBAL indices for event alignment
            'n_events': len(X_scaled)  # Total number of events
        }
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train LSTM autoencoder on normal user behavior sequences.
        
        Steps:
        1. Preprocess: sort, normalize
        2. Create sequences grouped by user
        3. Train autoencoder on normal sequences
        4. Set anomaly threshold
        5. Evaluate on full test set
        
        Args:
            df: DataFrame with engineered features and is_anomaly label
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training LSTM autoencoder on normal sequences...")
        df_copy = df.copy()
        X_scaled, preprocess_stats = self._preprocess_data(df_copy)
        
        logger.info("Creating sequences grouped by user...")
        user_ids = df_copy['user_id'].values
        is_anomaly = df_copy['is_anomaly'].values
        
        sequences_dict = self._create_user_sequences(X_scaled, user_ids, is_anomaly)
        X_train = sequences_dict['train']
        X_test = sequences_dict['test']
        y_test = sequences_dict['test_labels']
        
        logger.info("Building LSTM autoencoder...")
        self.model = LSTMAutoencoder(
            input_size=len(self.feature_columns),
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS
        ).to(DEVICE)
        
        logger.info("Training model on normal sequences...")
        self._train_model(X_train)
        
        logger.info("Computing reconstruction errors...")
        train_errors_seq = self._compute_reconstruction_errors(X_train)
        test_errors_seq = self._compute_reconstruction_errors(X_test)
        
        test_metadata = sequences_dict['metadata']
        n_events_test = sequences_dict['n_events']
        test_errors = self._convert_sequence_errors_to_events(
            test_errors_seq,
            test_metadata,
            n_events_test,
            fill_method='mean'
        )
        
        train_metadata = sequences_dict['metadata'][~sequences_dict['test_labels']]
        train_errors_event = self._convert_sequence_errors_to_events(
            train_errors_seq,
            train_metadata,
            sequences_dict['n_events'],
            fill_method='mean'
        )
        self.threshold = np.percentile(train_errors_event, self.PERCENTILE_THRESHOLD)
        
        y_test_event = np.zeros(n_events_test, dtype=int)
        if len(test_metadata) > 0:
            first_seq_start = test_metadata[0][0]
            y_test_event[:first_seq_start] = 0
        for seq_idx, (start_idx, end_idx) in enumerate(test_metadata):
            last_event_idx = end_idx - 1
            y_test_event[last_event_idx] = y_test[seq_idx]
        
        # Make event-level predictions
        predictions = (test_errors > self.threshold).astype(int)
        evaluation = self._evaluate(y_test_event, predictions, test_errors)
        
        self.evaluation_results = evaluation
        self.training_stats = {
            'n_train': len(X_train),
            'n_test_sequences': len(X_test),
            'n_events': n_events_test,
            'n_features': len(self.feature_columns),
            'sequence_length': self.SEQUENCE_LENGTH,
            'hidden_size': self.HIDDEN_SIZE,
            'threshold': float(self.threshold),
            'alignment_method': f'Assign sequence error to last event ({self.SEQUENCE_LENGTH} events per sequence)',
            'error_stats': {
                'train_mean': float(train_errors_seq.mean()),
                'train_std': float(train_errors_seq.std()),
                'test_mean': float(test_errors.mean()),
                'test_std': float(test_errors.std())
            }
        }
        
        logger.info("=" * 80)
        logger.info("✓ Training complete!")
        logger.info("=" * 80 + "\n")
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'train_errors_seq': train_errors_seq,
            'test_errors_event': test_errors,
            'evaluation': evaluation,
            'training_stats': self.training_stats
        }
    
    def _train_model(self, X_train: np.ndarray) -> None:
        """
        Train the autoencoder on normal sequences.
        
        Args:
            X_train: Training sequences array
        """
        # Create data loader
        train_dataset = SequenceDataset(X_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        
        # Optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        criterion = nn.MSELoss()
        
        logger.info(f"  Batch size: {self.BATCH_SIZE}")
        logger.info(f"  Epochs: {self.EPOCHS}")
        logger.info(f"  Learning rate: {self.LEARNING_RATE}")
        logger.info(f"  Loss: MSE (reconstruction error)")
        
        # Training loop
        self.model.train()
        for epoch in range(self.EPOCHS):
            total_loss = 0
            for batch_X in train_loader:
                batch_X = batch_X.to(DEVICE)
                
                # Forward pass
                reconstructed = self.model(batch_X)
                loss = criterion(reconstructed, batch_X)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % max(1, self.EPOCHS // 5) == 0:
                logger.info(f"  Epoch {epoch+1}/{self.EPOCHS}, Loss: {avg_loss:.6f}")
        
        logger.info(f"✓ Model trained on {len(X_train)} normal sequences")
    
    def _compute_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for sequences.
        
        Args:
            X: Sequences array (num_sequences, seq_len, num_features)
            
        Returns:
            Array of reconstruction errors (num_sequences,)
        """
        dataset = SequenceDataset(X)
        loader = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        
        errors = []
        self.model.eval()
        
        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X.to(DEVICE)
                reconstructed = self.model(batch_X)
                
                # MSE per sample in batch
                batch_errors = torch.mean((batch_X - reconstructed) ** 2, dim=(1, 2))
                errors.extend(batch_errors.cpu().numpy())
        
        return np.array(errors)
    
    def _convert_sequence_errors_to_events(self, seq_errors: np.ndarray,
                                          metadata: np.ndarray,
                                          n_events: int,
                                          fill_method: str = 'mean') -> np.ndarray:
        """Convert sequence errors to event-level by propagating to all events in sequence."""
        event_scores = np.zeros(n_events)
        for seq_idx, (start_idx, end_idx) in enumerate(metadata):
            seq_error = seq_errors[seq_idx]
            for event_idx in range(start_idx, end_idx):
                event_scores[event_idx] = max(event_scores[event_idx], seq_error)
        return event_scores
    
    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 errors: np.ndarray) -> Dict:
        """
        Evaluate predictions against ground truth.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            errors: Reconstruction errors
            
        Returns:
            Dictionary with evaluation metrics
        """
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = (y_pred == y_true).mean()
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # ROC-AUC
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, errors)
        else:
            roc_auc = None
        
        logger.info(f"\nClassification Metrics:")
        logger.info(f"  Precision:  {precision:.4f}")
        logger.info(f"  Recall:     {recall:.4f}")
        logger.info(f"  F1-Score:   {f1:.4f}")
        logger.info(f"  Accuracy:   {accuracy:.4f}")
        if roc_auc is not None:
            logger.info(f"  ROC-AUC:    {roc_auc:.4f}")
        
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  True Negatives:  {tn} (correct normal)")
        logger.info(f"  False Positives: {fp} (false alarms)")
        logger.info(f"  False Negatives: {fn} (missed anomalies)")
        logger.info(f"  True Positives:  {tp} (detected anomalies)")
        
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
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate event-level predictions on new data.
        
        Returns one anomaly score per EVENT (not per sequence), aligned with original data.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            Tuple of (event_level_anomaly_scores, event_level_predictions)
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model must be trained first")
        
        logger.info("Generating event-level predictions...")
        
        # Preprocess
        df_copy = df.copy()
        df_copy = df_copy.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        # Auto-engineer features if needed
        if not all(f in df_copy.columns for f in self.feature_columns):
            from features.feature_engineering import FeatureEngineer
            fe = FeatureEngineer()
            df_copy = fe.transform(df_copy)
        
        X = df_copy[self.feature_columns].copy()
        X_scaled = self.scaler.transform(X)
        
        user_ids = df_copy['user_id'].values
        
        # Create sequences with metadata for alignment
        sequences_dict = self._create_user_sequences(
            X_scaled, 
            user_ids, 
            np.zeros(len(X_scaled))  # No labels for prediction
        )
        X_seq = sequences_dict['test']
        metadata = sequences_dict['metadata']
        n_events = sequences_dict['n_events']
        
        # Compute sequence-level reconstruction errors
        seq_errors = self._compute_reconstruction_errors(X_seq)
        
        # Convert to event-level scores
        event_errors = self._convert_sequence_errors_to_events(
            seq_errors,
            metadata,
            n_events,
            fill_method='mean'
        )
        
        # Make event-level predictions
        predictions = (event_errors > self.threshold).astype(int)
        
        logger.info(f"✓ Event-level predictions generated")
        logger.info(f"  Total events: {n_events}")
        anomaly_pct = (100 * predictions.sum() / n_events) if n_events > 0 else 0
        logger.info(f"  Anomalies detected: {predictions.sum()} ({anomaly_pct:.1f}%)")
        
        return event_errors, predictions
    
    def save_model(self, model_path: str = 'models/lstm_model.pt',
                   scaler_path: str = 'models/lstm_scaler.pkl') -> None:
        """Save model, scaler, and threshold."""
        if self.model is None:
            raise RuntimeError("No trained model to save")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        
        import pickle
        metadata = {'scaler': self.scaler, 'threshold': self.threshold, 'feature_columns': self.feature_columns}
        with open(scaler_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"✓ Model, scaler, and threshold saved")
    
    def load_model(self, model_path: str = 'models/lstm_model.pt',
                   scaler_path: str = 'models/lstm_scaler.pkl') -> None:
        """Load model, scaler, and threshold."""
        self.model = LSTMAutoencoder(
            input_size=len(self.feature_columns),
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS
        ).to(DEVICE)
        
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        
        import pickle
        with open(scaler_path, 'rb') as f:
            metadata = pickle.load(f)
            if isinstance(metadata, dict):
                self.scaler = metadata.get('scaler')
                self.threshold = metadata.get('threshold')
                self.feature_columns = metadata.get('feature_columns', self.feature_columns)
            else:
                self.scaler = metadata
        
        logger.info(f"✓ Model, scaler, and threshold loaded")


def train_lstm_autoencoder(df: pd.DataFrame, verbose: bool = True) -> Tuple[Dict, 'LSTMAutoencoderAnomalyDetector']:
    """
    Convenience function: train LSTM autoencoder and return results.
    
    Returns event-level anomaly scores and predictions.
    
    Args:
        df: DataFrame with engineered features and is_anomaly label
        verbose: Enable logging
        
    Returns:
        Tuple of (results_dict, detector_instance)
    """
    detector = LSTMAutoencoderAnomalyDetector(verbose=verbose)
    results = detector.train(df)
    
    return results, detector



