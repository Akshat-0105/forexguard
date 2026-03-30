"""
Feature Engineering Module for ForexGuard Anomaly Detection

Generates advanced features from raw trading events with proper temporal ordering and no data leakage.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generates advanced features from raw trading events for anomaly detection."""
    
    # Rolling window sizes
    ROLLING_WINDOW_EVENTS = 20
    RECENT_WINDOW_EVENTS = 10
    ROLLING_TIME_WINDOW_MIN = 5
    ZSCORE_WINDOW = 20
    
    def __init__(self, verbose: bool = True):
        """Initialize feature engineer with engineered column names."""
        self.verbose = verbose
        self.df = None
        self.event_counts = {}
        self.feature_columns = [
            'time_since_last_event', 'events_last_5min', 'actions_per_minute',
            'withdrawal_zscore', 'deposit_withdraw_ratio',
            'avg_trade_volume_rolling', 'trade_volume_zscore', 'instrument_concentration',
            'unique_ips_last_10_events', 'ip_change_flag',
            'is_night_activity'
        ]
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias for engineer_features (sklearn compatibility)."""
        return self.engineer_features(df)
    
    def get_feature_columns(self):
        """Return list of engineered feature column names."""
        return self.feature_columns
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main entry point: compute all features from raw data."""
        df = self._prepare_data(df)
        self.df = df.copy()
        
        logger.info("Starting feature engineering pipeline...")
        
        df = self._add_time_features(df)
        df = self._add_financial_features(df)
        df = self._add_trading_features(df)
        df = self._add_identity_features(df)
        df = self._add_behavioral_features(df)
        
        logger.info(f"✓ Feature engineering complete: {df.shape[1] - self.df.shape[1]} new features")
        logger.info(f"  Total columns: {df.shape[1]} (original: {self.df.shape[1]})")
        
        df[self.feature_columns] = df[self.feature_columns].astype(float)
        
        return df
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate, convert types, and sort data by user and timestamp."""
        df = df.copy()
        
        required_cols = ['user_id', 'timestamp', 'event_type', 'ip_address', 'device_id', 
                         'amount', 'trade_volume', 'instrument']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0).astype(float)
        df['trade_volume'] = pd.to_numeric(df['trade_volume'], errors='coerce').fillna(0.0).astype(float)
        
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        logger.info(f"Data prepared: {len(df)} events from {df['user_id'].nunique()} users")
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features: time_since_last_event, events_last_5min, actions_per_minute."""
        logger.info("Computing time features...")
        
        df['time_since_last_event'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
        
        def count_events_in_window(group):
            timestamps = group['timestamp']
            result = []
            for i in range(len(timestamps)):
                if i == 0:
                    result.append(0)
                else:
                    window_start = timestamps.iloc[i] - timedelta(minutes=self.ROLLING_TIME_WINDOW_MIN)
                    count = timestamps.iloc[:i].between(window_start, timestamps.iloc[i] - pd.Timedelta(seconds=1)).sum()
                    result.append(count)
            return pd.Series(result, index=group.index)
        
        df['events_last_5min'] = df.groupby('user_id', group_keys=False).apply(
            count_events_in_window
        ).reset_index(level=0, drop=True)
        
        def actions_per_minute(group):
            """Rate of events over last 20 events."""
            result = pd.Series(0.0, index=group.index, dtype=float)
            for i, row in enumerate(group.itertuples()):
                if i < 2:
                    result.iloc[i] = 0.0
                else:
                    start_idx = max(0, i - self.ROLLING_WINDOW_EVENTS)
                    time_span_min = (row.timestamp - group['timestamp'].iloc[start_idx]).total_seconds() / 60.0
                    if time_span_min > 0:
                        event_count = max(0, i - start_idx)
                        result.iloc[i] = event_count / time_span_min
                    else:
                        result.iloc[i] = 0.0
            return result
        
        df['actions_per_minute'] = df.groupby('user_id', group_keys=False).apply(
            actions_per_minute
        ).reset_index(level=0, drop=True)
        
        df = df.assign(
            time_since_last_event=df['time_since_last_event'].fillna(0),
            events_last_5min=df['events_last_5min'].fillna(0),
            actions_per_minute=df['actions_per_minute'].fillna(0)
        )
        
        logger.info(f"  ✓ Time features: time_since_last_event, events_last_5min, actions_per_minute")
        
        return df
    
    def _add_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add financial features: withdrawal_zscore, deposit_withdraw_ratio."""
        logger.info("Computing financial features...")
        
        def withdrawal_zscore(group):
            """Z-score of withdrawal amounts."""
            result = pd.Series(0.0, index=group.index, dtype=float)
            for i in range(len(group)):
                past_data = group.iloc[:i]
                withdrawals = past_data[past_data['event_type'] == 'withdrawal']['amount']
                
                if len(withdrawals) >= 2:
                    mean_withdrawal = withdrawals.mean()
                    std_withdrawal = withdrawals.std()
                    
                    if std_withdrawal > 0 and group.iloc[i]['event_type'] == 'withdrawal':
                        z_score = (group.iloc[i]['amount'] - mean_withdrawal) / std_withdrawal
                        result.iloc[i] = z_score
                    else:
                        result.iloc[i] = 0.0
                else:
                    result.iloc[i] = 0.0
            
            return result
        
        df['withdrawal_zscore'] = df.groupby('user_id', group_keys=False).apply(
            withdrawal_zscore
        ).reset_index(level=0, drop=True)
        
        def deposit_withdraw_ratio(group):
            """Ratio of withdrawals to deposits over rolling window."""
            result = pd.Series(1.0, index=group.index, dtype=float)
            for i in range(1, len(group)):
                start_idx = max(0, i - self.ROLLING_WINDOW_EVENTS)
                past_window = group.iloc[start_idx:i]
                
                total_deposits = past_window[past_window['event_type'] == 'deposit']['amount'].sum()
                total_withdrawals = past_window[past_window['event_type'] == 'withdrawal']['amount'].sum()
                
                if total_deposits > 0:
                    ratio = total_withdrawals / total_deposits
                    result.iloc[i] = ratio
                else:
                    result.iloc[i] = 0.0
            
            return result
        
        df['deposit_withdraw_ratio'] = df.groupby('user_id', group_keys=False).apply(
            deposit_withdraw_ratio
        ).reset_index(level=0, drop=True)
        
        df = df.assign(
            withdrawal_zscore=df['withdrawal_zscore'].fillna(0),
            deposit_withdraw_ratio=df['deposit_withdraw_ratio'].fillna(0)
        )
        
        logger.info(f"  ✓ Financial features: withdrawal_zscore, deposit_withdraw_ratio")
        
        return df
    
    def _add_trading_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trading features: avg_trade_volume_rolling, trade_volume_zscore, instrument_concentration."""
        logger.info("Computing trading features...")
        
        def avg_trade_volume_rolling(group):
            """Mean trade volume over rolling window."""
            result = pd.Series(0.0, index=group.index, dtype=float)
            for i in range(len(group)):
                past_data = group.iloc[:i]
                trades = past_data[past_data['event_type'] == 'trade']['trade_volume']
                result.iloc[i] = trades.mean() if len(trades) > 0 else 0.0
            return result
        
        df['avg_trade_volume_rolling'] = df.groupby('user_id', group_keys=False).apply(
            avg_trade_volume_rolling
        ).reset_index(level=0, drop=True)
        
        def trade_volume_zscore(group):
            """Z-score of trade volumes."""
            result = pd.Series(0.0, index=group.index, dtype=float)
            for i in range(len(group)):
                if group.iloc[i]['event_type'] == 'trade' and i > 0:
                    past_data = group.iloc[:i].copy()
                    past_volumes = past_data.loc[past_data['event_type'] == 'trade', 'trade_volume']
                    
                    if len(past_volumes) >= 2:
                        mean_vol = past_volumes.mean()
                        std_vol = past_volumes.std()
                        if std_vol > 0:
                            z_score = (group.iloc[i]['trade_volume'] - mean_vol) / std_vol
                            result.iloc[i] = z_score
                        else:
                            result.iloc[i] = 0.0
                    else:
                        result.iloc[i] = 0.0
                else:
                    result.iloc[i] = 0.0
            
            return result
        
        df['trade_volume_zscore'] = df.groupby('user_id', group_keys=False).apply(
            trade_volume_zscore
        ).reset_index(level=0, drop=True)
        
        def instrument_concentration(group):
            """Max instrument frequency / total instruments."""
            group = group.reset_index(drop=True)
            result = pd.Series(0.0, index=range(len(group)), dtype=float)
            for i in range(len(group)):
                past_data = group.iloc[:i]
                past_trades = past_data[past_data['event_type'] == 'trade']
                
                if len(past_trades) > 0:
                    instr_counts = past_trades['instrument'].value_counts()
                    max_count = instr_counts.iloc[0] if len(instr_counts) > 0 else 0
                    total = len(past_trades)
                    concentration = max_count / total if total > 0 else 0.0
                    result.iloc[i] = concentration
                else:
                    result.iloc[i] = 0.0
            
            return result
        
        df['instrument_concentration'] = df.groupby('user_id', group_keys=False).apply(
            instrument_concentration
        ).reset_index(level=0, drop=True)
        
        df = df.assign(
            avg_trade_volume_rolling=df['avg_trade_volume_rolling'].fillna(0),
            trade_volume_zscore=df['trade_volume_zscore'].fillna(0),
            instrument_concentration=df['instrument_concentration'].fillna(0)
        )
        
        logger.info(f"  ✓ Trading features: avg_trade_volume_rolling, trade_volume_zscore, instrument_concentration")
        
        return df
    
    def _add_identity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add identity features: unique_ips_last_10_events, ip_change_flag."""
        logger.info("Computing identity features...")
        
        def unique_ips_last_10(group):
            """Unique IP count in last 10 events."""
            result = pd.Series(0, index=group.index, dtype=int)
            for i in range(len(group)):
                start_idx = max(0, i - self.RECENT_WINDOW_EVENTS + 1)
                window = group.iloc[start_idx:i]
                result.iloc[i] = window['ip_address'].nunique()
            return result
        
        df['unique_ips_last_10_events'] = df.groupby('user_id', group_keys=False).apply(
            unique_ips_last_10
        ).reset_index(level=0, drop=True)
        
        def ip_change_flag(group):
            """Flag IP changes from previous event."""
            result = pd.Series(0, index=group.index, dtype=int)
            for i in range(len(group)):
                if i > 0 and group.iloc[i]['ip_address'] != group.iloc[i - 1]['ip_address']:
                    result.iloc[i] = 1
                else:
                    result.iloc[i] = 0
            return result
        
        df['ip_change_flag'] = df.groupby('user_id', group_keys=False).apply(
            ip_change_flag
        ).reset_index(level=0, drop=True)
        
        logger.info(f"  ✓ Identity features: unique_ips_last_10_events, ip_change_flag")
        
        return df
    
    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral features: is_night_activity (0-5 UTC)."""
        logger.info("Computing behavioral features...")
        
        df['is_night_activity'] = df['timestamp'].dt.hour.apply(
            lambda h: 1 if 0 <= h <= 5 else 0
        )
        
        logger.info(f"  ✓ Behavioral features: is_night_activity")
        
        return df


def featurize_dataset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Apply feature engineering to a dataset."""
    engineer = FeatureEngineer(verbose=verbose)
    return engineer.engineer_features(df)


def main():
    """Load synthetic dataset and engineer features."""
    import os
    
    logger.info("=" * 70)
    logger.info("ForexGuard Feature Engineering Module")
    logger.info("=" * 70)
    
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_events.csv')
    
    if os.path.exists(csv_path):
        logger.info(f"\nLoading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        df_features = featurize_dataset(df, verbose=True)
        
        logger.info(f"\n{'='*70}")
        logger.info("FEATURE ENGINEERING RESULTS")
        logger.info(f"{'='*70}")
        logger.info(f"Original columns: {df.shape[1]}")
        logger.info(f"Engineered columns: {df_features.shape[1]}")
        logger.info(f"New features added: {df_features.shape[1] - df.shape[1]}")
        
        logger.info(f"\nNew feature columns:")
        new_cols = set(df_features.columns) - set(df.columns)
        for col in sorted(new_cols):
            logger.info(f"  - {col}")
        
        logger.info(f"\nSample engineered features (first 5 rows):")
        sample_cols = list(new_cols)[:6]
        logger.info(f"\n{df_features[['user_id', 'event_type', 'timestamp'] + sample_cols].head()}")
        
        logger.info(f"\n{'='*70}")
        logger.info("✓ Feature engineering complete!")
        logger.info(f"{'='*70}\n")
        
        return df_features
    else:
        logger.warning(f"Dataset not found at {csv_path}")
        logger.info("Generate synthetic dataset first using: python data/generator.py")
        return None


if __name__ == '__main__':
    df_engineered = main()
    if df_engineered is not None:
        output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'engineered_features.csv')
        df_engineered.to_csv(output_path, index=False)
        logger.info(f"✓ Engineered features saved to {output_path}")
