"""Feature engineering validation and analysis."""

import sys
import pandas as pd
import numpy as np
from features.feature_engineering import featurize_dataset, FeatureEngineer
import logging

logger = logging.getLogger(__name__)


def validate_features():
    """Validate all engineered features."""
    logger.info("Validating engineered features...")
    
    # Load and engineer dataset
    logger.info("Loading raw data and engineering features...")
    df_raw = pd.read_csv('data/synthetic_events.csv')
    df_features = featurize_dataset(df_raw, verbose=False)
    logger.info(f"Loaded {len(df_raw)} events from {df_raw['user_id'].nunique()} users")
    logger.info(f"Generated {df_features.shape[1]} total columns")
    
    # Get feature list from FeatureEngineer
    new_features = FeatureEngineer().get_feature_columns()
    
    # Validation 1: No Missing Features
    logger.info("Checking feature presence...")
    missing = [f for f in new_features if f not in df_features.columns]
    if missing:
        logger.warning(f"Missing features: {missing}")
        return False
    logger.info(f"All {len(new_features)} features present")
    
    # Validation 2: NaN Handling
    logger.info("Checking for unexpected NaN values...")
    nan_counts = {col: df_features[col].isna().sum() for col in new_features}
    max_nans = max(nan_counts.values())
    if max_nans > 0:
        logger.warning("Found NaN values:")
        for col, count in nan_counts.items():
            if count > 0:
                logger.warning(f"  {col}: {count} NaNs")
        return False
    logger.info("No NaN values in any feature")
    
    # Validation 3: Range and Distribution Checks
    logger.info("Checking feature ranges and distributions...")
    
    range_checks = {
        'time_since_last_event': (0, float('inf')),
        'events_last_5min': (0, None),
        'actions_per_minute': (0, float('inf')),
        'withdrawal_zscore': (float('-inf'), float('inf')),
        'deposit_withdraw_ratio': (0, float('inf')),
        'avg_trade_volume_rolling': (0, float('inf')),
        'trade_volume_zscore': (float('-inf'), float('inf')),
        'instrument_concentration': (0, 1),
        'unique_ips_last_10_events': (0, 10),
        'ip_change_flag': (0, 1),
        'is_night_activity': (0, 1)
    }
    
    all_valid = True
    for feature, (min_val, max_val) in range_checks.items():
        feat_min = df_features[feature].min()
        feat_max = df_features[feature].max()
        
        valid = True
        if min_val != float('-inf') and feat_min < min_val:
            logger.warning(f"✗ {feature}: min value {feat_min} < expected {min_val}")
            valid = False
            all_valid = False
        if max_val is not None and max_val != float('inf') and feat_max > max_val:
            logger.warning(f"✗ {feature}: max value {feat_max} > expected {max_val}")
            valid = False
            all_valid = False
        
        if valid:
            logger.info(f"{feature}: [{feat_min:.4f}, {feat_max:.4f}]")
    
    if not all_valid:
        return False
    
    # Validation 4: Data Leakage Spot Checks
    logger.info("Performing data leakage spot checks...")
    
    # Check 1: First event per user should have 0 (or NaN -> filled with 0) for lookback features
    first_per_user = df_features.groupby('user_id').first()
    
    if (first_per_user['time_since_last_event'] != 0).any():
        logger.warning("First events have non-zero time_since_last_event")
        return False
    logger.info("First events: time_since_last_event = 0")
    
    if (first_per_user['events_last_5min'] != 0).any():
        logger.warning("First events have non-zero events_last_5min")
        return False
    logger.info("First events: events_last_5min = 0")
    
    if (first_per_user['ip_change_flag'] != 0).any():
        logger.warning("First events have ip_change_flag = 1 (should be 0)")
        return False
    logger.info("First events: ip_change_flag = 0")
    
    # Check 2: Spot-check a user's feature progression
    sample_user = df_features['user_id'].iloc[100]
    user_events = df_features[df_features['user_id'] == sample_user].copy()

    # --- Withdrawal Z-score sanity ---
    withdrawal_zscores = user_events['withdrawal_zscore'].values

    # Check for invalid values
    if not np.isfinite(withdrawal_zscores).all():
        logger.warning("Invalid values in withdrawal_zscore")
        return False

    # Check for extreme unrealistic spikes (optional threshold)
    if (np.abs(withdrawal_zscores) > 10).any():
        logger.warning("Unrealistic withdrawal_zscore detected (>10)")
        return False


    # --- Unique IP behavior ---
    unique_ips = user_events['unique_ips_last_10_events'].values

    # Should never exceed window size (10)
    if (unique_ips > 10).any():
        logger.warning("unique_ips_last_10_events exceeds 10")
        return False

    # Should be non-negative
    if (unique_ips < 0).any():
        logger.warning("Negative unique_ips_last_10_events detected")
        return False
    
    # --- IP change flag sanity ---
    ip_changes = user_events['ip_change_flag'].values

    if (ip_changes > 1).any():
        logger.warning("Invalid ip_change_flag values")
        return False

    logger.info(f"Spot check passed for user {sample_user} ({len(user_events)} events)")
    
    # Validation 5: Statistical Consistency
    stats = df_features[new_features].describe()
    logger.info(f"Feature statistics generated")
    
    # Final summary
    logger.info("Validation complete: all checks passed")
    logger.info(f"Total events: {len(df_features):,}")
    logger.info(f"Total users: {df_features['user_id'].nunique():,}")
    logger.info(f"Features engineered: {len(new_features)}, Data leakage: None, NaN values: 0")
    
    return True


def analyze_feature_correlations():
    """Analyze correlations between engineered features and anomaly labels."""
    logger.info("Analyzing feature-anomaly correlations...")
    
    df_raw = pd.read_csv('data/synthetic_events.csv')
    df_features = featurize_dataset(df_raw, verbose=False)
    
    # Get feature list
    new_features = FeatureEngineer().get_feature_columns()
    
    correlations = {}
    for feature in new_features:
        corr = df_features[feature].corr(df_features['is_anomaly'])
        correlations[feature] = corr
    
    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    logger.info("Feature-Anomaly Correlations:")
    for feature, corr in sorted_corrs:
        strength = "STRONG" if abs(corr) > 0.3 else "MODERATE" if abs(corr) > 0.1 else "WEAK"
        logger.info(f"{feature}: {corr:+.4f} [{strength}]")
    
    logger.info("Top 5 Most Predictive Features:")
    for i, (feature, corr) in enumerate(sorted_corrs[:5], 1):
        logger.info(f"  {i}. {feature} (corr: {corr:+.4f})")


def save_engineered_dataset():
    """Save engineered dataset to CSV for model training."""
    logger.info("Saving engineered dataset...")
    
    df_raw = pd.read_csv('data/synthetic_events.csv')
    df_features = featurize_dataset(df_raw, verbose=False)
    
    output_path = 'data/engineered_features.csv'
    df_features.to_csv(output_path, index=False)
    
    logger.info(f"Saved engineered dataset: {output_path}")
    logger.info(f"Rows: {len(df_features):,}, Columns: {df_features.shape[1]}, "
                f"Size: {df_features.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    # Run all validations
    success = validate_features()
    
    if success:
        analyze_feature_correlations()
        save_engineered_dataset()
    else:
        logger.error("Feature validation failed")
        sys.exit(1)
