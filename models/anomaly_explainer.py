"""
Anomaly Explainability Layer

Generates human-readable explanations for detected anomalies based on
feature values and simple interpretable rules.

Key features:
- Rule-based explanations (no black-box AI)
- Identifies top contributing features
- Ranks reasons by impact
- Human-friendly output
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class AnomalyExplainer:
    """
    Provides human-readable explanations for detected anomalies.
    
    Uses simple rule-based logic to identify why an event was flagged as anomalous.
    Thresholds are data-driven and computed from feature distributions using percentiles.
    """
    
    # Feature columns organized by type for cleaner processing
    Z_SCORE_FEATURES = ['withdrawal_zscore', 'trade_volume_zscore']
    BINARY_FLAG_FEATURES = ['ip_change_flag', 'is_night_activity']
    COUNT_FEATURES = ['unique_ips_last_10_events', 'events_last_5min']
    RATE_FEATURES = ['actions_per_minute']
    CONCENTRATION_FEATURES = ['instrument_concentration']
    TIME_FEATURES = ['time_since_last_event']
    RATIO_FEATURES = ['deposit_withdraw_ratio']
    VOLUME_FEATURES = ['avg_trade_volume_rolling']
    
    # Percentile levels for threshold computation
    # Most features use 95th percentile (top 5%), flag features use fixed values
    PERCENTILE_MAPPING = {
        'withdrawal_zscore': 0.95,
        'unique_ips_last_10_events': 0.95,
        'events_last_5min': 0.95,
        'trade_volume_zscore': 0.95,
        'actions_per_minute': 0.95,
        'instrument_concentration': 0.95,
        'time_since_last_event': 0.05,  # 5th percentile (lower is anomalous)
        'ip_change_flag': 0.5,  # Binary flag
        'is_night_activity': 0.5,  # Binary flag
        'deposit_withdraw_ratio': (0.05, 0.95),  # Both extremes (tuple for dual thresholds)
        'avg_trade_volume_rolling': 0.95,
    }
    
    # Fallback thresholds (used if compute_feature_stats() not called)
    FALLBACK_THRESHOLDS = {
        'withdrawal_zscore': 1.5,
        'unique_ips_last_10_events': 2.5,
        'events_last_5min': 5.0,
        'trade_volume_zscore': 2.0,
        'actions_per_minute': 10.0,
        'instrument_concentration': 0.8,
        'time_since_last_event': 0.1,
        'ip_change_flag': 0.5,
        'is_night_activity': 0.5,
        'deposit_withdraw_ratio': 0.01,
        'avg_trade_volume_rolling': 100.0,
    }
    
    # Human-readable reason templates (descriptions for each feature)
    REASON_DESCRIPTIONS = {
        'withdrawal_zscore': 'high withdrawal activity',
        'unique_ips_last_10_events': 'multiple IP addresses',
        'events_last_5min': 'high activity in short time',
        'trade_volume_zscore': 'unusual trading volume',
        'actions_per_minute': 'high action frequency',
        'instrument_concentration': 'concentrated trading',
        'time_since_last_event': 'very rapid sequential events',
        'ip_change_flag': 'IP address changes',
        'is_night_activity': 'unusual time activity',
        'deposit_withdraw_ratio': 'imbalanced deposits/withdrawals',
        'avg_trade_volume_rolling': 'large volume trading',
    }
    
    # Human-readable reason templates (for formatting)
    REASON_TEMPLATES = {
        'withdrawal_zscore': "Sudden withdrawal spike (zscore={:.1f})",
        'unique_ips_last_10_events': "Multiple IP addresses detected ({} IPs)",
        'events_last_5min': "High activity burst ({} events in 5min)",
        'trade_volume_zscore': "Unusual trading volume (zscore={:.1f})",
        'actions_per_minute': "High action frequency ({:.1f} actions/min)",
        'instrument_concentration': "Concentrated trading ({:.1%} in single instrument)",
        'time_since_last_event': "Very rapid sequential events ({:.2f}s apart)",
        'ip_change_flag': "IP address changed",
        'is_night_activity': "Activity during unusual hours",
        'deposit_withdraw_ratio': "Imbalanced deposit/withdrawal ratio ({:.2f})",
        'avg_trade_volume_rolling': "Large volume trading (${:.0f})",
    }
    
    def __init__(self, verbose: bool = True, use_fallback: bool = False):
        """Initialize explainer."""
        self.verbose = verbose
        self.use_fallback = use_fallback
        self.feature_stats = None
        self.thresholds = None
        self.feature_columns = FeatureEngineer().get_feature_columns()
    
    def _get_feature_type(self, feature: str) -> str:
        if feature in self.Z_SCORE_FEATURES:
            return 'z_score'
        elif feature in self.BINARY_FLAG_FEATURES:
            return 'binary_flag'
        elif feature in self.COUNT_FEATURES:
            return 'count'
        elif feature in self.RATE_FEATURES:
            return 'rate'
        elif feature in self.CONCENTRATION_FEATURES:
            return 'concentration'
        elif feature in self.TIME_FEATURES:
            return 'time'
        elif feature in self.RATIO_FEATURES:
            return 'ratio'
        elif feature in self.VOLUME_FEATURES:
            return 'volume'
        return 'generic'
    
    def _is_z_score_feature(self, feature: str) -> bool:
        return feature in self.Z_SCORE_FEATURES
    
    def _is_binary_flag_feature(self, feature: str) -> bool:
        return feature in self.BINARY_FLAG_FEATURES
    
    def _is_count_feature(self, feature: str) -> bool:
        return feature in self.COUNT_FEATURES
    
    def _is_rate_feature(self, feature: str) -> bool:
        return feature in self.RATE_FEATURES
    
    def _is_concentration_feature(self, feature: str) -> bool:
        return feature in self.CONCENTRATION_FEATURES
    
    def _is_time_feature(self, feature: str) -> bool:
        return feature in self.TIME_FEATURES
    
    def _is_ratio_feature(self, feature: str) -> bool:
        return feature in self.RATIO_FEATURES
    
    def _is_volume_feature(self, feature: str) -> bool:
        return feature in self.VOLUME_FEATURES
        
    def compute_feature_stats(self, df: pd.DataFrame) -> Dict:
        """Compute statistics on features and derive adaptive thresholds."""
        logger.info("Computing feature statistics and thresholds")
        stats = {}
        for col in self.feature_columns:
            if col in df.columns:
                stats[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q05': df[col].quantile(0.05),
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75),
                    'q90': df[col].quantile(0.90),
                    'q95': df[col].quantile(0.95),
                }
        
        self.feature_stats = stats
        self._compute_adaptive_thresholds(df)
        if self.verbose:
            logger.info(f"Feature stats computed for {len(stats)} features")
            self._log_thresholds()
        return stats
    
    def _compute_adaptive_thresholds(self, df: pd.DataFrame) -> None:
        """Compute data-driven thresholds from feature distributions."""
        self.thresholds = {}
        
        for feature, percentile_spec in self.PERCENTILE_MAPPING.items():
            if feature not in df.columns:
                # Feature not in data, use fallback
                self.thresholds[feature] = self.FALLBACK_THRESHOLDS[feature]
                continue
            
            try:
                # Handle dual thresholds (for features with both lower and upper bounds)
                if isinstance(percentile_spec, tuple):
                    # Example: deposit_withdraw_ratio has (0.05, 0.95)
                    lower_p, upper_p = percentile_spec
                    lower_val = df[feature].quantile(lower_p)
                    upper_val = df[feature].quantile(upper_p)
                    self.thresholds[feature] = (lower_val, upper_val)
                    
                # Handle single percentile (most features)
                else:
                    self.thresholds[feature] = df[feature].quantile(percentile_spec)
                    
            except Exception as e:
                logger.warning(f"Failed to compute threshold for {feature}: {e}")
                self.thresholds[feature] = self.FALLBACK_THRESHOLDS[feature]
    
    def _log_thresholds(self) -> None:
        """Log computed thresholds."""
        if not self.thresholds:
            return
        for feature, threshold in sorted(self.thresholds.items()):
            if isinstance(threshold, tuple):
                logger.info(f"{feature}: [{threshold[0]:.3f}, {threshold[1]:.3f}]")
            else:
                logger.info(f"{feature}: {threshold:.3f}")
    
    def _get_threshold(self, feature: str) -> float:
        """Get threshold for feature with fallback mechanism."""
        if self.thresholds and feature in self.thresholds:
            return self.thresholds[feature]
        if self.use_fallback or self.thresholds is None:
            if self.verbose and self.thresholds is None:
                logger.warning(f"Using fallback threshold for {feature}")
            return self.FALLBACK_THRESHOLDS.get(feature, 0.5)
        raise ValueError(f"Threshold for '{feature}' not available")
    
    def _compute_feature_impact(self, row: pd.Series) -> Dict[str, float]:
        """Compute impact score for each feature."""
        impacts = {}
        for col in self.feature_columns:
            if col not in row.index or col not in self.feature_stats:
                continue
            value = row[col]
            if pd.isna(value):
                impacts[col] = 0.0
                continue
            feature_type = self._get_feature_type(col)
            if feature_type == 'z_score':
                impacts[col] = self._compute_zscore_impact(value)
            elif feature_type == 'binary_flag':
                impacts[col] = self._compute_flag_impact(value)
            elif feature_type == 'count':
                impacts[col] = self._compute_count_impact(col, value)
            elif feature_type == 'rate':
                impacts[col] = self._compute_rate_impact(col, value)
            elif feature_type == 'concentration':
                impacts[col] = self._compute_concentration_impact(col, value)
            elif feature_type == 'time':
                impacts[col] = self._compute_time_impact(col, value)
            elif feature_type == 'ratio':
                impacts[col] = self._compute_ratio_impact(col, value)
            elif feature_type == 'volume':
                impacts[col] = self._compute_volume_impact(col, value)
            else:
                impacts[col] = self._compute_generic_impact(col, value)
        
        return impacts
    
    def _compute_zscore_impact(self, value: float) -> float:
        impact = min(abs(value) / 3.0, 1.0)  # Normalize by ~3 std
        return max(0.0, min(1.0, impact))
    
    def _compute_flag_impact(self, value: float) -> float:
        return float(value) if value > 0.5 else 0.0
    
    def _compute_count_impact(self, feature: str, value: float) -> float:
        stats = self.feature_stats.get(feature, {})
        max_val = stats.get('max', 1)
        if max_val > 0:
            impact = min(value / max_val, 1.0)
        else:
            impact = 0.0
        return max(0.0, min(1.0, impact))
    
    def _compute_rate_impact(self, feature: str, value: float) -> float:
        stats = self.feature_stats.get(feature, {})
        max_val = stats.get('max', 1)
        if max_val > 0:
            impact = min(value / max_val, 1.0)
        else:
            impact = 0.0
        return max(0.0, min(1.0, impact))
    
    def _compute_concentration_impact(self, feature: str, value: float) -> float:
        stats = self.feature_stats.get(feature, {})
        max_val = max(abs(stats.get('min', 0)), abs(stats.get('max', 1)))
        if max_val > 0:
            impact = min(abs(value) / max_val, 1.0)
        else:
            impact = 0.0
        return max(0.0, min(1.0, impact))
    
    def _compute_time_impact(self, feature: str, value: float) -> float:
        stats = self.feature_stats.get(feature, {})
        max_val = stats.get('max', 1)
        if max_val > 0:
            impact = min(value / max_val, 1.0)
        else:
            impact = 0.0
        return max(0.0, min(1.0, impact))
    
    def _compute_ratio_impact(self, feature: str, value: float) -> float:
        stats = self.feature_stats.get(feature, {})
        max_val = max(abs(stats.get('min', 0)), abs(stats.get('max', 1)))
        if max_val > 0:
            impact = min(abs(value) / max_val, 1.0)
        else:
            impact = 0.0
        return max(0.0, min(1.0, impact))
    
    def _compute_volume_impact(self, feature: str, value: float) -> float:
        stats = self.feature_stats.get(feature, {})
        max_val = stats.get('max', 1)
        if max_val > 0:
            impact = min(value / max_val, 1.0)
        else:
            impact = 0.0
        return max(0.0, min(1.0, impact))
    
    def _compute_generic_impact(self, feature: str, value: float) -> float:
        """Compute impact for generic features using std deviation."""
        stats = self.feature_stats.get(feature, {})
        mean = stats.get('mean', 0)
        std = stats.get('std', 1)
        if std > 0:
            impact = min(abs(value - mean) / (3 * std), 1.0)
        else:
            impact = 0.0
        return max(0.0, min(1.0, impact))
    
    def _check_anomaly_rules(self, row: pd.Series) -> List[Tuple[str, float]]:
        """Apply anomaly rules to identify problematic features."""
        triggered_rules = []
        for feature in self.feature_columns:
            if feature not in row.index:
                continue
            
            value = row[feature]
            if pd.isna(value):
                continue
            try:
                threshold = self._get_threshold(feature)
            except ValueError:
                continue
            is_anomalous, severity = self._check_feature_rule(feature, value, threshold)
            
            if is_anomalous:
                triggered_rules.append((feature, severity))
        
        # Sort by severity (descending)
        triggered_rules.sort(key=lambda x: x[1], reverse=True)
        
        return triggered_rules
    
    def _check_feature_rule(self, feature: str, value: float, threshold: float) -> Tuple[bool, float]:
        """Check if feature violates anomaly rule."""
        feature_type = self._get_feature_type(feature)
        
        if feature_type == 'z_score':
            return self._check_zscore_rule(value, threshold)
        elif feature_type == 'binary_flag':
            return self._check_flag_rule(value, threshold)
        elif feature_type == 'count':
            return self._check_count_rule(value, threshold)
        elif feature_type == 'rate':
            return self._check_rate_rule(value, threshold)
        elif feature_type == 'concentration':
            return self._check_concentration_rule(value, threshold)
        elif feature_type == 'time':
            return self._check_time_rule(value, threshold)
        elif feature_type == 'ratio':
            return self._check_ratio_rule(value, threshold)
        elif feature_type == 'volume':
            return self._check_volume_rule(value, threshold)
        else:
            return self._check_generic_rule(value, threshold)
    
    def _check_zscore_rule(self, value: float, threshold: float) -> Tuple[bool, float]:
        abs_value = abs(value)
        if abs_value > threshold:
            severity = min(abs_value / threshold, 2.0)  # Cap at 2x threshold
            return True, severity
        return False, 0.0
    
    def _check_flag_rule(self, value: float, threshold: float) -> Tuple[bool, float]:
        is_anomalous = value > 0.5
        severity = float(value) if is_anomalous else 0.0
        return is_anomalous, severity
    
    def _check_count_rule(self, value: float, threshold: float) -> Tuple[bool, float]:
        if value > threshold:
            severity = min(value / threshold, 2.0)
            return True, severity
        return False, 0.0
    
    def _check_rate_rule(self, value: float, threshold: float) -> Tuple[bool, float]:
        if value > threshold:
            severity = min(value / threshold, 2.0)
            return True, severity
        return False, 0.0
    
    def _check_concentration_rule(self, value: float, threshold: float) -> Tuple[bool, float]:
        if value > threshold:
            severity = min(value / threshold, 2.0)
            return True, severity
        return False, 0.0
    
    def _check_time_rule(self, value: float, threshold: float) -> Tuple[bool, float]:
        if value < threshold:
            severity = 1.0 - (value / threshold) if threshold > 0 else 1.0
            return True, severity
        return False, 0.0
    
    def _check_ratio_rule(self, value: float, threshold: float) -> Tuple[bool, float]:
        # Threshold can be a tuple (lower_bound, upper_bound)
        if isinstance(threshold, tuple):
            lower, upper = threshold
            is_in_range = lower <= value <= upper
            if not is_in_range:
                # Severity based on distance from acceptable range
                if value < lower:
                    severity = min((lower - value) / max(lower, 1.0), 2.0)
                else:
                    severity = min((value - upper) / max(upper, 1.0), 2.0)
                return True, severity
        else:
            # Fallback to single threshold
            if value < threshold or value > (1.0 / threshold if threshold > 0 else 100):
                severity = min(max(abs(threshold - value), 1.0), 2.0)
                return True, severity
        
        return False, 0.0
    
    def _check_volume_rule(self, value: float, threshold: float) -> Tuple[bool, float]:
        if value > threshold:
            severity = min(value / threshold, 2.0)
            return True, severity
        return False, 0.0
    
    def _check_generic_rule(self, value: float, threshold: float) -> Tuple[bool, float]:
        if abs(value) > threshold:
            severity = min(abs(value) / threshold, 2.0)
            return True, severity
        return False, 0.0
    
    def _extract_top_reasons(self, triggered_rules: List[Tuple[str, float]], 
                             row: pd.Series, top_k: int = 3) -> List[Dict]:
        """Extract top-k reasons with deduplication."""
        reasons = []
        seen_texts = set()
        for feature, severity in triggered_rules:
            reason_text = self._format_reason(feature, row)
            if reason_text in seen_texts:
                continue
            seen_texts.add(reason_text)
            reasons.append({
                'text': reason_text,
                'severity': float(severity),
                'feature': feature
            })
        reasons.sort(key=lambda x: x['severity'], reverse=True)
        return reasons[:top_k]
    
    def _format_reason(self, feature: str, row: pd.Series) -> str:
        """Format human-readable reason for anomalous feature."""
        if feature not in row.index:
            return f"Anomalous pattern in {feature}"
        
        value = row[feature]
        template = self.REASON_TEMPLATES.get(feature, f"Anomalous {feature}")
        
        try:
            # Try to format with value if template has placeholders
            if isinstance(template, str) and '{' in template:
                return self._format_reason_with_value(template, feature, value)
            else:
                return template
        except Exception:
            return f"Anomalous {feature}"
    
    def _format_reason_with_value(self, template: str, feature: str, value: float) -> str:
        """Format template with appropriate value conversion."""
        if self._is_z_score_feature(feature):
            return template.format(float(value))
        elif self._is_count_feature(feature):
            return template.format(int(value))
        elif self._is_rate_feature(feature):
            return template.format(float(value))
        elif self._is_concentration_feature(feature):
            return template.format(float(value))
        elif self._is_time_feature(feature):
            return template.format(float(value))
        elif self._is_ratio_feature(feature):
            return template.format(float(value) if value > 0 else 1.0)
        elif self._is_volume_feature(feature):
            return template.format(float(value))
        else:
            return template.format(float(value))
    
    def explain_anomalies(self, df: pd.DataFrame,
                         anomaly_scores: np.ndarray,
                         predictions: np.ndarray,
                         top_k: int = 3) -> pd.DataFrame:
        """Generate explanations for anomalous events."""
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        if len(anomaly_scores) != len(df) or len(predictions) != len(df):
            raise ValueError("Input lengths do not match")
        
        if self.feature_stats is None:
            self.compute_feature_stats(df)
        explanations = []
        for idx, (score, pred) in enumerate(zip(anomaly_scores, predictions)):
            explanation = {
                'index': idx,
                'anomaly_score': float(score),
                'predicted_label': int(pred),
                'reasons': [],
                'num_reasons': 0
            }
            if pred == 1:
                row = df.iloc[idx]
                triggered_rules = self._check_anomaly_rules(row)
                reasons = self._extract_top_reasons(triggered_rules, row, top_k)
                explanation['reasons'] = reasons
                explanation['num_reasons'] = len(reasons)
            explanations.append(explanation)
        results_df = pd.DataFrame(explanations)
        return results_df
    
    def explain_event(self, row: pd.Series, anomaly_score: float,
                     is_anomaly: bool = True) -> Dict:
        """Generate explanation for a single event."""
        explanation = {
            'anomaly_score': float(anomaly_score),
            'predicted_label': 1 if is_anomaly else 0,
            'reasons': [],
            'feature_impacts': {}
        }
        if not is_anomaly:
            return explanation
        if self.feature_stats is not None:
            impacts = self._compute_feature_impact(row)
            explanation['feature_impacts'] = {k: float(v) for k, v in impacts.items()}
        triggered_rules = self._check_anomaly_rules(row)
        top_reasons = self._extract_top_reasons(triggered_rules, row, top_k=3)
        explanation['reasons'] = [
            {'reason': r['text'], 'severity': r['severity'], 'feature': r['feature']}
            for r in top_reasons
        ]
        return explanation
    
    def create_explainable_results(self, df: pd.DataFrame,
                                  anomaly_scores: np.ndarray,
                                  predictions: np.ndarray) -> pd.DataFrame:
        """Create output with scores, predictions, and explanations."""
        explanations = self.explain_anomalies(df, anomaly_scores, predictions)
        results = explanations[['anomaly_score', 'predicted_label', 'num_reasons']].copy()
        results['reasons'] = explanations['reasons'].apply(
            lambda reasons_list: self._format_reasons_for_output(reasons_list)
        )
        return results
    
    def _format_reasons_for_output(self, reasons: List[Dict]) -> str:
        """Format reason dicts for output."""
        if not reasons:
            return 'No clear reasons'
        reason_texts = [r['text'] for r in reasons]
        return ' | '.join(reason_texts)
