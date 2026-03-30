"""
ForexGuard Data Generation Module

Generates synthetic trading event datasets with realistic, sequential user timelines
and injected anomalies for model training and validation.

Key Design Principles:
- Event-driven: Each user has a continuous timeline
- Session-based: Realistic login → actions → logout flows
- Relative anomalies: Based on user baseline behavior (no data leakage)
- Data leak prevention: Only raw events, no derived features
- Time realism: Activity hours, inter-event gaps, realistic distributions

Anomaly Types (integrated into timelines):
1. Multi-IP login within short time
2. Impossible travel (geo jump)
3. Deposit → no trade → withdrawal
4. Sudden large withdrawal (relative to user baseline)
5. Dormant account → sudden activity burst
6. Trade volume spike (5x user baseline)
7. Single instrument concentration
8. High-frequency actions (bot-like)
9. Multi-user same IP (shared compromised IP)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class UserProfile:
    """Encapsulates user behavior characteristics and statistics."""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        
        # Profile characteristics
        self.risk_profile = np.random.choice(['conservative', 'moderate', 'aggressive'], 
                                             p=[0.3, 0.5, 0.2])
        self.is_active = np.random.choice([True, False], p=[0.75, 0.25])
        self.timezone_offset = np.random.randint(-12, 13)
        
        # Primary identifiers
        self.primary_ip = self._generate_ip()
        self.primary_device = f"device_{np.random.randint(1, 301)}"
        self.session_id = np.random.randint(10000, 99999)
        
        # Behavior statistics (updated during generation for relative anomalies)
        self.avg_withdrawal = {'conservative': 800, 'moderate': 2000, 'aggressive': 8000}[self.risk_profile]
        self.avg_deposit = {'conservative': 1000, 'moderate': 2500, 'aggressive': 10000}[self.risk_profile]
        self.avg_trade_volume = {'conservative': 0.1, 'moderate': 0.5, 'aggressive': 2.0}[self.risk_profile]
        self.max_trade_volume_seen = self.avg_trade_volume
        self.trading_hours_distribution = self._generate_trading_hours()
        
        # Timeline tracking
        self.last_event_time = None
        self.last_ip = self.primary_ip
        self.last_device = self.primary_device
        self.event_count = 0
        
        # Behavior shift tracking (NEW)
        self.behavior_shift_point = None  # Event index where behavior changes
        self.behavior_shifted = False
        self.post_shift_volume_multiplier = 1.0  # Volume increase post-shift
        self.post_shift_session_freq_multiplier = 1.0  # Session frequency increase
        self.post_shift_instruments = None  # Changed instruments set
        
        # Sequence tracking for deterministic anomalies
        self.deposit_no_trade_sequence = None  # (deposit_idx, next_withdrawal_idx)
        self.awaiting_withdrawal = False  # Flag to enforce no-trade between deposit and withdrawal
        
    def _generate_ip(self) -> str:
        """Generate a realistic IP address."""
        return f"192.168.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}"
    
    def _generate_trading_hours(self) -> np.ndarray:
        """
        Generate realistic trading hours distribution.
        More active during business hours of their timezone.
        """
        hours = np.zeros(24)
        # Peak during business hours (9-17 / 9AM-5PM in their timezone)
        center = (self.timezone_offset + 12) % 24
        for i in range(24):
            hours[i] = np.exp(-((i - center) ** 2) / 10)
        return hours / hours.sum()  # Normalize to probability distribution
    
    def get_activity_probability(self, hour: int) -> float:
        """Get probability of activity during a specific UTC hour."""
        return float(self.trading_hours_distribution[hour])
    
    def update_statistics(self, event_type: str, amount: float, trade_volume: float):
        """Update user statistics based on observed events (EMA smoothing)."""
        if event_type == 'deposit' and amount > 0:
            self.avg_deposit = 0.7 * self.avg_deposit + 0.3 * amount
        elif event_type == 'withdrawal' and amount > 0:
            self.avg_withdrawal = 0.7 * self.avg_withdrawal + 0.3 * amount
        elif event_type == 'trade' and trade_volume > 0:
            self.avg_trade_volume = 0.7 * self.avg_trade_volume + 0.3 * trade_volume
            self.max_trade_volume_seen = max(self.max_trade_volume_seen, trade_volume)
    
    def initialize_behavior_shift(self, total_events: int):
        """Initialize behavior shift point."""
        low = max(int(total_events * 0.5), 1)
        high = max(int(total_events * 0.7), low + 1)
        shift_point = np.random.randint(low, high)
        self.behavior_shift_point = shift_point
        self.post_shift_volume_multiplier = np.random.uniform(3, 8)
        self.post_shift_session_freq_multiplier = np.random.uniform(1.5, 3)
        num_post_instruments = np.random.randint(2, 4)
        self.post_shift_instruments = np.random.choice(
            DataGenerator.INSTRUMENTS,
            size=num_post_instruments, replace=False
        )


class DataGenerator:
    """Generates synthetic forex trading data with realistic sequential user timelines."""
    
    # Configuration constants
    NUM_EVENTS = 50000
    NUM_USERS = 500
    INSTRUMENTS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD',
                   'USDCAD', 'USDCHF', 'EURGBP', 'EURJPY', 'GBPJPY']
    
    # Anomaly injection parameters
    ANOMALY_RATE = 0.12  # ~12% of users will have anomalies
    ANOMALY_CONCENTRATION = 0.25  # When anomaly occurs, 25% of user's events are anomalous
    
    # Anomaly types: 1-10
    ANOMALY_TYPES = {
        1: 'multi_ip_login',
        2: 'impossible_travel',
        3: 'deposit_no_trade_withdrawal',
        4: 'large_withdrawal',
        5: 'dormant_activity',
        6: 'trade_volume_spike',
        7: 'instrument_concentration',
        8: 'high_frequency_bot',
        9: 'multi_user_same_ip',
        10: 'behavior_shift'  # NEW: account takeover/strategy change
    }
    
    def __init__(self, n_events: int = NUM_EVENTS, random_seed: int = 42):
        """Initialize the data generator with event count and random seed."""
        self.n_events = n_events
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # User profiles and tracking
        self.user_profiles: Dict[int, UserProfile] = {}
        self.user_anomaly_flags: Dict[int, bool] = {}
        self.anomaly_types: Dict[int, int] = {}
        self.shared_ip_groups: Dict[str, List[int]] = {}
        
        # Initialize users
        self._initialize_users()
        
        logger.info(f"DataGenerator initialized for ~{self.n_events} events across {len(self.user_profiles)} users")
    
    def _initialize_users(self):
        """Initialize user profiles and assign anomalies."""
        for user_id in range(1, self.NUM_USERS + 1):
            profile = UserProfile(user_id)
            self.user_profiles[user_id] = profile
            
            # Assign anomalies to ~12% of users
            has_anomaly = np.random.random() < self.ANOMALY_RATE
            self.user_anomaly_flags[user_id] = has_anomaly
            
            if has_anomaly:
                # Randomly select an anomaly type for this user
                self.anomaly_types[user_id] = np.random.randint(1, 11)
                
                # Initialize behavior shift if anomaly type is 10
                if self.anomaly_types[user_id] == 10:
                    profile.initialize_behavior_shift(self.n_events // self.NUM_USERS)
            
            # Build shared IP groups (multi-user same IP anomaly - type 9)
            if np.random.random() < 0.02:  # 2% of users share IPs
                ip = profile.primary_ip
                if ip not in self.shared_ip_groups:
                    self.shared_ip_groups[ip] = []
                self.shared_ip_groups[ip].append(user_id)
    
    def generate(self) -> pd.DataFrame:
        """Generate synthetic dataset with sequential user timelines and anomalies."""
        events = []
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        
        logger.info(f"Generating ~{self.n_events} events with sequential timelines...")
        
        # Calculate events per user
        events_per_user = self.n_events // self.NUM_USERS
        
        # Generate timeline for each user
        for user_id in range(1, self.NUM_USERS + 1):
            profile = self.user_profiles[user_id]
            
            # Skip inactive users occasionally
            if not profile.is_active and np.random.random() < 0.3:
                continue
            
            # Generate user's event timeline
            user_events = self._generate_user_timeline(
                user_id=user_id,
                num_events=events_per_user,
                base_time=base_time,
                profile=profile
            )
            events.extend(user_events)
            
            if user_id % 100 == 0:
                logger.info(f"  Generated timeline for user {user_id}/{self.NUM_USERS}")
        
        # Create DataFrame and sort by timestamp
        df = pd.DataFrame(events)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # FIX: ensure anomaly_type is proper nullable integer
        df['anomaly_type'] = df['anomaly_type'].astype('Int64')
        
        logger.info(f"Generated {len(df)} total events across {df['user_id'].nunique()} active users")
        
        # Post-processing for production-ready dataset
        logger.info("Applying post-processing enhancements...")
        
        # 1. Strengthen Type 9 multi-user same IP with overlapping sessions
        df = self._enforce_type9_overlapping_sessions(df)
        
        # 2. Add user-level anomaly labels
        df = self._add_user_level_labels(df)
        
        # 3. Ensure consistency in anomaly labeling
        df = self._ensure_label_consistency(df)
        
        logger.info("Post-processing complete")
        return df
    
    def _generate_user_timeline(self, user_id: int, num_events: int, 
                               base_time: datetime, profile: UserProfile) -> List[Dict]:
        """Generate a realistic timeline of events with anomalies injected proportionally."""
        events = []
        current_time = base_time
        in_session = False
        session_events = 0
        
        has_anomaly = self.user_anomaly_flags[user_id]
        anomaly_type = self.anomaly_types.get(user_id, None)
        anomaly_event_budget = int(num_events * self.ANOMALY_CONCENTRATION) if has_anomaly else 0
        anomaly_injected = 0
        
        # Burst phase tracking for high-frequency anomaly (type 8)
        in_burst_phase = False
        burst_end_time = None
        
        # Deposit-no-trade-withdrawal tracking (type 3)
        awaiting_withdrawal_for = None  # Timestamp when deposit happened
        
        for event_idx in range(num_events):
            # Check if we should enter behavior shift phase (anomaly type 10)
            if has_anomaly and anomaly_type == 10 and not profile.behavior_shifted:
                if event_idx >= profile.behavior_shift_point:
                    profile.behavior_shifted = True
                    logger.debug(f"User {user_id} behavior shift at event {event_idx}")
            
            # Trigger burst phase for high-frequency anomaly (type 8)
            if has_anomaly and anomaly_type == 8 and not in_burst_phase and np.random.random() < 0.05:
                in_burst_phase = True
                burst_end_time = current_time + timedelta(seconds=np.random.randint(30, 120))
            
            # Check if burst phase should end
            if in_burst_phase and current_time >= burst_end_time:
                in_burst_phase = False
            
            # Decide if we should start/end a session
            if not in_session and np.random.random() < 0.12:  # 12% chance to start session
                in_session = True
                session_events = 0
                current_time, login_event = self._create_login_event(user_id, current_time, profile, anomaly_type)
                login_event['is_anomaly'] = 0
                login_event['anomaly_type'] = None
                events.append(login_event)
            
            elif in_session:
                # Generate events within session
                session_events += 1
                
                # End session occasionally (or more frequently during behavior shift)
                session_end_prob = 0.18
                if profile.behavior_shifted and anomaly_type == 10:
                    session_end_prob = 0.25  # More frequent sessions during behavior shift
                
                if session_events > np.random.randint(5, 25) or np.random.random() < session_end_prob:
                    in_session = False
                    current_time, logout_event = self._create_logout_event(user_id, current_time, profile)
                    logout_event['is_anomaly'] = 0
                    logout_event['anomaly_type'] = None
                    events.append(logout_event)
                    continue
                
                # Determine if this event should be anomalous
                is_anomalous = (has_anomaly and anomaly_injected < anomaly_event_budget and 
                               np.random.random() < (anomaly_event_budget / (num_events - event_idx)))
                
                # Handle deposit-no-trade-withdrawal sequence (type 3)
                if is_anomalous and anomaly_type == 3 and awaiting_withdrawal_for is None and np.random.random() < 0.3:
                    # Start sequence with deposit (MARK AS ANOMALOUS)
                    current_time, event = self._create_session_event(
                        user_id=user_id,
                        current_time=current_time,
                        profile=profile,
                        is_anomalous=True,  # FIXED: Mark deposit as anomalous
                        anomaly_type=anomaly_type,
                        force_event_type='deposit'
                    )
                    if event:
                        event['is_anomaly'] = 1  # FIXED: Mark as anomalous
                        event['anomaly_type'] = 3  # Type 3
                        event['anomaly_severity'] = 0.3  # Mild severity for deposit
                        events.append(event)
                        awaiting_withdrawal_for = current_time
                        anomaly_injected += 1
                    continue
                
                # Enforce no trades in deposit-no-trade-withdrawal sequence
                if awaiting_withdrawal_for is not None:
                    # Generate withdrawal (MARK AS ANOMALOUS)
                    if np.random.random() < 0.3:  # 30% chance to complete withdrawal on each event
                        current_time, event = self._create_session_event(
                            user_id=user_id,
                            current_time=current_time,
                            profile=profile,
                            is_anomalous=True,  # Withdrawal is anomalous in this sequence
                            anomaly_type=anomaly_type,
                            force_event_type='withdrawal'
                        )
                        if event:
                            event['is_anomaly'] = 1  # FIXED: Mark as anomalous
                            event['anomaly_type'] = 3  # Type 3 (sequence anomaly)
                            event['anomaly_severity'] = 0.6  # Moderate-high severity for withdrawal in sequence
                            events.append(event)
                            awaiting_withdrawal_for = None
                        continue
                    else:
                        # Skip this event slot (don't generate trades)
                        gap = np.random.exponential(scale=30)
                        current_time += timedelta(seconds=gap)
                        continue
                
                # Generate normal event
                current_time, event = self._create_session_event(
                    user_id=user_id,
                    current_time=current_time,
                    profile=profile,
                    is_anomalous=is_anomalous,
                    anomaly_type=anomaly_type,
                    in_burst_phase=in_burst_phase,
                    behavior_shifted=profile.behavior_shifted
                )
                
                if event:
                    # Assign default severity for anomalies without specific magnitude calculations
                    if is_anomalous and event['anomaly_severity'] == 0.0:
                        if anomaly_type == 1:  # Multi-IP login
                            event['anomaly_severity'] = 0.4
                        elif anomaly_type == 2:  # Impossible travel
                            event['anomaly_severity'] = 0.7
                        elif anomaly_type == 5:  # Dormant → burst
                            event['anomaly_severity'] = 0.5
                        elif anomaly_type == 7:  # Instrument concentration
                            event['anomaly_severity'] = 0.3
                        elif anomaly_type == 9:  # Multi-user same IP
                            event['anomaly_severity'] = 0.5
                        else:
                            event['anomaly_severity'] = 0.5  # Default moderate
                    
                    events.append(event)
                    # Only update statistics for normal events (prevent anomaly cascading)
                    if not is_anomalous:
                        profile.update_statistics(event['event_type'], event['amount'], event['trade_volume'])
                    
                    if is_anomalous:
                        anomaly_injected += 1
            
            else:
                # No current session, advance time
                hours = np.random.exponential(scale=24)  # ~1 day average between sessions
                current_time += timedelta(hours=hours)
        
        # Close any open session
        if in_session:
            _, logout_event = self._create_logout_event(user_id, current_time, profile)
            logout_event['is_anomaly'] = 0
            logout_event['anomaly_type'] = None
            events.append(logout_event)
        
        return events
    
    def _create_login_event(self, user_id: int, current_time: datetime, 
                           profile: UserProfile, anomaly_type: Optional[int]) -> Tuple[datetime, Dict]:
        """Create a login event and advance time."""
        profile.last_event_time = current_time
        
        # Occasionally use different IP for anomaly
        if anomaly_type == 1 and np.random.random() < 0.08:  # Multi-IP login anomaly
            ip_address = self._generate_random_ip()
        else:
            ip_address = profile.primary_ip if np.random.random() < 0.96 else self._generate_random_ip()
        
        profile.last_ip = ip_address
        
        # Device consistency
        if np.random.random() < 0.92:
            device_id = profile.primary_device
        else:
            device_id = self._generate_device()
        
        profile.last_device = device_id
        
        event = {
            'user_id': user_id,
            'timestamp': current_time,
            'event_type': 'login',
            'ip_address': ip_address,
            'device_id': device_id,
            'amount': 0.0,
            'trade_volume': 0.0,
            'instrument': '',
            'session_id': profile.session_id,
            'is_anomaly': 0,
            'anomaly_type': None,
            'anomaly_severity': 0.0,
        }
        
        # Minimal advance time for login
        next_time = current_time + timedelta(seconds=np.random.randint(2, 8))
        
        return next_time, event
    
    def _create_logout_event(self, user_id: int, current_time: datetime,
                            profile: UserProfile) -> Tuple[datetime, Dict]:
        """Create a logout event and advance time for next session."""
        event = {
            'user_id': user_id,
            'timestamp': current_time,
            'event_type': 'logout',
            'ip_address': profile.last_ip,
            'device_id': profile.last_device,
            'amount': 0.0,
            'trade_volume': 0.0,
            'instrument': '',
            'session_id': profile.session_id,
            'is_anomaly': 0,
            'anomaly_type': None,
            'anomaly_severity': 0.0,
        }
        
        # Advance time to next possible session (inter-session gap)
        gap_hours = np.random.exponential(scale=8)  # ~8 hours average between sessions
        next_time = current_time + timedelta(hours=gap_hours)
        
        return next_time, event
    
    def _create_session_event(self, user_id: int, current_time: datetime,
                             profile: UserProfile, is_anomalous: bool, 
                             anomaly_type: Optional[int],
                             in_burst_phase: bool = False,
                             behavior_shifted: bool = False,
                             force_event_type: Optional[str] = None) -> Tuple[datetime, Optional[Dict]]:
        """Create an event within a session with ground truth labels and severity score."""
        # Determine event type
        if force_event_type:
            event_type = force_event_type
        else:
            # Modify distribution during behavior shift (more frequent sessions)
            if behavior_shifted and anomaly_type == 10:
                event_type_weights = [0.01, 0.01, 0.15, 0.08, 0.75]  # Slightly more trades
            else:
                event_type_weights = [0.02, 0.02, 0.15, 0.08, 0.73]
            
            event_type = np.random.choice(['login', 'logout', 'deposit', 'withdrawal', 'trade'], 
                                         p=event_type_weights)
        
        # Skip login/logout (handled separately)
        if event_type in ['login', 'logout']:
            gap = np.random.exponential(scale=180)  # ~3 min average
            next_time = current_time + timedelta(seconds=gap)
            return next_time, None
        
        # Advance time (reduced in burst phase for dense clustering)
        if in_burst_phase:
            gap = np.random.uniform(1, 8)  # 1-8 seconds for dense bursts
        else:
            gap = np.random.exponential(scale=90)  # ~1.5 min average between events
        
        next_time = current_time + timedelta(seconds=gap)
        
        # Generate amount, volume, and severity score
        amount, trade_volume, severity = self._generate_transaction_values(
            event_type=event_type,
            profile=profile,
            is_anomalous=is_anomalous,
            anomaly_type=anomaly_type,
            user_id=user_id,
            behavior_shifted=behavior_shifted
        )
        
        # Determine IP (mostly consistent in session)
        if is_anomalous and anomaly_type == 2:  # Impossible travel
            ip_address = self._generate_random_ip()
        else:
            ip_address = profile.last_ip if np.random.random() < 0.99 else self._generate_random_ip()
        
        # Determine device (mostly consistent)
        device_id = profile.last_device if np.random.random() < 0.97 else self._generate_device()
        
        # Instrument selection (behavior shift can force post-shift instruments)
        if behavior_shifted and anomaly_type == 10 and profile.post_shift_instruments is not None:
            instrument = np.random.choice(profile.post_shift_instruments)
        elif is_anomalous and anomaly_type == 7:  # Single instrument concentration
            instrument = self.INSTRUMENTS[user_id % len(self.INSTRUMENTS)]
        else:
            instrument = np.random.choice(self.INSTRUMENTS)
        
        event = {
            'user_id': user_id,
            'timestamp': next_time,
            'event_type': event_type,
            'ip_address': ip_address,
            'device_id': device_id,
            'amount': amount,
            'trade_volume': trade_volume,
            'instrument': instrument,
            'session_id': profile.session_id,
            'is_anomaly': 1 if is_anomalous else 0,
            'anomaly_type': anomaly_type if is_anomalous else None,
            'anomaly_severity': severity if is_anomalous else 0.0,  # NEW: severity score
        }
        
        return next_time, event
    
    def _generate_transaction_values(self, event_type: str, profile: UserProfile,
                                    is_anomalous: bool, anomaly_type: Optional[int],
                                    user_id: int, behavior_shifted: bool = False) -> Tuple[float, float, float]:
        """Generate transaction values with anomalies relative to user baseline."""
        amount = 0.0
        trade_volume = 0.0
        severity = 0.0
        
        if event_type == 'deposit':
            base = profile.avg_deposit
            # Anomaly type 3: Deposit with no trade... handled at sequence level
            amount = base * np.random.lognormal(mean=0, sigma=0.5)
        
        elif event_type == 'withdrawal':
            base = profile.avg_withdrawal
            if is_anomalous and anomaly_type == 4:  # Sudden large withdrawal
                # Spike: 4-12x user's baseline
                multiplier = np.random.uniform(4, 12)
                amount = base * multiplier
                # Severity: normalize by max multiplier
                severity = min(multiplier / 12.0, 1.0)
            else:
                amount = base * np.random.lognormal(mean=0, sigma=0.5)
        
        elif event_type == 'trade':
            base = profile.avg_trade_volume
            if is_anomalous and anomaly_type == 6:  # Trade volume spike
                # Spike: 6-25x user's baseline
                multiplier = np.random.uniform(6, 25)
                trade_volume = base * multiplier
                # Severity: normalize by max multiplier
                severity = min(multiplier / 25.0, 1.0)
            elif is_anomalous and anomaly_type == 8:  # High-frequency bot
                # Many small trades
                trade_volume = base * 0.05 * np.random.uniform(1, 3)
                severity = 0.6  # Moderate-high severity for bot behavior
            else:
                trade_volume = base * np.random.lognormal(mean=0, sigma=0.4)
            
            # Calculate transaction amount from volume (~$1000-3000 per lot)
            amount = trade_volume * np.random.uniform(1000, 3000)
        
        # Apply behavior shift multiplier if active (anomaly type 10)
        if behavior_shifted and anomaly_type == 10:
            if event_type == 'trade':
                trade_volume *= profile.post_shift_volume_multiplier
                amount = trade_volume * np.random.uniform(1000, 3000)
                severity = min(profile.post_shift_volume_multiplier / 8.0, 1.0)
        
        return round(amount, 2), round(trade_volume, 2), round(severity, 2)
    
    def _generate_random_ip(self) -> str:
        """Generate a random IP address."""
        return f"192.168.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}"
    
    def _generate_device(self) -> str:
        """Generate a random device ID."""
        return f"device_{np.random.randint(1, 301)}"
    
    def _enforce_type9_overlapping_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce overlapping sessions for Type 9 (multi-user same IP) anomalies."""
        # Find all Type 9 anomalies
        type9_events = df[df['anomaly_type'] == 9]
        if len(type9_events) == 0:
            return df
        
        # Group by IP address
        type9_users = type9_events['user_id'].unique()
        
        for ip_group in self.shared_ip_groups.values():
            if len(ip_group) < 2:
                continue
            
            # Check if these users have Type 9 anomalies
            type9_in_group = [u for u in ip_group if u in type9_users]
            if len(type9_in_group) < 2:
                continue
            
            # Get all logins for these users
            for user_id in type9_in_group:
                user_logins = df[(df['user_id'] == user_id) & (df['event_type'] == 'login')]
                
                if len(user_logins) > 0:
                    # For each login, find nearby logins from other Type 9 users
                    for idx, login in user_logins.iterrows():
                        login_time = pd.to_datetime(login['timestamp'])
                        
                        # Check if any other Type 9 user in same IP group has login within ±5 minutes
                        for other_user in type9_in_group:
                            if other_user == user_id:
                                continue
                            
                            other_logins = df[(df['user_id'] == other_user) & (df['event_type'] == 'login')]
                            mask = other_logins['timestamp'].apply(
                                lambda x: abs((pd.to_datetime(x) - login_time).total_seconds()) < 300
                            )
                            
                            if mask.any():
                                # Force both users to use same IP
                                df.loc[(df['user_id'] == user_id), 'ip_address'] = login['ip_address']
                                df.loc[(df['user_id'] == other_user), 'ip_address'] = login['ip_address']
                                
                                # Increase severity for Type 9 due to overlap
                                user_type9 = df[(df['user_id'] == user_id) & (df['anomaly_type'] == 9)]
                                df.loc[user_type9.index, 'anomaly_severity'] = 0.8
        
        return df
    
    def _add_user_level_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add user-level anomaly labels for each user."""
        # Group by user_id and check if any event has is_anomaly=1
        user_anomalies = df.groupby('user_id')['is_anomaly'].max().astype(bool)
        
        # Map back to full DataFrame
        df['user_is_anomalous'] = df['user_id'].map(user_anomalies).astype(bool)
        
        logger.info(f"Added user-level labels: {user_anomalies.sum()} users marked as anomalous")
        return df
    
    def _ensure_label_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistency in anomaly labeling across all fields."""
        initial_count = len(df)
        
        # Anomalies must have a type
        anomaly_mask = df['is_anomaly'] == 1
        missing_type = anomaly_mask & df['anomaly_type'].isna()
        if missing_type.any():
            logger.warning(f"Found {missing_type.sum()} anomalous events without type, assigning default")
            df.loc[missing_type, 'anomaly_type'] = 5  # Assign type 5 (dormant → burst)
        
        # Non-anomalies must have severity 0
        normal_mask = df['is_anomaly'] == 0
        df.loc[normal_mask, 'anomaly_severity'] = 0.0
        df.loc[normal_mask, 'anomaly_type'] = np.nan
        
        # Validate anomaly_type values
        invalid_type = anomaly_mask & (
            (df['anomaly_type'] < 1) | (df['anomaly_type'] > 10) | df['anomaly_type'].isna()
        )
        if invalid_type.any():
            logger.warning(f"Found {invalid_type.sum()} anomalies with invalid type, correcting")
            df.loc[invalid_type, 'anomaly_type'] = 5
        
        # Ensure severity is between 0 and 1
        df['anomaly_severity'] = df['anomaly_severity'].clip(0.0, 1.0)
        
        logger.info(f"Label consistency check complete: {initial_count} rows processed")
        return df
    
    def save_to_csv(self, df: pd.DataFrame, filepath: Optional[str] = None) -> str:
        """Save generated dataset to CSV file."""
        if filepath is None:
            filepath = os.path.join(os.path.dirname(__file__), 'synthetic_events.csv')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Ensure timestamp is string format for CSV
        df_copy = df.copy()
        df_copy['timestamp'] = df_copy['timestamp'].astype(str)
        
        df_copy.to_csv(filepath, index=False)
        logger.info(f"Dataset saved to {filepath} ({len(df)} events)")
        
        return filepath
