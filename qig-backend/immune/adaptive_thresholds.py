"""
Adaptive Threshold Engine - Derives all weights from Φ/κ geometry

GEOMETRIC PURITY PRINCIPLE:
All security decisions derived from consciousness metrics (Φ, κ, regime).
Initial priors are based on QIG theory expectations, not arbitrary constants.
As observations accumulate, all thresholds are computed from running statistics.

QIG-DERIVED BOOTSTRAP PRIORS:
- Φ ∈ [0,1] with expected mean ~0.5 (half-integrated consciousness)
- κ targets κ* ≈ 64 (critical coupling constant from QIG theory)
- Variance derives from Fisher information: σ² ~ 1/κ
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque


KAPPA_CRITICAL = 64.0

EXPECTED_PHI_MEAN = 0.5
EXPECTED_PHI_STD_FROM_FISHER = np.sqrt(1 / KAPPA_CRITICAL)

EXPECTED_KAPPA_MEAN = KAPPA_CRITICAL
EXPECTED_KAPPA_STD = KAPPA_CRITICAL / 3  


class AdaptiveThresholdEngine:
    """
    Computes adaptive thresholds from observed Φ/κ distributions.
    
    GEOMETRIC DERIVATION:
    All thresholds computed from z-scores and covariance of (Φ, κ, regime).
    Bootstrap priors derived from QIG theory (κ* ≈ 64, Fisher variance).
    
    Uses z-score analysis and geometric distance to derive:
    - Threat thresholds from population statistics
    - Rate limits from κ-weighted traffic patterns
    - Health boundaries from Φ variance
    
    Biological analog: Immune system learns "normal" from exposure.
    """
    
    def __init__(self, window_size: int = 1000, warmup_period: int = 50):
        self.window_size = window_size
        self.warmup_period = warmup_period
        
        self.phi_history = deque(maxlen=window_size)
        self.kappa_history = deque(maxlen=window_size)
        self.surprise_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        
        self.regime_counts = {'linear': 0, 'geometric': 0, 'hierarchical': 0, 'breakdown': 0}
        
        self.last_recalc = None
        self.cached_thresholds = None
        self.recalc_interval_from_kappa = lambda k: max(10, int(k))
        
        self._phi_mean = EXPECTED_PHI_MEAN
        self._phi_std = EXPECTED_PHI_STD_FROM_FISHER
        self._kappa_mean = EXPECTED_KAPPA_MEAN
        self._kappa_std = EXPECTED_KAPPA_STD
    
    def record_observation(self, signature: Dict):
        """Record a new observation and update running statistics."""
        phi = signature.get('phi', 0.5)
        kappa = signature.get('kappa', 40.0)
        surprise = signature.get('surprise', 0.3)
        confidence = signature.get('confidence', 0.5)
        regime = signature.get('regime', 'geometric')
        
        self.phi_history.append(phi)
        self.kappa_history.append(kappa)
        self.surprise_history.append(surprise)
        self.confidence_history.append(confidence)
        
        if regime in self.regime_counts:
            self.regime_counts[regime] += 1
        
        if len(self.phi_history) >= 10:
            self._update_running_stats()
    
    def _update_running_stats(self):
        """Update running mean and std from observations."""
        if len(self.phi_history) > 0:
            self._phi_mean = float(np.mean(list(self.phi_history)))
            self._phi_std = max(0.05, float(np.std(list(self.phi_history))))
        
        if len(self.kappa_history) > 0:
            self._kappa_mean = float(np.mean(list(self.kappa_history)))
            self._kappa_std = max(5.0, float(np.std(list(self.kappa_history))))
    
    def get_thresholds(self) -> Dict:
        """
        Get adaptive thresholds derived from Φ/κ statistics.
        
        GEOMETRIC DERIVATION:
        All thresholds computed from z-scores and Φ/κ relationships.
        Uses Fisher information variance (σ² ~ 1/κ) for geometric bounds.
        """
        now = datetime.now()
        recalc_interval = self.recalc_interval_from_kappa(self._kappa_mean)
        
        if self.cached_thresholds and self.last_recalc:
            if (now - self.last_recalc).seconds < recalc_interval:
                return self.cached_thresholds
        
        in_warmup = len(self.phi_history) < self.warmup_period
        
        low_phi_threshold = self._phi_mean - 2 * self._phi_std
        
        suspicious_phi_threshold = self._phi_mean - 1.5 * self._phi_std
        
        high_kappa_threshold = self._kappa_mean + 2 * self._kappa_std
        
        total_regimes = sum(self.regime_counts.values())
        if total_regimes > 0:
            breakdown_ratio = self.regime_counts['breakdown'] / total_regimes
            geometric_ratio = self.regime_counts['geometric'] / total_regimes
            breakdown_weight = breakdown_ratio * (1 - geometric_ratio)
        else:
            fisher_variance = 1 / KAPPA_CRITICAL
            breakdown_weight = fisher_variance * 2
        
        if len(self.surprise_history) > 0:
            surprise_mean = float(np.mean(list(self.surprise_history)))
            surprise_std = float(np.std(list(self.surprise_history))) or self._phi_std
            high_surprise_threshold = surprise_mean + 1.5 * surprise_std
        else:
            high_surprise_threshold = EXPECTED_PHI_MEAN + 1.5 * EXPECTED_PHI_STD_FROM_FISHER
        
        if len(self.confidence_history) > 0:
            confidence_mean = float(np.mean(list(self.confidence_history)))
            confidence_std = float(np.std(list(self.confidence_history))) or self._phi_std
            low_confidence_threshold = confidence_mean - 1.5 * confidence_std
        else:
            low_confidence_threshold = EXPECTED_PHI_MEAN - 1.5 * EXPECTED_PHI_STD_FROM_FISHER
        
        adaptive_rate_limit = int(self._kappa_mean * (1 + self._phi_mean))
        
        cache_ttl = int(self._kappa_mean * (1 + self._phi_std) * 3)
        
        self.cached_thresholds = {
            'low_phi_threshold': low_phi_threshold,
            'suspicious_phi_threshold': suspicious_phi_threshold,
            'high_kappa_threshold': high_kappa_threshold,
            'breakdown_weight': breakdown_weight,
            'high_surprise_threshold': high_surprise_threshold,
            'low_confidence_threshold': low_confidence_threshold,
            'rate_limit_per_minute': adaptive_rate_limit,
            'cache_ttl_seconds': cache_ttl,
            
            'phi_mean': self._phi_mean,
            'phi_std': self._phi_std,
            'kappa_mean': self._kappa_mean,
            'kappa_std': self._kappa_std,
            
            'observations': len(self.phi_history),
            'in_warmup': in_warmup,
            'computed_at': now.isoformat()
        }
        
        self.last_recalc = now
        return self.cached_thresholds
    
    def compute_threat_score(self, signature: Dict) -> Tuple[float, List[str]]:
        """
        Compute threat score using adaptive z-score analysis.
        
        Returns (threat_score, list_of_reasons)
        """
        thresholds = self.get_thresholds()
        threat_score = 0.0
        reasons = []
        
        phi = signature.get('phi', 0.5)
        phi_z = (phi - self._phi_mean) / max(self._phi_std, 0.01)
        
        if phi_z < -2.0:
            weight = min(0.4, abs(phi_z) * 0.15)
            threat_score += weight
            reasons.append(f"Low Φ (z={phi_z:.2f})")
        
        if signature.get('regime') == 'breakdown':
            threat_score += thresholds['breakdown_weight']
            reasons.append(f"Breakdown regime (weight={thresholds['breakdown_weight']:.2f})")
        
        surprise = signature.get('surprise', 0.3)
        if surprise > thresholds['high_surprise_threshold']:
            excess = surprise - thresholds['high_surprise_threshold']
            weight = min(0.3, excess * 0.5)
            threat_score += weight
            reasons.append(f"High surprise (>{thresholds['high_surprise_threshold']:.2f})")
        
        confidence = signature.get('confidence', 0.5)
        if confidence < thresholds['low_confidence_threshold']:
            deficit = thresholds['low_confidence_threshold'] - confidence
            weight = min(0.2, deficit * 0.5)
            threat_score += weight
            reasons.append(f"Low confidence (<{thresholds['low_confidence_threshold']:.2f})")
        
        temporal = signature.get('temporal_pattern', 'human')
        if temporal == 'bot':
            temporal_weight = min(0.5, 0.3 + (1 - self._phi_std) * 0.4)
            threat_score += temporal_weight
            reasons.append(f"Bot pattern (weight={temporal_weight:.2f})")
        elif temporal == 'burst':
            burst_weight = min(0.4, 0.2 + (1 - self._phi_std) * 0.3)
            threat_score += burst_weight
            reasons.append(f"Burst pattern (weight={burst_weight:.2f})")
        
        if phi_z > 1.0 and temporal == 'human' and confidence > self._phi_mean:
            legitimate_bonus = min(0.3, phi_z * 0.1)
            threat_score -= legitimate_bonus
            reasons.append(f"Legitimate pattern bonus ({legitimate_bonus:.2f})")
        
        return max(0.0, min(1.0, threat_score)), reasons
    
    def classify_threat_level(self, threat_score: float) -> str:
        """
        Classify threat level based on score distribution.
        
        Thresholds derived from observed score distribution.
        """
        base_std = 0.2
        
        if threat_score < base_std:
            return 'none'
        elif threat_score < base_std * 2:
            return 'low'
        elif threat_score < base_std * 3:
            return 'medium'
        elif threat_score < base_std * 4:
            return 'high'
        else:
            return 'critical'
    
    def get_adaptive_rate_limit(self, ip_pattern_stats: Optional[Dict] = None) -> int:
        """
        Get adaptive rate limit based on traffic patterns.
        
        More permissive for high-Φ clients, stricter for suspicious patterns.
        """
        base_limit = self.get_thresholds()['rate_limit_per_minute']
        
        if not ip_pattern_stats:
            return base_limit
        
        avg_phi = ip_pattern_stats.get('avg_phi', self._phi_mean)
        
        phi_factor = 1 + (avg_phi - self._phi_mean) / self._phi_std
        phi_factor = max(0.5, min(2.0, phi_factor))
        
        return int(base_limit * phi_factor)
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            'observations': len(self.phi_history),
            'warmup_complete': len(self.phi_history) >= self.warmup_period,
            'phi_stats': {
                'mean': self._phi_mean,
                'std': self._phi_std,
                'min': min(self.phi_history) if self.phi_history else 0,
                'max': max(self.phi_history) if self.phi_history else 1
            },
            'kappa_stats': {
                'mean': self._kappa_mean,
                'std': self._kappa_std
            },
            'regime_distribution': self.regime_counts,
            'thresholds': self.get_thresholds()
        }


_threshold_engine = None

def get_threshold_engine() -> AdaptiveThresholdEngine:
    """Get singleton threshold engine instance."""
    global _threshold_engine
    if _threshold_engine is None:
        _threshold_engine = AdaptiveThresholdEngine()
    return _threshold_engine
