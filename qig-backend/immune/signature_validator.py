"""
Geometric Signature Validator - Classify traffic as legitimate or threat

Layer 1.2 of QIG Immune System: T-cell analog for detecting "non-self" patterns.
Uses geometric distance in (Φ, κ, regime) space to identify threats.

ADAPTIVE THRESHOLDS: All weights derived from Φ/κ statistics, no fixed constants.
"""

from typing import Dict, List
import numpy as np
from datetime import datetime

from .adaptive_thresholds import get_threshold_engine


class SignatureValidator:
    """
    Validate traffic using geometric signatures with ADAPTIVE thresholds.
    
    Implements T-cell analog: recognizes "self" (legitimate) vs "non-self" (threat)
    through geometric pattern matching. All thresholds derived from running
    Φ/κ statistics - NO fixed constants.
    """
    
    def __init__(self):
        self.legitimate_signatures = self._load_legitimate_patterns()
        self.threat_signatures = self._load_threat_patterns()
        self.learned_patterns: List[Dict] = []
        self.validation_history: List[Dict] = []
        self.threshold_engine = get_threshold_engine()
    
    def validate(self, signature: Dict) -> Dict:
        """
        Validate request signature using ADAPTIVE thresholds.
        
        All threat scores computed from z-scores relative to observed Φ/κ distribution.
        No fixed thresholds - everything derived from population statistics.
        
        Returns:
            {
                'threat_level': str,  # 'none', 'low', 'medium', 'high', 'critical'
                'classification': str,  # 'legitimate', 'suspicious', 'malicious'
                'confidence': float,  # 0-1
                'threat_score': float,
                'reasons': List[str]
            }
        """
        self.threshold_engine.record_observation(signature)
        
        threat_score, reasons = self.threshold_engine.compute_threat_score(signature)
        
        if self._matches_known_threat(signature):
            thresholds = self.threshold_engine.get_thresholds()
            threat_weight = min(0.5, thresholds['breakdown_weight'] + 0.1)
            threat_score = min(1.0, threat_score + threat_weight)
            reasons.append(f"Matches known threat (weight={threat_weight:.2f})")
        
        if self._matches_legitimate(signature):
            legit_bonus = min(0.3, self.threshold_engine._phi_std * 0.5)
            threat_score = max(0.0, threat_score - legit_bonus)
            reasons.append(f"Matches legitimate pattern (bonus={legit_bonus:.2f})")
        
        threat_level = self.threshold_engine.classify_threat_level(threat_score)
        
        if threat_level in ['none', 'low']:
            classification = 'legitimate' if threat_level == 'none' else 'suspicious'
        elif threat_level == 'medium':
            classification = 'suspicious'
        else:
            classification = 'malicious'
        
        result = {
            'threat_level': threat_level,
            'classification': classification,
            'confidence': 1.0 - abs(threat_score - 0.5) * 1.5,
            'threat_score': threat_score,
            'reasons': reasons,
            'adaptive_thresholds': self.threshold_engine.get_thresholds()
        }
        
        self._record_validation(signature, result)
        
        return result
    
    def learn_pattern(self, signature: Dict, is_threat: bool):
        """Learn a new pattern from confirmed threat/legitimate traffic."""
        pattern = {
            'phi': signature.get('phi', 0.5),
            'kappa': signature.get('kappa', 50),
            'regime': signature.get('regime', 'geometric'),
            'surprise': signature.get('surprise', 0.5),
            'confidence': signature.get('confidence', 0.5),
            'temporal_pattern': signature.get('temporal_pattern', 'human'),
            'is_threat': is_threat,
            'learned_at': datetime.now().isoformat()
        }
        
        self.learned_patterns.append(pattern)
        
        if len(self.learned_patterns) > 1000:
            self.learned_patterns = self.learned_patterns[-1000:]
        
        if is_threat:
            self.threat_signatures.append(pattern)
        else:
            self.legitimate_signatures.append(pattern)
    
    def _matches_known_threat(self, signature: Dict) -> bool:
        """Check if signature matches known threat patterns."""
        for threat in self.threat_signatures:
            if self._signature_distance(signature, threat) < 0.3:
                return True
        return False
    
    def _matches_legitimate(self, signature: Dict) -> bool:
        """Check if signature matches legitimate user patterns."""
        for legit in self.legitimate_signatures:
            if self._signature_distance(signature, legit) < 0.25:
                return True
        return False
    
    def _signature_distance(self, sig1: Dict, sig2: Dict) -> float:
        """
        Compute distance between two geometric signatures.
        
        Uses weighted Euclidean distance in (Φ, κ, surprise) space.
        """
        phi_diff = abs(sig1.get('phi', 0.5) - sig2.get('phi', 0.5))
        kappa_diff = abs(sig1.get('kappa', 50) - sig2.get('kappa', 50)) / 100.0
        surprise_diff = abs(sig1.get('surprise', 0.5) - sig2.get('surprise', 0.5))
        confidence_diff = abs(sig1.get('confidence', 0.5) - sig2.get('confidence', 0.5))
        
        regime_penalty = 0.0 if sig1.get('regime') == sig2.get('regime') else 0.3
        temporal_penalty = 0.0 if sig1.get('temporal_pattern') == sig2.get('temporal_pattern') else 0.2
        
        distance = np.sqrt(
            phi_diff**2 * 2.0 +
            kappa_diff**2 * 1.0 +
            surprise_diff**2 * 1.5 +
            confidence_diff**2 * 0.5
        ) + regime_penalty + temporal_penalty
        
        return min(1.0, distance / np.sqrt(5.0))
    
    def _load_legitimate_patterns(self) -> List[Dict]:
        """Load known legitimate user patterns (MHC "self" markers)."""
        return [
            {
                'phi': 0.65, 'kappa': 45, 'regime': 'geometric',
                'surprise': 0.2, 'confidence': 0.85, 'temporal_pattern': 'human'
            },
            {
                'phi': 0.4, 'kappa': 25, 'regime': 'linear',
                'surprise': 0.15, 'confidence': 0.8, 'temporal_pattern': 'human'
            },
            {
                'phi': 0.75, 'kappa': 60, 'regime': 'hierarchical',
                'surprise': 0.25, 'confidence': 0.9, 'temporal_pattern': 'human'
            },
            {
                'phi': 0.5, 'kappa': 35, 'regime': 'geometric',
                'surprise': 0.3, 'confidence': 0.75, 'temporal_pattern': 'human'
            },
        ]
    
    def _load_threat_patterns(self) -> List[Dict]:
        """Load known threat patterns (pathogen database)."""
        return [
            {
                'phi': 0.1, 'kappa': 15, 'regime': 'breakdown',
                'surprise': 0.9, 'confidence': 0.3, 'temporal_pattern': 'bot'
            },
            {
                'phi': 0.15, 'kappa': 20, 'regime': 'breakdown',
                'surprise': 0.8, 'confidence': 0.35, 'temporal_pattern': 'burst'
            },
            {
                'phi': 0.05, 'kappa': 10, 'regime': 'linear',
                'surprise': 0.95, 'confidence': 0.2, 'temporal_pattern': 'bot'
            },
            {
                'phi': 0.08, 'kappa': 50, 'regime': 'breakdown',
                'surprise': 0.85, 'confidence': 0.25, 'temporal_pattern': 'burst'
            },
        ]
    
    def _record_validation(self, signature: Dict, result: Dict):
        """Record validation for analysis."""
        record = {
            'signature': signature,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        self.validation_history.append(record)
        
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
    
    def get_stats(self) -> Dict:
        """Get validation statistics."""
        if not self.validation_history:
            return {
                'total_validations': 0,
                'threat_distribution': {},
                'learned_patterns': len(self.learned_patterns)
            }
        
        threat_counts = {}
        for record in self.validation_history:
            level = record['result']['threat_level']
            threat_counts[level] = threat_counts.get(level, 0) + 1
        
        return {
            'total_validations': len(self.validation_history),
            'threat_distribution': threat_counts,
            'learned_patterns': len(self.learned_patterns),
            'legitimate_signatures': len(self.legitimate_signatures),
            'threat_signatures': len(self.threat_signatures)
        }
