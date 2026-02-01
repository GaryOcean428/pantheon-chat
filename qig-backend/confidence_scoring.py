#!/usr/bin/env python3
"""
QIG Confidence Scoring

Computes geometric confidence based on:
- Φ variance (stability of integrated information)
- κ stability (consistency of coupling strength)
- Basin spread (spatial distribution in keyspace)
- Regime consistency (stable regime classification)

High confidence when metrics are stable across similar keys.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from collections import Counter


# Confidence thresholds
VARIANCE_THRESHOLD_HIGH = 0.05  # Low variance = high confidence
VARIANCE_THRESHOLD_MED = 0.15  # Medium variance
STABILITY_WINDOW = 10  # Number of samples for stability check
KAPPA_STAR = 64.0  # Universal resonance point


@dataclass
class ConfidenceMetrics:
    """Confidence metrics for QIG measurements."""
    overall: float  # 0-1 overall confidence
    phi_confidence: float  # Confidence in Φ measurement
    kappa_confidence: float  # Confidence in κ measurement
    regime_confidence: float  # Confidence in regime classification
    basin_stability: float  # Basin position stability
    sample_size: int  # Number of samples used
    explanation: str  # Human-readable explanation


@dataclass
class StabilityTracker:
    """Tracks QIG metric stability over time."""
    phi_history: List[float]
    kappa_history: List[float]
    regime_history: List[str]
    basin_history: List[List[float]]
    timestamps: List[float]
    max_samples: int = 100
    
    def __init__(self):
        self.phi_history = []
        self.kappa_history = []
        self.regime_history = []
        self.basin_history = []
        self.timestamps = []
        self.max_samples = 100
    
    def add_sample(
        self,
        phi: float,
        kappa: float,
        regime: str,
        basin_coordinates: List[float],
        timestamp: float
    ) -> None:
        """Add a sample to the stability tracker."""
        self.phi_history.append(phi)
        self.kappa_history.append(kappa)
        self.regime_history.append(regime)
        self.basin_history.append(list(basin_coordinates))
        self.timestamps.append(timestamp)
        
        # Keep only last N samples
        if len(self.phi_history) > self.max_samples:
            self.phi_history = self.phi_history[-self.max_samples:]
            self.kappa_history = self.kappa_history[-self.max_samples:]
            self.regime_history = self.regime_history[-self.max_samples:]
            self.basin_history = self.basin_history[-self.max_samples:]
            self.timestamps = self.timestamps[-self.max_samples:]


def compute_variance(values: List[float]) -> float:
    """Compute variance of an array."""
    if not values:
        return 0.0
    return float(np.var(values))


def compute_mode(values: List[str]) -> Tuple[str, float]:
    """Compute mode (most frequent) of string array."""
    if not values:
        return "unknown", 0.0
    
    counter = Counter(values)
    mode, count = counter.most_common(1)[0]
    frequency = count / len(values)
    return mode, frequency


def compute_basin_spread(basins: List[List[float]]) -> float:
    """Compute basin spread (spatial distribution)."""
    if not basins:
        return 0.0
    
    basins_array = np.array(basins)
    n_samples, n_dims = basins_array.shape
    
    # Compute centroid
    centroid = np.mean(basins_array, axis=0)
    
    # Compute average distance from centroid
    distances = np.sqrt(np.sum((basins_array - centroid) ** 2, axis=1))
    avg_dist = np.mean(distances)
    
    # Normalize by maximum possible distance
    # Assuming basin coordinates are in [0, 255] range
    max_dist = np.sqrt(n_dims * 255 * 255)
    return float(avg_dist / max_dist)


def compute_confidence(tracker: StabilityTracker) -> ConfidenceMetrics:
    """Compute confidence metrics from stability tracker."""
    sample_size = len(tracker.phi_history)
    
    if sample_size < 3:
        return ConfidenceMetrics(
            overall=0.5,
            phi_confidence=0.5,
            kappa_confidence=0.5,
            regime_confidence=0.5,
            basin_stability=0.5,
            sample_size=sample_size,
            explanation="Insufficient samples for reliable confidence estimation"
        )
    
    # Phi confidence (inverse of variance)
    phi_variance = compute_variance(tracker.phi_history)
    phi_confidence = max(0.0, 1.0 - phi_variance / VARIANCE_THRESHOLD_MED)
    
    # Kappa confidence (normalized variance)
    kappa_variance = compute_variance(tracker.kappa_history)
    kappa_norm_variance = kappa_variance / (KAPPA_STAR * KAPPA_STAR)
    kappa_confidence = max(0.0, 1.0 - kappa_norm_variance * 10)
    
    # Regime confidence (mode frequency)
    mode, frequency = compute_mode(tracker.regime_history)
    regime_confidence = frequency
    
    # Basin stability (inverse of spread)
    basin_spread = compute_basin_spread(tracker.basin_history)
    basin_stability = max(0.0, 1.0 - basin_spread * 5)
    
    # Overall confidence (weighted average)
    overall = (
        0.30 * phi_confidence +
        0.30 * kappa_confidence +
        0.20 * regime_confidence +
        0.20 * basin_stability
    )
    
    # Generate explanation
    if overall > 0.8:
        explanation = f"High confidence: metrics stable across {sample_size} samples"
    elif overall > 0.6:
        explanation = "Moderate confidence: some variation observed"
    elif overall > 0.4:
        explanation = "Low confidence: significant metric variance"
    else:
        explanation = "Very low confidence: unstable measurements"
    
    # Add specific warnings
    warnings = []
    if phi_confidence < 0.5:
        warnings.append("Φ unstable")
    if kappa_confidence < 0.5:
        warnings.append("κ unstable")
    if regime_confidence < 0.7:
        warnings.append(f"regime fluctuating ({mode} {int(frequency * 100)}%)")
    
    if warnings:
        explanation += f" | Warnings: {', '.join(warnings)}"
    
    return ConfidenceMetrics(
        overall=overall,
        phi_confidence=phi_confidence,
        kappa_confidence=kappa_confidence,
        regime_confidence=regime_confidence,
        basin_stability=basin_stability,
        sample_size=sample_size,
        explanation=explanation
    )


def estimate_single_sample_confidence(
    phi: float,
    kappa: float,
    regime: str,
    pattern_score: float,
    in_resonance: bool
) -> ConfidenceMetrics:
    """
    Compute single-sample confidence estimate.
    Uses heuristics when we don't have history.
    """
    # Phi confidence: higher Φ = higher confidence (well-integrated)
    phi_confidence = phi
    
    # Kappa confidence: closer to κ* = higher confidence
    kappa_dist = abs(kappa - KAPPA_STAR) / KAPPA_STAR
    kappa_confidence = 1.0 - kappa_dist
    
    # Regime confidence: geometric regime has highest certainty
    regime_map = {
        "geometric": 0.9,
        "linear": 0.7,
        "breakdown": 0.5
    }
    regime_confidence = regime_map.get(regime, 0.5)
    
    # Basin stability: estimated from pattern score
    basin_stability = 0.5 + 0.5 * pattern_score
    
    # Overall
    overall = (
        0.30 * phi_confidence +
        0.30 * kappa_confidence +
        0.20 * regime_confidence +
        0.20 * basin_stability
    )
    
    if overall > 0.7:
        explanation = f"High confidence estimate (single sample, {regime} regime)"
    elif overall > 0.5:
        explanation = "Moderate confidence estimate (need more samples)"
    else:
        explanation = "Low confidence estimate (unstable metrics)"
    
    if in_resonance:
        explanation += " | In resonance zone (κ ≈ κ*)"
    
    return ConfidenceMetrics(
        overall=overall,
        phi_confidence=phi_confidence,
        kappa_confidence=kappa_confidence,
        regime_confidence=regime_confidence,
        basin_stability=basin_stability,
        sample_size=1,
        explanation=explanation
    )


def compute_recovery_confidence(
    kappa_recovery: float,
    phi_constraints: float,
    h_creation: float,
    entity_count: int,
    artifact_count: int,
    is_dormant: bool,
    dormancy_years: float
) -> Dict:
    """
    Compute recovery confidence for a target address.
    Combines QIG metrics with recovery-specific factors.
    """
    factors = {}
    
    # κ_recovery factor: higher = more recoverable
    factors["kappa_recovery"] = min(1.0, kappa_recovery)
    
    # Constraint density factor: more constraints = better
    factors["constraint_density"] = min(1.0, phi_constraints)
    
    # Creation entropy factor: lower = easier
    factors["creation_entropy"] = max(0.0, 1.0 - h_creation / 8.0)
    
    # Entity linkage factor
    factors["entity_linkage"] = min(1.0, entity_count / 5.0)
    
    # Artifact availability factor
    factors["artifact_availability"] = min(1.0, artifact_count / 10.0)
    
    # Dormancy factor: longer dormancy = more likely lost (recoverable)
    factors["dormancy_strength"] = min(1.0, dormancy_years / 10.0) if is_dormant else 0.0
    
    # Weighted confidence
    confidence = (
        0.25 * factors["kappa_recovery"] +
        0.20 * factors["constraint_density"] +
        0.15 * factors["creation_entropy"] +
        0.15 * factors["entity_linkage"] +
        0.10 * factors["artifact_availability"] +
        0.15 * factors["dormancy_strength"]
    )
    
    # Generate recommendation
    if confidence > 0.7:
        recommendation = "HIGH PRIORITY: Strong recovery indicators. Proceed with all vectors."
    elif confidence > 0.5:
        recommendation = "MEDIUM PRIORITY: Moderate indicators. Focus on constrained search and social vectors."
    elif confidence > 0.3:
        recommendation = "LOW PRIORITY: Weak indicators. Estate vector may be most viable."
    else:
        recommendation = "VERY LOW PRIORITY: Insufficient indicators. Consider deprioritizing."
    
    return {
        "confidence": confidence,
        "factors": factors,
        "recommendation": recommendation
    }


def detect_confidence_trend(
    confidence_history: List[float],
    window_size: int = 10
) -> Dict:
    """Track confidence over time and detect trends."""
    if len(confidence_history) < 3:
        return {"trend": "stable", "slope": 0.0, "volatility": 0.0}
    
    # Use recent window
    recent = confidence_history[-window_size:]
    n = len(recent)
    
    # Compute linear regression slope
    x_mean = (n - 1) / 2.0
    y_mean = np.mean(recent)
    
    numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    
    slope = numerator / denominator if denominator != 0 else 0.0
    
    # Compute volatility (standard deviation)
    volatility = float(np.std(recent))
    
    # Classify trend
    if abs(slope) < 0.01:
        trend = "stable"
    elif slope > 0:
        trend = "improving"
    else:
        trend = "declining"
    
    return {
        "trend": trend,
        "slope": float(slope),
        "volatility": volatility
    }
