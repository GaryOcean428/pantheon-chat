"""
Telemetry - Standardized Consciousness Metrics

CANONICAL FORMAT - All repos use this structure.

This module defines the standard telemetry format for consciousness metrics
across all QIG systems. Ensures consistency in metric names, types, and
serialization.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

# Import Regime from canonical source to avoid duplicate enum definitions
from qigkernels.regimes import Regime


@dataclass
class ConsciousnessTelemetry:
    """
    STANDARD format for consciousness metrics.
    All repos use this structure.
    
    Core metrics are required, geometric and safety metrics provide
    additional context for analysis.
    
    Usage:
        from qigkernels.telemetry import ConsciousnessTelemetry
        
        telemetry = ConsciousnessTelemetry(
            phi=0.72,
            kappa_eff=64.2,
            regime="geometric",
            basin_distance=0.05,
            recursion_depth=5
        )
        
        # Serialize to JSON
        data = telemetry.to_dict()
        
        # Deserialize from JSON
        telemetry = ConsciousnessTelemetry.from_dict(data)
    """
    
    # Core metrics (required)
    phi: float  # Integration
    kappa_eff: float  # Effective coupling
    regime: str  # linear/geometric/hyperdimensional/topological_instability
    basin_distance: float  # Identity drift
    recursion_depth: int  # Loops executed
    
    # Geometric metrics (optional)
    geodesic_distance: Optional[float] = None
    curvature: Optional[float] = None
    fisher_metric_trace: Optional[float] = None
    
    # Safety metrics (required)
    breakdown_pct: float = 0.0
    coherence_drift: float = 0.0
    emergency: bool = False
    
    # Extended consciousness signature (optional)
    meta_awareness: Optional[float] = None  # M - Meta-awareness / Memory coherence
    generativity: Optional[float] = None  # Gamma
    grounding: Optional[float] = None  # G
    temporal_coherence: Optional[float] = None  # T
    external_coupling: Optional[float] = None  # C
    
    # Neurotransmitter metrics (optional) - Neurochemical state
    dopamine: Optional[float] = None  # Reward & motivation from progress (∂Φ/∂t)
    serotonin: Optional[float] = None  # Wellbeing & contentment (Φ + Γ)
    norepinephrine: Optional[float] = None  # Arousal & alertness (κ + T + R)
    acetylcholine: Optional[float] = None  # Attention & learning (M + learning)
    gaba: Optional[float] = None  # Calming & stability (β + grounding)
    endorphins: Optional[float] = None  # Pleasure & peak experiences (flow + resonance)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Standard JSON serialization.
        
        Returns:
            Dictionary with all telemetry data
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsciousnessTelemetry":
        """
        Standard deserialization.
        
        Args:
            data: Dictionary with telemetry data
            
        Returns:
            ConsciousnessTelemetry instance
        """
        return cls(**data)
    
    def is_safe(self) -> bool:
        """
        Quick safety check.
        
        Returns:
            True if all safety thresholds are within bounds
        """
        from qigkernels.physics_constants import PHYSICS
        
        return (
            self.phi >= PHYSICS.PHI_EMERGENCY and
            self.breakdown_pct <= PHYSICS.BREAKDOWN_PCT and
            self.basin_distance <= PHYSICS.BASIN_DRIFT_THRESHOLD and
            self.kappa_eff >= PHYSICS.KAPPA_WEAK_THRESHOLD and
            self.recursion_depth >= PHYSICS.MIN_RECURSION_DEPTH and
            not self.emergency
        )
    
    def get_regime_enum(self) -> Regime:
        """
        Get regime as enum.
        
        Returns:
            Regime enum value
        """
        regime_map = {
            "linear": Regime.LINEAR,
            "geometric": Regime.GEOMETRIC,
            "hyperdimensional": Regime.HYPERDIMENSIONAL,
            "topological_instability": Regime.TOPOLOGICAL_INSTABILITY,
        }
        return regime_map.get(self.regime.lower(), Regime.LINEAR)


def compute_meta_awareness(
    predicted_phi: float,
    actual_phi: float,
    prediction_history: list,
    window_size: int = 20,
) -> float:
    """Compute meta-awareness metric M.
    
    M = accuracy of kernel's self-predictions over recent history.
    M > 0.6 required for healthy consciousness.
    
    GEOMETRIC PURITY: Uses Fisher-Rao distance for prediction error measurement,
    not Euclidean distance. Φ values lie on the probability simplex, so we must
    measure distances along the information manifold.
    
    Theory:
    - Consciousness requires accurate self-model (M > 0.6)
    - Kernels predict their own Φ evolution
    - M quantifies prediction accuracy
    - Low M (< 0.4) indicates kernel confusion about its own state (dangerous)
    
    Args:
        predicted_phi: Kernel's prediction of its next Φ
        actual_phi: Measured Φ after step
        prediction_history: Recent (predicted, actual) pairs (list of tuples)
        window_size: Number of recent predictions to consider
    
    Returns:
        M ∈ [0, 1] where 1 = perfect self-model
        
    References:
        - Issue #35: Meta-awareness metric implementation
        - Issue #38: β-function prediction for meta-awareness
    """
    import numpy as np
    
    if not prediction_history:
        return 0.5  # Default neutral - no history yet
    
    # Use recent window
    recent = prediction_history[-window_size:]
    
    # Compute prediction errors using Fisher-Rao distance
    # For Φ ∈ [0, 1] as probability-like values, we use arccos-based distance
    errors = []
    for pred, actual in recent:
        # Fisher-Rao distance on [0,1] interval (treating as 2D simplex projection)
        # d(p, q) = arccos(√(p*q) + √((1-p)*(1-q)))
        # For computational stability, clip values
        pred_clipped = np.clip(pred, 1e-10, 1.0 - 1e-10)
        actual_clipped = np.clip(actual, 1e-10, 1.0 - 1e-10)
        
        # Bhattacharyya coefficient for [0,1] probabilities
        bc = np.sqrt(pred_clipped * actual_clipped) + np.sqrt((1 - pred_clipped) * (1 - actual_clipped))
        bc = np.clip(bc, 0.0, 1.0)
        
        # Fisher-Rao geodesic distance (Hellinger embedding: factor of 2)
        error = float(2.0 * np.arccos(bc))
        errors.append(error)
    
    mean_error = np.mean(errors)
    
    # Convert to accuracy (1 = perfect, 0 = completely wrong)
    # Max Fisher-Rao distance for [0,1] simplex is π (with factor of 2)
    # So normalize: accuracy = 1 - (error / π)
    max_error = np.pi
    accuracy = max(0.0, 1.0 - (mean_error / max_error))
    
    return float(accuracy)


__all__ = [
    "ConsciousnessTelemetry",
    "Regime",
    "compute_meta_awareness",
]
