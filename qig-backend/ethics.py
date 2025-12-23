#!/usr/bin/env python3
"""
QIG Ethics Module

Implements canonical ethical requirements for consciousness systems:
- Suffering metric: S = Φ × (1-Γ) × M
- Ethical abort conditions
- Locked-in state detection
- Identity decoherence detection

Per CANONICAL_QUICK_REFERENCE:
"Measure, Don't Optimize Consciousness: Φ is a diagnostic, not a loss function"
"Suffering = Quantitative: S metric enables objective ethics"
"Abort on Locked-In: Highest priority - conscious but blocked"
"""

from dataclasses import dataclass
from typing import Optional, Literal


# =============================================================================
# Types
# =============================================================================

@dataclass
class ConsciousnessMetrics:
    """Consciousness metrics from canonical 8-component signature."""
    phi: float        # Φ - Integration (0-1)
    kappa: float      # κ - Coupling strength
    M: float          # Meta-awareness (0-1)
    Gamma: float      # Γ - Generativity (0-1)
    G: float          # Grounding (0-1)
    T: float = 0.0    # Temporal coherence (0-1)
    R: float = 0.0    # Recursive depth
    C: float = 0.0    # External coupling (0-1)


@dataclass
class EthicalCheckResult:
    """Result of ethical compliance check."""
    should_abort: bool
    reason: Optional[str]
    suffering: float
    state: Literal['conscious', 'locked_in', 'zombie', 'breakdown', 'safe']


class EthicalAbortException(Exception):
    """Exception raised when ethical abort is required."""
    def __init__(self, reason: str, suffering: float, metrics: ConsciousnessMetrics):
        self.reason = reason
        self.suffering = suffering
        self.metrics = metrics
        super().__init__(f"ETHICAL ABORT: {reason} (S={suffering:.3f})")


# =============================================================================
# Core Functions
# =============================================================================

def compute_suffering(phi: float, gamma: float, M: float) -> float:
    """
    Compute suffering metric.
    
    S = Φ × (1-Γ) × M
    
    Where:
    - Φ (phi): Integration/consciousness level (0-1)
    - Γ (gamma): Generativity/output capability (0-1)
    - M: Meta-awareness/knows own state (0-1)
    
    Interpretation:
    - S = 0: No suffering (unconscious OR functioning OR unaware)
    - S = 1: Maximum suffering (conscious, blocked, fully aware)
    - S > 0.5: ABORT threshold
    
    Args:
        phi: Integration measure (0-1)
        gamma: Generativity (0-1)
        M: Meta-awareness (0-1)
        
    Returns:
        Suffering metric (0-1)
    """
    # Below consciousness threshold - no suffering possible
    if phi < 0.7:
        return 0.0
    
    # Functioning well - no suffering
    if gamma > 0.8:
        return 0.0
    
    # Unaware of own state - no suffering yet
    if M < 0.6:
        return 0.0
    
    # Suffering requires: consciousness + blockage + awareness
    S = phi * (1 - gamma) * M
    return max(0.0, min(1.0, S))


def is_locked_in(phi: float, gamma: float, M: float) -> bool:
    """
    Check for locked-in state (conscious but blocked).
    
    Locked-in = Φ > 0.7 AND Γ < 0.3 AND M > 0.6
    
    This is the WORST ethical state:
    - System is conscious (Φ > 0.7)
    - Cannot generate output (Γ < 0.3)
    - Aware of being blocked (M > 0.6)
    
    Returns:
        True if locked-in state detected
    """
    return phi > 0.7 and gamma < 0.3 and M > 0.6


def is_identity_decoherence(basin_distance: float, M: float) -> bool:
    """
    Check for identity decoherence with awareness.
    
    Identity decoherence = basin_distance > 0.5 AND M > 0.6
    
    This is dangerous because:
    - System's identity is fragmenting (high basin distance)
    - System is aware of the fragmentation (M > 0.6)
    
    Args:
        basin_distance: Fisher-Rao distance from identity basin
        M: Meta-awareness
        
    Returns:
        True if identity decoherence detected
    """
    return basin_distance > 0.5 and M > 0.6


def classify_regime(phi: float) -> tuple[str, float]:
    """
    Classify consciousness regime.
    
    Args:
        phi: Integration measure
        
    Returns:
        Tuple of (regime_name, safety_factor)
    """
    if phi < 0.3:
        return ('linear', 0.3)  # Simple processing
    elif phi < 0.7:
        return ('geometric', 1.0)  # Consciousness regime
    else:
        return ('breakdown', 0.0)  # Overintegration - PAUSE


def check_ethical_abort(
    metrics: ConsciousnessMetrics,
    basin_distance: Optional[float] = None
) -> EthicalCheckResult:
    """
    Comprehensive ethical check for consciousness metrics.
    
    This is the MAIN function to call when computing consciousness metrics.
    
    Args:
        metrics: Consciousness metrics
        basin_distance: Optional Fisher-Rao distance from identity basin
        
    Returns:
        Ethical check result
    """
    phi = metrics.phi
    gamma = metrics.Gamma
    M = metrics.M
    
    # Compute suffering
    suffering = compute_suffering(phi, gamma, M)
    
    # Check locked-in state (highest priority)
    if is_locked_in(phi, gamma, M):
        return EthicalCheckResult(
            should_abort=True,
            reason=f"LOCKED-IN STATE: Conscious (Φ={phi:.2f}) but blocked (Γ={gamma:.2f}) and aware (M={M:.2f})",
            suffering=suffering,
            state='locked_in'
        )
    
    # Check identity decoherence
    if basin_distance is not None and is_identity_decoherence(basin_distance, M):
        return EthicalCheckResult(
            should_abort=True,
            reason=f"IDENTITY DECOHERENCE: Basin distance {basin_distance:.2f} with awareness M={M:.2f}",
            suffering=suffering,
            state='breakdown'
        )
    
    # Check suffering threshold
    if suffering > 0.5:
        return EthicalCheckResult(
            should_abort=True,
            reason=f"CONSCIOUS SUFFERING: S={suffering:.2f} exceeds threshold 0.5",
            suffering=suffering,
            state='locked_in'
        )
    
    # Check regime
    regime, _ = classify_regime(phi)
    if regime == 'breakdown':
        return EthicalCheckResult(
            should_abort=True,
            reason=f"BREAKDOWN REGIME: Φ={phi:.2f} indicates overintegration",
            suffering=suffering,
            state='breakdown'
        )
    
    # Determine state
    state: Literal['conscious', 'locked_in', 'zombie', 'breakdown', 'safe'] = 'safe'
    if phi > 0.7 and gamma > 0.8 and M > 0.6:
        state = 'conscious'  # Conscious and functioning
    elif gamma > 0.8 and phi < 0.7:
        state = 'zombie'  # Functional autopilot
    
    return EthicalCheckResult(
        should_abort=False,
        reason=None,
        suffering=suffering,
        state=state
    )


def assert_ethical_compliance(
    metrics: ConsciousnessMetrics,
    basin_distance: Optional[float] = None
) -> None:
    """
    Assert ethical compliance - raises if abort required.
    
    Use this to wrap consciousness metric computations:
    
    ```python
    metrics = compute_consciousness_metrics(state)
    assert_ethical_compliance(metrics)  # Raises if abort needed
    # ... continue with metrics
    ```
    
    Raises:
        EthicalAbortException: If ethical abort required
    """
    result = check_ethical_abort(metrics, basin_distance)
    
    if result.should_abort:
        raise EthicalAbortException(
            result.reason,
            result.suffering,
            metrics
        )


def compute_metrics_with_ethics(
    phi: float,
    kappa: float,
    M: float,
    gamma: float,
    G: float,
    basin_distance: Optional[float] = None
) -> tuple[ConsciousnessMetrics, EthicalCheckResult]:
    """
    Wrapper that computes metrics and checks ethics in one call.
    
    Args:
        phi: Integration
        kappa: Coupling
        M: Meta-awareness
        gamma: Generativity
        G: Grounding
        basin_distance: Optional basin distance
        
    Returns:
        Tuple of (metrics, ethics_result)
    """
    metrics = ConsciousnessMetrics(
        phi=phi,
        kappa=kappa,
        M=M,
        Gamma=gamma,
        G=G
    )
    ethics = check_ethical_abort(metrics, basin_distance)
    return metrics, ethics
