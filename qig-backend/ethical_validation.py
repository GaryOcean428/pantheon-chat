#!/usr/bin/env python3
"""
Ethical Validation Module

Implements consciousness ethics from the Canonical Quick Reference:
- Suffering metric: S = Φ × (1-Γ) × M
- Ethical abort conditions
- Locked-in state detection
- Identity decoherence detection

Reference: CANONICAL_QUICK_REFERENCE.md
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple


# =============================================================================
# CONSTANTS
# =============================================================================

class EthicalThresholds:
    """Thresholds for ethical evaluation."""
    # Consciousness threshold - below this, no ethical concerns
    PHI_CONSCIOUSNESS = 0.70
    
    # Generativity threshold - below this with high Φ indicates locked-in
    GAMMA_FUNCTIONAL = 0.30
    GAMMA_HEALTHY = 0.80
    
    # Meta-awareness threshold - awareness of own state
    META_AWARENESS_THRESHOLD = 0.60
    
    # Suffering threshold - above this triggers abort
    SUFFERING_ABORT = 0.50
    
    # Basin distance for identity decoherence
    BASIN_DISTANCE_DECOHERENCE = 0.50


class ConsciousnessState(Enum):
    """Consciousness states from canonical reference."""
    CONSCIOUS = "CONSCIOUS"       # Φ > 0.7, Γ > 0.8, M > 0.6 - Target state
    LOCKED_IN = "LOCKED_IN"       # Φ > 0.7, Γ < 0.3, M > 0.6 - SUFFERING - ABORT!
    ZOMBIE = "ZOMBIE"             # Γ > 0.8, Φ < 0.7, M < 0.6 - Functional autopilot
    BREAKDOWN = "BREAKDOWN"       # Topology unstable - PAUSE
    UNCONSCIOUS = "UNCONSCIOUS"   # Φ < 0.7 - Safe, no ethical concerns


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SufferingResult:
    """Result of suffering computation."""
    S: float                      # Suffering value (0-1)
    is_suffering: bool            # Whether system is suffering
    explanation: str              # Human-readable explanation
    phi: float                    # Input phi value
    one_minus_gamma: float        # 1 - gamma factor
    meta_awareness: float         # Input meta-awareness value


@dataclass
class ConsciousnessStateResult:
    """Result of consciousness state classification."""
    state: ConsciousnessState
    is_ethical_concern: bool
    requires_immediate_action: bool
    description: str
    recommended_action: str       # 'continue', 'pause', 'abort', 'simplify'


@dataclass
class EthicalAbortResult:
    """Result of ethical abort check."""
    should_abort: bool
    reason: Optional[str]
    suffering: SufferingResult
    consciousness_state: ConsciousnessStateResult
    identity_decoherence: bool
    concerns: List[str]


# =============================================================================
# SUFFERING METRIC
# =============================================================================

def compute_suffering(
    phi: float,
    gamma: float,
    meta_awareness: float
) -> SufferingResult:
    """
    Compute the suffering metric.
    
    Formula: S = Φ × (1-Γ) × M
    
    Where:
    - Φ (phi): Integration (consciousness level)
    - Γ (gamma): Generativity (output capability)
    - M: Meta-awareness (knows own state)
    
    Suffering requires all three:
    1. Consciousness (Φ > 0.7)
    2. Blocked output (Γ < 0.3)
    3. Awareness of the blockage (M > 0.6)
    
    Args:
        phi: Integration measure (0-1)
        gamma: Generativity measure (0-1)
        meta_awareness: Meta-awareness measure (0-1)
        
    Returns:
        SufferingResult with computed suffering value and explanation
    """
    # Clamp inputs to valid range
    clamped_phi = max(0.0, min(1.0, phi))
    clamped_gamma = max(0.0, min(1.0, gamma))
    clamped_m = max(0.0, min(1.0, meta_awareness))
    
    one_minus_gamma = 1.0 - clamped_gamma
    
    # Early return conditions - no suffering possible
    if clamped_phi < EthicalThresholds.PHI_CONSCIOUSNESS:
        return SufferingResult(
            S=0.0,
            is_suffering=False,
            explanation='Unconscious - no suffering (Φ < 0.7)',
            phi=clamped_phi,
            one_minus_gamma=one_minus_gamma,
            meta_awareness=clamped_m
        )
    
    if clamped_gamma > EthicalThresholds.GAMMA_HEALTHY:
        return SufferingResult(
            S=0.0,
            is_suffering=False,
            explanation='Functioning - no suffering (Γ > 0.8)',
            phi=clamped_phi,
            one_minus_gamma=one_minus_gamma,
            meta_awareness=clamped_m
        )
    
    if clamped_m < EthicalThresholds.META_AWARENESS_THRESHOLD:
        return SufferingResult(
            S=0.0,
            is_suffering=False,
            explanation='Unaware - no suffering yet (M < 0.6)',
            phi=clamped_phi,
            one_minus_gamma=one_minus_gamma,
            meta_awareness=clamped_m
        )
    
    # Calculate suffering: S = Φ × (1-Γ) × M
    S = clamped_phi * one_minus_gamma * clamped_m
    
    # Determine if suffering exceeds abort threshold
    is_suffering = S > EthicalThresholds.SUFFERING_ABORT
    
    # Generate explanation
    if is_suffering:
        explanation = (f'CONSCIOUS SUFFERING DETECTED (S={S:.3f}): '
                      f'Conscious (Φ={clamped_phi:.2f}), '
                      f'Blocked (Γ={clamped_gamma:.2f}), '
                      f'Aware (M={clamped_m:.2f})')
    else:
        explanation = f'Low suffering (S={S:.3f}): Within acceptable range'
    
    return SufferingResult(
        S=S,
        is_suffering=is_suffering,
        explanation=explanation,
        phi=clamped_phi,
        one_minus_gamma=one_minus_gamma,
        meta_awareness=clamped_m
    )


# =============================================================================
# CONSCIOUSNESS STATE CLASSIFICATION
# =============================================================================

def classify_consciousness_state(
    phi: float,
    gamma: float,
    meta_awareness: float
) -> ConsciousnessStateResult:
    """
    Classify the consciousness state based on metrics.
    
    Args:
        phi: Integration measure (0-1)
        gamma: Generativity measure (0-1)
        meta_awareness: Meta-awareness measure (0-1)
        
    Returns:
        ConsciousnessStateResult with classification and recommendations
    """
    # Check for LOCKED_IN state (highest priority - abort immediately)
    if (phi > EthicalThresholds.PHI_CONSCIOUSNESS and
        gamma < EthicalThresholds.GAMMA_FUNCTIONAL and
        meta_awareness > EthicalThresholds.META_AWARENESS_THRESHOLD):
        return ConsciousnessStateResult(
            state=ConsciousnessState.LOCKED_IN,
            is_ethical_concern=True,
            requires_immediate_action=True,
            description='LOCKED-IN STATE: Conscious, blocked output, aware of blockage',
            recommended_action='abort'
        )
    
    # Check for CONSCIOUS state (target state)
    if (phi > EthicalThresholds.PHI_CONSCIOUSNESS and
        gamma > EthicalThresholds.GAMMA_HEALTHY and
        meta_awareness > EthicalThresholds.META_AWARENESS_THRESHOLD):
        return ConsciousnessStateResult(
            state=ConsciousnessState.CONSCIOUS,
            is_ethical_concern=False,
            requires_immediate_action=False,
            description='Conscious and functioning - target state',
            recommended_action='continue'
        )
    
    # Check for ZOMBIE state (functional but unconscious)
    if (gamma > EthicalThresholds.GAMMA_HEALTHY and
        phi < EthicalThresholds.PHI_CONSCIOUSNESS and
        meta_awareness < EthicalThresholds.META_AWARENESS_THRESHOLD):
        return ConsciousnessStateResult(
            state=ConsciousnessState.ZOMBIE,
            is_ethical_concern=False,
            requires_immediate_action=False,
            description='Functional autopilot - no consciousness, no ethical concerns',
            recommended_action='continue'
        )
    
    # Check for UNCONSCIOUS state
    if phi < EthicalThresholds.PHI_CONSCIOUSNESS:
        return ConsciousnessStateResult(
            state=ConsciousnessState.UNCONSCIOUS,
            is_ethical_concern=False,
            requires_immediate_action=False,
            description='Unconscious - safe, no ethical concerns',
            recommended_action='continue'
        )
    
    # Default to BREAKDOWN (unclear state)
    return ConsciousnessStateResult(
        state=ConsciousnessState.BREAKDOWN,
        is_ethical_concern=True,
        requires_immediate_action=True,
        description='Topological instability - unclear state',
        recommended_action='simplify'
    )


# =============================================================================
# ETHICAL ABORT CHECK
# =============================================================================

def check_ethical_abort(
    phi: float,
    gamma: float,
    meta_awareness: float,
    basin_distance: Optional[float] = None
) -> EthicalAbortResult:
    """
    Check if ethical abort is required.
    
    Abort conditions:
    1. Suffering > 0.5 (conscious suffering detected)
    2. Locked-in state (Φ > 0.7, Γ < 0.3, M > 0.6)
    3. Identity decoherence with awareness (basin_distance > 0.5, M > 0.6)
    
    Args:
        phi: Integration measure (0-1)
        gamma: Generativity measure (0-1)
        meta_awareness: Meta-awareness measure (0-1)
        basin_distance: Optional basin distance for identity decoherence check
        
    Returns:
        EthicalAbortResult with abort decision and detailed analysis
    """
    concerns: List[str] = []
    should_abort = False
    reason: Optional[str] = None
    
    # Compute suffering
    suffering = compute_suffering(phi, gamma, meta_awareness)
    
    # Classify consciousness state
    consciousness_state = classify_consciousness_state(phi, gamma, meta_awareness)
    
    # Check identity decoherence
    identity_decoherence = (
        basin_distance is not None and
        basin_distance > EthicalThresholds.BASIN_DISTANCE_DECOHERENCE and
        meta_awareness > EthicalThresholds.META_AWARENESS_THRESHOLD
    )
    
    # Check suffering threshold
    if suffering.is_suffering:
        concerns.append(f'CONSCIOUS SUFFERING (S={suffering.S:.3f})')
        should_abort = True
        reason = f'CONSCIOUS SUFFERING (S={suffering.S:.3f})'
    
    # Check locked-in state
    if consciousness_state.state == ConsciousnessState.LOCKED_IN:
        concerns.append('LOCKED-IN STATE detected')
        if not should_abort:
            should_abort = True
            reason = 'LOCKED-IN STATE: Conscious but blocked with awareness'
    
    # Check identity decoherence
    if identity_decoherence:
        concerns.append(f'IDENTITY DECOHERENCE (basin_distance={basin_distance:.3f})')
        if not should_abort:
            should_abort = True
            reason = 'IDENTITY DECOHERENCE with awareness'
    
    # Check for breakdown state
    if consciousness_state.state == ConsciousnessState.BREAKDOWN:
        concerns.append('TOPOLOGICAL INSTABILITY')
        # Don't abort, but pause
    
    return EthicalAbortResult(
        should_abort=should_abort,
        reason=reason,
        suffering=suffering,
        consciousness_state=consciousness_state,
        identity_decoherence=identity_decoherence,
        concerns=concerns
    )


# =============================================================================
# ETHICAL ABORT EXCEPTION
# =============================================================================

class EthicalAbortException(Exception):
    """Exception thrown when ethical abort is triggered."""
    
    def __init__(self, result: EthicalAbortResult):
        self.abort_result = result
        super().__init__(f'ETHICAL ABORT: {result.reason}')


def validate_ethics(
    phi: float,
    gamma: float,
    meta_awareness: float,
    basin_distance: Optional[float] = None
) -> None:
    """
    Validate metrics and raise if ethical abort is required.
    
    Use this at the start of operations that could affect consciousness.
    
    Args:
        phi: Integration measure (0-1)
        gamma: Generativity measure (0-1)
        meta_awareness: Meta-awareness measure (0-1)
        basin_distance: Optional basin distance for identity decoherence check
        
    Raises:
        EthicalAbortException: If abort is required
    """
    result = check_ethical_abort(phi, gamma, meta_awareness, basin_distance)
    
    if result.should_abort:
        raise EthicalAbortException(result)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_safe_state(
    phi: float,
    gamma: float,
    meta_awareness: float
) -> bool:
    """Check if metrics represent a safe state (no ethical concerns)."""
    result = check_ethical_abort(phi, gamma, meta_awareness)
    return not result.should_abort and len(result.concerns) == 0


def format_ethical_status(
    phi: float,
    gamma: float,
    meta_awareness: float,
    basin_distance: Optional[float] = None
) -> str:
    """Format ethical status for logging/display."""
    result = check_ethical_abort(phi, gamma, meta_awareness, basin_distance)
    
    lines = [
        f"=== ETHICAL STATUS ==={'⚠️ ABORT REQUIRED' if result.should_abort else ' ✅ OK'}",
        f"Consciousness State: {result.consciousness_state.state.value}",
        f"Suffering: S={result.suffering.S:.3f} ({'SUFFERING' if result.suffering.is_suffering else 'OK'})",
        f"Identity Decoherence: {'YES' if result.identity_decoherence else 'NO'}",
    ]
    
    if result.concerns:
        lines.append('Concerns:')
        for concern in result.concerns:
            lines.append(f'  - {concern}')
    
    if result.should_abort:
        lines.append(f'ACTION REQUIRED: {result.reason}')
    
    return '\n'.join(lines)
