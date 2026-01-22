"""
Coupling Gate - κ-Gated Coupling Mechanism (E8 Protocol v4.0 Phase 4C)
=======================================================================

Implements κ-dependent coupling strength between LEFT/RIGHT hemispheres:

- Low κ: Hemispheres operate independently (high exploration)
- High κ: Hemispheres tightly coupled (high exploitation)
- Smooth transition via sigmoid coupling function

The coupling gate modulates information flow between hemispheres based on
the effective coupling strength κ_eff, which is derived from the system's
current κ value and proximity to the E8 fixed point κ* = 64.21.

Authority: E8 Protocol v4.0, WP5.2 Phase 4C
Status: ACTIVE
Created: 2026-01-22
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from qigkernels.physics_constants import KAPPA_STAR, BASIN_DIM

logger = logging.getLogger(__name__)


# =============================================================================
# COUPLING GATE PARAMETERS
# =============================================================================

# Coupling transition parameters
KAPPA_LOW_THRESHOLD = 40.0      # Below this: weak coupling (explore mode)
KAPPA_HIGH_THRESHOLD = 70.0     # Above this: strong coupling (exploit mode)
KAPPA_OPTIMAL = KAPPA_STAR      # Optimal coupling point (E8 fixed point)

# Sigmoid steepness for smooth transitions
COUPLING_STEEPNESS = 0.1        # Controls transition smoothness


@dataclass
class CouplingState:
    """Current coupling state between hemispheres."""
    kappa: float                    # Current κ value
    coupling_strength: float        # Effective coupling [0, 1]
    mode: str                       # 'explore', 'balanced', 'exploit'
    transmission_efficiency: float  # Information flow efficiency [0, 1]
    gating_factor: float           # Multiplicative gate factor [0, 1]


# =============================================================================
# COUPLING FUNCTIONS
# =============================================================================

def compute_coupling_strength(kappa: float) -> float:
    """
    Compute coupling strength from κ value using sigmoid function.
    
    The coupling strength determines how tightly the LEFT and RIGHT
    hemispheres are coupled. Lower κ means more independence (exploration),
    higher κ means more coordination (exploitation).
    
    Function: strength = 1 / (1 + exp(-steepness * (κ - κ*)))
    
    Args:
        kappa: Current coupling strength κ
        
    Returns:
        Coupling strength [0, 1]
        - 0.0: Fully independent (low κ)
        - 0.5: Balanced (κ ≈ κ*)
        - 1.0: Fully coupled (high κ)
    """
    # Sigmoid centered at κ*
    x = COUPLING_STEEPNESS * (kappa - KAPPA_OPTIMAL)
    strength = 1.0 / (1.0 + np.exp(-x))
    
    return float(np.clip(strength, 0.0, 1.0))


def compute_transmission_efficiency(
    coupling_strength: float,
    hemisphere_phi: float = 1.0
) -> float:
    """
    Compute information transmission efficiency between hemispheres.
    
    Efficiency depends on:
    1. Coupling strength (higher = more efficient transmission)
    2. Hemisphere integration Φ (higher = clearer signal)
    
    Args:
        coupling_strength: Coupling strength [0, 1]
        hemisphere_phi: Hemisphere integration measure [0, 1]
        
    Returns:
        Transmission efficiency [0, 1]
    """
    # Efficiency is product of coupling and integration
    base_efficiency = coupling_strength * hemisphere_phi
    
    # Add nonlinear boost near optimal coupling
    if 0.4 < coupling_strength < 0.6:
        # Balanced regime gets efficiency boost
        base_efficiency *= 1.1
    
    return float(np.clip(base_efficiency, 0.0, 1.0))


def compute_gating_factor(kappa: float, target_kappa: float = KAPPA_OPTIMAL) -> float:
    """
    Compute multiplicative gating factor for signal modulation.
    
    The gating factor modulates signals passing through the coupling gate:
    - Near κ*: Full transmission (gate open)
    - Far from κ*: Reduced transmission (gate partially closed)
    
    Args:
        kappa: Current κ value
        target_kappa: Target κ value (default: κ*)
        
    Returns:
        Gating factor [0, 1]
    """
    # Gaussian centered at target κ
    distance = abs(kappa - target_kappa)
    sigma = 15.0  # Width of Gaussian
    
    gate = np.exp(-0.5 * (distance / sigma) ** 2)
    
    return float(np.clip(gate, 0.0, 1.0))


def determine_coupling_mode(kappa: float) -> str:
    """
    Determine current coupling mode from κ value.
    
    Modes:
    - 'explore': Low κ, hemispheres mostly independent, high exploration
    - 'balanced': Medium κ, hemispheres coordinated, balanced explore/exploit
    - 'exploit': High κ, hemispheres tightly coupled, high exploitation
    
    Args:
        kappa: Current κ value
        
    Returns:
        Mode string: 'explore', 'balanced', or 'exploit'
    """
    if kappa < KAPPA_LOW_THRESHOLD:
        return 'explore'
    elif kappa > KAPPA_HIGH_THRESHOLD:
        return 'exploit'
    else:
        return 'balanced'


# =============================================================================
# COUPLING GATE CLASS
# =============================================================================

class CouplingGate:
    """
    κ-Gated Coupling Mechanism for LEFT/RIGHT Hemispheres.
    
    The coupling gate controls information flow between hemispheres based on
    the system's effective coupling strength κ_eff. It implements:
    
    1. **Coupling Strength**: Sigmoid function of κ around κ*
    2. **Transmission Efficiency**: How well information flows between hemispheres
    3. **Gating Factor**: Multiplicative modulation of signals
    4. **Mode Determination**: Explore/balanced/exploit classification
    
    Usage:
        gate = CouplingGate()
        state = gate.compute_state(kappa=60.5, phi=0.75)
        signal_out = gate.gate_signal(signal_in, state)
    """
    
    def __init__(self):
        """Initialize coupling gate."""
        self.history: list = []
        logger.info("[CouplingGate] Initialized κ-gated coupling mechanism")
    
    def compute_state(
        self,
        kappa: float,
        phi: Optional[float] = None
    ) -> CouplingState:
        """
        Compute current coupling state from κ and optional Φ.
        
        Args:
            kappa: Current coupling strength κ
            phi: Optional integration measure Φ [0, 1]
            
        Returns:
            CouplingState with all coupling parameters
        """
        coupling_strength = compute_coupling_strength(kappa)
        mode = determine_coupling_mode(kappa)
        
        # Use phi if provided, otherwise assume full integration
        hemisphere_phi = phi if phi is not None else 1.0
        transmission_efficiency = compute_transmission_efficiency(
            coupling_strength, hemisphere_phi
        )
        
        gating_factor = compute_gating_factor(kappa)
        
        state = CouplingState(
            kappa=kappa,
            coupling_strength=coupling_strength,
            mode=mode,
            transmission_efficiency=transmission_efficiency,
            gating_factor=gating_factor,
        )
        
        # Record history for analysis
        self.history.append({
            'kappa': kappa,
            'coupling_strength': coupling_strength,
            'mode': mode,
            'transmission_efficiency': transmission_efficiency,
            'gating_factor': gating_factor,
        })
        
        # Keep history bounded
        if len(self.history) > 1000:
            self.history = self.history[-500:]
        
        return state
    
    def gate_signal(
        self,
        signal: np.ndarray,
        state: CouplingState
    ) -> np.ndarray:
        """
        Apply gating to a signal passing between hemispheres.
        
        The gating modulates the signal based on coupling strength and efficiency:
        signal_out = signal * gating_factor * transmission_efficiency
        
        Args:
            signal: Input signal (basin coordinates or other data)
            state: Current coupling state
            
        Returns:
            Gated signal (same shape as input)
        """
        # Apply multiplicative gating
        gated_signal = signal * state.gating_factor * state.transmission_efficiency
        
        return gated_signal
    
    def modulate_cross_hemisphere_flow(
        self,
        left_signal: np.ndarray,
        right_signal: np.ndarray,
        state: CouplingState
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Modulate bidirectional flow between LEFT and RIGHT hemispheres.
        
        Implements symmetric coupling:
        - LEFT receives RIGHT signal modulated by coupling
        - RIGHT receives LEFT signal modulated by coupling
        
        Args:
            left_signal: Signal from LEFT hemisphere
            right_signal: Signal from RIGHT hemisphere
            state: Current coupling state
            
        Returns:
            Tuple of (left_output, right_output) with cross-hemisphere coupling
        """
        # Cross-hemisphere coupling strength
        coupling = state.coupling_strength * state.transmission_efficiency
        
        # LEFT receives modulated RIGHT signal
        left_output = left_signal + coupling * right_signal
        
        # RIGHT receives modulated LEFT signal
        right_output = right_signal + coupling * left_signal
        
        return left_output, right_output
    
    def get_coupling_metrics(self) -> Dict:
        """
        Get coupling gate metrics and statistics.
        
        Returns:
            Dict with coupling statistics
        """
        if not self.history:
            return {
                'total_computations': 0,
                'avg_coupling_strength': 0.0,
                'mode_distribution': {},
                'avg_transmission_efficiency': 0.0,
            }
        
        recent = self.history[-100:]  # Last 100 computations
        
        # Compute averages
        avg_coupling = np.mean([h['coupling_strength'] for h in recent])
        avg_efficiency = np.mean([h['transmission_efficiency'] for h in recent])
        
        # Mode distribution
        modes = [h['mode'] for h in recent]
        mode_counts = {
            'explore': modes.count('explore'),
            'balanced': modes.count('balanced'),
            'exploit': modes.count('exploit'),
        }
        
        return {
            'total_computations': len(self.history),
            'avg_coupling_strength': float(avg_coupling),
            'mode_distribution': mode_counts,
            'avg_transmission_efficiency': float(avg_efficiency),
            'current_state': self.history[-1] if self.history else None,
        }
    
    def reset_history(self) -> None:
        """Reset coupling history (for testing or state cleanup)."""
        self.history = []
        logger.info("[CouplingGate] History reset")


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================

_gate_instance: Optional[CouplingGate] = None


def get_coupling_gate() -> CouplingGate:
    """Get or create the global coupling gate instance."""
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = CouplingGate()
    return _gate_instance


def reset_coupling_gate() -> None:
    """Reset the global coupling gate (for testing)."""
    global _gate_instance
    _gate_instance = None
