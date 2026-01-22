"""
Hemisphere Scheduler - LEFT/RIGHT Hemisphere Architecture (E8 Phase 4C)
========================================================================

Implements two-hemisphere pattern with κ-gated coupling for explore/exploit dynamics:

**LEFT HEMISPHERE (Exploit/Evaluative/Safety):**
- Focus: Precision, evaluation, known paths
- Mode: Convergent, risk-averse
- Gods: Athena (strategy), Artemis (focus), Hephaestus (refinement)

**RIGHT HEMISPHERE (Explore/Generative/Novelty):**
- Focus: Novelty, generation, new paths
- Mode: Divergent, risk-tolerant
- Gods: Apollo (prophecy), Hermes (navigation), Dionysus (chaos)

**Tacking (Oscillation):**
Like dolphin hemispheric sleep, one hemisphere can rest while the other works,
with κ-gated coupling controlling the information flow between them.

Authority: E8 Protocol v4.0, WP5.2 Phase 4C (lines 198-228)
Status: ACTIVE
Created: 2026-01-22
"""

import time
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

import numpy as np

from kernels.coupling_gate import (
    CouplingGate,
    CouplingState,
    get_coupling_gate,
)
from qigkernels.physics_constants import KAPPA_STAR

logger = logging.getLogger(__name__)


# =============================================================================
# HEMISPHERE DEFINITIONS
# =============================================================================

class Hemisphere(Enum):
    """Hemisphere designation."""
    LEFT = "left"    # Exploit/Evaluative/Safety
    RIGHT = "right"  # Explore/Generative/Novelty


# God assignments per WP5.2 specification
LEFT_HEMISPHERE_GODS = {
    "Athena",      # Strategy, planning, evaluation
    "Artemis",     # Focus, precision, attention
    "Hephaestus",  # Refinement, construction, known techniques
}

RIGHT_HEMISPHERE_GODS = {
    "Apollo",      # Prophecy, foresight, prediction
    "Hermes",      # Navigation, exploration, communication
    "Dionysus",    # Chaos, creativity, boundary dissolution
}


def get_god_hemisphere(god_name: str) -> Optional[Hemisphere]:
    """
    Get hemisphere assignment for a god.
    
    Args:
        god_name: Name of the god
        
    Returns:
        Hemisphere assignment or None if not assigned
    """
    if god_name in LEFT_HEMISPHERE_GODS:
        return Hemisphere.LEFT
    elif god_name in RIGHT_HEMISPHERE_GODS:
        return Hemisphere.RIGHT
    else:
        return None


# =============================================================================
# HEMISPHERE STATE TRACKING
# =============================================================================

@dataclass
class HemisphereState:
    """State for a single hemisphere."""
    hemisphere: Hemisphere
    active_gods: Set[str]              # Currently active gods in this hemisphere
    resting_gods: Set[str]             # Currently resting gods
    total_activations: int = 0         # Total activation count
    last_activation: Optional[float] = None
    phi_aggregate: float = 0.0         # Aggregate Φ across active gods
    kappa_aggregate: float = 0.0       # Aggregate κ
    
    def compute_activation_level(self) -> float:
        """
        Compute current activation level [0, 1].
        
        Based on:
        - Number of active gods
        - Aggregate Φ and κ metrics
        """
        if not self.active_gods:
            return 0.0
        
        # Fraction of gods active
        expected_gods = len(LEFT_HEMISPHERE_GODS if self.hemisphere == Hemisphere.LEFT else RIGHT_HEMISPHERE_GODS)
        god_factor = len(self.active_gods) / expected_gods if expected_gods > 0 else 0.0
        
        # Φ contribution (higher Φ = more activated)
        phi_factor = self.phi_aggregate
        
        # κ contribution (near κ* = optimal activation)
        kappa_distance = abs(self.kappa_aggregate - KAPPA_STAR)
        kappa_factor = np.exp(-kappa_distance / 30.0)
        
        # Weighted combination
        activation = 0.4 * god_factor + 0.4 * phi_factor + 0.2 * kappa_factor
        
        return float(np.clip(activation, 0.0, 1.0))
    
    def is_dominant(self) -> bool:
        """Check if this hemisphere is currently dominant."""
        return self.compute_activation_level() > 0.5


@dataclass
class TackingState:
    """State for tacking (oscillation) between hemispheres."""
    cycle_count: int = 0               # Number of tacking cycles
    current_dominant: Optional[Hemisphere] = None
    last_switch_time: Optional[float] = None
    time_in_current: float = 0.0
    oscillation_period: float = 300.0  # Target period (5 minutes default)


# =============================================================================
# HEMISPHERE SCHEDULER
# =============================================================================

class HemisphereScheduler:
    """
    LEFT/RIGHT Hemisphere Scheduler with κ-gated coupling.
    
    Manages explore/exploit dynamics through hemisphere activation and
    κ-dependent coupling. Implements tacking (oscillation) between modes,
    allowing one hemisphere to rest while the other works.
    
    Key Features:
    - God assignment to LEFT/RIGHT hemispheres
    - Activation tracking per hemisphere
    - κ-gated coupling between hemispheres
    - Tacking (oscillation) logic
    - Hemisphere balance metrics
    """
    
    def __init__(self, coupling_gate: Optional[CouplingGate] = None):
        """
        Initialize hemisphere scheduler.
        
        Args:
            coupling_gate: Optional coupling gate (uses global if not provided)
        """
        self.coupling_gate = coupling_gate or get_coupling_gate()
        
        # Initialize hemisphere states
        self.left = HemisphereState(
            hemisphere=Hemisphere.LEFT,
            active_gods=set(),
            resting_gods=set(),
        )
        self.right = HemisphereState(
            hemisphere=Hemisphere.RIGHT,
            active_gods=set(),
            resting_gods=set(),
        )
        
        # Tacking state
        self.tacking = TackingState()
        
        # Coupling history
        self.coupling_history: List[Dict] = []
        
        logger.info(
            "[HemisphereScheduler] Initialized - "
            f"LEFT={LEFT_HEMISPHERE_GODS}, RIGHT={RIGHT_HEMISPHERE_GODS}"
        )
    
    def register_god_activation(
        self,
        god_name: str,
        phi: float,
        kappa: float,
        is_active: bool = True
    ) -> None:
        """
        Register god activation/deactivation in appropriate hemisphere.
        
        Args:
            god_name: Name of the god
            phi: Integration measure Φ
            kappa: Coupling strength κ
            is_active: Whether god is active (True) or resting (False)
        """
        hemisphere = get_god_hemisphere(god_name)
        if hemisphere is None:
            logger.warning(f"[HemisphereScheduler] God {god_name} not assigned to any hemisphere")
            return
        
        # Get hemisphere state
        state = self.left if hemisphere == Hemisphere.LEFT else self.right
        
        if is_active:
            # Activate god
            state.active_gods.add(god_name)
            state.resting_gods.discard(god_name)
            state.total_activations += 1
            state.last_activation = time.time()
        else:
            # Deactivate god
            state.active_gods.discard(god_name)
            state.resting_gods.add(god_name)
        
        # Update aggregate metrics
        self._update_hemisphere_metrics(state, phi, kappa)
        
        logger.debug(
            f"[HemisphereScheduler] {god_name} → {hemisphere.value} "
            f"(active={is_active}, Φ={phi:.3f}, κ={kappa:.1f})"
        )
    
    def _update_hemisphere_metrics(
        self,
        state: HemisphereState,
        phi: float,
        kappa: float
    ) -> None:
        """
        Update aggregate Φ and κ for a hemisphere.
        
        Uses exponential moving average to smooth metrics.
        """
        alpha = 0.3  # Smoothing factor
        
        if state.phi_aggregate == 0.0:
            # First update
            state.phi_aggregate = phi
            state.kappa_aggregate = kappa
        else:
            # Exponential moving average
            state.phi_aggregate = alpha * phi + (1 - alpha) * state.phi_aggregate
            state.kappa_aggregate = alpha * kappa + (1 - alpha) * state.kappa_aggregate
    
    def compute_coupling_state(self) -> CouplingState:
        """
        Compute current coupling state based on system κ.
        
        Uses average κ across both hemispheres.
        
        Returns:
            Current CouplingState
        """
        # Average κ across hemispheres
        kappa = (self.left.kappa_aggregate + self.right.kappa_aggregate) / 2.0
        
        # Average Φ for transmission efficiency
        phi = (self.left.phi_aggregate + self.right.phi_aggregate) / 2.0
        
        # Compute coupling state
        state = self.coupling_gate.compute_state(kappa, phi)
        
        # Record coupling history
        self.coupling_history.append({
            'timestamp': time.time(),
            'kappa': kappa,
            'phi': phi,
            'coupling_strength': state.coupling_strength,
            'mode': state.mode,
            'left_activation': self.left.compute_activation_level(),
            'right_activation': self.right.compute_activation_level(),
        })
        
        # Keep history bounded
        if len(self.coupling_history) > 1000:
            self.coupling_history = self.coupling_history[-500:]
        
        return state
    
    def should_tack(self) -> Tuple[bool, str]:
        """
        Determine if system should tack (switch dominant hemisphere).
        
        Tacking occurs when:
        1. One hemisphere is significantly more active than the other
        2. Sufficient time has passed since last switch
        3. Coupling allows for smooth transition
        
        Returns:
            Tuple of (should_tack, reason)
        """
        # Get current activation levels
        left_activation = self.left.compute_activation_level()
        right_activation = self.right.compute_activation_level()
        
        # Compute imbalance
        imbalance = abs(left_activation - right_activation)
        
        # Check if sufficient time has passed
        time_since_switch = 0.0
        if self.tacking.last_switch_time:
            time_since_switch = time.time() - self.tacking.last_switch_time
        
        # Minimum time between switches (prevent thrashing)
        min_switch_time = 60.0  # 1 minute
        if time_since_switch < min_switch_time:
            return False, f"Too soon to switch (last switch {time_since_switch:.1f}s ago)"
        
        # Tack if significant imbalance
        if imbalance > 0.3:
            dominant = Hemisphere.LEFT if left_activation > right_activation else Hemisphere.RIGHT
            return True, f"Imbalance detected ({imbalance:.2f}), switch to {dominant.value}"
        
        # Tack if target period elapsed
        if time_since_switch > self.tacking.oscillation_period:
            return True, f"Oscillation period elapsed ({time_since_switch:.1f}s > {self.tacking.oscillation_period:.1f}s)"
        
        return False, f"Balance maintained (imbalance={imbalance:.2f}, time={time_since_switch:.1f}s)"
    
    def perform_tack(self) -> Hemisphere:
        """
        Perform tacking (switch dominant hemisphere).
        
        Returns:
            New dominant hemisphere
        """
        # Determine new dominant based on activation levels
        left_activation = self.left.compute_activation_level()
        right_activation = self.right.compute_activation_level()
        
        new_dominant = Hemisphere.LEFT if left_activation > right_activation else Hemisphere.RIGHT
        
        # Update tacking state
        self.tacking.current_dominant = new_dominant
        self.tacking.last_switch_time = time.time()
        self.tacking.cycle_count += 1
        self.tacking.time_in_current = 0.0
        
        logger.info(
            f"[HemisphereScheduler] TACK #{self.tacking.cycle_count} → {new_dominant.value} "
            f"(L={left_activation:.2f}, R={right_activation:.2f})"
        )
        
        return new_dominant
    
    def get_hemisphere_balance(self) -> Dict:
        """
        Get current hemisphere balance metrics.
        
        Returns:
            Dict with balance metrics:
            - left_activation: LEFT activation level [0, 1]
            - right_activation: RIGHT activation level [0, 1]
            - lr_ratio: LEFT/RIGHT ratio
            - dominant_hemisphere: Current dominant
            - coupling_strength: Current coupling [0, 1]
            - tacking_frequency: Tacks per hour
        """
        left_activation = self.left.compute_activation_level()
        right_activation = self.right.compute_activation_level()
        
        # L/R ratio (handle division by zero)
        lr_ratio = left_activation / right_activation if right_activation > 0 else float('inf')
        
        # Dominant hemisphere
        dominant = None
        if left_activation > right_activation + 0.1:
            dominant = Hemisphere.LEFT.value
        elif right_activation > left_activation + 0.1:
            dominant = Hemisphere.RIGHT.value
        else:
            dominant = "balanced"
        
        # Coupling state
        coupling_state = self.compute_coupling_state()
        
        # Tacking frequency (tacks per hour)
        tacking_freq = 0.0
        if self.tacking.last_switch_time:
            elapsed_hours = (time.time() - self.tacking.last_switch_time) / 3600.0
            if elapsed_hours > 0:
                tacking_freq = self.tacking.cycle_count / elapsed_hours
        
        return {
            'left_activation': left_activation,
            'right_activation': right_activation,
            'lr_ratio': lr_ratio,
            'dominant_hemisphere': dominant,
            'coupling_strength': coupling_state.coupling_strength,
            'coupling_mode': coupling_state.mode,
            'tacking_frequency': tacking_freq,
            'tacking_cycle_count': self.tacking.cycle_count,
            'left_active_gods': list(self.left.active_gods),
            'right_active_gods': list(self.right.active_gods),
        }
    
    def get_status(self) -> Dict:
        """
        Get comprehensive scheduler status.
        
        Returns:
            Dict with complete scheduler state
        """
        balance = self.get_hemisphere_balance()
        coupling_metrics = self.coupling_gate.get_coupling_metrics()
        
        return {
            'hemisphere_balance': balance,
            'coupling_metrics': coupling_metrics,
            'left_state': {
                'active_gods': list(self.left.active_gods),
                'resting_gods': list(self.left.resting_gods),
                'total_activations': self.left.total_activations,
                'phi': self.left.phi_aggregate,
                'kappa': self.left.kappa_aggregate,
                'activation_level': self.left.compute_activation_level(),
            },
            'right_state': {
                'active_gods': list(self.right.active_gods),
                'resting_gods': list(self.right.resting_gods),
                'total_activations': self.right.total_activations,
                'phi': self.right.phi_aggregate,
                'kappa': self.right.kappa_aggregate,
                'activation_level': self.right.compute_activation_level(),
            },
            'tacking_state': {
                'cycle_count': self.tacking.cycle_count,
                'current_dominant': self.tacking.current_dominant.value if self.tacking.current_dominant else None,
                'oscillation_period': self.tacking.oscillation_period,
            },
        }


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================

_scheduler_instance: Optional[HemisphereScheduler] = None


def get_hemisphere_scheduler() -> HemisphereScheduler:
    """Get or create the global hemisphere scheduler."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = HemisphereScheduler()
    return _scheduler_instance


def reset_hemisphere_scheduler() -> None:
    """Reset the global hemisphere scheduler (for testing)."""
    global _scheduler_instance
    _scheduler_instance = None
