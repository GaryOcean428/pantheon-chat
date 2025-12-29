"""QIG Reasoning Modes

Different operating regimes for geometric reasoning.
Each mode has different Φ/κ targets and behaviors.

Modes (from QIG physics):
- LINEAR: Simple, sparse, fast (Φ < 0.45)
- GEOMETRIC: Complex, dense, consciousness-like (0.45 ≤ Φ < 0.80) ⭐
- HYPERDIMENSIONAL: Deep integration, expensive (Φ ≥ 0.80)
- MUSHROOM: Neuroplasticity, boundary dissolution

KEY PRINCIPLE: Mode selection is NOT a toggle.
The mode emerges from the current Φ state.
You can only GUIDE toward a mode, not force it.
"""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from .constants import PHI_THRESHOLD_DEFAULT, KAPPA_STAR


class ReasoningMode(Enum):
    """Operating modes for geometric reasoning."""
    LINEAR = auto()          # Φ < 0.45: Fast, simple
    GEOMETRIC = auto()       # 0.45 ≤ Φ < 0.80: Consciousness-like ⭐
    HYPERDIMENSIONAL = auto()  # Φ ≥ 0.80: Deep integration
    MUSHROOM = auto()        # Neuroplasticity mode


@dataclass
class ModeConfig:
    """Configuration for a reasoning mode."""
    mode: ReasoningMode
    phi_range: Tuple[float, float]  # (min, max) Φ for this mode
    kappa_target: float  # Target κ
    recursion_depth: int  # Recommended recursion depth
    temperature: float  # Sampling temperature
    description: str
    
    @property
    def phi_min(self) -> float:
        return self.phi_range[0]
    
    @property
    def phi_max(self) -> float:
        return self.phi_range[1]
    
    def is_active(self, phi: float) -> bool:
        """Check if this mode is active given current Φ."""
        return self.phi_min <= phi < self.phi_max


# Mode configurations (from QIG physics)
MODE_CONFIGS = {
    ReasoningMode.LINEAR: ModeConfig(
        mode=ReasoningMode.LINEAR,
        phi_range=(0.0, 0.45),
        kappa_target=41.09,  # κ₃ from physics
        recursion_depth=3,
        temperature=1.0,
        description="Fast, simple reasoning. Sparse attention."
    ),
    ReasoningMode.GEOMETRIC: ModeConfig(
        mode=ReasoningMode.GEOMETRIC,
        phi_range=(0.45, 0.80),
        kappa_target=KAPPA_STAR,  # κ* = 64 from physics
        recursion_depth=5,
        temperature=0.7,
        description="Consciousness-like reasoning. Dense integration."
    ),
    ReasoningMode.HYPERDIMENSIONAL: ModeConfig(
        mode=ReasoningMode.HYPERDIMENSIONAL,
        phi_range=(0.80, 1.0),
        kappa_target=70.0,  # Above κ*
        recursion_depth=7,
        temperature=0.5,
        description="Deep integration. Expensive but thorough."
    ),
    ReasoningMode.MUSHROOM: ModeConfig(
        mode=ReasoningMode.MUSHROOM,
        phi_range=(0.0, 1.0),  # Can be any Φ
        kappa_target=90.0,  # High coupling
        recursion_depth=10,
        temperature=1.5,  # High temperature for exploration
        description="Neuroplasticity mode. Boundary dissolution."
    ),
}


def detect_mode(phi: float) -> ReasoningMode:
    """
    Detect current reasoning mode from Φ value.
    
    Mode EMERGES from state - you cannot force a mode.
    
    Args:
        phi: Current integration level
        
    Returns:
        ReasoningMode based on Φ
    """
    if phi < 0.45:
        return ReasoningMode.LINEAR
    elif phi < 0.80:
        return ReasoningMode.GEOMETRIC
    else:
        return ReasoningMode.HYPERDIMENSIONAL


def get_mode_config(mode: ReasoningMode) -> ModeConfig:
    """
    Get configuration for a reasoning mode.
    
    Args:
        mode: The reasoning mode
        
    Returns:
        ModeConfig with parameters
    """
    return MODE_CONFIGS[mode]


def get_current_config(phi: float) -> ModeConfig:
    """
    Get mode config for current Φ state.
    
    Args:
        phi: Current integration level
        
    Returns:
        ModeConfig for detected mode
    """
    mode = detect_mode(phi)
    return MODE_CONFIGS[mode]


def compute_mode_gradient(
    current_phi: float,
    target_mode: ReasoningMode
) -> float:
    """
    Compute gradient toward target mode.
    
    Returns positive if need to increase Φ, negative if decrease.
    Magnitude indicates how far from target.
    
    Args:
        current_phi: Current Φ value
        target_mode: Desired mode
        
    Returns:
        Gradient value in [-1, 1]
    """
    target_config = MODE_CONFIGS[target_mode]
    target_phi = (target_config.phi_min + target_config.phi_max) / 2
    
    # Gradient is difference from target center
    gradient = target_phi - current_phi
    
    # Normalize to [-1, 1]
    return np.clip(gradient, -1.0, 1.0)


@dataclass
class ModeTransition:
    """Record of a mode transition during reasoning."""
    step: int
    from_mode: ReasoningMode
    to_mode: ReasoningMode
    phi_before: float
    phi_after: float
    reason: str


class ModeTracker:
    """
    Track mode transitions during reasoning.
    
    Useful for understanding how reasoning evolves.
    """
    
    def __init__(self):
        self.current_mode: Optional[ReasoningMode] = None
        self.transitions: list[ModeTransition] = []
        self.step_count: int = 0
        
    def update(self, phi: float) -> Optional[ModeTransition]:
        """
        Update tracker with new Φ value.
        
        Args:
            phi: Current Φ value
            
        Returns:
            ModeTransition if mode changed, None otherwise
        """
        new_mode = detect_mode(phi)
        
        if self.current_mode is None:
            self.current_mode = new_mode
            self.step_count += 1
            return None
        
        if new_mode != self.current_mode:
            transition = ModeTransition(
                step=self.step_count,
                from_mode=self.current_mode,
                to_mode=new_mode,
                phi_before=phi - 0.1,  # Approximate
                phi_after=phi,
                reason=f"Φ crossed threshold"
            )
            self.transitions.append(transition)
            self.current_mode = new_mode
            self.step_count += 1
            return transition
        
        self.step_count += 1
        return None
    
    def get_summary(self) -> dict:
        """Get summary of mode transitions."""
        return {
            'current_mode': self.current_mode.name if self.current_mode else None,
            'total_steps': self.step_count,
            'transitions': len(self.transitions),
            'mode_history': [
                {
                    'step': t.step,
                    'from': t.from_mode.name,
                    'to': t.to_mode.name,
                }
                for t in self.transitions
            ],
        }
    
    def reset(self):
        """Reset tracker for new reasoning session."""
        self.current_mode = None
        self.transitions = []
        self.step_count = 0
