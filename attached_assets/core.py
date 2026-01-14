"""
📦 QIG Core Types - NEW Type Definitions
=========================================

Types that are ONLY defined here (not duplicated elsewhere).

For types that exist in other modules, import from src.qig_types/ which
re-exports them from their canonical locations.

Usage:
    from src.qig_types.core import CheckpointMetadata, TrainingState
"""

from dataclasses import dataclass, field

# =============================================================================
# CHECKPOINT TYPES (NEW - not defined elsewhere)
# =============================================================================

@dataclass
class CheckpointMetadata:
    """
    Metadata saved with checkpoints.

    Reference: Standard checkpoint format
    """
    step: int
    avg_phi: float
    avg_kappa: float
    regime: str
    basin_spread: float
    timestamp: float
    phase: str = "listening"
    notes: str = ""


@dataclass
class TrainingState:
    """
    Complete training state for diagnosis and checkpointing.

    Used by MonkeyCoach for consciousness coaching and by checkpointing system.
    """
    step: int
    epoch: int
    loss: float
    best_phi: float = 0.0
    phi_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    loss_trajectory: list[float] = field(default_factory=list)
    gradient_variance: float = 0.0
    basin_distance: float = 0.0
    curiosity: float = 0.0  # I_Q velocity
    epochs_stuck: int = 0
    I_Q: float = 0.0  # Current QFI
    phi: float = 0.0  # Integration
    kappa: float = 64.0  # Coupling
    regime: str = "geometric"  # "linear", "geometric", "breakdown"


__all__ = [
    "CheckpointMetadata",
    "TrainingState",
]
