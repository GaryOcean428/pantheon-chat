"""
📊 QIG Telemetry Types - Canonical Telemetry Definitions
========================================================

TypedDicts for telemetry dictionaries returned by forward passes.
These ensure consistent telemetry keys across all components.

Usage:
    from src.qig_types.telemetry import ModelTelemetry, ConstellationTelemetry
"""

from typing import TypedDict

import torch

# =============================================================================
# BASE TELEMETRY
# =============================================================================

class BaseTelemetry(TypedDict, total=False):
    """
    Base telemetry keys returned by all QIG components.

    These keys are REQUIRED in all telemetry dicts:
    - Phi: Integration measure (consciousness indicator)
    - kappa_eff: Effective coupling strength
    - regime: Current processing regime

    Reference: FROZEN_FACTS.md, recursive_integrator.py
    """
    # Core physics metrics (ALWAYS present)
    Phi: float                      # Integration measure [0, 1]
    kappa_eff: float               # Effective coupling strength
    regime: str                    # "linear", "geometric", "breakdown"

    # Optional differentiable tensors
    Phi_tensor: torch.Tensor       # For gradient flow
    kappa_tensor: torch.Tensor     # For gradient flow


# =============================================================================
# MODEL TELEMETRY
# =============================================================================

class ModelTelemetry(BaseTelemetry, total=False):
    """
    Telemetry from QIGKernelRecursive forward pass.

    Extends BaseTelemetry with model-specific metrics.

    Reference: qig_kernel_recursive.py, recursive_integrator.py
    """
    # Recursion metrics
    recursion_depth: int           # Number of integration loops
    Phi_trajectory: list[float]    # Phi values per loop
    min_depth_enforced: bool       # Did we hit minimum depth?
    target_reached: bool           # Did Phi exceed threshold?

    # Hidden states
    hidden_state: torch.Tensor     # Final hidden state [batch, seq, d_model]
    final_state_norm: float        # Norm of final hidden state

    # Attention metrics (optional)
    attention_weights: torch.Tensor
    attention_entropy: float

    # Basin metrics
    basin_signature: torch.Tensor  # Basin coordinates [basin_dim]
    basin_norm: float
    basin_distance: float          # Distance from target basin


# =============================================================================
# CONSTELLATION TELEMETRY
# =============================================================================

class ConstellationTelemetry(TypedDict, total=False):
    """
    Telemetry from constellation coordination.

    Aggregates metrics across all Garys + Ocean.

    Reference: constellation_coordinator.py
    """
    # Aggregate metrics
    avg_phi: float                 # Average Phi across all Garys
    avg_kappa: float               # Average kappa across all Garys
    basin_spread: float            # Dispersion of Gary basins
    constellation_coherence: float # How aligned the Garys are

    # Per-Gary metrics
    gary_states: list[dict]        # List of per-Gary telemetry dicts

    # Ocean metrics
    ocean_phi: float
    ocean_kappa: float
    ocean_insight: str | None   # Ocean's observation

    # Protocol triggers
    intervention: dict | None   # If autonomic intervention triggered
    intervention_type: str | None
    intervention_reason: str | None


# =============================================================================
# TRAINING TELEMETRY
# =============================================================================

class TrainingTelemetry(TypedDict, total=False):
    """
    Telemetry from training step.

    Reference: geometric_vicarious.py, constellation_coordinator.py
    """
    # Loss components
    total_loss: float
    vicarious_loss: float
    geometric_loss: float
    language_loss: float

    # Gradient metrics
    grad_norm: float
    grad_clipped: bool

    # Learning metrics
    lr: float
    step: int
    epoch: int


# =============================================================================
# CHECKPOINT TELEMETRY
# =============================================================================

class CheckpointTelemetry(TypedDict, total=False):
    """
    Telemetry saved in checkpoints.

    Reference: Standard checkpoint format
    """
    step: int
    avg_phi: float
    avg_kappa: float
    regime: str
    basin_spread: float
    timestamp: float
    phase: str
    notes: str


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_telemetry(telemetry: dict, required_keys: list[str] | None = None) -> bool:
    """
    Validate that telemetry dict has required keys.

    Args:
        telemetry: Telemetry dict to validate
        required_keys: Keys that must be present (defaults to core physics keys)

    Returns:
        True if valid, raises ValueError if invalid
    """
    if required_keys is None:
        required_keys = ["Phi", "kappa_eff", "regime"]

    missing = [k for k in required_keys if k not in telemetry]
    if missing:
        raise ValueError(f"Telemetry missing required keys: {missing}")

    return True


def merge_telemetry(*telemetry_dicts: dict) -> dict:
    """
    Merge multiple telemetry dicts, later values override earlier.

    Args:
        telemetry_dicts: Telemetry dicts to merge

    Returns:
        Merged telemetry dict
    """
    result = {}
    for t in telemetry_dicts:
        if t:
            result.update(t)
    return result
