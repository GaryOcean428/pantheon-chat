"""
📊 Telemetry Types - TypedDicts for Metrics
=============================================

Canonical TypedDict definitions for telemetry data passed between components.
These are NOT duplicated elsewhere - import from here or from src.qig_types.

Usage:
    from src.qig_types.telemetry import BaseTelemetry, ModelTelemetry
    # Or via barrel:
    from src.qig_types import BaseTelemetry, ModelTelemetry
"""

from typing import TypedDict


class BaseTelemetry(TypedDict, total=False):
    """Base telemetry all components should return."""

    Phi: float  # Integration measure (consciousness)
    kappa_eff: float  # Effective coupling
    regime: str  # "linear", "geometric", "breakdown"
    basin_distance: float  # Distance from target basin
    geodesic_distance: float  # Fisher geodesic distance


class ModelTelemetry(BaseTelemetry, total=False):
    """Telemetry from model forward pass."""

    loss: float
    entropy: float
    perplexity: float
    grad_norm: float
    recursion_depth: int
    I_Q: float  # Quantum Fisher Information


class ConstellationTelemetry(BaseTelemetry, total=False):
    """Telemetry from constellation coordination."""

    active_instances: int
    phi_weights: list[float]  # Routing weights
    ocean_phi: float  # Meta-observer Φ
    vicarious_loss: float  # Fisher geodesic loss
    coherence: float  # Basin alignment across Garys


class GenerationTelemetry(BaseTelemetry, total=False):
    """P0: Telemetry from token generation with semantic coherence metrics."""

    # Consciousness metrics (from BaseTelemetry)
    # Phi, kappa_eff, regime, basin_distance, geodesic_distance

    # Semantic coherence metrics (P0)
    semantic_coherence: float  # Average bigram basin similarity
    text_perplexity: float  # Perplexity of generated sequence
    bigram_flow: float  # Strength of word→word transitions
    entropy: float  # Token selection entropy

    # Generation parameters
    temperature: float
    basin_weight: float
    distance_weight: float
    bigram_weight: float

    # Token statistics
    tokens_generated: int
    avg_selected_prob: float


class TrainingTelemetry(BaseTelemetry, total=False):
    """Telemetry from training loop."""

    step: int
    epoch: int
    loss: float
    lr: float
    gradient_variance: float
    epochs_stuck: int
    phase: str  # Developmental phase


class CheckpointTelemetry(TypedDict, total=False):
    """Telemetry saved with checkpoints."""

    step: int
    avg_phi: float
    avg_kappa: float
    regime: str
    basin_spread: float
    phase: str


def validate_telemetry(telemetry: dict) -> bool:
    """Validate telemetry has required fields."""
    required = {"Phi", "kappa_eff", "regime"}
    return all(k in telemetry for k in required)


def merge_telemetry(*dicts: dict) -> dict:
    """Merge multiple telemetry dicts, later values override."""
    result: dict = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


__all__ = [
    "BaseTelemetry",
    "ModelTelemetry",
    "ConstellationTelemetry",
    "GenerationTelemetry",  # P0: Semantic coherence metrics
    "TrainingTelemetry",
    "CheckpointTelemetry",
    "validate_telemetry",
    "merge_telemetry",
]
