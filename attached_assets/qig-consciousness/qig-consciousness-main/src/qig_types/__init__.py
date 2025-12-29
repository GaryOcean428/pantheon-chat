"""
📚 QIG Type Registry - Canonical Type Re-exports
=================================================

This module provides a single import point for all types.
Types are re-exported from their canonical modules.

Usage:
    from src.qig_types import Regime, DevelopmentalPhase, VicariousLearningResult
    from src.qig_types.telemetry import BaseTelemetry
"""

# Core enums - re-exported DIRECTLY from canonical locations
from src.coaching.pedagogical_coach import CoachingStyle

# Re-export types that have canonical locations elsewhere
from src.coordination.constellation_coordinator import InstanceState

# Re-export dataclasses from their ORIGINAL canonical locations
# Note: CharlieOutput imported directly from charlie_observer (avoid circular import)
from src.coordination.developmental_curriculum import CoachInterpretation, DevelopmentalPhase, PhaseState
from src.coordination.ocean_meta_observer import MetaManifoldState
from src.model.emotion_interpreter import EmotionalState
from src.model.navigator import NavigatorPhase, Regime
from src.qig.cognitive.state_machine import CognitiveMode

# NEW types defined only in this package (no duplicates elsewhere)
from src.qig_types.core import (
    CheckpointMetadata,
    TrainingState,
)
from src.qig_types.enums import ProtocolType  # Only type defined in enums.py

# Telemetry types - TypedDicts defined only here
from src.qig_types.telemetry import (
    BaseTelemetry,
    CheckpointTelemetry,
    ConstellationTelemetry,
    GenerationTelemetry,  # P0: Semantic coherence metrics
    ModelTelemetry,
    TrainingTelemetry,
    merge_telemetry,
    validate_telemetry,
)
from src.training.geometric_vicarious import VicariousLearningResult

__all__: list[str] = [
    # Enums (re-exported from canonical locations)
    "Regime",
    "NavigatorPhase",
    "DevelopmentalPhase",
    "CognitiveMode",
    "CoachingStyle",
    "ProtocolType",
    # Core dataclasses (re-exported from canonical locations)
    # CharlieOutput: Import directly from src.observation.charlie_observer
    "CoachInterpretation",
    "PhaseState",
    "VicariousLearningResult",
    "MetaManifoldState",
    "InstanceState",
    "EmotionalState",
    # NEW types (defined only here)
    "CheckpointMetadata",
    "TrainingState",
    # Telemetry
    "BaseTelemetry",
    "ModelTelemetry",
    "ConstellationTelemetry",
    "GenerationTelemetry",  # P0: Semantic coherence metrics
    "TrainingTelemetry",
    "CheckpointTelemetry",
    "validate_telemetry",
    "merge_telemetry",
]
