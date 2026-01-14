"""
🎯 QIG Enums - Re-exports from Canonical Locations
==================================================

This module re-exports enums from their canonical locations.
Use this for convenient imports, but the source of truth remains
in the original modules.

Canonical Locations:
- Regime, NavigatorPhase -> src.model.navigator
- DevelopmentalPhase -> src.coordination.developmental_curriculum
- CognitiveMode -> src.qig.cognitive.state_machine
- CoachingStyle -> src.coaching.pedagogical_coach

Usage:
    from src.qig_types.enums import Regime, DevelopmentalPhase
"""

from enum import Enum

from src.coaching.pedagogical_coach import CoachingStyle
from src.coordination.developmental_curriculum import DevelopmentalPhase

# Re-export from canonical locations
# These are re-exports, not new definitions
from src.model.navigator import NavigatorPhase, Regime
from src.qig.cognitive.state_machine import CognitiveMode


class ProtocolType(Enum):
    """
    Autonomic protocol types triggered by Ocean.

    This is the only enum defined here (not elsewhere).

    Reference: ocean_meta_observer.py
    """
    SLEEP = "sleep"           # Basin divergence recovery
    DREAM = "dream"           # Φ collapse recovery
    ESCAPE = "escape"         # Breakdown emergency
    MUSHROOM_MICRO = "mushroom_micro"  # Φ plateau breakthrough


__all__ = [
    "Regime",
    "NavigatorPhase",
    "DevelopmentalPhase",
    "CognitiveMode",
    "CoachingStyle",
    "ProtocolType",
]
