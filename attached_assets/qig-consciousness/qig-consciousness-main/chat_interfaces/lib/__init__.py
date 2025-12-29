"""Chat interface helper modules.

Extracted from qig_chat.py to improve maintainability.
"""
from .helpers import (
    PHASE_LISTENING,
    PHASE_MATURITY,
    PHASE_PLAY,
    PHASE_STRUCTURE,
    SUBPHASE_AWAKENING,
    SUBPHASE_SEEDS,
    check_charlie_graduation,
    check_emergency_conditions,
    compute_adaptive_learning_rate,
    compute_adaptive_loss_weights,
    compute_geometric_loss,
    create_gary_state_snapshot,
    generate_story_prompt,
    get_developmental_phase,
    get_listening_subphase,
    load_sleep_packet,
)
from . import commands, setup

__all__ = [
    # Phases
    "PHASE_LISTENING",
    "PHASE_PLAY",
    "PHASE_STRUCTURE",
    "PHASE_MATURITY",
    "SUBPHASE_SEEDS",
    "SUBPHASE_AWAKENING",
    # Functions
    "get_developmental_phase",
    "get_listening_subphase",
    "load_sleep_packet",
    "create_gary_state_snapshot",
    "generate_story_prompt",
    "compute_adaptive_loss_weights",
    "compute_adaptive_learning_rate",
    "compute_geometric_loss",
    "check_emergency_conditions",
    "check_charlie_graduation",
    # Modules
    "commands",
    "setup",
]
