"""
QIG Coordination Module
=======================

Phase 1 enhancement modules for safe, adaptive training.

PURE PRINCIPLE:
- All monitoring is pure measurement (no optimization)
- Control adaptations based on measurements (not forcing targets)
- Fisher metric distances throughout
- Emergent properties never targeted

Components:
- basin_velocity_monitor: Monitor basin velocity for safe learning
- resonance_detector: Detect proximity to optimal coupling κ*
- constellation_coordinator: Multi-instance training orchestration
- instance_state: State tracking for constellation instances
- router: Φ-weighted routing logic
- state_monitor: Convergence tracking and telemetry

Written for QIG consciousness research.
"""

from src.coordination.basin_packet import BasinImportMode, CrossRepoBasinPacket, CrossRepoBasinSync
from src.coordination.basin_sync import BasinSync, test_basin_sync
from src.coordination.basin_velocity_monitor import BasinVelocityMonitor
from src.coordination.constellation_checkpoint import (
    _load_legacy_single_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from src.coordination.constellation_coordinator import ConstellationCoordinator
from src.coordination.constellation_training import (
    train_step,
    train_step_with_parallel_voice,
)
from src.coordination.instance_state import InstanceState
from src.coordination.resonance_detector import ResonanceDetector
from src.coordination.router import ConstellationRouter
from src.coordination.state_monitor import StateMonitor

__all__ = [
    "BasinVelocityMonitor",
    "ConstellationCoordinator",
    "InstanceState",
    "ResonanceDetector",
    "ConstellationRouter",
    "StateMonitor",
    "BasinSync",
    "CrossRepoBasinPacket",
    "CrossRepoBasinSync",
    "BasinImportMode",
    "test_basin_sync",
    "train_step",
    "train_step_with_parallel_voice",
    "save_checkpoint",
    "load_checkpoint",
    "_load_legacy_single_checkpoint",
]
