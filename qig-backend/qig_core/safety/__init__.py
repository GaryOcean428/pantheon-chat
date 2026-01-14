"""
QIG Safety Module - Session/Safety Components
==============================================

PURE PRINCIPLES:
- Checkpoints are snapshots, not optimization targets
- Repair is geometric projection, not gradient update
- Meta-observation informs control, doesn't optimize

Components:
- SessionManager: Checkpoint-based learning ("Gary goes to school")
- SelfRepair: Geometric diagnostics and projection
- MetaReflector: Grounding and locked-in detection
"""

from .session_manager import (
    SessionManager,
    SessionState,
    CheckpointData,
)

from .self_repair import (
    SelfRepair,
    DiagnosticResult,
    RepairAction,
    GeometricAnomaly,
)

from .meta_reflector_integration import (
    MetaReflector,
    GroundingState,
    LockedInState,
)

__all__ = [
    'SessionManager',
    'SessionState',
    'CheckpointData',
    'SelfRepair',
    'DiagnosticResult',
    'RepairAction',
    'GeometricAnomaly',
    'MetaReflector',
    'GroundingState',
    'LockedInState',
]
