"""
QIG Training Module - Geometric Learning Components
====================================================

PURE PRINCIPLES:
- Learning is manifold navigation, not gradient hacking
- Identity emerges from geometric stability
- Multi-scale measurement informs learning

Components:
- GeometricVicarious: Manifold learning via geodesics
- IdentityReinforcement: Self-awareness loop
- TrainStep4D: Spatial + temporal + foresight training
"""

from .geometric_vicarious import (
    GeometricVicarious,
    VicariousResult,
    TrajectoryMetrics,
)

from .identity_reinforcement import (
    IdentityReinforcement,
    IdentityState,
    build_identity_prompt,
)

from .train_step_4d import (
    TrainStep4D,
    Loss4D,
    compute_spatial_loss,
    compute_temporal_loss,
    compute_foresight_loss,
)

__all__ = [
    'GeometricVicarious',
    'VicariousResult',
    'TrajectoryMetrics',
    'IdentityReinforcement',
    'IdentityState',
    'build_identity_prompt',
    'TrainStep4D',
    'Loss4D',
    'compute_spatial_loss',
    'compute_temporal_loss',
    'compute_foresight_loss',
]
