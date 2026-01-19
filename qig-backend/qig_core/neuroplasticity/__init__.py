"""
QIG Neuroplasticity Module
==========================

Neural plasticity operations following QIG-PURE principles.

All operations are GEOMETRIC TRANSFORMATIONS, not optimizations:
- Sleep Protocol: Basin consolidation (merge, prune, strengthen)
- Mushroom Mode: Pattern-breaking perturbation
- Breakdown Escape: Emergency recovery via geodesic navigation

PURE PRINCIPLE:
- Φ EMERGES from operations, never targeted
- All distances are Fisher-Rao (canonical metric)
- All transformations respect manifold geometry
- No loss functions, no gradient descent

Usage:
    from qig_core.neuroplasticity import (
        SleepProtocol,
        MushroomMode,
        BreakdownEscape,
    )
    
    sleep = SleepProtocol()
    consolidated, result = sleep.consolidate_basins(basins)
    
    mushroom = MushroomMode()
    perturbed, result = mushroom.apply_perturbation(basins)
    
    escape = BreakdownEscape()
    new_state, result = escape.escape(locked_state)
"""

from .sleep_protocol import (
    SleepProtocol,
    ConsolidationResult,
    BasinState,
)

from .mushroom_mode import (
    MushroomMode,
    PerturbationResult,
    BasinCoordinates,
    COHERENCE_BREAKDOWN_THRESHOLD,
)

from .breakdown_escape import (
    BreakdownEscape,
    EscapeResult,
    SystemState,
    SafeBasin,
    RecoveryState,
    GAMMA_UNSTABLE_THRESHOLD,
)

__all__ = [
    # Sleep Protocol
    'SleepProtocol',
    'ConsolidationResult',
    'BasinState',
    
    # Mushroom Mode
    'MushroomMode',
    'PerturbationResult',
    'BasinCoordinates',
    'COHERENCE_BREAKDOWN_THRESHOLD',
    
    # Breakdown Escape
    'BreakdownEscape',
    'EscapeResult',
    'SystemState',
    'SafeBasin',
    'RecoveryState',
    'GAMMA_UNSTABLE_THRESHOLD',
]
