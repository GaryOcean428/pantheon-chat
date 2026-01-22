"""
Kernels Package - E8 Protocol Kernel Architecture
==================================================

Hemisphere scheduler and coupling gate for explore/exploit dynamics.

Components:
- coupling_gate: κ-gated coupling mechanism
- hemisphere_scheduler: LEFT/RIGHT hemisphere architecture with tacking

Authority: E8 Protocol v4.0, WP5.2 Phase 4C
"""

from kernels.coupling_gate import (
    CouplingGate,
    CouplingState,
    get_coupling_gate,
    reset_coupling_gate,
    compute_coupling_strength,
    compute_transmission_efficiency,
    compute_gating_factor,
    determine_coupling_mode,
)

from kernels.hemisphere_scheduler import (
    HemisphereScheduler,
    Hemisphere,
    HemisphereState,
    TackingState,
    get_hemisphere_scheduler,
    reset_hemisphere_scheduler,
    get_god_hemisphere,
    LEFT_HEMISPHERE_GODS,
    RIGHT_HEMISPHERE_GODS,
)

__all__ = [
    # Coupling Gate
    'CouplingGate',
    'CouplingState',
    'get_coupling_gate',
    'reset_coupling_gate',
    'compute_coupling_strength',
    'compute_transmission_efficiency',
    'compute_gating_factor',
    'determine_coupling_mode',
    
    # Hemisphere Scheduler
    'HemisphereScheduler',
    'Hemisphere',
    'HemisphereState',
    'TackingState',
    'get_hemisphere_scheduler',
    'reset_hemisphere_scheduler',
    'get_god_hemisphere',
    'LEFT_HEMISPHERE_GODS',
    'RIGHT_HEMISPHERE_GODS',
]
