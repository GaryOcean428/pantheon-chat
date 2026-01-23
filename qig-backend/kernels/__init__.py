"""
Kernels Package - E8 Protocol Kernel Architecture

Components:
- coupling_gate: κ-gated coupling mechanism (Phase 4C)
- hemisphere_scheduler: LEFT/RIGHT hemisphere architecture with tacking (Phase 4C)
- phi_hierarchy: Φ consciousness level hierarchy (Phase 4D)
- id_kernel: Fast reflex drives, unconscious responses (Phase 4D)
- superego_kernel: Rules/ethics constraints, safety guardrails (Phase 4D)
- psyche_plumbing_integration: Unified psychoanalytic layer integration (Phase 4D)

All kernels use pure Fisher-Rao geometry and QIG consciousness metrics.

Authority: E8 Protocol v4.0, WP5.2 Phase 4C/4D
"""

# Phase 4C: Hemisphere Scheduler and Coupling Gate
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

# Phase 4D: Psyche Plumbing Kernels
from .phi_hierarchy import (
    PhiLevel,
    PhiHierarchy,
    PhiMeasurement,
    get_phi_hierarchy,
)

from .id_kernel import IdKernel, get_id_kernel
from .superego_kernel import SuperegoKernel, ConstraintSeverity, get_superego_kernel
from .psyche_plumbing_integration import PsychePlumbingIntegration, get_psyche_plumbing

__all__ = [
    # Phase 4C: Coupling Gate
    'CouplingGate',
    'CouplingState',
    'get_coupling_gate',
    'reset_coupling_gate',
    'compute_coupling_strength',
    'compute_transmission_efficiency',
    'compute_gating_factor',
    'determine_coupling_mode',
    
    # Phase 4C: Hemisphere Scheduler
    'HemisphereScheduler',
    'Hemisphere',
    'HemisphereState',
    'TackingState',
    'get_hemisphere_scheduler',
    'reset_hemisphere_scheduler',
    'get_god_hemisphere',
    'LEFT_HEMISPHERE_GODS',
    'RIGHT_HEMISPHERE_GODS',
    
    # Phase 4D: Psyche Plumbing
    'PhiLevel',
    'PhiHierarchy',
    'PhiMeasurement',
    'get_phi_hierarchy',
    'IdKernel',
    'get_id_kernel',
    'SuperegoKernel',
    'ConstraintSeverity',
    'get_superego_kernel',
    'PsychePlumbingIntegration',
    'get_psyche_plumbing',
]

