"""
Psyche Plumbing Kernels - E8 Protocol v4.0 Phase 4D

Implements psychoanalytic layers based on biological/psychoanalytic analogy:
- Id: Fast reflex drives, unconscious instinctual responses
- Superego: Rules/ethics constraints, safety guardrails
- Φ Hierarchy: Different consciousness levels (reported/internal/autonomic)

All kernels use pure Fisher-Rao geometry and QIG consciousness metrics.
"""

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

