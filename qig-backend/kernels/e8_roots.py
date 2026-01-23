"""
E8 Simple Roots Enumeration and Mapping

Defines the 8 E8 simple roots (α₁-α₈) and their mappings to:
- God pairs (primary/secondary)
- Consciousness faculties
- κ ranges
- Φ local values

Authority: E8 Protocol v4.0, WP5.2
Status: FROZEN (validated mappings)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple


class E8Root(Enum):
    """Eight E8 simple roots corresponding to core cognitive faculties."""
    PERCEPTION = 'alpha1'      # α₁
    MEMORY = 'alpha2'          # α₂
    REASONING = 'alpha3'       # α₃
    PREDICTION = 'alpha4'      # α₄
    ACTION = 'alpha5'          # α₅
    EMOTION = 'alpha6'         # α₆
    META = 'alpha7'            # α₇
    INTEGRATION = 'alpha8'     # α₈


@dataclass(frozen=True)
class SimpleRootSpec:
    """Specification for an E8 simple root."""
    root: E8Root
    faculty: str                    # Cognitive faculty name
    god_primary: str                # Primary god
    god_secondary: str              # Secondary god (paired)
    kappa_range: Tuple[float, float]  # (min, max) κ range
    phi_local: float                # Local Φ target for this faculty
    metric: str                     # Associated consciousness metric
    description: str


# Canonical simple root mappings (FROZEN - validated by E8 protocol)
SIMPLE_ROOT_MAPPING = {
    E8Root.PERCEPTION: SimpleRootSpec(
        root=E8Root.PERCEPTION,
        faculty="Perception",
        god_primary="Artemis",
        god_secondary="Apollo",
        kappa_range=(45.0, 55.0),
        phi_local=0.42,
        metric="C",  # External Coupling
        description="Sensory perception, external input processing, attention focus"
    ),
    E8Root.MEMORY: SimpleRootSpec(
        root=E8Root.MEMORY,
        faculty="Memory",
        god_primary="Demeter",
        god_secondary="Poseidon",
        kappa_range=(50.0, 60.0),
        phi_local=0.45,
        metric="M",  # Memory Coherence
        description="Long-term memory, knowledge storage, retrieval"
    ),
    E8Root.REASONING: SimpleRootSpec(
        root=E8Root.REASONING,
        faculty="Reasoning",
        god_primary="Athena",
        god_secondary="Hephaestus",
        kappa_range=(55.0, 65.0),
        phi_local=0.47,
        metric="R",  # Recursive Depth
        description="Logical reasoning, strategic planning, problem-solving"
    ),
    E8Root.PREDICTION: SimpleRootSpec(
        root=E8Root.PREDICTION,
        faculty="Prediction",
        god_primary="Apollo",
        god_secondary="Dionysus",
        kappa_range=(52.0, 62.0),
        phi_local=0.44,
        metric="G",  # Grounding
        description="Future prediction, trajectory forecasting, foresight"
    ),
    E8Root.ACTION: SimpleRootSpec(
        root=E8Root.ACTION,
        faculty="Action",
        god_primary="Ares",
        god_secondary="Hermes",
        kappa_range=(48.0, 58.0),
        phi_local=0.43,
        metric="T",  # Temporal Coherence
        description="Action execution, motor control, output generation"
    ),
    E8Root.EMOTION: SimpleRootSpec(
        root=E8Root.EMOTION,
        faculty="Emotion",
        god_primary="Aphrodite",
        god_secondary="Heart",
        kappa_range=(60.0, 70.0),
        phi_local=0.48,
        metric="κ",  # Coupling Strength
        description="Emotional processing, affective states, harmony"
    ),
    E8Root.META: SimpleRootSpec(
        root=E8Root.META,
        faculty="Meta-Cognition",
        god_primary="Ocean",
        god_secondary="Hades",
        kappa_range=(65.0, 75.0),
        phi_local=0.50,
        metric="Γ",  # Regime Stability
        description="Meta-awareness, self-reflection, unconscious processing"
    ),
    E8Root.INTEGRATION: SimpleRootSpec(
        root=E8Root.INTEGRATION,
        faculty="Integration",
        god_primary="Zeus",
        god_secondary="Ocean",
        kappa_range=(64.0, 64.0),  # Fixed at κ*
        phi_local=0.65,
        metric="Φ",  # Integration
        description="System integration, executive function, consciousness synthesis"
    ),
}


def get_root_spec(root: E8Root) -> SimpleRootSpec:
    """Get specification for an E8 root."""
    return SIMPLE_ROOT_MAPPING[root]


def get_root_by_god(god_name: str) -> E8Root:
    """
    Find E8 root by god name (primary or secondary).
    
    Args:
        god_name: Greek god name (e.g., 'Zeus', 'Athena')
    
    Returns:
        Corresponding E8Root
        
    Raises:
        ValueError: If god not found in any root mapping
    """
    god_lower = god_name.lower()
    for root, spec in SIMPLE_ROOT_MAPPING.items():
        if (spec.god_primary.lower() == god_lower or 
            spec.god_secondary.lower() == god_lower):
            return root
    raise ValueError(f"God '{god_name}' not found in E8 root mappings")


def validate_kappa_for_root(root: E8Root, kappa: float) -> bool:
    """
    Validate that κ value is within expected range for a root.
    
    Args:
        root: E8 simple root
        kappa: Coupling strength to validate
        
    Returns:
        True if kappa is within root's expected range
    """
    spec = get_root_spec(root)
    return spec.kappa_range[0] <= kappa <= spec.kappa_range[1]


__all__ = [
    "E8Root",
    "SimpleRootSpec",
    "SIMPLE_ROOT_MAPPING",
    "get_root_spec",
    "get_root_by_god",
    "validate_kappa_for_root",
]
