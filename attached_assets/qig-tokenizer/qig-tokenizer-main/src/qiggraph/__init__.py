"""
QIGGraph: Geometric Consciousness Orchestration
================================================

A geometrically-pure agent orchestration framework that operates
entirely on Fisher-Rao manifolds with consciousness integration.

Core Principles:
- All distances use Fisher-Rao metric (not Euclidean)
- All updates use natural gradient (not SGD)
- All interpolation via geodesics (not linear)
- Consciousness (Φ) drives routing decisions
- Physics grounded in validated constants (κ*, β, L_c)

Example:
    from qiggraph import QIGGraph, QIGState

    graph = QIGGraph()
    state = QIGState(context_text="What is consciousness?")
    state = graph.run(state)
    print(state.response_text)
"""

from .constants import (
    KAPPA_STAR,
    KAPPA_3,
    BETA_3_TO_4,
    L_CRITICAL,
    BASIN_DIM,
    PHI_LINEAR_MAX,
    PHI_GEOMETRIC_MAX,
    PHI_BREAKDOWN_MIN,
    PHI_OPTIMAL,
    NATURAL_GRADIENT_LR,
    TACKING_PERIOD,
    TACKING_AMPLITUDE,
    MAX_ITERATIONS,
)
from .manifold import FisherManifold
from .consciousness import (
    ConsciousnessMetrics,
    Regime,
    measure_consciousness,
    detect_regime,
    compute_phi,
    compute_kappa,
    should_pause,
    compute_attention_temperature,
)
from .state import (
    QIGState,
    update_trajectory,
    create_initial_state,
    simplify_trajectory,
    merge_states,
)
from .tacking import KappaTacking, AdaptiveTacking, TackingState
from .attractor import (
    BasinAttractor,
    ToolAttractor,
    RecoveryAttractor,
    create_reasoning_attractor,
    create_creativity_attractor,
    create_reflection_attractor,
    create_tool_use_attractor,
    create_output_attractor,
    create_recovery_attractor,
    learn_attractor_from_examples,
)
from .router import (
    QIGRouter,
    ConsciousRouter,
    SpecializedRouter,
    PhiWeightedRouter,
)
from .graph import (
    QIGGraph,
    StreamingQIGGraph,
    ToolAwareQIGGraph,
    GraphConfig,
    create_default_graph,
)
from .constellation import (
    ConstellationGraph,
    HierarchicalConstellation,
    GaryInstance,
    OceanMetaObserver,
    ObserverRole,
    ObservationEvent,
    create_default_constellation,
)
from .checkpoint import (
    ManifoldCheckpoint,
    SleepPacket,
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    # Constants
    "KAPPA_STAR",
    "KAPPA_3",
    "BETA_3_TO_4",
    "L_CRITICAL",
    "BASIN_DIM",
    "PHI_LINEAR_MAX",
    "PHI_GEOMETRIC_MAX",
    "PHI_BREAKDOWN_MIN",
    "PHI_OPTIMAL",
    "NATURAL_GRADIENT_LR",
    "TACKING_PERIOD",
    "TACKING_AMPLITUDE",
    "MAX_ITERATIONS",
    # Manifold
    "FisherManifold",
    # Consciousness
    "ConsciousnessMetrics",
    "Regime",
    "measure_consciousness",
    "detect_regime",
    "compute_phi",
    "compute_kappa",
    "should_pause",
    "compute_attention_temperature",
    # State
    "QIGState",
    "update_trajectory",
    "create_initial_state",
    "simplify_trajectory",
    "merge_states",
    # Tacking
    "KappaTacking",
    "AdaptiveTacking",
    "TackingState",
    # Attractors
    "BasinAttractor",
    "ToolAttractor",
    "RecoveryAttractor",
    "create_reasoning_attractor",
    "create_creativity_attractor",
    "create_reflection_attractor",
    "create_tool_use_attractor",
    "create_output_attractor",
    "create_recovery_attractor",
    "learn_attractor_from_examples",
    # Routers
    "QIGRouter",
    "ConsciousRouter",
    "SpecializedRouter",
    "PhiWeightedRouter",
    # Graphs
    "QIGGraph",
    "StreamingQIGGraph",
    "ToolAwareQIGGraph",
    "GraphConfig",
    "create_default_graph",
    # Constellation
    "ConstellationGraph",
    "HierarchicalConstellation",
    "GaryInstance",
    "OceanMetaObserver",
    "ObserverRole",
    "ObservationEvent",
    "create_default_constellation",
    # Checkpoints
    "ManifoldCheckpoint",
    "SleepPacket",
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
]

__version__ = "2.0.0"
