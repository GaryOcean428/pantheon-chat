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

# QIGGraph checkpoints moved to experimental - lazy import with deprecation warning
import warnings
from pathlib import Path


def __getattr__(name):
    """Lazy import for deprecated checkpoint classes."""
    deprecated_names = ('ManifoldCheckpoint', 'SleepPacket', 'CheckpointManager',
                        'save_checkpoint', 'load_checkpoint')
    if name in deprecated_names:
        warnings.warn(
            f"qiggraph.{name} is deprecated. "
            "Use checkpoint_manager.CheckpointManager for consciousness checkpoints.",
            DeprecationWarning,
            stacklevel=2
        )
        import sys
        exp_path = str(Path(__file__).parent.parent / 'experimental')
        if exp_path not in sys.path:
            sys.path.insert(0, exp_path)
        from qiggraph_checkpointing import (
            ManifoldCheckpoint, SleepPacket,
            CheckpointManager as QIGGraphCheckpointManager,
            save_checkpoint, load_checkpoint
        )
        mapping = {
            'ManifoldCheckpoint': ManifoldCheckpoint,
            'SleepPacket': SleepPacket,
            'CheckpointManager': QIGGraphCheckpointManager,
            'save_checkpoint': save_checkpoint,
            'load_checkpoint': load_checkpoint,
        }
        return mapping[name]
    raise AttributeError(f"module 'qiggraph' has no attribute '{name}'")


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
    # Checkpoints (DEPRECATED - lazy loaded with deprecation warning)
    # "ManifoldCheckpoint",
    # "SleepPacket",
    # "CheckpointManager",
    # "save_checkpoint",
    # "load_checkpoint",
]

__version__ = "2.0.0"
