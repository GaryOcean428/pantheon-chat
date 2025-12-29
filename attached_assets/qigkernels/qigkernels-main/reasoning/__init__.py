"""QIG Reasoning Infrastructure

Native geometric reasoning: chains, graphs, modes.

KEY PRINCIPLE: Reasoning is MANDATORY, not optional.
- There is NO forward pass without reasoning
- Training loss sees ALL chain steps
- Mode selection emerges from Φ state

Usage:
    from qigkernels.reasoning import QIGChain, GeometricStep, ChainResult
    
    # Define chain (MANDATORY steps)
    chain = QIGChain([
        GeometricStep("encode", model.encode),
        GeometricStep("integrate", model.integrate),
        GeometricStep("refine", model.refine),
    ])
    
    # Execute (NO bypass path)
    result = chain.run(input_basin)
    
    # Training uses ALL steps
    for step in result.trajectory:
        loss += compute_loss(step['basin_coords'], target)
"""

# Constants
from .constants import (
    BASIN_DIM,
    KAPPA_STAR,
    PHI_THRESHOLD,
    PHI_THRESHOLD_DEFAULT,
    PHI_DEGRADATION_THRESHOLD,
    KAPPA_RANGE_DEFAULT,
    GEODESIC_STEPS,
    MIN_RECURSIONS,
    MAX_RECURSIONS,
    BETA_RUNNING,
)

# Primitives
from .primitives import (
    basin_to_density_matrix,
    compute_phi_from_basin,
    compute_fisher_metric,
    compute_kappa,
    bures_distance,
    fisher_geodesic_distance,
    geodesic_interpolate,
    project_to_basin,
    normalize_basin,
)

# Re-export compute_kappa for convenience
__all_primitives__ = [
    "basin_to_density_matrix",
    "compute_phi_from_basin",
    "compute_fisher_metric",
    "compute_kappa",
    "bures_distance",
    "fisher_geodesic_distance",
    "geodesic_interpolate",
    "project_to_basin",
    "normalize_basin",
]

# Chain classes
from .qig_chain import (
    QIGChain,
    GeometricStep,
    ChainResult,
    create_reasoning_chain,
    create_deep_reasoning_chain,
)

# Modes
from .modes import (
    ReasoningMode,
    ModeConfig,
    ModeTracker,
    ModeTransition,
    MODE_CONFIGS,
    detect_mode,
    get_mode_config,
    get_current_config,
    compute_mode_gradient,
)

# 4D Temporal Components
from .temporal import (
    StateHistoryBuffer,
    HistoryEntry,
    BasinForesight,
    GeodesicFit,
    Phi4DMetrics,
    ForesightResult,
    measure_phi_4d,
    classify_regime,
    compute_trajectory_smoothness,
    fit_fisher_geodesic,
)

# 4D Chain
from .qig_chain_4d import (
    QIGChain4D,
    ChainResult4D,
    create_4d_reasoning_chain,
)

# Temporal Attention
from .temporal_attention import (
    TemporalAttention,
    TemporalAttentionNumpy,
    fisher_geodesic_mean,
    create_temporal_attention,
)

__all__ = [
    # Constants
    "BASIN_DIM",
    "KAPPA_STAR",
    "PHI_THRESHOLD",
    "PHI_THRESHOLD_DEFAULT",
    "PHI_DEGRADATION_THRESHOLD",
    "KAPPA_RANGE_DEFAULT",
    "GEODESIC_STEPS",
    "MIN_RECURSIONS",
    "MAX_RECURSIONS",
    "BETA_RUNNING",
    # Primitives
    "basin_to_density_matrix",
    "compute_phi_from_basin",
    "compute_fisher_metric",
    "compute_kappa",
    "bures_distance",
    "fisher_geodesic_distance",
    "geodesic_interpolate",
    "project_to_basin",
    "normalize_basin",
    # Chain
    "QIGChain",
    "GeometricStep",
    "ChainResult",
    "create_reasoning_chain",
    "create_deep_reasoning_chain",
    # Modes
    "ReasoningMode",
    "ModeConfig",
    "ModeTracker",
    "ModeTransition",
    "MODE_CONFIGS",
    "detect_mode",
    "get_mode_config",
    "get_current_config",
    "compute_mode_gradient",
    # 4D Temporal
    "StateHistoryBuffer",
    "HistoryEntry",
    "BasinForesight",
    "GeodesicFit",
    "Phi4DMetrics",
    "ForesightResult",
    "measure_phi_4d",
    "classify_regime",
    "compute_trajectory_smoothness",
    "fit_fisher_geodesic",
    # 4D Chain
    "QIGChain4D",
    "ChainResult4D",
    "create_4d_reasoning_chain",
    # Temporal Attention
    "TemporalAttention",
    "TemporalAttentionNumpy",
    "fisher_geodesic_mean",
    "create_temporal_attention",
]
