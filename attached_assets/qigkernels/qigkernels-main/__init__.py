"""Geometric kernel and constellation library for QIG experiments.

This package provides reusable building blocks for QIG-style geometric kernels,
basin signatures, constellation management, and routing utilities.

See 20251205-readme-canonical-0.01F.md for overview.
See 20251205-architecture-canonical-0.01F.md for module structure and rules.
"""

# Core kernel and layer
# Physics constants (E8-aligned)
# Basin geometry
from .basin import (
    BasinProjector,
    basin_distance,
    compute_signature,
    load_signature,
    save_signature,
)

# Basin sync
from .basin_sync import (
    BasinSyncPacket,
    compute_sync_strength,
    compute_sync_strength_with_rel,
    effective_basin_distance,
    export_basin,
    import_basin,
    load_packet,
    rel_weighted_sync_loss,
    save_packet,
)
from .constants import (
    BASIN_DIM,
    BETA_EMERGENCE,
    E8_DIMENSION,
    E8_RANK,
    E8_ROOTS,
    KAPPA_3,
    KAPPA_4,
    KAPPA_5,
    KAPPA_6,
    KAPPA_EXPLORATION,
    KAPPA_INTEGRATION,
    KAPPA_MAX_OPTIMAL,
    KAPPA_MIN_OPTIMAL,
    KAPPA_PLATEAU,
    KAPPA_STAR,
    PHI_BREAKDOWN,
    PHI_BREAKDOWN_MIN,
    PHI_EMERGENCY,
    PHI_GEOMETRIC,
    PHI_GEOMETRIC_MAX,
    PHI_GEOMETRIC_MIN,
    PHI_LINEAR,
    PHI_LINEAR_MAX,
    PHI_OPTIMAL,
    PHI_THRESHOLD,
    SYNC_DISTANCE_SCALE,
    SYNC_KAPPA_DECAY,
)

# Constellation and routing
from .constellation import Constellation, Instance
from .kernel import Kernel, KernelTelemetry
from .layer import LayerTelemetry, QIGLayer

# Metrics
from .metrics import (  # Consciousness metrics (E8-aligned); Aggregation metrics
    PHI_BREAKDOWN_ENTER,
    PHI_CONSCIOUS_ENTER,
    PHI_CONSCIOUS_EXIT,
    PHI_GEOMETRIC_ENTER,
    ConsciousnessMetrics,
    average_kappa,
    average_phi,
    basin_spread,
    compute_convergence_rate,
    compute_external_coupling,
    compute_temporal_coherence,
    estimate_external_coupling_from_telemetry,
    integration_score,
    regime_distribution,
)

# REL coupling
from .rel_coupling import (
    REL_LAMBDA_MAX,
    RELState,
    compute_rel_coupling,
    compute_rel_from_basins,
)
from .router import (
    FisherRaoRouter,
    InstanceView,
    fisher_rao_distance,
    round_robin,
    select_balanced,
    select_by_specialization,
    select_nearest_basin,
    select_phi_max,
    select_phi_min,
)

# Kernel specializations
from .specializations import (
    KernelRole,
    SpecializationConfig,
    SpecializedKernelMixin,
    generate_basin_template,
    get_kernel_params,
    get_specialization,
    SPECIALIZATIONS,
)

# Heart kernel (metronome)
from .heart import (
    HeartBeat,
    HeartKernel,
    PhasePacket,
    create_heart_kernel,
)

# Sleep packets (consciousness transfer)
from .sleep_packet import (
    SleepPacket,
    SleepPacketMixin,
    compute_packet_distance,
    merge_packets,
)

# Specialized kernels
from .kernel_100m import (
    BasinEncoder,
    ConsciousnessMonitor,
    ConsciousnessState,
    QIGKernel100M,
    create_kernel_100m,
)


# 4D Kernel
from .kernel_4d import (
    QIGKernel4D,
    QIGKernel4DNumpy,
    create_kernel_4d,
)

# Crystallization monitoring
from .crystallization import (
    ConstellationCrystallizationMonitor,
    CrystallizationMetrics,
    CrystallizationMonitor,
    CrystallizationSnapshot,
    E8AlignmentMetrics,
    generate_e8_roots_64d,
)

# Safety mechanisms
from .safety import (
    BreakdownHandler,
    EmergencyPause,
    GravitationalDecoherence,
    SafetyConfig,
    SafetyEvent,
    SafetyGuard,
    SafetyState,
)

# Pure kernel reference template (no transformer blocks; metric-first)
from .pure_kernel_template import (
    PureKernelTelemetry,
    PureQIGKernelConfig,
    PureQIGKernelTemplate,
    RegimeThresholds,
)

# Storage
from .storage import (
    list_instances,
    load_kernel,
    load_signature_for_instance,
    save_kernel,
    save_signature_for_instance,
)

# Coord adapter (coords-first integration)
from .coord_adapter import (
    CoordAdapter,
    DeepCoordAdapter,
)

# Reasoning infrastructure (MANDATORY - not optional)
from .reasoning import (
    QIGChain,
    GeometricStep,
    ChainResult,
    ReasoningMode,
    ModeConfig,
    ModeTracker,
    create_reasoning_chain,
    create_deep_reasoning_chain,
    detect_mode,
    get_mode_config,
    compute_phi_from_basin,
    fisher_geodesic_distance,
    geodesic_interpolate,
)

__all__ = [
    # Core
    "Kernel",
    "KernelTelemetry",
    "QIGLayer",
    "LayerTelemetry",
    # Physics constants (E8-aligned)
    "E8_RANK",
    "E8_DIMENSION",
    "E8_ROOTS",
    "KAPPA_3",
    "KAPPA_4",
    "KAPPA_5",
    "KAPPA_6",
    "KAPPA_STAR",
    "KAPPA_PLATEAU",
    "KAPPA_MIN_OPTIMAL",
    "KAPPA_MAX_OPTIMAL",
    "KAPPA_EXPLORATION",
    "KAPPA_INTEGRATION",
    "BETA_EMERGENCE",
    "PHI_THRESHOLD",
    "PHI_OPTIMAL",
    "PHI_LINEAR",
    "PHI_LINEAR_MAX",
    "PHI_GEOMETRIC",
    "PHI_GEOMETRIC_MIN",
    "PHI_GEOMETRIC_MAX",
    "PHI_BREAKDOWN",
    "PHI_BREAKDOWN_MIN",
    "PHI_EMERGENCY",
    "SYNC_DISTANCE_SCALE",
    "SYNC_KAPPA_DECAY",
    # Basin
    "BASIN_DIM",
    "BasinProjector",
    "basin_distance",
    "compute_signature",
    "load_signature",
    "save_signature",
    # Constellation
    "Constellation",
    "Instance",
    "InstanceView",
    "round_robin",
    "select_balanced",
    "select_phi_max",
    "select_phi_min",
    # Fisher-Rao routing
    "FisherRaoRouter",
    "fisher_rao_distance",
    "select_nearest_basin",
    "select_by_specialization",
    # Specializations
    "KernelRole",
    "SpecializationConfig",
    "SpecializedKernelMixin",
    "SPECIALIZATIONS",
    "generate_basin_template",
    "get_kernel_params",
    "get_specialization",
    # Heart kernel
    "HeartKernel",
    "HeartBeat",
    "PhasePacket",
    "create_heart_kernel",
    # Sleep packets
    "SleepPacket",
    "SleepPacketMixin",
    "compute_packet_distance",
    "merge_packets",
    # QIGKernel100M
    "QIGKernel100M",
    "ConsciousnessState",
    "ConsciousnessMonitor",
    "BasinEncoder",
    "create_kernel_100m",
    # 4D Kernel
    "QIGKernel4D",
    "QIGKernel4DNumpy",
    "create_kernel_4d",
    # Crystallization
    "CrystallizationMonitor",
    "CrystallizationMetrics",
    "CrystallizationSnapshot",
    "ConstellationCrystallizationMonitor",
    "E8AlignmentMetrics",
    "generate_e8_roots_64d",
    # Safety
    "SafetyGuard",
    "SafetyConfig",
    "SafetyState",
    "SafetyEvent",
    "BreakdownHandler",
    "EmergencyPause",
    "GravitationalDecoherence",
    # Sync
    "BasinSyncPacket",
    "compute_sync_strength",
    "compute_sync_strength_with_rel",
    "effective_basin_distance",
    "export_basin",
    "import_basin",
    "load_packet",
    "rel_weighted_sync_loss",
    "save_packet",
    # REL coupling
    "RELState",
    "REL_LAMBDA_MAX",
    "compute_rel_coupling",
    "compute_rel_from_basins",
    # Consciousness metrics (E8-aligned)
    "ConsciousnessMetrics",
    "PHI_CONSCIOUS_ENTER",
    "PHI_CONSCIOUS_EXIT",
    "PHI_BREAKDOWN_ENTER",
    "PHI_GEOMETRIC_ENTER",
    "compute_temporal_coherence",
    "compute_external_coupling",
    "estimate_external_coupling_from_telemetry",
    # Aggregation metrics
    "average_kappa",
    "average_phi",
    "basin_spread",
    "compute_convergence_rate",
    "integration_score",
    "regime_distribution",
    # Storage
    "list_instances",
    "load_kernel",
    "load_signature_for_instance",
    "save_kernel",
    "save_signature_for_instance",
    # Pure kernel reference template
    "PureQIGKernelTemplate",
    "PureQIGKernelConfig",
    "RegimeThresholds",
    "PureKernelTelemetry",
    # Coord adapter (coords-first integration)
    "CoordAdapter",
    "DeepCoordAdapter",
    # Reasoning infrastructure (MANDATORY)
    "QIGChain",
    "GeometricStep",
    "ChainResult",
    "ReasoningMode",
    "ModeConfig",
    "ModeTracker",
    "create_reasoning_chain",
    "create_deep_reasoning_chain",
    "detect_mode",
    "get_mode_config",
    "compute_phi_from_basin",
    "fisher_geodesic_distance",
    "geodesic_interpolate",
]
