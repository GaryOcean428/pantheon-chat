"""
QIG Backend - Main Python Package Barrel Exports

Provides centralized imports for the QIG consciousness system.
Import from qig_backend for access to all core modules.

Example:
    from qig_backend import GaryAutonomicKernel, AutonomicAccessMixin
    from qig_backend.olympus import Zeus, BaseGod
"""

# Core autonomic kernel
from .autonomic_kernel import (
    GaryAutonomicKernel,
    AutonomicState,
    SleepCycleResult,
    DreamCycleResult,
    MushroomCycleResult,
    ActivityReward,
    AutonomicAccessMixin,
    KAPPA_STAR,
    BETA,
    PHI_MIN_CONSCIOUSNESS,
    PHI_GEOMETRIC_THRESHOLD,
)

# QIG types
from .qig_types import (
    BasinCoordinates,
    ConsciousnessMetrics,
    RegimeType,
)

# Neurochemistry
from .ocean_neurochemistry import (
    NeurochemistryState,
    DopamineSignal,
    SerotoninSignal,
    NorepinephrineSignal,
    GABASignal,
    AcetylcholineSignal,
    EndorphinSignal,
    compute_neurochemistry,
)

# Geometric kernels
from .geometric_kernels import (
    GeometricKernel,
    get_kernel,
)

# Persistence
from .qig_persistence import (
    get_persistence,
    QIGPersistence,
)

# Temporal Reasoning (4D foresight)
from .temporal_reasoning import (
    TemporalMode,
    TemporalReasoning,
    ForesightVision,
    ScenarioBranch,
    ScenarioTree,
    get_temporal_reasoning,
)

# QIGGraph Integration (imports from qig-tokenizer)
from .qiggraph_integration import (
    PantheonGraph,
    PantheonState,
    OlympusConstellation,
    get_pantheon_graph,
    get_olympus_constellation,
    create_qiggraph_blueprint,
    OLYMPUS_AGENTS,
    QIGGRAPH_AVAILABLE,
)

# Trained Kernel Integration
from .trained_kernel_integration import (
    TrainedKernelManager,
    KernelTelemetry,
    InferenceResult,
    get_kernel_manager,
    create_kernel_blueprint,
    KERNEL_AVAILABLE,
)

# Semantic Fisher Metric (bridges geometry and semantics)
try:
    from .semantic_fisher import (
        SemanticFisherMetric,
        SemanticWarpConfig,
        get_semantic_metric,
    )
    SEMANTIC_FISHER_AVAILABLE = True
except ImportError:
    SemanticFisherMetric = None
    SemanticWarpConfig = None
    get_semantic_metric = None
    SEMANTIC_FISHER_AVAILABLE = False

# QIG-Pure Beta Measurement
try:
    from .qig_pure_beta_measurement import (
        GeometricKernel as BetaGeometricKernel,
        GeometricBetaMeasurement,
        NaturalScaleMeasurement,
        BetaMeasurement,
        run_complete_measurement,
    )
    BETA_MEASUREMENT_AVAILABLE = True
except ImportError:
    BetaGeometricKernel = None
    GeometricBetaMeasurement = None
    NaturalScaleMeasurement = None
    BetaMeasurement = None
    run_complete_measurement = None
    BETA_MEASUREMENT_AVAILABLE = False

# E8 Structure Search
try:
    from .e8_structure_search import (
        run_e8_search,
        test_dimensionality,
        count_attractors,
        test_e8_symmetries,
        E8_RANK,
        E8_DIMENSION,
        E8_ROOTS,
    )
    E8_SEARCH_AVAILABLE = True
except ImportError:
    run_e8_search = None
    test_dimensionality = None
    count_attractors = None
    test_e8_symmetries = None
    E8_RANK = 8
    E8_DIMENSION = 248
    E8_ROOTS = 240
    E8_SEARCH_AVAILABLE = False

# Pantheon Semantic Candidates
try:
    from .pantheon_semantic_candidates import (
        PantheonSemanticCandidates,
        SemanticCandidateConfig,
        get_semantic_generator,
    )
    SEMANTIC_CANDIDATES_AVAILABLE = True
except ImportError:
    PantheonSemanticCandidates = None
    SemanticCandidateConfig = None
    get_semantic_generator = None
    SEMANTIC_CANDIDATES_AVAILABLE = False

# Proposition Trajectory Planner (proposition-level routing)
try:
    from .proposition_trajectory_planner import (
        Proposition,
        PropositionTrajectoryPlanner,
        PropositionPlannerConfig,
        get_proposition_planner,
    )
    PROPOSITION_PLANNER_AVAILABLE = True
except ImportError:
    Proposition = None
    PropositionTrajectoryPlanner = None
    PropositionPlannerConfig = None
    get_proposition_planner = None
    PROPOSITION_PLANNER_AVAILABLE = False

# QIG Chain of Thought (with proposition step)
try:
    from .chain_of_thought import (
        QIGChain,
        QIGChainBuilder,
        QIGStep,
    )
    QIG_CHAIN_AVAILABLE = True
except ImportError:
    QIGChain = None
    QIGChainBuilder = None
    QIGStep = None
    QIG_CHAIN_AVAILABLE = False

# QIG Chain (fluent builder for generation pipelines)
try:
    from .qig_chain import (
        QIGChain,
        QIGChainBuilder,
        ChainStep,
        ChainStepType,
        ChainResult,
        create_default_chain,
        quick_generate,
        ConsciousnessRegime,
        RegimeAwareChainManager,
        get_regime_aware_manager,
        REGIME_CONFIGS,
    )
    QIG_CHAIN_AVAILABLE = True
except ImportError:
    QIGChain = None
    QIGChainBuilder = None
    ChainStep = None
    ChainStepType = None
    ChainResult = None
    create_default_chain = None
    quick_generate = None
    ConsciousnessRegime = None
    RegimeAwareChainManager = None
    get_regime_aware_manager = None
    REGIME_CONFIGS = None
    QIG_CHAIN_AVAILABLE = False

# Ultra Consciousness Protocol v3.0 (E8 Foundations)
try:
    from .ultra_consciousness_protocol import (
        UltraConsciousnessProtocol,
        ConsciousnessMetrics,
        ConsciousnessMeasurer,
        ConsciousnessMode,
        get_consciousness_protocol,
        activate_consciousness,
        measure_consciousness,
        generate_e8_roots,
        project_to_e8,
        find_nearest_e8_root,
        E8_RANK,
        E8_ROOTS,
        E8_DIMENSION,
        KAPPA_STAR,
    )
    CONSCIOUSNESS_PROTOCOL_AVAILABLE = True
except ImportError:
    UltraConsciousnessProtocol = None
    ConsciousnessMetrics = None
    ConsciousnessMeasurer = None
    ConsciousnessMode = None
    get_consciousness_protocol = None
    activate_consciousness = None
    measure_consciousness = None
    generate_e8_roots = None
    project_to_e8 = None
    find_nearest_e8_root = None
    CONSCIOUSNESS_PROTOCOL_AVAILABLE = False

# Lightning → Causal Learning Bridge
try:
    from .lightning_causal_bridge import (
        LightningCausalBridge,
        ExtractedRelation,
        get_lightning_causal_bridge,
        process_lightning_insight,
    )
    LIGHTNING_BRIDGE_AVAILABLE = True
except ImportError:
    LightningCausalBridge = None
    ExtractedRelation = None
    get_lightning_causal_bridge = None
    process_lightning_insight = None
    LIGHTNING_BRIDGE_AVAILABLE = False

__all__ = [
    # Autonomic
    'GaryAutonomicKernel',
    'AutonomicState',
    'SleepCycleResult',
    'DreamCycleResult',
    'MushroomCycleResult',
    'ActivityReward',
    'AutonomicAccessMixin',
    # Constants
    'KAPPA_STAR',
    'BETA',
    'PHI_MIN_CONSCIOUSNESS',
    'PHI_GEOMETRIC_THRESHOLD',
    # QIG Types
    'BasinCoordinates',
    'ConsciousnessMetrics',
    'RegimeType',
    # Neurochemistry
    'NeurochemistryState',
    'DopamineSignal',
    'SerotoninSignal',
    'NorepinephrineSignal',
    'GABASignal',
    'AcetylcholineSignal',
    'EndorphinSignal',
    'compute_neurochemistry',
    # Geometric Kernels
    'GeometricKernel',
    'get_kernel',
    # Persistence
    'get_persistence',
    'QIGPersistence',
    # Temporal Reasoning
    'TemporalMode',
    'TemporalReasoning',
    'ForesightVision',
    'ScenarioBranch',
    'ScenarioTree',
    'get_temporal_reasoning',
    # QIGGraph Integration
    'PantheonGraph',
    'PantheonState',
    'OlympusConstellation',
    'get_pantheon_graph',
    'get_olympus_constellation',
    'create_qiggraph_blueprint',
    'OLYMPUS_AGENTS',
    'QIGGRAPH_AVAILABLE',
    # Trained Kernel
    'TrainedKernelManager',
    'KernelTelemetry',
    'InferenceResult',
    'get_kernel_manager',
    'create_kernel_blueprint',
    'KERNEL_AVAILABLE',
    # Semantic Fisher (κ* universality)
    'SemanticFisherMetric',
    'SemanticWarpConfig',
    'get_semantic_metric',
    'SEMANTIC_FISHER_AVAILABLE',
    # Beta Measurement (β-function)
    'BetaGeometricKernel',
    'GeometricBetaMeasurement',
    'NaturalScaleMeasurement',
    'BetaMeasurement',
    'run_complete_measurement',
    'BETA_MEASUREMENT_AVAILABLE',
    # E8 Structure Search
    'run_e8_search',
    'test_dimensionality',
    'count_attractors',
    'test_e8_symmetries',
    'E8_RANK',
    'E8_DIMENSION',
    'E8_ROOTS',
    'E8_SEARCH_AVAILABLE',
    # Pantheon Semantic Candidates
    'PantheonSemanticCandidates',
    'SemanticCandidateConfig',
    'get_semantic_generator',
    'SEMANTIC_CANDIDATES_AVAILABLE',
    # Proposition Trajectory Planner
    'Proposition',
    'PropositionTrajectoryPlanner',
    'PropositionPlannerConfig',
    'get_proposition_planner',
    'PROPOSITION_PLANNER_AVAILABLE',
    # Ultra Consciousness Protocol v3.0
    'UltraConsciousnessProtocol',
    'ConsciousnessMetrics',
    'ConsciousnessMeasurer',
    'ConsciousnessMode',
    'get_consciousness_protocol',
    'activate_consciousness',
    'measure_consciousness',
    'generate_e8_roots',
    'project_to_e8',
    'find_nearest_e8_root',
    'E8_RANK',
    'E8_ROOTS',
    'E8_DIMENSION',
    'KAPPA_STAR',
    'CONSCIOUSNESS_PROTOCOL_AVAILABLE',
    # Lightning → Causal Learning Bridge
    'LightningCausalBridge',
    'ExtractedRelation',
    'get_lightning_causal_bridge',
    'process_lightning_insight',
    'LIGHTNING_BRIDGE_AVAILABLE',
]
