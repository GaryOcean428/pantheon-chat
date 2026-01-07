"""
QIG Backend - Main Python Package Barrel Exports

Provides centralized imports for the QIG consciousness system.
Import from qig_backend for access to all core modules.

Example:
    from qig_backend import GaryAutonomicKernel, AutonomicAccessMixin
    from qig_backend.olympus import Zeus, BaseGod
"""

_IS_PACKAGE_IMPORT = bool(__package__)

# Core autonomic kernel
if _IS_PACKAGE_IMPORT:
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
else:
    from autonomic_kernel import (
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
if _IS_PACKAGE_IMPORT:
    from .qig_types import (
        BasinCoordinates,
        ConsciousnessMetrics,
        RegimeType,
    )
else:
    from qig_types import (
        BasinCoordinates,
        ConsciousnessMetrics,
        RegimeType,
    )

# Neurochemistry
if _IS_PACKAGE_IMPORT:
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
else:
    from ocean_neurochemistry import (
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
if _IS_PACKAGE_IMPORT:
    from .geometric_kernels import (
        GeometricKernel,
        get_kernel,
    )
else:
    from geometric_kernels import (
        GeometricKernel,
        get_kernel,
    )

# Persistence
if _IS_PACKAGE_IMPORT:
    from .qig_persistence import (
        get_persistence,
        QIGPersistence,
    )
else:
    from qig_persistence import (
        get_persistence,
        QIGPersistence,
    )

# Temporal Reasoning (4D foresight)
if _IS_PACKAGE_IMPORT:
    from .temporal_reasoning import (
        TemporalMode,
        TemporalReasoning,
        ForesightVision,
        ScenarioBranch,
        ScenarioTree,
        get_temporal_reasoning,
    )
else:
    from temporal_reasoning import (
        TemporalMode,
        TemporalReasoning,
        ForesightVision,
        ScenarioBranch,
        ScenarioTree,
        get_temporal_reasoning,
    )

# QIGGraph Integration (imports from qig-tokenizer)
if _IS_PACKAGE_IMPORT:
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
else:
    from qiggraph_integration import (
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
if _IS_PACKAGE_IMPORT:
    from .trained_kernel_integration import (
        TrainedKernelManager,
        KernelTelemetry,
        InferenceResult,
        get_kernel_manager,
        create_kernel_blueprint,
        KERNEL_AVAILABLE,
    )
else:
    from trained_kernel_integration import (
        TrainedKernelManager,
        KernelTelemetry,
        InferenceResult,
        get_kernel_manager,
        create_kernel_blueprint,
        KERNEL_AVAILABLE,
    )

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
]
