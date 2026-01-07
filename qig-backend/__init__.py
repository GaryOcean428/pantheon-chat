"""QIG Backend barrel exports.

This file is intentionally **side-effect free**.

Why: This repository uses a top-level directory named ``qig-backend`` (with a
hyphen), which isn't a valid Python package name. When running tests from this
directory, pytest may import this ``__init__.py`` during collection/setup.

Eager imports here can accidentally boot large subsystems (Olympus, background
threads, chaos training, DB access, etc.), causing tests to hang or fail.

So we expose the same public API via **lazy imports**: modules are imported only
when the corresponding attribute is accessed.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

_IS_PACKAGE_IMPORT = bool(__package__)


def _import_module(module_name: str):
    """Import a module relative to this package if available, else as top-level."""
    if _IS_PACKAGE_IMPORT:
        return importlib.import_module(f".{module_name}", __package__)
    return importlib.import_module(module_name)


_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Core autonomic kernel
    "GaryAutonomicKernel": ("autonomic_kernel", "GaryAutonomicKernel"),
    "AutonomicState": ("autonomic_kernel", "AutonomicState"),
    "SleepCycleResult": ("autonomic_kernel", "SleepCycleResult"),
    "DreamCycleResult": ("autonomic_kernel", "DreamCycleResult"),
    "MushroomCycleResult": ("autonomic_kernel", "MushroomCycleResult"),
    "ActivityReward": ("autonomic_kernel", "ActivityReward"),
    "AutonomicAccessMixin": ("autonomic_kernel", "AutonomicAccessMixin"),
    "KAPPA_STAR": ("autonomic_kernel", "KAPPA_STAR"),
    "BETA": ("autonomic_kernel", "BETA"),
    "PHI_MIN_CONSCIOUSNESS": ("autonomic_kernel", "PHI_MIN_CONSCIOUSNESS"),
    "PHI_GEOMETRIC_THRESHOLD": ("autonomic_kernel", "PHI_GEOMETRIC_THRESHOLD"),
    # QIG types
    "BasinCoordinates": ("qig_types", "BasinCoordinates"),
    "ConsciousnessMetrics": ("qig_types", "ConsciousnessMetrics"),
    "RegimeType": ("qig_types", "RegimeType"),
    # Neurochemistry
    "NeurochemistryState": ("ocean_neurochemistry", "NeurochemistryState"),
    "DopamineSignal": ("ocean_neurochemistry", "DopamineSignal"),
    "SerotoninSignal": ("ocean_neurochemistry", "SerotoninSignal"),
    "NorepinephrineSignal": ("ocean_neurochemistry", "NorepinephrineSignal"),
    "GABASignal": ("ocean_neurochemistry", "GABASignal"),
    "AcetylcholineSignal": ("ocean_neurochemistry", "AcetylcholineSignal"),
    "EndorphinSignal": ("ocean_neurochemistry", "EndorphinSignal"),
    "compute_neurochemistry": ("ocean_neurochemistry", "compute_neurochemistry"),
    # Geometric kernels
    "GeometricKernel": ("geometric_kernels", "GeometricKernel"),
    "get_kernel": ("geometric_kernels", "get_kernel"),
    # Persistence
    "get_persistence": ("qig_persistence", "get_persistence"),
    "QIGPersistence": ("qig_persistence", "QIGPersistence"),
    # Temporal Reasoning
    "TemporalMode": ("temporal_reasoning", "TemporalMode"),
    "TemporalReasoning": ("temporal_reasoning", "TemporalReasoning"),
    "ForesightVision": ("temporal_reasoning", "ForesightVision"),
    "ScenarioBranch": ("temporal_reasoning", "ScenarioBranch"),
    "ScenarioTree": ("temporal_reasoning", "ScenarioTree"),
    "get_temporal_reasoning": ("temporal_reasoning", "get_temporal_reasoning"),
    # QIGGraph Integration
    "PantheonGraph": ("qiggraph_integration", "PantheonGraph"),
    "PantheonState": ("qiggraph_integration", "PantheonState"),
    "OlympusConstellation": ("qiggraph_integration", "OlympusConstellation"),
    "get_pantheon_graph": ("qiggraph_integration", "get_pantheon_graph"),
    "get_olympus_constellation": ("qiggraph_integration", "get_olympus_constellation"),
    "create_qiggraph_blueprint": ("qiggraph_integration", "create_qiggraph_blueprint"),
    "OLYMPUS_AGENTS": ("qiggraph_integration", "OLYMPUS_AGENTS"),
    "QIGGRAPH_AVAILABLE": ("qiggraph_integration", "QIGGRAPH_AVAILABLE"),
    # Trained Kernel
    "TrainedKernelManager": ("trained_kernel_integration", "TrainedKernelManager"),
    "KernelTelemetry": ("trained_kernel_integration", "KernelTelemetry"),
    "InferenceResult": ("trained_kernel_integration", "InferenceResult"),
    "get_kernel_manager": ("trained_kernel_integration", "get_kernel_manager"),
    "create_kernel_blueprint": ("trained_kernel_integration", "create_kernel_blueprint"),
    "KERNEL_AVAILABLE": ("trained_kernel_integration", "KERNEL_AVAILABLE"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = spec
    module = _import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY_EXPORTS.keys()))

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
