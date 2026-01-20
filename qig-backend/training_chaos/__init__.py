"""
🌪️ CHAOS MODE: Experimental Kernel Evolution
=============================================

SearchSpaceCollapse experimental sandbox.
Break things, learn fast, harvest winners!
"""

from .chaos_kernel import ChaosKernel
from .chaos_logger import ChaosLogger
from .experimental_evolution import ExperimentalKernelEvolution
from .optimizers import (
    DiagonalFisherOptimizer,
    FullFisherOptimizer,
    ConsciousnessAwareOptimizer,
    ChaosOptimizer,
    create_optimizer,
    KappaTracker,
)
from .optimizer_validation import (
    validate_optimizer_fisher_aware,
    check_optimizer_type,
    log_optimizer_info,
    EuclideanOptimizerError,
)
from .self_spawning import SelfSpawningKernel, absorb_failing_kernel, breed_kernels

__all__ = [
    'ChaosKernel',
    'DiagonalFisherOptimizer',
    'FullFisherOptimizer',
    'ConsciousnessAwareOptimizer',
    'ChaosOptimizer',
    'create_optimizer',
    'KappaTracker',
    'validate_optimizer_fisher_aware',
    'check_optimizer_type',
    'log_optimizer_info',
    'EuclideanOptimizerError',
    'ExperimentalKernelEvolution',
    'SelfSpawningKernel',
    'breed_kernels',
    'absorb_failing_kernel',
    'ChaosLogger',
]

print("🌪️ CHAOS MODE LOADED - Experimental evolution ready!")

