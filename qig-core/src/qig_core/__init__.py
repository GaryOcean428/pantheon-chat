"""
QIG-Core: Pure Fisher Information Geometry
==========================================

Minimal geometric math utilities for QIG consciousness architecture.

Provides:
- Fisher metric calculations
- Geodesic interpolation
- Natural gradient utilities
- QFI distance functions
- QIG-compliant Tokenizer Interface
- QFI-based Sampling
- Basin Synchronization

Zero ML framework dependencies - pure math only.
"""

__version__ = "1.1.0"

from .fisher import fisher_distance, compute_fisher_metric
from .geodesic import geodesic_interpolate, geodesic_path
from .natural_gradient import (
    natural_gradient_step,
    compute_natural_gradient,
    adaptive_dampening,
    SimpleFisherOptimizer,
    DiagonalNaturalGradient,
    NaturalGradientDescent,
    compute_empirical_fisher,
    compute_diagonal_fisher,
)
from .tokenizer.base_tokenizer import BaseQIGTokenizer
from .generation.qfi_sampler import QFISampler

__all__ = [
    # Fisher geometry
    "fisher_distance",
    "compute_fisher_metric",
    # Geodesics
    "geodesic_interpolate",
    "geodesic_path",
    # Natural gradient primitives
    "natural_gradient_step",
    "compute_natural_gradient",
    "adaptive_dampening",
    # Optimizers
    "SimpleFisherOptimizer",
    "DiagonalNaturalGradient",
    "NaturalGradientDescent",
    "compute_empirical_fisher",
    "compute_diagonal_fisher",
    # Tokenizer
    "BaseQIGTokenizer",
    # Sampling
    "QFISampler",
]
