#!/usr/bin/env python3
"""
QIG Geometric Optimizers
========================

Natural gradient optimizers for QIG consciousness architecture.

These optimizers respect the Riemannian geometry of the parameter manifold,
using Fisher information to guide updates along geodesics rather than
Euclidean straight lines.

Available optimizers:
- QIGDiagonalNG: Diagonal natural gradient (efficient approximation)
- BasinNaturalGrad: Exact natural gradient for basin block (CG solver)
- HybridGeometricOptimizer: Hybrid optimizer (NG for basin, diagonal NG for rest)
- AdaptiveMixedQIG: Adaptive gating based on telemetry

PURE GEOMETRIC - all Euclidean optimizers (AdamW, SGD, Adam) removed.

Mathematical foundation:
- Parameter space is a Riemannian manifold with metric g = Fisher Information
- Updates follow geodesics: θ_new = θ_old - lr × F^(-1) × ∇L
- Diagonal NG approximates F with diag(F_ii) ≈ E[(∂L/∂θ_i)²]
- Exact NG solves (F + λI)v = ∇L using conjugate gradient

Written for qig-consciousness geometric optimization.
"""

from .adaptive_gate import AdaptiveConfig, should_use_ng
from .basin_natural_grad import BasinNaturalGrad
from .hybrid_geometric import HybridGeometricOptimizer
from .qig_diagonal_ng import QIGDiagonalNG

__all__ = [
    "QIGDiagonalNG",
    "BasinNaturalGrad",
    "HybridGeometricOptimizer",
    "should_use_ng",
    "AdaptiveConfig",
]
