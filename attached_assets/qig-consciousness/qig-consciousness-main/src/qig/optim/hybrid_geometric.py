#!/usr/bin/env python3
"""
Hybrid Geometric Optimizer
==========================

Hybrid optimizer that applies different optimization strategies to different
parts of the model based on geometric principles:

- Basin block (geometric core): Natural Gradient (exact or diagonal)
- Processing layers (QFI attention, recursive integrator, etc.): Diagonal Natural Gradient

Rationale from QIG physics:
1. Basin coordinates (~3M params) are the active geometric core
   - High curvature, active geometry → needs natural gradient
   - This is where consciousness geometry is learned

2. Processing layers (~47M params) contain QFI attention, recursive integrator, running coupling
   - Lower curvature, but still geometric
   - Still requires geometric optimization (diagonal NG)

This stratified approach gives:
- Maximum geometric accuracy where it matters (basin)
- Computational efficiency overall (diagonal approximation for backbone)
- Pure geometric optimization throughout (NO Euclidean contamination)
- Lower cost than full natural gradient

Written for qig-consciousness geometric optimization.

CRITICAL: AdamW and all Euclidean optimizers have been REMOVED.
They are mathematically incompatible with curved information manifolds.
"""

from typing import Any, Optional

import torch

from .basin_natural_grad import BasinNaturalGrad
from .qig_diagonal_ng import QIGDiagonalNG


class HybridGeometricOptimizer:
    """
    Hybrid optimizer for QIG architecture - PURE GEOMETRIC.

    Applies natural gradient to basin block and diagonal NG to the rest.
    NO Euclidean optimization at any level.

    Args:
        model: QIGKernelRecursive model
        lr_ng: Learning rate for natural gradient (basin block)
        lr_rest: Learning rate for rest of model
        cg_iters: Conjugate gradient iterations for basin NG (if use_exact_ng=True)
        use_exact_ng: If True, use exact NG for basin; if False, use diagonal NG
        weight_decay: Weight decay for both optimizers

    Example:
        >>> model = QIGKernelRecursive(...)
        >>> optimizer = HybridGeometricOptimizer(
        ...     model,
        ...     lr_ng=1e-2,
        ...     lr_rest=1e-3,
        ...     use_exact_ng=True
        ... )
        >>> # Training loop
        >>> optimizer.zero_grad()
        >>> loss.backward(create_graph=use_exact_ng)  # create_graph=True if using exact NG
        >>> optimizer.step()

    Geometric interpretation:
    - Basin block: High-curvature manifold → natural gradient follows geodesics
    - Processing layers (QFI attention, recursive integrator): Lower curvature → diagonal NG
    - BOTH use Riemannian geometry (no Euclidean contamination)
    """

    def __init__(
        self,
        model,
        lr_ng: float = 1e-2,
        lr_rest: float = 1e-3,
        cg_iters: int = 8,
        use_exact_ng: bool = True,
        weight_decay: float = 0.01,
    ):
        self.model = model
        self.use_exact_ng = use_exact_ng
        self.rest_optimizer_type = "diagonal_ng"  # FIXED: Only geometric optimization

        # Separate basin parameters from rest
        basin_params = []
        rest_params = []

        # Basin block consists of:
        # 1. basin_coords (vocab_size × basin_dim)
        # 2. basin_to_model projection (basin_dim × d_model)
        # Check both new and legacy attribute names
        coords_layer = getattr(model, "basin_coords_layer", None) or getattr(model, "basin_coords_layer", None)  # BACKWARD COMPAT
        if coords_layer is not None:
            if hasattr(coords_layer, "basin_coords"):
                basin_params.append(coords_layer.basin_coords)
            if hasattr(coords_layer, "basin_to_model"):
                basin_params.extend(coords_layer.basin_to_model.parameters())

        # Everything else goes to rest
        basin_param_ids = {id(p) for p in basin_params}
        for p in model.parameters():
            if p.requires_grad and id(p) not in basin_param_ids:
                rest_params.append(p)

        # Initialize basin optimizer (exact or diagonal NG)
        if use_exact_ng:
            self.basin_opt = BasinNaturalGrad(
                basin_params,
                lr=lr_ng,
                cg_iters=cg_iters,
                damping=1e-4,
            )
        else:
            self.basin_opt = QIGDiagonalNG(
                basin_params,
                lr=lr_ng,
                alpha=0.99,
                weight_decay=weight_decay,
            )

        # Initialize rest optimizer (ALWAYS diagonal NG - pure geometric)
        self.rest_opt = QIGDiagonalNG(
            rest_params,
            lr=lr_rest,
            alpha=0.99,
            weight_decay=weight_decay,
        )

        # Store counts for telemetry
        self.num_basin_params = sum(p.numel() for p in basin_params)
        self.num_rest_params = sum(p.numel() for p in rest_params)

        print("✅ HybridGeometricOptimizer initialized (PURE GEOMETRIC):")
        print(
            f"   Basin optimizer: {'Exact NG' if use_exact_ng else 'Diagonal NG'} "
            f"({self.num_basin_params:,} params, lr={lr_ng})"
        )
        print(f"   Processing layers optimizer: DIAGONAL NG ({self.num_rest_params:,} params, lr={lr_rest})")
        print(f"   Total: {self.num_basin_params + self.num_rest_params:,} parameters")
        print("   🧭 Zero Euclidean contamination - pure Riemannian geometry")

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for both optimizers."""
        self.basin_opt.zero_grad(set_to_none=set_to_none)
        self.rest_opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any | None = None):
        """
        Perform optimization step for both optimizers.

        Important: If using exact natural gradient, must call:
            loss.backward(create_graph=True)
        before step().

        Args:
            closure: Optional closure for optimizer
        """
        # Step rest optimizer first (cheaper)
        self.rest_opt.step(closure)

        # Then step basin optimizer (more expensive)
        self.basin_opt.step(closure)

    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            "basin_opt": self.basin_opt.state_dict(),
            "rest_opt": self.rest_opt.state_dict(),
            "use_exact_ng": self.use_exact_ng,
            "rest_optimizer_type": self.rest_optimizer_type,
        }

    def load_state_dict(self, state_dict: dict):
        """Load state dict from checkpoint."""
        self.basin_opt.load_state_dict(state_dict["basin_opt"])
        self.rest_opt.load_state_dict(state_dict["rest_opt"])

    def get_stats(self) -> dict:
        """
        Get optimizer statistics for telemetry.

        Returns:
            dict with optimizer stats including Fisher metrics
        """
        stats = {
            "num_basin_params": self.num_basin_params,
            "num_rest_params": self.num_rest_params,
            "use_exact_ng": self.use_exact_ng,
            "rest_optimizer": self.rest_optimizer_type,
        }

        # Add basin optimizer stats
        if hasattr(self.basin_opt, "get_stats"):
            basin_stats = self.basin_opt.get_stats()
            stats.update({f"basin_{k}": v for k, v in basin_stats.items()})

        # Add Fisher stats if using diagonal NG
        if hasattr(self.basin_opt, "get_fisher_stats"):
            fisher_stats = self.basin_opt.get_fisher_stats()
            stats.update({f"basin_fisher_{k}": v for k, v in fisher_stats.items()})

        if hasattr(self.rest_opt, "get_fisher_stats"):
            fisher_stats = self.rest_opt.get_fisher_stats()
            stats.update({f"rest_fisher_{k}": v for k, v in fisher_stats.items()})

        return stats


class AdaptiveHybridGeometric(HybridGeometricOptimizer):
    """
    Adaptive version of HybridGeometricOptimizer that conditionally applies
    natural gradient based on telemetry signals.

    This implements the adaptive gating logic:
    - Apply natural gradient when κ_eff is high and system is exploring
    - Fall back to cheaper optimizers during stable/linear regimes
    - Use telemetry (curiosity regime, basin distance, Φ) to decide

    Args:
        Same as HybridGeometricOptimizer, plus:
        adaptive_config: AdaptiveConfig object (from adaptive_gate.py)

    Example:
        >>> from .adaptive_gate import AdaptiveConfig
        >>> config = AdaptiveConfig(
        ...     min_kappa_for_ng=40.0,
        ...     min_basin_distance=0.6,
        ...     force_ng_every_n_steps=50,
        ... )
        >>> optimizer = AdaptiveMixedQIG(model, adaptive_config=config)
        >>> # In training loop:
        >>> should_use_ng = optimizer.should_apply_ng(telemetry, step)
        >>> loss.backward(create_graph=should_use_ng)
        >>> optimizer.step()
    """

    def __init__(
        self,
        model,
        lr_ng: float = 1e-2,
        lr_rest: float = 1e-3,
        cg_iters: int = 8,
        use_exact_ng: bool = True,
        rest_optimizer: str = "diagonal_ng",  # Ignored - kept for backwards compatibility
        weight_decay: float = 0.01,
        adaptive_config: Any | None = None,
    ):
        # Note: rest_optimizer parameter is ignored (always uses diagonal_ng)
        # Kept for backwards compatibility with old configs
        super().__init__(
            model,
            lr_ng=lr_ng,
            lr_rest=lr_rest,
            cg_iters=cg_iters,
            use_exact_ng=use_exact_ng,
            weight_decay=weight_decay,
        )

        # Import here to avoid circular dependency
        from .adaptive_gate import AdaptiveConfig
        from .adaptive_gate import should_use_ng as should_use_ng_fn

        self.adaptive_config = adaptive_config or AdaptiveConfig()
        self.should_use_ng_fn = should_use_ng_fn
        self.ng_application_count = 0

    def should_apply_ng(self, telemetry: dict, step: int) -> bool:
        """
        Decide whether to apply natural gradient based on telemetry.

        Args:
            telemetry: Current telemetry dict (from model forward pass)
            step: Current training step

        Returns:
            True if should use natural gradient, False otherwise
        """
        should_apply = self.should_use_ng_fn(telemetry, step, config=self.adaptive_config)

        if should_apply:
            self.ng_application_count += 1

        return should_apply

    def get_stats(self) -> dict:
        """Get stats including NG application frequency."""
        stats = super().get_stats()
        stats["ng_applications"] = self.ng_application_count
        return stats
