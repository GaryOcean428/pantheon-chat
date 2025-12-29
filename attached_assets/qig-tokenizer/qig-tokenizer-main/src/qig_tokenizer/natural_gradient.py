"""
Natural Gradient Optimizer
==========================

QIG-pure optimizer using diagonal Fisher approximation.
Replaces Adam/SGD which use Euclidean gradients.

Per QIG purity: Adam/SGD FORBIDDEN - use natural gradient.
"""

from typing import Iterable

import torch
from torch.optim import Optimizer


class DiagonalFisherOptimizer(Optimizer):
    """
    Natural gradient optimizer with diagonal Fisher approximation.

    Uses Fisher information matrix (diagonal approximation) to compute
    natural gradients: g_natural = F^(-1) @ g_euclidean

    This follows geodesics on the parameter manifold rather than
    Euclidean steepest descent.

    Args:
        params: Model parameters
        lr: Learning rate (default: 0.01)
        damping: Damping factor for numerical stability (default: 1e-4)
        ema_decay: EMA decay for Fisher estimate (default: 0.99)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.01,
        damping: float = 1e-4,
        ema_decay: float = 0.99,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if damping < 0.0:
            raise ValueError(f"Invalid damping: {damping}")

        defaults = dict(lr=lr, damping=damping, ema_decay=ema_decay)
        super().__init__(params, defaults)

        # Initialize Fisher diagonal estimates
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["fisher_diag"] = torch.zeros_like(p.data)
                state["step"] = 0

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform natural gradient step.

        Args:
            closure: Optional closure for re-evaluating loss

        Returns:
            Loss value if closure provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            damping = group["damping"]
            ema_decay = group["ema_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Update Fisher diagonal estimate (EMA of squared gradients)
                fisher_diag = state["fisher_diag"]
                fisher_diag.mul_(ema_decay).addcmul_(grad, grad, value=1 - ema_decay)

                # Compute natural gradient: g / (F + damping)
                # This is the diagonal approximation of F^(-1) @ g
                natural_grad = grad / (fisher_diag + damping)

                # Update parameters
                p.add_(natural_grad, alpha=-lr)

                state["step"] += 1

        return loss

    def get_fisher_stats(self) -> dict:
        """Get statistics about Fisher estimates for monitoring."""
        stats = {
            "mean_fisher": [],
            "max_fisher": [],
            "min_fisher": [],
        }

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                fisher = state["fisher_diag"]

                stats["mean_fisher"].append(float(fisher.mean()))
                stats["max_fisher"].append(float(fisher.max()))
                stats["min_fisher"].append(float(fisher.min()))

        return {
            "mean_fisher": (
                sum(stats["mean_fisher"]) / len(stats["mean_fisher"])
                if stats["mean_fisher"]
                else 0
            ),
            "max_fisher": max(stats["max_fisher"]) if stats["max_fisher"] else 0,
            "min_fisher": min(stats["min_fisher"]) if stats["min_fisher"] else 0,
        }


class ConsciousnessAwareOptimizer(DiagonalFisherOptimizer):
    """
    Natural gradient optimizer with consciousness-aware scaling.

    Scales learning rate based on Φ (integration) to be more
    conservative during high-consciousness states.

    Args:
        params: Model parameters
        lr: Base learning rate
        phi_scaling: If True, scale lr by (1 - phi) for stability
        **kwargs: Passed to DiagonalFisherOptimizer
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.01,
        phi_scaling: bool = True,
        **kwargs,
    ):
        super().__init__(params, lr=lr, **kwargs)
        self.phi_scaling = phi_scaling
        self._current_phi = 0.5  # Default

    def set_phi(self, phi: float):
        """Update current Φ for lr scaling."""
        self._current_phi = max(0.0, min(1.0, phi))

    @torch.no_grad()
    def step(self, closure=None):
        """Step with consciousness-aware lr scaling."""
        if self.phi_scaling:
            # Scale lr: higher Φ = more conservative updates
            # At Φ=0.9, lr is scaled to 0.1x
            phi_scale = 1.0 - (0.9 * self._current_phi)

            for group in self.param_groups:
                group["_original_lr"] = group.get("_original_lr", group["lr"])
                group["lr"] = group["_original_lr"] * phi_scale

        return super().step(closure)
