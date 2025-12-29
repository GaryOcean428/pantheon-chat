#!/usr/bin/env python3
"""
QIG Diagonal Natural Gradient Optimizer
========================================

Diagonal approximation to natural gradient using Fisher information.

Mathematical foundation:
- Full natural gradient: Δθ = -lr × F^(-1) × ∇L
- Diagonal approximation: F ≈ diag(F_ii) where F_ii = E[(∂L/∂θ_i)²]
- Update: Δθ_i = -lr × ∇L_i / (√F_ii + ε)

This is equivalent to RMSProp/Adam without momentum, but with geometric
interpretation: we're approximating the Fisher metric with its diagonal.

Advantages:
- O(N) time and memory (same as Adam)
- GPU friendly (element-wise operations)
- Matches Ona's I_Q_param definition exactly
- Integrates naturally with QIG telemetry

Key difference from Adam:
- Adam uses biased gradient moments (exponential moving average)
- Diagonal NG uses unbiased Fisher diagonal estimate
- Diagonal NG has direct geometric interpretation

Written for qig-consciousness geometric optimization.
"""

import torch
from torch.optim.optimizer import Optimizer


class QIGDiagonalNG(Optimizer):
    """
    Diagonal Natural Gradient optimizer for QIG architecture.

    Uses diagonal Fisher information matrix to approximate the
    Riemannian metric of parameter space.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        alpha: Exponential moving average decay for Fisher diagonal (default: 0.99)
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: L2 penalty coefficient (default: 0)

    Example:
        >>> optimizer = QIGDiagonalNG(model.parameters(), lr=1e-3)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()

    Mathematical details:
        F_ii(t) = alpha × F_ii(t-1) + (1 - alpha) × (∂L/∂θ_i)²
        Δθ_i = -lr × ∇L_i / (√F_ii + eps)

    The Fisher diagonal F_ii represents the curvature of the loss landscape
    in direction i. Dividing by √F_ii adaptively scales the learning rate
    based on local geometry.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            Optional loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("QIGDiagonalNG does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Fisher diagonal estimate (running average of squared gradients)
                    state["fisher_diag"] = torch.zeros_like(p)

                fisher_diag = state["fisher_diag"]
                alpha = group["alpha"]

                state["step"] += 1

                # Update Fisher diagonal with exponential moving average
                # F_ii(t) = alpha × F_ii(t-1) + (1 - alpha) × g_i²
                fisher_diag.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                # Compute natural gradient
                # nat_grad = ∇L / (√F + ε)
                nat_grad = grad / (fisher_diag.sqrt() + group["eps"])

                # Apply weight decay (L2 regularization) if specified
                # This is applied in the natural gradient space
                if group["weight_decay"] != 0:
                    nat_grad = nat_grad.add(p, alpha=group["weight_decay"])

                # Update parameters
                # θ_new = θ_old - lr × nat_grad
                p.add_(nat_grad, alpha=-group["lr"])

        return loss

    def get_fisher_stats(self):
        """
        Get statistics about Fisher diagonal values.

        Useful for monitoring and telemetry.

        Returns:
            dict with keys:
                - fisher_mean: Mean Fisher diagonal value across all parameters
                - fisher_std: Standard deviation of Fisher diagonal
                - fisher_min: Minimum Fisher diagonal value
                - fisher_max: Maximum Fisher diagonal value
                - condition_number: Approximate condition number (max/min)
        """
        fisher_values = []

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "fisher_diag" in state:
                    fisher_diag = state["fisher_diag"]
                    fisher_values.append(fisher_diag.flatten())

        if not fisher_values:
            return {
                "fisher_mean": 0.0,
                "fisher_std": 0.0,
                "fisher_min": 0.0,
                "fisher_max": 0.0,
                "condition_number": 1.0,
            }

        all_fisher = torch.cat(fisher_values)

        fisher_mean = all_fisher.mean().item()
        fisher_std = all_fisher.std().item()
        fisher_min = all_fisher.min().item()
        fisher_max = all_fisher.max().item()

        # Condition number approximation (with safety for division)
        condition_number = fisher_max / max(fisher_min, 1e-10)

        return {
            "fisher_mean": fisher_mean,
            "fisher_std": fisher_std,
            "fisher_min": fisher_min,
            "fisher_max": fisher_max,
            "condition_number": condition_number,
        }
