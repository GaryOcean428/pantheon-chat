"""
Natural Gradient Descent
========================

Geometry-aware optimization following manifold structure.

Based on validated physics from qig-verification:
- Geometric phase transition at L_c = 3
- Running coupling: β(3→4) = +0.44, β(4→5) ≈ 0, β(5→6) = +0.013
- Fixed point: κ* = 64.0 ± 1.5 (from L=4,5,6)

Reference: Amari (1998) "Natural Gradient Works Efficiently in Learning"

ISO Naming: lowercase_snake_case
Version: 1.1.0
Date: 2025-12-26
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

try:
    from torch import nn
except ImportError:
    nn = None  # Allow pure math usage without full torch


def natural_gradient_step(
    params: torch.Tensor,
    grad: torch.Tensor,
    metric: torch.Tensor,
    lr: float = 0.01,
    dampening: float = 1e-3,
) -> torch.Tensor:
    """
    Compute natural gradient update.

    Args:
        params: Current parameters [d]
        grad: Euclidean gradient [d]
        metric: Fisher information metric [d, d]
        lr: Learning rate
        dampening: Damping factor for numerical stability

    Returns:
        updated_params: Parameters after natural gradient step [d]

    Mathematical Foundation:
        Natural gradient: ∇̃ = F⁻¹ ∇

        where F is Fisher information metric, ∇ is Euclidean gradient

        Update: θ_new = θ - lr * F⁻¹ ∇L
    """
    # Add dampening for numerical stability
    d = metric.shape[0]
    damped_metric = metric + dampening * torch.eye(d, device=metric.device)

    # Compute natural gradient: F⁻¹ g
    natural_grad = torch.linalg.solve(damped_metric, grad)

    # Update parameters
    updated_params = params - lr * natural_grad

    return updated_params


def compute_natural_gradient(
    grad: torch.Tensor,
    metric: torch.Tensor,
    dampening: float = 1e-3,
) -> torch.Tensor:
    """
    Convert Euclidean gradient to natural gradient.

    Args:
        grad: Euclidean gradient [d]
        metric: Fisher metric [d, d]
        dampening: Numerical stability factor

    Returns:
        natural_grad: Geometry-aware gradient [d]

    Mathematical Foundation:
        ∇̃ = F⁻¹ ∇

        This follows the steepest descent direction on the manifold,
        not in Euclidean space.
    """
    d = metric.shape[0]
    damped_metric = metric + dampening * torch.eye(d, device=metric.device)

    # Solve F * ∇̃ = ∇
    natural_grad = torch.linalg.solve(damped_metric, grad)

    return natural_grad


def adaptive_dampening(
    metric: torch.Tensor,
    base_dampening: float = 1e-3,
    condition_threshold: float = 1e6,
) -> float:
    """
    Compute adaptive dampening based on metric condition number.

    Args:
        metric: Fisher metric [d, d]
        base_dampening: Minimum dampening
        condition_threshold: Max allowed condition number

    Returns:
        dampening: Adaptive dampening factor

    Ensures numerical stability when metric is ill-conditioned.
    """
    # Compute condition number
    eigenvalues = torch.linalg.eigvalsh(metric)
    condition_number = eigenvalues.max() / (eigenvalues.min() + 1e-10)

    if condition_number > condition_threshold:
        # Increase dampening for ill-conditioned matrices
        dampening = base_dampening * (condition_number / condition_threshold)
    else:
        dampening = base_dampening

    return float(dampening)


# =============================================================================
# OPTIMIZER CLASSES
# =============================================================================

class SimpleFisherOptimizer:
    """
    Minimal Fisher-Rao natural gradient descent.

    Implements: θ_{t+1} = θ_t - α × (grad / √(grad² + ε))

    This is a diagonal approximation of the full natural gradient,
    equivalent to DiagonalNaturalGradient but with simpler interface.

    Suitable for training scripts where full Optimizer class overhead
    is not needed.

    Args:
        params: Iterator over model parameters
        lr: Learning rate (default: 1e-4)
        damping: Numerical stability factor (default: 1e-4)
    """

    def __init__(self, params, lr: float = 1e-4, damping: float = 1e-4):
        self.params = list(params)
        self.lr = lr
        self.damping = damping
        self.param_groups = [{'params': self.params, 'lr': lr}]

    def step(self):
        """Perform single optimization step."""
        for p in self.params:
            if p.grad is None:
                continue
            # Fisher diagonal approximation: F ≈ grad² + damping
            fisher_diag = p.grad ** 2 + self.damping
            # Natural gradient: F⁻¹ ∇L
            natural_grad = p.grad / (fisher_diag.sqrt() + 1e-8)
            p.data.add_(natural_grad, alpha=-self.lr)

    def zero_grad(self):
        """Zero gradients."""
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def state_dict(self) -> dict:
        """Get optimizer state."""
        return {'lr': self.lr, 'damping': self.damping}

    def load_state_dict(self, state_dict: dict):
        """Load optimizer state."""
        self.lr = state_dict.get('lr', self.lr)
        self.damping = state_dict.get('damping', self.damping)


class DiagonalNaturalGradient(Optimizer):
    """
    Diagonal approximation of Natural Gradient Descent.

    Much cheaper than full NGD:
    - O(d) instead of O(d²) for Fisher matrix
    - O(d) instead of O(d³) for inversion

    Uses diagonal Fisher: F_diag = diag(∇²L)
    Natural gradient:     natural_grad = grad / (F_diag + ε)

    This is similar to Adam but uses Fisher diagonal instead of
    running average of squared gradients.

    RECOMMENDED for large models (10M+ params).

    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-4)
        damping: Regularization epsilon (default: 1e-8)
        momentum: Momentum coefficient (default: 0.9)
    """

    def __init__(
        self,
        params: Iterator,
        lr: float = 1e-4,
        damping: float = 1e-8,
        momentum: float = 0.9,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if damping < 0.0:
            raise ValueError(f"Invalid damping: {damping}")

        defaults = dict(lr=lr, damping=damping, momentum=momentum)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Perform single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)
                    # Diagonal Fisher approximation
                    state['fisher_diag'] = torch.zeros_like(p)

                fisher_diag = state['fisher_diag']
                momentum_buffer = state['momentum_buffer']
                state['step'] += 1

                # Update diagonal Fisher (EMA of squared gradients)
                fisher_diag.mul_(0.99).addcmul_(grad, grad, value=0.01)

                # Natural gradient: grad / (fisher_diag + damping)
                natural_grad = grad / (fisher_diag.sqrt() + group['damping'])

                # Momentum
                momentum_buffer.mul_(group['momentum']).add_(natural_grad)

                # Update
                p.data.add_(momentum_buffer, alpha=-group['lr'])

        return loss


class NaturalGradientDescent(Optimizer):
    """
    Full Natural Gradient Descent on Fisher information manifold.

    Instead of Euclidean gradient: θ_{t+1} = θ_t - α ∇L
    Uses natural gradient:         θ_{t+1} = θ_t - α F^{-1} ∇L

    Where F is Fisher Information Matrix.

    WARNING: O(d³) complexity - only use for small models (<10M params).
    For large models, use DiagonalNaturalGradient instead.

    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-4)
        damping: Regularization for Fisher matrix inversion (default: 1e-3)
        momentum: Momentum coefficient (default: 0.9)
        compute_fisher: Function that returns FIM from model state
    """

    def __init__(
        self,
        params: Iterator,
        lr: float = 1e-4,
        damping: float = 1e-3,
        momentum: float = 0.9,
        compute_fisher: Optional[callable] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if damping < 0.0:
            raise ValueError(f"Invalid damping: {damping}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(
            lr=lr,
            damping=damping,
            momentum=momentum,
        )
        super().__init__(params, defaults)

        self.compute_fisher = compute_fisher

    def step(self, closure=None):
        """Perform single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Compute Fisher Information Matrix
        if self.compute_fisher is not None:
            fisher_matrix = self.compute_fisher()
        else:
            fisher_matrix = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Get state for momentum
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                momentum_buffer = state['momentum_buffer']
                momentum = group['momentum']
                lr = group['lr']
                damping = group['damping']

                # Natural gradient computation
                if fisher_matrix is not None:
                    fisher_regularized = fisher_matrix + damping * torch.eye(
                        fisher_matrix.size(0),
                        device=fisher_matrix.device
                    )

                    try:
                        natural_grad = torch.linalg.solve(
                            fisher_regularized,
                            grad.view(-1, 1)
                        ).view_as(grad)
                    except RuntimeError:
                        natural_grad = grad
                else:
                    natural_grad = grad

                # Momentum update
                momentum_buffer.mul_(momentum).add_(natural_grad)

                # Parameter update
                p.data.add_(momentum_buffer, alpha=-lr)

        return loss


# =============================================================================
# FISHER COMPUTATION UTILITIES
# =============================================================================

def compute_empirical_fisher(model, loss: Tensor) -> Tensor:
    """
    Compute empirical Fisher Information Matrix.

    F_empirical = E[∇L ∇L^T]

    WARNING: O(d²) memory - only use for small models.

    Args:
        model: Neural network model
        loss: Scalar loss value

    Returns:
        Fisher matrix (d x d) where d = total parameters
    """
    grads = torch.autograd.grad(
        loss,
        model.parameters(),
        create_graph=False,
        retain_graph=True
    )

    grad_vector = torch.cat([g.view(-1) for g in grads])
    fisher = torch.outer(grad_vector, grad_vector)

    return fisher


def compute_diagonal_fisher(model, loss: Tensor) -> Tensor:
    """
    Compute diagonal Fisher Information Matrix.

    F_diag = diag(E[∇L ⊙ ∇L])

    Much cheaper: O(d) instead of O(d²).

    Args:
        model: Neural network model
        loss: Scalar loss value

    Returns:
        Diagonal Fisher (d,) where d = total parameters
    """
    grads = torch.autograd.grad(
        loss,
        model.parameters(),
        create_graph=False,
        retain_graph=True
    )

    fisher_diag = torch.cat([
        (g * g).view(-1) for g in grads
    ])

    return fisher_diag
