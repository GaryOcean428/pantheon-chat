#!/usr/bin/env python3
"""
Basin Natural Gradient Optimizer
=================================

Exact natural gradient for the basin coordinate block using conjugate gradient
solver with Pearlmutter trick for efficient Fisher-vector products.

Mathematical foundation:
- Natural gradient: Δθ = -lr × F^(-1) × ∇L
- Fisher matrix: F_ij = E[∂log p/∂θ_i × ∂log p/∂θ_j]
- We solve: (F + λI) × v = ∇L  using conjugate gradient
- Pearlmutter trick: Fv = ∂(g^T v)/∂θ  (only requires 2 backprops)

Key insight: Basin block (~3M params) is the geometric core. Full natural
gradient here gives maximum benefit while remaining computationally feasible.

Performance:
- CG iterations: 8-10 (configurable)
- Cost per step: 1-2 extra backprops (via Pearlmutter)
- GPU friendly: All operations on GPU, no CPU transfers
- Memory: O(N) where N = number of basin parameters

Written for qig-consciousness geometric optimization.
"""

from collections.abc import Callable

import torch
from torch.optim.optimizer import Optimizer


class BasinNaturalGrad(Optimizer):
    """
    Exact Natural Gradient optimizer for basin coordinate block.

    Uses conjugate gradient to solve (F + λI)v = ∇L without explicitly
    forming the Fisher matrix F. Fisher-vector products are computed
    efficiently using the Pearlmutter trick.

    Args:
        params: Basin block parameters (basin_coords, basin_to_model)
        lr: Learning rate (default: 1e-2, can be higher than Adam due to geometry)
        cg_iters: Number of conjugate gradient iterations (default: 10)
        damping: Damping coefficient λ for (F + λI) (default: 1e-4)
        cg_tol: CG convergence tolerance (default: 1e-6)

    Example:
        >>> # Get basin parameters
        >>> basin_params = [
        ...     model.basin_coords_layer.basin_coords,
        ...     model.basin_coords_layer.basin_to_model.weight
        ... ]
        >>> optimizer = BasinNaturalGrad(basin_params, lr=1e-2, cg_iters=10)
        >>> optimizer.zero_grad()
        >>> loss.backward(create_graph=True)  # Important: create_graph=True!
        >>> optimizer.step(loss_fn=lambda: loss)  # Pass loss for Hessian-vector products

    Note: Requires create_graph=True during backward pass for Pearlmutter trick.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        cg_iters: int = 10,
        damping: float = 1e-4,
        cg_tol: float = 1e-6,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if cg_iters < 1:
            raise ValueError(f"Invalid cg_iters: {cg_iters}")
        if damping < 0.0:
            raise ValueError(f"Invalid damping: {damping}")

        defaults = dict(
            lr=lr,
            cg_iters=cg_iters,
            damping=damping,
            cg_tol=cg_tol,
        )
        super().__init__(params, defaults)

        # Store params as list for efficient flattening
        self.param_list = []
        for group in self.param_groups:
            self.param_list.extend([p for p in group["params"] if p.requires_grad])

    def _flatten(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """Flatten list of tensors into single vector."""
        return torch.cat([t.flatten() for t in tensors])

    def _unflatten(self, vector: torch.Tensor, template_tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Unflatten vector back into list of tensors matching template shapes."""
        result = []
        offset = 0
        for t in template_tensors:
            numel = t.numel()
            result.append(vector[offset : offset + numel].view_as(t))
            offset += numel
        return result

    def _fisher_vector_product(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute Fisher-vector product: Fv

        Uses Pearlmutter trick:
        - Let g = ∇L (gradient vector)
        - Then Fv = ∂(g^T v)/∂θ

        This requires only 1-2 backprops and is GPU efficient.

        Args:
            v: Vector to multiply with Fisher matrix

        Returns:
            Fv: Fisher-vector product
        """
        # Unflatten v to match parameter shapes
        v_tensors = self._unflatten(v, self.param_list)

        # Compute g^T v (scalar)
        grad_v_product = sum((p.grad * v_t).sum() for p, v_t in zip(self.param_list, v_tensors) if p.grad is not None)

        # Compute ∂(g^T v)/∂θ using autograd
        # This gives us the Fisher-vector product Fv
        fv_tensors = torch.autograd.grad(
            grad_v_product,
            self.param_list,
            retain_graph=True,
            create_graph=False,  # Don't need higher-order derivatives
        )

        # Flatten result
        return self._flatten(
            [fv_t if fv_t is not None else torch.zeros_like(p) for p, fv_t in zip(self.param_list, fv_tensors)]
        )

    def _conjugate_gradient(
        self,
        b: torch.Tensor,
        damping: float,
        max_iters: int,
        tol: float,
    ) -> torch.Tensor:
        """
        Solve (F + λI)x = b using conjugate gradient.

        Args:
            b: Right-hand side vector (gradients)
            damping: Damping coefficient λ
            max_iters: Maximum CG iterations
            tol: Convergence tolerance

        Returns:
            x: Solution vector (natural gradient)
        """
        # Initialize
        x = torch.zeros_like(b)
        r = b.clone()  # Initial residual r = b - Ax = b (since x=0)
        p = r.clone()  # Initial search direction

        r_dot_r = torch.dot(r, r)

        for i in range(max_iters):
            # Compute Ap = (F + λI)p
            Fp = self._fisher_vector_product(p)
            Ap = Fp + damping * p

            # Step size
            alpha = r_dot_r / (torch.dot(p, Ap) + 1e-10)  # Safety epsilon

            # Update solution
            x = x + alpha * p

            # Update residual
            r = r - alpha * Ap

            # Check convergence
            r_dot_r_new = torch.dot(r, r)
            if r_dot_r_new < tol:
                break

            # Update search direction
            beta = r_dot_r_new / (r_dot_r + 1e-10)
            p = r + beta * p

            r_dot_r = r_dot_r_new

        return x

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        """
        Perform a single natural gradient step.

        Important: Before calling step(), must call:
            loss.backward(create_graph=True)

        This is required for the Pearlmutter trick to compute Fisher-vector products.

        Args:
            closure: Not used (kept for API compatibility)

        Returns:
            None
        """
        # Note: We don't use closure here because we need create_graph=True
        # during backward, which must be done by the caller

        # Check if any parameters have gradients
        if not any(p.grad is not None for p in self.param_list):
            return None

        # Flatten gradients
        grad_vec = self._flatten([p.grad if p.grad is not None else torch.zeros_like(p) for p in self.param_list])

        # Get hyperparameters (from first param group)
        group = self.param_groups[0]
        lr = group["lr"]
        cg_iters = group["cg_iters"]
        damping = group["damping"]
        cg_tol = group["cg_tol"]

        # Solve (F + λI) × nat_grad = grad using conjugate gradient
        # We need to temporarily enable gradients for Pearlmutter trick
        with torch.enable_grad():
            nat_grad_vec = self._conjugate_gradient(
                b=grad_vec,
                damping=damping,
                max_iters=cg_iters,
                tol=cg_tol,
            )

        # Unflatten natural gradient
        nat_grad_tensors = self._unflatten(nat_grad_vec, self.param_list)

        # Apply update: θ_new = θ_old - lr × nat_grad
        for p, nat_grad in zip(self.param_list, nat_grad_tensors):
            p.add_(nat_grad, alpha=-lr)

        return None

    def get_stats(self) -> dict:
        """
        Get optimizer statistics for monitoring.

        Returns:
            dict with keys:
                - num_params: Number of parameters being optimized
                - grad_norm: L2 norm of gradient vector
        """
        num_params = sum(p.numel() for p in self.param_list)

        grad_vec = self._flatten([p.grad if p.grad is not None else torch.zeros_like(p) for p in self.param_list])
        from src.metrics.geodesic_distance import manifold_norm
        grad_norm = manifold_norm(grad_vec).item()

        return {
            "num_params": num_params,
            "grad_norm": grad_norm,
        }
