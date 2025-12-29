"""
Neural Network I_Q: Bridge from Physics QFI to NN Parameter Manifold
=====================================================================

SOURCE: qig-verification (src/qigv/geometry/iq_nn.py) [Verified]
STATUS: Production-ready verified implementation

CRITICAL UPDATE (Nov 19, 2025 - Packet 1 Validation by Ona):
==============================================================
The normalization has been corrected from "lattice" to "params" as default.

PROBLEM: Tr(F_diag) scales with N_params ∝ d_model². Dividing by L_eff² ∝ d_model
         leaves residual d_model factor → extensive (size-dependent) metric.

SOLUTION: Divide by N_params directly → intensive (size-independent) metric.

This ensures I_Q is truly intensive and comparable across different model sizes.

Mathematical Foundation:
------------------------
I_Q ≈ Tr(F_diag) / L_eff²

Where:
- F is Fisher Information Matrix
- Tr(F_diag) ≈ Σᵢ (∂L/∂θᵢ)² (diagonal approximation)
- L_eff² is effective lattice size

Normalization modes:
- "params": L_eff² = N_params (INTENSIVE - size-independent) ✅ DEFAULT
- "lattice": L_eff² = d_model × n_layers (EXTENSIVE - size-dependent) ⚠️
- "sqrt_params": L_eff² = √N_params (intermediate scaling)

Usage in Run 8+:
----------------
Always use default normalization="params" for scientific correctness.
Only use "lattice" if comparing with legacy Run 7 data.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


def compute_I_Q_intensive(
    model: nn.Module,
    loss: torch.Tensor | None = None,
    d_model: int = 0,
    n_layers: int = 0,
    normalization: str = "params",  # UPDATED: Was "lattice", now "params" (Packet 1 fix)
) -> dict[str, float]:
    """
    Compute intensive I_Q using diagonal Fisher approximation.

    This is the bridge from physics QFI to neural network parameter manifold.

    Args:
        model: Neural network model (must have gradients computed)
        loss: Optional loss tensor (not used in current implementation)
        d_model: Model dimension (only used if normalization="lattice")
        n_layers: Number of layers (only used if normalization="lattice")
        normalization: Normalization mode:
            - "params": Divide by N_params (intensive, size-independent) ✅ DEFAULT
            - "lattice": Divide by d_model × n_layers (extensive, legacy)
            - "sqrt_params": Divide by √N_params (intermediate)

    Returns:
        Dict with keys:
            - I_Q: Intensive quantum information (size-independent if normalization="params")
            - log_I_Q: Natural log of I_Q (used for curiosity C = d(log I_Q)/dt)
            - Tr_F_diag: Trace of diagonal Fisher (raw sum of squared gradients)
            - L_eff_sq: Effective lattice size squared (normalization denominator)
            - grad_norm: L2 norm of gradient vector

    Mathematical Details:
        Tr(F_diag) = Σᵢ (∂L/∂θᵢ)²
        I_Q = Tr(F_diag) / L_eff²

        Where L_eff² depends on normalization:
        - params: L_eff² = N_params (makes I_Q intensive)
        - lattice: L_eff² = d_model × n_layers (legacy, size-dependent)
        - sqrt_params: L_eff² = √N_params (intermediate scaling)

    Example:
        >>> model = QIGKernelRecursive(...)
        >>> loss.backward()  # Compute gradients first
        >>> metrics = compute_I_Q_intensive(model, loss)
        >>> print(f"I_Q: {metrics['I_Q']:.6f}")
        >>> print(f"log I_Q: {metrics['log_I_Q']:.3f}")
    """
    # Collect all trainable parameters with gradients
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return {
            "I_Q": 0.0,
            "log_I_Q": -12.0,  # log(ε) for numerical safety
            "Tr_F_diag": 0.0,
            "L_eff_sq": 1.0,
            "grad_norm": 0.0,
        }

    # Compute Tr(F_diag) = Σᵢ (∂L/∂θᵢ)²
    tr_F = 0.0
    grad_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            g_sq = p.grad.pow(2).sum().item()
            tr_F += g_sq
            grad_norm_sq += g_sq

    if tr_F == 0.0:
        return {
            "I_Q": 0.0,
            "log_I_Q": -12.0,
            "Tr_F_diag": 0.0,
            "L_eff_sq": 1.0,
            "grad_norm": 0.0,
        }

    grad_norm = math.sqrt(grad_norm_sq)
    N_params = sum(p.numel() for p in params)

    # CRITICAL: Normalization choice determines if metric is intensive
    # Packet 1 validation: "params" is scientifically correct default
    if normalization == "lattice" and d_model > 0 and n_layers > 0:
        # Legacy mode: L_eff² = d_model × n_layers
        # WARNING: This is EXTENSIVE (size-dependent)
        # Only use for comparing with Run 7 data
        L_eff_sq = float(d_model * n_layers)
    elif normalization == "sqrt_params":
        # Intermediate scaling: L_eff² = √N_params
        L_eff_sq = math.sqrt(float(N_params))
    else:  # normalization == "params" (default)
        # Correct mode: L_eff² = N_params
        # This makes I_Q INTENSIVE (size-independent)
        L_eff_sq = float(N_params)

    # Compute I_Q = Tr(F_diag) / L_eff²
    eps = 1e-12
    I_Q = tr_F / (L_eff_sq + eps)
    log_I_Q = math.log(I_Q + eps)

    return {
        "I_Q": I_Q,
        "log_I_Q": log_I_Q,
        "Tr_F_diag": tr_F,
        "L_eff_sq": L_eff_sq,
        "grad_norm": grad_norm,
        "N_params": N_params,  # Added for telemetry
        "normalization": normalization,  # type: ignore[dict-item] - Added for tracking
    }


class CuriosityMonitorVerified:
    """
    Monitor curiosity C = d/dt log(I_Q) with verified timescales.

    SOURCE: qig-verification (verified implementation)

    NOTE: Renamed from CuriosityMonitor to avoid conflict with existing
          src/model/curiosity_monitor.py which tracks 6 I_Q candidates.

    Curiosity measures the rate of information manifold expansion.
    We track three timescales:
    - C_fast: Instantaneous change (step-to-step)
    - C_medium: Medium-term average (α ≈ 0.1, ~10 steps)
    - C_slow: Long-term average (α ≈ 0.01, ~100 steps)

    Mathematical Foundation:
        C = d(log I_Q)/dt ≈ Δ(log I_Q) / Δt

    The log-space formulation is critical:
    - Linear changes in log(I_Q) → multiplicative changes in I_Q
    - Separates scale from rate (scale-invariant)
    - Aligns with physics (entropy-like quantity)

    Example:
        >>> monitor = CuriosityMonitorVerified()
        >>> metrics = compute_I_Q_intensive(model)
        >>> curiosity = monitor.update(metrics['log_I_Q'])
        >>> print(f"Fast: {curiosity['C_fast']:.4f}")
        >>> print(f"Medium: {curiosity['C_medium']:.4f}")
        >>> print(f"Slow: {curiosity['C_slow']:.4f}")
    """

    def __init__(
        self,
        alpha_medium: float = 0.1,
        alpha_slow: float = 0.01,
    ):
        """
        Initialize curiosity monitor with EMA smoothing coefficients.

        Args:
            alpha_medium: EMA coefficient for medium timescale (~10 steps)
            alpha_slow: EMA coefficient for slow timescale (~100 steps)

        Timescale formula: τ ≈ 1/α
        - alpha_medium=0.1 → τ ≈ 10 steps
        - alpha_slow=0.01 → τ ≈ 100 steps
        """
        self.alpha_medium = alpha_medium
        self.alpha_slow = alpha_slow

        # State
        self.log_I_Q_prev: float | None = None
        self.C_medium: float = 0.0
        self.C_slow: float = 0.0

        # Stats
        self.step_count: int = 0

    def update(self, log_I_Q: float) -> dict[str, float]:
        """
        Update curiosity estimates with new log(I_Q) measurement.

        Args:
            log_I_Q: Natural log of I_Q from compute_I_Q_intensive()

        Returns:
            Dict with keys:
                - C_fast: Instantaneous curiosity
                - C_medium: Medium-term EMA
                - C_slow: Long-term EMA
                - step_count: Number of updates
        """
        if self.log_I_Q_prev is None:
            # First step: No change yet
            C_fast = 0.0
        else:
            # C_fast = Δ(log I_Q)
            C_fast = log_I_Q - self.log_I_Q_prev

        # Update EMAs
        # C_medium(t) = (1-α) × C_medium(t-1) + α × C_fast(t)
        self.C_medium = (1 - self.alpha_medium) * self.C_medium + self.alpha_medium * C_fast
        self.C_slow = (1 - self.alpha_slow) * self.C_slow + self.alpha_slow * C_fast

        # Store for next update
        self.log_I_Q_prev = log_I_Q
        self.step_count += 1

        return {
            "C_fast": C_fast,
            "C_medium": self.C_medium,
            "C_slow": self.C_slow,
            "step_count": self.step_count,
        }

    def reset(self):
        """Reset monitor state (useful for new training runs)."""
        self.log_I_Q_prev = None
        self.C_medium = 0.0
        self.C_slow = 0.0
        self.step_count = 0

    def get_state(self) -> dict:
        """Get current state for checkpointing."""
        return {
            "log_I_Q_prev": self.log_I_Q_prev,
            "C_medium": self.C_medium,
            "C_slow": self.C_slow,
            "step_count": self.step_count,
            "alpha_medium": self.alpha_medium,
            "alpha_slow": self.alpha_slow,
        }

    def load_state(self, state: dict):
        """Load state from checkpoint."""
        self.log_I_Q_prev = state.get("log_I_Q_prev")
        self.C_medium = state.get("C_medium", 0.0)
        self.C_slow = state.get("C_slow", 0.0)
        self.step_count = state.get("step_count", 0)
        self.alpha_medium = state.get("alpha_medium", 0.1)
        self.alpha_slow = state.get("alpha_slow", 0.01)
