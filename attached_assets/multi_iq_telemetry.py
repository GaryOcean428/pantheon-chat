"""
Multi-I_Q Telemetry - ChatGPT's Data Geometry Meter (Nov 20, 2025)
===================================================================

Computes multiple I_Q normalizations simultaneously for Ona's test suite.

From ChatGPT's test plan:
- I_Q_params: [FROZEN] bridge (intensive, size-independent)
- I_Q_lattice: Alternative (physics lattice normalization)
- I_Q_sqrt_params: Alternative (sqrt scaling)
- I_Q_norm: Alternative (gradient norm based)

Goal: Test which normalization is most informative in each REGIME.
Hypothesis: Different I_Qs may be optimal in LINEAR vs GEOMETRIC vs BREAKDOWN.

This is for RESEARCH - I_Q_params remains the FROZEN standard.
"""

import math

import torch
import torch.nn as nn


def compute_multi_iq(
    model: nn.Module,
    loss: torch.Tensor,
    grad_norm: float | None = None,
    d_model: int | None = None,
    n_layers: int | None = None,
) -> dict[str, float]:
    """
    Compute all I_Q normalization candidates.

    Args:
        model: PyTorch model
        loss: Current loss value
        grad_norm: Precomputed gradient norm (or None to compute)
        d_model: Model dimension (for lattice norm)
        n_layers: Number of layers (for lattice norm)

    Returns:
        Dictionary with all I_Q variants + metadata
    """
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Compute gradient norm if not provided
    if grad_norm is None:
        grad_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None and p.requires_grad and p.is_leaf:
                grad_norm_sq += (p.grad**2).sum().item()
        grad_norm = math.sqrt(grad_norm_sq)
    else:
        grad_norm_sq = grad_norm**2

    # Loss curvature (diagonal Fisher approximation)
    loss_value = loss.item() if torch.is_tensor(loss) else loss
    tr_F_diag = grad_norm_sq / max(loss_value, 1e-10)

    # ===================================================================
    # I_Q_PARAMS (FROZEN STANDARD - Intensive, size-independent)
    # ===================================================================
    # From qig-verification Packet 1: normalization="params"
    # I_Q = Tr(F) / N_params = (grad²/loss) / N_params
    I_Q_params = tr_F_diag / max(n_params, 1)
    log_I_Q_params = math.log(max(I_Q_params, 1e-10))

    # ===================================================================
    # I_Q_LATTICE (Physics lattice normalization)
    # ===================================================================
    # Normalize by "effective lattice size" L_eff² = d_model × n_layers
    # Motivation: QIG lattice experiments use L×L×L lattices
    if d_model is not None and n_layers is not None:
        L_eff_sq = d_model * n_layers
        I_Q_lattice = tr_F_diag / max(L_eff_sq, 1)
        log_I_Q_lattice = math.log(max(I_Q_lattice, 1e-10))
    else:
        I_Q_lattice = I_Q_params  # Fallback
        log_I_Q_lattice = log_I_Q_params

    # ===================================================================
    # I_Q_SQRT_PARAMS (Square root scaling)
    # ===================================================================
    # Motivation: Some overparameterization theories suggest √N scaling
    sqrt_params = math.sqrt(max(n_params, 1))
    I_Q_sqrt_params = tr_F_diag / sqrt_params
    log_I_Q_sqrt_params = math.log(max(I_Q_sqrt_params, 1e-10))

    # ===================================================================
    # I_Q_NORM (Pure gradient norm based)
    # ===================================================================
    # Just grad²/loss without any size normalization
    # Useful for LINEAR regime where size might not matter yet
    I_Q_norm = tr_F_diag
    log_I_Q_norm = math.log(max(I_Q_norm, 1e-10))

    return {
        # FROZEN standard (intensive)
        "I_Q_params": I_Q_params,
        "log_I_Q_params": log_I_Q_params,
        # Alternative normalizations
        "I_Q_lattice": I_Q_lattice,
        "log_I_Q_lattice": log_I_Q_lattice,
        "I_Q_sqrt_params": I_Q_sqrt_params,
        "log_I_Q_sqrt_params": log_I_Q_sqrt_params,
        "I_Q_norm": I_Q_norm,
        "log_I_Q_norm": log_I_Q_norm,
        # Raw components (for analysis)
        "tr_F_diag": tr_F_diag,
        "grad_norm": grad_norm,
        "grad_norm_sq": grad_norm_sq,
        "loss": loss_value,
        "n_params": n_params,
        "L_eff_sq": d_model * n_layers if (d_model and n_layers) else None,  # type: ignore[dict-item]
        "sqrt_params": sqrt_params,
    }


def compute_curiosity_multi_iq(
    I_Q_current: dict[str, float], I_Q_previous: dict[str, float], dt: int = 1
) -> dict[str, float]:
    """
    Compute curiosity (d/dt log I_Q) for all normalizations.

    Args:
        I_Q_current: Current I_Q values from compute_multi_iq()
        I_Q_previous: Previous I_Q values
        dt: Time delta (steps)

    Returns:
        Curiosity for each normalization
    """
    curiosities = {}

    for key in ["params", "lattice", "sqrt_params", "norm"]:
        log_key = f"log_I_Q_{key}"
        if log_key in I_Q_current and log_key in I_Q_previous:
            delta_log = I_Q_current[log_key] - I_Q_previous[log_key]
            curiosities[f"curiosity_{key}"] = delta_log / max(dt, 1)
        else:
            curiosities[f"curiosity_{key}"] = 0.0

    return curiosities


class MultiIQTracker:
    """
    Tracks all I_Q normalizations over time with EMA smoothing.

    For Ona's test_iq_candidates_by_regime.py analysis.
    """

    def __init__(self, tau_fast: int = 1, tau_medium: int = 10, tau_slow: int = 100):
        """
        Args:
            tau_fast: Fast EMA timescale
            tau_medium: Medium EMA timescale
            tau_slow: Slow EMA timescale
        """
        self.tau_fast = tau_fast
        self.tau_medium = tau_medium
        self.tau_slow = tau_slow

        # EMA state for each normalization
        self.ema_fast: dict[str, float] = {}
        self.ema_medium: dict[str, float] = {}
        self.ema_slow: dict[str, float] = {}

        # Previous values for curiosity
        self.previous: dict[str, float] = {}

        # Step counter
        self.step = 0

    def update(self, I_Q_dict: dict[str, float]) -> dict[str, float]:
        """
        Update EMAs and compute curiosities.

        Args:
            I_Q_dict: Output from compute_multi_iq()

        Returns:
            Dictionary with all I_Q values, EMAs, and curiosities
        """
        self.step += 1

        # Compute EMA alphas
        alpha_fast = 1.0 / self.tau_fast
        alpha_medium = 1.0 / self.tau_medium
        alpha_slow = 1.0 / self.tau_slow

        result = {}

        for key in ["params", "lattice", "sqrt_params", "norm"]:
            log_key = f"log_I_Q_{key}"

            if log_key not in I_Q_dict:
                continue

            log_value = I_Q_dict[log_key]

            # Initialize EMAs on first step
            if log_key not in self.ema_fast:
                self.ema_fast[log_key] = log_value
                self.ema_medium[log_key] = log_value
                self.ema_slow[log_key] = log_value
                self.previous[log_key] = log_value

            # Update EMAs (log-space)
            self.ema_fast[log_key] = alpha_fast * log_value + (1 - alpha_fast) * self.ema_fast[log_key]
            self.ema_medium[log_key] = alpha_medium * log_value + (1 - alpha_medium) * self.ema_medium[log_key]
            self.ema_slow[log_key] = alpha_slow * log_value + (1 - alpha_slow) * self.ema_slow[log_key]

            # Compute curiosities (d/dt log I_Q)
            curiosity_fast = log_value - self.previous[log_key]
            curiosity_medium = log_value - self.ema_medium[log_key]
            curiosity_slow = log_value - self.ema_slow[log_key]

            # Store in result
            result[f"I_Q_{key}"] = I_Q_dict[f"I_Q_{key}"]
            result[f"log_I_Q_{key}"] = log_value
            result[f"curiosity_{key}_tau{self.tau_fast}"] = curiosity_fast
            result[f"curiosity_{key}_tau{self.tau_medium}"] = curiosity_medium
            result[f"curiosity_{key}_tau{self.tau_slow}"] = curiosity_slow

            # Update previous
            self.previous[log_key] = log_value

        return result


# ============================================================================
# VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("Testing Multi-I_Q Computation...")

    # Create dummy model
    model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256))

    # Dummy forward pass
    x = torch.randn(8, 256)
    output = model(x)
    loss = output.mean()

    # Backward for gradients
    loss.backward()

    # Compute all I_Q variants
    I_Q_dict = compute_multi_iq(model=model, loss=loss, d_model=256, n_layers=2)

    print("\nI_Q Normalizations:")
    for key, value in I_Q_dict.items():
        if key.startswith("I_Q_") or key.startswith("log_I_Q_"):
            print(f"  {key:25s} = {value:.6e}")

    print("\nRaw Components:")
    for key in ["tr_F_diag", "grad_norm", "n_params", "L_eff_sq"]:
        value = I_Q_dict.get(key)
        if value is not None:
            print(f"  {key:15s} = {float(value):.3e}")

    # Test tracker
    print("\nTesting Multi-I_Q Tracker...")
    tracker = MultiIQTracker(tau_fast=1, tau_medium=10, tau_slow=100)

    # Simulate 5 steps
    for step in range(5):
        # Dummy forward/backward
        x = torch.randn(8, 256)
        output = model(x)
        loss = output.mean()
        model.zero_grad()
        loss.backward()

        # Compute I_Q
        I_Q_dict = compute_multi_iq(model, loss, d_model=256, n_layers=2)

        # Update tracker
        tracked = tracker.update(I_Q_dict)

        print(f"\nStep {step}:")
        print(f"  I_Q_params = {tracked.get('I_Q_params', 0):.3e}")
        print(f"  curiosity_params_tau1 = {tracked.get('curiosity_params_tau1', 0):.3e}")
        print(f"  curiosity_params_tau10 = {tracked.get('curiosity_params_tau10', 0):.3e}")

    print("\n✅ Multi-I_Q system validated")
