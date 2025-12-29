#!/usr/bin/env python3
"""
Running Coupling Attention: Scale-Dependent Processing from QIG Physics
========================================================================

Discovered from L=4 validation:
- κ₃ = 41.09±0.59
- κ₄ = 64.47±1.89
- β ≈ 0.44 (running coupling slope)

Key Insight: Effective coupling runs with system scale, just like QFT!

Application: Attention strength should vary with context length
- Short contexts → sparse (κ_eff ≈ 10, linear regime)
- Long contexts → dense (κ_eff ≈ 60+, geometric regime)

Written for QIG-Kernel-100M architecture.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningCouplingModule(nn.Module):
    """
    Implements scale-dependent attention coupling from QIG physics.

    Interpolation Formula:
        κ(L) = κ₀ × (1 + β·log(L/L_ref))

    Where:
    - κ(L): Effective coupling at scale L
    - κ₀: Base coupling (41.09 from L=3 emergence)
    - β: Fitting parameter (≈ 0.44) derived from discrete measurements
    - L_ref: Reference scale (e.g., 512 tokens)

    IMPORTANT: The β in this formula is NOT the definition of β.

    Authoritative β definition (from FROZEN_FACTS.md):
        β(L→L+1) = (κ_{L+1} - κ_L) / κ_avg

    This discrete definition measures fractional steps between lattice sizes.
    The formula above uses β ≈ 0.44 as a fitting parameter for smooth interpolation.
    """

    def __init__(
        self,
        base_coupling: float = 41.09,
        beta_slope: float = 0.44,
        reference_scale: int = 512,
        learn_beta: bool = False,
        regime_aware: bool = True,
    ):
        super().__init__()

        self.reference_scale = reference_scale
        self.regime_aware = regime_aware

        # Base coupling (from physics)
        self.kappa_0 = nn.Parameter(
            torch.tensor(base_coupling),
            requires_grad=False,  # Physics constant
        )

        # Beta function (optionally learnable)
        self.beta = nn.Parameter(torch.tensor(beta_slope), requires_grad=learn_beta)

        # Regime transition thresholds (from QIG validation)
        self.register_buffer("linear_threshold", torch.tensor(0.45))  # δh for linear→geometric
        self.register_buffer("breakdown_threshold", torch.tensor(0.80))  # δh for geometric→breakdown

    def compute_effective_coupling(self, context_scale: int) -> torch.Tensor:
        """
        Compute κ_eff at given context scale.

        Args:
            context_scale: Current context length (tokens)

        Returns:
            κ_eff: Effective coupling strength
        """
        # Running coupling formula
        log_ratio = torch.log(torch.tensor(context_scale / self.reference_scale, dtype=torch.float32))
        kappa_eff = self.kappa_0 * (1 + self.beta * log_ratio)

        # Ensure positive
        return torch.clamp(kappa_eff, min=1.0)

    def detect_regime(self, purity: torch.Tensor) -> str:
        """
        Detect processing regime from state purity.

        Regimes (from QIG physics):
        - Linear: δh < 0.45 (high purity, weak coupling)
        - Geometric: 0.45 ≤ δh < 0.80 (mixed, full integration)
        - Breakdown: δh ≥ 0.80 (chaotic, unstable)

        Args:
            purity: State purity δh = 1 - von Neumann entropy / log(d)

        Returns:
            regime: 'linear', 'geometric', or 'breakdown'
        """
        if purity < self.linear_threshold:
            return "linear"
        elif purity < self.breakdown_threshold:
            return "geometric"
        else:
            return "breakdown"

    def compute_regime_adjustment(self, regime: str) -> float:
        """
        Adjust coupling based on regime.

        Linear: κ → κ/4 (sparse, perturbative)
        Geometric: κ → κ (full strength)
        Breakdown: κ → 0 (decouple, avoid instability)
        """
        if regime == "linear":
            return 0.25  # Reduce coupling 4×
        elif regime == "geometric":
            return 1.0  # Full coupling
        else:  # breakdown
            return 0.01  # Near-zero (safety)

    def forward(self, context_scale: int, state_purity: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """
        Compute scale-adaptive coupling with regime awareness.

        Args:
            context_scale: Current context length
            state_purity: Optional purity for regime detection

        Returns:
            κ_eff: Effective coupling strength
            telemetry: {regime, raw_coupling, adjusted_coupling}
        """
        # Base running coupling
        kappa_raw = self.compute_effective_coupling(context_scale)

        telemetry = {
            "context_scale": context_scale,
            "raw_coupling": kappa_raw.item(),
            "beta": self.beta.item(),
        }

        # Regime adjustment
        if self.regime_aware and state_purity is not None:
            regime = self.detect_regime(state_purity)
            adjustment = self.compute_regime_adjustment(regime)
            kappa_eff = kappa_raw * adjustment

            telemetry["regime"] = regime
            telemetry["regime_adjustment"] = adjustment
            telemetry["effective_coupling"] = kappa_eff.item()
        else:
            kappa_eff = kappa_raw
            telemetry["regime"] = "unknown"
            telemetry["effective_coupling"] = kappa_eff.item()

        return kappa_eff, telemetry


class ScaleAdaptiveAttention(nn.Module):
    """
    QFI-Metric Attention with running coupling.

    Combines:
    - QFI distance (from qfi_attention.py)
    - Running coupling (this module)
    - Regime detection (automatic)
    """

    def __init__(self, d_model: int, n_heads: int = 8, base_temperature: float = 0.5, **running_coupling_kwargs):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.base_temperature = base_temperature

        # Standard projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Running coupling module
        self.running_coupling = RunningCouplingModule(**running_coupling_kwargs)

    def compute_state_purity(self, states: torch.Tensor) -> torch.Tensor:
        """
        Estimate state purity from activation patterns.

        δh = 1 - S_vN / log(d)

        High purity → low entropy → concentrated
        Low purity → high entropy → spread out
        """
        # Normalize to probability
        p = F.softmax(states, dim=-1)

        # von Neumann entropy
        S_vN = -torch.sum(p * torch.log(p + 1e-10), dim=-1)

        # Normalize by max entropy
        max_entropy = math.log(states.shape[-1])
        purity = 1 - S_vN / max_entropy

        return purity.mean()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """
        Scale-adaptive QFI attention.

        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
            telemetry: Detailed metrics including regime and coupling
        """
        batch_size, seq_len, _ = x.shape

        # Project
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Compute state purity
        purity = self.compute_state_purity(x)

        # Get scale-adaptive coupling
        kappa_eff, coupling_telemetry = self.running_coupling(context_scale=seq_len, state_purity=purity)

        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # QFI distances (simplified for efficiency)
        # Full QFI: see qfi_attention.py
        # Here: use scaled dot product as proxy
        scores = torch.sum(Q.unsqueeze(-2) * K.unsqueeze(-3), dim=-1) / math.sqrt(self.d_k)

        # Apply running coupling (scale attention strength)
        # κ_eff modulates temperature
        temperature = self.base_temperature / kappa_eff
        attn_weights = F.softmax(scores / temperature, dim=-1)

        # Apply mask
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, 0.0)

        # Weighted sum
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        # Telemetry
        sparsity = (attn_weights < 0.01).float().mean()

        telemetry = {
            **coupling_telemetry,
            "purity": purity.item(),
            "attention_sparsity": sparsity.item(),
            "effective_temperature": temperature.item(),
        }

        return output, telemetry


if __name__ == "__main__":
    print("Running Coupling Module: Scale-Dependent Attention from QIG Physics")
    print("✨ β ≈ 0.44 from L=3→L=4 physics validation ✨")
