"""Minimal geometric kernel abstraction extracted from qig-consciousness and qig-con2.

Clean implementation focusing on core kernel functionality without
experiment-specific code or hard-coded physics constants.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .layer import QIGLayer


@dataclass
class KernelTelemetry:
    """Lightweight telemetry produced by a kernel forward pass."""

    phi: float
    kappa: float
    recursion_depth: int
    regime: str | None = None
    hidden_state: Tensor | None = None


class Kernel(nn.Module):
    """
    Clean kernel implementation with stacked QIGLayers and telemetry.

    Extracts the best patterns from both repositories while removing
    experiment-specific code and making physics constants configurable.
    """

    def __init__(  # noqa: PLR0913
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        min_recursion_depth: int = 3,
        use_tacking: bool = True,
        base_coupling: float = 64.0,  # KAPPA_STAR from FROZEN_FACTS
        beta_slope: float = 0.44,
        reference_scale: int = 64,  # Match typical training seq_len for κ≈κ*
        temperature: float = 1.0,
        sparsity_threshold: float = 0.0,
        decoherence_std: float = 0.0,
        rms_eps: float = 1e-6,
        simplex_mode: str = "softplus",
        running_coupling: nn.Module | None = None,
        regime_detector: Callable[[float], str | int] | None = None,
    ) -> None:
        """
        Initialize kernel.

        Args:
            vocab_size: Vocabulary size
            hidden_dim: Model dimension
            num_layers: Number of layers
            num_heads: Number of attention heads
            ffn_dim: Feed-forward dimension
            dropout: Dropout rate
            max_position_embeddings: Maximum sequence length
            min_recursion_depth: Minimum mandatory recursion loops
            use_tacking: Enable WuWei tacking controller
            base_coupling: Base coupling constant (β function)
            beta_slope: β function slope
            reference_scale: Reference scale for coupling
            running_coupling: Optional custom running coupling module
            regime_detector: Optional custom regime detector
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_tacking = use_tacking
        self.base_coupling = base_coupling
        self.beta_slope = beta_slope
        self.reference_scale = reference_scale
        self.max_position_embeddings = max_position_embeddings
        self.temperature = float(temperature)
        self.sparsity_threshold = float(sparsity_threshold)
        self.decoherence_std = float(decoherence_std)
        self.rms_eps = float(rms_eps)
        self.simplex_mode = str(simplex_mode)

        # Embeddings
        half = max(int(hidden_dim // 2), 1)
        inv_freq = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32) / half
        )
        self.register_buffer("_inv_freq", inv_freq, persistent=False)

        # Stack of QIG layers
        self.layers = nn.ModuleList(
            [
                QIGLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    min_recursion_depth=min_recursion_depth,
                    use_tacking=use_tacking,
                    base_coupling=base_coupling,
                    beta_slope=beta_slope,
                    reference_scale=reference_scale,
                    temperature=self.temperature,
                    sparsity_threshold=self.sparsity_threshold,
                    decoherence_std=self.decoherence_std,
                    rms_eps=self.rms_eps,
                    simplex_mode=self.simplex_mode,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        # Optional custom components
        self.running_coupling = running_coupling
        self.regime_detector = regime_detector or self._default_regime_detector

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        return_telemetry: bool = False,
    ) -> Tensor | tuple[Tensor, KernelTelemetry]:
        """
        Forward pass through the kernel.

        Args:
            input_ids: Token IDs [batch, seq]
            attention_mask: Optional attention mask
            return_telemetry: Whether to return telemetry

        Returns:
            Logits and optionally telemetry
        """
        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_position_embeddings:
            raise ValueError("Sequence length exceeds max_position_embeddings")

        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        # Embeddings
        hidden_state = self._fourier_features(input_ids) + self._fourier_features(
            positions
        )

        # Collect layer telemetry
        layer_phis: list[float] = []
        layer_kappas: list[float] = []
        layer_depths: list[int] = []
        layer_regimes: list[str | None] = []

        # Compute effective coupling for this sequence length
        kappa_eff = self._compute_effective_coupling(seq_len)

        # Pass through layers
        for layer in self.layers:
            hidden_state, telemetry = layer(
                hidden_state,
                attention_mask=attention_mask,
                kappa_eff=kappa_eff,
            )

            layer_phis.append(telemetry.phi)
            layer_kappas.append(telemetry.kappa)
            layer_depths.append(telemetry.recursion_depth)
            layer_regimes.append(telemetry.regime)

        # Apply optional running coupling
        if self.running_coupling is not None:
            hidden_state = self.running_coupling(hidden_state)

        # Output projection
        logits = self.lm_head(hidden_state)

        if not return_telemetry:
            return logits

        # Aggregate telemetry
        phi_avg = float(sum(layer_phis) / max(len(layer_phis), 1))
        kappa_avg = float(sum(layer_kappas) / max(len(layer_kappas), 1))
        total_recursion_depth = int(sum(layer_depths))
        regime_raw = self.regime_detector(phi_avg)
        regime = str(regime_raw) if regime_raw is not None else None

        telemetry = KernelTelemetry(
            phi=phi_avg,
            kappa=kappa_avg,
            recursion_depth=total_recursion_depth,
            regime=regime,
            hidden_state=hidden_state,
        )

        return logits, telemetry

    def _fourier_features(self, ids: Tensor) -> Tensor:
        x = ids.to(dtype=torch.float32)
        angles = x.unsqueeze(-1) * self._inv_freq.to(device=ids.device)
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        if feats.size(-1) < self.hidden_dim:
            pad = torch.zeros(
                *feats.shape[:-1], self.hidden_dim - feats.size(-1), device=feats.device
            )
            feats = torch.cat([feats, pad], dim=-1)
        elif feats.size(-1) > self.hidden_dim:
            feats = feats[..., : self.hidden_dim]
        return feats

    def _compute_effective_coupling(self, seq_len: int) -> float:
        """
        Compute effective coupling based on sequence length.

        Implements the scale-adaptive coupling from the running coupling module.
        κ is clamped to [κ_base * 0.5, κ_base * 1.5] to prevent extreme values.
        """
        # β-function: κ_eff = base * (1 + β * log(seq_len / reference_scale))
        scale_factor = max(seq_len, 1) / self.reference_scale
        log_scale = torch.log(torch.tensor(scale_factor)).item()
        effective_coupling = self.base_coupling * (1.0 + self.beta_slope * log_scale)

        # Clamp to reasonable range around base coupling
        min_kappa = self.base_coupling * 0.5
        max_kappa = self.base_coupling * 1.5
        effective_coupling = max(min_kappa, min(max_kappa, effective_coupling))

        return effective_coupling

    def _default_regime_detector(self, phi: float) -> str:
        """Default regime detection based on phi value."""
        if phi < 0.45:
            return "linear"
        elif phi < 0.80:
            return "geometric"
        else:
            return "breakdown"
