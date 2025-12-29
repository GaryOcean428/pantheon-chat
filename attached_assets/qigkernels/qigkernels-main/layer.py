"""QIG-style layer extracted from qig-consciousness and qig-con2.

Clean implementation with attention, recursion, optional tacking, and telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class LayerTelemetry:
    """Telemetry data for a single layer forward pass."""

    phi: float
    kappa: float
    recursion_depth: int
    regime: str | None = None


class QIGLayer(nn.Module):
    """
    Clean QIG-style transformer layer with telemetry.

    Extracts the core patterns from both repositories while removing
    experiment-specific code and hard-coded physics constants.
    """

    def __init__(  # noqa: PLR0913
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
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
    ) -> None:
        """
        Initialize QIG layer.

        Args:
            hidden_dim: Model dimension
            num_heads: Number of attention heads
            ffn_dim: Feed-forward dimension
            dropout: Dropout rate
            min_recursion_depth: Minimum mandatory recursion loops
            use_tacking: Enable WuWei tacking controller
            base_coupling: Base coupling constant (β function)
            beta_slope: β function slope
            reference_scale: Reference scale for coupling
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_tacking = use_tacking
        self.min_recursion_depth = min_recursion_depth
        self.base_coupling = base_coupling
        self.beta_slope = beta_slope
        self.reference_scale = reference_scale

        self.temperature = float(temperature)
        self.sparsity_threshold = float(sparsity_threshold)
        self.decoherence_std = float(decoherence_std)
        self.rms_eps = float(rms_eps)
        self.simplex_mode = str(simplex_mode)
        self.norm_scale_1 = nn.Parameter(torch.ones(hidden_dim))
        self.norm_scale_2 = nn.Parameter(torch.ones(hidden_dim))

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # Placeholder for recursive integrator
        self.recursive_integrator = RecursiveIntegrator(
            hidden_dim=hidden_dim,
            min_depth=min_recursion_depth,
        )

        # Placeholder for tacking controller
        self.tacking_controller: WuWeiController | None
        if use_tacking:
            self.tacking_controller = WuWeiController(hidden_dim)
        else:
            self.tacking_controller = None

        # Placeholder for regime detector
        self.regime_detector = RegimeDetector()

    def forward(
        self,
        hidden_state: Tensor,
        attention_mask: Tensor | None = None,
        kappa_eff: float | None = None,
    ) -> tuple[Tensor, LayerTelemetry]:
        """
        Forward pass through QIG layer.

        Args:
            hidden_state: Input hidden state [batch, seq, hidden_dim]
            attention_mask: Optional attention mask
            kappa_eff: Optional effective coupling

        Returns:
            Tuple of (output_hidden_state, layer_telemetry)
        """
        # Compute effective coupling if not provided
        if kappa_eff is None:
            seq_len = hidden_state.size(1)
            kappa_eff = self._compute_effective_coupling(seq_len)

        # 1. Attention with residual connection
        attn_output = self._metric_attention(
            hidden_state, attention_mask=attention_mask
        )
        hidden_state = self._rms_norm(
            hidden_state + attn_output, scale=self.norm_scale_1
        )

        # 2. Recursive integration
        recursive_output, recursive_telemetry = self.recursive_integrator(hidden_state)
        hidden_state = hidden_state + recursive_output

        # 3. Optional tacking
        if self.use_tacking and self.tacking_controller is not None:
            logic_weight, mode, _ = self.tacking_controller(hidden_state)
            if isinstance(logic_weight, Tensor):  # noqa: SIM108 (ternary too long)
                mode_scale = float(logic_weight.mean().detach())
            else:
                mode_scale = logic_weight
        else:
            mode_scale = 1.0

        # 4. Feed-forward with mode scaling
        ffn_output = self.ffn(hidden_state)
        hidden_state = self._rms_norm(
            hidden_state + ffn_output * mode_scale,
            scale=self.norm_scale_2,
        )

        # 5. Regime detection
        phi = recursive_telemetry.get("phi", 0.5)
        regime = self.regime_detector(phi, kappa_eff)

        # Compile telemetry
        telemetry = LayerTelemetry(
            phi=phi,
            kappa=kappa_eff,
            recursion_depth=recursive_telemetry.get("depth", 1),
            regime=regime,
        )

        return hidden_state, telemetry

    def _compute_effective_coupling(self, seq_len: int) -> float:
        """
        Compute effective coupling based on sequence length.

        This implements the scale-adaptive coupling from the running coupling module.
        κ is clamped to [κ_base * 0.5, κ_base * 1.5] to prevent extreme values.
        """
        # Simplified β-function: κ_eff = base * (1 + β * log(seq_len / reference_scale))
        scale_factor = max(seq_len, 1) / self.reference_scale
        log_scale = torch.log(torch.tensor(scale_factor)).item()
        effective_coupling = self.base_coupling * (1.0 + self.beta_slope * log_scale)

        # Clamp to reasonable range around base coupling
        min_kappa = self.base_coupling * 0.5
        max_kappa = self.base_coupling * 1.5
        effective_coupling = max(min_kappa, min(max_kappa, effective_coupling))

        return effective_coupling

    def _metric_attention(
        self, hidden_state: Tensor, *, attention_mask: Tensor | None
    ) -> Tensor:
        if hidden_state.dim() != 3:
            raise ValueError("hidden_state expected shape (B, T, D)")

        b, t, _d = hidden_state.shape
        p = self._project_to_simplex(hidden_state)
        p_i = p[:, :, None, :]
        p_j = p[:, None, :, :]

        dist = self._fisher_rao_distance_simplex(p_i, p_j)
        logits = -dist / max(float(self.temperature), 1e-6)
        weights = torch.softmax(logits, dim=-1)

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                key_padding_mask = attention_mask
            else:
                key_padding_mask = attention_mask == 0
            if key_padding_mask.shape != (b, t):
                raise ValueError("attention_mask expected shape (B, T)")
            weights = weights.masked_fill(key_padding_mask[:, None, :], 0.0)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        if self.sparsity_threshold > 0.0:
            keep = weights > float(self.sparsity_threshold)
            weights = weights * keep
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        output = torch.einsum("bij,bjd->bid", weights, hidden_state)
        if self.decoherence_std > 0.0:
            output = output + torch.randn_like(output) * float(self.decoherence_std)
        return output

    def _project_to_simplex(self, x: Tensor, eps: float = 1e-8) -> Tensor:
        if self.simplex_mode == "abs":
            x = x.abs()
        else:
            x = F.softplus(x)
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        return x.clamp_min(eps)

    def _fisher_rao_distance_simplex(
        self, p: Tensor, q: Tensor, eps: float = 1e-8
    ) -> Tensor:
        p = p / (p.sum(dim=-1, keepdim=True) + eps)
        q = q / (q.sum(dim=-1, keepdim=True) + eps)

        inner = torch.sum(torch.sqrt(p * q + eps), dim=-1).clamp(
            -1.0 + 1e-6, 1.0 - 1e-6
        )
        return 2.0 * torch.acos(inner)

    def _rms_norm(self, x: Tensor, *, scale: Tensor, eps: float = 1e-6) -> Tensor:
        denom = torch.sqrt(
            torch.mean(x * x, dim=-1, keepdim=True)
            + max(float(self.rms_eps), float(eps))
        )
        return (x / denom) * scale


class RecursiveIntegrator(nn.Module):
    """Recursive integrator with Φ computation from hidden state integration."""

    def __init__(self, hidden_dim: int, min_depth: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.min_depth = min_depth
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.integration_gate = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Recursive integration with Φ estimation."""
        # Apply recursive projection
        output = hidden_state
        integration_scores = []

        for _ in range(self.min_depth):
            projected = self.projection(output)
            # Integration gate measures how much new info integrates
            gate = torch.sigmoid(self.integration_gate(projected))
            output = output + gate * (projected - output)
            integration_scores.append(gate.mean().item())

        # Φ estimation: based on integration coherence
        # High integration = high Φ (information integrated across the system)
        mean_integration = sum(integration_scores) / len(integration_scores)

        # Also measure cross-position coherence (how similar are different positions)
        if hidden_state.dim() == 3 and hidden_state.size(1) > 1:
            # Compute variance across sequence positions
            pos_var = hidden_state.var(dim=1).mean().item()
            # Lower variance = higher coherence = higher Φ
            coherence = 1.0 / (1.0 + pos_var)
        else:
            coherence = 0.5

        # Combine integration and coherence for Φ
        phi = 0.3 + 0.4 * mean_integration + 0.3 * coherence
        phi = max(0.0, min(1.0, phi))  # Clamp to [0, 1]

        telemetry = {
            "phi": phi,
            "depth": self.min_depth,
            "integration": mean_integration,
            "coherence": coherence,
        }

        return output, telemetry


class WuWeiController(nn.Module):
    """Placeholder WuWei tacking controller."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, str, dict]:
        """Simplified tacking control."""
        logic_weight = torch.sigmoid(self.projection(hidden_state.mean(dim=1)))
        mode = "logic" if logic_weight.mean() > 0.5 else "feeling"
        return logic_weight, mode, {}


class RegimeDetector:
    """Simple regime detector."""

    def __call__(self, phi: float, kappa: float) -> str:
        """Detect regime based on phi and kappa."""
        if phi < 0.45:
            return "linear"
        elif phi < 0.80:
            return "geometric"
        else:
            return "breakdown"
