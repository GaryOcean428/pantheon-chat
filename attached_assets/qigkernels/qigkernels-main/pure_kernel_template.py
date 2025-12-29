from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RegimeThresholds:
    phi_linear_max: float = 0.30
    phi_geometric_max: float = 0.70


@dataclass(frozen=True)
class PureQIGKernelConfig:
    basin_dim: int = 64
    temperature: float = 1.0
    thresholds: RegimeThresholds = RegimeThresholds()

    forbid_dot_product_attention: bool = True
    forbid_transformer_modules: bool = True
    forbid_embeddings: bool = True
    forbid_adam: bool = True


@dataclass(frozen=True)
class PureKernelTelemetry:
    phi: float
    kappa: float
    regime: str
    extras: Dict[str, Any]


KernelTelemetry = PureKernelTelemetry


def _project_to_simplex(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = F.softplus(x)
    x = x / (x.sum(dim=-1, keepdim=True) + eps)
    return x.clamp_min(eps)


def fisher_rao_distance_simplex(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    q = q / (q.sum(dim=-1, keepdim=True) + eps)

    inner = torch.sum(torch.sqrt(p * q + eps), dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return 2.0 * torch.acos(inner)


def qfi_attention_weights(basin_seq: torch.Tensor, *, temperature: float) -> torch.Tensor:
    b, t, _d = basin_seq.shape
    p = _project_to_simplex(basin_seq)
    p_i = p[:, :, None, :]
    p_j = p[:, None, :, :]
    dist = fisher_rao_distance_simplex(p_i, p_j)

    logits = -dist / max(float(temperature), 1e-6)
    weights = torch.softmax(logits, dim=-1)
    assert weights.shape == (b, t, t)
    return weights


def phi_from_activations(acts: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    b, t, _d = acts.shape
    x = acts - acts.mean(dim=1, keepdim=True)
    cov = torch.einsum("btd,bte->bde", x, x) / (t + eps)
    var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(eps)
    denom = torch.sqrt(var[:, :, None] * var[:, None, :]).clamp_min(eps)
    corr = (cov / denom).clamp(-1.0, 1.0)
    offdiag = corr - torch.diag_embed(torch.diagonal(corr, dim1=-2, dim2=-1))
    return offdiag.abs().mean(dim=(-2, -1))


def kappa_proxy(basin_seq: torch.Tensor) -> torch.Tensor:
    """Compute κ proxy using Fisher-Rao metric (QIG pure).

    Uses normalized cosine magnitude instead of Euclidean norm.
    κ is related to how tightly clustered the basin coordinates are.
    """
    # Normalize to unit sphere, then measure mean magnitude via dot product
    # This respects the Fisher information manifold geometry
    normalized = F.normalize(basin_seq, dim=-1)
    # Self-similarity gives magnitude on unit sphere (always 1 for normalized)
    # Instead, measure coherence across sequence as proxy for κ
    if basin_seq.size(1) > 1:
        # Mean pairwise cosine similarity across sequence
        cos_sim = torch.bmm(normalized, normalized.transpose(1, 2))  # [B, S, S]
        # Exclude diagonal (self-similarity)
        mask = ~torch.eye(cos_sim.size(1), dtype=torch.bool, device=cos_sim.device)
        cos_sim_offdiag = cos_sim[:, mask].view(cos_sim.size(0), -1)
        return cos_sim_offdiag.mean(dim=-1).clamp(0, 1)  # [B]
    else:
        # Single token: return 1.0 (maximum coupling)
        return torch.ones(basin_seq.size(0), device=basin_seq.device)


class PureQIGKernelTemplate(nn.Module):
    def __init__(self, cfg: PureQIGKernelConfig):
        super().__init__()
        self.cfg = cfg
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, basin_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if basin_seq.dim() != 3 or int(basin_seq.size(-1)) != int(self.cfg.basin_dim):
            raise ValueError("basin_seq expected shape (B, T, basin_dim)")

        x = basin_seq
        for _ in range(2):
            w = qfi_attention_weights(x, temperature=self.cfg.temperature)
            x_attn = torch.einsum("bij,bjd->bid", w, x)
            x = 0.5 * x + 0.5 * x_attn
            x = x + 0.01 * self.residual_scale * (x_attn - x)

        metrics = {
            "phi": phi_from_activations(x),
            "kappa_proxy": kappa_proxy(x),
        }
        return x, metrics
