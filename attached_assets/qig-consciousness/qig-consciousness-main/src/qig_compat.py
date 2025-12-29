"""QIG Ecosystem Compatibility Layer.

This module provides imports from the qigkernels package with fallbacks
to local implementations for backwards compatibility.

Usage:
    # Instead of: from src.kernel import QIGKernel
    # Use: from src.qig_compat import Kernel, KernelTelemetry

    # Instead of: from src.model.basin import BasinMatcher
    # Use: from src.qig_compat import BasinProjector, basin_distance

After migration is complete, replace with direct qigkernels imports:
    from qigkernels import Kernel, BasinProjector, Constellation
"""
from __future__ import annotations

import warnings

# Try importing from qigkernels first (preferred)
try:
    from qigkernels import (
        BASIN_DIM,
        BasinProjector,
        Constellation,
        Instance,
        InstanceView,
        Kernel,
        KernelTelemetry,
        LayerTelemetry,
        QIGLayer,
        basin_distance,
        compute_signature,
        round_robin,
        select_balanced,
        select_phi_max,
        select_phi_min,
    )
    from qigkernels.constants import BETA_EMERGENCE, KAPPA_3, KAPPA_STAR, SYNC_KAPPA_DECAY

    USING_QIGKERNELS = True

except ImportError:
    warnings.warn(
        "qigkernels not installed. Using local implementations. "
        "Install with: uv pip install -e ../qigkernels",
        ImportWarning,
        stacklevel=2,
    )
    USING_QIGKERNELS = False

    # Fall back to local implementations
    import torch
    from torch import Tensor, nn

    from src.kernel import QIGKernel as Kernel

    # Provide minimal stubs for missing features
    BASIN_DIM = 64

    class BasinProjector(nn.Module):  # type: ignore[no-redef]
        def __init__(self, hidden_dim: int, signature_dim: int = BASIN_DIM) -> None:
            super().__init__()
            self.proj = nn.Linear(hidden_dim, signature_dim)

        def forward(self, hidden_state: Tensor) -> Tensor:
            if hidden_state.dim() != 3:
                raise ValueError(f"expected hidden_state [batch, seq, hidden_dim], got {tuple(hidden_state.shape)}")
            return self.proj(hidden_state.mean(dim=1))

    # Constants (match qigkernels defaults)
    BETA_EMERGENCE = 0.443
    SYNC_KAPPA_DECAY = 10.0

    # Physics constants (fallback)
    KAPPA_3 = 41.09
    KAPPA_STAR = 64.0

    def basin_distance(a, b):  # type: ignore[misc]
        """Fisher-Rao distance between basin signatures (Bures approximation).

        GEOMETRIC PURITY: Uses Fisher metric, NOT Euclidean L2.
        """
        import torch.nn.functional as F

        # Bures approximation: d² = 2(1 - cos_sim)
        cos_sim = F.cosine_similarity(
            a.unsqueeze(0) if a.dim() == 1 else a,
            b.unsqueeze(0) if b.dim() == 1 else b,
            dim=-1,
        )
        distance_sq = 2.0 * (1.0 - cos_sim)
        return torch.sqrt(torch.clamp(distance_sq, min=1e-8)).squeeze()

    def compute_signature(projector, hidden_state):  # type: ignore[misc]
        """Compute basin signature."""
        sig = projector(hidden_state)
        return sig.mean(dim=0) if sig.dim() > 1 else sig

    # Telemetry types as simple dicts
    class KernelTelemetry:  # type: ignore[no-redef]
        """Stub for kernel telemetry."""
        def __init__(self, **kwargs):  # type: ignore[misc]
            for k, v in kwargs.items():
                setattr(self, k, v)

    LayerTelemetry = KernelTelemetry

    # Constellation stubs
    class Instance:  # type: ignore[no-redef]
        def __init__(self, name: str, kernel, phi=None, signature=None):  # type: ignore[misc]
            self.name = name
            self.kernel = kernel
            self.phi = phi
            self.signature = signature

    class Constellation:  # type: ignore[no-redef]
        def __init__(self):  # type: ignore[misc]
            self.instances = []

        def add_instance(self, instance):  # type: ignore[misc]
            self.instances.append(instance)

    # Routing stubs
    def round_robin(last_idx: int, n: int) -> int:  # type: ignore[misc]
        return (last_idx + 1) % n

    def select_phi_max(views) -> int:  # type: ignore[misc]
        return max(range(len(views)), key=lambda i: views[i].phi or 0)

    # Other stubs
    QIGLayer = None
    InstanceView = None


# Provide a simple check for migration status
def check_qigkernels_available() -> bool:
    """Check if qigkernels is properly installed."""
    return USING_QIGKERNELS


# Compatibility aliases
BETA_3_TO_4 = BETA_EMERGENCE


__all__ = [
    # Core
    "Kernel",
    "KernelTelemetry",
    "QIGLayer",
    "LayerTelemetry",
    # Basin
    "BASIN_DIM",
    "BasinProjector",
    "basin_distance",
    "compute_signature",
    # Constellation
    "Constellation",
    "Instance",
    "InstanceView",
    "round_robin",
    "select_phi_max",
    "select_phi_min",
    "select_balanced",
    # Physics constants
    "KAPPA_3",
    "BETA_3_TO_4",
    "KAPPA_STAR",
    "BETA_EMERGENCE",
    "SYNC_KAPPA_DECAY",
    # Status
    "USING_QIGKERNELS",
    "check_qigkernels_available",
]
