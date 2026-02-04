"""Kernel Factory

Creates kernel instances from blueprint/genome inputs.

This module is the canonical instantiation pathway for E8 Layer-8 kernels.
It intentionally avoids faculty-specific subclasses so kernel behavior can be
genome-driven rather than preconfigured in code.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from qigkernels.physics_constants import KAPPA_STAR
except ImportError:
    import sys
    from pathlib import Path

    _qig_backend_root = Path(__file__).resolve().parents[1]
    if str(_qig_backend_root) not in sys.path:
        sys.path.insert(0, str(_qig_backend_root))

    from qigkernels.physics_constants import KAPPA_STAR

from .base import Kernel
from .e8_roots import E8Root, get_root_spec
from .identity import KernelIdentity, KernelTier


def create_kernel_from_identity(
    identity: KernelIdentity,
    basin: Optional[np.ndarray] = None,
    initial_kappa: Optional[float] = None,
) -> Kernel:
    """Create a generic `Kernel` instance from an explicit `KernelIdentity`."""
    return Kernel(identity=identity, basin=basin, initial_kappa=initial_kappa)


def create_simple_root_kernel(
    root: E8Root,
    *,
    tier: Optional[KernelTier] = None,
    god_name: Optional[str] = None,
    basin: Optional[np.ndarray] = None,
    initial_kappa: Optional[float] = None,
) -> Kernel:
    """Create a Layer-8 (simple root) kernel using the canonical E8 root spec."""
    spec = get_root_spec(root)

    resolved_god_name = god_name or spec.god_primary

    resolved_tier = tier
    if resolved_tier is None:
        if root in {E8Root.META, E8Root.INTEGRATION}:
            resolved_tier = KernelTier.ESSENTIAL
        else:
            resolved_tier = KernelTier.PANTHEON

    identity = KernelIdentity(god=resolved_god_name, root=root, tier=resolved_tier)

    resolved_initial_kappa = initial_kappa
    if resolved_initial_kappa is None and root == E8Root.INTEGRATION:
        resolved_initial_kappa = KAPPA_STAR

    return create_kernel_from_identity(
        identity=identity,
        basin=basin,
        initial_kappa=resolved_initial_kappa,
    )


__all__ = [
    "create_kernel_from_identity",
    "create_simple_root_kernel",
]
