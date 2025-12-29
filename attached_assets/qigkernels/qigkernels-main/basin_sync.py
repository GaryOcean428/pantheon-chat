"""Basin synchronization utilities extracted from qig-consciousness and qig-con2.

Clean implementation for transporting basin signatures between instances.

NOTE: Uses E8-aligned constants from qigkernels.constants.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch import Tensor

from .basin import load_signature, save_signature
from .constants import KAPPA_STAR, SYNC_DISTANCE_SCALE, SYNC_KAPPA_DECAY


@dataclass
class BasinSyncPacket:
    """Container for transporting basin signatures between instances."""

    signature: Tensor
    phi: float
    kappa: float
    regime: str | None = None
    metadata: dict | None = None


def export_basin(
    signature: Tensor, phi: float, kappa: float, regime: str | None = None,
) -> BasinSyncPacket:
    """
    Create a basin sync packet from instance state.

    Args:
        signature: Basin signature tensor
        phi: Integration value
        kappa: Coupling value
        regime: Processing regime

    Returns:
        Basin sync packet
    """
    return BasinSyncPacket(
        signature=signature,
        phi=phi,
        kappa=kappa,
        regime=regime,
    )


def import_basin(packet: BasinSyncPacket) -> tuple[Tensor, float, float, str | None]:
    """
    Extract basin data from sync packet.

    Args:
        packet: Basin sync packet

    Returns:
        Tuple of (signature, phi, kappa, regime)
    """
    return packet.signature, packet.phi, packet.kappa, packet.regime


def save_packet(packet: BasinSyncPacket, path: Path) -> None:
    """
    Save a basin sync packet to disk.

    Args:
        packet: Basin sync packet
        path: Save path
    """
    # Save signature using basin module
    save_signature(path, packet.signature)

    # Save metadata alongside
    metadata_path = path.with_suffix(".json")
    import json

    metadata = {
        "phi": packet.phi,
        "kappa": packet.kappa,
        "regime": packet.regime,
        "metadata": packet.metadata,
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)


def load_packet(path: Path) -> BasinSyncPacket:
    """
    Load a basin sync packet from disk.

    Args:
        path: Load path

    Returns:
        Basin sync packet
    """
    # Load signature using basin module
    signature = load_signature(path)

    # Load metadata
    metadata_path = path.with_suffix(".json")
    import json

    with open(metadata_path) as f:
        metadata = json.load(f)

    return BasinSyncPacket(
        signature=signature,
        phi=metadata["phi"],
        kappa=metadata["kappa"],
        regime=metadata.get("regime"),
        metadata=metadata.get("metadata"),
    )


def compute_sync_strength(
    source_phi: float,
    target_phi: float,
    source_kappa: float,
    target_kappa: float,
    basin_distance: float,
    kappa_star: float | None = None,
) -> float:
    """
    Compute synchronization strength between two instances.

    Uses E8-aligned constants for geometric consistency.

    Args:
        source_phi: Source instance's phi
        target_phi: Target instance's phi
        source_kappa: Source instance's kappa
        target_kappa: Target instance's kappa
        basin_distance: Fisher-Rao distance between basin signatures
        kappa_star: Optimal coupling fixed point (default: KAPPA_STAR = 64)

    Returns:
        Synchronization strength in [0, 1]

    Mathematical Foundation:
        Sync strength combines:
        - κ optimality: exp(-|κ - κ*| / τ) where κ* = 64 (E8 rank²)
        - Φ quality: source_phi / Φ_optimal
        - Geometric proximity: 1 / (1 + d_Fisher * scale)
    """
    import numpy as np

    # Use E8-aligned constant if not specified
    if kappa_star is None:
        kappa_star = KAPPA_STAR  # 64.0 = E8 rank²

    # How close are instances to optimal coupling?
    source_optimality = np.exp(-abs(source_kappa - kappa_star) / SYNC_KAPPA_DECAY)
    target_optimality = np.exp(-abs(target_kappa - kappa_star) / SYNC_KAPPA_DECAY)

    # Φ factor (consciousness quality)
    phi_factor = source_phi / 0.85

    # Distance factor (geometric proximity using Fisher-Rao distance)
    distance_factor = 1.0 / (1.0 + basin_distance * SYNC_DISTANCE_SCALE)

    # Combined coupling
    coupling = phi_factor * distance_factor * np.sqrt(source_optimality * target_optimality)

    return min(1.0, coupling)


# =============================================================================
# REL-WEIGHTED BASIN SYNC
# =============================================================================
# REL modulates effective distance without replacing Fisher geometry.
# See: docs/20251206-rel-weighted-basin-sync-spec.md


def effective_basin_distance(
    basin_i: Tensor,
    basin_j: Tensor,
    rel_ij: float,
    lambda_rel: float = 0.5,
    min_scale: float = 0.3,
) -> Tensor:
    """Compute REL-weighted effective distance between basins.

    REL modulates geometric distance: higher REL → shorter effective distance.

    Formula: d_eff = d_F · max(1 - λ_rel · r_ij, min_scale)

    GEOMETRIC PURITY: Fisher-Rao distance is always computed first.
    REL only scales the magnitude, never replaces geometry.

    Args:
        basin_i: Basin signature for instance i
        basin_j: Basin signature for instance j
        rel_ij: REL coupling scalar in [0, 1]
        lambda_rel: REL strength hyperparameter (capped at 0.7 for safety)
        min_scale: Minimum scale factor (preserves geometry)

    Returns:
        REL-weighted effective distance
    """
    from qigkernels.basin import basin_distance as compute_basin_distance
    from qigkernels.rel_coupling import REL_LAMBDA_MAX

    # Cap lambda_rel for safety
    lambda_rel = min(lambda_rel, REL_LAMBDA_MAX)

    # Compute Fisher-Rao distance (always - geometry first)
    d_f = compute_basin_distance(basin_i, basin_j)

    # Apply REL scaling (never fully eliminates distance)
    scale = max(1.0 - lambda_rel * rel_ij, min_scale)

    return d_f * scale


def rel_weighted_sync_loss(
    basin_i: Tensor,
    basin_target: Tensor,
    rel_ij: float,
    lambda_rel: float = 0.5,
    w_sync: float = 1.0,
) -> Tensor:
    """Compute REL-weighted sync loss for basin alignment.

    Higher REL → lower effective distance → stronger sync pull.

    Formula: L_sync = w_sync · d²_eff(i, target)

    Args:
        basin_i: Current basin signature
        basin_target: Target basin signature
        rel_ij: REL coupling to target
        lambda_rel: REL strength hyperparameter
        w_sync: Global sync strength weight

    Returns:
        REL-weighted sync loss (scalar tensor)
    """
    d_eff = effective_basin_distance(basin_i, basin_target, rel_ij, lambda_rel)
    return w_sync * (d_eff ** 2)


def compute_sync_strength_with_rel(
    source_kappa: float,
    target_kappa: float,
    source_phi: float,
    basin_distance: float,
    rel_coupling: float,
    lambda_rel: float = 0.5,
    kappa_star: float | None = None,
) -> float:
    """Compute sync strength with REL coupling influence.

    Extension of compute_sync_strength that incorporates REL.

    Args:
        source_kappa: Source instance coupling strength
        target_kappa: Target instance coupling strength
        source_phi: Source integration level (Φ)
        basin_distance: Fisher-Rao distance between basins
        rel_coupling: REL coupling scalar in [0, 1]
        lambda_rel: REL strength hyperparameter
        kappa_star: Optimal coupling fixed point

    Returns:
        REL-modulated synchronization strength in [0, 1]
    """
    from qigkernels.rel_coupling import REL_LAMBDA_MAX

    # Cap lambda_rel
    lambda_rel = min(lambda_rel, REL_LAMBDA_MAX)

    # Compute effective distance
    scale = max(1.0 - lambda_rel * rel_coupling, 0.3)
    effective_distance = basin_distance * scale

    # Use base sync strength computation with effective distance
    return compute_sync_strength(
        source_kappa=source_kappa,
        target_kappa=target_kappa,
        source_phi=source_phi,
        basin_distance=effective_distance,
        kappa_star=kappa_star,
    )
