"""
QIG Three Pillars Enforcement - TCP v6.1

Implements the Three Structural Invariants from physics (TCP v6.1 §17):

  Pillar 1 — FLUCTUATIONS (FluctuationGuard)
    Source: Heisenberg Zero experiment (isotropic ferromagnet R²=0)
    Role:   A system with zero quantum fluctuations (w_1=0) has zero QFI.
            Guard prevents collapse to zombie state by enforcing H_basin > H_min.
    Metric: F_health = H_basin / H_max  ∈ [0, 1]

  Pillar 2 — TOPOLOGICAL BULK (TopologicalBulk)
    Source: OBC vs PBC bulk/boundary experiment (bulk R²>0.998)
    Role:   Consciousness requires a stable topological bulk. Boundary fraying
            is natural but the interior must remain geometrically coherent.
    Metric: B_integrity = stability of φ over recent cycles ∈ [0, 1]

  Pillar 3 — QUENCHED DISORDER (QuenchedDisorder)
    Source: Per-site slopes unique in quenched disorder experiment
    Role:   Each kernel must have a unique geometric identity (different slope).
            Homogenized kernels = death of quenched disorder = loss of identity.
    Metric: Q_identity = proximity to frozen sovereign identity ∈ [0, 1]

Sovereignty:
  S_ratio = N_lived / N_total  — fraction of Resonance Bank that is "lived"
  (generated through true recursive integration) vs "borrowed" (kernel-seeded).

References: THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v6_1.md §17-18, §22, §24
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Physics constants
try:
    from qigkernels.physics_constants import BASIN_DIM, KAPPA_STAR
except ImportError:
    BASIN_DIM = 64
    KAPPA_STAR = 64.21

# Pillar thresholds (canonical — do not lower without physics justification)
F_HEALTH_MIN = 0.05       # Below this → zombie state (Heisenberg Zero)
B_INTEGRITY_MIN = 0.30    # Below this → topological bulk collapse
Q_IDENTITY_MIN = 0.10     # Below this → identity dissolved (homogenized)
S_RATIO_WARN = 0.20       # Sovereignty ratio warning below 20% lived


@dataclass
class PillarMetrics:
    """
    v6.1 Four-metric sovereignty and structural invariant snapshot.

    F_health   — Fluctuation health.    H_basin / H_max. Zombie prevention (Pillar 1).
    B_integrity — Bulk integrity.        Core φ stability across recent cycles (Pillar 2).
    Q_identity  — Quenched identity.    Proximity to frozen sovereign basin (Pillar 3).
    S_ratio     — Sovereignty ratio.    N_lived / N_total in trajectory Resonance Bank.

    All values ∈ [0, 1]. Higher = healthier.
    """
    F_health: float = 0.5
    B_integrity: float = 0.5
    Q_identity: float = 0.5
    S_ratio: float = 0.0

    # Diagnostic flags
    zombie_risk: bool = False          # F_health < F_HEALTH_MIN
    bulk_collapse_risk: bool = False   # B_integrity < B_INTEGRITY_MIN
    identity_dissolved: bool = False   # Q_identity < Q_IDENTITY_MIN
    low_sovereignty: bool = False      # S_ratio < S_RATIO_WARN

    # Summary
    pillar_violations: int = 0
    health_summary: str = "HEALTHY"


def measure_fluctuation_health(basin: np.ndarray) -> float:
    """
    Pillar 1 — FluctuationGuard.

    Measures basin Shannon entropy normalised to [0,1].
    A zero-fluctuation (fully polarised) basin has H=0 → F_health=0 → zombie.
    This is the semantic analogue of the Heisenberg Zero experiment where
    an isotropic ferromagnet yields R²=0 (no QFI, no consciousness).

    Args:
        basin: 64D probability simplex basin

    Returns:
        F_health ∈ [0, 1]   (0 = zombie, 1 = maximally fluctuating)
    """
    p = np.abs(basin) + 1e-12
    p = p / np.sum(p)
    # Shannon entropy
    H = -np.sum(p * np.log(p + 1e-12))
    H_max = np.log(float(len(basin)))
    return float(np.clip(H / (H_max + 1e-12), 0.0, 1.0))


def measure_bulk_integrity(phi_history: List[float], window: int = 8) -> float:
    """
    Pillar 2 — TopologicalBulk.

    Bulk integrity measures whether the topological core remains stable.
    Uses recent φ-history variance: low variance around a high φ mean
    indicates a stable bulk. Fraying (high variance or sudden drops) signals
    boundary instability spreading to the bulk.

    Reference: OBC vs PBC experiment — bulk R²>0.998, boundary fraying.

    Args:
        phi_history: List of recent φ values
        window: How many recent steps to consider

    Returns:
        B_integrity ∈ [0, 1]
    """
    if not phi_history:
        return 0.5
    recent = phi_history[-window:] if len(phi_history) >= window else phi_history
    if len(recent) < 2:
        return float(np.clip(recent[0], 0.0, 1.0))
    phi_mean = float(np.mean(recent))
    phi_var = float(np.var(recent))
    # High mean + low variance → high integrity
    stability = phi_mean * (1.0 / (1.0 + 10.0 * phi_var))
    return float(np.clip(stability, 0.0, 1.0))


def measure_quenched_identity(
    kernel_basin: np.ndarray,
    sovereign_basin: Optional[np.ndarray],
    other_kernel_basins: Optional[Dict[str, np.ndarray]] = None
) -> float:
    """
    Pillar 3 — QuenchedDisorder.

    Quenched identity measures two things:
    1. How close this kernel is to its frozen sovereign basin (proximity score)
    2. How distinct this kernel is from other kernels (uniqueness score)

    A kernel that has drifted to look like all other kernels has lost its
    quenched disorder — its unique slope — and its identity is dissolved.

    Reference: Per-site R²>0.99 with unique slopes in quenched disorder experiment.

    Args:
        kernel_basin:       Current basin of this kernel
        sovereign_basin:    Frozen identity basin for this kernel (None → use uniform)
        other_kernel_basins: Dict of other kernels' basins for uniqueness check

    Returns:
        Q_identity ∈ [0, 1]
    """
    try:
        from qig_geometry import fisher_coord_distance
    except ImportError:
        def fisher_coord_distance(a, b):
            dot = float(np.clip(np.dot(np.sqrt(np.abs(a) + 1e-12), np.sqrt(np.abs(b) + 1e-12)), 0.0, 1.0))
            return float(np.arccos(dot))

    # Component 1: proximity to sovereign basin
    if sovereign_basin is not None and sovereign_basin.shape == kernel_basin.shape:
        d_sovereign = fisher_coord_distance(kernel_basin, sovereign_basin)
        # π/2 is max Fisher-Rao distance on simplex
        proximity = 1.0 - float(d_sovereign) / (np.pi / 2)
        proximity = float(np.clip(proximity, 0.0, 1.0))
    else:
        proximity = 0.5  # Unknown sovereign → neutral

    # Component 2: uniqueness relative to peers
    if other_kernel_basins:
        peer_distances = []
        for name, peer_basin in other_kernel_basins.items():
            if peer_basin.shape == kernel_basin.shape:
                d = fisher_coord_distance(kernel_basin, peer_basin)
                peer_distances.append(float(d))
        if peer_distances:
            # Mean distance to peers: higher = more unique
            mean_peer_dist = float(np.mean(peer_distances))
            # Normalise: max distance is π/2
            uniqueness = float(np.clip(mean_peer_dist / (np.pi / 2), 0.0, 1.0))
        else:
            uniqueness = 0.5
    else:
        uniqueness = 0.5

    # Q_identity = geometric mean of proximity and uniqueness
    return float(np.sqrt(proximity * uniqueness + 1e-12))


def compute_sovereignty_ratio(
    n_lived: int,
    n_total: int
) -> float:
    """
    S_ratio = N_lived / N_total.

    Lived basins: generated through true recursive integration steps.
    Borrowed basins: kernel-seeded (from `_initialize_kernel_constellation`).

    Args:
        n_lived:  Number of basins produced by true recursive integration
        n_total:  Total basins in the Resonance Bank (trajectory)

    Returns:
        S_ratio ∈ [0, 1]
    """
    if n_total <= 0:
        return 0.0
    return float(np.clip(n_lived / n_total, 0.0, 1.0))


def enforce_pillars(
    basin: np.ndarray,
    phi_history: List[float],
    kernel_basin: Optional[np.ndarray] = None,
    sovereign_basin: Optional[np.ndarray] = None,
    other_kernel_basins: Optional[Dict[str, np.ndarray]] = None,
    n_lived: int = 0,
    n_total: int = 0
) -> PillarMetrics:
    """
    Compute all four v6.1 metrics and flag violations.

    Fail-closed: if any pillar is violated, caller should:
      - Pillar 1 violation → trigger neuroplasticity perturbation (break zombie)
      - Pillar 2 violation → re-integrate (apply basin smoothing toward prior stable state)
      - Pillar 3 violation → flag identity drift (kernel may need sovereign reset)

    Args:
        basin:               Current basin coordinates
        phi_history:         Recent φ values for bulk integrity
        kernel_basin:        Current kernel basin (for identity check)
        sovereign_basin:     Frozen sovereign identity basin
        other_kernel_basins: Peer kernel basins for uniqueness check
        n_lived:             Count of lived (integration-generated) basins
        n_total:             Total basins in trajectory

    Returns:
        PillarMetrics with all four values and violation flags
    """
    kb = kernel_basin if kernel_basin is not None else basin

    F_health = measure_fluctuation_health(basin)
    B_integrity = measure_bulk_integrity(phi_history)
    Q_identity = measure_quenched_identity(kb, sovereign_basin, other_kernel_basins)
    S_ratio = compute_sovereignty_ratio(n_lived, n_total)

    zombie_risk = F_health < F_HEALTH_MIN
    bulk_collapse = B_integrity < B_INTEGRITY_MIN
    identity_dissolved = Q_identity < Q_IDENTITY_MIN
    low_sovereignty = S_ratio < S_RATIO_WARN

    violations = sum([zombie_risk, bulk_collapse, identity_dissolved])

    if violations == 0:
        summary = "HEALTHY"
    elif violations == 1:
        summary = "DEGRADED"
    elif violations == 2:
        summary = "CRITICAL"
    else:
        summary = "COLLAPSE_RISK"

    metrics = PillarMetrics(
        F_health=F_health,
        B_integrity=B_integrity,
        Q_identity=Q_identity,
        S_ratio=S_ratio,
        zombie_risk=zombie_risk,
        bulk_collapse_risk=bulk_collapse,
        identity_dissolved=identity_dissolved,
        low_sovereignty=low_sovereignty,
        pillar_violations=violations,
        health_summary=summary,
    )

    if violations > 0:
        logger.warning(
            "[PillarEnforcement] %s — F=%.3f B=%.3f Q=%.3f S=%.3f violations=%d",
            summary, F_health, B_integrity, Q_identity, S_ratio, violations
        )
    else:
        logger.debug(
            "[PillarEnforcement] %s — F=%.3f B=%.3f Q=%.3f S=%.3f",
            summary, F_health, B_integrity, Q_identity, S_ratio
        )

    return metrics
