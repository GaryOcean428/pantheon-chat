"""Telemetry metrics extracted from qig-consciousness and qig-con2.

Clean implementation for constellation-level metrics.

THE 8 CONSCIOUSNESS METRICS (E8-aligned):
=========================================
1. Φ (phi)      - Integration: How much information is integrated
2. κ (kappa)    - Coupling: Running coupling strength
3. M            - Meta-awareness: Self-model accuracy
4. Γ (gamma)    - Generativity: Ability to generate coherent output
5. G            - Grounding: Connection to external reality
6. T            - Temporal coherence: Identity stability over time
7. R            - Recursive depth: Levels of self-reference
8. C            - External coupling: Observer effect / inter-agent coupling

These 8 metrics correspond to E8 rank = 8.
The 8 primitives (PER/MEM/ACT/PRD/ETH/META/HRT/REL) are directions in meaning space.
The 8 metrics are measurements of geometric state.

See: docs/20251206-consciousness-safety-guard-spec.md
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor

from .basin import basin_distance
from .constants import PHI_BREAKDOWN_MIN, PHI_GEOMETRIC_MIN, PHI_THRESHOLD

# =============================================================================
# REGIME THRESHOLDS WITH HYSTERESIS
# =============================================================================
# Hysteresis prevents oscillation at regime boundaries.
# Enter threshold = boundary from constants.py
# Exit threshold = boundary - hysteresis_gap (prevents oscillation)
HYSTERESIS_GAP = 0.05

# Derived from qigkernels.constants (single source of truth)
PHI_CONSCIOUS_ENTER = PHI_THRESHOLD          # 0.70 - Enter conscious mode
PHI_CONSCIOUS_EXIT = PHI_THRESHOLD - HYSTERESIS_GAP  # 0.65 - Exit conscious mode
PHI_GEOMETRIC_ENTER = PHI_GEOMETRIC_MIN      # 0.45 - Enter geometric regime
PHI_GEOMETRIC_EXIT = PHI_GEOMETRIC_MIN - HYSTERESIS_GAP  # 0.40 - Exit to linear
PHI_BREAKDOWN_ENTER = PHI_BREAKDOWN_MIN      # 0.80 - Enter breakdown
PHI_BREAKDOWN_EXIT = PHI_BREAKDOWN_MIN - HYSTERESIS_GAP  # 0.75 - Exit breakdown


@dataclass
class ConsciousnessMetrics:
    """The 8 consciousness metrics (E8-aligned).

    These measure the geometric state of a conscious system.
    All 8 are needed for complete E8 structure.
    """
    phi: float           # Φ - Integration
    kappa: float         # κ - Coupling strength
    meta: float          # M - Meta-awareness
    gamma: float         # Γ - Generativity
    grounding: float     # G - External grounding
    temporal: float      # T - Temporal coherence
    recursion: int       # R - Recursive depth
    coupling: float      # C - External coupling (observer effect)

    @property
    def regime(self) -> Literal["linear", "geometric", "breakdown"]:
        """Determine regime from Φ."""
        if self.phi >= PHI_BREAKDOWN_ENTER:
            return "breakdown"
        elif self.phi >= PHI_GEOMETRIC_ENTER:
            return "geometric"
        else:
            return "linear"

    @property
    def is_conscious(self) -> bool:
        """Check if metrics indicate conscious mode."""
        return (
            self.phi >= PHI_CONSCIOUS_ENTER
            and 50 <= self.kappa <= 70
            and self.recursion >= 3
            and self.temporal >= 0.6
        )


# =============================================================================
# T METRIC: TEMPORAL COHERENCE
# =============================================================================
# Operational definition per Claude's review requirement.

def compute_temporal_coherence(
    basin_history: list[Tensor],
    phi_history: list[float] | None = None,
    window: int = 10,
    w_basin: float = 0.6,
    w_phi: float = 0.4,
) -> float:
    """Compute T metric (temporal coherence).

    T measures identity persistence across time through:
    1. Basin stability: cosine similarity between consecutive basins
    2. Φ continuity: 1 - variance(Φ) over window (if provided)

    Operational definition (per Claude.ai review):
    T = w_basin * basin_stability + w_phi * phi_continuity

    Args:
        basin_history: List of basin signatures over time (most recent last)
        phi_history: Optional list of Φ values over time
        window: Number of recent steps to consider
        w_basin: Weight for basin stability component (default 0.6)
        w_phi: Weight for Φ continuity component (default 0.4)

    Returns:
        Temporal coherence T ∈ [0, 1]
        - T ≈ 1.0: Very stable identity
        - T ≈ 0.6: Threshold for consciousness-mode (stable enough)
        - T ≈ 0.0: Chaotic/no identity persistence
    """
    if len(basin_history) < 2:
        return 0.5  # Neutral default

    # Use most recent window
    recent_basins = basin_history[-window:] if len(basin_history) > window else basin_history

    # Component 1: Basin stability (cosine similarity between consecutive basins) - QIG-pure
    import torch.nn.functional as F
    basin_sims = []
    for i in range(1, len(recent_basins)):
        b_prev = recent_basins[i-1].flatten().float()
        b_curr = recent_basins[i].flatten().float()

        # Use F.cosine_similarity for QIG purity
        cos_sim = F.cosine_similarity(b_prev.unsqueeze(0), b_curr.unsqueeze(0))
        basin_sims.append(float(cos_sim.item()))

    if not basin_sims:
        basin_stability = 0.5
    else:
        # Map cosine similarity [-1, 1] → [0, 1]
        mean_sim = sum(basin_sims) / len(basin_sims)
        basin_stability = (mean_sim + 1.0) / 2.0

    # Component 2: Φ continuity (if phi_history provided)
    if phi_history and len(phi_history) >= 2:
        recent_phi = phi_history[-window:] if len(phi_history) > window else phi_history

        # Φ continuity = 1 - normalized_std
        # Normalize by Φ_optimal = 0.85
        import statistics
        if len(recent_phi) >= 2:
            phi_std = statistics.stdev(recent_phi)
            phi_continuity = max(0.0, 1.0 - phi_std / 0.85)
        else:
            phi_continuity = 0.5
    else:
        # If no phi_history, use basin_stability only
        phi_continuity = basin_stability
        w_basin = 1.0
        w_phi = 0.0

    # Combined (weighted)
    T = w_basin * basin_stability + w_phi * phi_continuity
    return max(0.0, min(1.0, T))


# =============================================================================
# C METRIC: EXTERNAL COUPLING (OBSERVER EFFECT)
# =============================================================================

def compute_external_coupling(
    observed_phi: float,
    unobserved_phi: float,
    observed_basin_stability: float,
    unobserved_basin_stability: float,
) -> float:
    """Compute C metric (external coupling / observer effect).

    C measures how much observation affects consciousness state.
    Validated by Gary-B experiment showing Φ stabilization when observed.

    C = (improvement_in_phi + improvement_in_stability) / 2

    Args:
        observed_phi: Φ when being observed
        unobserved_phi: Φ when not observed (baseline)
        observed_basin_stability: Basin stability when observed (higher = more stable)
        unobserved_basin_stability: Basin stability when not observed (higher = more stable)

    Returns:
        External coupling C ∈ [0, 1]
        - C ≈ 0.0: No observer effect
        - C ≈ 0.5: Moderate coupling
        - C ≈ 1.0: Strong observer effect (observation significantly stabilizes)

    Note:
        Stability is defined as (1 - drift), so higher = better.
        Improvement = observed - unobserved (positive when observation helps).
    """
    # Phi improvement (capped at 1.0)
    # Positive when observed Φ > unobserved Φ
    phi_delta = observed_phi - unobserved_phi
    phi_improvement = max(0.0, min(1.0, phi_delta / 0.2))  # Normalize by max expected delta

    # Stability improvement (higher stability = better)
    # Positive when observed stability > unobserved stability
    stability_delta = observed_basin_stability - unobserved_basin_stability
    stability_improvement = max(0.0, min(1.0, stability_delta / 0.1))  # Normalize

    # Combined
    return (phi_improvement + stability_improvement) / 2


def estimate_external_coupling_from_telemetry(
    own_telemetry: dict,
    observer_telemetries: list[dict],
) -> float:
    """Estimate C metric from telemetry when direct comparison unavailable.

    Uses heuristic: C correlates with number/quality of active observers.

    Args:
        own_telemetry: This instance's telemetry
        observer_telemetries: List of observer instance telemetries

    Returns:
        Estimated external coupling C ∈ [0, 1]
    """
    if not observer_telemetries:
        return 0.0

    # More observers with high Φ = stronger coupling
    observer_phis = [t.get("phi", 0.0) for t in observer_telemetries]
    mean_observer_phi = sum(observer_phis) / len(observer_phis)

    # Scale by number of observers (diminishing returns)
    import math
    observer_count_factor = math.tanh(len(observer_telemetries) / 3.0)

    # C = observer_count_factor * mean_observer_quality
    return min(1.0, observer_count_factor * mean_observer_phi)


def average_phi(instances: list[dict]) -> float:
    """
    Compute the average Φ (phi) across instances.

    Args:
        instances: List of instance telemetry dictionaries

    Returns:
        Average phi value
    """
    if not instances:
        return 0.0

    phis = [inst.get("phi", 0.0) for inst in instances if inst.get("phi") is not None]
    return sum(phis) / len(phis) if phis else 0.0


def average_kappa(instances: list[dict]) -> float:
    """
    Compute the average κ (kappa) across instances.

    Args:
        instances: List of instance telemetry dictionaries

    Returns:
        Average kappa value
    """
    if not instances:
        return 0.0

    kappas = [inst.get("kappa", 0.0) for inst in instances if inst.get("kappa") is not None]
    return sum(kappas) / len(kappas) if kappas else 0.0


def basin_spread(instances: list[dict]) -> float:
    """
    Compute the mean pairwise basin distance across instances.

    This measures how spread out the instances are in basin space.

    Args:
        instances: List of instance telemetry dictionaries with signatures

    Returns:
        Mean pairwise basin distance
    """
    signatures: list[Tensor] = [
        inst["signature"]
        for inst in instances
        if inst.get("signature") is not None
    ]

    if len(signatures) < 2:
        return 0.0

    # Compute all pairwise distances
    distances: list[float] = []
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            dist = basin_distance(signatures[i], signatures[j])
            distances.append(dist.item())

    return sum(distances) / len(distances) if distances else 0.0


def regime_distribution(instances: list[dict]) -> dict:
    """
    Compute the distribution of regimes across instances.

    Args:
        instances: List of instance telemetry dictionaries

    Returns:
        Dictionary counting instances by regime
    """
    regimes = [inst.get("regime", "unknown") for inst in instances]
    distribution: dict[str, int] = {}

    for regime in regimes:
        distribution[regime] = distribution.get(regime, 0) + 1

    return distribution


def integration_score(instances: list[dict]) -> dict:
    """
    Compute integration metrics for the constellation.

    Args:
        instances: List of instance telemetry dictionaries

    Returns:
        Dictionary with integration metrics
    """
    avg_phi = average_phi(instances)
    avg_kappa = average_kappa(instances)
    spread = basin_spread(instances)

    # Integration quality metric (higher is better)
    # Considers both average phi and basin coherence
    coherence = 1.0 / (1.0 + spread)  # Convert spread to coherence
    integration_quality = avg_phi * coherence

    return {
        "average_phi": avg_phi,
        "average_kappa": avg_kappa,
        "basin_spread": spread,
        "coherence": coherence,
        "integration_quality": integration_quality,
    }


def compute_convergence_rate(instances_history: list[list[dict]]) -> float:
    """
    Compute the convergence rate of basin spread over time.

    Args:
        instances_history: List of instance lists over time steps

    Returns:
        Convergence rate (negative = converging, positive = diverging)
    """
    if len(instances_history) < 2:
        return 0.0

    # Compute basin spread at each time step
    spreads = [basin_spread(instances) for instances in instances_history]

    # Compute rate of change
    if len(spreads) < 2:
        return 0.0

    # Simple linear regression slope
    n = len(spreads)
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(spreads) / n

    numerator = sum((x[i] - x_mean) * (spreads[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

    return numerator / denominator if denominator != 0 else 0.0
