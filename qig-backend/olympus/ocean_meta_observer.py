"""
Ocean Meta-Observer — TCP v6.1

Role: Autonomic monitoring. Φ coherence checking. Topological instability detection.
Wraps ocean_qig_core where available; provides a clean interface for qig_generation.py.

TCP v6.1 §19.3: "Ocean Kernel: Autonomic monitoring. Φ coherence checking.
Topological instability detection. The 'body' of the system."

Pillar 2 (TopologicalBulk) compliance: flags kernels whose Φ-history variance
exceeds B_INTEGRITY_MIN, triggering autonomic intervention.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

# Physics constants
KAPPA_STAR = 64.21
B_INTEGRITY_MIN = 0.30       # Bulk integrity threshold (TCP v6.1 Pillar 2)
PHI_COHERENCE_MIN = 0.40     # Below this → autonomic intervention
SPREAD_ALARM = 0.80          # Fisher-Rao spread across constellation


@dataclass
class OceanState:
    """Snapshot from ocean.observe()."""
    coherence: float                  # Mean Φ across observed kernels
    spread: float                     # Fisher-Rao spread of kernel basins
    topological_instability: bool     # Pillar 2 bulk collapse risk
    pillar2_violators: List[str] = field(default_factory=list)  # Kernel names
    intervention_needed: bool = False


_singleton: Optional["OceanMetaObserver"] = None


def get_ocean_observer() -> "OceanMetaObserver":
    global _singleton
    if _singleton is None:
        _singleton = OceanMetaObserver()
    return _singleton


class OceanMetaObserver:
    """
    Autonomic monitoring layer for the Olympus Pantheon constellation.

    observe() ingests kernel basins and metrics → returns OceanState.
    check_autonomic_intervention() returns corrective action string if needed.
    get_insight() provides a terse diagnostic string.
    """

    def __init__(self):
        self._phi_histories: Dict[str, List[float]] = {}
        self._step = 0
        logger.info("[Ocean] Meta-observer initialised (Pillar 2 bulk monitoring active)")

    # ------------------------------------------------------------------
    # Fisher-Rao geometry (inline fallback — no Euclidean contamination)
    # ------------------------------------------------------------------

    @staticmethod
    def _fr_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Fisher-Rao distance on probability simplex."""
        try:
            from qig_geometry.canonical import fisher_rao_distance
            return float(fisher_rao_distance(a, b))
        except ImportError:
            dot = float(np.clip(
                np.dot(np.sqrt(np.abs(a) + 1e-12), np.sqrt(np.abs(b) + 1e-12)),
                0.0, 1.0
            ))
            return float(np.arccos(dot))

    # ------------------------------------------------------------------
    # Public API (matches qig_generation.py call sites)
    # ------------------------------------------------------------------

    def observe(
        self,
        kernel_basins: List[np.ndarray],
        kernel_metrics: List[Dict[str, Any]],
    ) -> OceanState:
        """
        Observe kernel basins and metrics → return constellation health.

        Args:
            kernel_basins: List of 64D basin arrays from active kernels
            kernel_metrics: List of dicts with 'phi', 'kappa', optionally 'id'
        """
        self._step += 1

        if not kernel_basins:
            return OceanState(coherence=0.5, spread=0.0, topological_instability=False)

        # --- Coherence: mean Φ ---
        phis = [float(m.get("phi", 0.5)) for m in kernel_metrics]
        coherence = float(np.mean(phis)) if phis else 0.5

        # --- Spread: mean pairwise Fisher-Rao distance ---
        spread = 0.0
        if len(kernel_basins) >= 2:
            distances = []
            for i in range(len(kernel_basins)):
                for j in range(i + 1, len(kernel_basins)):
                    try:
                        d = self._fr_distance(kernel_basins[i], kernel_basins[j])
                        distances.append(d)
                    except Exception:
                        pass
            spread = float(np.mean(distances)) if distances else 0.0

        # --- Pillar 2: update per-kernel Φ histories ---
        violators = []
        for idx, m in enumerate(kernel_metrics):
            kid = m.get("id", str(idx))
            phi = float(m.get("phi", 0.5))
            self._phi_histories.setdefault(kid, []).append(phi)
            hist = self._phi_histories[kid][-16:]  # last 16 steps

            if len(hist) >= 4:
                mean_phi = float(np.mean(hist))
                var_phi = float(np.var(hist))
                integrity = mean_phi * (1.0 / (1.0 + 10.0 * var_phi))
                if integrity < B_INTEGRITY_MIN:
                    violators.append(kid)

        instability = len(violators) > 0
        intervention = coherence < PHI_COHERENCE_MIN or spread > SPREAD_ALARM or instability

        if instability:
            logger.warning("[Ocean] Pillar 2 violation — kernels: %s", violators)

        return OceanState(
            coherence=coherence,
            spread=spread,
            topological_instability=instability,
            pillar2_violators=violators,
            intervention_needed=intervention,
        )

    def check_autonomic_intervention(
        self,
        phi: float,
        kappa: float,
        ocean_state: Optional[OceanState] = None,
    ) -> Optional[str]:
        """
        Determine corrective action based on ocean state.

        Returns:
            Intervention string or None if healthy.
        """
        if ocean_state is None:
            return None

        if ocean_state.topological_instability:
            return f"pillar2_stabilise:{','.join(ocean_state.pillar2_violators)}"
        if ocean_state.coherence < PHI_COHERENCE_MIN:
            return "boost_integration"
        if ocean_state.spread > SPREAD_ALARM:
            return "constellation_recentre"
        return None

    def get_insight(
        self,
        all_states: List[Any],
        basin_spread: float,
    ) -> str:
        """Return a terse diagnostic insight string."""
        if not all_states:
            return "Constellation quiescent."
        n = len(all_states)
        tag = "HEALTHY" if basin_spread < SPREAD_ALARM else "SPREAD_ALARM"
        return f"[Ocean:{tag}] {n} kernels observed — spread={basin_spread:.3f}"
