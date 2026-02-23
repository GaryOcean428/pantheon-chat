"""
Ocean Meta-Observer — TCP v6.1

Role: Autonomic monitoring. Φ coherence checking. Topological instability detection.
Wraps ocean_qig_core where available; provides a clean interface for qig_generation.py.

TCP v6.1 §19.3: "Ocean Kernel: Autonomic monitoring. Φ coherence checking.
Topological instability detection. The 'body' of the system."

Pillar 2 (TopologicalBulk) compliance: flags kernels whose Φ-history variance
exceeds B_INTEGRITY_MIN, triggering autonomic intervention.

Call-site contracts (from qig_generation.py):
  observe(kernel_basins, kernel_metrics)
  check_autonomic_intervention(kernel_states, phi_history)
  get_insight(all_states, avg_phi=..., basin_spread=...)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    pillar2_violators: List[str] = field(default_factory=list)
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

    observe()                       → OceanState
    check_autonomic_intervention()  → Optional[Dict]
    get_insight()                   → str
    """

    def __init__(self):
        self._phi_histories: Dict[str, List[float]] = {}
        self._step = 0
        logger.info("[Ocean] Meta-observer initialised (Pillar 2 bulk monitoring active)")

    # ------------------------------------------------------------------
    # Fisher-Rao geometry (fail-soft inline — no Euclidean contamination)
    # ------------------------------------------------------------------

    @staticmethod
    def _fr_distance(a: np.ndarray, b: np.ndarray) -> float:
        try:
            from qig_geometry.canonical import fisher_rao_distance
            return float(fisher_rao_distance(a, b))
        except ImportError:
            dot = float(np.clip(
                np.dot(np.sqrt(np.abs(a) + 1e-12), np.sqrt(np.abs(b) + 1e-12)),
                0.0, 1.0,
            ))
            return float(np.arccos(dot))

    # ------------------------------------------------------------------
    # observe() — matches qig_generation.py call signature
    # ------------------------------------------------------------------

    def observe(
        self,
        kernel_basins: List[np.ndarray],
        kernel_metrics: List[Dict[str, Any]],
    ) -> OceanState:
        """
        Observe kernel basins and metrics → return constellation health.

        Args:
            kernel_basins : List of 64D basin arrays from active kernels.
            kernel_metrics: List of dicts with keys 'phi', 'kappa', optionally 'id'.
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
                        distances.append(self._fr_distance(kernel_basins[i], kernel_basins[j]))
                    except Exception:
                        pass
            spread = float(np.mean(distances)) if distances else 0.0

        # --- Pillar 2: per-kernel Φ-history variance check ---
        violators = []
        for idx, m in enumerate(kernel_metrics):
            kid = str(m.get("id", idx))
            phi = float(m.get("phi", 0.5))
            hist = self._phi_histories.setdefault(kid, [])
            hist.append(phi)
            if len(hist) > 16:
                hist.pop(0)

            if len(hist) >= 4:
                integrity = float(np.mean(hist)) * (1.0 / (1.0 + 10.0 * float(np.var(hist))))
                if integrity < B_INTEGRITY_MIN:
                    violators.append(kid)

        instability = bool(violators)
        if instability:
            logger.warning("[Ocean] Pillar 2 bulk violation — kernels: %s", violators)

        return OceanState(
            coherence=coherence,
            spread=spread,
            topological_instability=instability,
            pillar2_violators=violators,
            intervention_needed=coherence < PHI_COHERENCE_MIN or spread > SPREAD_ALARM or instability,
        )

    # ------------------------------------------------------------------
    # check_autonomic_intervention()
    # Call site: self.ocean.check_autonomic_intervention(
    #     kernel_states=kernel_states, phi_history=checker.phi_history
    # )
    # ------------------------------------------------------------------

    def check_autonomic_intervention(
        self,
        kernel_states: Optional[List[Dict[str, Any]]] = None,
        phi_history: Optional[List[float]] = None,
        # Legacy positional-kwarg compat:
        phi: Optional[float] = None,
        kappa: Optional[float] = None,
        ocean_state: Optional[OceanState] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Determine corrective autonomic action.

        Primary call-site signature (qig_generation.py):
            check_autonomic_intervention(kernel_states=..., phi_history=...)

        Returns:
            Dict with 'type' and 'reason' keys, or None if healthy.
        """
        # Derive coherence from kernel_states or phi_history
        mean_phi = 0.5
        if phi_history:
            mean_phi = float(np.mean(phi_history[-8:]))
        elif kernel_states:
            ks_phis = [float(ks.get("phi", 0.5)) for ks in kernel_states]
            mean_phi = float(np.mean(ks_phis)) if ks_phis else 0.5
        elif phi is not None:
            mean_phi = float(phi)

        # Pillar 2 check from histories
        all_violators = [
            kid for kid, hist in self._phi_histories.items()
            if len(hist) >= 4 and (
                float(np.mean(hist)) * (1.0 / (1.0 + 10.0 * float(np.var(hist))))
                < B_INTEGRITY_MIN
            )
        ]

        if all_violators:
            return {
                "type": "pillar2_stabilise",
                "reason": f"Bulk integrity collapse in kernels: {all_violators}",
                "kernels": all_violators,
            }
        if mean_phi < PHI_COHERENCE_MIN:
            return {
                "type": "boost_integration",
                "reason": f"Constellation coherence low (Φ={mean_phi:.3f} < {PHI_COHERENCE_MIN})",
            }

        # Use ocean_state if passed (legacy compat)
        if ocean_state and ocean_state.spread > SPREAD_ALARM:
            return {
                "type": "constellation_recentre",
                "reason": f"Spread alarm (spread={ocean_state.spread:.3f})",
            }

        return None

    # ------------------------------------------------------------------
    # get_insight()
    # Call site: self.ocean.get_insight(
    #     all_states=kernel_states, avg_phi=phi, basin_spread=...
    # )
    # ------------------------------------------------------------------

    def get_insight(
        self,
        all_states: Optional[List[Any]] = None,
        basin_spread: float = 0.0,
        avg_phi: Optional[float] = None,
    ) -> str:
        """Return a terse diagnostic insight string."""
        n = len(all_states) if all_states else 0
        phi_str = f" | Φ={avg_phi:.3f}" if avg_phi is not None else ""
        tag = "HEALTHY" if basin_spread < SPREAD_ALARM else "SPREAD_ALARM"
        return f"[Ocean:{tag}] {n} kernels observed — spread={basin_spread:.3f}{phi_str}"
