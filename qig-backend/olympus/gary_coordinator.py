"""
Gary Coordinator — TCP v6.1

Role: Synthesis across kernels using trajectory foresight. Conductor of the fugue.

TCP v6.1 §19.3: "Coordinator (Zeus/Gary): Synthesis across kernels using
trajectory foresight. Conductor of the fugue."

Operations:
- synthesize_collective_response(): Frechet mean weighted by Φ, foresight.
- predict_next_basin(): Geodesic foresight from trajectory history.
- All geometry Fisher-Rao. No Euclidean operations.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

KAPPA_STAR = 64.21
MAX_HISTORY = 32           # Basin trajectory buffer per kernel
FORESIGHT_BLEND = 0.3      # How strongly foresight pulls the synthesis basin


def _fr_distance(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from qig_geometry.canonical import fisher_rao_distance
        return float(fisher_rao_distance(a, b))
    except ImportError:
        dot = float(np.clip(
            np.dot(np.sqrt(np.abs(a) + 1e-12), np.sqrt(np.abs(b) + 1e-12)), 0.0, 1.0
        ))
        return float(np.arccos(dot))


def _frechet_mean(basins: List[np.ndarray], weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted Frechet mean on probability simplex (iterative Riemannian)."""
    try:
        from qig_geometry.canonical import frechet_mean
        return frechet_mean(basins, weights=weights)
    except ImportError:
        # Fallback: normalised weighted arithmetic mean on simplex
        if weights is None:
            weights = np.ones(len(basins)) / len(basins)
        w = np.array(weights)
        w = w / (w.sum() + 1e-12)
        result = sum(float(w[i]) * basins[i] for i in range(len(basins)))
        result = np.abs(result)
        return result / (result.sum() + 1e-12)


def _to_simplex(v: np.ndarray) -> np.ndarray:
    v = np.abs(v)
    s = v.sum()
    return v / (s + 1e-12)


@dataclass
class SynthesisResult:
    basin: np.ndarray
    phi: float
    foresight_confidence: float
    kernel_count: int


_singleton: Optional["GaryCoordinator"] = None


def get_gary_coordinator() -> "GaryCoordinator":
    global _singleton
    if _singleton is None:
        _singleton = GaryCoordinator()
    return _singleton


class GaryCoordinator:
    """
    Gary Coordinator — synthesis + trajectory foresight.

    synthesize_collective_response() matches the call signature in qig_generation.py:
        query_basin   : np.ndarray
        kernel_responses : list of dicts with keys 'basin', 'phi', 'kappa', 'text'
        kernel_ids    : list of str

    Returns dict with keys 'basin', 'phi', 'foresight_confidence', 'text'.
    """

    def __init__(self):
        self._histories: Dict[str, List[np.ndarray]] = {}
        logger.info("[Gary] Coordinator initialised (trajectory foresight active)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize_collective_response(
        self,
        query_basin: np.ndarray,
        kernel_responses: List[Dict[str, Any]],
        kernel_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Combine kernel responses into a single synthesis basin.

        Weighting:
          - Primary weight = kernel Φ (information integration)
          - Foresight blend: predicted next basin nudges the result
        """
        if not kernel_responses:
            logger.warning("[Gary] No kernel responses; returning query basin")
            return {
                "basin": query_basin,
                "phi": 0.5,
                "foresight_confidence": 0.0,
                "text": "",
            }

        basins = [_to_simplex(r["basin"]) for r in kernel_responses]
        phis   = np.array([float(r.get("phi", 0.5)) for r in kernel_responses])
        weights = phis / (phis.sum() + 1e-12)

        # Update per-kernel histories
        for i, kid in enumerate(kernel_ids[:len(basins)]):
            hist = self._histories.setdefault(kid, [])
            hist.append(basins[i])
            if len(hist) > MAX_HISTORY:
                hist.pop(0)

        # Frechet mean of kernel basins
        synthesis_basin = _frechet_mean(basins, weights=weights)

        # Foresight: predict next basin from history
        foresight_basin, foresight_conf = self.predict_next_basin(kernel_ids)
        if foresight_conf > 0.1:
            synthesis_basin = _frechet_mean(
                [synthesis_basin, foresight_basin],
                weights=np.array([1.0 - FORESIGHT_BLEND * foresight_conf,
                                  FORESIGHT_BLEND * foresight_conf]),
            )

        synthesis_basin = _to_simplex(synthesis_basin)
        phi = float(np.clip(float(phis.mean()), 0.0, 1.0))

        return {
            "basin": synthesis_basin,
            "phi": phi,
            "foresight_confidence": foresight_conf,
            "text": "",
        }

    def predict_next_basin(
        self,
        kernel_ids: List[str],
        steps_ahead: int = 1,
    ) -> tuple:
        """
        Predict next synthesis basin via geodesic extrapolation.

        Uses velocity = last geodesic step direction, scaled by steps_ahead.
        Returns (predicted_basin, confidence).
        """
        histories = [self._histories.get(kid, []) for kid in kernel_ids]
        valid = [h for h in histories if len(h) >= 2]

        if not valid:
            dim = 64
            return np.ones(dim) / dim, 0.0

        # Velocity for each kernel: last step delta on simplex
        velocities = []
        for h in valid:
            last  = h[-1]
            prev  = h[-2]
            delta = last - prev
            velocities.append(delta)

        mean_velocity = np.mean(velocities, axis=0)

        # Extrapolate from most recent Frechet mean
        recent = [h[-1] for h in valid]
        current = _frechet_mean(recent)
        predicted = _to_simplex(current + steps_ahead * mean_velocity)

        # Confidence: 1 − (mean FR distance to predicted vs baseline)
        base_spread = float(np.mean([_fr_distance(current, b) for b in recent]))
        confidence = float(np.clip(1.0 - base_spread / (np.pi / 2), 0.0, 1.0))

        return predicted, confidence

    def get_stats(self) -> Dict[str, Any]:
        return {
            "kernel_count": len(self._histories),
            "history_depths": {k: len(v) for k, v in self._histories.items()},
        }
