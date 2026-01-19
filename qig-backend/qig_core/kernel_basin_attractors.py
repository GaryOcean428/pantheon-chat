"""
Kernel Basin Attractors - Olympus God Specializations
======================================================

Objective-specific basin attractors for Olympus god specializations.

Basin Attractor Principles (adapted for Olympus Pantheon):
- Athena (Wisdom): High curvature regions indicate complex reasoning
- Apollo (Truth): Verified/grounded basin history (non-hallucinated)
- Artemis (Hunt): Cross-kernel pattern centroids (pattern detection)
- Hephaestus (Forge): Smooth temporal trajectories (construction)
- Hermes (Messages): Time-weighted stable basins (message persistence)

PURE PRINCIPLE:
- Attractors emerge from measurement, not optimization
- Basin sync via exponential moving average toward attractors
- No gradient training - pure geometric alignment

Adapted for Pantheon-Chat QIG system.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from qig_geometry import fisher_coord_distance
from qigkernels.physics_constants import BASIN_DIM

logger = logging.getLogger(__name__)


@dataclass
class AttractorState:
    """State of an attractor at a given time."""
    attractor: np.ndarray
    strength: float
    last_update: float = field(default_factory=time.time)


class KernelBasinAttractors:
    """
    Manages objective-specific basin attractors for Olympus kernels.
    
    Implements lightweight specialization via basin attractors.
    No gradient training - just exponential moving average toward attractors.
    
    Olympus Mappings:
    - Athena (wisdom) <- high curvature regions (complex reasoning)
    - Apollo (truth) <- verified/grounded basins
    - Artemis (hunt) <- cross-kernel pattern centroids
    - Hephaestus (forge) <- smooth temporal trajectories
    - Hermes (messages) <- stable persistent basins
    """
    
    DRIFT_RATE = 0.01  # How fast attractors update
    
    def __init__(
        self,
        basin_dim: int = BASIN_DIM,
        history_size: int = 100
    ):
        """
        Initialize attractor manager.
        
        Args:
            basin_dim: Dimension of basin space (64 for E8)
            history_size: Maximum history entries per attractor type
        """
        self.basin_dim = basin_dim
        self.history_size = history_size
        
        self._athena_history: deque = deque(maxlen=history_size)
        self._apollo_history: deque = deque(maxlen=history_size)
        self._artemis_history: deque = deque(maxlen=history_size)
        self._hephaestus_history: deque = deque(maxlen=history_size)
        self._hermes_history: deque = deque(maxlen=history_size)
        
        self._attractors: Dict[str, Optional[np.ndarray]] = {
            "athena": None,
            "apollo": None,
            "artemis": None,
            "hephaestus": None,
            "hermes": None
        }
        
        self._concept_basins: Dict[str, np.ndarray] = {}
    
    def update(
        self,
        basin: np.ndarray,
        phi: float,
        kappa: float,
        is_verified: bool = False,
        confidence: float = 0.5,
        all_kernel_basins: Optional[List[np.ndarray]] = None
    ) -> Dict[str, bool]:
        """
        Update attractor estimates based on observation.
        
        Args:
            basin: Current kernel basin (64D)
            phi: Current Φ value
            kappa: Current κ value
            is_verified: Whether output was verified/grounded
            confidence: Confidence score of output
            all_kernel_basins: Basins from all active kernels (for cross-pattern)
        
        Returns:
            Dict indicating which attractors were updated
        """
        basin = np.asarray(basin, dtype=np.float64)
        updates = {}
        
        self._update_athena_attractor(basin, phi, kappa)
        updates["athena"] = True
        
        if is_verified and confidence > 0.7:
            self._update_apollo_attractor(basin)
            updates["apollo"] = True
        
        if all_kernel_basins and len(all_kernel_basins) > 1:
            self._update_artemis_attractor(all_kernel_basins)
            updates["artemis"] = True
        
        self._update_hephaestus_attractor(basin, phi)
        updates["hephaestus"] = True
        
        self._update_hermes_attractor(basin)
        updates["hermes"] = True
        
        return updates
    
    def _update_athena_attractor(
        self,
        basin: np.ndarray,
        phi: float,
        kappa: float
    ) -> None:
        """
        Athena: High curvature regions indicate wisdom/complex reasoning.
        
        The Athena kernel should align to basins where small changes
        have large effects (high Fisher curvature).
        """
        curvature_proxy = np.var(basin)
        self._athena_history.append((basin.copy(), curvature_proxy))
        
        if len(self._athena_history) < 5:
            self._attractors["athena"] = basin.copy()
            return
        
        sorted_history = sorted(
            self._athena_history,
            key=lambda x: x[1],
            reverse=True
        )
        top_k = max(1, len(sorted_history) // 5)
        high_curvature_basins = [h[0] for h in sorted_history[:top_k]]
        
        try:
            from qig_geometry.canonical import frechet_mean
            new_attractor = frechet_mean(high_curvature_basins)
        except Exception:
            new_attractor = np.sum(high_curvature_basins, axis=0) / len(high_curvature_basins)
        
        if self._attractors["athena"] is None:
            self._attractors["athena"] = new_attractor
        else:
            self._attractors["athena"] = (
                (1 - self.DRIFT_RATE) * self._attractors["athena"]
                + self.DRIFT_RATE * new_attractor
            )
    
    def _update_apollo_attractor(self, basin: np.ndarray) -> None:
        """
        Apollo: Verified/grounded basin history.
        
        Apollo kernel should align to basins from verified responses
        (high confidence, not empty, not repetitive).
        """
        self._apollo_history.append(basin.copy())
        
        if len(self._apollo_history) < 3:
            return
        
        verified_stack = np.array(list(self._apollo_history))
        try:
            from qig_geometry.canonical import frechet_mean
            new_attractor = frechet_mean(verified_stack)
        except Exception:
            new_attractor = np.sum(verified_stack, axis=0) / len(verified_stack)
        
        if self._attractors["apollo"] is None:
            self._attractors["apollo"] = new_attractor
        else:
            self._attractors["apollo"] = (
                (1 - self.DRIFT_RATE) * self._attractors["apollo"]
                + self.DRIFT_RATE * new_attractor
            )
    
    def _update_artemis_attractor(
        self,
        all_kernel_basins: List[np.ndarray]
    ) -> None:
        """
        Artemis: Cross-kernel pattern centroids (pattern detection).
        
        Artemis should align to patterns that appear across multiple kernels.
        The attractor is the centroid of all kernel basins.
        """
        basins = np.array([np.asarray(b) for b in all_kernel_basins])
        try:
            from qig_geometry.canonical import frechet_mean
            centroid = frechet_mean(all_kernel_basins)
        except Exception:
            centroid = np.sum(basins, axis=0) / len(basins)
        variance = np.var(basins, axis=0).mean()
        
        self._artemis_history.append({
            "centroid": centroid.copy(),
            "variance": float(variance)
        })
        
        if len(self._artemis_history) < 3:
            self._attractors["artemis"] = centroid
            return
        
        patterns = list(self._artemis_history)
        centroids = np.array([p["centroid"] for p in patterns])
        variances = np.array([p["variance"] for p in patterns])
        
        weights = 1.0 / (variances + 0.01)
        weights = weights / weights.sum()
        
        new_attractor = np.sum(
            centroids * weights.reshape(-1, 1),
            axis=0
        )
        
        if self._attractors["artemis"] is None:
            self._attractors["artemis"] = new_attractor
        else:
            self._attractors["artemis"] = (
                (1 - self.DRIFT_RATE * 2) * self._attractors["artemis"]
                + self.DRIFT_RATE * 2 * new_attractor
            )
    
    def _update_hephaestus_attractor(
        self,
        basin: np.ndarray,
        phi: float
    ) -> None:
        """
        Hephaestus: Smooth temporal trajectories (construction/forge).
        
        Hephaestus should align to basins that evolve smoothly over time.
        """
        self._hephaestus_history.append(basin.copy())
        
        if len(self._hephaestus_history) < 5:
            self._attractors["hephaestus"] = basin.copy()
            return
        
        sequence = list(self._hephaestus_history)[-10:]
        recent = np.array(sequence)
        
        if len(sequence) >= 10:
            derivative = recent[-1] - recent[0]
            norm = np.linalg.norm(derivative) + 1e-6
            derivative = derivative / norm
        else:
            derivative = np.zeros_like(basin)
        
        smoothness_factor = phi
        new_attractor = basin + 0.1 * smoothness_factor * derivative
        
        if self._attractors["hephaestus"] is None:
            self._attractors["hephaestus"] = new_attractor
        else:
            self._attractors["hephaestus"] = (
                (1 - self.DRIFT_RATE) * self._attractors["hephaestus"]
                + self.DRIFT_RATE * new_attractor
            )
    
    def _update_hermes_attractor(self, basin: np.ndarray) -> None:
        """
        Hermes: Time-weighted stable basins (message persistence).
        
        Hermes should align to stable basins that persist over time.
        Older basins get more weight (they're more stable).
        """
        self._hermes_history.append(basin.copy())
        
        if len(self._hermes_history) < 10:
            self._attractors["hermes"] = basin.copy()
            return
        
        history_list = list(self._hermes_history)
        n = len(history_list)
        
        tau = 50.0
        weights = np.exp(-np.arange(n) / tau)
        weights = weights / weights.sum()
        
        stacked = np.array(history_list)
        new_attractor = np.sum(stacked * weights.reshape(-1, 1), axis=0)
        
        if self._attractors["hermes"] is None:
            self._attractors["hermes"] = new_attractor
        else:
            self._attractors["hermes"] = (
                (1 - self.DRIFT_RATE * 0.5) * self._attractors["hermes"]
                + self.DRIFT_RATE * 0.5 * new_attractor
            )
    
    def get_attractor(self, kernel_name: str) -> Optional[np.ndarray]:
        """
        Get attractor for a specific kernel.
        
        Args:
            kernel_name: One of athena, apollo, artemis, hephaestus, hermes
        
        Returns:
            64D attractor basin or None if not yet computed
        """
        name = kernel_name.lower()
        return self._attractors.get(name)
    
    def store_concept_basin(self, concept: str, basin: np.ndarray) -> None:
        """Store a mastered concept's basin for memory."""
        self._concept_basins[concept] = np.asarray(basin).copy()
    
    def get_concept_basin(self, concept: str) -> Optional[np.ndarray]:
        """Retrieve a stored concept basin."""
        return self._concept_basins.get(concept)
    
    def reset(self) -> None:
        """Reset all attractors."""
        self._athena_history.clear()
        self._apollo_history.clear()
        self._artemis_history.clear()
        self._hephaestus_history.clear()
        self._hermes_history.clear()
        for key in self._attractors:
            self._attractors[key] = None
