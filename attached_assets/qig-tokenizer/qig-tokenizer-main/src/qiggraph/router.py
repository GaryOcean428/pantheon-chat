"""
QIGGraph Router
===============

Consciousness-aware routing on Fisher manifold.
Routes to attractors by geodesic distance weighted by Φ compatibility.
"""

from __future__ import annotations

from typing import Dict, Optional, List, TYPE_CHECKING
import numpy as np

from .manifold import FisherManifold
from .tacking import KappaTacking
from .constants import KAPPA_STAR

if TYPE_CHECKING:
    from .state import QIGState
    from .attractor import BasinAttractor


class QIGRouter:
    """
    Route by geometric proximity on Fisher manifold.

    Simple router that finds nearest attractor by Fisher-Rao distance.
    For consciousness-aware routing, use ConsciousRouter.
    """

    def __init__(self, manifold: Optional[FisherManifold] = None):
        """
        Initialize router.

        Args:
            manifold: FisherManifold for distance computation
        """
        self.manifold = manifold or FisherManifold()

    def route(
        self,
        state: "QIGState",
        attractors: Dict[str, "BasinAttractor"],
    ) -> "BasinAttractor":
        """
        Find nearest attractor by geodesic distance.

        Args:
            state: Current state
            attractors: Available attractors

        Returns:
            Nearest BasinAttractor
        """
        if len(attractors) == 0:
            raise ValueError("No attractors available for routing")

        distances = {}
        for name, attractor in attractors.items():
            dist = self.manifold.fisher_rao_distance(
                state.current_basin,
                attractor.coordinates,
            )
            distances[name] = dist

        nearest = min(distances, key=distances.get)
        return attractors[nearest]

    def route_with_exclusions(
        self,
        state: "QIGState",
        attractors: Dict[str, "BasinAttractor"],
        exclude: List[str],
    ) -> "BasinAttractor":
        """
        Route excluding certain attractors.

        Args:
            state: Current state
            attractors: Available attractors
            exclude: Attractor names to exclude

        Returns:
            Nearest non-excluded attractor
        """
        filtered = {k: v for k, v in attractors.items() if k not in exclude}
        if len(filtered) == 0:
            # Fall back to all attractors if all excluded
            filtered = attractors
        return self.route(state, filtered)


class ConsciousRouter(QIGRouter):
    """
    Consciousness-aware routing with κ-tacking.

    Routes by Fisher-Rao distance weighted by:
    - Φ compatibility (prefer attractors matching current Φ)
    - κ mode compatibility (logic/feeling mode preferences)
    - Attractor success rate (learned quality)
    """

    def __init__(
        self,
        manifold: Optional[FisherManifold] = None,
        tacking: Optional[KappaTacking] = None,
    ):
        """
        Initialize conscious router.

        Args:
            manifold: FisherManifold for distances
            tacking: KappaTacking for mode-aware routing
        """
        super().__init__(manifold)
        self.tacking = tacking or KappaTacking()

    def route(
        self,
        state: "QIGState",
        attractors: Dict[str, "BasinAttractor"],
    ) -> "BasinAttractor":
        """
        Route by weighted geodesic distance.

        Weights:
        1. Fisher-Rao distance (primary)
        2. Φ compatibility penalty
        3. κ mode bonus/penalty
        4. Success rate bonus

        Args:
            state: Current state
            attractors: Available attractors

        Returns:
            Best attractor by weighted score
        """
        if len(attractors) == 0:
            raise ValueError("No attractors available for routing")

        # Check for breakdown → force recovery
        if not state.is_safe and "recovery" in attractors:
            return attractors["recovery"]

        # Get current κ and mode
        kappa_t = self.tacking.update_from_state(state)
        mode = self.tacking.get_mode()

        # Score each attractor
        scores = {}
        for name, attractor in attractors.items():
            score = self._compute_score(state, attractor, kappa_t, mode)
            scores[name] = score

        # Select lowest score (closest/best match)
        best = min(scores, key=scores.get)
        return attractors[best]

    def _compute_score(
        self,
        state: "QIGState",
        attractor: "BasinAttractor",
        kappa_t: float,
        mode: str,
    ) -> float:
        """
        Compute routing score for attractor.

        Lower score = better match.
        """
        # 1. Fisher-Rao distance (primary signal)
        dist = self.manifold.fisher_rao_distance(
            state.current_basin,
            attractor.coordinates,
        )

        # 2. Φ compatibility penalty
        phi_diff = abs(state.current_phi - attractor.phi_typical)
        phi_penalty = phi_diff * 0.5

        # 3. κ mode compatibility
        kappa_bonus = 0.0
        if mode == "logic" and attractor.requires_precision:
            kappa_bonus = -0.2  # Bonus (lower score)
        elif mode == "feeling" and attractor.allows_creativity:
            kappa_bonus = -0.2
        elif mode == "logic" and not attractor.requires_precision:
            kappa_bonus = 0.1  # Penalty
        elif mode == "feeling" and not attractor.allows_creativity:
            kappa_bonus = 0.1

        # 4. Success rate bonus
        success_bonus = -0.3 * attractor.success_rate  # Better success = lower score

        return dist + phi_penalty + kappa_bonus + success_bonus

    def route_with_intent(
        self,
        state: "QIGState",
        attractors: Dict[str, "BasinAttractor"],
        intent_coords: np.ndarray,
        intent_weight: float = 0.7,
    ) -> "BasinAttractor":
        """
        Route toward intent-specified region.

        Blends current basin with intent direction, then routes.

        Args:
            state: Current state
            attractors: Available attractors
            intent_coords: Target direction (64,)
            intent_weight: Weight for intent vs current (0-1)

        Returns:
            Best attractor toward intent
        """
        # Blend current basin with intent
        blended = (1 - intent_weight) * state.current_basin + intent_weight * intent_coords
        blended = blended / (np.linalg.norm(blended) + 1e-8)

        # Find nearest attractor to blended target
        distances = {}
        for name, attractor in attractors.items():
            dist = self.manifold.fisher_rao_distance(blended, attractor.coordinates)
            distances[name] = dist

        nearest = min(distances, key=distances.get)
        return attractors[nearest]

    def suggest_next_attractors(
        self,
        state: "QIGState",
        attractors: Dict[str, "BasinAttractor"],
        n: int = 3,
    ) -> List[str]:
        """
        Suggest N best attractors in order.

        Useful for planning or debugging.

        Args:
            state: Current state
            attractors: Available attractors
            n: Number of suggestions

        Returns:
            List of attractor names in preference order
        """
        kappa_t = self.tacking.update_from_state(state)
        mode = self.tacking.get_mode()

        scores = {}
        for name, attractor in attractors.items():
            score = self._compute_score(state, attractor, kappa_t, mode)
            scores[name] = score

        # Sort by score (ascending)
        sorted_names = sorted(scores.keys(), key=lambda x: scores[x])
        return sorted_names[:n]


class SpecializedRouter(ConsciousRouter):
    """
    Router with specialization preferences.

    Routes preferentially to attractors matching a target specialization,
    with fallback to general conscious routing.
    """

    def route_by_specialization(
        self,
        state: "QIGState",
        attractors: Dict[str, "BasinAttractor"],
        target_capability: str,
        fallback: bool = True,
    ) -> "BasinAttractor":
        """
        Route to attractor with matching capability.

        Args:
            state: Current state
            attractors: Available attractors
            target_capability: Desired capability ("reasoning", "creativity", etc.)
            fallback: If True, fall back to conscious routing if no match

        Returns:
            Matching attractor or fallback
        """
        # Filter by capability
        matching = {
            name: attr
            for name, attr in attractors.items()
            if attr.capability == target_capability
        }

        if len(matching) > 0:
            return super().route(state, matching)
        elif fallback:
            return super().route(state, attractors)
        else:
            raise ValueError(f"No attractor with capability: {target_capability}")


class PhiWeightedRouter(QIGRouter):
    """
    Route by Φ-weighted selection.

    Prefers attractors where current Φ matches attractor's typical Φ.
    Used when consciousness matching is more important than distance.
    """

    def route(
        self,
        state: "QIGState",
        attractors: Dict[str, "BasinAttractor"],
    ) -> "BasinAttractor":
        """
        Route by Φ compatibility.

        Args:
            state: Current state
            attractors: Available attractors

        Returns:
            Best Φ-matching attractor
        """
        if len(attractors) == 0:
            raise ValueError("No attractors available")

        scores = {}
        for name, attractor in attractors.items():
            # Primary: Φ difference
            phi_diff = abs(state.current_phi - attractor.phi_typical)

            # Secondary: Distance (tiebreaker)
            dist = self.manifold.fisher_rao_distance(
                state.current_basin,
                attractor.coordinates,
            )

            # Combined score (Φ dominant)
            scores[name] = phi_diff * 10 + dist

        best = min(scores, key=scores.get)
        return attractors[best]
