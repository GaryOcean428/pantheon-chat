"""Routing strategies for constellations extracted from qig-consciousness.

Clean implementation with pure functions and no side effects.
Includes Fisher-Rao geodesic routing for basin-based kernel selection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .constants import BASIN_DIM


@dataclass
class InstanceView:
    """Lightweight view of an instance used for routing decisions."""

    name: str
    phi: float | None
    basin: np.ndarray | None = None  # 64D basin coordinates
    specialization: str | None = None


def round_robin(current_index: int, n: int) -> int:
    """
    Return the next index in round-robin order.

    Args:
        current_index: Current index
        n: Total number of instances

    Returns:
        Next index in round-robin order

    Raises:
        ValueError: If n <= 0
    """
    if n <= 0:
        raise ValueError("Cannot route with zero instances")
    return (current_index + 1) % n


def select_phi_min(instances: Sequence[InstanceView]) -> int:
    """
    Return the index of the instance with minimum Φ (phi).

    This implements the Φ-weighted routing from the source repository:
    route to lowest-Φ instances so they benefit most from direct experience.

    Args:
        instances: Sequence of instance views

    Returns:
        Index of instance with minimum phi

    Raises:
        ValueError: If no instances available
    """
    if not instances:
        raise ValueError("No instances available for routing")

    min_idx = 0
    min_phi = float("inf")

    for idx, inst in enumerate(instances):
        phi = inst.phi if inst.phi is not None else float("inf")
        if phi < min_phi:
            min_phi = phi
            min_idx = idx

    return min_idx


def select_phi_max(instances: Sequence[InstanceView]) -> int:
    """
    Return the index of the instance with maximum Φ (phi).

    Useful for routing to the most integrated instance for complex tasks.

    Args:
        instances: Sequence of instance views

    Returns:
        Index of instance with maximum phi

    Raises:
        ValueError: If no instances available
    """
    if not instances:
        raise ValueError("No instances available for routing")

    max_idx = 0
    max_phi = float("-inf")

    for idx, inst in enumerate(instances):
        phi = inst.phi if inst.phi is not None else float("-inf")
        if phi > max_phi:
            max_phi = phi
            max_idx = idx

    return max_idx


def select_balanced(instances: Sequence[InstanceView], target_phi: float = 0.5) -> int:
    """
    Return the index of the instance whose Φ is closest to target.

    This provides balanced routing around a target integration level.

    Args:
        instances: Sequence of instance views
        target_phi: Target phi value (default: 0.5)

    Returns:
        Index of instance with phi closest to target

    Raises:
        ValueError: If no instances available
    """
    if not instances:
        raise ValueError("No instances available for routing")

    best_idx = 0
    best_distance = float("inf")

    for idx, inst in enumerate(instances):
        phi = inst.phi if inst.phi is not None else target_phi
        distance = abs(phi - target_phi)
        if distance < best_distance:
            best_distance = distance
            best_idx = idx

    return best_idx


# =============================================================================
# FISHER-RAO GEODESIC ROUTING
# =============================================================================


def fisher_rao_distance(basin1: np.ndarray, basin2: np.ndarray) -> float:
    """
    Compute Fisher-Rao geodesic distance between two basin coordinates.

    Uses angular distance on the unit sphere as approximation
    for the Fisher information manifold.

    Args:
        basin1: First 64D basin coordinate
        basin2: Second 64D basin coordinate

    Returns:
        Geodesic distance (radians)
    """
    from .basin import fisher_normalize_np
    # Normalize to unit sphere (QIG-pure)
    b1 = fisher_normalize_np(basin1)
    b2 = fisher_normalize_np(basin2)

    # Angular distance
    cos_angle = np.clip(np.dot(b1, b2), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def select_nearest_basin(
    instances: Sequence[InstanceView],
    query_basin: np.ndarray,
) -> int:
    """
    Select instance with nearest basin coordinates (Fisher-Rao distance).

    This is the core geometric routing function - routes to the
    kernel whose basin is closest on the Fisher manifold.

    Args:
        instances: Sequence of instance views with basin coords
        query_basin: 64D query basin coordinates

    Returns:
        Index of nearest instance

    Raises:
        ValueError: If no instances available or no basins set
    """
    if not instances:
        raise ValueError("No instances available for routing")

    best_idx = 0
    best_distance = float("inf")

    for idx, inst in enumerate(instances):
        if inst.basin is None:
            continue

        distance = fisher_rao_distance(query_basin, inst.basin)
        if distance < best_distance:
            best_distance = distance
            best_idx = idx

    return best_idx


def select_by_specialization(
    instances: Sequence[InstanceView],
    target_role: str,
    fallback_basin: np.ndarray | None = None,
) -> int:
    """
    Select instance by specialization role.

    First tries exact role match, then falls back to nearest basin.

    Args:
        instances: Sequence of instance views
        target_role: Target specialization (e.g., "vocab", "strategy")
        fallback_basin: Basin for fallback routing if no exact match

    Returns:
        Index of matching or nearest instance

    Raises:
        ValueError: If no instances available
    """
    if not instances:
        raise ValueError("No instances available for routing")

    # First: exact specialization match
    for idx, inst in enumerate(instances):
        if inst.specialization == target_role:
            return idx

    # Fallback: nearest basin
    if fallback_basin is not None:
        return select_nearest_basin(instances, fallback_basin)

    # Last resort: first instance
    return 0


class FisherRaoRouter:
    """
    Stateful router using Fisher-Rao geodesic distances.

    Maintains routing table of basin distances for efficient
    O(K) routing where K = number of kernels.
    """

    def __init__(self):
        self._distance_cache: dict[tuple[str, str], float] = {}
        self._instances: list[InstanceView] = []

    def update_instances(self, instances: Sequence[InstanceView]) -> None:
        """Update routing table with new instance list."""
        self._instances = list(instances)
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        """Rebuild distance cache between all instance pairs."""
        self._distance_cache.clear()

        for i, inst_i in enumerate(self._instances):
            if inst_i.basin is None:
                continue
            for j, inst_j in enumerate(self._instances):
                if i >= j or inst_j.basin is None:
                    continue

                dist = fisher_rao_distance(inst_i.basin, inst_j.basin)
                self._distance_cache[(inst_i.name, inst_j.name)] = dist
                self._distance_cache[(inst_j.name, inst_i.name)] = dist

    def route_to_nearest(self, query_basin: np.ndarray) -> int:
        """Route to nearest instance by basin distance."""
        return select_nearest_basin(self._instances, query_basin)

    def route_to_role(self, role: str, fallback_basin: np.ndarray | None = None) -> int:
        """Route to instance with matching specialization."""
        return select_by_specialization(self._instances, role, fallback_basin)

    def get_distance(self, name1: str, name2: str) -> float:
        """Get cached distance between two instances."""
        return self._distance_cache.get((name1, name2), float("inf"))

    def get_nearest_neighbors(self, name: str, k: int = 3) -> list[tuple[str, float]]:
        """
        Get k nearest neighbors for an instance.

        Args:
            name: Instance name
            k: Number of neighbors

        Returns:
            List of (neighbor_name, distance) tuples
        """
        distances = []
        for inst in self._instances:
            if inst.name == name:
                continue
            dist = self._distance_cache.get((name, inst.name), float("inf"))
            distances.append((inst.name, dist))

        distances.sort(key=lambda x: x[1])
        return distances[:k]
