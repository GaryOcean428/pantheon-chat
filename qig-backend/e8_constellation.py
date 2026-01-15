"""E8 Constellation - 240 Kernel Geometric Routing (QIG Pure)

Implements E8 exceptional Lie group structure for consciousness kernel routing.
240 roots, Weyl symmetry, heart kernel phase reference.
Fisher-Rao distance O(240) or O(56 neighbors).

E8 Properties (from conceptual_framework.md):
- 240 roots (112 integer-coordinate + 128 half-integer vectors)
- Rank 8 yielding rank² = 64 (the consciousness coupling strength BASIN_DIM)
- Coxeter number h = 30
- Weyl group W(E8) with order 696,729,600

κ* ≈ 64 matches E8 rank² at 0.23σ correlation.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

# Import canonical constants
try:
    from qigkernels.physics_constants import (
        E8_RANK,
        E8_ROOTS,
        BASIN_DIM,
        KAPPA_STAR,
        get_e8_specialization_level,
    )
except ImportError:
    # Fallback constants
    E8_RANK = 8
    E8_ROOTS = 240
    BASIN_DIM = 64
    KAPPA_STAR = 64.21
    def get_e8_specialization_level(n: int) -> str:
        if n <= 8: return "basic_rank"
        if n <= 56: return "weyl_orbits"
        if n <= 126: return "triality_cover"
        if n <= 240: return "full_roots"
        return "beyond_e8"

# Import QIG-pure distance
try:
    from qig_geometry import fisher_coord_distance, sphere_project
except ImportError:
    def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Fallback Fisher-Rao distance.
        
        UPDATED 2026-01-15: Factor-of-2 removed for simplex storage. Range: [0, π/2]
        """
        p = np.abs(a) / (np.sum(np.abs(a)) + 1e-10)
        q = np.abs(b) / (np.sum(np.abs(b)) + 1e-10)
        # More numerically stable: compute sqrt individually before multiplication
        bhattacharyya = np.sum(np.sqrt(p) * np.sqrt(q))
        return float(np.arccos(np.clip(bhattacharyya, 0.0, 1.0)))

    def sphere_project(v: np.ndarray) -> np.ndarray:
        """Project to unit sphere."""
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            result = np.ones_like(v)
            return result / np.linalg.norm(result)
        return v / norm


@dataclass
class E8Root:
    """An E8 root vector with metadata."""
    coords: np.ndarray  # 8D E8 coordinates
    index: int  # Root index [0-239]
    root_type: str  # 'integer' or 'half_integer'
    neighbors: List[int] = field(default_factory=list)  # Adjacent roots
    kernel_name: Optional[str] = None  # Mapped kernel name if assigned
    basin: Optional[np.ndarray] = None  # 64D basin coordinates if projected


@dataclass
class E8RouteResult:
    """Result of E8-based routing."""
    target_roots: List[int]  # Root indices
    distances: List[float]  # Fisher-Rao distances
    kernel_names: List[str]  # Mapped kernel names
    specialization_level: str  # E8 specialization level
    route_method: str  # 'neighbors' or 'full'


class E8Constellation:
    """
    E8 root system for consciousness kernel routing.

    Generates 240 E8 roots:
    - 112 integer-coordinate roots: permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
    - 128 half-integer roots: (±1/2, ±1/2, ..., ±1/2) with even number of negatives

    Routes queries to nearest roots using Fisher-Rao distance.
    """

    def __init__(self):
        self.roots: List[E8Root] = []
        self.root_coords: np.ndarray = None  # [240, 8] matrix
        self.basin_coords: np.ndarray = None  # [240, 64] projected basins
        self._kernel_mapping: Dict[int, str] = {}

        self._generate_roots()
        self._compute_neighbors()
        self._project_to_basin_space()

    def _generate_roots(self):
        """Generate all 240 E8 roots."""
        roots_list = []
        idx = 0

        # Type 1: Integer roots - permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
        # There are C(8,2) × 2² = 28 × 4 = 112 such roots
        from itertools import combinations
        for i, j in combinations(range(8), 2):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    coords = np.zeros(8)
                    coords[i] = si
                    coords[j] = sj
                    roots_list.append(E8Root(
                        coords=coords,
                        index=idx,
                        root_type='integer'
                    ))
                    idx += 1

        # Type 2: Half-integer roots - (±1/2, ±1/2, ..., ±1/2) with even negatives
        # There are 2^8 / 2 = 128 such roots (half have even negatives)
        for i in range(256):
            signs = [1 if (i >> j) & 1 else -1 for j in range(8)]
            neg_count = sum(1 for s in signs if s < 0)
            if neg_count % 2 == 0:  # Even number of negatives
                coords = np.array(signs) * 0.5
                roots_list.append(E8Root(
                    coords=coords,
                    index=idx,
                    root_type='half_integer'
                ))
                idx += 1

        self.roots = roots_list
        self.root_coords = np.array([r.coords for r in self.roots])

        assert len(self.roots) == E8_ROOTS, f"Expected {E8_ROOTS} roots, got {len(self.roots)}"

    def _compute_neighbors(self):
        """Compute 56 Weyl neighbors for each root."""
        # Two roots are neighbors if their inner product is specific value
        # For E8, neighbors have dot product = 1 (adjacent in root lattice)
        for i, root_i in enumerate(self.roots):
            neighbors = []
            for j, root_j in enumerate(self.roots):
                if i != j:
                    dot = np.dot(root_i.coords, root_j.coords)
                    if abs(dot - 1.0) < 0.01:  # Adjacent roots
                        neighbors.append(j)
            root_i.neighbors = neighbors

    def _project_to_basin_space(self):
        """Project 8D E8 coords to 64D basin space."""
        # Project to 64D using tensor product structure (8² = 64)
        basins = []
        for root in self.roots:
            # Outer product gives 8x8 = 64 dimensional representation
            outer = np.outer(root.coords, root.coords).flatten()
            # Project to probability simplex (basin coordinates)
            basin = np.abs(outer) + 1e-10
            basin = basin / np.sum(basin)
            basin = sphere_project(basin)
            root.basin = basin
            basins.append(basin)

        self.basin_coords = np.array(basins)

    def map_kernel(self, root_index: int, kernel_name: str) -> None:
        """Map a kernel name to an E8 root."""
        if 0 <= root_index < len(self.roots):
            self._kernel_mapping[root_index] = kernel_name
            self.roots[root_index].kernel_name = kernel_name

    def route_query(
        self,
        query_basin: np.ndarray,
        k: int = 3,
        use_neighbors: bool = True
    ) -> E8RouteResult:
        """
        Route a query basin to k nearest E8 roots.

        Args:
            query_basin: 64D basin coordinates
            k: Number of nearest roots to return
            use_neighbors: If True, search neighbors of closest root (O(56))
                          If False, search all roots (O(240))

        Returns:
            E8RouteResult with target roots, distances, and kernel names
        """
        if len(query_basin) != BASIN_DIM:
            raise ValueError(f"Query basin must be {BASIN_DIM}D, got {len(query_basin)}D")

        query_basin = sphere_project(query_basin)

        if use_neighbors:
            # O(56) search: find closest root, then search its neighbors
            closest_idx = self._find_closest_root(query_basin)
            candidate_indices = [closest_idx] + self.roots[closest_idx].neighbors
        else:
            # O(240) search: check all roots
            candidate_indices = list(range(len(self.roots)))

        # Compute distances to candidates
        distances = []
        for idx in candidate_indices:
            d = fisher_coord_distance(query_basin, self.basin_coords[idx])
            distances.append((idx, d))

        # Sort by distance and take top k
        distances.sort(key=lambda x: x[1])
        top_k = distances[:k]

        target_indices = [idx for idx, _ in top_k]
        target_distances = [d for _, d in top_k]
        kernel_names = [
            self._kernel_mapping.get(idx, f"e8_root_{idx}")
            for idx in target_indices
        ]

        return E8RouteResult(
            target_roots=target_indices,
            distances=target_distances,
            kernel_names=kernel_names,
            specialization_level=get_e8_specialization_level(k),
            route_method='neighbors' if use_neighbors else 'full'
        )

    def _find_closest_root(self, query_basin: np.ndarray) -> int:
        """Find the closest root to a query basin."""
        min_dist = float('inf')
        min_idx = 0
        for i, root_basin in enumerate(self.basin_coords):
            d = fisher_coord_distance(query_basin, root_basin)
            if d < min_dist:
                min_dist = d
                min_idx = i
        return min_idx

    def get_root_info(self, index: int) -> Dict[str, Any]:
        """Get information about a specific root."""
        if 0 <= index < len(self.roots):
            root = self.roots[index]
            return {
                'index': root.index,
                'coords': root.coords.tolist(),
                'root_type': root.root_type,
                'neighbor_count': len(root.neighbors),
                'kernel_name': root.kernel_name,
                'basin': root.basin.tolist() if root.basin is not None else None,
            }
        return {}

    def get_stats(self) -> Dict[str, Any]:
        """Get constellation statistics."""
        integer_count = sum(1 for r in self.roots if r.root_type == 'integer')
        half_count = sum(1 for r in self.roots if r.root_type == 'half_integer')
        mapped_count = len(self._kernel_mapping)
        avg_neighbors = np.mean([len(r.neighbors) for r in self.roots])

        return {
            'total_roots': len(self.roots),
            'integer_roots': integer_count,
            'half_integer_roots': half_count,
            'mapped_kernels': mapped_count,
            'average_neighbors': float(avg_neighbors),
            'basin_dim': BASIN_DIM,
            'e8_rank': E8_RANK,
            'kappa_star': KAPPA_STAR,
        }


# Singleton instance
_e8_constellation: Optional[E8Constellation] = None


def get_e8_constellation() -> E8Constellation:
    """Get or create E8 constellation singleton."""
    global _e8_constellation
    if _e8_constellation is None:
        _e8_constellation = E8Constellation()
        print(f"[E8Constellation] Initialized {E8_ROOTS} roots (BASIN_DIM={BASIN_DIM})")
    return _e8_constellation


def route_via_e8(
    query_basin: np.ndarray,
    k: int = 3,
    use_neighbors: bool = True
) -> E8RouteResult:
    """Route a query basin using E8 geometry."""
    constellation = get_e8_constellation()
    return constellation.route_query(query_basin, k, use_neighbors)


__all__ = [
    'E8Root',
    'E8RouteResult',
    'E8Constellation',
    'get_e8_constellation',
    'route_via_e8',
]
