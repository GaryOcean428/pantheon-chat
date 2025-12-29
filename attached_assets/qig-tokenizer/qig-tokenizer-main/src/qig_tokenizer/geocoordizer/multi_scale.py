"""
MultiScaleCoordizer: Hierarchical Granularity Management
=========================================================

Manages simultaneous representations at multiple scales:
    char → subword → word → phrase → concept

Enables the model to zoom in (novel input) or out (familiar phrases)
dynamically based on context.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .types import BasinCoordinate

if TYPE_CHECKING:
    from .fisher_coordizer import FisherCoordizer


# Scale hierarchy (lowest to highest)
SCALE_ORDER = ["byte", "char", "subword", "word", "phrase", "concept"]


@dataclass
class ScaleNode:
    """Node in multi-scale hierarchy."""

    coord_id: int
    scale: str
    children: list[int] = field(default_factory=list)  # Lower-scale coord_ids
    parent: int | None = None  # Higher-scale coord_id

    @property
    def scale_level(self) -> int:
        return SCALE_ORDER.index(self.scale) if self.scale in SCALE_ORDER else 0


@dataclass
class StructuredCoords:
    """Multi-scale coordization result."""

    levels: dict[str, list[int]]  # scale -> coord_ids at that scale
    hierarchy: dict[int, ScaleNode]  # coord_id -> node info
    primary_scale: str  # Main scale used for output

    def get_coords(self, scale: str | None = None) -> list[int]:
        """Get coord_ids at specified scale (or primary)."""
        scale = scale or self.primary_scale
        return self.levels.get(scale, [])

    def get_coverage(self, coord_id: int) -> tuple[int, int]:
        """Get character span covered by a coordinate."""
        node = self.hierarchy.get(coord_id)
        if not node:
            return (0, 0)

        # Recursively find byte coverage
        if node.scale == "byte":
            return (coord_id, coord_id + 1)

        if not node.children:
            return (0, 0)

        start = float("inf")
        end = 0
        for child_id in node.children:
            child_start, child_end = self.get_coverage(child_id)
            start = min(start, child_start)
            end = max(end, child_end)

        return (int(start), int(end))


class MultiScaleCoordizer:
    """
    Multi-scale coordinate hierarchy management.

    Maintains simultaneous representations at multiple granularity
    levels (char, subword, word, phrase, concept) and enables
    dynamic scale selection.

    Attributes:
        active_scales: Currently enabled scales
        scale_thresholds: Fisher distance thresholds for scale promotion
    """

    def __init__(
        self,
        active_scales: list[str] | None = None,
        promotion_threshold: float = 0.3,
        min_cluster_size: int = 2,
    ):
        self.active_scales = active_scales or ["subword", "word"]
        self.promotion_threshold = promotion_threshold
        self.min_cluster_size = min_cluster_size

        # Scale-specific vocabularies (coord_id -> scale)
        self.coord_scales: dict[int, str] = {}

        # Hierarchy mappings
        self.children_map: dict[int, list[int]] = defaultdict(list)
        self.parent_map: dict[int, int] = {}

        # Known multi-word units at each scale
        self.known_units: dict[str, dict[tuple[int, ...], int]] = {
            scale: {} for scale in SCALE_ORDER
        }

    def segment(
        self,
        text: str,
        levels: list[str],
        coordizer: FisherCoordizer,
    ) -> StructuredCoords:
        """
        Segment text at multiple scales simultaneously.

        Args:
            text: Input text
            levels: Desired scales (e.g., ["char", "word", "concept"])
            coordizer: FisherCoordizer for base coordization

        Returns:
            StructuredCoords with multi-scale hierarchy
        """
        # Get base coordization
        result = coordizer.coordize(text)
        base_coords = result.coord_ids

        # Build hierarchy from base
        hierarchy: dict[int, ScaleNode] = {}
        levels_dict: dict[str, list[int]] = defaultdict(list)

        # Start with base level
        base_scale = "subword"  # Default base from coordizer
        for cid in base_coords:
            coord = coordizer.vocab.get(cid)
            scale = coord.scale if coord else "byte"

            hierarchy[cid] = ScaleNode(
                coord_id=cid,
                scale=scale,
                children=[],
                parent=None,
            )
            levels_dict[scale].append(cid)

        # Try to promote to higher scales if requested
        for target_scale in levels:
            if SCALE_ORDER.index(target_scale) > SCALE_ORDER.index(base_scale):
                promoted = self._try_promote(
                    base_coords, target_scale, coordizer, hierarchy
                )
                if promoted:
                    levels_dict[target_scale].extend(promoted)

        # Determine primary scale (highest with content)
        primary = base_scale
        for scale in reversed(levels):
            if levels_dict.get(scale):
                primary = scale
                break

        return StructuredCoords(
            levels=dict(levels_dict),
            hierarchy=hierarchy,
            primary_scale=primary,
        )

    def _try_promote(
        self,
        coords: list[int],
        target_scale: str,
        coordizer: FisherCoordizer,
        hierarchy: dict[int, ScaleNode],
    ) -> list[int]:
        """Try to promote coordinate sequences to higher scale."""
        promoted = []

        # Check for known units at target scale
        i = 0
        while i < len(coords):
            # Try longest match first
            for length in range(min(5, len(coords) - i), 1, -1):
                seq = tuple(coords[i : i + length])

                if seq in self.known_units.get(target_scale, {}):
                    # Found known unit
                    unit_id = self.known_units[target_scale][seq]
                    promoted.append(unit_id)

                    # Update hierarchy
                    if unit_id not in hierarchy:
                        hierarchy[unit_id] = ScaleNode(
                            coord_id=unit_id,
                            scale=target_scale,
                            children=list(seq),
                            parent=None,
                        )

                    for child_id in seq:
                        if child_id in hierarchy:
                            hierarchy[child_id].parent = unit_id

                    i += length
                    break
            else:
                # No match, keep at current scale
                i += 1

        return promoted

    def promote_to_scale(
        self,
        coord_ids: list[int],
        scale: str,
        coordizer: FisherCoordizer,
        name: str | None = None,
    ) -> int:
        """
        Promote a coordinate sequence to a higher scale.

        Creates a new coordinate representing the sequence at
        the target scale level.

        Args:
            coord_ids: Sequence of coord_ids to promote
            scale: Target scale (e.g., "phrase", "concept")
            coordizer: FisherCoordizer to update
            name: Optional name for new coordinate

        Returns:
            New coordinate ID at target scale
        """
        if len(coord_ids) < self.min_cluster_size:
            raise ValueError(f"Need at least {self.min_cluster_size} coords to promote")

        seq = tuple(coord_ids)

        # Check if already exists
        if seq in self.known_units.get(scale, {}):
            return self.known_units[scale][seq]

        # Create new coordinate via geodesic centroid
        new_id = len(coordizer.vocab)

        # Compute geodesic centroid of components
        vectors = [coordizer.vocab[cid].vector for cid in coord_ids]

        # Iterative geodesic mean
        centroid = vectors[0].copy()
        for v in vectors[1:]:
            norm_c = np.linalg.norm(centroid)
            norm_v = np.linalg.norm(v)

            if norm_c > 1e-10 and norm_v > 1e-10:
                c_unit = centroid / norm_c
                v_unit = v / norm_v
                mid = c_unit + v_unit
                mid = mid / (np.linalg.norm(mid) + 1e-10)
                centroid = mid * ((norm_c + norm_v) / 2)

        # Generate name if not provided
        if name is None:
            component_names = []
            for cid in coord_ids[:3]:  # First 3 components
                coord = coordizer.vocab.get(cid)
                if coord and coord.name:
                    component_names.append(coord.name)
            name = (
                "+".join(component_names) if component_names else f"<{scale}_{new_id}>"
            )

        # Create coordinate
        new_coord = BasinCoordinate(
            coord_id=new_id,
            vector=centroid,
            name=name,
            scale=scale,
        )

        coordizer.vocab[new_id] = new_coord
        coordizer.name_to_id[name] = new_id

        # Register in known units
        if scale not in self.known_units:
            self.known_units[scale] = {}
        self.known_units[scale][seq] = new_id

        # Update hierarchy
        self.coord_scales[new_id] = scale
        self.children_map[new_id] = list(coord_ids)
        for cid in coord_ids:
            self.parent_map[cid] = new_id

        return new_id

    def enable_levels(self, levels: list[str]) -> None:
        """Enable specific scale levels."""
        self.active_scales = [lvl for lvl in levels if lvl in SCALE_ORDER]

    def disable_level(self, level: str) -> None:
        """Disable a specific scale level."""
        if level in self.active_scales:
            self.active_scales.remove(level)

    def get_scale(self, coord_id: int, coordizer: FisherCoordizer) -> str:
        """Get scale of a coordinate."""
        if coord_id in self.coord_scales:
            return self.coord_scales[coord_id]

        coord = coordizer.vocab.get(coord_id)
        return coord.scale if coord else "byte"

    def get_children(self, coord_id: int) -> list[int]:
        """Get child coordinates (lower scale components)."""
        return self.children_map.get(coord_id, [])

    def get_parent(self, coord_id: int) -> int | None:
        """Get parent coordinate (higher scale container)."""
        return self.parent_map.get(coord_id)

    def discover_units(
        self,
        coordizer: FisherCoordizer,
        corpus_coords: list[list[int]],
        target_scale: str,
        min_frequency: int = 5,
        max_length: int = 5,
    ) -> list[tuple[tuple[int, ...], int]]:
        """
        Discover multi-coordinate units from corpus.

        Finds frequently occurring sequences that should be
        promoted to a higher scale.

        Args:
            coordizer: FisherCoordizer instance
            corpus_coords: List of coordized sequences
            target_scale: Scale to promote units to
            min_frequency: Minimum occurrences
            max_length: Maximum sequence length

        Returns:
            List of (sequence, new_coord_id) pairs
        """
        # Count n-gram frequencies
        ngram_counts: dict[tuple[int, ...], int] = defaultdict(int)

        for coords in corpus_coords:
            for n in range(2, max_length + 1):
                for i in range(len(coords) - n + 1):
                    ngram = tuple(coords[i : i + n])
                    ngram_counts[ngram] += 1

        # Filter by frequency and check basin tightness
        candidates = []

        for ngram, count in ngram_counts.items():
            if count < min_frequency:
                continue

            # Check if components are geometrically close
            if self._is_tight_cluster(ngram, coordizer):
                candidates.append((ngram, count))

        # Sort by frequency and promote top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)

        promoted = []
        for ngram, count in candidates[:50]:  # Limit to top 50
            new_id = self.promote_to_scale(list(ngram), target_scale, coordizer)
            promoted.append((ngram, new_id))

        return promoted

    def _is_tight_cluster(
        self,
        coord_ids: tuple[int, ...],
        coordizer: FisherCoordizer,
    ) -> bool:
        """Check if coordinates form a tight cluster (low Fisher variance)."""
        if len(coord_ids) < 2:
            return True

        coords = [coordizer.vocab.get(cid) for cid in coord_ids]
        coords = [c for c in coords if c is not None]

        if len(coords) < 2:
            return False

        # Compute average pairwise Fisher distance
        total_dist = 0.0
        count = 0

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                total_dist += coords[i].fisher_distance(coords[j])
                count += 1

        avg_dist = total_dist / max(count, 1)

        return avg_dist < self.promotion_threshold

    def get_scale_distribution(
        self,
        coords: list[int],
        coordizer: FisherCoordizer,
    ) -> dict[str, int]:
        """Get distribution of scales in a coordinate sequence."""
        distribution: dict[str, int] = defaultdict(int)

        for cid in coords:
            scale = self.get_scale(cid, coordizer)
            distribution[scale] += 1

        return dict(distribution)
