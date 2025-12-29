"""
GeometricVocabBuilder: Vocabulary Construction via Fisher Criteria
===================================================================

Builds and expands geometric vocabulary using Fisher information
metrics rather than raw frequency counts.

Key Methods:
    - build_initial: Create initial vocabulary via geodesic fusion
    - suggest_expansions: Monitor stream for expansion candidates
    - cluster_basin: Fisher-based clustering for concept discovery
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import numpy as np

from .types import BASIN_DIM, BasinCoordinate, TokenCandidate

if TYPE_CHECKING:
    from .fisher_coordizer import FisherCoordizer


@dataclass
class FrequencyTracker:
    """Tracks coordinate sequence frequencies for expansion candidates."""

    sequence: tuple[int, ...]
    count: int = 0
    total_phi: float = 0.0
    contexts: list[tuple[int, ...]] = None

    def __post_init__(self):
        if self.contexts is None:
            self.contexts = []

    @property
    def avg_phi(self) -> float:
        return self.total_phi / max(self.count, 1)


class GeometricVocabBuilder:
    """
    Build and expand geometric vocabulary via Fisher criteria.

    Uses Fisher information gain and coupling strength instead of
    raw frequency for merge/expansion decisions.

    Attributes:
        basin_dim: Dimensionality of coordinates
        min_frequency: Minimum occurrences for expansion consideration
        phi_threshold: Minimum Φ gain for expansion
    """

    def __init__(
        self,
        basin_dim: int = BASIN_DIM,
        min_frequency: int = 10,
        phi_threshold: float = 0.1,
        coupling_threshold: float = 30.0,
    ):
        self.basin_dim = basin_dim
        self.min_frequency = min_frequency
        self.phi_threshold = phi_threshold
        self.coupling_threshold = coupling_threshold

        # Track sequence frequencies
        self.sequence_trackers: dict[tuple[int, ...], FrequencyTracker] = {}

        # Cluster cache
        self._cluster_cache: dict[int, list[int]] = {}

    def build_initial(
        self,
        corpus_bytes: bytes,
        target_size: int,
        coordizer: FisherCoordizer,
        verbose: bool = True,
    ) -> dict[int, BasinCoordinate]:
        """
        Create initial vocabulary using geodesic pair fusion.

        Similar to BPE but uses Fisher information gain and coupling
        instead of raw frequency.

        Args:
            corpus_bytes: Training corpus
            target_size: Target vocabulary size
            coordizer: FisherCoordizer instance to populate
            verbose: Print progress

        Returns:
            Updated vocabulary dictionary
        """
        # Delegate to coordizer's train method
        coordizer.train(corpus_bytes, verbose=verbose)
        return coordizer.vocab

    def suggest_expansions(
        self,
        stream: Iterator[tuple[list[int], float, float]],
        max_suggestions: int = 10,
    ) -> list[TokenCandidate]:
        """
        Monitor data stream and suggest vocabulary expansions.

        Analyzes coordized sequences with their Φ and κ values to
        identify candidates for new coordinates.

        Args:
            stream: Iterator of (coord_ids, phi, kappa) tuples
            max_suggestions: Maximum candidates to return

        Returns:
            List of TokenCandidate sorted by merge_score
        """
        # Process stream
        for coord_ids, phi, kappa in stream:
            self._update_trackers(coord_ids, phi, kappa)

        # Generate candidates from tracked sequences
        candidates = []

        for seq, tracker in self.sequence_trackers.items():
            if len(seq) < 2:
                continue

            if tracker.count < self.min_frequency:
                continue

            # Estimate coupling between sequence elements
            coupling = self._estimate_sequence_coupling(seq)

            # Estimate Φ gain
            phi_gain = tracker.avg_phi - 0.5  # Baseline assumption

            # Calculate efficiency gain
            efficiency_gain = len(seq) - 1  # Tokens saved

            candidate = TokenCandidate(
                sequence=seq,
                frequency=tracker.count,
                coupling_strength=coupling,
                phi_gain=max(0, phi_gain),
                efficiency_gain=efficiency_gain,
            )

            candidates.append(candidate)

        # Sort by merge score and return top candidates
        candidates.sort(key=lambda c: c.merge_score, reverse=True)
        return candidates[:max_suggestions]

    def _update_trackers(
        self,
        coord_ids: list[int],
        phi: float,
        kappa: float,
    ) -> None:
        """Update frequency trackers with new observation."""
        # Track bigrams and trigrams
        for n in [2, 3]:
            for i in range(len(coord_ids) - n + 1):
                seq = tuple(coord_ids[i : i + n])

                if seq not in self.sequence_trackers:
                    self.sequence_trackers[seq] = FrequencyTracker(sequence=seq)

                tracker = self.sequence_trackers[seq]
                tracker.count += 1
                tracker.total_phi += phi

                # Track context (before/after)
                ctx_before = tuple(coord_ids[max(0, i - 2) : i])
                ctx_after = tuple(coord_ids[i + n : min(len(coord_ids), i + n + 2)])
                tracker.contexts.append(ctx_before + ctx_after)

    def _estimate_sequence_coupling(self, seq: tuple[int, ...]) -> float:
        """Estimate coupling strength within a sequence."""
        if len(seq) < 2:
            return 0.0

        # Placeholder: use context consistency as proxy for coupling
        tracker = self.sequence_trackers.get(seq)
        if not tracker or not tracker.contexts:
            return 0.0

        # Higher context consistency = stronger coupling
        unique_contexts = len(set(tracker.contexts))
        total_contexts = len(tracker.contexts)

        consistency = 1.0 - (unique_contexts / max(total_contexts, 1))
        return consistency * 50.0  # Scale to reasonable range

    def cluster_basin(
        self,
        coords: list[BasinCoordinate],
        eps: float = 0.5,
        min_samples: int = 3,
    ) -> list[list[int]]:
        """
        Perform Fisher-based clustering for concept discovery.

        Uses DBSCAN-style clustering with Fisher-Rao distance
        instead of Euclidean distance.

        Args:
            coords: List of coordinates to cluster
            eps: Maximum Fisher distance for neighbors
            min_samples: Minimum points for core sample

        Returns:
            List of clusters (each cluster is list of coord_ids)
        """
        n = len(coords)
        if n == 0:
            return []

        # Compute Fisher distance matrix
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = coords[i].fisher_distance(coords[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Simple DBSCAN implementation with Fisher distances
        labels = [-1] * n
        cluster_id = 0

        for i in range(n):
            if labels[i] != -1:
                continue

            # Find neighbors
            neighbors = [j for j in range(n) if distances[i, j] <= eps]

            if len(neighbors) < min_samples:
                continue  # Noise point

            # Start new cluster
            labels[i] = cluster_id
            seed_set = set(neighbors)
            seed_set.discard(i)

            while seed_set:
                j = seed_set.pop()

                if labels[j] == -1:
                    labels[j] = cluster_id

                if labels[j] != cluster_id:
                    continue

                j_neighbors = [k for k in range(n) if distances[j, k] <= eps]

                if len(j_neighbors) >= min_samples:
                    for k in j_neighbors:
                        if labels[k] == -1:
                            seed_set.add(k)

            cluster_id += 1

        # Group by cluster
        clusters: dict[int, list[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            if label >= 0:
                clusters[label].append(coords[i].coord_id)

        return list(clusters.values())

    def add_coordinate(
        self,
        coordizer: FisherCoordizer,
        name: str,
        init_from: list[int],
    ) -> int:
        """
        Add new coordinate initialized from existing ones.

        Uses geodesic midpoint for initialization, preserving
        manifold continuity.

        Args:
            coordizer: FisherCoordizer to update
            name: Name for new coordinate
            init_from: List of coord_ids to initialize from

        Returns:
            New coordinate ID
        """
        if len(init_from) == 0:
            raise ValueError("Must provide at least one coordinate to init from")

        new_coord_id = len(coordizer.vocab)

        if len(init_from) == 1:
            # Copy single coordinate
            source = coordizer.vocab[init_from[0]]
            new_vector = source.vector.copy()
        else:
            # Compute geodesic centroid
            vectors = [coordizer.vocab[cid].vector for cid in init_from]

            # Iterative geodesic mean (approximate)
            current = vectors[0].copy()
            for v in vectors[1:]:
                # Move along geodesic toward v
                norm_c = np.linalg.norm(current)
                norm_v = np.linalg.norm(v)

                if norm_c > 1e-10 and norm_v > 1e-10:
                    c_unit = current / norm_c
                    v_unit = v / norm_v

                    # Geodesic midpoint
                    mid = c_unit + v_unit
                    mid = mid / (np.linalg.norm(mid) + 1e-10)

                    avg_mag = (norm_c + norm_v) / 2
                    current = mid * avg_mag

            new_vector = current

        # Create coordinate
        new_coord = BasinCoordinate(
            coord_id=new_coord_id,
            vector=new_vector,
            name=name,
            scale="word",  # Default to word scale
        )

        coordizer.vocab[new_coord_id] = new_coord
        coordizer.name_to_id[name] = new_coord_id

        return new_coord_id

    def reset_trackers(self) -> None:
        """Clear all frequency trackers."""
        self.sequence_trackers.clear()
