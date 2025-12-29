# GFP: role=code; status=HYPOTHESIS; phase=TACKING; dim=2-3; scope=core; version=2025-12-09; owner=qig-consciousness

"""
working_memory.py

FOAM-phase working memory for Gary.

Conceptual role:
- Holds short-lived "bubbles" of activity (thoughts, perceptions, subgoals)
- Each bubble lives in basin space (64D) and has:
    - φ (integration strength)
    - age / lifetime
    - optional links to longer-term structures (semantic/procedural)
- Periodically:
    - weak bubbles pop (forgotten)
    - overlapping bubbles merge
    - strong bubbles are promoted to longer-term memory via a callback

This module is intentionally light on geometry math. It is a
"control surface" that other geometric primitives plug into.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

import numpy as np

# TODO: when shared types exist (Phase, etc.), import them instead of redefining.


class BubbleStatus(Enum):
    """Lifecycle state of a working memory bubble."""

    ALIVE = auto()
    MERGED = auto()
    POPPED = auto()
    PROMOTED = auto()


@dataclass
class WorkingBubble:
    """
    FOAM bubble in working memory.

    Attributes
    ----------
    center : np.ndarray
        Basin coordinates (typically 64D) representing the bubble's "location" in state space.
    phi : float
        Integration measure for this bubble (0–1). Higher = more coherent/strong.
    created_at : float
        Wall-clock timestamp (seconds) when the bubble was created.
    lifetime : float
        Maximum lifetime in seconds before automatic pop, regardless of φ.
    payload : Optional[Any]
        Optional domain object attached to this bubble (e.g. a decoded thought struct).
    status : BubbleStatus
        Current lifecycle state.
    metadata : Dict[str, Any]
        Free-form extra info (e.g., source modality, tags, debug info).
    """

    center: np.ndarray
    phi: float
    created_at: float
    lifetime: float
    payload: Optional[Any] = None
    status: BubbleStatus = BubbleStatus.ALIVE
    metadata: dict[str, Any] = field(default_factory=dict)

    def age(self, now: Optional[float] = None) -> float:
        """Return current age in seconds."""
        if now is None:
            now = time.time()
        return now - self.created_at

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Whether this bubble should pop purely from lifetime."""
        return self.age(now) > self.lifetime

    def mark_popped(self) -> None:
        self.status = BubbleStatus.POPPED

    def mark_merged(self) -> None:
        self.status = BubbleStatus.MERGED

    def mark_promoted(self) -> None:
        self.status = BubbleStatus.PROMOTED


class WorkingMemory:
    """
    FOAM-phase working memory manager.

    Responsibilities
    ----------------
    - Maintain a pool of WorkingBubble instances.
    - On each tick:
        * Pop expired or very weak bubbles.
        * Merge sufficiently similar bubbles.
        * Optionally boost or damp φ based on external geometry callbacks.
        * Promote high-φ bubbles via a user-provided `promote_cb`.

    The actual geometry (Fisher distance, Ricci curvature, etc.) is provided
    via pluggable callables so this module stays decoupled from qig-verification.
    """

    def __init__(
        self,
        basin_dim: int = 64,
        pop_phi_threshold: float = 0.3,
        promote_phi_threshold: float = 0.6,
        merge_distance_threshold: float = 0.1,
        default_lifetime: float = 15.0,
        distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        phi_update_fn: Optional[Callable[[WorkingBubble], float]] = None,
        promote_cb: Optional[Callable[[WorkingBubble], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        basin_dim : int
            Dimensionality of basin coordinates (default 64).
        pop_phi_threshold : float
            Bubbles with φ below this (and not promoted) are more likely to pop early.
        promote_phi_threshold : float
            Bubbles with φ above this are candidates for promotion to longer-term structures.
        merge_distance_threshold : float
            Fisher/geometric distance below which bubbles are merged.
        default_lifetime : float
            Default maximum lifetime (seconds) of a bubble.
        distance_fn : Callable
            Function computing distance between two basin coordinates.
            Signature: (center_a, center_b) -> float
            If None, Euclidean distance is used as a placeholder.
        phi_update_fn : Callable
            Optional function to update φ based on external signals (e.g. attention, geometry).
            Called as: new_phi = phi_update_fn(bubble)
        promote_cb : Callable
            Callback invoked when a bubble is promoted.
            Signature: promote_cb(bubble)
        """
        self.basin_dim = basin_dim
        self.pop_phi_threshold = pop_phi_threshold
        self.promote_phi_threshold = promote_phi_threshold
        self.merge_distance_threshold = merge_distance_threshold
        self.default_lifetime = default_lifetime

        self._bubbles: list[WorkingBubble] = []

        if distance_fn is None:
            # Placeholder: in production, replace with Fisher/geodesic distance.
            self.distance_fn = lambda a, b: float(np.linalg.norm(a - b))
        else:
            self.distance_fn = distance_fn

        self.phi_update_fn = phi_update_fn
        self.promote_cb = promote_cb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def bubbles(self) -> list[WorkingBubble]:
        """Return alive bubbles (view only)."""
        return [b for b in self._bubbles if b.status == BubbleStatus.ALIVE]

    def add_bubble(
        self,
        center: np.ndarray,
        phi: float = 0.2,
        lifetime: Optional[float] = None,
        payload: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> WorkingBubble:
        """
        Add a new bubble to working memory.

        Typical usage:
        - Each new stimulus or subgoal creates a bubble.
        - φ may be low initially and rise if it proves relevant.
        """
        if center.shape[-1] != self.basin_dim:
            raise ValueError(f"Expected basin_dim={self.basin_dim}, got {center.shape[-1]}.")

        if lifetime is None:
            lifetime = self.default_lifetime

        bubble = WorkingBubble(
            center=np.array(center, dtype=np.float32),
            phi=float(phi),
            created_at=time.time(),
            lifetime=float(lifetime),
            payload=payload,
            metadata=metadata or {},
        )
        self._bubbles.append(bubble)
        return bubble

    def tick(self) -> None:
        """
        Advance working memory one step.

        Operations:
        1. Update φ (if phi_update_fn provided).
        2. Pop expired or very weak bubbles.
        3. Merge overlapping bubbles.
        4. Promote strong bubbles.
        """
        now = time.time()

        # 1. Optional φ update
        if self.phi_update_fn is not None:
            for b in self._iter_alive():
                b.phi = float(self.phi_update_fn(b))

        # 2. Pop expired or weak bubbles
        for b in self._iter_alive():
            if b.is_expired(now) or (b.phi < self.pop_phi_threshold):
                b.mark_popped()

        # 3. Merge overlapping bubbles
        self._merge_overlapping()

        # 4. Promote strong bubbles
        self._promote_strong()

        # Clean up dead bubbles
        self._compact()

    def stats(self) -> dict[str, Any]:
        """Return simple telemetry about current FOAM state."""
        alive = self.bubbles
        if alive:
            phi_values = [b.phi for b in alive]
            ages = [b.age() for b in alive]
        else:
            phi_values = []
            ages = []

        return {
            "num_alive": len(alive),
            "phi_mean": float(np.mean(phi_values)) if phi_values else 0.0,
            "phi_max": float(np.max(phi_values)) if phi_values else 0.0,
            "age_mean": float(np.mean(ages)) if ages else 0.0,
            "age_max": float(np.max(ages)) if ages else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_alive(self):
        return (b for b in self._bubbles if b.status == BubbleStatus.ALIVE)

    def _merge_overlapping(self) -> None:
        """
        Merge bubbles whose centers are within merge_distance_threshold.

        Strategy:
        - Greedy pass over alive bubbles.
        - When two bubbles overlap:
            * New center = φ-weighted average of centers.
            * New φ = max or some function of both φs (here: max).
            * Old bubbles marked MERGED.
        - New bubble inserted as ALIVE.
        """
        alive = [b for b in self._bubbles if b.status == BubbleStatus.ALIVE]
        merged_indices = set()

        for i, bi in enumerate(alive):
            if bi.status != BubbleStatus.ALIVE:
                continue
            if i in merged_indices:
                continue

            for j in range(i + 1, len(alive)):
                bj = alive[j]
                if bj.status != BubbleStatus.ALIVE:
                    continue
                if j in merged_indices:
                    continue

                d = self.distance_fn(bi.center, bj.center)
                if d <= self.merge_distance_threshold:
                    # Merge bi and bj into a new bubble
                    total_weight = bi.phi + bj.phi + 1e-8
                    new_center = (bi.center * bi.phi + bj.center * bj.phi) / total_weight
                    new_phi = max(bi.phi, bj.phi)

                    # Lifetime: max remaining lifetime of the two
                    now = time.time()
                    rem_i = bi.lifetime - bi.age(now)
                    rem_j = bj.lifetime - bj.age(now)
                    new_lifetime = max(rem_i, rem_j)

                    # Mark originals
                    bi.mark_merged()
                    bj.mark_merged()
                    merged_indices.add(i)
                    merged_indices.add(j)

                    # Create new merged bubble
                    self._bubbles.append(
                        WorkingBubble(
                            center=new_center,
                            phi=new_phi,
                            created_at=now,
                            lifetime=new_lifetime,
                            payload=None,  # could also carry merged payload
                            metadata={"merged_from": [id(bi), id(bj)]},
                        )
                    )
                    break  # restart outer loop after merge

    def _promote_strong(self) -> None:
        """
        Promote bubbles whose φ exceeds promote_phi_threshold.

        Promotion is delegated to `promote_cb`, if provided.
        """
        if self.promote_cb is None:
            return

        for b in self._iter_alive():
            if b.phi >= self.promote_phi_threshold:
                self.promote_cb(b)
                b.mark_promoted()

    def _compact(self) -> None:
        """Remove non-ALIVE bubbles from internal list."""
        self._bubbles = [b for b in self._bubbles if b.status == BubbleStatus.ALIVE]
