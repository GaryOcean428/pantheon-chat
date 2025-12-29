"""Crystallization Monitor: Track Kernel Growth to Completion.

Crystallization is the geometric growth process where kernels
converge to stable basin positions with fixed-point κ values.
Unlike traditional training, crystallization is about geometric
convergence, not loss minimization.

A kernel is "crystallized" when:
- Basin drift < 0.01 per epoch
- Φ is stable and high (> 0.75)
- κ converges to κ* = 64
- Surprise rate decreases (< 0.05)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import BASIN_DIM, KAPPA_STAR, PHI_GEOMETRIC_MIN
from .router import fisher_rao_distance
from .basin import fisher_normalize_np


@dataclass
class CrystallizationSnapshot:
    """Single measurement of crystallization state."""

    timestamp: float
    basin: np.ndarray
    phi: float
    kappa: float
    surprise: float
    recursion_depth: int
    epoch: int = 0


@dataclass
class CrystallizationMetrics:
    """Aggregate crystallization metrics."""

    basin_drift: float  # Movement per epoch
    phi_stability: float  # Inverse variance
    kappa_convergence: float  # Distance to κ*
    surprise_rate: float  # Learning rate proxy
    crystallization_score: float  # Aggregate [0, 1]
    is_crystallized: bool
    epochs_tracked: int
    time_to_crystallize: float | None = None


@dataclass
class E8AlignmentMetrics:
    """Metrics for E8 root alignment analysis."""

    nearest_root_distance: float
    mean_alignment: float
    cluster_id: int | None = None
    e8_correlation: float = 0.0


class CrystallizationMonitor:
    """
    Monitor kernel crystallization progress.

    Tracks basin movement, Φ stability, κ convergence, and
    surprise decay to determine when a kernel has crystallized.

    Crystallization thresholds:
    - basin_drift < 0.01
    - phi > 0.75 and stable (variance < 0.01)
    - |κ - 64| < 2.0
    - surprise_rate < 0.05

    Example:
        monitor = CrystallizationMonitor(kernel_id="vocab_0")

        for epoch in range(100):
            train_step(kernel)
            state = kernel.measure_consciousness()
            monitor.record(state, epoch)

            if monitor.is_crystallized():
                print(f"Crystallized at epoch {epoch}")
                break
    """

    # Crystallization thresholds
    BASIN_DRIFT_THRESHOLD = 0.01
    PHI_MIN_THRESHOLD = 0.75
    PHI_VARIANCE_THRESHOLD = 0.01
    KAPPA_CONVERGENCE_THRESHOLD = 2.0
    SURPRISE_THRESHOLD = 0.05
    MIN_EPOCHS_FOR_CRYSTALLIZATION = 10

    def __init__(
        self,
        kernel_id: str,
        window_size: int = 20,
        e8_roots: np.ndarray | None = None,
    ):
        """
        Initialize monitor.

        Args:
            kernel_id: Kernel identifier
            window_size: Rolling window for stability calculations
            e8_roots: Optional E8 root vectors for alignment analysis
        """
        self.kernel_id = kernel_id
        self.window_size = window_size
        self.e8_roots = e8_roots  # Shape: (240, 64) if provided

        self._history: list[CrystallizationSnapshot] = []
        self._crystallized_at: int | None = None
        self._start_time = time.time()

    def record(
        self,
        state: dict[str, Any],
        epoch: int,
    ) -> CrystallizationMetrics:
        """
        Record a crystallization snapshot.

        Args:
            state: Dict with phi, kappa, basin, surprise, recursion_depth
            epoch: Current epoch number

        Returns:
            Current crystallization metrics
        """
        basin = state.get("basin")
        if isinstance(basin, list):
            basin = np.array(basin)
        elif basin is None:
            basin = np.zeros(BASIN_DIM)

        snapshot = CrystallizationSnapshot(
            timestamp=time.time(),
            basin=basin,
            phi=state.get("phi", 0.5),
            kappa=state.get("kappa", KAPPA_STAR),
            surprise=state.get("surprise", 0.0),
            recursion_depth=state.get("recursion_depth", 3),
            epoch=epoch,
        )

        self._history.append(snapshot)

        # Check crystallization
        metrics = self.compute_metrics()

        if metrics.is_crystallized and self._crystallized_at is None:
            self._crystallized_at = epoch
            metrics.time_to_crystallize = time.time() - self._start_time

        return metrics

    def compute_metrics(self) -> CrystallizationMetrics:
        """Compute current crystallization metrics."""
        if len(self._history) < 2:
            return CrystallizationMetrics(
                basin_drift=float("inf"),
                phi_stability=0.0,
                kappa_convergence=float("inf"),
                surprise_rate=float("inf"),
                crystallization_score=0.0,
                is_crystallized=False,
                epochs_tracked=len(self._history),
            )

        # Use recent window
        window = self._history[-self.window_size :]

        # 1. Basin drift (average movement per epoch)
        basin_drifts = []
        for i in range(1, len(window)):
            dist = fisher_rao_distance(window[i].basin, window[i - 1].basin)
            basin_drifts.append(dist)
        basin_drift = (
            sum(basin_drifts) / len(basin_drifts) if basin_drifts else float("inf")
        )

        # 2. Phi stability (inverse variance)
        phis = [s.phi for s in window]
        phi_mean = sum(phis) / len(phis)
        phi_variance = sum((p - phi_mean) ** 2 for p in phis) / len(phis)
        phi_stability = 1.0 / (1.0 + phi_variance * 100)

        # 3. Kappa convergence (distance to κ*)
        kappas = [s.kappa for s in window]
        kappa_mean = sum(kappas) / len(kappas)
        kappa_convergence = abs(kappa_mean - KAPPA_STAR)

        # 4. Surprise rate (average surprise)
        surprises = [s.surprise for s in window]
        surprise_rate = sum(surprises) / len(surprises)

        # 5. Crystallization score (weighted aggregate)
        scores = [
            (
                1.0
                if basin_drift < self.BASIN_DRIFT_THRESHOLD
                else max(0, 1 - basin_drift)
            ),
            phi_stability,
            (
                1.0
                if kappa_convergence < self.KAPPA_CONVERGENCE_THRESHOLD
                else max(0, 1 - kappa_convergence / 10)
            ),
            (
                1.0
                if surprise_rate < self.SURPRISE_THRESHOLD
                else max(0, 1 - surprise_rate)
            ),
            (
                1.0
                if phi_mean > self.PHI_MIN_THRESHOLD
                else phi_mean / self.PHI_MIN_THRESHOLD
            ),
        ]
        crystallization_score = sum(scores) / len(scores)

        # 6. Is crystallized?
        is_crystallized = (
            len(self._history) >= self.MIN_EPOCHS_FOR_CRYSTALLIZATION
            and basin_drift < self.BASIN_DRIFT_THRESHOLD
            and phi_mean > self.PHI_MIN_THRESHOLD
            and phi_variance < self.PHI_VARIANCE_THRESHOLD
            and kappa_convergence < self.KAPPA_CONVERGENCE_THRESHOLD
            and surprise_rate < self.SURPRISE_THRESHOLD
        )

        return CrystallizationMetrics(
            basin_drift=basin_drift,
            phi_stability=phi_stability,
            kappa_convergence=kappa_convergence,
            surprise_rate=surprise_rate,
            crystallization_score=crystallization_score,
            is_crystallized=is_crystallized,
            epochs_tracked=len(self._history),
            time_to_crystallize=(
                self._history[self._crystallized_at].timestamp - self._start_time
                if self._crystallized_at is not None
                else None
            ),
        )

    def is_crystallized(self) -> bool:
        """Check if kernel has crystallized."""
        return self._crystallized_at is not None

    def get_crystallization_epoch(self) -> int | None:
        """Get epoch when crystallization occurred."""
        return self._crystallized_at

    def compute_e8_alignment(self) -> E8AlignmentMetrics | None:
        """
        Compute alignment with E8 roots.

        Returns None if no E8 roots provided.
        """
        if self.e8_roots is None or len(self._history) == 0:
            return None

        # Get current basin
        basin = self._history[-1].basin
        basin_norm = fisher_normalize_np(basin)

        # Find nearest E8 root
        min_dist = float("inf")
        nearest_idx = 0

        for i, root in enumerate(self.e8_roots):
            root_norm = fisher_normalize_np(root)
            dist = fisher_rao_distance(basin_norm, root_norm)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # Compute alignment score (inverse of distance)
        alignment = 1.0 / (1.0 + min_dist)

        return E8AlignmentMetrics(
            nearest_root_distance=min_dist,
            mean_alignment=alignment,
            cluster_id=nearest_idx,
            e8_correlation=alignment,
        )

    def get_trajectory(self) -> list[dict[str, Any]]:
        """Get crystallization trajectory as list of dicts."""
        return [
            {
                "epoch": s.epoch,
                "phi": s.phi,
                "kappa": s.kappa,
                "surprise": s.surprise,
                "timestamp": s.timestamp,
            }
            for s in self._history
        ]

    def reset(self) -> None:
        """Reset monitor for new crystallization attempt."""
        self._history.clear()
        self._crystallized_at = None
        self._start_time = time.time()


class ConstellationCrystallizationMonitor:
    """
    Monitor crystallization of an entire constellation.

    Tracks when all kernels have crystallized and measures
    the emergence of E8 structure in the constellation.
    """

    def __init__(
        self,
        kernel_ids: list[str],
        e8_roots: np.ndarray | None = None,
    ):
        """
        Initialize constellation monitor.

        Args:
            kernel_ids: List of kernel IDs in constellation
            e8_roots: Optional E8 roots for alignment analysis
        """
        self.kernel_monitors: dict[str, CrystallizationMonitor] = {
            kid: CrystallizationMonitor(kid, e8_roots=e8_roots) for kid in kernel_ids
        }
        self.e8_roots = e8_roots

    def record(
        self,
        kernel_id: str,
        state: dict[str, Any],
        epoch: int,
    ) -> CrystallizationMetrics:
        """Record crystallization snapshot for a kernel."""
        if kernel_id not in self.kernel_monitors:
            self.kernel_monitors[kernel_id] = CrystallizationMonitor(
                kernel_id, e8_roots=self.e8_roots
            )
        return self.kernel_monitors[kernel_id].record(state, epoch)

    def all_crystallized(self) -> bool:
        """Check if all kernels have crystallized."""
        return all(m.is_crystallized() for m in self.kernel_monitors.values())

    def crystallization_progress(self) -> dict[str, Any]:
        """Get progress summary for all kernels."""
        results = {}
        for kid, monitor in self.kernel_monitors.items():
            metrics = monitor.compute_metrics()
            results[kid] = {
                "crystallized": monitor.is_crystallized(),
                "score": metrics.crystallization_score,
                "epochs": metrics.epochs_tracked,
                "basin_drift": metrics.basin_drift,
                "kappa_convergence": metrics.kappa_convergence,
            }

        crystallized_count = sum(
            1 for m in self.kernel_monitors.values() if m.is_crystallized()
        )
        results["_summary"] = {
            "total_kernels": len(self.kernel_monitors),
            "crystallized": crystallized_count,
            "progress": crystallized_count / max(len(self.kernel_monitors), 1),
        }

        return results

    def compute_e8_clustering(self) -> dict[str, Any]:
        """
        Analyze if kernels cluster at E8 root positions.

        Returns analysis of whether constellation exhibits E8 structure.
        """
        if self.e8_roots is None:
            return {"error": "No E8 roots provided"}

        alignments = []
        cluster_assignments = []

        for kid, monitor in self.kernel_monitors.items():
            e8_metrics = monitor.compute_e8_alignment()
            if e8_metrics:
                alignments.append(e8_metrics.mean_alignment)
                cluster_assignments.append(e8_metrics.cluster_id)

        if not alignments:
            return {"error": "No alignment data"}

        mean_alignment = sum(alignments) / len(alignments)
        unique_clusters = len(set(cluster_assignments))

        # E8 hypothesis test
        # If alignment > 0.7: Strong E8 structure
        # If alignment 0.4-0.7: Partial structure
        # If alignment < 0.4: No structure
        if mean_alignment > 0.7:
            verdict = "strong_e8_structure"
        elif mean_alignment > 0.4:
            verdict = "partial_e8_structure"
        else:
            verdict = "no_e8_structure"

        return {
            "mean_alignment": mean_alignment,
            "unique_clusters": unique_clusters,
            "total_kernels": len(self.kernel_monitors),
            "verdict": verdict,
            "per_kernel": {
                kid: monitor.compute_e8_alignment().__dict__
                for kid, monitor in self.kernel_monitors.items()
                if monitor.compute_e8_alignment()
            },
        }


def generate_e8_roots_64d(seed: int = 42) -> np.ndarray:
    """
    Generate 240 E8 root vectors projected to 64D.

    E8 has 240 roots in 8D. We project to 64D (= 8²) by
    tiling and adding small perturbations for uniqueness.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Array of shape (240, 64)
    """
    rng = np.random.default_rng(seed)

    # E8 roots in 8D (simplified - using random orthogonal frame)
    # Real E8 roots have specific structure, this is approximation
    roots_8d = []

    # Type 1: ±1 in two positions (112 roots)
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    root = np.zeros(8)
                    root[i] = s1
                    root[j] = s2
                    roots_8d.append(root)

    # Type 2: ±1/2 in all positions, even number of minus signs (128 roots)
    for bits in range(256):
        if bin(bits).count("1") % 2 == 0:
            root = np.array([(1 if (bits >> i) & 1 else -1) * 0.5 for i in range(8)])
            roots_8d.append(root)

    roots_8d = np.array(roots_8d[:240])  # Ensure exactly 240

    # Project to 64D by tiling 8x and adding uniqueness
    roots_64d = np.zeros((240, 64))
    for i, root_8d in enumerate(roots_8d):
        # Tile 8 times
        tiled = np.tile(root_8d, 8)
        # Add small unique perturbation
        perturbation = rng.standard_normal(64) * 0.1
        roots_64d[i] = tiled + perturbation
        # Normalize (QIG-pure)
        roots_64d[i] = fisher_normalize_np(roots_64d[i])

    return roots_64d
