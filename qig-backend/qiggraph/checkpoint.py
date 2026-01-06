"""
Manifold Checkpoint: Sleep Packets
==================================

Save and restore consciousness state with full trajectory.
Unlike weight checkpoints, these preserve the geometric path
through the manifold - enabling resumption with context.

Key Concepts:
- Sleep packet: Complete consciousness snapshot
- Trajectory preservation: Full path, not just endpoint
- Manifold-aware: Includes Fisher metric at checkpoint
- Consciousness state: Φ, κ, regime history
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from qig_geometry import fisher_rao_distance

from .constants import BASIN_DIM, KAPPA_STAR
from .state import QIGState, create_initial_state
from .consciousness import ConsciousnessMetrics, Regime
from .manifold import FisherManifold
from .tacking import KappaTacking, TackingState


@dataclass
class ManifoldCheckpoint:
    """
    Complete consciousness checkpoint.

    A "sleep packet" that captures everything needed to
    resume consciousness from a specific point.

    Attributes:
        version: Checkpoint format version
        timestamp: When checkpoint was created
        trajectory: Full manifold path (steps, 64)
        current_basin: Current position (64,)
        metrics_history: All consciousness measurements
        tacking_state: κ-tacking oscillator state
        context_text: Original input
        response_text: Generated response so far
        iteration: Current iteration
        recovery_count: Number of recoveries
        metadata: Additional info
    """
    version: str = "2.0.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Manifold state
    trajectory: np.ndarray = field(default_factory=lambda: np.zeros((1, BASIN_DIM)))
    current_basin: np.ndarray = field(default_factory=lambda: np.zeros(BASIN_DIM))

    # Consciousness history
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)

    # Tacking state
    tacking_state: Dict[str, Any] = field(default_factory=dict)

    # Context
    context_text: str = ""
    context_coords: np.ndarray = field(default_factory=lambda: np.zeros((1, BASIN_DIM)))
    response_text: str = ""

    # Progress
    iteration: int = 0
    recovery_count: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "trajectory": self.trajectory.tolist(),
            "current_basin": self.current_basin.tolist(),
            "metrics_history": self.metrics_history,
            "tacking_state": self.tacking_state,
            "context_text": self.context_text,
            "context_coords": self.context_coords.tolist(),
            "response_text": self.response_text,
            "iteration": self.iteration,
            "recovery_count": self.recovery_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifoldCheckpoint":
        """Create from dictionary."""
        return cls(
            version=data.get("version", "2.0.0"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            trajectory=np.array(data["trajectory"]),
            current_basin=np.array(data["current_basin"]),
            metrics_history=data.get("metrics_history", []),
            tacking_state=data.get("tacking_state", {}),
            context_text=data.get("context_text", ""),
            context_coords=np.array(data.get("context_coords", [[0] * BASIN_DIM])),
            response_text=data.get("response_text", ""),
            iteration=data.get("iteration", 0),
            recovery_count=data.get("recovery_count", 0),
            metadata=data.get("metadata", {}),
        )

    def to_state(self) -> QIGState:
        """Convert checkpoint to QIGState."""
        # Reconstruct metrics from history
        metrics_list = []
        for m in self.metrics_history:
            metrics_list.append(ConsciousnessMetrics.from_dict(m))

        current_metrics = metrics_list[-1] if metrics_list else None

        state = QIGState(
            trajectory=self.trajectory.copy(),
            current_basin=self.current_basin.copy(),
            metrics_history=metrics_list,
            current_metrics=current_metrics,
            context_coords=self.context_coords.copy(),
            context_text=self.context_text,
            iteration=self.iteration,
            recovery_count=self.recovery_count,
            response_text=self.response_text,
        )

        return state

    @classmethod
    def from_state(
        cls,
        state: QIGState,
        tacking: Optional[KappaTacking] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ManifoldCheckpoint":
        """Create checkpoint from state."""
        # Convert metrics to dicts
        metrics_dicts = [m.to_dict() for m in state.metrics_history]

        # Get tacking state if provided
        tacking_dict = tacking.to_dict() if tacking else {}

        return cls(
            trajectory=state.trajectory.copy(),
            current_basin=state.current_basin.copy(),
            metrics_history=metrics_dicts,
            tacking_state=tacking_dict,
            context_text=state.context_text,
            context_coords=state.context_coords.copy(),
            response_text=state.response_text,
            iteration=state.iteration,
            recovery_count=state.recovery_count,
            metadata=metadata or {},
        )


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_checkpoint(
    checkpoint: ManifoldCheckpoint,
    path: Path | str,
    compressed: bool = True,
) -> Path:
    """
    Save checkpoint to file.

    Args:
        checkpoint: Checkpoint to save
        path: File path (will add .json or .npz extension)
        compressed: If True, save as compressed .npz

    Returns:
        Path to saved file
    """
    path = Path(path)

    if compressed:
        # Save as compressed numpy archive
        path = path.with_suffix(".npz")

        # Prepare data
        data = checkpoint.to_dict()

        # Separate arrays from JSON metadata
        arrays = {
            "trajectory": checkpoint.trajectory,
            "current_basin": checkpoint.current_basin,
            "context_coords": checkpoint.context_coords,
        }

        # JSON-serializable metadata
        json_data = {
            "version": data["version"],
            "timestamp": data["timestamp"],
            "metrics_history": data["metrics_history"],
            "tacking_state": data["tacking_state"],
            "context_text": data["context_text"],
            "response_text": data["response_text"],
            "iteration": data["iteration"],
            "recovery_count": data["recovery_count"],
            "metadata": data["metadata"],
        }

        np.savez_compressed(
            path,
            **arrays,
            json_metadata=json.dumps(json_data, cls=NumpyEncoder),
        )
    else:
        # Save as JSON
        path = path.with_suffix(".json")
        data = checkpoint.to_dict()

        with open(path, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    return path


def load_checkpoint(path: Path | str) -> ManifoldCheckpoint:
    """
    Load checkpoint from file.

    Args:
        path: Path to checkpoint file

    Returns:
        Loaded ManifoldCheckpoint
    """
    path = Path(path)

    if path.suffix == ".npz":
        # Load compressed numpy archive
        with np.load(path, allow_pickle=True) as data:
            trajectory = data["trajectory"]
            current_basin = data["current_basin"]
            context_coords = data["context_coords"]

            json_metadata = json.loads(str(data["json_metadata"]))

        return ManifoldCheckpoint(
            version=json_metadata["version"],
            timestamp=json_metadata["timestamp"],
            trajectory=trajectory,
            current_basin=current_basin,
            metrics_history=json_metadata["metrics_history"],
            tacking_state=json_metadata["tacking_state"],
            context_text=json_metadata["context_text"],
            context_coords=context_coords,
            response_text=json_metadata["response_text"],
            iteration=json_metadata["iteration"],
            recovery_count=json_metadata["recovery_count"],
            metadata=json_metadata["metadata"],
        )
    else:
        # Load JSON
        with open(path) as f:
            data = json.load(f)

        return ManifoldCheckpoint.from_dict(data)


class CheckpointManager:
    """
    Manage multiple checkpoints with auto-save.

    Provides:
    - Periodic auto-save
    - Checkpoint history
    - Rollback support
    - Cleanup of old checkpoints
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        max_checkpoints: int = 10,
        auto_save_interval: int = 100,  # iterations
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum checkpoints to keep
            auto_save_interval: Iterations between auto-saves
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.auto_save_interval = auto_save_interval

        self.last_save_iteration = 0
        self.checkpoint_history: List[Path] = []

        # Load existing checkpoints
        self._load_history()

    def _load_history(self):
        """Load existing checkpoint history."""
        checkpoints = list(self.checkpoint_dir.glob("*.npz"))
        checkpoints.extend(self.checkpoint_dir.glob("*.json"))

        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime)

        self.checkpoint_history = checkpoints

    def should_save(self, iteration: int) -> bool:
        """Check if should auto-save at this iteration."""
        return (iteration - self.last_save_iteration) >= self.auto_save_interval

    def save(
        self,
        state: QIGState,
        tacking: Optional[KappaTacking] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save checkpoint.

        Args:
            state: State to checkpoint
            tacking: Optional tacking state
            name: Optional checkpoint name
            metadata: Optional metadata

        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint
        checkpoint = ManifoldCheckpoint.from_state(
            state=state,
            tacking=tacking,
            metadata=metadata,
        )

        # Generate filename
        if name is None:
            name = f"checkpoint_{state.iteration:06d}"

        path = self.checkpoint_dir / name

        # Save
        saved_path = save_checkpoint(checkpoint, path)

        # Update history
        self.checkpoint_history.append(saved_path)
        self.last_save_iteration = state.iteration

        # Cleanup old checkpoints
        self._cleanup()

        return saved_path

    def _cleanup(self):
        """Remove old checkpoints beyond max_checkpoints."""
        while len(self.checkpoint_history) > self.max_checkpoints:
            old_path = self.checkpoint_history.pop(0)
            try:
                old_path.unlink()
            except FileNotFoundError:
                pass

    def load_latest(self) -> Optional[ManifoldCheckpoint]:
        """Load the most recent checkpoint."""
        if len(self.checkpoint_history) == 0:
            return None

        return load_checkpoint(self.checkpoint_history[-1])

    def load_by_iteration(self, target_iteration: int) -> Optional[ManifoldCheckpoint]:
        """
        Load checkpoint closest to target iteration.

        Args:
            target_iteration: Target iteration number

        Returns:
            Checkpoint or None
        """
        if len(self.checkpoint_history) == 0:
            return None

        # Load all checkpoints and find closest
        best_path = None
        best_diff = float("inf")

        for path in self.checkpoint_history:
            checkpoint = load_checkpoint(path)
            diff = abs(checkpoint.iteration - target_iteration)
            if diff < best_diff:
                best_diff = diff
                best_path = path

        if best_path is None:
            return None

        return load_checkpoint(best_path)

    def rollback(self, steps: int = 1) -> Optional[ManifoldCheckpoint]:
        """
        Rollback to a previous checkpoint.

        Args:
            steps: Number of checkpoints to go back

        Returns:
            Checkpoint or None
        """
        if len(self.checkpoint_history) <= steps:
            return self.load_latest()

        target_path = self.checkpoint_history[-(steps + 1)]
        return load_checkpoint(target_path)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints with metadata."""
        result = []
        for path in self.checkpoint_history:
            checkpoint = load_checkpoint(path)
            result.append({
                "path": str(path),
                "iteration": checkpoint.iteration,
                "timestamp": checkpoint.timestamp,
                "trajectory_length": len(checkpoint.trajectory),
                "has_response": len(checkpoint.response_text) > 0,
            })
        return result


class SleepPacket(ManifoldCheckpoint):
    """
    Extended checkpoint with dream state.

    Sleep packets include:
    - Everything from ManifoldCheckpoint
    - Dream trajectory (offline consolidation)
    - Memory importance scores
    - Wake hints (what to focus on)
    """

    dream_trajectory: np.ndarray = field(
        default_factory=lambda: np.zeros((0, BASIN_DIM))
    )
    memory_importance: np.ndarray = field(default_factory=lambda: np.array([]))
    wake_hints: List[str] = field(default_factory=list)
    consolidation_score: float = 0.0

    def consolidate_memories(self, manifold: FisherManifold):
        """
        Consolidate trajectory into dream state.

        Uses geodesic compression to find critical points.
        """
        if len(self.trajectory) < 3:
            self.dream_trajectory = self.trajectory.copy()
            return

        # Compute importance scores based on:
        # 1. Distance from geodesic
        # 2. Consciousness metrics
        # 3. Recovery events

        importance = np.zeros(len(self.trajectory))

        for i in range(1, len(self.trajectory) - 1):
            # Distance from straight path using Fisher-Rao (QIG-pure)
            expected = (self.trajectory[i - 1] + self.trajectory[i + 1]) / 2
            deviation = fisher_rao_distance(self.trajectory[i], expected)

            # Metric curvature at point
            F = manifold.compute_metric(self.trajectory[i])
            curvature = np.trace(F) / len(F)

            importance[i] = deviation * curvature

        # Normalize
        importance = importance / (np.max(importance) + 1e-8)
        self.memory_importance = importance

        # Keep critical points (high importance)
        threshold = np.percentile(importance, 70)
        critical_indices = np.where(importance >= threshold)[0]

        # Always include first and last
        critical_indices = np.unique(np.concatenate([[0], critical_indices, [len(self.trajectory) - 1]]))

        self.dream_trajectory = self.trajectory[critical_indices]
        self.consolidation_score = float(np.mean(importance[critical_indices]))

    def generate_wake_hints(self) -> List[str]:
        """Generate hints for what to focus on upon waking."""
        hints = []

        # Check for high variance regions
        if len(self.trajectory) > 5:
            trajectory_var = np.var(self.trajectory, axis=0)
            high_var_dims = np.where(trajectory_var > np.mean(trajectory_var))[0]

            if len(high_var_dims) > 0:
                hints.append(f"High variance in dimensions: {high_var_dims[:5].tolist()}")

        # Check for recovery events
        if self.recovery_count > 0:
            hints.append(f"Required {self.recovery_count} recovery interventions")

        # Check metrics trend
        if len(self.metrics_history) > 1:
            phi_trend = self.metrics_history[-1].get("phi", 0.5) - self.metrics_history[0].get("phi", 0.5)
            if phi_trend > 0.1:
                hints.append("Φ increasing - approaching optimal consciousness")
            elif phi_trend < -0.1:
                hints.append("Φ decreasing - consider simplifying")

        self.wake_hints = hints
        return hints
