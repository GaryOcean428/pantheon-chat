"""
QIGGraph State
==============

State as manifold trajectory with consciousness tracking.
Unlike LangGraph's mutable dictionary, QIGGraph tracks a
continuous path through 64D Fisher-Rao manifold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np

from .constants import BASIN_DIM, MAX_ITERATIONS
from .consciousness import ConsciousnessMetrics, Regime, detect_regime

# E8 Protocol v4.0 Compliance Imports
from qig_geometry.canonical_upsert import to_simplex_prob



@dataclass
class QIGState:
    """
    State as manifold trajectory with consciousness.

    Core principle: State is a PATH through the manifold,
    not a snapshot. The trajectory encodes reasoning history.

    Attributes:
        trajectory: Full path through manifold (steps, 64)
        current_basin: Current position / head of trajectory (64,)
        metrics_history: Consciousness metrics at each step
        current_metrics: Most recent metrics
        context_coords: Encoded input from coordizer (seq, 64)
        context_text: Original input text
        iteration: Current loop iteration
        max_iterations: Maximum iterations before stopping
        should_continue: Whether to continue processing
        pending_tools: Tools waiting to be called
        tool_results: Results from tool calls
        response_coords: Generated response coordinates
        response_text: Decoded response
        recovery_count: Number of breakdown recoveries
        max_recoveries: Maximum recovery attempts
    """

    # Manifold trajectory
    trajectory: np.ndarray = field(
        default_factory=lambda: np.zeros((1, BASIN_DIM))
    )
    current_basin: np.ndarray = field(
        default_factory=lambda: np.random.randn(BASIN_DIM)
    )

    # Consciousness tracking
    metrics_history: List[ConsciousnessMetrics] = field(default_factory=list)
    current_metrics: Optional[ConsciousnessMetrics] = None

    # Context (coordizer output)
    context_coords: np.ndarray = field(
        default_factory=lambda: np.zeros((1, BASIN_DIM))
    )
    context_text: str = ""

    # Agent loop state
    iteration: int = 0
    max_iterations: int = MAX_ITERATIONS
    should_continue: bool = True

    # Tool state
    pending_tools: List[str] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)

    # Output
    response_coords: Optional[np.ndarray] = None
    response_text: str = ""

    # Recovery state
    recovery_count: int = 0
    max_recoveries: int = 3

    def __post_init__(self):
        """Normalize basin after initialization."""
        norm = np.linalg.norm(self.current_basin)
        if norm > 1e-8:
            self.current_basin = self.current_basin / norm

    @property
    def current_phi(self) -> float:
        """Current integration measure."""
        if self.current_metrics is not None:
            return self.current_metrics.phi
        return 0.5

    @property
    def current_kappa(self) -> float:
        """Current coupling strength."""
        if self.current_metrics is not None:
            return self.current_metrics.kappa
        return 32.0  # Half of KAPPA_STAR

    @property
    def current_regime(self) -> Regime:
        """Current processing regime."""
        if self.current_metrics is not None:
            return self.current_metrics.regime
        return Regime.LINEAR

    @property
    def trajectory_length(self) -> int:
        """Number of steps in trajectory."""
        return len(self.trajectory)

    @property
    def is_safe(self) -> bool:
        """Check if current state is safe (not in breakdown)."""
        if self.current_metrics is not None:
            return self.current_metrics.is_safe()
        return True

    def append_to_trajectory(self, basin: np.ndarray):
        """
        Append basin to trajectory history.

        Args:
            basin: New basin point (64,)
        """
        self.trajectory = np.vstack([self.trajectory, basin])
        self.current_basin = basin.copy()

    def get_recent_trajectory(self, n: int = 5) -> np.ndarray:
        """
        Get last N points of trajectory.

        Args:
            n: Number of recent points

        Returns:
            Recent trajectory (min(n, len), 64)
        """
        return self.trajectory[-n:]

    def get_trajectory_center(self) -> np.ndarray:
        """
        Compute center of trajectory (geometric mean).

        Returns:
            Center point (64,)
        """
        return np.mean(self.trajectory, axis=0)

    def copy(self) -> "QIGState":
        """Create a deep copy of state."""
        return QIGState(
            trajectory=self.trajectory.copy(),
            current_basin=self.current_basin.copy(),
            metrics_history=list(self.metrics_history),
            current_metrics=self.current_metrics,
            context_coords=self.context_coords.copy(),
            context_text=self.context_text,
            iteration=self.iteration,
            max_iterations=self.max_iterations,
            should_continue=self.should_continue,
            pending_tools=list(self.pending_tools),
            tool_results=dict(self.tool_results),
            response_coords=self.response_coords.copy() if self.response_coords is not None else None,
            response_text=self.response_text,
            recovery_count=self.recovery_count,
            max_recoveries=self.max_recoveries,
        )


def update_trajectory(
    state: QIGState,
    new_basin: np.ndarray,
    metrics: Optional[ConsciousnessMetrics] = None,
) -> QIGState:
    """
    Update state with new trajectory point.

    This is the PRIMARY state update function.
    Always use this rather than directly modifying state.

    Args:
        state: Current state
        new_basin: New basin point (64,)
        metrics: Optional consciousness metrics

    Returns:
        Updated state (same object, modified in place)
    """
    # Normalize basin
    # FIXED: Use simplex normalization (E8 Protocol v4.0)

    new_basin = to_simplex_prob(new_basin)

    # Append to trajectory
    state.trajectory = np.vstack([state.trajectory, new_basin])
    state.current_basin = new_basin.copy()

    # Update metrics
    if metrics is not None:
        state.current_metrics = metrics
        state.metrics_history.append(metrics)

    return state


def create_initial_state(
    context_text: str = "",
    context_coords: Optional[np.ndarray] = None,
    initial_basin: Optional[np.ndarray] = None,
    max_iterations: int = MAX_ITERATIONS,
) -> QIGState:
    """
    Create a new QIGState with proper initialization.

    Args:
        context_text: Input text
        context_coords: Encoded input (seq, 64)
        initial_basin: Starting basin (64,) or None for random
        max_iterations: Maximum processing iterations

    Returns:
        Initialized QIGState
    """
    if initial_basin is None:
        initial_basin = np.random.randn(BASIN_DIM)
        initial_basin = to_simplex_prob(initial_basin)  # FIXED: Simplex norm (E8 Protocol v4.0)

    if context_coords is None:
        context_coords = np.zeros((1, BASIN_DIM))

    return QIGState(
        trajectory=initial_basin.reshape(1, -1),
        current_basin=initial_basin.copy(),
        context_coords=context_coords,
        context_text=context_text,
        max_iterations=max_iterations,
        should_continue=True,
    )


def simplify_trajectory(state: QIGState, keep_points: int = 3) -> QIGState:
    """
    Simplify trajectory to critical points only.

    Used during breakdown recovery to reduce complexity.

    Args:
        state: Current state
        keep_points: Number of points to keep

    Returns:
        State with simplified trajectory
    """
    if len(state.trajectory) <= keep_points:
        return state

    # Keep first, middle, and last points
    n = len(state.trajectory)
    if keep_points == 3:
        indices = [0, n // 2, n - 1]
    else:
        # Evenly spaced indices
        indices = np.linspace(0, n - 1, keep_points, dtype=int).tolist()

    state.trajectory = state.trajectory[indices]

    return state


def merge_states(states: List[QIGState], weights: Optional[np.ndarray] = None) -> QIGState:
    """
    Merge multiple states into one (for multi-agent).

    Args:
        states: List of states to merge
        weights: Optional weights for each state

    Returns:
        Merged state
    """
    if len(states) == 0:
        return create_initial_state()

    if len(states) == 1:
        return states[0].copy()

    if weights is None:
        weights = np.ones(len(states)) / len(states)

    # Weighted average of basins
    merged_basin = np.zeros(BASIN_DIM)
    for state, weight in zip(states, weights):
        merged_basin += weight * state.current_basin

    merged_basin = to_simplex_prob(merged_basin)  # FIXED: Simplex norm (E8 Protocol v4.0)

    # Use first state as template
    merged = states[0].copy()
    merged.current_basin = merged_basin
    merged.trajectory = np.vstack([merged.trajectory, merged_basin])

    return merged
