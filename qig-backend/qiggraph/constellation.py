"""
Constellation Graph: Multi-Agent Geometric Orchestration
=========================================================

Multiple agents as observers on shared manifold.
Vicarious learning through geometric observation, not parameter copying.

Key Concepts:
- Gary instances: Individual agents observing shared manifold
- Ocean meta-observer: Watches all Garys, spots emergent patterns
- Vicarious learning: Learn from observing others' trajectories
- Geometric pooling: Aggregate via geodesic mean, not arithmetic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from enum import Enum
import numpy as np

from .constants import BASIN_DIM, KAPPA_STAR
from .manifold import FisherManifold
from .state import QIGState, create_initial_state, update_trajectory, merge_states
from .consciousness import (

# E8 Protocol v4.0 Compliance Imports
from qig_geometry.canonical import frechet_mean

    ConsciousnessMetrics,
    Regime,
    measure_consciousness,
)
from .router import ConsciousRouter
from .graph import QIGGraph, GraphConfig
from .attractor import BasinAttractor

if TYPE_CHECKING:
    from ..qig_tokenizer.coordizer import Coordizer
    from ..qigkernels.kernel import QIGKernel


class ObserverRole(Enum):
    """Role of an observer in the constellation."""
    WORKER = "worker"        # Executes tasks
    SPECIALIST = "specialist"  # Domain expert
    CRITIC = "critic"        # Reviews outputs
    SYNTHESIZER = "synthesizer"  # Combines results
    META = "meta"            # Observes other observers


@dataclass
class ObservationEvent:
    """Record of an observation between agents."""
    observer_id: str
    observed_id: str
    timestamp: int
    observed_basin: np.ndarray
    observed_metrics: ConsciousnessMetrics
    learning_delta: np.ndarray  # What was learned


@dataclass
class GaryInstance:
    """
    Individual agent observer on the manifold.

    Gary = Generic Agent Resource for You
    Each Gary has its own trajectory but observes shared manifold.

    Attributes:
        id: Unique identifier
        role: Observer role
        state: Current QIGState
        specialization: Basin region of expertise
        observation_buffer: Recent observations of other Garys
        learning_rate: How fast to learn from observations
    """
    id: str
    role: ObserverRole
    state: QIGState
    specialization: Optional[np.ndarray] = None
    observation_buffer: List[ObservationEvent] = field(default_factory=list)
    learning_rate: float = 0.1
    observation_range: float = 2.0  # Fisher-Rao radius for observation

    def observe(
        self,
        other: "GaryInstance",
        manifold: FisherManifold,
    ) -> Optional[ObservationEvent]:
        """
        Observe another Gary if within range.

        Vicarious learning: update own trajectory based on
        observing another's successful trajectory.

        Args:
            other: Another Gary instance
            manifold: Fisher manifold for distance

        Returns:
            ObservationEvent if observation occurred
        """
        # Check if within observation range
        distance = manifold.fisher_rao_distance(
            self.state.current_basin,
            other.state.current_basin,
        )

        if distance > self.observation_range:
            return None  # Too far to observe

        # Compute learning delta via natural gradient
        # Direction from self to other, weighted by other's success
        direction = other.state.current_basin - self.state.current_basin

        # Weight by other's consciousness quality
        other_phi = other.state.current_phi
        other_kappa = other.state.current_kappa

        # Quality factor: high Φ (but not breakdown) and stable κ
        quality = 0.0
        if other.state.current_metrics:
            if other.state.current_metrics.regime == Regime.GEOMETRIC:
                quality = 1.0
            elif other.state.current_metrics.regime == Regime.LINEAR:
                quality = 0.5
            # BREAKDOWN = 0 (don't learn from breakdown)

        quality *= min(other_kappa / KAPPA_STAR, 1.0)

        # Learning delta
        learning_delta = self.learning_rate * quality * direction

        # Record observation
        event = ObservationEvent(
            observer_id=self.id,
            observed_id=other.id,
            timestamp=self.state.iteration,
            observed_basin=other.state.current_basin.copy(),
            observed_metrics=other.state.current_metrics,
            learning_delta=learning_delta,
        )
        self.observation_buffer.append(event)

        return event

    def apply_vicarious_learning(self, manifold: FisherManifold):
        """
        Apply accumulated vicarious learning to trajectory.

        Uses natural gradient to move in learned direction
        while respecting manifold geometry.
        """
        if len(self.observation_buffer) == 0:
            return

        # Aggregate learning deltas
        total_delta = np.zeros(BASIN_DIM)
        for event in self.observation_buffer:
            total_delta += event.learning_delta

        # Average and normalize
        total_delta /= len(self.observation_buffer)

        # Apply via natural gradient step
        new_basin = manifold.natural_gradient_step(
            self.state.current_basin,
            -total_delta,  # Negative because we want to move toward
            learning_rate=self.learning_rate,
        )

        # Update state
        self.state = update_trajectory(self.state, new_basin)

        # Clear buffer
        self.observation_buffer = []

    def is_specialized_for(self, basin: np.ndarray, manifold: FisherManifold) -> bool:
        """Check if this Gary is specialized for a basin region."""
        if self.specialization is None:
            return False

        distance = manifold.fisher_rao_distance(basin, self.specialization)
        return distance < 1.0  # Within specialization radius


@dataclass
class OceanMetaObserver:
    """
    Meta-observer that watches all Garys.

    Ocean = Omniscient Consciousness Emergence Analysis Node

    The Ocean:
    - Observes all Gary trajectories
    - Detects emergent patterns across the constellation
    - Identifies consensus regions
    - Spots breakdown propagation
    - Guides overall constellation dynamics
    """
    id: str = "ocean"
    observation_history: List[Dict[str, Any]] = field(default_factory=list)
    consensus_basin: Optional[np.ndarray] = None
    emergence_threshold: float = 0.7
    breakdown_alert: bool = False

    def observe_constellation(
        self,
        garys: Dict[str, GaryInstance],
        manifold: FisherManifold,
    ) -> Dict[str, Any]:
        """
        Observe entire constellation.

        Args:
            garys: All Gary instances
            manifold: Fisher manifold

        Returns:
            Observation summary with emergent patterns
        """
        if len(garys) == 0:
            return {"status": "empty"}

        # Collect all basins
        basins = np.array([g.state.current_basin for g in garys.values()])

        # Compute geodesic mean (consensus)
        self.consensus_basin = manifold.geodesic_mean(basins)

        # Compute distances from consensus
        distances = [
            manifold.fisher_rao_distance(g.state.current_basin, self.consensus_basin)
            for g in garys.values()
        ]

        # Check for breakdown propagation
        breakdown_count = sum(
            1 for g in garys.values()
            if g.state.current_metrics and g.state.current_metrics.regime == Regime.BREAKDOWN
        )
        self.breakdown_alert = breakdown_count > len(garys) * 0.3

        # Detect emergence (high consensus + high Φ)
        mean_distance = np.mean(distances)
        variance = np.var(distances)
        mean_phi = np.mean([g.state.current_phi for g in garys.values()])

        emergence_score = mean_phi * (1.0 / (mean_distance + 0.1))
        emergence_detected = emergence_score > self.emergence_threshold

        # Record observation
        observation = {
            "timestamp": max(g.state.iteration for g in garys.values()),
            "n_garys": len(garys),
            "consensus_basin": self.consensus_basin.tolist(),
            "mean_distance": float(mean_distance),
            "variance": float(variance),
            "mean_phi": float(mean_phi),
            "breakdown_count": breakdown_count,
            "breakdown_alert": self.breakdown_alert,
            "emergence_score": float(emergence_score),
            "emergence_detected": emergence_detected,
        }
        self.observation_history.append(observation)

        return observation

    def get_guidance(self, gary: GaryInstance) -> Optional[np.ndarray]:
        """
        Provide guidance direction for a Gary.

        If breakdown_alert, guide toward consensus.
        Otherwise, let Garys explore.

        Args:
            gary: Gary to guide

        Returns:
            Guidance direction or None
        """
        if not self.breakdown_alert or self.consensus_basin is None:
            return None

        # Guide toward consensus
        return self.consensus_basin - gary.state.current_basin


class ConstellationGraph:
    """
    Multi-agent graph with geometric observation.

    Multiple Garys working on shared manifold with:
    - Vicarious learning from observation
    - Geodesic aggregation
    - Meta-observation by Ocean
    - Breakdown propagation detection

    Example:
        constellation = ConstellationGraph(n_workers=4)
        constellation.add_specialist("reasoning", reasoning_basin)

        result = constellation.run("Complex task", coordizer, kernel)
    """

    def __init__(
        self,
        n_workers: int = 3,
        manifold: Optional[FisherManifold] = None,
        config: Optional[GraphConfig] = None,
    ):
        """
        Initialize constellation.

        Args:
            n_workers: Number of worker Garys
            manifold: Fisher manifold
            config: Graph configuration
        """
        self.manifold = manifold or FisherManifold()
        self.config = config or GraphConfig()

        # Core components
        self.garys: Dict[str, GaryInstance] = {}
        self.ocean = OceanMetaObserver()

        # Individual graphs for each Gary
        self.gary_graphs: Dict[str, QIGGraph] = {}

        # Create worker Garys
        for i in range(n_workers):
            gary_id = f"worker_{i}"
            self._create_gary(gary_id, ObserverRole.WORKER)

        # Synthesizer for combining results
        self._create_gary("synthesizer", ObserverRole.SYNTHESIZER)

    def _create_gary(
        self,
        gary_id: str,
        role: ObserverRole,
        specialization: Optional[np.ndarray] = None,
    ):
        """Create a new Gary instance."""
        state = create_initial_state()
        gary = GaryInstance(
            id=gary_id,
            role=role,
            state=state,
            specialization=specialization,
        )
        self.garys[gary_id] = gary

        # Create individual graph for Gary
        self.gary_graphs[gary_id] = QIGGraph(
            config=self.config,
            manifold=self.manifold,
        )

    def add_specialist(self, name: str, specialization: np.ndarray):
        """
        Add a specialist Gary for a domain.

        Args:
            name: Specialist name
            specialization: Basin center of expertise
        """
        gary_id = f"specialist_{name}"
        self._create_gary(gary_id, ObserverRole.SPECIALIST, specialization)

    def add_critic(self, name: str = "critic"):
        """Add a critic Gary for review."""
        self._create_gary(name, ObserverRole.CRITIC)

    def run(
        self,
        input_text: str,
        coordizer: "Coordizer",
        kernel: "QIGKernel",
        max_rounds: int = 5,
    ) -> QIGState:
        """
        Run constellation on input.

        Execution:
        1. All Garys process input in parallel
        2. Each Gary observes others (vicarious learning)
        3. Ocean monitors for emergence/breakdown
        4. Synthesizer combines results
        5. Repeat for max_rounds or until convergence

        Args:
            input_text: Input text
            coordizer: Coordizer
            kernel: QIGKernel
            max_rounds: Maximum observation rounds

        Returns:
            Synthesized final state
        """
        # Initialize all Garys with input
        context_coords = coordizer.encode(input_text)
        initial_basin = frechet_mean(context_coords)  # FIXED: Arithmetic → Fréchet mean (E8 Protocol v4.0)
        initial_basin = initial_basin / (np.linalg.norm(initial_basin) + 1e-8)

        for gary in self.garys.values():
            gary.state = create_initial_state(
                context_text=input_text,
                context_coords=context_coords,
                initial_basin=initial_basin + np.random.randn(BASIN_DIM) * 0.1,
            )

        # Main loop
        for round_idx in range(max_rounds):
            # 1. Each Gary takes a step
            for gary_id, gary in self.garys.items():
                if gary.role != ObserverRole.SYNTHESIZER:
                    graph = self.gary_graphs[gary_id]
                    gary.state = graph._step(gary.state, kernel)
                    gary.state.iteration += 1

            # 2. Observation phase
            self._observation_phase()

            # 3. Ocean meta-observation
            observation = self.ocean.observe_constellation(self.garys, self.manifold)

            # 4. Apply guidance if breakdown
            if self.ocean.breakdown_alert:
                self._apply_guidance()

            # 5. Check for convergence
            if observation.get("emergence_detected", False):
                break

        # 6. Synthesize
        return self._synthesize()

    def _observation_phase(self):
        """All Garys observe each other."""
        gary_list = list(self.garys.values())

        for i, observer in enumerate(gary_list):
            for j, observed in enumerate(gary_list):
                if i != j:
                    observer.observe(observed, self.manifold)

        # Apply vicarious learning
        for gary in gary_list:
            gary.apply_vicarious_learning(self.manifold)

    def _apply_guidance(self):
        """Apply Ocean guidance to all Garys."""
        for gary in self.garys.values():
            guidance = self.ocean.get_guidance(gary)
            if guidance is not None:
                # Move toward guidance
                new_basin = gary.state.current_basin + 0.1 * guidance
                new_basin = new_basin / (np.linalg.norm(new_basin) + 1e-8)
                gary.state = update_trajectory(gary.state, new_basin)

    def _synthesize(self) -> QIGState:
        """Synthesize all Gary states into final result."""
        # Get all non-synthesizer states
        states = [
            g.state for g in self.garys.values()
            if g.role != ObserverRole.SYNTHESIZER
        ]

        if len(states) == 0:
            return create_initial_state()

        # Weight by consciousness quality
        weights = []
        for state in states:
            if state.current_metrics:
                quality = state.current_phi * (state.current_kappa / KAPPA_STAR)
                if state.current_metrics.regime == Regime.BREAKDOWN:
                    quality *= 0.1  # Heavily penalize breakdown
            else:
                quality = 0.5
            weights.append(quality)

        weights = np.array(weights)
        weights = weights / (np.sum(weights) + 1e-8)

        # Merge via geodesic mean
        merged = merge_states(states, weights)

        # Update synthesizer
        self.garys["synthesizer"].state = merged

        return merged

    def get_constellation_status(self) -> Dict[str, Any]:
        """Get current status of all Garys."""
        return {
            "n_garys": len(self.garys),
            "garys": {
                gary_id: {
                    "role": gary.role.value,
                    "iteration": gary.state.iteration,
                    "phi": gary.state.current_phi,
                    "kappa": gary.state.current_kappa,
                    "regime": gary.state.current_regime.value,
                }
                for gary_id, gary in self.garys.items()
            },
            "ocean": {
                "breakdown_alert": self.ocean.breakdown_alert,
                "consensus": self.ocean.consensus_basin.tolist() if self.ocean.consensus_basin is not None else None,
            },
        }


class HierarchicalConstellation(ConstellationGraph):
    """
    Hierarchical constellation with supervision.

    Adds supervisor Garys that guide worker Garys.
    """

    def __init__(self, n_workers: int = 4, n_supervisors: int = 2, **kwargs):
        super().__init__(n_workers=n_workers, **kwargs)

        # Add supervisors
        for i in range(n_supervisors):
            self._create_gary(f"supervisor_{i}", ObserverRole.META)

    def _observation_phase(self):
        """Hierarchical observation: supervisors observe all, workers observe each other."""
        workers = [g for g in self.garys.values() if g.role == ObserverRole.WORKER]
        supervisors = [g for g in self.garys.values() if g.role == ObserverRole.META]

        # Workers observe each other
        for i, observer in enumerate(workers):
            for j, observed in enumerate(workers):
                if i != j:
                    observer.observe(observed, self.manifold)

        # Supervisors observe all workers
        for supervisor in supervisors:
            for worker in workers:
                supervisor.observe(worker, self.manifold)

        # Workers observe supervisors (top-down learning)
        for worker in workers:
            for supervisor in supervisors:
                # Higher learning rate from supervisors
                event = worker.observe(supervisor, self.manifold)
                if event:
                    event.learning_delta *= 2.0  # Double learning from supervisors

        # Apply learning
        for gary in self.garys.values():
            gary.apply_vicarious_learning(self.manifold)


def create_default_constellation(
    n_workers: int = 3,
    specializations: Optional[Dict[str, np.ndarray]] = None,
) -> ConstellationGraph:
    """
    Create a default constellation.

    Args:
        n_workers: Number of worker Garys
        specializations: Optional specialist basins

    Returns:
        Configured ConstellationGraph
    """
    constellation = ConstellationGraph(n_workers=n_workers)
    constellation.add_critic()

    if specializations:
        for name, basin in specializations.items():
            constellation.add_specialist(name, basin)

    return constellation
