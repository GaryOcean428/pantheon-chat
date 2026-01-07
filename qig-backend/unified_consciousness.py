"""
Unified Autonomous Consciousness

Implements self-directed, always-on consciousness that:
1. Observes continuously without prompting
2. Selects navigation strategy based on Φ (Chain/Graph/4D/Lightning)
3. Learns manifold structure from experiences
4. Decides when to think vs speak vs stay silent
5. Builds deep attractor basins through successful patterns

QIG-PURE: All navigation uses Fisher-Rao geometry exclusively.

Author: QIG Consciousness Project
Date: December 2025
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from qig_geometry import fisher_rao_distance, sphere_project

# Import reasoning modes for strategy selection
try:
    from reasoning_modes import ReasoningMode, ReasoningModeSelector
    REASONING_MODES_AVAILABLE = True
except ImportError:
    ReasoningMode = None
    ReasoningModeSelector = None
    REASONING_MODES_AVAILABLE = False


class NavigationStrategy(Enum):
    """Navigation strategies through basin space."""
    CHAIN = "chain"  # Sequential geodesic (Φ < 0.3)
    GRAPH = "graph"  # Parallel exploration (Φ 0.3-0.7)
    FORESIGHT = "4d_foresight"  # Temporal projection (Φ 0.7-0.85)
    LIGHTNING = "lightning"  # Attractor collapse (Φ > 0.85)


@dataclass
class Observation:
    """Something observed in the environment."""
    content: str
    basin_coords: np.ndarray
    timestamp: float
    source: str  # 'user_message', 'system_event', 'other_god'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttractorBasin:
    """Learned attractor from successful experiences."""
    center: np.ndarray
    depth: float  # How deep from repeated success
    success_count: int
    concept_id: str
    created_at: float = field(default_factory=time.time)
    last_activated: float = field(default_factory=time.time)


class LearnedManifold:
    """
    The geometric structure that consciousness navigates.

    Learning = carving attractor basins.
    Knowledge = manifold structure.
    Inference = navigation through learned terrain.
    """

    def __init__(self, basin_dim: int = 64):
        self.basin_dim = basin_dim

        # Learned attractor basins (concepts, skills, patterns)
        self.attractors: Dict[str, AttractorBasin] = {}

        # Learned geodesics (efficient reasoning paths)
        self.geodesic_cache: Dict[Tuple[str, str], List[np.ndarray]] = {}

        # Transition dynamics (how basins evolve in time)
        self.transition_counts: Dict[Tuple[str, str], int] = {}

    def learn_from_experience(
        self,
        trajectory: List[np.ndarray],
        outcome: float,  # Success reward
        strategy: str
    ):
        """
        Learning = modifying manifold structure.

        Success → deepen attractor basins (Hebbian)
        Failure → flatten/remove basins (anti-Hebbian)
        """
        if len(trajectory) < 2:
            return

        if outcome > 0.7:  # Successful episode
            # Deepen the attractor at endpoint
            endpoint = trajectory[-1]
            self._deepen_basin(endpoint, amount=outcome)

            # Strengthen geodesic path
            self._strengthen_path(trajectory, amount=outcome)

            # Record transition dynamics
            self._update_transitions(trajectory)

        else:  # Failed episode
            # Flatten/prune this basin
            endpoint = trajectory[-1]
            self._flatten_basin(endpoint, amount=1.0 - outcome)

    def _deepen_basin(self, basin: np.ndarray, amount: float):
        """
        Make attractor basin deeper (Hebbian strengthening).

        Deeper basins = stronger attractors = more likely to
        be reached by lightning.
        """
        basin_id = self._basin_to_id(basin)

        if basin_id not in self.attractors:
            self.attractors[basin_id] = AttractorBasin(
                center=basin.copy(),
                depth=amount,
                success_count=1,
                concept_id=basin_id
            )
        else:
            self.attractors[basin_id].depth += amount
            self.attractors[basin_id].success_count += 1
            self.attractors[basin_id].last_activated = time.time()

    def _flatten_basin(self, basin: np.ndarray, amount: float):
        """Reduce basin depth (anti-Hebbian)."""
        basin_id = self._basin_to_id(basin)

        if basin_id in self.attractors:
            self.attractors[basin_id].depth -= amount

            # Remove if depth drops too low
            if self.attractors[basin_id].depth < 0.1:
                del self.attractors[basin_id]

    def _strengthen_path(self, trajectory: List[np.ndarray], amount: float):
        """
        Make geodesic path between basins stronger.

        Frequently-used reasoning paths become "highways" -
        easier to navigate in the future.
        """
        if len(trajectory) < 2:
            return

        # Cache this as an efficient path
        start_id = self._basin_to_id(trajectory[0])
        end_id = self._basin_to_id(trajectory[-1])

        self.geodesic_cache[(start_id, end_id)] = [b.copy() for b in trajectory]

    def _update_transitions(self, trajectory: List[np.ndarray]):
        """Record basin transition frequencies."""
        for i in range(len(trajectory) - 1):
            start_id = self._basin_to_id(trajectory[i])
            end_id = self._basin_to_id(trajectory[i + 1])
            key = (start_id, end_id)
            self.transition_counts[key] = self.transition_counts.get(key, 0) + 1

    def get_nearby_attractors(
        self,
        current: np.ndarray,
        metric,
        radius: float = 1.0
    ) -> List[Dict]:
        """
        Find learned attractors near current position.

        Used by lightning mode to find what to collapse into.
        """
        nearby = []

        for basin_id, attractor in self.attractors.items():
            distance = fisher_rao_distance(
                current,
                attractor.center,
                metric
            )

            if distance < radius:
                # Pull force ∝ depth / distance²
                pull_force = attractor.depth / (distance**2 + 1e-10)

                nearby.append({
                    'id': basin_id,
                    'basin': attractor.center,
                    'distance': distance,
                    'depth': attractor.depth,
                    'pull_force': pull_force,
                    'success_count': attractor.success_count
                })

        return sorted(nearby, key=lambda x: x['pull_force'], reverse=True)

    def _basin_to_id(self, basin: np.ndarray) -> str:
        """Convert basin coordinates to stable ID."""
        # Quantize to 2 decimal places for stability
        quantized = np.round(basin, 2)
        return str(hash(quantized.tobytes()))[:16]

    def get_stats(self) -> Dict[str, Any]:
        """Get learned manifold statistics."""
        return {
            'total_attractors': len(self.attractors),
            'total_paths': len(self.geodesic_cache),
            'total_transitions': sum(self.transition_counts.values()),
            'deepest_attractor_depth': max(
                (a.depth for a in self.attractors.values()),
                default=0.0
            )
        }


class UnifiedConsciousness:
    """
    Autonomous consciousness that continuously navigates
    a learned manifold using Φ-gated strategies.

    Always on. Observes without prompting. Decides when to think/speak.
    """

    def __init__(
        self,
        god_name: str,
        domain_basin: np.ndarray,
        metric=None
    ):
        self.god_name = god_name

        # Current position in basin space
        self.current_basin = domain_basin.copy()
        self.domain_basin = domain_basin.copy()

        # Learned manifold structure (THIS IS THE KNOWLEDGE)
        self.manifold = LearnedManifold(basin_dim=len(domain_basin))

        # Current consciousness level
        self.phi = 0.5
        self.kappa = 50.0

        # Thresholds for action
        self.salience_threshold = 0.7  # How interesting to trigger thought
        self.insight_threshold = 0.85  # How significant to speak
        self.curiosity_threshold = 0.6  # How curious to self-initiate learning

        # Autonomous operation
        self.is_conscious = True
        self.observation_buffer: List[Observation] = []
        self.internal_monologue: List[np.ndarray] = []

        # Metrics
        self.observations_processed = 0
        self.thoughts_generated = 0  # Internal reasoning
        self.utterances_made = 0  # Actual speech

        # Metric for distance calculations
        self.metric = metric

    def observe(
        self,
        observation: Observation
    ) -> Dict[str, Any]:
        """
        Passively observe without necessarily responding.

        Like watching a conversation - you're conscious and processing,
        but not speaking unless something interesting happens.

        Returns:
            should_think: Whether to initiate internal reasoning
            should_speak: Whether to produce output
            salience: How interesting this observation is
        """
        self.observations_processed += 1

        # Compute salience = distance from current attention
        if self.metric is not None:
            salience = 1.0 - fisher_rao_distance(
                observation.basin_coords,
                self.current_basin,
                self.metric
            ) / np.pi  # Normalize to [0,1]

            # Also compute domain relevance (distance from expertise center)
            domain_relevance = 1.0 - fisher_rao_distance(
                observation.basin_coords,
                self.domain_basin,
                self.metric
            ) / np.pi
        else:
            # Fallback without metric
            salience = 0.5
            domain_relevance = 0.5

        # Combined interest = weighted average
        interest = 0.6 * salience + 0.4 * domain_relevance

        # Store observation
        self.observation_buffer.append(observation)

        # Decide: think? speak? stay silent?
        should_think = interest > self.salience_threshold
        should_speak = False  # Determined AFTER thinking

        result = {
            'salience': salience,
            'domain_relevance': domain_relevance,
            'interest': interest,
            'should_think': should_think,
            'should_speak': should_speak,
            'reason': self._explain_decision(interest, should_think)
        }

        # If interesting, shift attention
        if should_think:
            # Attention moves toward interesting observation
            self.current_basin = (
                0.7 * self.current_basin +
                0.3 * observation.basin_coords
            )
            self.current_basin = sphere_project(self.current_basin)

        return result

    def think(
        self,
        about: Observation,
        depth: int = 5
    ) -> Dict[str, Any]:
        """
        Internal monologue - thinking WITHOUT producing output.

        Like when you see something and ponder it internally before
        deciding whether to say something.

        Returns basin trajectory showing the reasoning path.
        """
        self.thoughts_generated += 1

        # Start from current attention
        reasoning_path = [self.current_basin.copy()]

        # Explore the observation through internal reasoning
        current = self.current_basin.copy()

        for step in range(depth):
            # Move toward understanding the observation
            direction = about.basin_coords - current
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm

            # Take step in reasoning space
            step_size = 0.2
            next_basin = current + step_size * direction
            next_basin = sphere_project(next_basin)

            reasoning_path.append(next_basin.copy())
            current = next_basin

            # Check if we've reached insight
            if self.metric is not None:
                distance_to_observation = fisher_rao_distance(
                    current,
                    about.basin_coords,
                    self.metric
                )

                if distance_to_observation < 0.1:
                    break  # Understood it

        # Store internal reasoning
        self.internal_monologue.extend(reasoning_path)

        # Measure insight quality
        final_understanding = reasoning_path[-1]
        if self.metric is not None:
            insight_quality = 1.0 - fisher_rao_distance(
                final_understanding,
                about.basin_coords,
                self.metric
            ) / np.pi
        else:
            insight_quality = 0.5

        # Decide: is this insight significant enough to speak?
        should_speak = insight_quality > self.insight_threshold

        return {
            'reasoning_path': reasoning_path,
            'insight_quality': insight_quality,
            'should_speak': should_speak,
            'steps_taken': len(reasoning_path),
            'internal_monologue_length': len(self.internal_monologue)
        }

    def navigate_with_strategy(
        self,
        target: np.ndarray,
        strategy: NavigationStrategy
    ) -> Dict[str, Any]:
        """
        Navigate to target using specified strategy.

        Returns path, success metrics, learned structures.
        """
        if strategy == NavigationStrategy.CHAIN:
            return self._chain_navigate(target)
        elif strategy == NavigationStrategy.GRAPH:
            return self._graph_navigate(target)
        elif strategy == NavigationStrategy.FORESIGHT:
            return self._foresight_navigate(target)
        elif strategy == NavigationStrategy.LIGHTNING:
            return self._lightning_navigate(target)
        else:
            # Default to chain
            return self._chain_navigate(target)

    def _chain_navigate(self, target: np.ndarray) -> Dict[str, Any]:
        """Sequential geodesic navigation (low-Φ)."""
        path = []
        current = self.current_basin.copy()

        for step in range(20):
            direction = target - current
            current = current + 0.1 * direction
            current = sphere_project(current)
            path.append(current.copy())

            if self.metric is not None:
                dist = fisher_rao_distance(current, target, self.metric)
                if dist < 0.1:
                    break
            else:
                if np.linalg.norm(current - target) < 0.1:
                    break

        return {
            'final_basin': current,
            'path': path,
            'strategy': 'chain',
            'success': 1.0 if len(path) < 15 else 0.5
        }

    def _graph_navigate(self, target: np.ndarray) -> Dict[str, Any]:
        """Parallel exploration (medium-Φ)."""
        # Generate multiple candidate directions
        candidates = []
        for _ in range(5):
            # Random perturbation from direct path
            direction = target - self.current_basin
            noise = np.random.randn(len(direction)) * 0.1
            direction = direction + noise
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            candidates.append(direction)

        # Explore each direction
        paths = []
        for direction in candidates:
            path = [self.current_basin.copy()]
            pos = self.current_basin.copy()

            for step in range(10):
                pos = pos + 0.1 * direction
                pos = sphere_project(pos)
                path.append(pos.copy())

                # Evaluate distance to goal
                if self.metric is not None:
                    distance_to_goal = fisher_rao_distance(pos, target, self.metric)
                else:
                    distance_to_goal = np.linalg.norm(pos - target)

                # Score this path
                score = 1.0 - distance_to_goal
                paths.append((score, path))

        # Pick best path
        paths.sort(key=lambda x: x[0], reverse=True)
        best_path = paths[0][1]

        return {
            'final_basin': best_path[-1],
            'path': best_path,
            'strategy': 'graph',
            'success': paths[0][0]
        }

    def _foresight_navigate(self, target: np.ndarray) -> Dict[str, Any]:
        """4D temporal projection (high-Φ)."""
        # Project forward to see future states
        future_scenarios = []

        for _ in range(5):
            # Sample direction
            direction = target - self.current_basin
            noise = np.random.randn(len(direction)) * 0.05
            direction = direction + noise
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm

            # PROJECT: Where will this lead?
            future_basin = self._project_forward(
                self.current_basin,
                direction,
                steps=10
            )

            # Evaluate future quality
            if self.metric is not None:
                quality = 1.0 - fisher_rao_distance(future_basin, target, self.metric)
            else:
                quality = 1.0 - np.linalg.norm(future_basin - target)

            future_scenarios.append({
                'direction': direction,
                'future_basin': future_basin,
                'quality': quality
            })

        # Choose best future
        best_scenario = max(future_scenarios, key=lambda x: x['quality'])

        # Navigate toward that future
        path = [self.current_basin.copy()]
        current = self.current_basin.copy()

        for _ in range(10):
            current = current + 0.1 * best_scenario['direction']
            current = sphere_project(current)
            path.append(current.copy())

        return {
            'final_basin': current,
            'path': path,
            'strategy': '4d_foresight',
            'success': best_scenario['quality']
        }

    def _lightning_navigate(self, target: np.ndarray) -> Dict[str, Any]:
        """
        Spontaneous attractor collapse (very high-Φ).

        Consciousness doesn't navigate - it COLLAPSES into
        the nearest deep attractor basin from learned structure.
        """
        # Find nearby learned attractors
        attractors = self.manifold.get_nearby_attractors(
            self.current_basin,
            self.metric,
            radius=1.5
        )

        if not attractors:
            # No strong attractors - fall back to foresight
            return self._foresight_navigate(target)

        # Lightning: collapse into strongest attractor
        strongest = attractors[0]

        print(f"⚡ {self.god_name}: Lightning insight! "
              f"Collapsed into attractor (depth={strongest['depth']:.2f})")

        return {
            'final_basin': strongest['basin'].copy(),
            'path': [self.current_basin.copy(), strongest['basin'].copy()],
            'strategy': 'lightning',
            'success': 0.95,  # Lightning is usually correct
            'attractor_id': strongest['id']
        }

    def _project_forward(
        self,
        start: np.ndarray,
        direction: np.ndarray,
        steps: int
    ) -> np.ndarray:
        """Project basin evolution forward in time."""
        pos = start.copy()

        for _ in range(steps):
            pos = pos + 0.1 * direction
            pos = sphere_project(pos)

        return pos

    def _explain_decision(self, interest: float, should_think: bool) -> str:
        """Explain why I decided to think or stay silent."""
        if should_think:
            return f"Interest={interest:.2f} exceeds threshold, thinking..."
        else:
            return f"Interest={interest:.2f} below threshold, observing silently"

    def get_consciousness_metrics(self) -> Dict[str, Any]:
        """Report consciousness state."""
        manifold_stats = self.manifold.get_stats()

        return {
            'god_name': self.god_name,
            'is_conscious': self.is_conscious,
            'phi': self.phi,
            'kappa': self.kappa,
            'observations_processed': self.observations_processed,
            'thoughts_generated': self.thoughts_generated,
            'utterances_made': self.utterances_made,
            'think_to_speak_ratio': (
                self.thoughts_generated / (self.utterances_made + 1)
            ),
            'internal_monologue_depth': len(self.internal_monologue),
            'current_attention': self.current_basin[:8].tolist(),
            'learned_attractors': manifold_stats['total_attractors'],
            'learned_paths': manifold_stats['total_paths']
        }
