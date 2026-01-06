"""
4D Temporal Reasoning: Foresight & Scenario Planning

Two temporal modes for hyperdimensional (Φ > 0.75) reasoning:

1. FORESIGHT (Future→Present): Geodesic prophecy
   - "Seeing" what will happen - singular, intuitive
   - Follow natural geodesic to attractor, trace backwards
   - Fast, one clear vision

2. SCENARIO PLANNING (Present→Future): Branching exploration
   - Running possibilities - analytical, branching
   - Simulate multiple paths, evaluate outcomes
   - Slower, multiple simulations

QIG Purity Note:
  All distance computations use Fisher-Rao from qig_geometry.py.
  The sphere_project() function uses np.linalg.norm() which is APPROVED
  per QIG Purity Addendum Section 3: normalization for numerical stability
  and projection to unit sphere in embedding space. This is NOT used for
  basin coordinate distance comparisons (which use fisher_coord_distance).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qig_geometry import (
    fisher_coord_distance,
    fisher_rao_distance,
    geodesic_interpolation,
    sphere_project,
)
from qigkernels.physics_constants import PHI_HYPERDIMENSIONAL
from prediction_self_improvement import (
    get_prediction_improvement,
    PredictionFailureReason,
    PredictionSelfImprovement,
)
from generative_reasoning import get_generative_reasoning


class TemporalMode(Enum):
    """Two modes of temporal reasoning."""
    FORESIGHT = "foresight"
    SCENARIO = "scenario"


@dataclass
class ForesightVision:
    """
    A singular vision of the future.
    
    Foresight = geodesic extrapolation on Fisher manifold.
    
    Usage:
    - arrival_time: Steps until reaching attractor basin
    - confidence: How certain the prediction is (0-1)
    - attractor_strength: How strong the pull of the destination basin is
    - geodesic_naturalness: How smooth/natural the predicted path is (1.0 = perfect geodesic)
    - future_basin: The 64D coordinates of the predicted destination
    
    Decision-making:
    - High confidence (>0.7) + high attractor_strength (>0.5) → trust the vision, navigate toward it
    - Low confidence (<0.3) → uncertain future, increase exploration/scenario planning
    - Low geodesic_naturalness (<0.5) → bumpy path ahead, proceed cautiously
    """
    future_basin: np.ndarray
    arrival_time: int
    confidence: float
    path_backwards: List[np.ndarray]
    attractor_strength: float
    geodesic_naturalness: float
    
    def __str__(self):
        """Verbose representation showing what the vision means."""
        basin_list = self.future_basin.tolist() if hasattr(self.future_basin, 'tolist') else list(self.future_basin)
        basin_str = ', '.join(f'{v:.4f}' for v in basin_list)
        return (f"Vision: arrive={self.arrival_time} steps, "
                f"conf={self.confidence:.1%}, attractor={self.attractor_strength:.2f}, "
                f"naturalness={self.geodesic_naturalness:.2f}, "
                f"basin=[{basin_str}]")
    
    def is_actionable(self) -> bool:
        """Check if this vision is strong enough to guide decisions."""
        return self.confidence > 0.5 and self.attractor_strength > 0.3
    
    def get_guidance(self) -> str:
        """Get guidance text based on vision quality using generative capability."""
        reasoning = get_generative_reasoning()
        return reasoning.generate_foresight_guidance(
            confidence=self.confidence,
            attractor_strength=self.attractor_strength,
            naturalness=self.geodesic_naturalness,
            future_basin=self.future_basin
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'future_basin': self.future_basin.tolist() if hasattr(self.future_basin, 'tolist') else list(self.future_basin),
            'arrival_time': self.arrival_time,
            'confidence': self.confidence,
            'path_length': len(self.path_backwards),
            'attractor_strength': self.attractor_strength,
            'geodesic_naturalness': self.geodesic_naturalness,
        }


@dataclass  
class ScenarioBranch:
    """One possible future path."""
    path_forward: List[np.ndarray]
    final_basin: np.ndarray
    probability: float
    quality: float
    action_description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'final_basin': self.final_basin.tolist() if hasattr(self.final_basin, 'tolist') else list(self.final_basin),
            'probability': self.probability,
            'quality': self.quality,
            'path_length': len(self.path_forward),
            'action': self.action_description,
        }


@dataclass
class ScenarioTree:
    """Multiple branching futures."""
    root_basin: np.ndarray
    branches: List[ScenarioBranch]
    most_probable: ScenarioBranch
    
    def __str__(self):
        return (f"Scenarios: {len(self.branches)} possibilities, "
                f"most probable: {self.most_probable.probability:.1%}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_branches': len(self.branches),
            'most_probable': self.most_probable.to_dict(),
            'branches': [b.to_dict() for b in self.branches],
        }


class TemporalReasoning:
    """
    4D reasoning with foresight and scenario planning.
    
    Requires Φ > 0.75 (hyperdimensional consciousness).
    
    Now includes self-improvement through:
    - Failure analysis: Understand WHY predictions fail
    - Chain/graph analysis: Learn prediction patterns
    - Recursive loops: Self-directed improvement
    """
    
    PHI_THRESHOLD = PHI_HYPERDIMENSIONAL
    
    def __init__(self, basin_dim: int = 64):
        self.basin_dim = basin_dim
        
        self.foresight_horizon = 50
        # Increased threshold so attractors are detected with moderate convergence
        # Old: 0.1 was too strict - only perfect stationarity detected as attractor
        # New: 0.2 allows detection when movement significantly slows
        self.attractor_detection_threshold = 0.2
        
        self.scenario_branches = 5
        self.scenario_depth = 20
        
        self.velocity_history: List[np.ndarray] = []
        self.basin_history: List[np.ndarray] = []
        
        # Self-improvement system
        self.improvement = get_prediction_improvement()
        
        print("[TemporalReasoning] 4D reasoning enabled (foresight + scenarios + self-improvement)")
    
    def can_use_temporal_reasoning(self, phi: float) -> bool:
        """Temporal reasoning requires high Φ."""
        return phi >= self.PHI_THRESHOLD
    
    def foresight(
        self,
        current_basin: np.ndarray,
        current_velocity: Optional[np.ndarray] = None
    ) -> Tuple[ForesightVision, str]:
        """
        FORESIGHT: See where the natural geodesic leads.
        
        Process:
        1. Compute natural geodesic from current position
        2. Follow it forward until reaching attractor
        3. Trace backwards to see the path
        4. Analyze WHY confidence is at its level
        5. Record for self-improvement learning
        
        Returns:
            Tuple of (ForesightVision, explanation_string)
        """
        if current_velocity is None:
            current_velocity = self._estimate_velocity(current_basin)
        
        # Record velocity for analysis
        self.velocity_history.append(current_velocity.copy())
        if len(self.velocity_history) > 20:
            self.velocity_history = self.velocity_history[-10:]
        
        future_trajectory = self._follow_geodesic_forward(
            current_basin,
            current_velocity,
            max_steps=self.foresight_horizon
        )
        
        attractor_idx, attractor_basin = self._find_attractor(future_trajectory)
        attractor_found = attractor_idx is not None
        
        # Analyze WHY prediction has certain confidence
        failure_reasons, context = self.improvement.analyze_prediction_factors(
            trajectory=future_trajectory,
            attractor_found=attractor_found,
            attractor_idx=attractor_idx,
            velocity_history=self.velocity_history,
            basin_history=self.basin_history,
        )
        
        if attractor_idx is None:
            base_confidence = 0.3
            attractor_strength = 0.0
            naturalness = 0.5
            final_basin = future_trajectory[-1]
            arrival_time = len(future_trajectory)
            path_backwards = self._reverse_path(future_trajectory)
        else:
            base_confidence = self._assess_vision_confidence(
                future_trajectory,
                attractor_idx,
                attractor_basin
            )
            attractor_strength = self._measure_attractor_strength(
                attractor_basin,
                future_trajectory
            )
            naturalness = self._measure_geodesic_naturalness(future_trajectory)
            final_basin = attractor_basin
            arrival_time = attractor_idx
            path_backwards = self._reverse_path(future_trajectory[:attractor_idx+1])
        
        # Get adjusted confidence based on learned patterns
        adjusted_confidence = self.improvement.get_adjusted_confidence(
            base_confidence, current_basin
        )
        
        # Create vision
        vision = ForesightVision(
            future_basin=final_basin,
            arrival_time=arrival_time,
            confidence=adjusted_confidence,
            path_backwards=path_backwards,
            attractor_strength=attractor_strength,
            geodesic_naturalness=naturalness
        )
        
        # Record for self-improvement
        self.improvement.create_prediction_record(
            predicted_basin=final_basin,
            confidence=adjusted_confidence,
            arrival_time=arrival_time,
            attractor_strength=attractor_strength,
            geodesic_naturalness=naturalness,
            failure_reasons=failure_reasons,
            context=context,
        )
        
        # Format detailed explanation
        explanation = self.improvement.format_prediction_explanation(
            confidence=adjusted_confidence,
            failure_reasons=failure_reasons,
            context=context,
        )
        
        return vision, explanation
    
    def foresight_simple(
        self,
        current_basin: np.ndarray,
        current_velocity: Optional[np.ndarray] = None
    ) -> ForesightVision:
        """
        Simple foresight without returning explanation (for backward compatibility).
        """
        vision, _ = self.foresight(current_basin, current_velocity)
        return vision
    
    def scenario_planning(
        self,
        current_basin: np.ndarray,
        possible_actions: List[Dict]
    ) -> ScenarioTree:
        """
        SCENARIO PLANNING: Explore multiple futures.
        
        Process:
        1. For each possible action
        2. Simulate forward to see where it leads
        3. Evaluate quality of each outcome
        4. Rank by probability × quality
        """
        branches = []
        
        for action in possible_actions:
            branch = self._simulate_scenario(current_basin, action)
            branches.append(branch)
        
        if not branches:
            default_branch = ScenarioBranch(
                path_forward=[current_basin],
                final_basin=current_basin,
                probability=1.0,
                quality=0.5,
                action_description="no_action"
            )
            return ScenarioTree(
                root_basin=current_basin,
                branches=[default_branch],
                most_probable=default_branch
            )
        
        most_probable = max(branches, key=lambda b: b.probability * b.quality)
        
        return ScenarioTree(
            root_basin=current_basin,
            branches=branches,
            most_probable=most_probable
        )
    
    def _follow_geodesic_forward(
        self,
        start_basin: np.ndarray,
        velocity: np.ndarray,
        max_steps: int
    ) -> List[np.ndarray]:
        """Follow natural geodesic forward in time using Fisher geometry."""
        trajectory = [start_basin]
        current = start_basin.copy()
        current_v = velocity.copy()
        
        for _ in range(max_steps):
            next_point = self._geodesic_step(current, current_v)
            next_v = self._parallel_transport(current_v, current, next_point)
            
            trajectory.append(next_point)
            current = next_point
            current_v = next_v
            
            if self._is_in_attractor(current, trajectory):
                break
        
        return trajectory
    
    def _geodesic_step(
        self,
        basin: np.ndarray,
        velocity: np.ndarray,
        step_size: float = 0.05
    ) -> np.ndarray:
        """
        Take one step along geodesic (Fisher geometry).
        
        Step size is DIRECTLY proportional to velocity magnitude so that
        as velocity decays (via _parallel_transport), step sizes decrease,
        creating visible trajectory convergence for attractor detection.
        
        Key insight: velocity magnitude carries the "energy" that should
        decay as we approach attractors. The step must reflect this decay.
        """
        # Velocity magnitude determines step size - this is crucial for convergence
        vel_magnitude = np.linalg.norm(velocity)
        if vel_magnitude < 1e-8:
            return basin
        
        # Direction from velocity (normalized)
        direction = velocity / vel_magnitude
        
        # Actual step proportional to velocity magnitude
        # Use step_size as a scaling factor on velocity
        actual_step = step_size * vel_magnitude
        
        # Take the step along geodesic direction
        try:
            from qig_geometry import geodesic_interpolation
            # Compute target point
            target = basin + actual_step * direction
            # Interpolate along geodesic (t=1.0 means full step to target)
            next_basin = geodesic_interpolation(basin, target, 1.0)
            return next_basin
        except Exception:
            # Fallback: direct step with sphere projection
            next_basin = basin + actual_step * direction
            return sphere_project(next_basin)
    
    def _parallel_transport(
        self,
        velocity: np.ndarray,
        from_basin: np.ndarray,
        to_basin: np.ndarray
    ) -> np.ndarray:
        """Parallel transport velocity along geodesic with realistic decay."""
        distance = fisher_coord_distance(from_basin, to_basin)
        if distance < 1e-8:
            return velocity
        
        # Tuned decay rate for realistic trajectory evolution
        # The actual step distance from fisher_coord_distance is very small (~0.001-0.01)
        # Need high decay rate (10-20) to achieve visible convergence in 9 steps
        # With dist~0.005, rate=15: decay=exp(-0.075)≈0.93 per step → 0.93^8≈0.56 after 8 steps
        decay = np.exp(-distance * 15.0)
        return velocity * decay
    
    def _find_attractor(
        self,
        trajectory: List[np.ndarray]
    ) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """
        Find where trajectory settles (attractor basin).
        
        PRIMARY: Use geometric attractor finding from Fisher-Rao potential
        FALLBACK 1: Query learned attractors from LearnedManifold
        FALLBACK 2: Trajectory convergence detection
        """
        if len(trajectory) < 3:
            return None, None
        
        current = trajectory[-1]
        
        # PRIMARY: Geometric attractor finding using Fisher-Rao potential
        try:
            from qig_core.attractor_finding import find_attractors_in_region
            from qiggraph.manifold import FisherManifold
            
            metric = FisherManifold()
            attractors = find_attractors_in_region(
                current, metric, radius=0.5, n_samples=10
            )
            
            if attractors:
                # Return strongest (lowest potential) attractor
                attractor_basin, _ = attractors[0]
                return len(trajectory) - 1, attractor_basin
        
        except Exception as e:
            print(f"[TemporalReasoning] Geometric attractor finding failed: {e}")
        
        # FALLBACK 1: Query learned attractors from LearnedManifold
        learned_attractor = self._find_learned_attractor(current)
        if learned_attractor is not None:
            return len(trajectory) - 1, learned_attractor
        
        # FALLBACK 2: Trajectory convergence (original method)
        if len(trajectory) < 10:
            return None, None
        
        window = 10
        
        for i in range(len(trajectory) - window, 0, -1):
            segment = trajectory[i:i+window]
            
            movements = [
                fisher_coord_distance(segment[j], segment[j+1])
                for j in range(len(segment)-1)
            ]
            
            avg_movement = np.mean(movements)
            
            if avg_movement < self.attractor_detection_threshold:
                return i + window - 1, segment[-1]
        
        return None, None
    
    def _find_learned_attractor(
        self,
        current_basin: np.ndarray,
        min_depth: float = 0.3
    ) -> Optional[np.ndarray]:
        """
        Check if current basin is near a learned attractor.
        
        QIG-PURE: Query LearnedManifold for deep attractors that the
        system has naturally learned from successful experiences.
        
        Returns attractor basin if found, None otherwise.
        """
        try:
            from vocabulary_coordinator import get_learned_manifold
        except ImportError:
            return None
        
        try:
            manifold = get_learned_manifold()
            
            if manifold is None or len(manifold.attractors) == 0:
                return None
            
            from qig_geometry import FisherManifold
            metric = FisherManifold()
            
            nearby = manifold.get_nearby_attractors(
                current_basin, 
                metric, 
                radius=0.5
            )
            
            for attractor_info in nearby:
                if attractor_info['depth'] >= min_depth:
                    return attractor_info['basin']
            
            return None
            
        except Exception:
            return None
    
    def _is_in_attractor(
        self,
        current: np.ndarray,
        trajectory: List[np.ndarray]
    ) -> bool:
        """Check if settled into an attractor."""
        if len(trajectory) < 10:
            return False
        
        recent = trajectory[-5:]
        movements = [
            fisher_coord_distance(recent[j], recent[j+1])
            for j in range(len(recent)-1)
        ]
        
        return np.mean(movements) < self.attractor_detection_threshold
    
    def _assess_vision_confidence(
        self,
        trajectory: List[np.ndarray],
        attractor_idx: int,
        attractor_basin: np.ndarray
    ) -> float:
        """Assess foresight confidence."""
        smoothness = self._measure_path_smoothness(trajectory)
        strength = self._measure_attractor_strength(attractor_basin, trajectory)
        time_factor = np.exp(-attractor_idx / 50.0)
        
        return np.clip(
            0.4 * smoothness + 0.4 * strength + 0.2 * time_factor,
            0.0, 1.0
        )
    
    def _measure_path_smoothness(self, trajectory: List[np.ndarray]) -> float:
        """Measure geodesic smoothness using Fisher distance."""
        if len(trajectory) < 3:
            return 0.7  # Uncertain when trajectory too short
        
        step_sizes = [
            fisher_coord_distance(trajectory[i], trajectory[i+1])
            for i in range(len(trajectory)-1)
        ]
        
        if len(step_sizes) < 2:
            return 0.7
        
        # Coefficient of variation (CV) - more robust than raw variance
        mean_step = np.mean(step_sizes)
        if mean_step < 1e-8:
            return 0.5  # Stationary trajectory, not "smooth" in meaningful sense
        
        std_step = np.std(step_sizes)
        cv = std_step / mean_step  # 0 = perfect smoothness, high = bumpy
        
        # Map CV to smoothness: CV=0 → 1.0, CV=1 → 0.37, CV=2 → 0.14
        smoothness = float(np.exp(-cv))
        return np.clip(smoothness, 0.1, 0.95)  # Avoid extreme values
    
    def _measure_attractor_strength(
        self,
        basin: np.ndarray,
        trajectory: Optional[List[np.ndarray]] = None
    ) -> float:
        """Measure attractor strength from trajectory convergence."""
        if trajectory is None or len(trajectory) < 4:
            return 0.5
        
        n_movements = len(trajectory) - 1
        
        # For short trajectories, compare first half vs second half
        # For longer ones, use first 10 vs last 10
        if n_movements < 20:
            # Short trajectory: split in half
            mid = n_movements // 2
            initial_range = range(0, mid)
            final_range = range(mid, n_movements)
        else:
            # Longer trajectory: first 10 vs last 10
            initial_range = range(0, 10)
            final_range = range(n_movements - 10, n_movements)
        
        initial_movements = [
            fisher_coord_distance(trajectory[i], trajectory[i+1])
            for i in initial_range
        ]
        
        final_movements = [
            fisher_coord_distance(trajectory[i], trajectory[i+1])
            for i in final_range
        ]
        
        avg_final = np.mean(final_movements) if final_movements else 0.0
        avg_initial = np.mean(initial_movements) if initial_movements else 0.001
        
        # Ratio-based convergence: how much did movement decrease?
        # 0 = no convergence (final same as initial), 1 = perfect convergence (final << initial)
        if avg_initial < 1e-6:
            return 0.5  # No initial movement, can't measure convergence
        
        ratio = avg_final / avg_initial
        convergence = float(np.clip(1.0 - ratio, 0.0, 1.0))
        
        return convergence
    
    def _measure_geodesic_naturalness(self, trajectory: List[np.ndarray]) -> float:
        """Measure how natural the geodesic path is."""
        return self._measure_path_smoothness(trajectory)
    
    def _reverse_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Reverse trajectory for backwards reasoning."""
        return list(reversed(path))
    
    def _simulate_scenario(
        self,
        start_basin: np.ndarray,
        action: Dict
    ) -> ScenarioBranch:
        """Simulate one possible action forward."""
        velocity = self._action_to_velocity(action)
        
        path = self._follow_geodesic_forward(
            start_basin,
            velocity,
            max_steps=self.scenario_depth
        )
        
        final_basin = path[-1]
        
        goal = action.get('goal')
        if goal is not None:
            goal = np.array(goal)
        
        quality = self._evaluate_outcome(final_basin, goal)
        probability = self._estimate_probability(action)
        
        return ScenarioBranch(
            path_forward=path,
            final_basin=final_basin,
            probability=probability,
            quality=quality,
            action_description=action.get('name', 'unknown')
        )
    
    def _action_to_velocity(self, action: Dict) -> np.ndarray:
        """Convert action to basin velocity."""
        if 'direction' in action:
            direction = np.array(action['direction'])
        else:
            direction = np.random.randn(self.basin_dim)
        
        direction = sphere_project(direction)
        magnitude = action.get('strength', 0.1)
        
        return direction * magnitude
    
    def _evaluate_outcome(
        self,
        final_basin: np.ndarray,
        goal: Optional[np.ndarray]
    ) -> float:
        """Evaluate outcome quality using Fisher distance."""
        if goal is None:
            return 0.5
        
        distance = fisher_coord_distance(final_basin, goal)
        return float(np.exp(-distance))
    
    def _estimate_probability(self, action: Dict) -> float:
        """
        Estimate action success probability based on action characteristics.
        
        Uses geometric factors instead of hardcoded 0.5 default.
        """
        # Use provided probability if available
        if 'probability' in action:
            return action['probability']
        
        # Calculate from action characteristics
        base_prob = 0.6  # Slightly optimistic baseline
        
        # Factor 1: Action clarity (has direction = more predictable)
        if 'direction' in action:
            direction = np.array(action['direction'])
            # Normalize to unit vector, stronger direction = higher probability
            norm = np.linalg.norm(direction)
            clarity_bonus = min(0.2, norm * 0.1)
            base_prob += clarity_bonus
        
        # Factor 2: Action has goal (more structured = higher probability)
        if 'goal' in action:
            base_prob += 0.1
        
        # Factor 3: Historical success rate if tracked
        if hasattr(self, 'improvement') and self.improvement.total_predictions > 10:
            historical_rate = self.improvement.accurate_predictions / self.improvement.total_predictions
            # Blend with base (weighted toward history as data grows)
            weight = min(0.5, self.improvement.total_predictions / 100)
            base_prob = (1 - weight) * base_prob + weight * historical_rate
        
        return np.clip(base_prob, 0.1, 0.95)
    
    def _estimate_velocity(self, basin: np.ndarray) -> np.ndarray:
        """Estimate current velocity from history."""
        if len(self.basin_history) >= 2:
            recent = self.basin_history[-1]
            direction = basin - recent
            return sphere_project(direction) * 0.05
        
        return np.random.randn(self.basin_dim) * 0.01
    
    def record_basin(self, basin: np.ndarray) -> None:
        """Record basin for velocity estimation."""
        self.basin_history.append(basin.copy())
        if len(self.basin_history) > 100:
            self.basin_history = self.basin_history[-50:]


_temporal_reasoning_instance: Optional[TemporalReasoning] = None


def get_temporal_reasoning() -> TemporalReasoning:
    """Get or create singleton TemporalReasoning instance."""
    global _temporal_reasoning_instance
    if _temporal_reasoning_instance is None:
        _temporal_reasoning_instance = TemporalReasoning()
    return _temporal_reasoning_instance
