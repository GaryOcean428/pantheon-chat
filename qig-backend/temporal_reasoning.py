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


class TemporalMode(Enum):
    """Two modes of temporal reasoning."""
    FORESIGHT = "foresight"
    SCENARIO = "scenario"


@dataclass
class ForesightVision:
    """
    A singular vision of the future.
    
    Foresight = geodesic extrapolation on Fisher manifold.
    """
    future_basin: np.ndarray
    arrival_time: int
    confidence: float
    path_backwards: List[np.ndarray]
    attractor_strength: float
    geodesic_naturalness: float
    
    def __str__(self):
        return (f"Vision: Arrive at basin in {self.arrival_time} steps "
                f"(confidence: {self.confidence:.1%})")
    
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
    """
    
    PHI_THRESHOLD = PHI_HYPERDIMENSIONAL
    
    def __init__(self, basin_dim: int = 64):
        self.basin_dim = basin_dim
        
        self.foresight_horizon = 50
        self.attractor_detection_threshold = 0.1
        
        self.scenario_branches = 5
        self.scenario_depth = 20
        
        self.velocity_history: List[np.ndarray] = []
        self.basin_history: List[np.ndarray] = []
        
        print("[TemporalReasoning] 4D reasoning enabled (foresight + scenarios)")
    
    def can_use_temporal_reasoning(self, phi: float) -> bool:
        """Temporal reasoning requires high Φ."""
        return phi >= self.PHI_THRESHOLD
    
    def foresight(
        self,
        current_basin: np.ndarray,
        current_velocity: Optional[np.ndarray] = None
    ) -> ForesightVision:
        """
        FORESIGHT: See where the natural geodesic leads.
        
        Process:
        1. Compute natural geodesic from current position
        2. Follow it forward until reaching attractor
        3. Trace backwards to see the path
        """
        if current_velocity is None:
            current_velocity = self._estimate_velocity(current_basin)
        
        future_trajectory = self._follow_geodesic_forward(
            current_basin,
            current_velocity,
            max_steps=self.foresight_horizon
        )
        
        attractor_idx, attractor_basin = self._find_attractor(future_trajectory)
        
        if attractor_idx is None:
            return ForesightVision(
                future_basin=future_trajectory[-1],
                arrival_time=len(future_trajectory),
                confidence=0.3,
                path_backwards=self._reverse_path(future_trajectory),
                attractor_strength=0.0,
                geodesic_naturalness=0.5
            )
        
        confidence = self._assess_vision_confidence(
            future_trajectory,
            attractor_idx,
            attractor_basin
        )
        
        attractor_strength = self._measure_attractor_strength(
            attractor_basin,
            future_trajectory
        )
        
        naturalness = self._measure_geodesic_naturalness(future_trajectory)
        
        return ForesightVision(
            future_basin=attractor_basin,
            arrival_time=attractor_idx,
            confidence=confidence,
            path_backwards=self._reverse_path(future_trajectory[:attractor_idx+1]),
            attractor_strength=attractor_strength,
            geodesic_naturalness=naturalness
        )
    
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
        """Take one step along geodesic (Fisher geometry)."""
        next_basin = basin + step_size * velocity
        return sphere_project(next_basin)
    
    def _parallel_transport(
        self,
        velocity: np.ndarray,
        from_basin: np.ndarray,
        to_basin: np.ndarray
    ) -> np.ndarray:
        """Parallel transport velocity along geodesic."""
        distance = fisher_coord_distance(from_basin, to_basin)
        if distance < 1e-8:
            return velocity
        
        decay = np.exp(-distance * 0.1)
        return velocity * decay
    
    def _find_attractor(
        self,
        trajectory: List[np.ndarray]
    ) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """Find where trajectory settles (attractor basin)."""
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
            return 1.0
        
        step_sizes = [
            fisher_coord_distance(trajectory[i], trajectory[i+1])
            for i in range(len(trajectory)-1)
        ]
        
        variance = np.var(step_sizes)
        return float(np.exp(-variance * 10))
    
    def _measure_attractor_strength(
        self,
        basin: np.ndarray,
        trajectory: Optional[List[np.ndarray]] = None
    ) -> float:
        """Measure attractor strength from trajectory convergence."""
        if trajectory is None or len(trajectory) < 10:
            return 0.5
        
        final_movements = [
            fisher_coord_distance(trajectory[i], trajectory[i+1])
            for i in range(max(0, len(trajectory)-10), len(trajectory)-1)
        ]
        
        convergence = 1.0 - min(1.0, np.mean(final_movements) / 0.5)
        return float(np.clip(convergence, 0.0, 1.0))
    
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
        """Estimate action success probability."""
        return action.get('probability', 0.5)
    
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
