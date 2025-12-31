"""
Autonomous Reasoning Learner

Self-learning system that discovers and refines reasoning strategies
through experience. Strategies are selected, executed, and updated
based on success/failure outcomes.

QIG-PURE: All geometric operations use Fisher-Rao distance exclusively.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
import uuid

from qig_geometry import fisher_rao_distance, sphere_project, geodesic_interpolation


@dataclass
class ReasoningStrategy:
    """
    A learnable reasoning strategy with geometric parameters.
    
    Strategies are defined by their preferred Φ range and exploration parameters.
    They are updated through reinforcement learning based on success/failure.
    """
    name: str
    description: str
    preferred_phi_range: Tuple[float, float]
    step_size_alpha: float = 0.1
    exploration_beta: float = 0.2
    task_features: Optional[Dict[str, float]] = None
    success_count: int = 0
    failure_count: int = 0
    avg_efficiency: float = 0.5
    created_at: float = field(default_factory=time.time)
    
    def success_rate(self) -> float:
        """Calculate success rate of this strategy."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total
    
    def copy(self) -> 'ReasoningStrategy':
        """Create a copy of this strategy."""
        return ReasoningStrategy(
            name=f"{self.name}_copy",
            description=self.description,
            preferred_phi_range=self.preferred_phi_range,
            step_size_alpha=self.step_size_alpha,
            exploration_beta=self.exploration_beta,
            task_features=dict(self.task_features) if self.task_features else None,
            success_count=0,
            failure_count=0,
            avg_efficiency=self.avg_efficiency
        )


@dataclass
class ReasoningEpisode:
    """Record of a reasoning episode for learning."""
    strategy_name: str
    task: Dict[str, Any]
    start_basin: np.ndarray
    target_basin: np.ndarray
    steps: List[Dict[str, Any]]
    final_basin: np.ndarray
    success: bool
    efficiency: float
    duration: float
    converged: bool = False
    timestamp: float = field(default_factory=time.time)


class AutonomousReasoningLearner:
    """
    Autonomous system that learns reasoning strategies through experience.
    
    Key capabilities:
    1. Strategy selection based on task-strategy matching
    2. Episode execution with basin trajectory tracking
    3. Learning from success/failure (reinforcement)
    4. Strategy consolidation during sleep
    5. Novel strategy generation through exploration
    
    QIG-PURE: All operations use Fisher-Rao geometry.
    """
    
    def __init__(self, basin_dim: int = 64):
        """
        Initialize the autonomous reasoning learner.
        
        Args:
            basin_dim: Dimensionality of basin coordinates
        """
        self.basin_dim = basin_dim
        self.strategies: List[ReasoningStrategy] = []
        self.reasoning_episodes: List[ReasoningEpisode] = []
        self.exploration_rate = 0.2
        self.max_episodes = 1000
        
        self._current_basin: Optional[np.ndarray] = None
        
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """Initialize with baseline strategies for each reasoning mode."""
        self.strategies = [
            ReasoningStrategy(
                name="linear_basic",
                description="Fast sequential reasoning for simple problems",
                preferred_phi_range=(0.0, 0.3),
                step_size_alpha=0.2,
                exploration_beta=0.05,
                task_features={'complexity': 0.2, 'novelty': 0.2}
            ),
            ReasoningStrategy(
                name="geometric_balanced",
                description="Balanced multi-path exploration",
                preferred_phi_range=(0.3, 0.7),
                step_size_alpha=0.15,
                exploration_beta=0.2,
                task_features={'complexity': 0.5, 'novelty': 0.5}
            ),
            ReasoningStrategy(
                name="hyperdimensional_temporal",
                description="4D temporal reasoning with timeline branching",
                preferred_phi_range=(0.75, 0.85),
                step_size_alpha=0.1,
                exploration_beta=0.3,
                task_features={'complexity': 0.8, 'novelty': 0.7}
            ),
            ReasoningStrategy(
                name="mushroom_exploration",
                description="Controlled chaos for radical novelty",
                preferred_phi_range=(0.85, 1.0),
                step_size_alpha=0.05,
                exploration_beta=0.5,
                task_features={'complexity': 0.9, 'novelty': 0.9}
            )
        ]
    
    def get_current_basin(self) -> np.ndarray:
        """Get current basin coordinates."""
        if self._current_basin is None:
            self._current_basin = np.zeros(self.basin_dim)
        return self._current_basin
    
    def set_current_basin(self, basin: np.ndarray):
        """Set current basin coordinates."""
        self._current_basin = basin
    
    def identify_target_basin(self, task: Dict) -> np.ndarray:
        """
        Identify target basin for a task using QIG-pure encoding.
        
        Encodes task description to basin coordinates on the Fisher manifold.
        """
        task_text = task.get('description', str(task))
        
        import hashlib
        content_hash = hashlib.sha256(task_text.encode()).digest()
        
        target = np.zeros(self.basin_dim, dtype=np.float64)
        for i in range(min(self.basin_dim, len(content_hash))):
            target[i] = (content_hash[i] - 128) / 256.0
        
        target = sphere_project(target)
        return target
    
    def measure_phi_at_basin(self, basin: np.ndarray) -> float:
        """
        Estimate Φ at a basin location using Fisher-Rao distance.
        
        QIG-PURE: Uses Fisher-Rao distance from origin.
        """
        origin = np.zeros_like(basin)
        fr_distance = fisher_rao_distance(basin, origin)
        phi = min(fr_distance / 2.0, 1.0)
        return phi
    
    def compute_geodesic_direction(
        self,
        current: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """
        Compute direction toward target along geodesic.
        
        QIG-PURE: Uses geodesic_interpolation for Fisher-Rao direction.
        """
        next_point = geodesic_interpolation(current, target, t=0.1)
        direction = next_point - current
        
        dir_norm = fisher_rao_distance(direction, np.zeros_like(direction))
        if dir_norm > 1e-10:
            direction = direction / dir_norm
        
        return direction
    
    def project_to_manifold(self, basin: np.ndarray) -> np.ndarray:
        """Project basin back onto the Fisher manifold."""
        return sphere_project(basin)
    
    def select_strategy(self, task: Dict, current_phi: float) -> ReasoningStrategy:
        """
        Select best strategy for a task.
        
        Strategy selection considers:
        1. Φ compatibility (is current Φ in strategy's range?)
        2. Task feature matching (does strategy work for this task type?)
        3. Historical success rate
        4. Exploration (sometimes try new strategies)
        """
        if np.random.random() < self.exploration_rate:
            return self.generate_novel_strategy(task)
        
        scored_strategies = []
        
        for strategy in self.strategies:
            score = 0.0
            
            phi_low, phi_high = strategy.preferred_phi_range
            if phi_low <= current_phi <= phi_high:
                score += 1.0
            else:
                distance_to_range = min(
                    abs(current_phi - phi_low),
                    abs(current_phi - phi_high)
                )
                score += max(0, 1.0 - distance_to_range * 2)
            
            if strategy.task_features:
                task_features = self.extract_task_features(task)
                feature_score = 0.0
                for key, value in task_features.items():
                    strategy_value = strategy.task_features.get(key, 0.5)
                    feature_score += 1.0 - abs(value - strategy_value)
                score += feature_score / len(task_features)
            
            score += strategy.success_rate() * 0.5
            
            score += strategy.avg_efficiency * 0.3
            
            scored_strategies.append((strategy, score))
        
        scored_strategies.sort(key=lambda x: x[1], reverse=True)
        
        return scored_strategies[0][0]
    
    def generate_novel_strategy(self, task: Dict) -> ReasoningStrategy:
        """
        Create a new reasoning strategy through exploration.
        
        Samples parameters from prior distribution.
        """
        phi_center = np.random.uniform(0.3, 0.8)
        phi_width = np.random.uniform(0.1, 0.3)
        
        strategy = ReasoningStrategy(
            name=f"novel_{uuid.uuid4().hex[:8]}",
            description="Autonomously discovered strategy",
            preferred_phi_range=(
                max(0, phi_center - phi_width),
                min(1.0, phi_center + phi_width)
            ),
            step_size_alpha=np.random.uniform(0.05, 0.3),
            exploration_beta=np.random.uniform(0.1, 0.5),
            task_features=self.extract_task_features(task)
        )
        
        self.strategies.append(strategy)
        return strategy
    
    def execute_strategy(
        self,
        strategy: ReasoningStrategy,
        task: Dict,
        max_steps: int = 20
    ) -> ReasoningEpisode:
        """
        Execute a reasoning strategy and record results.
        
        Tracks basin trajectory and computes geometric efficiency.
        """
        start_time = time.time()
        
        current_basin = self.get_current_basin().copy()
        target_basin = self.identify_target_basin(task)
        
        steps = []
        converged = False
        
        for step_num in range(max_steps):
            direction = self.compute_geodesic_direction(current_basin, target_basin)
            
            step_vector = strategy.step_size_alpha * direction
            
            if np.random.random() < strategy.exploration_beta:
                noise = np.random.randn(*step_vector.shape) * 0.1
                step_vector = step_vector + noise
            
            next_basin = current_basin + step_vector
            next_basin = self.project_to_manifold(next_basin)
            
            distance_to_target = fisher_rao_distance(next_basin, target_basin)
            
            steps.append({
                'step': step_num,
                'basin': next_basin.copy(),
                'distance_to_target': distance_to_target
            })
            
            if distance_to_target < 0.1:
                converged = True
                break
            
            current_basin = next_basin
        
        duration = time.time() - start_time
        final_basin = current_basin
        
        if len(steps) > 1:
            actual_distance = sum(
                fisher_rao_distance(
                    steps[i]['basin'],
                    steps[i + 1]['basin']
                )
                for i in range(len(steps) - 1)
            )
        else:
            actual_distance = 0.0
        
        optimal_distance = fisher_rao_distance(
            self.get_current_basin(),
            target_basin
        )
        
        efficiency = optimal_distance / (actual_distance + 1e-10)
        efficiency = min(efficiency, 1.0)
        
        success = converged or (len(steps) > 0 and steps[-1]['distance_to_target'] < 0.5)
        
        episode = ReasoningEpisode(
            strategy_name=strategy.name,
            task=task,
            start_basin=self.get_current_basin().copy(),
            target_basin=target_basin,
            steps=steps,
            final_basin=final_basin,
            success=success,
            efficiency=efficiency,
            duration=duration,
            converged=converged
        )
        
        self.set_current_basin(final_basin)
        
        return episode
    
    def learn_from_episode(self, episode: ReasoningEpisode):
        """
        Update strategy based on episode results.
        
        Reinforcement learning:
        - Success → strengthen strategy
        - Failure → weaken strategy
        - High efficiency → increase step size
        - Low efficiency → decrease step size
        """
        strategy = next(
            (s for s in self.strategies if s.name == episode.strategy_name),
            None
        )
        
        if strategy is None:
            strategy = ReasoningStrategy(
                name=episode.strategy_name,
                description="Learned from experience",
                preferred_phi_range=(0.5, 0.7),
                step_size_alpha=0.1,
                exploration_beta=0.2
            )
            self.strategies.append(strategy)
        
        if episode.success:
            strategy.success_count += 1
            
            if episode.steps:
                avg_step_distance = np.mean([
                    fisher_rao_distance(
                        step['basin'],
                        episode.start_basin
                    )
                    for step in episode.steps
                ])
                strategy.step_size_alpha = (
                    0.9 * strategy.step_size_alpha +
                    0.1 * min(avg_step_distance, 0.5)
                )
                
                avg_phi = np.mean([
                    self.measure_phi_at_basin(step['basin'])
                    for step in episode.steps
                ])
                
                phi_center = np.mean(strategy.preferred_phi_range)
                phi_width = strategy.preferred_phi_range[1] - strategy.preferred_phi_range[0]
                
                new_phi_center = 0.9 * phi_center + 0.1 * avg_phi
                strategy.preferred_phi_range = (
                    max(0, new_phi_center - phi_width / 2),
                    min(1.0, new_phi_center + phi_width / 2)
                )
                
                task_features = self.extract_task_features(episode.task)
                if strategy.task_features is None:
                    strategy.task_features = task_features
                else:
                    for key in task_features:
                        strategy.task_features[key] = (
                            0.9 * strategy.task_features.get(key, 0.5) +
                            0.1 * task_features[key]
                        )
        else:
            strategy.failure_count += 1
        
        total_episodes = strategy.success_count + strategy.failure_count
        if total_episodes > 0:
            strategy.avg_efficiency = (
                (total_episodes - 1) * strategy.avg_efficiency + episode.efficiency
            ) / total_episodes
        
        self.reasoning_episodes.append(episode)
        
        if len(self.reasoning_episodes) > self.max_episodes:
            self.reasoning_episodes = self.reasoning_episodes[-self.max_episodes:]
    
    def consolidate_strategies(self):
        """
        Called during sleep: consolidate successful strategies.
        
        Actions:
        1. Prune failed strategies
        2. Merge similar strategies
        3. Strengthen successful patterns
        """
        min_success_rate = 0.2
        min_episodes = 5
        
        pruned = []
        pruned_count = 0
        for strategy in self.strategies:
            total = strategy.success_count + strategy.failure_count
            if total >= min_episodes and strategy.success_rate() < min_success_rate:
                pruned_count += 1
                continue
            pruned.append(strategy)
        
        self.strategies = pruned
        
        merged = []
        used = set()
        
        for i, strategy_a in enumerate(self.strategies):
            if i in used:
                continue
            
            similar_strategies = [strategy_a]
            
            for j, strategy_b in enumerate(self.strategies[i + 1:], start=i + 1):
                if j in used:
                    continue
                
                if self._strategies_similar(strategy_a, strategy_b):
                    similar_strategies.append(strategy_b)
                    used.add(j)
            
            if len(similar_strategies) > 1:
                merged_strategy = self._merge_strategies(similar_strategies)
                merged.append(merged_strategy)
            else:
                merged.append(strategy_a)
        
        self.strategies = merged
        
        # Reduced logging - only log when merges happen
        # logger.debug(f"Strategy consolidation complete: {len(self.strategies)} strategies")
    
    def _strategies_similar(
        self,
        strategy_a: ReasoningStrategy,
        strategy_b: ReasoningStrategy,
        threshold: float = 0.2
    ) -> bool:
        """Are two strategies similar enough to merge?"""
        param_distance = np.sqrt(
            (strategy_a.step_size_alpha - strategy_b.step_size_alpha) ** 2 +
            (strategy_a.exploration_beta - strategy_b.exploration_beta) ** 2
        )
        return param_distance < threshold
    
    def _merge_strategies(
        self,
        strategies: List[ReasoningStrategy]
    ) -> ReasoningStrategy:
        """Merge similar strategies into one stronger strategy."""
        weights = [s.success_rate() for s in strategies]
        total_weight = sum(weights)
        
        if total_weight == 0:
            weights = [1.0] * len(strategies)
            total_weight = len(strategies)
        
        merged = ReasoningStrategy(
            name=f"merged_{strategies[0].name}",
            description=f"Merged from {len(strategies)} strategies",
            preferred_phi_range=(
                sum(w * s.preferred_phi_range[0] for w, s in zip(weights, strategies)) / total_weight,
                sum(w * s.preferred_phi_range[1] for w, s in zip(weights, strategies)) / total_weight
            ),
            step_size_alpha=sum(w * s.step_size_alpha for w, s in zip(weights, strategies)) / total_weight,
            exploration_beta=sum(w * s.exploration_beta for w, s in zip(weights, strategies)) / total_weight,
            success_count=sum(s.success_count for s in strategies),
            failure_count=sum(s.failure_count for s in strategies),
            avg_efficiency=sum(w * s.avg_efficiency for w, s in zip(weights, strategies)) / total_weight
        )
        
        return merged
    
    def extract_task_features(self, task: Dict) -> Dict[str, float]:
        """Extract features from task for strategy matching."""
        return {
            'complexity': task.get('complexity', 0.5),
            'novelty': task.get('novelty', 0.5),
            'time_pressure': task.get('time_pressure', 0.5),
            'precision_required': task.get('precision_required', 0.5)
        }
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get statistics about all strategies."""
        return {
            'total_strategies': len(self.strategies),
            'total_episodes': len(self.reasoning_episodes),
            'exploration_rate': self.exploration_rate,
            'strategies': [
                {
                    'name': s.name,
                    'success_rate': s.success_rate(),
                    'avg_efficiency': s.avg_efficiency,
                    'phi_range': s.preferred_phi_range,
                    'episodes': s.success_count + s.failure_count
                }
                for s in self.strategies
            ]
        }
