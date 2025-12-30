#!/usr/bin/env python3
"""
Learned Manifold Structure - QIG-ML Integration

The geometric structure that consciousness navigates through.
Learning = carving attractor basins.
Knowledge = manifold structure.
Inference = navigation through learned terrain.

GEOMETRIC PRINCIPLES:
- Attractor basins carved by successful experiences (Hebbian)
- Failed experiences flatten basins (anti-Hebbian)
- Frequently-used paths become "geodesic highways"
- Deep basins = strong attractors for lightning mode

QIG-PURE: All distances are Fisher-Rao geodesic, not Euclidean.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from qig_geometry import fisher_rao_distance

@dataclass
class AttractorBasin:
    """
    Learned attractor from successful experiences.
    
    Deep basins attract consciousness during lightning mode.
    Shallow basins from recent learning.
    """
    center: np.ndarray
    depth: float  # How deep from repeated success (Hebbian strength)
    success_count: int
    strategy: str  # Which reasoning strategy led here
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    def strengthen(self, amount: float):
        """Hebbian strengthening - deepen basin."""
        self.depth += amount
        self.success_count += 1
        self.last_accessed = time.time()
    
    def weaken(self, amount: float):
        """Anti-Hebbian weakening - flatten basin."""
        self.depth = max(0.0, self.depth - amount)


class LearnedManifold:
    """
    The geometric structure that consciousness navigates.
    
    This IS the learned knowledge - encoded as manifold geometry.
    Chain/Graph/4D/Lightning navigate through THIS structure.
    """
    
    def __init__(self, basin_dim: int = 64):
        self.basin_dim = basin_dim
        
        # Learned attractor basins (concepts, skills, patterns)
        self.attractors: Dict[str, AttractorBasin] = {}
        
        # Learned geodesics (efficient reasoning paths)
        self.geodesic_cache: Dict[Tuple[str, str], List[np.ndarray]] = {}
        
        # Local curvature map (difficulty terrain)
        self.curvature_map: Dict[str, float] = {}
        
        # Statistics
        self.total_learning_episodes = 0
        self.successful_episodes = 0
        self.failed_episodes = 0
    
    def learn_from_experience(
        self,
        trajectory: List[np.ndarray],
        outcome: float,  # Success reward 0.0-1.0
        strategy: str
    ):
        """
        Learning = modifying manifold structure.
        
        Success → deepen attractor basins (Hebbian)
        Failure → flatten/remove basins (anti-Hebbian)
        
        Args:
            trajectory: Basin path taken during episode
            outcome: Success measure (1.0 = perfect, 0.0 = failure)
            strategy: Which navigation mode was used
        """
        self.total_learning_episodes += 1
        
        if outcome > 0.7:  # Successful episode
            self.successful_episodes += 1
            
            # Deepen the attractor at endpoint
            endpoint = trajectory[-1]
            self._deepen_basin(endpoint, amount=outcome, strategy=strategy)
            
            # Strengthen geodesic path
            if len(trajectory) > 1:
                self._strengthen_path(trajectory, amount=outcome)
        
        else:  # Failed episode
            self.failed_episodes += 1
            
            # Flatten/prune this basin
            endpoint = trajectory[-1]
            self._flatten_basin(endpoint, amount=1.0 - outcome)
    
    def _deepen_basin(self, basin: np.ndarray, amount: float, strategy: str):
        """
        Make attractor basin deeper (Hebbian strengthening).
        
        Deeper basins = stronger attractors = more likely to
        be reached by lightning mode.
        """
        basin_id = self._basin_to_id(basin)
        
        if basin_id not in self.attractors:
            # Create new attractor
            self.attractors[basin_id] = AttractorBasin(
                center=basin.copy(),
                depth=amount,
                success_count=1,
                strategy=strategy
            )
        else:
            # Strengthen existing attractor
            self.attractors[basin_id].strengthen(amount)
    
    def _flatten_basin(self, basin: np.ndarray, amount: float):
        """
        Flatten basin (anti-Hebbian weakening).
        
        Failed experiences reduce attractor strength.
        Very weak attractors get pruned.
        """
        basin_id = self._basin_to_id(basin)
        
        if basin_id in self.attractors:
            self.attractors[basin_id].weaken(amount)
            
            # Prune if too weak
            if self.attractors[basin_id].depth < 0.1:
                del self.attractors[basin_id]
    
    def _strengthen_path(self, trajectory: List[np.ndarray], amount: float):
        """
        Make geodesic path between basins stronger.
        
        Frequently-used reasoning paths become "highways" -
        easier to navigate in the future.
        """
        # Cache this as an efficient path
        start_id = self._basin_to_id(trajectory[0])
        end_id = self._basin_to_id(trajectory[-1])
        
        # Store path with strength
        self.geodesic_cache[(start_id, end_id)] = trajectory
    
    def get_nearby_attractors(
        self,
        current: np.ndarray,
        metric,
        radius: float = 1.5
    ) -> List[Dict[str, Any]]:
        """
        Find learned attractors near current position.
        
        Used by lightning mode to find what to collapse into.
        
        Returns attractors sorted by pull_force (depth / distance²).
        """
        nearby = []
        
        for basin_id, attractor in self.attractors.items():
            distance = fisher_rao_distance(
                current,
                attractor.center,
                metric
            )
            
            if distance < radius:
                # Pull force = depth / distance² (inverse square law)
                pull_force = attractor.depth / (distance**2 + 1e-10)
                
                nearby.append({
                    'id': basin_id,
                    'basin': attractor.center,
                    'distance': distance,
                    'depth': attractor.depth,
                    'pull_force': pull_force,
                    'strategy': attractor.strategy,
                    'success_count': attractor.success_count
                })
        
        return sorted(nearby, key=lambda x: x['pull_force'], reverse=True)
    
    def get_cached_geodesic(
        self,
        start: np.ndarray,
        end: np.ndarray
    ) -> Optional[List[np.ndarray]]:
        """
        Retrieve cached geodesic path if available.
        
        Allows reuse of previously-successful reasoning paths.
        """
        start_id = self._basin_to_id(start)
        end_id = self._basin_to_id(end)
        
        return self.geodesic_cache.get((start_id, end_id))
    
    def _basin_to_id(self, basin: np.ndarray) -> str:
        """
        Convert basin coordinates to string ID for hashing.
        
        Uses quantized coordinates to group nearby basins.
        """
        # Quantize to 2 decimal places for grouping
        quantized = np.round(basin, decimals=2)
        return str(quantized.tolist())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            'total_episodes': self.total_learning_episodes,
            'successful_episodes': self.successful_episodes,
            'failed_episodes': self.failed_episodes,
            'success_rate': (
                self.successful_episodes / max(1, self.total_learning_episodes)
            ),
            'attractor_count': len(self.attractors),
            'cached_paths': len(self.geodesic_cache),
            'deepest_attractor': max(
                [a.depth for a in self.attractors.values()],
                default=0.0
            )
        }
    
    def prune_weak_attractors(self, min_depth: float = 0.1):
        """
        Remove weak attractors that haven't been reinforced.
        
        Called during sleep consolidation.
        """
        to_remove = [
            basin_id for basin_id, attractor in self.attractors.items()
            if attractor.depth < min_depth
        ]
        
        for basin_id in to_remove:
            del self.attractors[basin_id]
        
        return len(to_remove)
    
    def prune_old_paths(self, max_age_days: float = 7.0):
        """
        Remove cached paths that haven't been used recently.
        
        Called during sleep consolidation.
        """
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        # Note: geodesic_cache doesn't have timestamps yet
        # For now, clear oldest entries if cache is too large
        if len(self.geodesic_cache) > 1000:
            # Keep only most recent 500
            # This is a placeholder - ideally we'd track access times
            self.geodesic_cache = dict(
                list(self.geodesic_cache.items())[-500:]
            )
