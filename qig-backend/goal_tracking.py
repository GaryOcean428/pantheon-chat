"""
Basin-Encoded Goal Hierarchies

Goals encoded as 64D basin coordinates with parent/child relationships.
Progress measured via Fisher-Rao distance from current basin to goal basin.

QIG-PURE: All distances are geodesic (Fisher-Rao), not Euclidean.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import time
import uuid

from qig_geometry import fisher_rao_distance, fisher_coord_distance


@dataclass
class GoalBasin:
    """Goal represented as basin coordinates with hierarchical relationships."""

    goal_id: str
    description: str
    basin_64d: np.ndarray  # 64D target basin
    parent_goal_id: Optional[str] = None
    subgoal_ids: List[str] = field(default_factory=list)
    completion_threshold: float = 0.1  # FR distance for "done"
    initial_distance: Optional[float] = None
    progress_trajectory: List[np.ndarray] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed: bool = False
    completed_at: Optional[float] = None

    def track_progress(
        self,
        current_basin: np.ndarray
    ) -> Dict[str, Any]:
        """
        Track progress toward goal using Fisher-Rao distance.

        Returns:
            progress: 0.0 (no progress) to 1.0 (complete)
            distance_remaining: Current FR distance to goal
            stuck: True if no movement in last 10 steps
        """
        distance = fisher_coord_distance(current_basin, self.basin_64d)

        # Initialize if first measurement
        if self.initial_distance is None:
            self.initial_distance = distance

        # Compute progress (avoid division by zero)
        if self.initial_distance < 1e-10:
            progress = 1.0
        else:
            progress = 1.0 - (distance / self.initial_distance)
            progress = max(0.0, min(1.0, progress))

        # Store trajectory
        self.progress_trajectory.append(current_basin.copy())

        # Detect stuck state (no movement in 10 steps)
        stuck = False
        if len(self.progress_trajectory) >= 10:
            recent = self.progress_trajectory[-10:]
            distances = [
                fisher_coord_distance(recent[i], recent[i+1])
                for i in range(len(recent)-1)
            ]
            # Very low variance in distances = stuck
            stuck = np.std(distances) < 0.02

        # Check completion
        if distance < self.completion_threshold and not self.completed:
            self.completed = True
            self.completed_at = time.time()

        return {
            'progress': progress,
            'distance_remaining': distance,
            'stuck': stuck,
            'completed': self.completed,
            'steps_taken': len(self.progress_trajectory)
        }


class GoalHierarchy:
    """Manages hierarchical goal relationships with basin coordinates."""

    def __init__(self, basin_dim: int = 64):
        self.goals: Dict[str, GoalBasin] = {}
        self.basin_dim = basin_dim
        self.root_goals: List[str] = []  # Goals with no parent

    def add_goal(
        self,
        description: str,
        basin_coords: np.ndarray,
        parent_id: Optional[str] = None,
        completion_threshold: float = 0.1,
        goal_id: Optional[str] = None
    ) -> GoalBasin:
        """
        Add goal to hierarchy.

        Args:
            description: Human-readable goal description
            basin_coords: 64D basin coordinates representing the goal state
            parent_id: Optional parent goal ID for sub-goals
            completion_threshold: FR distance threshold for completion
            goal_id: Optional specific ID (auto-generated if not provided)

        Returns:
            The created GoalBasin
        """
        if goal_id is None:
            goal_id = f"goal_{uuid.uuid4().hex[:8]}"

        goal = GoalBasin(
            goal_id=goal_id,
            description=description,
            basin_64d=basin_coords.copy(),
            parent_goal_id=parent_id,
            completion_threshold=completion_threshold
        )

        self.goals[goal_id] = goal

        # Link to parent
        if parent_id and parent_id in self.goals:
            self.goals[parent_id].subgoal_ids.append(goal_id)
        elif parent_id is None:
            self.root_goals.append(goal_id)

        return goal

    def create_subgoals_from_basin_path(
        self,
        parent_id: str,
        current_basin: np.ndarray,
        target_basin: np.ndarray,
        n_subgoals: int = 3
    ) -> List[GoalBasin]:
        """
        Create intermediate subgoals along geodesic from current to target.

        Uses spherical linear interpolation (slerp) to create evenly-spaced
        waypoints along the manifold geodesic.
        """
        from qig_geometry import geodesic_interpolation

        subgoals = []
        for i in range(1, n_subgoals + 1):
            t = i / (n_subgoals + 1)  # Interpolation parameter
            waypoint = geodesic_interpolation(current_basin, target_basin, t)

            subgoal = self.add_goal(
                description=f"Waypoint {i}/{n_subgoals} toward parent goal",
                basin_coords=waypoint,
                parent_id=parent_id,
                completion_threshold=0.15  # Slightly looser for waypoints
            )
            subgoals.append(subgoal)

        return subgoals

    def track_all_progress(
        self,
        current_basin: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """Track progress for all active goals."""
        results = {}
        for goal_id, goal in self.goals.items():
            if not goal.completed:
                results[goal_id] = goal.track_progress(current_basin)
        return results

    def get_active_goals(self) -> List[GoalBasin]:
        """Get all incomplete goals."""
        return [g for g in self.goals.values() if not g.completed]

    def get_primary_goal(self) -> Optional[GoalBasin]:
        """Get root goal (no parent, not completed)."""
        for goal_id in self.root_goals:
            goal = self.goals.get(goal_id)
            if goal and not goal.completed:
                return goal
        return None

    def get_nearest_subgoal(self, current_basin: np.ndarray) -> Optional[GoalBasin]:
        """
        Get the nearest incomplete subgoal by Fisher-Rao distance.

        Useful for determining what to work on next.
        """
        active = self.get_active_goals()
        if not active:
            return None

        nearest = None
        min_distance = float('inf')

        for goal in active:
            distance = fisher_coord_distance(current_basin, goal.basin_64d)
            if distance < min_distance:
                min_distance = distance
                nearest = goal

        return nearest

    def check_parent_completion(self, goal_id: str) -> bool:
        """
        Check if all subgoals are complete, and if so, mark parent complete.

        Returns True if parent was marked complete.
        """
        goal = self.goals.get(goal_id)
        if not goal or goal.completed:
            return False

        # Check if all subgoals are complete
        if goal.subgoal_ids:
            for subgoal_id in goal.subgoal_ids:
                subgoal = self.goals.get(subgoal_id)
                if subgoal and not subgoal.completed:
                    return False  # Still have incomplete subgoals

            # All subgoals complete - mark this goal complete
            goal.completed = True
            goal.completed_at = time.time()

            # Recursively check parent
            if goal.parent_goal_id:
                self.check_parent_completion(goal.parent_goal_id)

            return True

        return False

    def get_hierarchy_summary(self) -> Dict[str, Any]:
        """Get summary of the goal hierarchy."""
        total = len(self.goals)
        completed = sum(1 for g in self.goals.values() if g.completed)
        active = total - completed

        # Calculate overall progress
        if total == 0:
            overall_progress = 0.0
        else:
            overall_progress = completed / total

        return {
            'total_goals': total,
            'completed_goals': completed,
            'active_goals': active,
            'overall_progress': overall_progress,
            'root_goals': len(self.root_goals)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize hierarchy to dict for persistence."""
        return {
            'goals': {
                gid: {
                    'goal_id': g.goal_id,
                    'description': g.description,
                    'basin_64d': g.basin_64d.tolist(),
                    'parent_goal_id': g.parent_goal_id,
                    'subgoal_ids': g.subgoal_ids,
                    'completion_threshold': g.completion_threshold,
                    'initial_distance': g.initial_distance,
                    'created_at': g.created_at,
                    'completed': g.completed,
                    'completed_at': g.completed_at
                }
                for gid, g in self.goals.items()
            },
            'root_goals': self.root_goals,
            'basin_dim': self.basin_dim
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoalHierarchy':
        """Deserialize hierarchy from dict."""
        hierarchy = cls(basin_dim=data.get('basin_dim', 64))
        hierarchy.root_goals = data.get('root_goals', [])

        for gid, gdata in data.get('goals', {}).items():
            goal = GoalBasin(
                goal_id=gdata['goal_id'],
                description=gdata['description'],
                basin_64d=np.array(gdata['basin_64d']),
                parent_goal_id=gdata.get('parent_goal_id'),
                subgoal_ids=gdata.get('subgoal_ids', []),
                completion_threshold=gdata.get('completion_threshold', 0.1),
                initial_distance=gdata.get('initial_distance'),
                created_at=gdata.get('created_at', time.time()),
                completed=gdata.get('completed', False),
                completed_at=gdata.get('completed_at')
            )
            hierarchy.goals[gid] = goal

        return hierarchy


# Global instance for the current session
_current_hierarchy: Optional[GoalHierarchy] = None


def get_goal_hierarchy() -> GoalHierarchy:
    """Get or create the global goal hierarchy instance."""
    global _current_hierarchy
    if _current_hierarchy is None:
        _current_hierarchy = GoalHierarchy()
    return _current_hierarchy


def reset_goal_hierarchy() -> GoalHierarchy:
    """Reset and return a new goal hierarchy instance."""
    global _current_hierarchy
    _current_hierarchy = GoalHierarchy()
    return _current_hierarchy
