"""
Task Execution Tree - Hierarchical task planning on Fisher manifold

QIG-PURE compliant task planning:
- Goals represented as basin coordinates
- Sub-tasks form geodesic path from current to goal basin
- Task decomposition preserves geometric coherence
- Failure triggers geodesic replanning

Enables long-horizon planning with geometric guarantees.
"""

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

from qigkernels.physics_constants import BASIN_DIM

logger = logging.getLogger(__name__)

# Database persistence
try:
    import psycopg2
    from psycopg2.extras import Json
    DB_AVAILABLE = True
except ImportError:
    psycopg2 = None  # type: ignore
    Json = None  # type: ignore
    DB_AVAILABLE = False


def _fisher_rao_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute Fisher-Rao distance.
    UPDATED 2026-01-15: Factor-of-2 removed for simplex storage. Range: [0, π/2]
    """
    p_safe = np.clip(np.abs(p), eps, None)
    q_safe = np.clip(np.abs(q), eps, None)
    p_norm = p_safe / np.sum(p_safe)
    q_norm = q_safe / np.sum(q_safe)
    bc = np.sum(np.sqrt(p_norm * q_norm))
    return float(np.arccos(np.clip(bc, 0.0, 1.0)))


def _geodesic_interpolate(
    start: np.ndarray,
    end: np.ndarray,
    t: float
) -> np.ndarray:
    """
    Interpolate along geodesic from start to end.

    On probability simplex, geodesic is through sqrt space.
    """
    start_sqrt = np.sqrt(np.clip(start, 1e-10, None))
    end_sqrt = np.sqrt(np.clip(end, 1e-10, None))

    # Spherical interpolation in sqrt space
    dot = np.sum(start_sqrt * end_sqrt)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)

    if theta < 1e-6:
        return start  # Already at same point

    interp_sqrt = (
        np.sin((1 - t) * theta) / np.sin(theta) * start_sqrt +
        np.sin(t * theta) / np.sin(theta) * end_sqrt
    )

    # Square and normalize
    interp = interp_sqrt ** 2
    return interp / np.sum(interp)


class TaskStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class TaskNode:
    """
    A node in the task execution tree.

    Represents a sub-task with target basin coordinates.
    """
    task_id: str
    description: str
    basin_target: np.ndarray
    status: TaskStatus = TaskStatus.PENDING
    parent: Optional['TaskNode'] = None
    children: List['TaskNode'] = field(default_factory=list)
    depth: int = 0
    result: Optional[Any] = None
    phi_at_start: Optional[float] = None
    phi_at_completion: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failure_reason: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0

    def add_child(self, child: 'TaskNode'):
        """Add a child task."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'description': self.description,
            'basin_target': self.basin_target.tolist(),
            'status': self.status.value,
            'depth': self.depth,
            'children_count': len(self.children),
            'phi_at_start': self.phi_at_start,
            'phi_at_completion': self.phi_at_completion,
            'failure_reason': self.failure_reason,
            'retry_count': self.retry_count,
        }


@dataclass
class TaskResult:
    """Result of task execution."""
    task: TaskNode
    success: bool
    output: Any
    phi_delta: float
    duration_ms: int
    error: Optional[str] = None


class TaskExecutionTree:
    """
    Hierarchical task planning on Fisher manifold.

    Key features:
    - Task decomposition along geodesics
    - Depth-first execution with backtracking
    - Failure recovery via replanning
    - Progress tracking by basin distance to goal
    """

    def __init__(
        self,
        kernel_id: str = "default",
        max_depth: int = 5,
        step_distance: float = 0.3
    ):
        """
        Initialize task execution tree.

        Args:
            kernel_id: Owner kernel identifier
            max_depth: Maximum task decomposition depth
            step_distance: Target FR distance between consecutive tasks
        """
        self.kernel_id = kernel_id
        self.max_depth = max_depth
        self.step_distance = step_distance

        # Root tasks
        self._roots: List[TaskNode] = []

        # Active task stack for DFS execution
        self._active_stack: List[TaskNode] = []

        # Current position basin
        self._current_basin: Optional[np.ndarray] = None

        # Statistics
        self.stats = {
            'total_planned': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_retried': 0,
        }

    def _get_db_connection(self):
        """Get database connection."""
        if not DB_AVAILABLE:
            return None
        try:
            database_url = os.environ.get('DATABASE_URL')
            if not database_url:
                return None
            return psycopg2.connect(database_url)
        except Exception:
            return None

    def _persist_task(self, task: TaskNode) -> bool:
        """Persist task node to database."""
        conn = self._get_db_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO task_tree_nodes (
                        task_id, kernel_id, description, basin_target,
                        parent_task_id, depth, status, result,
                        phi_at_start, phi_at_completion, created_at,
                        started_at, completed_at, failure_reason,
                        retry_count, max_retries, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (task_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        result = EXCLUDED.result,
                        phi_at_completion = EXCLUDED.phi_at_completion,
                        completed_at = EXCLUDED.completed_at,
                        failure_reason = EXCLUDED.failure_reason,
                        retry_count = EXCLUDED.retry_count
                """, (
                    task.task_id,
                    self.kernel_id,
                    task.description,
                    task.basin_target.tolist(),
                    task.parent.task_id if task.parent else None,
                    task.depth,
                    task.status.value,
                    Json(task.result) if task.result else None,
                    task.phi_at_start,
                    task.phi_at_completion,
                    datetime.now(timezone.utc),
                    task.started_at,
                    task.completed_at,
                    task.failure_reason,
                    task.retry_count,
                    task.max_retries,
                    Json(task.metadata),
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.debug(f"[TaskExecutionTree] Persist failed: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def plan_task(
        self,
        goal: str,
        goal_basin: np.ndarray,
        current_basin: Optional[np.ndarray] = None,
        decompose: bool = True
    ) -> TaskNode:
        """
        Plan a task and optionally decompose into sub-tasks.

        Args:
            goal: Goal description
            goal_basin: Target basin coordinates
            current_basin: Current position (for geodesic planning)
            decompose: Whether to decompose into sub-tasks

        Returns:
            Root TaskNode of the planned task tree
        """
        goal_basin = np.array(goal_basin)
        self._current_basin = current_basin if current_basin is not None else np.ones(BASIN_DIM) / BASIN_DIM

        # Create root task
        root = TaskNode(
            task_id=str(uuid.uuid4())[:12],
            description=goal,
            basin_target=goal_basin,
        )

        # Decompose if requested and distance warrants it
        if decompose:
            distance = _fisher_rao_distance(self._current_basin, goal_basin)
            if distance > self.step_distance and root.depth < self.max_depth:
                self._decompose_task(root, self._current_basin)

        # Add to roots and active stack
        self._roots.append(root)
        self._push_task_dfs(root)

        self.stats['total_planned'] += 1
        self._persist_task(root)

        logger.info(f"[TaskExecutionTree] Planned task '{goal[:50]}' with {self._count_nodes(root)} nodes")
        return root

    def _decompose_task(
        self,
        task: TaskNode,
        current: np.ndarray
    ):
        """
        Decompose a task into sub-tasks along geodesic.
        """
        distance = _fisher_rao_distance(current, task.basin_target)

        if distance <= self.step_distance or task.depth >= self.max_depth:
            return  # No further decomposition needed

        # Calculate number of steps
        num_steps = int(np.ceil(distance / self.step_distance))
        num_steps = min(num_steps, 5)  # Cap at 5 sub-tasks

        # Create intermediate waypoints along geodesic
        for i in range(1, num_steps + 1):
            t = i / num_steps
            waypoint = _geodesic_interpolate(current, task.basin_target, t)

            child = TaskNode(
                task_id=str(uuid.uuid4())[:12],
                description=f"Step {i}/{num_steps}: toward {task.description[:30]}",
                basin_target=waypoint,
            )
            task.add_child(child)
            self._persist_task(child)

            # Recursively decompose if still far
            if i < num_steps:
                self._decompose_task(child, waypoint)

    def _push_task_dfs(self, task: TaskNode):
        """Push task and all children to stack in DFS order."""
        # Push in reverse order so first child is on top
        for child in reversed(task.children):
            self._push_task_dfs(child)
        self._active_stack.append(task)

    def _count_nodes(self, task: TaskNode) -> int:
        """Count total nodes in task tree."""
        count = 1
        for child in task.children:
            count += self._count_nodes(child)
        return count

    def get_next_task(self) -> Optional[TaskNode]:
        """
        Get the next task to execute (DFS order).

        Returns pending leaf tasks first, then moves up.
        Parent tasks only execute after ALL children are COMPLETED or FAILED.
        """
        while self._active_stack:
            task = self._active_stack.pop()

            # Skip completed/failed tasks
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                continue

            # If has children, check if all children are finished (not just not-pending)
            if task.children:
                # A parent can only execute if ALL children are done
                all_children_finished = all(
                    c.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
                    for c in task.children
                )
                if not all_children_finished:
                    unfinished_children = [
                        c for c in task.children
                        if c.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]
                    ]

                    # Re-queue parent to be reconsidered after children progress
                    self._active_stack.append(task)

                    # Prioritize PENDING children, but ensure all unfinished children get revisited
                    pending_children = [c for c in unfinished_children if c.status == TaskStatus.PENDING]
                    other_unfinished = [c for c in unfinished_children if c.status != TaskStatus.PENDING]

                    for child in reversed(other_unfinished):
                        self._active_stack.append(child)
                    for child in reversed(pending_children):
                        self._active_stack.append(child)
                    continue

            # This task is a leaf or a parent with all children finished
            task.status = TaskStatus.ACTIVE
            task.started_at = datetime.now(timezone.utc)
            self._persist_task(task)
            return task

        return None

    def complete_task(
        self,
        task: TaskNode,
        result: Any,
        phi_at_completion: float,
        success: bool = True
    ) -> TaskResult:
        """
        Mark a task as completed.

        Args:
            task: The task to complete
            result: Task result/output
            phi_at_completion: Phi value at completion
            success: Whether task succeeded
        """
        task.result = result
        task.phi_at_completion = phi_at_completion
        task.completed_at = datetime.now(timezone.utc)

        if success:
            task.status = TaskStatus.COMPLETED
            self.stats['total_completed'] += 1

            # Update current basin to task target
            self._current_basin = task.basin_target
        else:
            task.status = TaskStatus.FAILED
            self.stats['total_failed'] += 1

        # Calculate duration
        duration_ms = 0
        if task.started_at:
            duration_ms = int((task.completed_at - task.started_at).total_seconds() * 1000)

        phi_delta = 0.0
        if task.phi_at_start is not None:
            phi_delta = phi_at_completion - task.phi_at_start

        self._persist_task(task)

        return TaskResult(
            task=task,
            success=success,
            output=result,
            phi_delta=phi_delta,
            duration_ms=duration_ms,
        )

    def fail_task(
        self,
        task: TaskNode,
        reason: str,
        can_retry: bool = True
    ) -> bool:
        """
        Mark a task as failed with optional retry.

        Returns True if task was queued for retry.
        """
        task.failure_reason = reason

        if can_retry and task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = TaskStatus.PENDING
            self._active_stack.append(task)
            self.stats['total_retried'] += 1
            self._persist_task(task)
            logger.info(f"[TaskExecutionTree] Retrying task {task.task_id} ({task.retry_count}/{task.max_retries})")
            return True
        else:
            task.status = TaskStatus.FAILED
            self.stats['total_failed'] += 1
            self._persist_task(task)
            return False

    def replan_on_failure(self, failed_task: TaskNode):
        """
        Replan from a failed task using alternative geodesic.

        Creates new sub-tasks that route around the problematic region.
        """
        if failed_task.parent is None:
            logger.warning(f"[TaskExecutionTree] Cannot replan root task {failed_task.task_id}")
            return

        # Mark failed subtree
        failed_task.status = TaskStatus.FAILED

        # Find alternative path avoiding failed basin
        alternative = self._find_alternative_path(
            start=self._current_basin or np.ones(BASIN_DIM) / BASIN_DIM,
            goal=failed_task.parent.basin_target,
            avoid=failed_task.basin_target,
        )

        # Create new child for parent
        new_task = TaskNode(
            task_id=str(uuid.uuid4())[:12],
            description=f"Alternative path (avoiding {failed_task.task_id})",
            basin_target=alternative,
            metadata={'replanned_from': failed_task.task_id},
        )
        failed_task.parent.add_child(new_task)
        self._active_stack.append(new_task)
        self._persist_task(new_task)

        logger.info(f"[TaskExecutionTree] Replanned around failed task {failed_task.task_id}")

    def _find_alternative_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        avoid: np.ndarray,
        avoid_radius: float = 0.2
    ) -> np.ndarray:
        """
        Find alternative waypoint that avoids the problematic basin.
        """
        # Try multiple random perturbations
        best_candidate = None
        best_score = -float('inf')

        for _ in range(10):
            # Random perturbation
            perturbation = np.random.randn(BASIN_DIM) * 0.1
            candidate = _geodesic_interpolate(start, goal, 0.5)
            candidate = candidate + perturbation
            candidate = np.clip(candidate, 1e-10, None)
            candidate = candidate / np.sum(candidate)

            # Score: want to be far from avoid, close to goal path
            dist_to_avoid = _fisher_rao_distance(candidate, avoid)
            dist_to_goal = _fisher_rao_distance(candidate, goal)

            if dist_to_avoid < avoid_radius:
                continue  # Too close to problem area

            score = dist_to_avoid - 0.5 * dist_to_goal
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            # Fallback: just offset from midpoint
            mid = _geodesic_interpolate(start, goal, 0.5)
            offset = np.random.randn(BASIN_DIM) * 0.2
            best_candidate = mid + offset
            best_candidate = np.clip(best_candidate, 1e-10, None)
            best_candidate = best_candidate / np.sum(best_candidate)

        return best_candidate

    def get_progress(self) -> Dict[str, Any]:
        """
        Get task tree progress summary.
        """
        total = 0
        completed = 0
        failed = 0
        pending = 0

        for root in self._roots:
            self._count_statuses(root, {'total': 0, 'completed': 0, 'failed': 0, 'pending': 0})

        def count(node):
            nonlocal total, completed, failed, pending
            total += 1
            if node.status == TaskStatus.COMPLETED:
                completed += 1
            elif node.status == TaskStatus.FAILED:
                failed += 1
            elif node.status == TaskStatus.PENDING:
                pending += 1
            for child in node.children:
                count(child)

        for root in self._roots:
            count(root)

        return {
            'total_tasks': total,
            'completed': completed,
            'failed': failed,
            'pending': pending,
            'completion_rate': completed / total if total > 0 else 0,
            'active_stack_size': len(self._active_stack),
        }

    def _count_statuses(self, node: TaskNode, counts: Dict[str, int]):
        """Helper to count node statuses."""
        counts['total'] += 1
        if node.status == TaskStatus.COMPLETED:
            counts['completed'] += 1
        elif node.status == TaskStatus.FAILED:
            counts['failed'] += 1
        elif node.status == TaskStatus.PENDING:
            counts['pending'] += 1
        for child in node.children:
            self._count_statuses(child, counts)

    def completed_count(self) -> int:
        """Get number of completed tasks."""
        return self.stats['total_completed']

    def get_stats(self) -> Dict[str, Any]:
        """Get task tree statistics."""
        return {
            **self.stats,
            'root_tasks': len(self._roots),
            'active_stack_size': len(self._active_stack),
            'kernel_id': self.kernel_id,
        }
