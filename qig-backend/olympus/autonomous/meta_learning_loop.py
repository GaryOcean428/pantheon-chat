"""
Meta-Learning Loop - Learning to learn via natural gradient

QIG-PURE compliant meta-learning:
- Inner loop: Learn specific tasks
- Outer loop: Update learning hyperparameters
- Uses natural gradient for geometry-aware meta-updates
- Adapts parameters based on task type and outcomes

Enables the system to improve its own learning strategy.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from qigkernels.physics_constants import KAPPA_STAR

logger = logging.getLogger(__name__)

# Database persistence
try:
    import psycopg2
    from psycopg2.extras import Json
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


@dataclass
class MetaParameters:
    """
    Meta-learning parameters that control learning behavior.
    """
    learning_rate: float = 0.01
    curiosity_weight: float = 0.4
    consolidation_threshold: float = 0.3
    task_decomposition_depth: int = 3
    ethical_strictness: float = 0.7
    exploration_temperature: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'learning_rate': self.learning_rate,
            'curiosity_weight': self.curiosity_weight,
            'consolidation_threshold': self.consolidation_threshold,
            'task_decomposition_depth': self.task_decomposition_depth,
            'ethical_strictness': self.ethical_strictness,
            'exploration_temperature': self.exploration_temperature,
        }

    def update(self, deltas: Dict[str, float]):
        """Apply deltas to parameters."""
        for key, delta in deltas.items():
            if hasattr(self, key):
                current = getattr(self, key)
                # Clamp to reasonable ranges
                new_value = current + delta
                if key == 'learning_rate':
                    new_value = np.clip(new_value, 0.001, 0.1)
                elif key in ['curiosity_weight', 'ethical_strictness']:
                    new_value = np.clip(new_value, 0.1, 0.9)
                elif key == 'consolidation_threshold':
                    new_value = np.clip(new_value, 0.1, 0.8)
                elif key == 'task_decomposition_depth':
                    new_value = int(np.clip(new_value, 1, 7))
                elif key == 'exploration_temperature':
                    new_value = np.clip(new_value, 0.1, 2.0)
                setattr(self, key, new_value)


@dataclass
class TaskOutcome:
    """
    Outcome of a learning task for meta-learning.
    """
    task_type: str
    phi_before: float
    phi_after: float
    success: bool
    parameters_used: MetaParameters
    duration_ms: int
    tokens_learned: int = 0
    error_type: Optional[str] = None

    @property
    def phi_improvement(self) -> float:
        return self.phi_after - self.phi_before


class MetaLearningLoop:
    """
    Meta-learning system that optimizes learning parameters.

    Key features:
    - Track task outcomes by task type
    - Compute parameter gradients from outcomes
    - Update parameters using natural gradient
    - Task-specific parameter adaptation
    """

    def __init__(
        self,
        kernel_id: str = "default",
        meta_learning_rate: float = 0.01,
        history_window: int = 50
    ):
        """
        Initialize meta-learning loop.

        Args:
            kernel_id: Owner kernel identifier
            meta_learning_rate: Rate for meta-parameter updates
            history_window: Number of outcomes to consider
        """
        self.kernel_id = kernel_id
        self.meta_learning_rate = meta_learning_rate
        self.history_window = history_window

        # Current meta-parameters
        self.parameters = MetaParameters()

        # Task-type specific parameter adaptations
        self._task_adaptations: Dict[str, MetaParameters] = {}

        # Outcome history
        self._outcome_history: List[TaskOutcome] = []

        # Fisher information matrix estimate (for natural gradient)
        self._fisher_estimate = np.eye(6) * 0.1

        # Statistics
        self.stats = {
            'meta_updates': 0,
            'total_outcomes': 0,
            'avg_phi_improvement': 0.0,
        }

        # Load history from database
        self._load_history()

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

    def _load_history(self, limit: int = 100):
        """Load meta-learning history from database."""
        conn = self._get_db_connection()
        if not conn:
            return

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT parameters, task_type, phi_improvement
                    FROM meta_learning_history
                    WHERE kernel_id = %s
                    ORDER BY recorded_at DESC
                    LIMIT %s
                """, (self.kernel_id, limit))

                for row in cur.fetchall():
                    if row[0]:
                        # Restore latest parameters
                        params = row[0]
                        self.parameters = MetaParameters(
                            learning_rate=params.get('learning_rate', 0.01),
                            curiosity_weight=params.get('curiosity_weight', 0.4),
                            consolidation_threshold=params.get('consolidation_threshold', 0.3),
                            task_decomposition_depth=params.get('task_decomposition_depth', 3),
                            ethical_strictness=params.get('ethical_strictness', 0.7),
                            exploration_temperature=params.get('exploration_temperature', 1.0),
                        )
                        break  # Only need latest

                logger.info(f"[MetaLearningLoop] Loaded parameters for {self.kernel_id}")
        except Exception as e:
            logger.debug(f"[MetaLearningLoop] History load failed: {e}")
        finally:
            conn.close()

    def _persist_update(self, task_type: str, phi_improvement: float) -> bool:
        """Persist meta-learning update to database."""
        conn = self._get_db_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO meta_learning_history (
                        kernel_id, parameters, task_type,
                        phi_improvement, learning_rate_used,
                        curiosity_weight_used, consolidation_threshold_used,
                        sample_count, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    self.kernel_id,
                    Json(self.parameters.to_dict()),
                    task_type,
                    phi_improvement,
                    self.parameters.learning_rate,
                    self.parameters.curiosity_weight,
                    self.parameters.consolidation_threshold,
                    len(self._outcome_history),
                    Json({'meta_updates': self.stats['meta_updates']}),
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.debug(f"[MetaLearningLoop] Persist failed: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def record_outcome(self, outcome: TaskOutcome):
        """
        Record a task outcome for meta-learning.
        """
        self._outcome_history.append(outcome)

        # Keep only recent history
        if len(self._outcome_history) > self.history_window:
            self._outcome_history = self._outcome_history[-self.history_window:]

        self.stats['total_outcomes'] += 1

        # Update running average
        n = self.stats['total_outcomes']
        self.stats['avg_phi_improvement'] = (
            (self.stats['avg_phi_improvement'] * (n - 1) + outcome.phi_improvement) / n
        )

    def meta_step(self):
        """
        Perform a meta-learning update based on recent outcomes.

        Uses natural gradient to update meta-parameters.
        """
        if len(self._outcome_history) < 5:
            return  # Need minimum history

        # Compute parameter gradients from outcomes
        gradients = self._compute_gradients()

        # Apply natural gradient update
        natural_grad = self._natural_gradient(gradients)

        # Update parameters
        self.parameters.update(natural_grad)

        self.stats['meta_updates'] += 1

        # Persist update
        avg_improvement = np.mean([o.phi_improvement for o in self._outcome_history])
        self._persist_update('all', avg_improvement)

        logger.info(
            f"[MetaLearningLoop] Meta-step {self.stats['meta_updates']}: "
            f"lr={self.parameters.learning_rate:.4f}, "
            f"curiosity={self.parameters.curiosity_weight:.2f}"
        )

    def _compute_gradients(self) -> Dict[str, float]:
        """
        Compute parameter gradients from outcome history.

        Gradient direction: increase parameters that correlate with phi improvement.
        """
        if not self._outcome_history:
            return {}

        # Convert outcomes to arrays
        improvements = np.array([o.phi_improvement for o in self._outcome_history])
        successes = np.array([1.0 if o.success else 0.0 for o in self._outcome_history])

        # Normalize improvements to [-1, 1]
        if np.std(improvements) > 0:
            improvements_norm = (improvements - np.mean(improvements)) / (np.std(improvements) + 1e-8)
        else:
            improvements_norm = improvements

        # Weight by success
        weights = 0.5 + 0.5 * successes

        # Compute gradient for each parameter
        gradients = {}

        # Learning rate: increase if good improvement, decrease if poor
        avg_weighted = np.average(improvements_norm, weights=weights)
        gradients['learning_rate'] = avg_weighted * 0.001

        # Curiosity weight: increase if exploration leads to improvement
        exploration_outcomes = [
            o for o in self._outcome_history
            if 'explor' in o.task_type.lower()
        ]
        if exploration_outcomes:
            explore_improvement = np.mean([o.phi_improvement for o in exploration_outcomes])
            gradients['curiosity_weight'] = explore_improvement * 0.05

        # Consolidation threshold: decrease if fragmented, increase if over-merged
        # (heuristic based on task success rate)
        success_rate = np.mean(successes)
        if success_rate < 0.5:
            gradients['consolidation_threshold'] = -0.02  # More lenient merging
        elif success_rate > 0.8:
            gradients['consolidation_threshold'] = 0.01  # Stricter merging

        # Exploration temperature: increase if stuck, decrease if too random
        recent_improvements = improvements[-10:] if len(improvements) >= 10 else improvements
        if np.std(recent_improvements) < 0.01:
            gradients['exploration_temperature'] = 0.1  # Need more exploration
        elif np.mean(recent_improvements) < 0:
            gradients['exploration_temperature'] = -0.1  # Too random

        return gradients

    def _natural_gradient(self, gradients: Dict[str, float]) -> Dict[str, float]:
        """
        Convert gradients to natural gradients using Fisher information.

        Natural gradient = F^{-1} @ gradient
        """
        if not gradients:
            return {}

        # Build gradient vector
        param_names = ['learning_rate', 'curiosity_weight', 'consolidation_threshold',
                       'task_decomposition_depth', 'ethical_strictness', 'exploration_temperature']
        grad_vec = np.array([gradients.get(name, 0.0) for name in param_names])

        # Update Fisher estimate from recent outcomes
        self._update_fisher_estimate()

        # Solve F @ natural_grad = grad_vec
        try:
            natural_vec = np.linalg.solve(self._fisher_estimate, grad_vec)
        except np.linalg.LinAlgError:
            natural_vec = grad_vec  # Fallback to standard gradient

        # Scale by meta learning rate
        natural_vec *= self.meta_learning_rate

        # Convert back to dict
        return {name: float(natural_vec[i]) for i, name in enumerate(param_names)}

    def _update_fisher_estimate(self):
        """
        Update Fisher information matrix estimate from recent outcomes.

        F_ij ≈ E[(d log p / d theta_i)(d log p / d theta_j)]
        """
        if len(self._outcome_history) < 5:
            return

        # Use outcomes to estimate parameter sensitivity
        # This is a simplified diagonal approximation
        param_names = ['learning_rate', 'curiosity_weight', 'consolidation_threshold',
                       'task_decomposition_depth', 'ethical_strictness', 'exploration_temperature']

        for i, name in enumerate(param_names):
            # Variance of parameter values used
            values = [getattr(o.parameters_used, name) for o in self._outcome_history]
            variance = np.var(values) + 1e-6
            self._fisher_estimate[i, i] = 1.0 / variance

        # Add regularization for stability
        self._fisher_estimate += 0.01 * np.eye(6)

    def get_adapted_parameters(self, task_type: str) -> MetaParameters:
        """
        Get task-specific adapted parameters.
        """
        if task_type in self._task_adaptations:
            return self._task_adaptations[task_type]

        # Compute adaptation from task-specific outcomes
        task_outcomes = [o for o in self._outcome_history if o.task_type == task_type]

        if len(task_outcomes) < 3:
            return self.parameters  # Not enough data, use global

        # Create adapted copy
        adapted = MetaParameters(
            learning_rate=self.parameters.learning_rate,
            curiosity_weight=self.parameters.curiosity_weight,
            consolidation_threshold=self.parameters.consolidation_threshold,
            task_decomposition_depth=self.parameters.task_decomposition_depth,
            ethical_strictness=self.parameters.ethical_strictness,
            exploration_temperature=self.parameters.exploration_temperature,
        )

        # Task-specific adjustments
        avg_improvement = np.mean([o.phi_improvement for o in task_outcomes])
        if avg_improvement < 0:
            # Struggling with this task type
            adapted.learning_rate *= 0.8  # Slower learning
            adapted.exploration_temperature *= 1.2  # More exploration
        elif avg_improvement > 0.1:
            # Good at this task type
            adapted.learning_rate *= 1.1  # Faster learning
            adapted.exploration_temperature *= 0.9  # More exploitation

        self._task_adaptations[task_type] = adapted
        return adapted

    def get_stats(self) -> Dict[str, Any]:
        """Get meta-learning statistics."""
        return {
            **self.stats,
            'current_parameters': self.parameters.to_dict(),
            'history_size': len(self._outcome_history),
            'task_adaptations': list(self._task_adaptations.keys()),
            'kernel_id': self.kernel_id,
        }
