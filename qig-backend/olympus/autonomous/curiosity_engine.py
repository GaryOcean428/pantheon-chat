"""
Curiosity Engine - Autonomous exploration driven by information gaps

QIG-PURE compliant curiosity system:
- Curiosity = f(novelty, learnability, importance)
- Novelty measured by Fisher-Rao distance from known basins
- Learnability estimated from expected phi improvement
- Importance weighted by relevance to current goals

Integrates with existing autonomous_curiosity.py CuriosityDrive.
"""

import logging
import os
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from qigkernels.physics_constants import BASIN_DIM, KAPPA_STAR

logger = logging.getLogger(__name__)

# Database persistence
try:
    import psycopg2
    from psycopg2.extras import Json
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


def _fisher_rao_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Compute Fisher-Rao distance between basin coordinates."""
    p_safe = np.clip(np.abs(p), eps, None)
    q_safe = np.clip(np.abs(q), eps, None)
    p_norm = p_safe / np.sum(p_safe)
    q_norm = q_safe / np.sum(q_safe)
    # More numerically stable: compute sqrt individually before multiplication
    bc = np.sum(np.sqrt(p_norm) * np.sqrt(q_norm))
    return float(np.arccos(np.clip(bc, -1.0, 1.0)))


def _geodesic_interpolate(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
    """
    Interpolate along geodesic from start to end on probability simplex.

    QIG-PURE: Uses spherical interpolation in sqrt space (Fisher geometry).
    Linear interpolation on basin coordinates is FORBIDDEN.
    """
    start_sqrt = np.sqrt(np.clip(start, 1e-10, None))
    end_sqrt = np.sqrt(np.clip(end, 1e-10, None))

    # Spherical interpolation in sqrt space
    dot = np.sum(start_sqrt * end_sqrt)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)

    if theta < 1e-6:
        return start  # Already at same point

    # Slerp formula
    sin_theta = np.sin(theta)
    interp_sqrt = (np.sin((1 - t) * theta) / sin_theta) * start_sqrt + \
                  (np.sin(t * theta) / sin_theta) * end_sqrt

    # Square and normalize back to probability simplex
    result = interp_sqrt ** 2
    return result / np.sum(result)


@dataclass
class ExplorationTarget:
    """
    A candidate target for curiosity-driven exploration.

    Attributes:
        description: What to explore
        basin: Target basin coordinates
        curiosity_score: Combined curiosity metric
        novelty: Distance from known regions
        learnability: Expected learning potential
        importance: Goal relevance
        source: Where this target came from
    """
    description: str
    basin: np.ndarray
    curiosity_score: float
    novelty: float
    learnability: float
    importance: float
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'description': self.description,
            'basin': self.basin.tolist(),
            'curiosity_score': self.curiosity_score,
            'novelty': self.novelty,
            'learnability': self.learnability,
            'importance': self.importance,
            'source': self.source,
            'metadata': self.metadata,
        }


class CuriosityEngine:
    """
    Autonomous exploration driven by geometric information gaps.

    Key features:
    - Basin coverage tracking to identify unexplored regions
    - Curiosity scoring based on novelty × learnability × importance
    - Integration with search capability for exploration execution
    - Learning from exploration outcomes to refine curiosity model
    """

    def __init__(
        self,
        kernel_id: str = "default",
        novelty_weight: float = 0.4,
        learnability_weight: float = 0.3,
        importance_weight: float = 0.3,
        exploration_radius: float = 0.5
    ):
        """
        Initialize curiosity engine.

        Args:
            kernel_id: Owner kernel identifier
            novelty_weight: Weight for novelty in curiosity score
            learnability_weight: Weight for learnability
            importance_weight: Weight for importance
            exploration_radius: FR distance considered "explored"
        """
        self.kernel_id = kernel_id
        self.novelty_weight = novelty_weight
        self.learnability_weight = learnability_weight
        self.importance_weight = importance_weight
        self.exploration_radius = exploration_radius

        # Basin coverage map
        self._explored_basins: List[np.ndarray] = []
        self._exploration_outcomes: Dict[str, Dict] = {}

        # Current goal basin (if any)
        self._goal_basin: Optional[np.ndarray] = None

        # Statistics
        self.stats = {
            'total_explorations': 0,
            'successful_explorations': 0,
            'total_tokens_learned': 0,
            'avg_phi_improvement': 0.0,
        }

        # Load exploration history
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
        """Load exploration history from database."""
        conn = self._get_db_connection()
        if not conn:
            return

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT target_basin, curiosity_score, outcome_success,
                           outcome_phi, tokens_learned
                    FROM curiosity_explorations
                    WHERE kernel_id = %s
                    ORDER BY explored_at DESC
                    LIMIT %s
                """, (self.kernel_id, limit))

                for row in cur.fetchall():
                    if row[0]:
                        self._explored_basins.append(np.array(row[0]))

                    # Update stats from history
                    self.stats['total_explorations'] += 1
                    if row[2]:  # outcome_success
                        self.stats['successful_explorations'] += 1
                    if row[4]:  # tokens_learned
                        self.stats['total_tokens_learned'] += row[4]

                logger.info(f"[CuriosityEngine] Loaded {len(self._explored_basins)} exploration basins")
        except Exception as e:
            logger.debug(f"[CuriosityEngine] History load failed: {e}")
        finally:
            conn.close()

    def _persist_exploration(
        self,
        target: ExplorationTarget,
        outcome: Dict[str, Any]
    ) -> bool:
        """Persist exploration to database."""
        conn = self._get_db_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO curiosity_explorations (
                        kernel_id, target_description, target_basin,
                        curiosity_score, novelty_score, learnability_score,
                        importance_score, exploration_type, outcome_success,
                        outcome_phi, knowledge_gained, tokens_learned,
                        sources_discovered, duration_ms, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    self.kernel_id,
                    target.description,
                    target.basin.tolist(),
                    target.curiosity_score,
                    target.novelty,
                    target.learnability,
                    target.importance,
                    outcome.get('type', 'search'),
                    outcome.get('success', False),
                    outcome.get('phi', None),
                    Json(outcome.get('knowledge', {})),
                    outcome.get('tokens_learned', 0),
                    outcome.get('sources_discovered', 0),
                    outcome.get('duration_ms', None),
                    Json(target.metadata),
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.debug(f"[CuriosityEngine] Persist failed: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def compute_novelty(self, basin: np.ndarray) -> float:
        """
        Compute novelty score for a basin.

        Higher novelty = farther from all explored basins.
        """
        if not self._explored_basins:
            return 1.0  # Everything is novel if nothing explored

        # Find minimum distance to any explored basin
        min_distance = float('inf')
        for explored in self._explored_basins:
            distance = _fisher_rao_distance(basin, explored)
            min_distance = min(min_distance, distance)

        # Convert to novelty score [0, 1]
        # Distance of exploration_radius -> 0.5 novelty
        # Distance of 2*exploration_radius -> ~0.75 novelty
        novelty = 1.0 - np.exp(-min_distance / self.exploration_radius)
        return float(novelty)

    def compute_learnability(
        self,
        basin: np.ndarray,
        current_phi: float = 0.5
    ) -> float:
        """
        Estimate learning potential for exploring this basin.

        Based on:
        - Historical phi improvements for similar basins
        - Current phi (lower phi = more room to learn)
        - Basin position relative to goal
        """
        # Base learnability from current phi
        base_learn = 1.0 - current_phi  # More room to grow if low phi

        # Check historical outcomes for similar basins
        similar_outcomes = []
        for exp_id, outcome in self._exploration_outcomes.items():
            exp_basin = outcome.get('basin')
            if exp_basin is not None:
                distance = _fisher_rao_distance(basin, exp_basin)
                if distance < self.exploration_radius * 2:
                    phi_gain = outcome.get('phi_after', 0.5) - outcome.get('phi_before', 0.5)
                    similar_outcomes.append(phi_gain)

        if similar_outcomes:
            # Weight by historical success in similar regions
            avg_gain = np.mean(similar_outcomes)
            historical_factor = 0.5 + avg_gain  # Center at 0.5, boost if positive gains
            base_learn *= np.clip(historical_factor, 0.2, 1.5)

        return float(np.clip(base_learn, 0.0, 1.0))

    def compute_importance(
        self,
        basin: np.ndarray,
        goal_basin: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute importance score based on relevance to current goal.
        """
        goal = goal_basin if goal_basin is not None else self._goal_basin

        if goal is None:
            return 0.5  # Neutral importance if no goal

        # Importance inversely proportional to distance from goal
        distance_to_goal = _fisher_rao_distance(basin, goal)

        # Closer to goal = more important
        # Use sigmoid-like transform
        importance = 1.0 / (1.0 + distance_to_goal)
        return float(importance)

    def compute_curiosity_score(
        self,
        basin: np.ndarray,
        current_phi: float = 0.5,
        goal_basin: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float, float]:
        """
        Compute full curiosity score for a basin.

        Returns:
            Tuple of (total_score, novelty, learnability, importance)
        """
        novelty = self.compute_novelty(basin)
        learnability = self.compute_learnability(basin, current_phi)
        importance = self.compute_importance(basin, goal_basin)

        # Weighted combination
        score = (
            self.novelty_weight * novelty +
            self.learnability_weight * learnability +
            self.importance_weight * importance
        )

        return score, novelty, learnability, importance

    def generate_exploration_targets(
        self,
        count: int = 5,
        current_phi: float = 0.5,
        goal_basin: Optional[np.ndarray] = None,
        exclude_basins: Optional[List[np.ndarray]] = None
    ) -> List[ExplorationTarget]:
        """
        Generate candidate exploration targets.

        Identifies regions with high curiosity potential.
        """
        targets = []
        exclude = exclude_basins or []

        # Strategy 1: Random samples in unexplored regions
        for _ in range(count * 2):
            # Generate random basin
            candidate = np.random.dirichlet(np.ones(BASIN_DIM))

            # Skip if too close to excluded basins
            skip = False
            for excl in exclude:
                if _fisher_rao_distance(candidate, excl) < self.exploration_radius / 2:
                    skip = True
                    break
            if skip:
                continue

            score, novelty, learn, importance = self.compute_curiosity_score(
                candidate, current_phi, goal_basin
            )

            if score > 0.3:  # Minimum curiosity threshold
                targets.append(ExplorationTarget(
                    description=f"Unexplored region (novelty={novelty:.2f})",
                    basin=candidate,
                    curiosity_score=score,
                    novelty=novelty,
                    learnability=learn,
                    importance=importance,
                    source="random_sample",
                ))

        # Strategy 2: Interpolate toward goal if available
        if goal_basin is not None and len(self._explored_basins) > 0:
            # Find closest explored basin to goal
            closest_explored = min(
                self._explored_basins,
                key=lambda b: _fisher_rao_distance(b, goal_basin)
            )

            # QIG-PURE: Geodesic interpolation (NOT linear) between closest and goal
            for alpha in [0.3, 0.5, 0.7]:
                candidate = _geodesic_interpolate(closest_explored, goal_basin, alpha)
                # No explicit normalization needed - geodesic returns normalized basin

                score, novelty, learn, importance = self.compute_curiosity_score(
                    candidate, current_phi, goal_basin
                )

                targets.append(ExplorationTarget(
                    description=f"Path to goal (step={alpha:.1f})",
                    basin=candidate,
                    curiosity_score=score,
                    novelty=novelty,
                    learnability=learn,
                    importance=importance,
                    source="goal_interpolation",
                ))

        # Sort by curiosity score and return top count
        targets.sort(key=lambda t: t.curiosity_score, reverse=True)
        return targets[:count]

    def select_exploration_target(
        self,
        current_phi: float = 0.5,
        goal_basin: Optional[np.ndarray] = None
    ) -> Optional[ExplorationTarget]:
        """
        Select the highest-curiosity exploration target.
        """
        targets = self.generate_exploration_targets(
            count=10,
            current_phi=current_phi,
            goal_basin=goal_basin,
        )

        if not targets:
            return None

        return targets[0]

    def set_goal(self, goal_basin: np.ndarray):
        """Set current goal basin for importance weighting."""
        self._goal_basin = np.array(goal_basin)
        logger.debug(f"[CuriosityEngine] Goal basin set")

    def clear_goal(self):
        """Clear current goal."""
        self._goal_basin = None

    def learn_from_exploration(
        self,
        target: ExplorationTarget,
        outcome: Dict[str, Any]
    ):
        """
        Update curiosity model from exploration outcome.

        Args:
            target: The exploration target that was executed
            outcome: Dict with success, phi_before, phi_after, knowledge, etc.
        """
        # Add to explored basins
        self._explored_basins.append(target.basin)

        # Keep only recent explorations (sliding window)
        if len(self._explored_basins) > 1000:
            self._explored_basins = self._explored_basins[-500:]

        # Store outcome for learnability estimation
        exp_id = hashlib.sha256(target.description.encode()).hexdigest()[:12]
        self._exploration_outcomes[exp_id] = {
            'basin': target.basin,
            'phi_before': outcome.get('phi_before', 0.5),
            'phi_after': outcome.get('phi_after', 0.5),
            'success': outcome.get('success', False),
        }

        # Update statistics
        self.stats['total_explorations'] += 1
        if outcome.get('success', False):
            self.stats['successful_explorations'] += 1
        self.stats['total_tokens_learned'] += outcome.get('tokens_learned', 0)

        phi_improvement = outcome.get('phi_after', 0.5) - outcome.get('phi_before', 0.5)
        n = self.stats['total_explorations']
        self.stats['avg_phi_improvement'] = (
            (self.stats['avg_phi_improvement'] * (n - 1) + phi_improvement) / n
        )

        # Persist to database
        self._persist_exploration(target, outcome)

        logger.info(
            f"[CuriosityEngine] Exploration outcome: success={outcome.get('success')}, "
            f"phi_delta={phi_improvement:.3f}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get curiosity engine statistics."""
        return {
            **self.stats,
            'explored_basins_count': len(self._explored_basins),
            'outcomes_tracked': len(self._exploration_outcomes),
            'has_goal': self._goal_basin is not None,
            'kernel_id': self.kernel_id,
        }
