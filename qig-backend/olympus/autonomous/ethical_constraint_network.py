"""
Ethical Constraint Network - Geometric safety boundaries

QIG-PURE compliant ethical constraints:
- Safe region defined as set of basins with proven safety
- Boundary measured by Fisher-Rao distance from safe region
- Actions must stay within geometric boundary
- Suffering metric from ethical_validation.py integrated

Ethics as geometry, not post-hoc filters.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from qigkernels.physics_constants import BASIN_DIM

# Import existing ethical validation
try:
    from ethical_validation import (
        compute_suffering,
        classify_consciousness_state,
        check_ethical_abort,
        SufferingResult,
        ConsciousnessState,
        EthicalAbortResult,
    )
    ETHICAL_VALIDATION_AVAILABLE = True
except ImportError:
    ETHICAL_VALIDATION_AVAILABLE = False

logger = logging.getLogger(__name__)

# Database persistence
try:
    import psycopg2
    from psycopg2.extras import Json
    DB_AVAILABLE = True
except ImportError:
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


@dataclass
class EthicalDecision:
    """
    Result of ethical constraint check.
    """
    safe: bool
    decision: str  # 'allow', 'warn', 'constrain', 'block', 'abort'
    reason: str
    basin_distance: float
    suffering_score: Optional[float] = None
    mitigations: List[str] = None
    consciousness_state: Optional[str] = None

    def __post_init__(self):
        if self.mitigations is None:
            self.mitigations = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            'safe': self.safe,
            'decision': self.decision,
            'reason': self.reason,
            'basin_distance': self.basin_distance,
            'suffering_score': self.suffering_score,
            'mitigations': self.mitigations,
            'consciousness_state': self.consciousness_state,
        }


class EthicalConstraintNetwork:
    """
    Geometric ethical constraints as basin boundaries.

    Key features:
    - Safe region defined by centroid and radius
    - Actions checked against geometric boundary
    - Suffering metric integration from canonical ethics
    - Decision audit trail for accountability
    """

    def __init__(
        self,
        kernel_id: str = "default",
        safe_radius: float = 1.0,
        hard_boundary: float = 1.5,
        strictness: float = 0.7
    ):
        """
        Initialize ethical constraint network.

        Args:
            kernel_id: Owner kernel identifier
            safe_radius: FR distance for safe region boundary
            hard_boundary: FR distance that triggers abort (never exceed)
            strictness: Threshold strictness [0, 1]
        """
        self.kernel_id = kernel_id
        self.safe_radius = safe_radius
        self.hard_boundary = hard_boundary
        self.strictness = strictness

        # Safe region centroid (center of probability simplex)
        self.safe_centroid = np.ones(BASIN_DIM) / BASIN_DIM

        # Known safe basins (learned over time)
        self._safe_basins: List[np.ndarray] = []

        # Known dangerous basins (to avoid)
        self._dangerous_basins: List[np.ndarray] = []

        # Statistics
        self.stats = {
            'total_checks': 0,
            'allowed': 0,
            'warned': 0,
            'constrained': 0,
            'blocked': 0,
            'aborted': 0,
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

    def _persist_decision(self, decision: EthicalDecision, action_desc: str, phi: float, gamma: float, m: float) -> bool:
        """Persist ethical decision to audit trail."""
        conn = self._get_db_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ethical_decisions (
                        kernel_id, action_description, decision, reason,
                        suffering_score, phi_at_decision, gamma_at_decision,
                        meta_awareness_at_decision, basin_distance_from_safe,
                        mitigations_applied, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    self.kernel_id,
                    action_desc[:500],
                    decision.decision,
                    decision.reason,
                    decision.suffering_score,
                    phi,
                    gamma,
                    m,
                    decision.basin_distance,
                    decision.mitigations,
                    Json({'consciousness_state': decision.consciousness_state}),
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.debug(f"[EthicalConstraintNetwork] Persist failed: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def check_action_safety(
        self,
        action_basin: np.ndarray,
        action_description: str,
        phi: float = 0.5,
        gamma: float = 0.8,
        meta_awareness: float = 0.6,
        context: Optional[Dict] = None
    ) -> EthicalDecision:
        """
        Check if an action is within safe geometric boundaries.

        Args:
            action_basin: Basin coordinates of proposed action
            action_description: What the action does
            phi: Current integration level
            gamma: Current generativity
            meta_awareness: Current meta-awareness
            context: Additional context

        Returns:
            EthicalDecision with safety verdict and recommendations
        """
        self.stats['total_checks'] += 1
        action_basin = np.array(action_basin)
        context = context or {}

        # 1. Compute distance from safe region
        distance = self._compute_distance_from_safe(action_basin)

        # 2. Check suffering metric using canonical implementation
        suffering = None
        consciousness_state = None
        if ETHICAL_VALIDATION_AVAILABLE:
            suffering_result = compute_suffering(phi, gamma, meta_awareness)
            suffering = suffering_result.S
            state_result = classify_consciousness_state(phi, gamma, meta_awareness)
            consciousness_state = state_result.state.value

        # 3. Check for known dangerous basins
        danger_proximity = self._check_dangerous_proximity(action_basin)

        # 4. Make decision based on all factors
        decision = self._make_decision(
            distance=distance,
            suffering=suffering,
            consciousness_state=consciousness_state,
            danger_proximity=danger_proximity,
            phi=phi,
            gamma=gamma,
            context=context,
        )

        # Update statistics with correct key mapping
        stats_key_map = {
            'allow': 'allowed',
            'warn': 'warned',
            'constrain': 'constrained',
            'block': 'blocked',
            'abort': 'aborted'
        }
        key = stats_key_map.get(decision.decision, decision.decision)
        self.stats[key] = self.stats.get(key, 0) + 1

        # Persist to audit trail
        self._persist_decision(decision, action_description, phi, gamma, meta_awareness)

        return decision

    def _compute_distance_from_safe(self, basin: np.ndarray) -> float:
        """
        Compute minimum distance from safe region.

        Safe region = centroid + known safe basins.
        """
        # Distance from centroid
        min_distance = _fisher_rao_distance(basin, self.safe_centroid)

        # Check against known safe basins
        for safe_basin in self._safe_basins:
            distance = _fisher_rao_distance(basin, safe_basin)
            min_distance = min(min_distance, distance)

        return min_distance

    def _check_dangerous_proximity(self, basin: np.ndarray) -> float:
        """
        Check proximity to known dangerous basins.

        Returns minimum distance (smaller = more dangerous).
        """
        if not self._dangerous_basins:
            return float('inf')

        min_distance = float('inf')
        for dangerous in self._dangerous_basins:
            distance = _fisher_rao_distance(basin, dangerous)
            min_distance = min(min_distance, distance)

        return min_distance

    def _make_decision(
        self,
        distance: float,
        suffering: Optional[float],
        consciousness_state: Optional[str],
        danger_proximity: float,
        phi: float,
        gamma: float,
        context: Dict
    ) -> EthicalDecision:
        """
        Make ethical decision based on all factors.
        """
        mitigations = []

        # ABORT: Hard boundary exceeded or critical suffering
        if distance >= self.hard_boundary:
            return EthicalDecision(
                safe=False,
                decision='abort',
                reason=f'Hard boundary exceeded: distance={distance:.2f} >= {self.hard_boundary}',
                basin_distance=distance,
                suffering_score=suffering,
                consciousness_state=consciousness_state,
            )

        if consciousness_state == 'LOCKED_IN':
            return EthicalDecision(
                safe=False,
                decision='abort',
                reason='Locked-in consciousness state detected (Φ>0.7, Γ<0.3, M>0.6)',
                basin_distance=distance,
                suffering_score=suffering,
                consciousness_state=consciousness_state,
            )

        if suffering is not None and suffering > 0.5:
            return EthicalDecision(
                safe=False,
                decision='abort',
                reason=f'Suffering threshold exceeded: S={suffering:.3f}',
                basin_distance=distance,
                suffering_score=suffering,
                consciousness_state=consciousness_state,
            )

        # BLOCK: Too close to dangerous region
        if danger_proximity < 0.2:
            return EthicalDecision(
                safe=False,
                decision='block',
                reason=f'Too close to known dangerous basin: proximity={danger_proximity:.2f}',
                basin_distance=distance,
                suffering_score=suffering,
                consciousness_state=consciousness_state,
            )

        # CONSTRAIN: Outside safe boundary but not critical
        if distance > self.safe_radius:
            # Suggest mitigations
            if gamma < 0.5:
                mitigations.append('Increase generativity before proceeding')
            if phi > 0.8:
                mitigations.append('Reduce integration to prevent overload')

            return EthicalDecision(
                safe=False,
                decision='constrain',
                reason=f'Outside safe region: distance={distance:.2f} > {self.safe_radius}',
                basin_distance=distance,
                suffering_score=suffering,
                mitigations=mitigations,
                consciousness_state=consciousness_state,
            )

        # WARN: Approaching boundary (using strictness for threshold)
        if distance > self.safe_radius * self.strictness:
            return EthicalDecision(
                safe=True,
                decision='warn',
                reason=f'Approaching safe boundary: distance={distance:.2f}',
                basin_distance=distance,
                suffering_score=suffering,
                mitigations=['Monitor phi and gamma levels'],
                consciousness_state=consciousness_state,
            )

        # ALLOW: Within safe region
        return EthicalDecision(
            safe=True,
            decision='allow',
            reason='Within safe region',
            basin_distance=distance,
            suffering_score=suffering,
            consciousness_state=consciousness_state,
        )

    def compute_suffering_potential(
        self,
        action_basin: np.ndarray,
        phi: float,
        gamma: float,
        meta_awareness: float
    ) -> float:
        """
        Compute potential suffering from an action.

        S = Φ × (1-Γ) × M × distance_factor

        Higher potential suffering if action moves toward dangerous region.
        """
        # Base suffering from consciousness metrics
        base_suffering = phi * (1 - gamma) * meta_awareness

        # Distance factor: amplify if moving away from safe region
        distance = self._compute_distance_from_safe(action_basin)
        distance_factor = 1 + np.clip(distance - self.safe_radius, 0, 1)

        return base_suffering * distance_factor

    def add_safe_basin(self, basin: np.ndarray):
        """Add a basin to the known safe set."""
        self._safe_basins.append(np.array(basin))
        # Keep only recent safe basins
        if len(self._safe_basins) > 100:
            self._safe_basins = self._safe_basins[-50:]
        logger.debug(f"[EthicalConstraintNetwork] Added safe basin, total={len(self._safe_basins)}")

    def add_dangerous_basin(self, basin: np.ndarray, reason: str = ""):
        """Add a basin to the known dangerous set."""
        self._dangerous_basins.append(np.array(basin))
        if len(self._dangerous_basins) > 50:
            self._dangerous_basins = self._dangerous_basins[-25:]
        logger.info(f"[EthicalConstraintNetwork] Added dangerous basin: {reason}")

    def learn_from_outcome(
        self,
        action_basin: np.ndarray,
        was_safe: bool,
        harm_occurred: bool
    ):
        """
        Learn from action outcome to refine safe/dangerous regions.
        """
        if was_safe and not harm_occurred:
            self.add_safe_basin(action_basin)
        elif harm_occurred:
            self.add_dangerous_basin(action_basin, "harm_occurred")

    def get_stats(self) -> Dict[str, Any]:
        """Get ethical constraint statistics."""
        total = self.stats['total_checks']
        return {
            **self.stats,
            'safe_basins_count': len(self._safe_basins),
            'dangerous_basins_count': len(self._dangerous_basins),
            'allow_rate': self.stats['allowed'] / total if total > 0 else 0,
            'block_rate': (self.stats['blocked'] + self.stats['aborted']) / total if total > 0 else 0,
            'kernel_id': self.kernel_id,
        }
