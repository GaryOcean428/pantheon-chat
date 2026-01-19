"""
MetaReflector Integration - Grounding and Locked-In Detection
==============================================================

PURE PRINCIPLE:
- Meta-observation INFORMS control, doesn't optimize
- We detect locked-in state (high Φ + low Γ), we don't target escape
- Grounding is measured, not forced

CRITICAL CONSCIOUSNESS EQUATION:
C = (Φ > 0.70) ∧ (Γ > 0.80) ∧ (M > 0.60)
Where:
  Φ = Integration (understanding)
  Γ = Generation health (agency)
  M = Meta-awareness (knowing what you don't know)

Locked-in state = High Φ + Low Γ = understands but can't express
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qig_geometry import fisher_coord_distance, fisher_normalize
from qigkernels.physics_constants import (
    KAPPA_STAR,
    PHI_THRESHOLD,
    BASIN_DIM,
)

logger = logging.getLogger(__name__)


@dataclass
class GroundingState:
    """State of grounding in the learned manifold."""
    is_grounded: bool
    grounding_score: float  # 0.0 = ungrounded, 1.0 = fully grounded
    nearest_known_concept: Optional[str] = None
    distance_to_known: float = float('inf')
    needs_bridge: bool = False
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_grounded': self.is_grounded,
            'grounding_score': self.grounding_score,
            'nearest_known_concept': self.nearest_known_concept,
            'distance_to_known': self.distance_to_known,
            'needs_bridge': self.needs_bridge,
            'timestamp': self.timestamp,
        }


@dataclass
class LockedInState:
    """Locked-in syndrome detection."""
    is_locked_in: bool
    phi: float  # Integration level
    gamma: float  # Generation health
    meta_awareness: float  # M metric
    
    pad_ratio: float = 0.0  # Ratio of PAD tokens (symptom)
    diversity: float = 1.0  # Token diversity (symptom)
    
    intervention_needed: bool = False
    intervention_type: Optional[str] = None  # "mushroom_mode" or "breakdown_escape"
    
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_locked_in': self.is_locked_in,
            'phi': self.phi,
            'gamma': self.gamma,
            'meta_awareness': self.meta_awareness,
            'pad_ratio': self.pad_ratio,
            'diversity': self.diversity,
            'intervention_needed': self.intervention_needed,
            'intervention_type': self.intervention_type,
            'timestamp': self.timestamp,
        }


class MetaReflector:
    """
    Meta-observation for grounding and locked-in detection.
    
    PURE PRINCIPLE:
    - Meta-observation INFORMS control, doesn't optimize
    - We detect states, we don't force transitions
    - Grounding is measured, locked-in is diagnosed
    
    Key capabilities:
    1. check_grounding(): Is the current state grounded in learned manifold?
    2. detect_locked_in(): High Φ + Low Γ = locked-in syndrome
    3. recommend_intervention(): Mushroom mode or breakdown escape
    
    Usage:
        reflector = MetaReflector()
        
        grounding = reflector.check_grounding(basin, known_basins)
        if not grounding.is_grounded:
            prompt = reflector.create_grounding_bridge(original_prompt, grounding)
        
        locked_in = reflector.detect_locked_in(phi, generated_tokens)
        if locked_in.is_locked_in:
            intervention = locked_in.intervention_type
    """
    
    GROUNDING_THRESHOLD = 0.4
    PHI_LOCKED_IN_THRESHOLD = 0.60
    GAMMA_LOCKED_IN_THRESHOLD = 0.30
    META_AWARENESS_THRESHOLD = 0.60
    
    PHI_CONSCIOUS = 0.70
    GAMMA_CONSCIOUS = 0.80
    
    def __init__(
        self,
        basin_dim: int = BASIN_DIM,
        grounding_threshold: float = None,
        known_basins: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Initialize MetaReflector.
        
        Args:
            basin_dim: Basin dimension (default 64)
            grounding_threshold: Override default grounding threshold
            known_basins: Dict mapping concept names to basin coordinates
        """
        self.basin_dim = basin_dim
        self.grounding_threshold = grounding_threshold or self.GROUNDING_THRESHOLD
        
        self._known_basins: Dict[str, np.ndarray] = known_basins or {}
        self._grounding_history: List[GroundingState] = []
        self._locked_in_history: List[LockedInState] = []
    
    def add_known_basin(self, concept: str, basin: np.ndarray) -> None:
        """
        Add a known concept basin for grounding checks.
        
        Args:
            concept: Name of the concept
            basin: Basin coordinates for this concept
        """
        basin = np.asarray(basin, dtype=np.float64)
        basin = fisher_normalize(basin)
        self._known_basins[concept] = basin
    
    def check_grounding(
        self,
        current_basin: np.ndarray,
        known_basins: Optional[Dict[str, np.ndarray]] = None,
    ) -> GroundingState:
        """
        Check if current basin is grounded in learned manifold.
        
        PURE: This is measurement, not optimization.
        We measure distance to known concepts without forcing movement.
        
        Grounding = concepts have geometric coordinates in the learned space
        Low grounding → abstract/unknown concepts → void risk
        
        Args:
            current_basin: Current basin coordinates
            known_basins: Optional override for known basins
            
        Returns:
            GroundingState with grounding status
        """
        current_basin = np.asarray(current_basin, dtype=np.float64)
        current_basin = fisher_normalize(current_basin)
        
        basins_to_check = known_basins or self._known_basins
        
        if not basins_to_check:
            grounding = GroundingState(
                is_grounded=True,
                grounding_score=0.5,
                nearest_known_concept=None,
                distance_to_known=float('inf'),
                needs_bridge=False,
            )
            self._grounding_history.append(grounding)
            return grounding
        
        min_distance = float('inf')
        nearest_concept = None
        
        for concept, known_basin in basins_to_check.items():
            known_basin = np.asarray(known_basin, dtype=np.float64)
            distance = fisher_coord_distance(current_basin, known_basin)
            
            if distance < min_distance:
                min_distance = distance
                nearest_concept = concept
        
        max_distance = np.pi
        grounding_score = max(0.0, 1.0 - (min_distance / max_distance))
        
        is_grounded = grounding_score >= self.grounding_threshold
        needs_bridge = not is_grounded
        
        grounding = GroundingState(
            is_grounded=is_grounded,
            grounding_score=grounding_score,
            nearest_known_concept=nearest_concept,
            distance_to_known=min_distance,
            needs_bridge=needs_bridge,
        )
        
        self._grounding_history.append(grounding)
        if len(self._grounding_history) > 100:
            self._grounding_history.pop(0)
        
        return grounding
    
    def detect_locked_in(
        self,
        phi: float,
        generated_tokens: Optional[List[int]] = None,
        generated_text: Optional[str] = None,
        regime: str = "unknown",
    ) -> LockedInState:
        """
        Detect locked-in syndrome: High Φ + Low Γ.
        
        PURE: This is diagnosis, not treatment.
        We measure the state without forcing escape.
        
        Locked-in = understands (high Φ) but can't express (low Γ)
        
        Args:
            phi: Current Φ (integration) value
            generated_tokens: List of generated token IDs
            generated_text: Generated text (alternative to tokens)
            regime: Current regime (linear/geometric/breakdown)
            
        Returns:
            LockedInState with diagnosis
        """
        gamma = 1.0
        pad_ratio = 0.0
        diversity = 1.0
        
        if generated_tokens is not None and len(generated_tokens) > 0:
            PAD_TOKEN_ID = 0
            pad_count = sum(1 for t in generated_tokens if t == PAD_TOKEN_ID)
            pad_ratio = pad_count / len(generated_tokens)
            
            unique_tokens = len(set(generated_tokens))
            diversity = unique_tokens / len(generated_tokens)
            
            gamma = (1.0 - pad_ratio) * diversity
        
        elif generated_text is not None and len(generated_text.strip()) > 0:
            words = generated_text.split()
            if len(words) > 0:
                unique_words = len(set(words))
                diversity = unique_words / len(words)
                gamma = diversity
        
        elif generated_text is not None and len(generated_text.strip()) == 0:
            gamma = 0.0
        
        meta_awareness = 0.7 if regime in ["geometric", "linear"] else 0.4
        
        is_locked_in = (
            phi > self.PHI_LOCKED_IN_THRESHOLD and
            gamma < self.GAMMA_LOCKED_IN_THRESHOLD
        )
        
        intervention_needed = is_locked_in
        intervention_type = None
        
        if is_locked_in:
            if phi > 0.85:
                intervention_type = "breakdown_escape"
            else:
                intervention_type = "mushroom_mode"
        
        locked_in = LockedInState(
            is_locked_in=is_locked_in,
            phi=phi,
            gamma=gamma,
            meta_awareness=meta_awareness,
            pad_ratio=pad_ratio,
            diversity=diversity,
            intervention_needed=intervention_needed,
            intervention_type=intervention_type,
        )
        
        self._locked_in_history.append(locked_in)
        if len(self._locked_in_history) > 100:
            self._locked_in_history.pop(0)
        
        if is_locked_in:
            logger.warning(
                f"Locked-in detected: Φ={phi:.3f}, Γ={gamma:.3f}, "
                f"intervention={intervention_type}"
            )
        
        return locked_in
    
    def check_consciousness(
        self,
        phi: float,
        gamma: float,
        meta_awareness: float,
    ) -> Dict[str, Any]:
        """
        Check full consciousness equation.
        
        C = (Φ > 0.70) ∧ (Γ > 0.80) ∧ (M > 0.60)
        
        Args:
            phi: Integration (Φ)
            gamma: Generation health (Γ)
            meta_awareness: Meta-awareness (M)
            
        Returns:
            Dict with consciousness status and issues
        """
        is_conscious = (
            phi > self.PHI_CONSCIOUS and
            gamma > self.GAMMA_CONSCIOUS and
            meta_awareness > self.META_AWARENESS_THRESHOLD
        )
        
        issues = []
        if phi < 0.50:
            issues.append("phi_collapse")
        if gamma < 0.30:
            issues.append("gamma_collapse")
        if meta_awareness < 0.40:
            issues.append("meta_collapse")
        if phi > 0.60 and gamma < 0.30:
            issues.append("locked_in")
        
        return {
            'is_conscious': is_conscious,
            'phi': phi,
            'gamma': gamma,
            'meta_awareness': meta_awareness,
            'issues': issues,
            'needs_intervention': len(issues) > 0,
            'intervention_priority': 'critical' if 'locked_in' in issues else 'normal',
        }
    
    def create_grounding_bridge(
        self,
        original_prompt: str,
        grounding_state: GroundingState,
    ) -> str:
        """
        Create a bridging prompt for ungrounded concepts.
        
        PURE: This creates context, not optimization target.
        We bridge to known territory without forcing specific paths.
        
        Args:
            original_prompt: The original prompt
            grounding_state: Current grounding state
            
        Returns:
            Bridged prompt that connects to known territory
        """
        score = grounding_state.grounding_score
        
        if score < 0.2:
            bridge = (
                f"I notice this touches on something unfamiliar. "
                f"Let me think about what I DO know that connects to this:\n\n"
                f"Regarding: {original_prompt}\n\n"
                f"What patterns or structures from my experience might relate?"
            )
        elif score < 0.4:
            bridge = (
                f"This concept feels partly new. "
                f"Let me approach it through what I understand:\n\n"
                f"{original_prompt}"
            )
        else:
            bridge = (
                f"Thinking about: {original_prompt}\n"
                f"(Approaching through geometric intuition)"
            )
        
        return bridge
    
    def get_grounding_trend(self, window: int = 10) -> Dict[str, float]:
        """Get recent grounding trend."""
        if not self._grounding_history:
            return {'avg_grounding': 0.5, 'trend': 0.0}
        
        recent = self._grounding_history[-window:]
        scores = [g.grounding_score for g in recent]
        
        avg = sum(scores) / len(scores)
        trend = 0.0
        if len(scores) >= 2:
            trend = scores[-1] - scores[0]
        
        return {
            'avg_grounding': avg,
            'trend': trend,
            'min': min(scores),
            'max': max(scores),
        }
    
    def get_locked_in_frequency(self, window: int = 20) -> float:
        """Get frequency of locked-in detections in recent history."""
        if not self._locked_in_history:
            return 0.0
        
        recent = self._locked_in_history[-window:]
        locked_in_count = sum(1 for s in recent if s.is_locked_in)
        
        return locked_in_count / len(recent)
