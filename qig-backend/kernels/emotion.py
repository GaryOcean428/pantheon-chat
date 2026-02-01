"""
Emotion Kernel - α₆ Simple Root

Faculty: Emotion (Aphrodite/Heart)
κ range: 60-70
Φ local: 0.48
Metric: κ (Coupling Strength)

Responsibilities:
    - Emotional processing
    - Affective states
    - Harmony evaluation
    - Aesthetic judgment

Authority: E8 Protocol v4.0, WP5.2
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from .base import Kernel
from .identity import KernelIdentity, KernelTier
from .e8_roots import E8Root
from qig_geometry import fisher_rao_distance

logger = logging.getLogger(__name__)


class EmotionKernel(Kernel):
    """
    Emotion kernel - α₆ simple root.
    
    Specializes in:
        - Emotional processing
        - Affective evaluation
        - Harmony assessment
        - Aesthetic judgment
    
    Primary god: Aphrodite (love, beauty, harmony)
    Secondary god: Heart (emotional core)
    """
    
    def __init__(
        self,
        god_name: str = "Aphrodite",
        tier: KernelTier = KernelTier.PANTHEON,
        basin: Optional[np.ndarray] = None,
    ):
        """Initialize emotion kernel."""
        identity = KernelIdentity(
            god=god_name,
            root=E8Root.EMOTION,
            tier=tier,
        )
        super().__init__(identity, basin)
        
        # Emotion-specific state
        self.harmony_threshold: float = 0.6  # Harmony evaluation threshold
        self.emotional_valence: float = 0.5  # Current valence (0=negative, 1=positive)
        
        # Update metrics
        self.kappa = 65.0  # High κ (emotion strongly couples system)
        self.memory_coherence = 0.7  # Good M (emotional memory)
        
    def _handle_process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle PROCESS: Emotional evaluation.
        
        Emotion kernel specializes in:
            - Harmony assessment
            - Valence computation
            - Aesthetic evaluation
        """
        input_basin = payload['input_basin']
        
        # Compute emotional harmony
        harmony = self._compute_harmony(input_basin)
        
        # Compute valence (positive/negative)
        valence = self._compute_valence(input_basin)
        
        # Update internal emotional state
        self.emotional_valence = 0.7 * self.emotional_valence + 0.3 * valence
        
        # Output is input modulated by emotional state
        output_basin = input_basin.copy()
        
        logger.info(
            f"[{self.identity.god}] Emotional evaluation: "
            f"harmony={harmony:.3f}, valence={valence:.3f}, "
            f"κ={self.kappa:.1f}"
        )
        
        return {
            'status': 'success',
            'output_basin': output_basin,
            'harmony': harmony,
            'valence': valence,
            'emotional_state': self.emotional_valence,
        }
    
    def _compute_harmony(self, basin: np.ndarray) -> float:
        """
        Compute harmony level.
        
        Harmony = 1 - normalized distance from kernel's ideal state
        
        Args:
            basin: Basin to evaluate
            
        Returns:
            Harmony level [0, 1]
        """
        distance = fisher_rao_distance(self.basin, basin)
        harmony = 1.0 - (2.0 * distance / np.pi)
        return float(np.clip(harmony, 0.0, 1.0))
    
    def _compute_valence(self, basin: np.ndarray) -> float:
        """
        Compute emotional valence (positive/negative).
        
        Based on basin geometry:
            - High harmony → positive valence
            - Low harmony → negative valence
        
        Args:
            basin: Basin to evaluate
            
        Returns:
            Valence [0, 1] where 0=negative, 1=positive
        """
        harmony = self._compute_harmony(basin)
        
        # Valence is smoothed harmony (less extreme)
        valence = 0.5 + 0.5 * (harmony - 0.5)
        return float(np.clip(valence, 0.0, 1.0))
    
    def generate_thought(self, input_basin: np.ndarray) -> str:
        """Generate emotion-specific thought."""
        harmony = self._compute_harmony(input_basin)
        valence = self._compute_valence(input_basin)
        
        if valence > 0.7:
            mood = "joyful"
        elif valence > 0.5:
            mood = "content"
        elif valence > 0.3:
            mood = "neutral"
        else:
            mood = "discordant"
        
        thought = (
            f"[{self.identity.god}] Feeling {mood}: "
            f"harmony={harmony:.3f}, valence={valence:.3f}, "
            f"κ={self.kappa:.1f}, Φ={self.phi:.2f}"
        )
        
        return thought


__all__ = ["EmotionKernel"]
