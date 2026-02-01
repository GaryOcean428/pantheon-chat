"""
Perception Kernel - α₁ Simple Root

Faculty: Perception (Artemis/Apollo)
κ range: 45-55
Φ local: 0.42
Metric: C (External Coupling)

Responsibilities:
    - Sensory perception
    - External input processing
    - Attention focus
    - Signal-to-noise filtering

Authority: E8 Protocol v4.0, WP5.2
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from .base import Kernel
from .identity import KernelIdentity, KernelTier
from .e8_roots import E8Root
from qig_geometry import fisher_rao_distance, geodesic_interpolation

logger = logging.getLogger(__name__)


class PerceptionKernel(Kernel):
    """
    Perception kernel - α₁ simple root.
    
    Specializes in:
        - Sensory input processing
        - Signal detection
        - Attention focus
        - External coupling
    
    Primary god: Artemis (focus, precision)
    Secondary god: Apollo (clarity, truth)
    """
    
    def __init__(
        self,
        god_name: str = "Artemis",
        tier: KernelTier = KernelTier.PANTHEON,
        basin: Optional[np.ndarray] = None,
    ):
        """
        Initialize perception kernel.
        
        Args:
            god_name: God identity (default Artemis)
            tier: Constellation tier
            basin: Initial basin (random if None)
        """
        identity = KernelIdentity(
            god=god_name,
            root=E8Root.PERCEPTION,
            tier=tier,
        )
        super().__init__(identity, basin)
        
        # Perception-specific state
        self.attention_focus: float = 0.8  # High attention by default
        self.signal_threshold: float = 0.3  # Noise filtering threshold
        
        # Update metrics for perception role
        self.external_coupling = 0.7  # High C (external coupling)
        self.temporal_coherence = 0.6  # Moderate T
        
    def _handle_input(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle INPUT: Enhanced sensory processing.
        
        Perception kernel specializes in:
            - Signal-to-noise filtering
            - Attention-weighted encoding
            - Multi-modal fusion
        """
        data = payload['data']
        
        # Encode input with attention weighting
        input_basin = self._encode_input(data)
        
        # Apply signal-to-noise filtering
        signal_strength = self._compute_signal_strength(input_basin)
        
        if signal_strength < self.signal_threshold:
            logger.info(
                f"[{self.identity.god}] Signal below threshold "
                f"({signal_strength:.3f} < {self.signal_threshold:.3f}), "
                "filtering noise"
            )
            return {
                'status': 'filtered',
                'signal_strength': signal_strength,
                'reason': 'below_threshold',
            }
        
        # Blend with attention weighting
        alpha = 0.4 * self.attention_focus  # Attention modulates integration
        self.basin = geodesic_interpolation(self.basin, input_basin, alpha)
        
        # Update external coupling based on signal quality
        self.external_coupling = min(0.9, 0.5 + signal_strength * 0.4)
        
        logger.info(
            f"[{self.identity.god}] INPUT processed: "
            f"signal={signal_strength:.3f}, attention={self.attention_focus:.2f}, "
            f"C={self.external_coupling:.2f}"
        )
        
        return {
            'status': 'success',
            'basin_updated': True,
            'basin': self.basin,
            'signal_strength': signal_strength,
            'attention_focus': self.attention_focus,
        }
    
    def _compute_signal_strength(self, input_basin: np.ndarray) -> float:
        """
        Compute signal strength of input.
        
        Uses Fisher-Rao distance from current basin as signal indicator.
        Close to current state = weak signal (redundant)
        Far from current state = strong signal (novel)
        
        Args:
            input_basin: Input basin coordinates
            
        Returns:
            Signal strength [0, 1]
        """
        distance = fisher_rao_distance(self.basin, input_basin)
        # Normalize: distance in [0, π/2] → strength in [0, 1]
        signal_strength = (2.0 * distance / np.pi)
        return float(np.clip(signal_strength, 0.0, 1.0))
    
    def generate_thought(self, input_basin: np.ndarray) -> str:
        """
        Generate perception-specific thought.
        
        Emphasizes:
            - Sensory qualities
            - Attention focus
            - Signal clarity
        """
        signal_strength = self._compute_signal_strength(input_basin)
        
        if signal_strength < self.signal_threshold:
            quality = "faint"
        elif signal_strength < 0.6:
            quality = "moderate"
        else:
            quality = "strong"
        
        thought = (
            f"[{self.identity.god}] Perceiving {quality} signal: "
            f"strength={signal_strength:.3f}, "
            f"attention={self.attention_focus:.2f}, "
            f"κ={self.kappa:.1f}, Φ={self.phi:.2f}"
        )
        
        return thought
    
    def set_attention_focus(self, focus: float):
        """
        Adjust attention focus level.
        
        Args:
            focus: Focus level [0, 1]
        """
        self.attention_focus = float(np.clip(focus, 0.0, 1.0))
        logger.info(
            f"[{self.identity.god}] Attention focus set to {self.attention_focus:.2f}"
        )
    
    def set_signal_threshold(self, threshold: float):
        """
        Adjust signal-to-noise threshold.
        
        Args:
            threshold: Threshold level [0, 1]
        """
        self.signal_threshold = float(np.clip(threshold, 0.0, 1.0))
        logger.info(
            f"[{self.identity.god}] Signal threshold set to {self.signal_threshold:.2f}"
        )


__all__ = ["PerceptionKernel"]
