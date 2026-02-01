"""
Action Kernel - α₅ Simple Root

Faculty: Action (Ares/Hermes)
κ range: 48-58
Φ local: 0.43
Metric: T (Temporal Coherence)

Responsibilities:
    - Action execution
    - Motor control
    - Output generation
    - Behavioral sequences

Authority: E8 Protocol v4.0, WP5.2
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base import Kernel
from .identity import KernelIdentity, KernelTier
from .e8_roots import E8Root
from qig_geometry import fisher_rao_distance

logger = logging.getLogger(__name__)


class ActionKernel(Kernel):
    """
    Action kernel - α₅ simple root.
    
    Specializes in:
        - Action execution
        - Output generation
        - Motor sequences
        - Behavioral control
    
    Primary god: Ares (energy, drive)
    Secondary god: Hermes (communication, action)
    """
    
    def __init__(
        self,
        god_name: str = "Ares",
        tier: KernelTier = KernelTier.PANTHEON,
        basin: Optional[np.ndarray] = None,
    ):
        """Initialize action kernel."""
        identity = KernelIdentity(
            god=god_name,
            root=E8Root.ACTION,
            tier=tier,
        )
        super().__init__(identity, basin)
        
        # Action-specific state
        self.action_threshold: float = 0.4  # Minimum activation for action
        self.action_history: List[str] = []
        
        # Update metrics
        self.temporal_coherence = 0.8  # High T (time sequencing)
        self.external_coupling = 0.6   # Good C (output to external)
        
    def _handle_output(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle OUTPUT: Action execution.
        
        Action kernel specializes in:
            - Activation thresholding
            - Action sequencing
            - Output formatting
        """
        basin = payload['basin']
        output_format = payload.get('format', 'text')
        
        # Compute action activation (distance from rest state)
        activation = self._compute_activation(basin)
        
        if activation < self.action_threshold:
            logger.info(
                f"[{self.identity.god}] Action below threshold "
                f"({activation:.3f} < {self.action_threshold:.3f}), "
                "no action"
            )
            return {
                'status': 'suppressed',
                'activation': activation,
                'reason': 'below_threshold',
            }
        
        # Generate action output
        thought = self.generate_thought(basin)
        
        # Record action
        self.action_history.append(thought)
        if len(self.action_history) > 20:
            self.action_history = self.action_history[-20:]
        
        logger.info(
            f"[{self.identity.god}] Action executed: "
            f"activation={activation:.3f}, "
            f"history={len(self.action_history)}"
        )
        
        return {
            'status': 'success',
            'thought': thought,
            'format': output_format,
            'activation': activation,
            'action_count': len(self.action_history),
        }
    
    def _compute_activation(self, basin: np.ndarray) -> float:
        """
        Compute action activation level.
        
        Args:
            basin: Basin state to evaluate
            
        Returns:
            Activation level [0, 1]
        """
        # Distance from kernel's rest state indicates activation
        distance = fisher_rao_distance(self.basin, basin)
        activation = (2.0 * distance / np.pi)  # Normalize to [0, 1]
        return float(np.clip(activation, 0.0, 1.0))
    
    def generate_thought(self, input_basin: np.ndarray) -> str:
        """Generate action-specific thought."""
        activation = self._compute_activation(input_basin)
        
        if activation < self.action_threshold:
            intensity = "idle"
        elif activation < 0.7:
            intensity = "moderate"
        else:
            intensity = "vigorous"
        
        thought = (
            f"[{self.identity.god}] Executing {intensity} action: "
            f"activation={activation:.3f}, "
            f"sequence={len(self.action_history)}, "
            f"κ={self.kappa:.1f}, Φ={self.phi:.2f}"
        )
        
        return thought
    
    def set_action_threshold(self, threshold: float):
        """Set action activation threshold."""
        self.action_threshold = float(np.clip(threshold, 0.0, 1.0))
        logger.info(
            f"[{self.identity.god}] Action threshold set to {self.action_threshold:.2f}"
        )


__all__ = ["ActionKernel"]
