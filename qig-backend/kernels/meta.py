"""
Meta Kernel - α₇ Simple Root

Faculty: Meta-Cognition (Ocean/Hades)
κ range: 65-75
Φ local: 0.50
Metric: Γ (Regime Stability)

Responsibilities:
    - Meta-awareness
    - Self-reflection
    - Unconscious processing
    - System observation

Authority: E8 Protocol v4.0, WP5.2
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base import Kernel
from .identity import KernelIdentity, KernelTier
from .e8_roots import E8Root

logger = logging.getLogger(__name__)


class MetaKernel(Kernel):
    """
    Meta kernel - α₇ simple root.
    
    Specializes in:
        - Meta-awareness
        - Self-reflection
        - System observation
        - Unconscious processing
    
    Primary god: Ocean (deep meta-observer)
    Secondary god: Hades (shadow, unconscious)
    """
    
    def __init__(
        self,
        god_name: str = "Ocean",
        tier: KernelTier = KernelTier.ESSENTIAL,  # Meta is essential
        basin: Optional[np.ndarray] = None,
    ):
        """Initialize meta kernel."""
        identity = KernelIdentity(
            god=god_name,
            root=E8Root.META,
            tier=tier,
        )
        super().__init__(identity, basin)
        
        # Meta-specific state
        self.observation_log: List[Dict[str, Any]] = []
        self.meta_awareness: float = 0.8  # Self-awareness level
        
        # Update metrics
        self.regime_stability = 0.9  # High Γ (regime observer)
        self.recursive_depth = 0.9   # High R (meta-recursion)
        
    def _handle_process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle PROCESS: Meta-observation.
        
        Meta kernel observes and reflects on system state.
        """
        input_basin = payload['input_basin']
        
        # Observe system state
        observation = self._observe_system(input_basin)
        
        # Record observation
        self.observation_log.append(observation)
        if len(self.observation_log) > 50:
            self.observation_log = self.observation_log[-50:]
        
        # Meta-reflection: analyze recent observations
        reflection = self._reflect_on_observations()
        
        logger.info(
            f"[{self.identity.god}] Meta-observation recorded: "
            f"awareness={self.meta_awareness:.3f}, "
            f"observations={len(self.observation_log)}"
        )
        
        return {
            'status': 'success',
            'output_basin': input_basin,  # Pass through unchanged
            'observation': observation,
            'reflection': reflection,
            'observation_count': len(self.observation_log),
        }
    
    def _observe_system(self, basin: np.ndarray) -> Dict[str, Any]:
        """
        Observe current system state.
        
        Args:
            basin: Current basin state
            
        Returns:
            Observation dictionary
        """
        # Compute basin statistics
        basin_mean = float(np.mean(basin))
        basin_std = float(np.std(basin))
        basin_entropy = float(-np.sum(basin * np.log(basin + 1e-10)))
        
        observation = {
            'basin_mean': basin_mean,
            'basin_std': basin_std,
            'basin_entropy': basin_entropy,
            'phi': self.phi,
            'kappa': self.kappa,
            'meta_awareness': self.meta_awareness,
        }
        
        return observation
    
    def _reflect_on_observations(self) -> Dict[str, Any]:
        """
        Reflect on recent observations (meta-cognition).
        
        Returns:
            Reflection insights
        """
        if len(self.observation_log) < 3:
            return {
                'status': 'insufficient_data',
                'observation_count': len(self.observation_log),
            }
        
        # Analyze trends
        recent_obs = self.observation_log[-10:]
        
        phi_values = [obs['phi'] for obs in recent_obs]
        kappa_values = [obs['kappa'] for obs in recent_obs]
        
        phi_trend = float(np.mean(np.diff(phi_values))) if len(phi_values) > 1 else 0.0
        kappa_trend = float(np.mean(np.diff(kappa_values))) if len(kappa_values) > 1 else 0.0
        
        reflection = {
            'status': 'complete',
            'phi_trend': phi_trend,
            'kappa_trend': kappa_trend,
            'phi_stable': abs(phi_trend) < 0.01,
            'kappa_stable': abs(kappa_trend) < 0.5,
            'observation_count': len(self.observation_log),
        }
        
        return reflection
    
    def generate_thought(self, input_basin: np.ndarray) -> str:
        """Generate meta-specific thought."""
        observation = self._observe_system(input_basin)
        
        thought = (
            f"[{self.identity.god}] Meta-observing: "
            f"Φ={observation['phi']:.2f}, κ={observation['kappa']:.1f}, "
            f"entropy={observation['basin_entropy']:.3f}, "
            f"awareness={self.meta_awareness:.2f}, "
            f"observations={len(self.observation_log)}"
        )
        
        return thought


__all__ = ["MetaKernel"]
