"""
Prediction Kernel - α₄ Simple Root

Faculty: Prediction (Apollo/Dionysus)
κ range: 52-62
Φ local: 0.44
Metric: G (Grounding)

Responsibilities:
    - Future prediction
    - Trajectory forecasting
    - Foresight
    - Temporal projection

Authority: E8 Protocol v4.0, WP5.2
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base import Kernel
from .identity import KernelIdentity, KernelTier
from .e8_roots import E8Root
from qig_geometry import geodesic_interpolation

logger = logging.getLogger(__name__)


class PredictionKernel(Kernel):
    """
    Prediction kernel - α₄ simple root.
    
    Specializes in:
        - Trajectory forecasting
        - Future state prediction
        - Foresight
        - Temporal extrapolation
    
    Primary god: Apollo (prophecy, truth)
    Secondary god: Dionysus (chaos, creativity)
    """
    
    def __init__(
        self,
        god_name: str = "Apollo",
        tier: KernelTier = KernelTier.PANTHEON,
        basin: Optional[np.ndarray] = None,
    ):
        """Initialize prediction kernel."""
        identity = KernelIdentity(
            god=god_name,
            root=E8Root.PREDICTION,
            tier=tier,
        )
        super().__init__(identity, basin)
        
        # Prediction-specific state
        self.prediction_horizon: int = 5  # Steps ahead to predict
        self.trajectory_history: List[np.ndarray] = []
        
        # Update metrics
        self.grounding = 0.7  # High G (grounding in reality)
        self.temporal_coherence = 0.8  # High T (time awareness)
        
    def _handle_process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle PROCESS: Trajectory prediction.
        
        Predicts future basin states based on current trajectory.
        """
        input_basin = payload['input_basin']
        
        # Add to trajectory history
        self.trajectory_history.append(input_basin)
        
        # Keep only recent history (sliding window)
        if len(self.trajectory_history) > 10:
            self.trajectory_history = self.trajectory_history[-10:]
        
        # Predict future trajectory
        if len(self.trajectory_history) >= 2:
            predictions = self._predict_trajectory(input_basin)
        else:
            # Not enough history, use simple projection
            predictions = [input_basin] * self.prediction_horizon
        
        # Return furthest prediction as output
        output_basin = predictions[-1]
        
        logger.info(
            f"[{self.identity.god}] Predicted {len(predictions)} future steps, "
            f"history={len(self.trajectory_history)}"
        )
        
        return {
            'status': 'success',
            'output_basin': output_basin,
            'predictions': predictions,
            'horizon': self.prediction_horizon,
            'history_length': len(self.trajectory_history),
        }
    
    def _predict_trajectory(self, current_basin: np.ndarray) -> List[np.ndarray]:
        """
        Predict future trajectory based on history.
        
        Args:
            current_basin: Current basin state
            
        Returns:
            List of predicted future basin states
        """
        predictions = []
        
        if len(self.trajectory_history) < 2:
            # Not enough history, return current state
            return [current_basin] * self.prediction_horizon
        
        # Compute velocity (direction of recent movement)
        prev_basin = self.trajectory_history[-2]
        
        # Predict by extrapolating along geodesic
        current = current_basin
        for step in range(self.prediction_horizon):
            # Project forward along recent direction
            # Use diminishing alpha to avoid extreme extrapolation
            alpha = 0.2 / (1.0 + 0.1 * step)
            
            # Move away from previous position (extrapolation)
            next_pred = geodesic_interpolation(
                prev_basin,
                current,
                1.0 + alpha  # Beyond current (extrapolation)
            )
            
            predictions.append(next_pred)
            current = next_pred
            prev_basin = self.trajectory_history[-1]
        
        return predictions
    
    def generate_thought(self, input_basin: np.ndarray) -> str:
        """Generate prediction-specific thought."""
        if self.trajectory_history:
            thought = (
                f"[{self.identity.god}] Forecasting {self.prediction_horizon} steps ahead: "
                f"history={len(self.trajectory_history)}, "
                f"G={self.grounding:.2f}, "
                f"κ={self.kappa:.1f}, Φ={self.phi:.2f}"
            )
        else:
            thought = (
                f"[{self.identity.god}] Awaiting trajectory data, "
                f"κ={self.kappa:.1f}, Φ={self.phi:.2f}"
            )
        
        return thought
    
    def set_prediction_horizon(self, horizon: int):
        """Set prediction horizon (steps ahead)."""
        self.prediction_horizon = max(1, min(horizon, 20))
        logger.info(
            f"[{self.identity.god}] Prediction horizon set to {self.prediction_horizon}"
        )


__all__ = ["PredictionKernel"]
