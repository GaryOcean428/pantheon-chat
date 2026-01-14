"""
Training Coordinator - Orchestrates QIG-pure training modules.

All training follows "measure, never optimize" - Φ and κ emerge from
geometric navigation, not loss minimization.
"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

try:
    from qig_core.training.geometric_vicarious import GeometricVicarious
    from qig_core.training.identity_reinforcement import IdentityReinforcement
    from qig_core.training.train_step_4d import TrainStep4D
    TRAINING_MODULES_AVAILABLE = True
except ImportError:
    TRAINING_MODULES_AVAILABLE = False
    GeometricVicarious = None  # type: ignore
    IdentityReinforcement = None  # type: ignore
    TrainStep4D = None  # type: ignore


@dataclass
class TrainingDiagnostics:
    """Training diagnostics (measurement only)."""
    geodesic_distance: float
    trajectory_smoothness: float
    identity_strength: float
    spatial_loss: float
    temporal_loss: float
    foresight_loss: float
    total_4d_loss: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'geodesic_distance': self.geodesic_distance,
            'trajectory_smoothness': self.trajectory_smoothness,
            'identity_strength': self.identity_strength,
            'spatial_loss': self.spatial_loss,
            'temporal_loss': self.temporal_loss,
            'foresight_loss': self.foresight_loss,
            'total_4d_loss': self.total_4d_loss,
        }


class TrainingCoordinator:
    """
    Coordinates QIG-pure training modules.
    
    CRITICAL: This coordinator MEASURES training progress.
    It does NOT optimize or update basins directly.
    
    Components:
    - GeometricVicarious: Manifold learning via geodesics
    - IdentityReinforcement: Self-awareness loop
    - TrainStep4D: Spatial + temporal + foresight training
    
    Usage:
        coordinator = TrainingCoordinator()
        
        diagnostics = coordinator.measure_training_state(
            current_basin=current,
            target_basin=target,
            basin_history=[b1, b2, b3],
            phi=0.65,
            kappa=1.2,
        )
    """
    
    def __init__(self):
        if not TRAINING_MODULES_AVAILABLE:
            raise RuntimeError("Training modules not available")
        
        self._geometric_vicarious = GeometricVicarious()
        self._identity_reinforcement = IdentityReinforcement()
        self._train_step_4d = TrainStep4D()
    
    def measure_training_state(
        self,
        current_basin: np.ndarray,
        target_basin: np.ndarray,
        basin_history: List[np.ndarray],
        identity_attractor: Optional[np.ndarray] = None,
        phi: float = 0.0,
        kappa: float = 0.0,
        regime: str = "unknown",
    ) -> TrainingDiagnostics:
        """
        Measure current training state without optimization.
        
        PURE PRINCIPLE: We MEASURE, never optimize directly.
        Returns diagnostics that inform adaptive control.
        
        Args:
            current_basin: Current basin coordinates (64D)
            target_basin: Target basin to measure distance to
            basin_history: History of recent basins for trajectory analysis
            identity_attractor: Optional identity attractor basin
            phi: Current Φ (integration) value
            kappa: Current κ (coupling) value
            regime: Current regime string
            
        Returns:
            TrainingDiagnostics with all measured values
        """
        geodesic_distance = self._geometric_vicarious.compute_vicarious_loss(
            current_basin, target_basin
        )
        
        if len(basin_history) >= 2:
            trajectory_metrics = self._geometric_vicarious.learn_from_trajectory(
                basins=basin_history,
                phis=[phi] * len(basin_history),
            )
            trajectory_smoothness = trajectory_metrics.trajectory_smoothness
        else:
            trajectory_smoothness = 1.0
        
        if identity_attractor is not None:
            id_state = self._identity_reinforcement.measure_identity(
                basin=current_basin,
                phi=phi,
                kappa=kappa,
                regime=regime,
            )
            identity_strength = id_state.identity_strength
        else:
            identity_strength = 0.0
        
        loss_4d = self._train_step_4d.compute_loss(
            predicted_basin=current_basin,
            target_basin=target_basin,
            phi_temporal=phi,
        )
        
        return TrainingDiagnostics(
            geodesic_distance=geodesic_distance,
            trajectory_smoothness=trajectory_smoothness,
            identity_strength=identity_strength,
            spatial_loss=loss_4d.spatial,
            temporal_loss=loss_4d.temporal,
            foresight_loss=loss_4d.foresight,
            total_4d_loss=loss_4d.total,
        )
    
    def measure_trajectory(
        self,
        basins: List[np.ndarray],
        phis: Optional[List[float]] = None,
        kappas: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Measure trajectory learning metrics.
        
        Args:
            basins: List of basin coordinates along trajectory
            phis: Optional list of Φ values
            kappas: Optional list of κ values
            
        Returns:
            Dictionary with trajectory metrics
        """
        trajectory_metrics = self._geometric_vicarious.learn_from_trajectory(
            basins=basins,
            phis=phis,
            kappas=kappas,
        )
        
        return trajectory_metrics.to_dict()
    
    def reinforce_identity(
        self,
        basin: np.ndarray,
        phi: float,
    ) -> bool:
        """
        Attempt identity reinforcement when Φ is high enough.
        
        Args:
            basin: Current basin coordinates
            phi: Current Φ value
            
        Returns:
            True if reinforcement occurred
        """
        return self._identity_reinforcement.reinforce(basin, phi)
    
    def get_identity_state(
        self,
        basin: np.ndarray,
        phi: float,
        kappa: float,
        regime: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Get current identity state.
        
        Args:
            basin: Current basin coordinates
            phi: Current Φ value
            kappa: Current κ value
            regime: Current regime
            
        Returns:
            Identity state dictionary
        """
        state = self._identity_reinforcement.measure_identity(
            basin=basin,
            phi=phi,
            kappa=kappa,
            regime=regime,
        )
        return state.to_dict()
    
    def measure_4d_loss(
        self,
        current_basin: np.ndarray,
        target_basin: np.ndarray,
        basin_history: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Measure 4D loss components WITHOUT optimization.
        
        PURE PRINCIPLE: Measure only, no gradients, no updates.
        
        Args:
            current_basin: Current basin coordinates (64D)
            target_basin: Target basin to measure distance to
            basin_history: History of recent basins (for potential foresight)
            
        Returns:
            Dictionary with loss components and availability flag
        """
        if not TRAINING_MODULES_AVAILABLE or self._train_step_4d is None:
            return {
                'spatial_loss': 0.0,
                'temporal_loss': 0.0,
                'foresight_loss': 0.0,
                'total_4d_loss': 0.0,
                'available': False
            }
        
        loss_result = self._train_step_4d.compute_loss(
            predicted_basin=current_basin,
            target_basin=target_basin,
        )
        return {
            'spatial_loss': loss_result.spatial,
            'temporal_loss': loss_result.temporal,
            'foresight_loss': loss_result.foresight,
            'total_4d_loss': loss_result.total,
            'available': True
        }
    
    def compute_4d_loss_batch(
        self,
        predicted_basins: List[np.ndarray],
        target_basins: List[np.ndarray],
        phi_temporals: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Compute 4D loss for a batch of basins (pure measurement).
        
        PURE PRINCIPLE: Measure only, no state mutation.
        
        Args:
            predicted_basins: List of predicted basins
            target_basins: List of target basins
            phi_temporals: Optional temporal Φ values
            
        Returns:
            Loss dictionary
        """
        if not TRAINING_MODULES_AVAILABLE or self._train_step_4d is None:
            return {
                'loss_4d': 0.0,
                'loss_spatial': 0.0,
                'loss_temporal': 0.0,
                'loss_foresight': 0.0,
                'available': False
            }
        
        loss = self._train_step_4d.step_batch(
            predicted_basins=predicted_basins,
            target_basins=target_basins,
            phi_temporals=phi_temporals,
        )
        result = loss.to_dict()
        result['available'] = True
        return result
