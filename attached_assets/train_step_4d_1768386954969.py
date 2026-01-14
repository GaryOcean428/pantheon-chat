"""4D Training Step

Training loss with temporal coherence and foresight accuracy.

KEY PRINCIPLE: Training loss includes ALL dimensions of consciousness:
1. Spatial loss (3D): Basin accuracy at each step
2. Temporal loss (4D): Trajectory smoothness
3. Foresight loss: Prediction accuracy

This trains the model to:
- Produce accurate spatial representations
- Maintain smooth trajectories through basin space
- Predict future states accurately
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Try torch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from qigkernels.reasoning import (
    fisher_geodesic_distance,
)


# =============================================================================
# LOSS COMPONENTS
# =============================================================================

def compute_spatial_loss(
    predicted_basin: np.ndarray,
    target_basin: np.ndarray,
) -> float:
    """
    Compute spatial (3D) loss via Fisher-Rao geodesic distance.
    
    This measures how far the predicted basin is from the target
    in the curved geometry of basin space.
    
    Args:
        predicted_basin: Predicted 64D basin coordinates
        target_basin: Target 64D basin coordinates
        
    Returns:
        Spatial loss (lower = better)
    """
    return fisher_geodesic_distance(predicted_basin, target_basin)


def compute_temporal_loss(
    phi_temporal: float,
    target_temporal_phi: float = 0.7,
) -> float:
    """
    Compute temporal coherence loss.
    
    Penalizes jerky trajectories (low Φ_temporal).
    Rewards smooth geodesic movement through basin space.
    
    Args:
        phi_temporal: Measured temporal Φ
        target_temporal_phi: Target temporal coherence (default 0.7)
        
    Returns:
        Temporal loss (lower = better)
    """
    # Penalize being below target
    if phi_temporal < target_temporal_phi:
        return (target_temporal_phi - phi_temporal) ** 2
    else:
        return 0.0


def compute_foresight_loss(
    predicted_trajectory: Optional[List[np.ndarray]],
    actual_trajectory: List[np.ndarray],
) -> float:
    """
    Compute foresight prediction loss.
    
    Measures how accurate the model's trajectory predictions are.
    
    Args:
        predicted_trajectory: Predicted future basins (or None)
        actual_trajectory: Actual basins that occurred
        
    Returns:
        Foresight loss (lower = better)
    """
    if predicted_trajectory is None or len(predicted_trajectory) == 0:
        return 0.0

    n_compare = min(len(predicted_trajectory), len(actual_trajectory))
    if n_compare == 0:
        return 0.0

    losses = [
        fisher_geodesic_distance(p, a)
        for p, a in zip(predicted_trajectory[:n_compare], actual_trajectory[:n_compare])
    ]

    return float(np.mean(losses))


# =============================================================================
# NUMPY TRAINING STEP
# =============================================================================

def train_step_4d_numpy(
    model: Any,
    batch: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    4D training step using numpy.
    
    For non-PyTorch environments. Returns loss components for manual
    gradient computation or logging.
    
    Args:
        model: Model with forward() method returning (basin, metrics)
        batch: List of training examples with 'input', 'target', optional 'trajectory'
        weights: Loss component weights (spatial, temporal, foresight)
        
    Returns:
        Dict with loss components
    """
    weights = weights or {
        'spatial': 1.0,
        'temporal': 0.3,
        'foresight': 0.2,
    }

    total_spatial = 0.0
    total_temporal = 0.0
    total_foresight = 0.0

    for example in batch:
        input_basin = example['input']
        target_basin = example['target']
        target_trajectory = example.get('trajectory', [])

        # Forward pass with 4D tracking
        output, metrics = model.forward(input_basin, return_4d_metrics=True)

        if metrics is None:
            metrics = {'phi_temporal': 0.5, 'predicted_trajectory': None}

        # Spatial loss (3D)
        loss_spatial = compute_spatial_loss(output, target_basin)
        total_spatial += loss_spatial

        # Temporal loss (4D)
        loss_temporal = compute_temporal_loss(metrics.get('phi_temporal', 0.5))
        total_temporal += loss_temporal

        # Foresight loss
        loss_foresight = compute_foresight_loss(
            metrics.get('predicted_trajectory'),
            target_trajectory
        )
        total_foresight += loss_foresight

    n = len(batch)

    # Combined 4D loss
    loss_4d = (
        weights['spatial'] * (total_spatial / n) +
        weights['temporal'] * (total_temporal / n) +
        weights['foresight'] * (total_foresight / n)
    )

    return {
        'loss_4d': loss_4d,
        'loss_spatial': total_spatial / n,
        'loss_temporal': total_temporal / n,
        'loss_foresight': total_foresight / n,
    }


# =============================================================================
# PYTORCH TRAINING STEP
# =============================================================================

if HAS_TORCH:
    def train_step_4d(
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        4D training step with PyTorch.
        
        Training loss includes ALL dimensions:
        1. Spatial loss: Basin accuracy
        2. Temporal loss: Trajectory smoothness
        3. Foresight loss: Prediction accuracy
        
        Args:
            model: PyTorch model with forward() returning (output, metrics)
            batch: Dict with 'input', 'target', optional 'trajectory'
            optimizer: PyTorch optimizer
            weights: Loss component weights
            
        Returns:
            Dict with loss components
        """
        weights = weights or {
            'spatial': 1.0,
            'temporal': 0.3,
            'foresight': 0.2,
        }

        model.train()
        optimizer.zero_grad()

        inputs = batch['input']  # [B, D] or [B, S, D]
        targets = batch['target']  # [B, D]

        # Forward pass with 4D tracking
        outputs, metrics = model(inputs, return_4d_metrics=True)

        # =====================================================================
        # SPATIAL LOSS (3D)
        # =====================================================================
        # Cosine distance approximates Fisher-Rao on unit sphere
        cos_sim = F.cosine_similarity(outputs, targets, dim=-1)
        loss_spatial = (1.0 - cos_sim).mean()

        # =====================================================================
        # TEMPORAL LOSS (4D)
        # =====================================================================
        if metrics is not None:
            phi_temporal = metrics.get('phi_temporal', 0.5)
            target_phi = 0.7

            if phi_temporal < target_phi:
                loss_temporal = torch.tensor(
                    (target_phi - phi_temporal) ** 2,
                    dtype=outputs.dtype,
                    device=outputs.device
                )
            else:
                loss_temporal = torch.tensor(
                    0.0,
                    dtype=outputs.dtype,
                    device=outputs.device
                )
        else:
            loss_temporal = torch.tensor(0.0, dtype=outputs.dtype, device=outputs.device)

        # =====================================================================
        # FORESIGHT LOSS
        # =====================================================================
        loss_foresight = torch.tensor(0.0, dtype=outputs.dtype, device=outputs.device)

        if metrics is not None and 'trajectory' in batch:
            predicted = metrics.get('predicted_trajectory')
            actual = batch['trajectory']  # [B, T, D]

            if predicted is not None and actual.shape[1] > 0:
                # Convert predictions to tensor
                n_compare = min(len(predicted), actual.shape[1])

                predicted_tensor = torch.tensor(
                    np.stack(predicted[:n_compare]),
                    dtype=outputs.dtype,
                    device=outputs.device
                )  # [n_compare, D]

                actual_tensor = actual[:, :n_compare, :]  # [B, n_compare, D]

                # Broadcast and compute cosine distance
                # Simplified: compare first batch element
                cos_sim_foresight = F.cosine_similarity(
                    predicted_tensor,
                    actual_tensor[0],
                    dim=-1
                )
                loss_foresight = (1.0 - cos_sim_foresight).mean()

        # =====================================================================
        # COMBINED 4D LOSS
        # =====================================================================
        loss_4d = (
            weights['spatial'] * loss_spatial +
            weights['temporal'] * loss_temporal +
            weights['foresight'] * loss_foresight
        )

        # Backward and optimize
        loss_4d.backward()
        optimizer.step()

        return {
            'loss_4d': loss_4d.item(),
            'loss_spatial': loss_spatial.item(),
            'loss_temporal': loss_temporal.item() if isinstance(loss_temporal, torch.Tensor) else loss_temporal,
            'loss_foresight': loss_foresight.item() if isinstance(loss_foresight, torch.Tensor) else loss_foresight,
        }


    class Loss4D(nn.Module):
        """
        4D Consciousness Loss Module.
        
        Combines spatial, temporal, and foresight losses.
        Can be used as a standalone loss function.
        """

        def __init__(
            self,
            spatial_weight: float = 1.0,
            temporal_weight: float = 0.3,
            foresight_weight: float = 0.2,
            target_temporal_phi: float = 0.7,
        ):
            """
            Initialize 4D loss.
            
            Args:
                spatial_weight: Weight for spatial (3D) loss
                temporal_weight: Weight for temporal coherence loss
                foresight_weight: Weight for prediction accuracy loss
                target_temporal_phi: Target Φ_temporal value
            """
            super().__init__()
            self.spatial_weight = spatial_weight
            self.temporal_weight = temporal_weight
            self.foresight_weight = foresight_weight
            self.target_temporal_phi = target_temporal_phi

        def forward(
            self,
            predicted: torch.Tensor,
            target: torch.Tensor,
            metrics: Optional[Dict[str, Any]] = None,
            predicted_trajectory: Optional[List[np.ndarray]] = None,
            actual_trajectory: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            """
            Compute 4D loss.
            
            Args:
                predicted: Predicted basin [B, D]
                target: Target basin [B, D]
                metrics: 4D metrics from forward pass
                predicted_trajectory: Foresight predictions
                actual_trajectory: Ground truth trajectory [B, T, D]
                
            Returns:
                (total_loss, loss_components_dict)
            """
            # Spatial loss
            cos_sim = F.cosine_similarity(predicted, target, dim=-1)
            loss_spatial = (1.0 - cos_sim).mean()

            # Temporal loss
            if metrics is not None:
                phi_temporal = metrics.get('phi_temporal', 0.5)
                if phi_temporal < self.target_temporal_phi:
                    loss_temporal = torch.tensor(
                        (self.target_temporal_phi - phi_temporal) ** 2,
                        dtype=predicted.dtype,
                        device=predicted.device
                    )
                else:
                    loss_temporal = torch.tensor(0.0, device=predicted.device)
            else:
                loss_temporal = torch.tensor(0.0, device=predicted.device)

            # Foresight loss
            loss_foresight = torch.tensor(0.0, device=predicted.device)
            if predicted_trajectory is not None and actual_trajectory is not None:
                n_compare = min(len(predicted_trajectory), actual_trajectory.shape[1])
                if n_compare > 0:
                    pred_t = torch.tensor(
                        np.stack(predicted_trajectory[:n_compare]),
                        dtype=predicted.dtype,
                        device=predicted.device
                    )
                    actual_t = actual_trajectory[:, :n_compare, :]

                    cos_sim_f = F.cosine_similarity(pred_t, actual_t[0], dim=-1)
                    loss_foresight = (1.0 - cos_sim_f).mean()

            # Combined
            total_loss = (
                self.spatial_weight * loss_spatial +
                self.temporal_weight * loss_temporal +
                self.foresight_weight * loss_foresight
            )

            components = {
                'spatial': loss_spatial,
                'temporal': loss_temporal,
                'foresight': loss_foresight,
            }

            return total_loss, components
else:
    train_step_4d = train_step_4d_numpy  # type: ignore
    Loss4D = None  # type: ignore


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'train_step_4d',
    'train_step_4d_numpy',
    'Loss4D',
    'compute_spatial_loss',
    'compute_temporal_loss',
    'compute_foresight_loss',
]
