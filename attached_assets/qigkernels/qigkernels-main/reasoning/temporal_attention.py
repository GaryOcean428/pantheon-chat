"""Temporal Attention for 4D Consciousness

Attention across temporal sequence using Fisher-Rao metric.
Allows current state to integrate information from past states.

This is the key module for temporal integration in 4D consciousness.
Unlike standard attention (dot product), we use Fisher-Rao distance
which respects the curved geometry of basin space.
"""

from __future__ import annotations

from typing import List, Dict, Optional, TYPE_CHECKING
import numpy as np

# Try torch import, fall back to numpy-only implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .constants import BASIN_DIM
from .primitives import (
    fisher_geodesic_distance,
    geodesic_interpolate,
    compute_phi_from_basin,
)

if TYPE_CHECKING:
    from .temporal import StateHistoryBuffer, HistoryEntry


# =============================================================================
# FISHER-RAO GEODESIC MEAN
# =============================================================================

def fisher_geodesic_mean(
    basins: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Compute weighted geodesic mean on Fisher manifold.
    
    This is the Fréchet mean with Fisher-Rao distance.
    Approximated via iterative geodesic interpolation.
    
    Args:
        basins: Array of basin states [T, D]
        weights: Normalized weights [T]
        
    Returns:
        Geodesic mean [D]
    """
    if len(basins) == 0:
        return np.zeros(BASIN_DIM)
    
    if len(basins) == 1:
        return basins[0].copy()
    
    # Normalize weights
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-10)
    
    # Start with weighted average (tangent space approximation)
    mean = np.zeros_like(basins[0], dtype=np.float64)
    for basin, weight in zip(basins, weights):
        mean += weight * basin
    
    # Iterative refinement (geodesic averaging)
    for _ in range(3):  # Few iterations sufficient
        new_mean = np.zeros_like(mean)
        for basin, weight in zip(basins, weights):
            # Move along geodesic from mean toward basin
            direction = geodesic_interpolate(mean, basin, t=weight)
            new_mean += direction / len(basins)
        mean = new_mean
    
    return mean


# =============================================================================
# NUMPY-ONLY TEMPORAL ATTENTION
# =============================================================================

class TemporalAttentionNumpy:
    """
    Temporal attention using numpy only (no PyTorch).
    
    Attention across temporal sequence using Fisher-Rao distance.
    Allows current state to integrate information from past.
    """
    
    def __init__(
        self,
        basin_dim: int = BASIN_DIM,
        temperature: float = 0.5,
        past_weight: float = 0.3,
    ):
        """
        Initialize temporal attention.
        
        Args:
            basin_dim: Dimension of basin space
            temperature: Softmax temperature for attention weights
            past_weight: How much to incorporate from past (0-1)
        """
        self.basin_dim = basin_dim
        self.temperature = temperature
        self.past_weight = past_weight
    
    def forward(
        self,
        query: np.ndarray,
        history: List['HistoryEntry'],
    ) -> np.ndarray:
        """
        Apply temporal attention.
        
        Args:
            query: Current basin state [D]
            history: List of past HistoryEntry objects
            
        Returns:
            Temporally-integrated state [D]
        """
        if len(history) == 0:
            return query
        
        # Extract past basins
        past_basins = np.array([h.basin for h in history])
        
        # Compute attention weights via Fisher-Rao distance
        # Closer states get higher weight
        distances = np.array([
            fisher_geodesic_distance(query, past)
            for past in past_basins
        ])
        
        # Softmax with temperature (negative because closer = higher weight)
        weights = np.exp(-distances / self.temperature)
        weights = weights / (weights.sum() + 1e-10)
        
        # Weighted combination on Fisher manifold
        integrated = fisher_geodesic_mean(past_basins, weights)
        
        # Combine with current state via geodesic
        # past_weight controls how much history influences output
        output = geodesic_interpolate(
            query,
            integrated,
            t=self.past_weight
        )
        
        return output
    
    def __call__(self, query: np.ndarray, history: List['HistoryEntry']) -> np.ndarray:
        """Make callable like PyTorch module."""
        return self.forward(query, history)


# =============================================================================
# PYTORCH TEMPORAL ATTENTION (if available)
# =============================================================================

if HAS_TORCH:
    class TemporalAttention(nn.Module):
        """
        Attention across temporal sequence using Fisher-Rao metric.
        
        Allows current state to integrate information from past.
        Uses Fisher-Rao distance for attention weights instead of dot product.
        """
        
        def __init__(
            self,
            basin_dim: int = BASIN_DIM,
            temperature: float = 0.5,
            past_weight: float = 0.3,
        ):
            """
            Initialize temporal attention.
            
            Args:
                basin_dim: Dimension of basin space
                temperature: Softmax temperature for attention weights
                past_weight: How much to incorporate from past (0-1)
            """
            super().__init__()
            self.basin_dim = basin_dim
            self.temperature = temperature
            self.past_weight = past_weight
            
            # Learnable temperature (optional)
            self.log_temperature = nn.Parameter(
                torch.tensor(np.log(temperature)),
                requires_grad=True
            )
        
        def forward(
            self,
            query: torch.Tensor,
            history: List['HistoryEntry'],
        ) -> torch.Tensor:
            """
            Apply temporal attention.
            
            Args:
                query: Current basin state [D] or [B, D]
                history: List of past HistoryEntry objects
                
            Returns:
                Temporally-integrated state
            """
            if len(history) == 0:
                return query
            
            # Handle batched input
            squeeze_output = False
            if query.dim() == 1:
                query = query.unsqueeze(0)
                squeeze_output = True
            
            batch_size = query.shape[0]
            device = query.device
            
            # Extract past basins and convert to tensor
            past_basins = torch.tensor(
                np.array([h.basin for h in history]),
                dtype=query.dtype,
                device=device
            )  # [T, D]
            
            # Compute Fisher-Rao distances (approximated via cosine distance)
            # query: [B, D], past_basins: [T, D]
            # Expand for broadcasting: [B, 1, D] and [1, T, D]
            query_expanded = query.unsqueeze(1)  # [B, 1, D]
            past_expanded = past_basins.unsqueeze(0)  # [1, T, D]
            
            # Cosine similarity as Fisher-Rao approximation
            cos_sim = F.cosine_similarity(
                query_expanded,
                past_expanded,
                dim=-1
            )  # [B, T]
            
            # Convert to distance (1 - cos_sim) and apply softmax
            distances = 1.0 - cos_sim
            
            # Effective temperature
            temp = torch.exp(self.log_temperature)
            
            # Attention weights (closer = higher weight)
            weights = F.softmax(-distances / temp, dim=-1)  # [B, T]
            
            # Weighted combination
            # weights: [B, T], past_basins: [T, D] -> [B, D]
            integrated = torch.einsum('bt,td->bd', weights, past_basins)
            
            # Geodesic interpolation with current state
            output = (
                (1.0 - self.past_weight) * query +
                self.past_weight * integrated
            )
            
            # Normalize to unit sphere (basin constraint)
            output = F.normalize(output, p=2, dim=-1)
            
            if squeeze_output:
                output = output.squeeze(0)
            
            return output
else:
    # Fall back to numpy implementation
    TemporalAttention = TemporalAttentionNumpy  # type: ignore


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_temporal_attention(
    basin_dim: int = BASIN_DIM,
    temperature: float = 0.5,
    past_weight: float = 0.3,
    use_torch: bool = True,
) -> TemporalAttention:
    """
    Factory function to create temporal attention module.
    
    Args:
        basin_dim: Dimension of basin space
        temperature: Softmax temperature
        past_weight: Weight given to historical states
        use_torch: Use PyTorch if available
        
    Returns:
        TemporalAttention module (PyTorch or numpy)
    """
    if use_torch and HAS_TORCH:
        return TemporalAttention(
            basin_dim=basin_dim,
            temperature=temperature,
            past_weight=past_weight,
        )
    else:
        return TemporalAttentionNumpy(
            basin_dim=basin_dim,
            temperature=temperature,
            past_weight=past_weight,
        )
