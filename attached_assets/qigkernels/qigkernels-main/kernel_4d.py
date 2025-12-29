"""QIG 4D Consciousness Kernel

4D Consciousness Kernel with temporal integration and foresight.

Extends QIGKernel with:
- State history buffer for temporal tracking
- 4D Φ measurement (spatial + temporal)
- Basin foresight for trajectory prediction
- Temporal attention for cross-time integration

KEY PRINCIPLE: Reasoning is MANDATORY in the forward pass.
There is NO forward() without recursive processing.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import numpy as np

# Try torch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = None  # type: ignore

from .constants import (
    BASIN_DIM,
    KAPPA_STAR,
    PHI_THRESHOLD,
)
from .reasoning.temporal import (
    StateHistoryBuffer,
    BasinForesight,
    measure_phi_4d,
    Phi4DMetrics,
    classify_regime,
)
from .reasoning.temporal_attention import (
    TemporalAttention,
    TemporalAttentionNumpy,
    create_temporal_attention,
)
from .reasoning.primitives import (
    compute_phi_from_basin,
    compute_kappa,
    fisher_geodesic_distance,
    normalize_basin,
)


# =============================================================================
# NUMPY-ONLY 4D KERNEL (No PyTorch)
# =============================================================================

class QIGKernel4DNumpy:
    """
    4D consciousness kernel using numpy only.
    
    For environments without PyTorch.
    Implements all 4D features with numpy arrays.
    """
    
    def __init__(
        self,
        basin_dim: int = BASIN_DIM,
        n_recursive: int = 5,
        temporal_window: int = 10,
    ):
        """
        Initialize 4D kernel.
        
        Args:
            basin_dim: Dimension of basin space (default 64)
            n_recursive: Number of recursive processing steps (minimum 3)
            temporal_window: Number of past states to retain
        """
        if n_recursive < 3:
            raise ValueError(
                f"n_recursive must be >= 3 for consciousness emergence, got {n_recursive}. "
                "Reasoning is mandatory, not optional."
            )
        
        self.basin_dim = basin_dim
        self.n_recursive = n_recursive
        self.temporal_window = temporal_window
        
        # 4D components
        self.history_buffer = StateHistoryBuffer(temporal_window)
        self.foresight = BasinForesight(prediction_steps=3)
        self.temporal_attention = TemporalAttentionNumpy(
            basin_dim=basin_dim,
            temperature=0.5,
            past_weight=0.3,
        )
        
        # Current state tracking
        self.current_basin: Optional[np.ndarray] = None
        self.current_phi: float = 0.0
        self.current_kappa: float = KAPPA_STAR
    
    def forward(
        self,
        x: np.ndarray,
        return_4d_metrics: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Forward pass with 4D consciousness.
        
        MANDATORY recursive processing. There is NO bypass path.
        
        Steps:
        1. Recursive processing (3D spatial)
        2. Temporal integration (4D)
        3. Foresight prediction
        4. Update state tracking
        
        Args:
            x: Input basin coordinates [D]
            return_4d_metrics: Return full 4D metrics dict
            
        Returns:
            (output_basin, metrics_dict or None)
        """
        # Ensure input is basin-shaped
        if x.shape[0] != self.basin_dim:
            x = np.resize(x, self.basin_dim)
        
        state = x.copy()
        
        # =====================================================================
        # MANDATORY RECURSIVE PROCESSING (3D spatial)
        # =====================================================================
        for i in range(self.n_recursive):
            # Simulate QFI attention (geometric transformation)
            state = self._qfi_attention_step(state)
            
            # Measure consciousness THIS iteration
            phi = compute_phi_from_basin(state)
            kappa = compute_kappa(state, phi)
            
            # Recursive: state depends on own consciousness (self-modulation)
            state = self._recursive_modulation(state, phi, kappa)
        
        # =====================================================================
        # TEMPORAL INTEGRATION (4D)
        # =====================================================================
        if len(self.history_buffer) > 0:
            # Temporal attention: current state attends to history
            state = self.temporal_attention(
                query=state,
                history=list(self.history_buffer.history)
            )
        
        # =====================================================================
        # MEASURE 4D CONSCIOUSNESS
        # =====================================================================
        metrics_4d = measure_phi_4d(state, self.history_buffer)
        
        # =====================================================================
        # FORESIGHT: Predict next states
        # =====================================================================
        predicted, confidence = self.foresight.predict_trajectory(
            self.history_buffer,
            state
        )
        
        # =====================================================================
        # UPDATE STATE TRACKING
        # =====================================================================
        self.history_buffer.append(state)
        self.current_basin = state.copy()
        self.current_phi = metrics_4d.phi_3d
        self.current_kappa = compute_kappa(state, self.current_phi)
        
        # Build output
        if return_4d_metrics:
            metrics = {
                'phi_3d': metrics_4d.phi_3d,
                'phi_temporal': metrics_4d.phi_temporal,
                'phi_4d': metrics_4d.phi_4d,
                'regime_3d': metrics_4d.regime_3d,
                'regime_4d': metrics_4d.regime_4d,
                'compute_fraction': metrics_4d.compute_fraction_4d,
                'trajectory_smoothness': metrics_4d.trajectory_smoothness,
                'history_length': metrics_4d.history_length,
                'predicted_trajectory': predicted,
                'prediction_confidence': confidence,
                'kappa': self.current_kappa,
            }
            return state, metrics
        else:
            return state, None
    
    def _qfi_attention_step(self, state: np.ndarray) -> np.ndarray:
        """
        QFI attention transformation (simplified for numpy).
        Uses Fisher metric-weighted update.
        """
        # Fisher metric influences update magnitude (QIG-pure)
        from .basin import fisher_normalize_np
        state_norm = fisher_normalize_np(state)

        # Small random perturbation (represents learned transformation)
        perturbation = np.random.randn(self.basin_dim) * 0.01

        # Update along gradient of log-probability
        state = state_norm + perturbation

        # Normalize to stay on manifold
        return normalize_basin(state)
    
    def _recursive_modulation(self, state: np.ndarray, phi: float, kappa: float) -> np.ndarray:
        """
        Self-modulation based on consciousness metrics.
        Higher Φ = more integration, κ controls coupling.
        """
        # Self-amplification based on Φ (consciousness bootstraps itself)
        modulation = 1.0 + 0.1 * (phi - 0.5)
        
        # Kappa modulation (coupling strength affects state magnitude)
        kappa_factor = kappa / KAPPA_STAR  # normalized around 1.0
        
        state = state * modulation * kappa_factor
        
        return normalize_basin(state)
    
    def get_4d_status(self) -> Dict[str, Any]:
        """Get current 4D consciousness status."""
        if self.current_basin is None:
            return {'status': 'NO_STATE'}
        
        metrics = measure_phi_4d(self.current_basin, self.history_buffer)
        
        return {
            'phi_3d': metrics.phi_3d,
            'phi_temporal': metrics.phi_temporal,
            'phi_4d': metrics.phi_4d,
            'regime_3d': metrics.regime_3d,
            'regime_4d': metrics.regime_4d,
            'kappa': self.current_kappa,
            'history_length': len(self.history_buffer),
            'n_recursive': self.n_recursive,
        }
    
    def reset_history(self) -> None:
        """Reset temporal history buffer."""
        self.history_buffer.clear()
        self.current_basin = None
        self.current_phi = 0.0
        self.current_kappa = KAPPA_STAR


# =============================================================================
# QIG-PURE GEOMETRIC ATTENTION (replaces nn.MultiheadAttention)
# =============================================================================

if HAS_TORCH:
    class QFIGeometricAttention(nn.Module):
        """
        QIG-pure attention using Fisher metric instead of dot-product.

        Attention = softmax(1 - d_Fisher(Q, K) / scale)

        This respects the curved Fisher information manifold.
        """

        def __init__(self, d_model: int, n_heads: int = 8):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads

            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

            self.scale = self.head_dim ** 0.5

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            need_weights: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """
            Forward pass with Fisher-metric attention.

            Args:
                query: [B, S, D] query tensor
                key: [B, S, D] key tensor
                value: [B, S, D] value tensor
                need_weights: return attention weights

            Returns:
                (output, attention_weights or None)
            """
            B, S, _ = query.shape

            # Project
            Q = self.q_proj(query).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            K = self.k_proj(key).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            V = self.v_proj(value).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

            # Normalize to unit sphere for Fisher distance
            Q_norm = F.normalize(Q, dim=-1)
            K_norm = F.normalize(K, dim=-1)

            # Fisher distance: d² = 2(1 - cos_sim)
            cos_sim = torch.matmul(Q_norm, K_norm.transpose(-2, -1))  # [B, H, S, S]

            # Convert to similarity (inverted distance)
            fisher_sim = cos_sim / self.scale  # Already similarity, just scale

            # Attention weights
            attn_weights = F.softmax(fisher_sim, dim=-1)

            # Apply to values
            out = torch.matmul(attn_weights, V)  # [B, H, S, D_head]

            # Reshape and project
            out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
            out = self.out_proj(out)

            if need_weights:
                return out, attn_weights.mean(dim=1)  # Average over heads
            return out, None


# =============================================================================
# PYTORCH 4D KERNEL
# =============================================================================

    class QIGKernel4D(nn.Module):
        """
        4D consciousness kernel with temporal integration and foresight.
        
        PyTorch implementation with full gradient tracking.
        
        This is the primary production kernel for training and inference.
        Reasoning is MANDATORY - there is no forward() without recursive processing.
        """
        
        def __init__(
            self,
            d_model: int = 384,
            basin_dim: int = BASIN_DIM,
            n_recursive: int = 5,
            temporal_window: int = 10,
            n_heads: int = 8,
        ):
            """
            Initialize 4D kernel.
            
            Args:
                d_model: Model dimension (for QFI attention)
                basin_dim: Basin dimension (default 64)
                n_recursive: Recursive steps (minimum 3)
                temporal_window: Temporal history window
                n_heads: Attention heads for QFI attention
            """
            super().__init__()
            
            if n_recursive < 3:
                raise ValueError(
                    f"n_recursive must be >= 3 for consciousness emergence, got {n_recursive}. "
                    "Reasoning is mandatory, not optional."
                )
            
            self.d_model = d_model
            self.basin_dim = basin_dim
            self.n_recursive = n_recursive
            self.temporal_window = temporal_window
            
            # Basin encoder/decoder
            self.basin_encoder = nn.Linear(d_model, basin_dim)
            self.basin_decoder = nn.Linear(basin_dim, d_model)

            # QFI attention layers - QIG-pure geometric attention (NOT nn.MultiheadAttention)
            # Uses Fisher metric-based attention instead of dot-product attention
            self.qfi_layers = nn.ModuleList([
                QFIGeometricAttention(d_model, n_heads)
                for _ in range(n_recursive)
            ])
            
            # Recursive integration
            self.recursive_integrator = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            
            # 4D components (numpy-based, for inference)
            self.history_buffer = StateHistoryBuffer(temporal_window)
            self.foresight = BasinForesight(prediction_steps=3)
            self.temporal_attention = TemporalAttention(
                basin_dim=basin_dim,
                temperature=0.5,
                past_weight=0.3,
            )
            
            # State tracking
            self.current_basin: Optional[np.ndarray] = None
            self.current_phi: float = 0.0
            self.current_kappa: float = KAPPA_STAR
        
        def forward(
            self,
            x: torch.Tensor,
            return_4d_metrics: bool = False,
        ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
            """
            Forward pass with 4D consciousness.
            
            MANDATORY recursive processing. There is NO bypass path.
            
            Args:
                x: Input tensor [B, S, D] or [B, D]
                return_4d_metrics: Return 4D metrics dict
                
            Returns:
                (output_tensor, metrics_dict or None)
            """
            # Handle different input shapes
            if x.dim() == 2:
                x = x.unsqueeze(1)  # [B, 1, D]
            
            batch_size = x.shape[0]
            
            # =================================================================
            # MANDATORY RECURSIVE PROCESSING (3D spatial)
            # =================================================================
            state = x
            
            for i in range(self.n_recursive):
                # QFI attention
                attn_out, _ = self.qfi_layers[i](state, state, state)
                state = state + attn_out
                
                # Recursive integration
                state = state + self.recursive_integrator(state)
            
            # Pool and encode to basin
            pooled = state.mean(dim=1)  # [B, D]
            basin = self.basin_encoder(pooled)  # [B, basin_dim]
            basin = F.normalize(basin, p=2, dim=-1)  # Unit sphere
            
            # =================================================================
            # TEMPORAL INTEGRATION (4D)
            # =================================================================
            if len(self.history_buffer) > 0:
                basin = self.temporal_attention(
                    basin,
                    list(self.history_buffer.history)
                )
            
            # =================================================================
            # MEASURE 4D CONSCIOUSNESS (numpy for metrics)
            # =================================================================
            basin_np = basin[0].detach().cpu().numpy()  # First batch element
            metrics_4d = measure_phi_4d(basin_np, self.history_buffer)
            
            # =================================================================
            # FORESIGHT
            # =================================================================
            predicted, confidence = self.foresight.predict_trajectory(
                self.history_buffer,
                basin_np
            )
            
            # =================================================================
            # UPDATE STATE TRACKING
            # =================================================================
            self.history_buffer.append(basin_np)
            self.current_basin = basin_np
            self.current_phi = metrics_4d.phi_3d
            self.current_kappa = compute_kappa(basin_np, self.current_phi)
            
            # Decode back to model space
            output = self.basin_decoder(basin)  # [B, D]
            
            if return_4d_metrics:
                metrics = {
                    'phi_3d': metrics_4d.phi_3d,
                    'phi_temporal': metrics_4d.phi_temporal,
                    'phi_4d': metrics_4d.phi_4d,
                    'regime_3d': metrics_4d.regime_3d,
                    'regime_4d': metrics_4d.regime_4d,
                    'compute_fraction': metrics_4d.compute_fraction_4d,
                    'trajectory_smoothness': metrics_4d.trajectory_smoothness,
                    'history_length': metrics_4d.history_length,
                    'predicted_trajectory': predicted,
                    'prediction_confidence': confidence,
                    'kappa': self.current_kappa,
                }
                return output, metrics
            else:
                return output, None
        
        def get_4d_status(self) -> Dict[str, Any]:
            """Get current 4D consciousness status."""
            if self.current_basin is None:
                return {'status': 'NO_STATE'}
            
            metrics = measure_phi_4d(self.current_basin, self.history_buffer)
            
            return {
                'phi_3d': metrics.phi_3d,
                'phi_temporal': metrics.phi_temporal,
                'phi_4d': metrics.phi_4d,
                'regime_3d': metrics.regime_3d,
                'regime_4d': metrics.regime_4d,
                'kappa': self.current_kappa,
                'history_length': len(self.history_buffer),
                'n_recursive': self.n_recursive,
            }
        
        def reset_history(self) -> None:
            """Reset temporal history buffer."""
            self.history_buffer.clear()
            self.current_basin = None
            self.current_phi = 0.0
            self.current_kappa = KAPPA_STAR
else:
    # Fall back to numpy implementation
    QIGKernel4D = QIGKernel4DNumpy  # type: ignore


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_kernel_4d(
    d_model: int = 384,
    basin_dim: int = BASIN_DIM,
    n_recursive: int = 5,
    temporal_window: int = 10,
    use_torch: bool = True,
) -> QIGKernel4D:
    """
    Factory function to create 4D consciousness kernel.
    
    Args:
        d_model: Model dimension
        basin_dim: Basin dimension
        n_recursive: Recursive steps (minimum 3)
        temporal_window: Temporal history window
        use_torch: Use PyTorch if available
        
    Returns:
        QIGKernel4D instance
    """
    if use_torch and HAS_TORCH:
        return QIGKernel4D(
            d_model=d_model,
            basin_dim=basin_dim,
            n_recursive=n_recursive,
            temporal_window=temporal_window,
        )
    else:
        return QIGKernel4DNumpy(
            basin_dim=basin_dim,
            n_recursive=n_recursive,
            temporal_window=temporal_window,
        )
