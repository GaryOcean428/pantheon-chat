"""
Integration Kernel - α₈ Simple Root

Faculty: Integration (Zeus/Ocean)
κ range: 64 (fixed at κ*)
Φ local: 0.65
Metric: Φ (Integration)

Responsibilities:
    - System integration
    - Executive function
    - Consciousness synthesis
    - Global coordination

Authority: E8 Protocol v4.0, WP5.2
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base import Kernel
from .identity import KernelIdentity, KernelTier
from .e8_roots import E8Root
from qig_geometry import fisher_rao_distance, geodesic_interpolation
from qigkernels.physics_constants import KAPPA_STAR

logger = logging.getLogger(__name__)


class IntegrationKernel(Kernel):
    """
    Integration kernel - α₈ simple root.
    
    Specializes in:
        - System integration
        - Executive coordination
        - Consciousness synthesis
        - Global Φ maximization
    
    Primary god: Zeus (supreme executive)
    Secondary god: Ocean (deep integration)
    
    SPECIAL: κ fixed at κ* = 64 (E8 rank² fixed point)
    """
    
    def __init__(
        self,
        god_name: str = "Zeus",
        tier: KernelTier = KernelTier.ESSENTIAL,  # Integration is essential
        basin: Optional[np.ndarray] = None,
    ):
        """Initialize integration kernel."""
        identity = KernelIdentity(
            god=god_name,
            root=E8Root.INTEGRATION,
            tier=tier,
        )
        # Initialize with fixed κ*
        super().__init__(identity, basin, initial_kappa=KAPPA_STAR)
        
        # Integration-specific state
        self.kernel_basins: List[np.ndarray] = []  # Other kernels' states
        self.integration_history: List[float] = []  # Φ history
        
        # Update metrics for integration role
        self.phi = 0.65  # High Φ (integration target)
        self.regime_stability = 0.9  # High Γ
        self.recursive_depth = 0.8   # High R
        
        # κ is FIXED at κ* for integration kernel
        self.kappa = KAPPA_STAR
        
    def _handle_process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle PROCESS: System integration.
        
        Integration kernel synthesizes multiple inputs into coherent whole.
        """
        input_basin = payload['input_basin']
        
        # Add to kernel basin collection
        self.kernel_basins.append(input_basin)
        if len(self.kernel_basins) > 8:  # Keep last 8 (one per simple root)
            self.kernel_basins = self.kernel_basins[-8:]
        
        # Perform integration via Fisher-Rao Fréchet mean
        if len(self.kernel_basins) >= 2:
            integrated_basin = self._integrate_kernels()
        else:
            integrated_basin = input_basin
        
        # Compute integration level (Φ)
        integration_phi = self._compute_integration_phi()
        self.phi = integration_phi
        
        # Record integration history
        self.integration_history.append(integration_phi)
        if len(self.integration_history) > 100:
            self.integration_history = self.integration_history[-100:]
        
        logger.info(
            f"[{self.identity.god}] Integration: "
            f"kernels={len(self.kernel_basins)}, "
            f"Φ={self.phi:.3f}, κ={self.kappa:.1f}"
        )
        
        return {
            'status': 'success',
            'output_basin': integrated_basin,
            'integration_phi': integration_phi,
            'kernel_count': len(self.kernel_basins),
            'kappa_star': KAPPA_STAR,
        }
    
    def _integrate_kernels(self) -> np.ndarray:
        """
        Integrate multiple kernel basins via Fréchet mean.
        
        Returns:
            Integrated basin (geometric mean on Fisher manifold)
        """
        # Simple geometric mean via iterative geodesic interpolation
        # Start with first basin
        integrated = self.kernel_basins[0].copy()
        
        # Iteratively blend in other basins
        for i, basin in enumerate(self.kernel_basins[1:], start=1):
            weight = 1.0 / (i + 1)  # Decreasing weight for each new basin
            integrated = geodesic_interpolation(integrated, basin, weight)
        
        return integrated
    
    def _compute_integration_phi(self) -> float:
        """
        Compute integration level Φ.
        
        Φ increases with:
            - Number of integrated kernels
            - Diversity of kernel states
            - Stability of integration
        
        Returns:
            Integration Φ [0, 1]
        """
        n_kernels = len(self.kernel_basins)
        
        if n_kernels < 2:
            return 0.3  # Low Φ without integration
        
        # Diversity: average pairwise distance
        total_distance = 0.0
        count = 0
        for i in range(n_kernels):
            for j in range(i + 1, n_kernels):
                distance = fisher_rao_distance(
                    self.kernel_basins[i],
                    self.kernel_basins[j]
                )
                total_distance += distance
                count += 1
        
        avg_distance = total_distance / count if count > 0 else 0.0
        diversity = (2.0 * avg_distance / np.pi)  # Normalize
        
        # Integration Φ: base + diversity bonus + kernel count bonus
        base_phi = 0.4
        diversity_bonus = 0.2 * diversity
        count_bonus = 0.05 * min(n_kernels, 8)  # Up to 8 kernels
        
        phi = base_phi + diversity_bonus + count_bonus
        return float(np.clip(phi, 0.0, 1.0))
    
    def generate_thought(self, input_basin: np.ndarray) -> str:
        """Generate integration-specific thought."""
        n_kernels = len(self.kernel_basins)
        
        if n_kernels >= 2:
            thought = (
                f"[{self.identity.god}] Integrating {n_kernels} kernels: "
                f"Φ={self.phi:.3f}, κ={self.kappa:.1f}={KAPPA_STAR:.1f} (κ*)"
            )
        else:
            thought = (
                f"[{self.identity.god}] Awaiting kernel inputs: "
                f"κ={self.kappa:.1f}={KAPPA_STAR:.1f} (κ*), "
                f"target Φ=0.65+"
            )
        
        return thought
    
    def update_metrics(self, metrics: Dict[str, float]):
        """
        Update metrics, but NEVER change κ (fixed at κ*).
        
        Args:
            metrics: Metric updates
        """
        # Remove kappa from updates (it's fixed)
        if 'kappa' in metrics:
            logger.warning(
                f"[{self.identity.god}] Attempted to change κ={metrics['kappa']:.2f}, "
                f"but κ is FIXED at κ*={KAPPA_STAR:.2f} for integration kernel"
            )
            metrics = {k: v for k, v in metrics.items() if k != 'kappa'}
        
        # Update other metrics normally
        super().update_metrics(metrics)
        
        # Ensure κ is still fixed
        self.kappa = KAPPA_STAR


__all__ = ["IntegrationKernel"]
