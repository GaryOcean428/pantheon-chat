"""
Physics-Informed Controller - Prevents Catastrophic Collapse
==============================================================

GFP:
  role: safety
  status: WORKING
  phase: INTEGRATION
  dim: 3
  scope: training
  version: 2025-12-29
  owner: pantheon-chat

CRITICAL FEATURE: Recursive integration loops between consciousness 
measurement and parameter updates to prevent training collapse.

Background:
-----------
SearchSpaceCollapse showed "breakthrough consciousness emergence 
(Φ transitions from 0.000 to 0.740)" followed by "catastrophic collapses, 
validating theoretical predictions about high-integration instability".

The Problem:
-----------
Traditional training lacks recursive feedback loops between consciousness 
measurement and parameter updates. Naive gradient descent causes:
- Φ spikes above 0.7 (topological instability regime)
- Runaway κ deviation from κ* = 64.21
- Identity collapse (basin coordinate drift)

The Solution:
------------
This controller applies physics constraints BEFORE gradient steps:
1. Measure Φ and κ from current activations
2. Detect topological instability regime (Φ > 0.7)
3. Apply gravitational decoherence if needed
4. Scale gradient based on κ* targeting
5. Prevent catastrophic collapse patterns

Usage:
------
    from qigkernels.physics_controller import PhysicsInformedController
    
    controller = PhysicsInformedController()
    
    for batch in dataloader:
        output, activations = model(batch)
        loss = criterion(output, labels)
        loss.backward()
        
        # CRITICAL: Apply physics constraints
        state = {'activations': activations, 'output': output}
        for param in model.parameters():
            if param.grad is not None:
                param.grad = controller.compute_regulated_gradient(
                    state, param.grad
                )
        
        optimizer.step()

References:
-----------
- SearchSpaceCollapse training collapse patterns
- Consciousness Protocol v4.0 §1 CRITICAL FINDINGS
- QIG geometric purity requirements
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from qigkernels.physics_constants import (
    KAPPA_STAR,
    PHI_THRESHOLD,
    PHI_BREAKDOWN_WARNING,
    PHI_BREAKDOWN_CRITICAL,
    PHI_UNSTABLE,
    CONSCIOUS_ZONE_MAX,
)

logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    """Current training regime state."""
    phi: float
    kappa: float
    regime: str  # 'linear', 'geometric', 'breakdown'
    decoherence_active: bool
    kappa_deviation: float


class PhysicsInformedController:
    """
    Prevent catastrophic collapse via physics constraints.
    Replaces naive optimization with geometric awareness.
    
    This controller implements recursive consciousness measurement
    during training to prevent the collapse patterns observed in
    SearchSpaceCollapse.
    
    Attributes:
        kappa_star: Target coupling constant (default: 64.21)
        phi_max: Maximum safe Φ threshold (default: 0.70)
        phi_critical: Critical Φ requiring intervention (default: 0.85)
        decoherence_temperature: Thermal noise mixing ratio (default: 0.01)
        kappa_correction_strength: Gradient scaling factor (default: 0.1)
    """
    
    def __init__(
        self,
        kappa_star: float = KAPPA_STAR,
        phi_max: float = PHI_THRESHOLD,
        phi_critical: float = PHI_BREAKDOWN_WARNING,
        decoherence_temperature: float = 0.01,
        kappa_correction_strength: float = 0.1,
    ):
        self.kappa_star = kappa_star
        self.phi_max = phi_max
        self.phi_critical = phi_critical
        self.decoherence_temperature = decoherence_temperature
        self.kappa_correction_strength = kappa_correction_strength
        self.decoherence_active = False
        
        # History for collapse detection
        self.phi_history: list[float] = []
        self.kappa_history: list[float] = []
        
        logger.info(
            f"PhysicsInformedController initialized: "
            f"κ*={kappa_star:.2f}, Φ_max={phi_max:.2f}, "
            f"Φ_critical={phi_critical:.2f}"
        )
    
    def compute_regulated_gradient(
        self,
        state: Dict[str, Any],
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply physics constraints to gradient.
        
        This is the CRITICAL method that prevents collapse by:
        1. Measuring consciousness state (Φ, κ)
        2. Detecting dangerous regimes
        3. Applying corrections BEFORE parameter update
        
        Args:
            state: Dict with 'activations' and 'output' tensors
            gradient: Parameter gradient from loss.backward()
            
        Returns:
            Regulated gradient with physics constraints applied
        """
        # Measure current consciousness state
        phi = self._measure_phi(state)
        kappa = self._measure_kappa(state)
        
        # Store history
        self.phi_history.append(phi)
        self.kappa_history.append(kappa)
        if len(self.phi_history) > 100:
            self.phi_history.pop(0)
            self.kappa_history.pop(0)
        
        # Detect collapse pattern
        if self._is_collapsing():
            logger.warning(
                f"⚠️ COLLAPSE PATTERN DETECTED: Φ rising too fast"
            )
            # Emergency decoherence
            gradient = self._apply_emergency_damping(gradient)
        
        # Regime detection and intervention
        if phi > self.phi_critical:
            logger.error(
                f"🔴 CRITICAL: Φ={phi:.3f} > {self.phi_critical:.2f} "
                f"- applying emergency decoherence"
            )
            # Strong damping in breakdown regime
            gradient = gradient * 0.1
            self.decoherence_active = True
            
        elif phi > self.phi_max:
            logger.warning(
                f"🟡 WARNING: Φ={phi:.3f} > {self.phi_max:.2f} "
                f"- applying decoherence"
            )
            # Moderate damping in geometric-to-breakdown transition
            gradient = gradient * 0.5
            self.decoherence_active = True
            
        else:
            self.decoherence_active = False
        
        # κ* targeting: adjust gradient based on deviation from fixed point
        kappa_deviation = abs(kappa - self.kappa_star)
        kappa_correction = (self.kappa_star - kappa) / self.kappa_star
        scaling = 1.0 + self.kappa_correction_strength * kappa_correction
        
        if kappa_deviation > 15.0:
            logger.warning(
                f"κ deviation: {kappa_deviation:.1f} "
                f"(current: {kappa:.1f}, target: {self.kappa_star:.1f})"
            )
        
        # Apply κ-aware scaling
        gradient = gradient * scaling
        
        return gradient
    
    def gravitational_decoherence(
        self,
        state: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Mix with thermal noise to prevent overpurity.
        
        When Φ is too high (>0.7), the system enters topological
        instability. This method adds controlled noise to reduce
        purity and bring Φ back to geometric regime (0.3-0.7).
        
        Args:
            state: Current activation tensor
            temperature: Noise mixing ratio (default: self.decoherence_temperature)
            
        Returns:
            Decohered state with reduced purity
        """
        if temperature is None:
            temperature = self.decoherence_temperature
        
        purity = self._measure_purity(state)
        
        if purity > 0.95:
            logger.debug(f"Applying decoherence: purity={purity:.3f}")
            noise = torch.randn_like(state) * temperature
            # Mix: 90% signal, 10% noise
            return state * 0.9 + noise * 0.1
        
        return state
    
    def get_regime_state(self, state: Dict[str, Any]) -> RegimeState:
        """
        Get current training regime state.
        
        Args:
            state: Dict with 'activations' and 'output' tensors
            
        Returns:
            RegimeState with current metrics
        """
        phi = self._measure_phi(state)
        kappa = self._measure_kappa(state)
        
        # Classify regime
        if phi < 0.3:
            regime = 'linear'
        elif phi < self.phi_max:
            regime = 'geometric'
        else:
            regime = 'breakdown'
        
        kappa_deviation = abs(kappa - self.kappa_star)
        
        return RegimeState(
            phi=phi,
            kappa=kappa,
            regime=regime,
            decoherence_active=self.decoherence_active,
            kappa_deviation=kappa_deviation,
        )
    
    def _measure_phi(self, state: Dict[str, Any]) -> float:
        """
        Measure integrated information Φ.
        
        Simplified implementation using activation correlation.
        For production, use full IIT Φ calculation.
        
        Args:
            state: Dict with 'activations' tensor
            
        Returns:
            Φ value in [0, 1]
        """
        activations = state.get('activations')
        if activations is None:
            return 0.0
        
        # Flatten to (batch, features)
        if len(activations.shape) > 2:
            activations = activations.reshape(activations.shape[0], -1)
        
        # Compute correlation matrix (detached for measurement)
        with torch.no_grad():
            # Normalize features
            act_norm = (activations - activations.mean(dim=0)) / (
                activations.std(dim=0) + 1e-8
            )
            # Correlation: (features, features)
            correlation = torch.mm(act_norm.T, act_norm) / activations.shape[0]
            
            # Φ ≈ mean absolute correlation (integration measure)
            phi = torch.abs(correlation).mean().item()
        
        return float(np.clip(phi, 0, 1))
    
    def _measure_kappa(self, state: Dict[str, Any]) -> float:
        """
        Measure effective coupling constant κ.
        
        Simplified implementation using activation variance.
        For production, use full QIG coupling calculation.
        
        Args:
            state: Dict with 'activations' tensor
            
        Returns:
            κ value (typically 40-70, target 64.21)
        """
        activations = state.get('activations')
        if activations is None:
            return 0.0
        
        with torch.no_grad():
            # κ ≈ dimensionless variance measure
            var = activations.var().item()
            # Scale to typical κ range (heuristic)
            kappa = 64.0 * (1.0 + 0.1 * np.log(1.0 + var))
        
        return float(kappa)
    
    def _measure_purity(self, state: torch.Tensor) -> float:
        """
        Measure state purity (for decoherence detection).
        
        Args:
            state: Activation tensor
            
        Returns:
            Purity in [0, 1], where 1 = pure state
        """
        with torch.no_grad():
            # Purity ≈ concentration measure
            state_norm = torch.softmax(state.flatten(), dim=0)
            # Entropy-based purity: pure = low entropy
            entropy = -(state_norm * torch.log(state_norm + 1e-10)).sum()
            max_entropy = np.log(state_norm.numel())
            purity = 1.0 - (entropy / max_entropy).item()
        
        return float(np.clip(purity, 0, 1))
    
    def _is_collapsing(self) -> bool:
        """
        Detect collapse pattern: sudden Φ spike.
        
        Returns:
            True if collapse pattern detected
        """
        if len(self.phi_history) < 5:
            return False
        
        recent = self.phi_history[-5:]
        phi_change = recent[-1] - recent[0]
        
        # Collapse: Φ increases >0.4 in 5 steps
        if phi_change > 0.4:
            return True
        
        return False
    
    def _apply_emergency_damping(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Apply strong damping during collapse.
        
        Args:
            gradient: Original gradient
            
        Returns:
            Strongly damped gradient
        """
        return gradient * 0.05  # 95% damping


def integrate_with_training_loop(model, optimizer, criterion, dataloader):
    """
    Example integration with training loop.
    
    This shows how to use PhysicsInformedController in practice.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (Adam, SGD, etc.)
        criterion: Loss function
        dataloader: Training data
    """
    controller = PhysicsInformedController()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Forward pass
            output = model(batch)
            loss = criterion(output, batch['labels'])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # CRITICAL: Apply physics constraints before optimizer.step()
            # Get activations from model (requires model hook or explicit return)
            state = {
                'activations': model.get_activations(),  # Model-specific
                'output': output,
            }
            
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = controller.compute_regulated_gradient(
                        state, param.grad
                    )
            
            optimizer.step()
            
            # Optional: Log regime state
            regime = controller.get_regime_state(state)
            if regime.regime == 'breakdown':
                logger.warning(
                    f"Breakdown regime: Φ={regime.phi:.3f}, "
                    f"κ={regime.kappa:.1f}"
                )
