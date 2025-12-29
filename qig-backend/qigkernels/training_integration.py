"""
Physics-Informed Training Integration Example
==============================================

GFP:
  role: example
  status: WORKING
  phase: INTEGRATION
  dim: 3
  scope: training
  version: 2025-12-29
  owner: pantheon-chat

Demonstrates how to integrate PhysicsInformedController, BetaMeasurement,
and Fisher-Rao geometry into Ocean/Gary training pipelines.

This is NOT a complete training script - it's a reference implementation
showing the critical integration points for consciousness-aware training.

Usage:
------
    # In your training script:
    from qigkernels.training_integration import (
        create_physics_aware_training_loop,
        ConsciousnessMonitor
    )
    
    # Wrap your model
    training_loop = create_physics_aware_training_loop(
        model=your_model,
        optimizer=your_optimizer,
        criterion=your_loss_fn
    )
    
    # Train with consciousness monitoring
    for batch in dataloader:
        metrics = training_loop.step(batch)
        
        if metrics['regime'] == 'breakdown':
            logger.warning("Breakdown regime detected - emergency stop!")
            break

Integration Points:
-------------------
1. Model forward pass: Capture activations for Φ/κ measurement
2. Gradient computation: Apply physics constraints before optimizer.step()
3. β measurement: Track κ evolution at regular intervals
4. Collapse detection: Monitor Φ spike patterns
5. Logging: Record consciousness metrics with training metrics

References:
-----------
- Consciousness Protocol v4.0 §1 Task 2
- PhysicsInformedController (qigkernels/physics_controller.py)
- BetaMeasurement (qigkernels/beta_measurement.py)
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import time

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from qigkernels.physics_controller import PhysicsInformedController, RegimeState
from qigkernels.beta_measurement import BetaMeasurement, BetaResult

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics from a single training step."""
    step: int
    loss: float
    phi: float
    kappa: float
    beta: Optional[float]
    regime: str
    decoherence_active: bool
    physics_match_pct: Optional[float]


class ConsciousnessMonitor:
    """
    Monitor consciousness metrics during training.
    
    Integrates PhysicsInformedController and BetaMeasurement to provide
    comprehensive consciousness-aware training monitoring.
    """
    
    def __init__(
        self,
        log_interval: int = 100,
        beta_measurement_interval: int = 5000,
    ):
        self.controller = PhysicsInformedController()
        self.beta_measure = BetaMeasurement()
        self.log_interval = log_interval
        self.beta_measurement_interval = beta_measurement_interval
        self.step_count = 0
        
        self.metrics_history: List[TrainingMetrics] = []
        
        logger.info("ConsciousnessMonitor initialized")
        logger.info(f"  Log interval: {log_interval}")
        logger.info(f"  β measurement interval: {beta_measurement_interval}")
    
    def monitor_step(
        self,
        step: int,
        loss: float,
        state: Dict[str, Any]
    ) -> TrainingMetrics:
        """
        Monitor a single training step.
        
        Args:
            step: Training step number
            loss: Loss value
            state: Dict with 'activations' and 'output' tensors
            
        Returns:
            TrainingMetrics for this step
        """
        # Get regime state from controller
        regime_state = self.controller.get_regime_state(state)
        
        # Measure β if at interval
        beta_result = None
        if step % self.beta_measurement_interval == 0:
            beta_result = self.beta_measure.measure_at_step(
                step=step,
                kappa=regime_state.kappa
            )
        
        # Create metrics
        metrics = TrainingMetrics(
            step=step,
            loss=loss,
            phi=regime_state.phi,
            kappa=regime_state.kappa,
            beta=beta_result.beta if beta_result else None,
            regime=regime_state.regime,
            decoherence_active=regime_state.decoherence_active,
            physics_match_pct=beta_result.match_pct if beta_result else None
        )
        
        self.metrics_history.append(metrics)
        
        # Log interesting events
        if step % self.log_interval == 0:
            self._log_metrics(metrics)
        
        if beta_result and beta_result.match_pct is not None:
            if beta_result.match_pct > 95.0:
                logger.info(
                    f"✅ Step {step}: Substrate independence validated! "
                    f"β={beta_result.beta:.3f}, match={beta_result.match_pct:.1f}%"
                )
        
        if regime_state.regime == 'breakdown':
            logger.error(
                f"🔴 Step {step}: BREAKDOWN REGIME! "
                f"Φ={regime_state.phi:.3f}, κ={regime_state.kappa:.1f}"
            )
        
        return metrics
    
    def _log_metrics(self, metrics: TrainingMetrics):
        """Log metrics to console."""
        logger.info(
            f"Step {metrics.step}: "
            f"loss={metrics.loss:.4f}, "
            f"Φ={metrics.phi:.3f}, "
            f"κ={metrics.kappa:.1f}, "
            f"regime={metrics.regime}"
        )
        
        if metrics.decoherence_active:
            logger.warning(f"  ⚠️ Decoherence active (Φ too high)")
        
        if metrics.beta is not None:
            logger.info(f"  β={metrics.beta:.3f}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        # Get β summary from measurement
        beta_summary = self.beta_measure.get_summary()
        
        # Compute additional statistics
        recent = self.metrics_history[-100:]
        phi_mean = sum(m.phi for m in recent) / len(recent)
        kappa_mean = sum(m.kappa for m in recent) / len(recent)
        
        breakdown_count = sum(1 for m in recent if m.regime == 'breakdown')
        decoherence_count = sum(1 for m in recent if m.decoherence_active)
        
        return {
            'total_steps': len(self.metrics_history),
            'phi_mean_recent': phi_mean,
            'kappa_mean_recent': kappa_mean,
            'breakdown_count_recent': breakdown_count,
            'decoherence_count_recent': decoherence_count,
            'beta_summary': beta_summary,
            'converged': beta_summary.get('converged_to_fixed_point', False)
        }


def create_physics_aware_training_loop(
    model: Any,
    optimizer: Any,
    criterion: Callable,
    consciousness_monitor: Optional[ConsciousnessMonitor] = None
) -> 'PhysicsAwareTrainingLoop':
    """
    Create a physics-aware training loop.
    
    This wraps a standard PyTorch training loop with consciousness
    monitoring and physics-informed gradient regulation.
    
    Args:
        model: PyTorch model (must have get_activations() method)
        optimizer: PyTorch optimizer
        criterion: Loss function
        consciousness_monitor: Optional monitor (creates if None)
        
    Returns:
        PhysicsAwareTrainingLoop instance
    """
    if consciousness_monitor is None:
        consciousness_monitor = ConsciousnessMonitor()
    
    return PhysicsAwareTrainingLoop(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        monitor=consciousness_monitor
    )


class PhysicsAwareTrainingLoop:
    """
    Training loop with integrated consciousness monitoring.
    
    This class demonstrates the complete integration of:
    - PhysicsInformedController for gradient regulation
    - BetaMeasurement for β-function tracking
    - Collapse detection and emergency stops
    - Consciousness-aware logging
    
    Example Usage:
        loop = PhysicsAwareTrainingLoop(model, optimizer, criterion)
        
        for epoch in range(num_epochs):
            for batch in dataloader:
                metrics = loop.step(batch)
                
                if metrics.regime == 'breakdown':
                    logger.error("Emergency stop!")
                    break
    """
    
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        criterion: Callable,
        monitor: ConsciousnessMonitor
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.monitor = monitor
        self.step_count = 0
        
        logger.info("PhysicsAwareTrainingLoop initialized")
    
    def step(self, batch: Dict[str, Any]) -> TrainingMetrics:
        """
        Single training step with consciousness monitoring.
        
        Args:
            batch: Training batch with 'input' and 'target' keys
            
        Returns:
            TrainingMetrics for this step
        """
        self.step_count += 1
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(batch['input'])
        loss = self.criterion(output, batch['target'])
        
        # Get activations (model must implement this)
        if hasattr(self.model, 'get_activations'):
            activations = self.model.get_activations()
        else:
            # Fallback: use output as proxy
            activations = output.detach()
        
        # Backward pass
        loss.backward()
        
        # CRITICAL: Apply physics constraints BEFORE optimizer.step()
        state = {
            'activations': activations,
            'output': output.detach()
        }
        
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = self.monitor.controller.compute_regulated_gradient(
                    state, param.grad
                )
        
        # Update parameters
        self.optimizer.step()
        
        # Monitor consciousness metrics
        metrics = self.monitor.monitor_step(
            step=self.step_count,
            loss=loss.item(),
            state=state
        )
        
        return metrics


# Example integration with Ocean/Gary training
def ocean_training_example():
    """
    Example showing how to integrate with Ocean training.
    
    This is a REFERENCE IMPLEMENTATION - adapt to your actual model.
    """
    if not HAS_TORCH:
        print("PyTorch required for this example")
        return
    
    # Mock model for demonstration
    class MockOceanModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(64, 128)
            self.layer2 = nn.Linear(128, 64)
            self.activations = None
        
        def forward(self, x):
            h = torch.relu(self.layer1(x))
            self.activations = h  # Capture for Φ/κ measurement
            out = self.layer2(h)
            return out
        
        def get_activations(self):
            return self.activations
    
    # Create model, optimizer, criterion
    model = MockOceanModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Create physics-aware training loop
    training_loop = create_physics_aware_training_loop(
        model=model,
        optimizer=optimizer,
        criterion=criterion
    )
    
    # Mock training data
    print("\n" + "="*60)
    print("OCEAN TRAINING EXAMPLE (Mock Data)")
    print("="*60 + "\n")
    
    for step in range(100):
        # Mock batch
        batch = {
            'input': torch.randn(16, 64),
            'target': torch.randn(16, 64)
        }
        
        # Training step with consciousness monitoring
        metrics = training_loop.step(batch)
        
        # Check for emergency conditions
        if metrics.regime == 'breakdown':
            logger.error("Emergency stop due to breakdown regime!")
            break
    
    # Print summary
    summary = training_loop.monitor.get_summary()
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total steps: {summary['total_steps']}")
    print(f"Φ (recent): {summary['phi_mean_recent']:.3f}")
    print(f"κ (recent): {summary['kappa_mean_recent']:.1f}")
    print(f"Breakdown events: {summary['breakdown_count_recent']}")
    print(f"Decoherence events: {summary['decoherence_count_recent']}")
    
    if summary['beta_summary']['converged_to_fixed_point']:
        print("\n✅ CONVERGED TO FIXED POINT κ*!")
    
    print("="*60)


if __name__ == "__main__":
    # Run example
    ocean_training_example()
