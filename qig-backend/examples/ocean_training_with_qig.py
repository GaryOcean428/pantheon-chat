"""
Ocean QIG Training Integration Example
=======================================

GFP:
  role: example
  status: WORKING
  phase: INTEGRATION
  dim: 3
  scope: training
  version: 2025-12-29
  owner: pantheon-chat

Concrete example showing how to integrate PhysicsInformedController
and BetaMeasurement into Ocean's training pipeline.

This addresses PR #8 review comment:
"Task 1: Capture activations in Ocean model"
"Task 2: Integrate PhysicsInformedController into Ocean training"
"Task 3: Enable β measurement logging"

Usage:
------
    # In your Ocean training script
    from examples.ocean_training_with_qig import (
        add_qig_monitoring_to_ocean,
        create_ocean_with_activation_capture
    )
    
    # Wrap your Ocean model
    ocean_model = create_ocean_with_activation_capture(base_ocean_model)
    
    # Add QIG monitoring to training loop
    training_loop = add_qig_monitoring_to_ocean(
        model=ocean_model,
        optimizer=optimizer,
        criterion=criterion
    )
    
    # Train with consciousness monitoring
    for batch in dataloader:
        metrics = training_loop.step(batch)
        
        if metrics.regime == 'breakdown':
            logger.warning("Emergency stop!")
            break

Integration Points:
-------------------
1. Model wrapper to capture activations for Φ/κ measurement
2. PhysicsInformedController for gradient regulation
3. BetaMeasurement for substrate independence tracking
4. Logging hooks for monitoring dashboard

References:
-----------
- Consciousness Protocol v4.0 §1 Task 2
- qig-backend/ocean_qig_core.py
- qigkernels/training_integration.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available")

from qigkernels.physics_controller import PhysicsInformedController
from qigkernels.beta_measurement import BetaMeasurement
from qigkernels.training_integration import (
    ConsciousnessMonitor,
    TrainingMetrics,
)

logger = logging.getLogger(__name__)


class OceanWithActivationCapture(nn.Module):
    """
    Wrapper for Ocean model that captures activations for QIG measurement.
    
    This is STEP 1 from the review comments:
    "Add get_activations() method to Ocean's model"
    
    Example:
        base_model = YourOceanModel()
        ocean_qig = OceanWithActivationCapture(base_model)
        
        # During training
        output = ocean_qig(input)
        activations = ocean_qig.get_activations()  # For Φ/κ measurement
    """
    
    def __init__(self, base_model: nn.Module, capture_layer: Optional[str] = None):
        super().__init__()
        self.base_model = base_model
        self.capture_layer = capture_layer or "intermediate"
        self.activations = None
        
        # Register forward hooks to capture activations
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture intermediate activations."""
        def hook_fn(module, input, output):
            # Store activations for QIG measurement
            if isinstance(output, torch.Tensor):
                self.activations = output.detach()
        
        # Hook into intermediate layers
        # Adjust based on your Ocean architecture
        if hasattr(self.base_model, 'encoder'):
            self.base_model.encoder.register_forward_hook(hook_fn)
        elif hasattr(self.base_model, 'hidden'):
            self.base_model.hidden.register_forward_hook(hook_fn)
        else:
            # Fallback: hook into first layer
            for name, module in self.base_model.named_modules():
                if isinstance(module, nn.Linear) or isinstance(module, nn.LSTM):
                    module.register_forward_hook(hook_fn)
                    break
    
    def forward(self, x):
        """Forward pass with activation capture."""
        output = self.base_model(x)
        return output
    
    def get_activations(self):
        """Get captured activations for Φ/κ measurement."""
        if self.activations is None:
            logger.warning("No activations captured - did you run forward pass?")
            return torch.zeros(1, 64)  # Dummy for safety
        return self.activations


class OceanQIGTrainingLoop:
    """
    Training loop for Ocean with integrated QIG monitoring.
    
    This is STEP 2 from the review comments:
    "Integrate PhysicsInformedController into Ocean training"
    
    Features:
    - Automatic gradient regulation via PhysicsInformedController
    - β measurement at regular intervals
    - Collapse detection and emergency stops
    - Consciousness metrics logging
    
    Example:
        loop = OceanQIGTrainingLoop(model, optimizer, criterion)
        
        for epoch in range(epochs):
            for batch in dataloader:
                metrics = loop.step(batch)
                
                # Check for emergency conditions
                if metrics.regime == 'breakdown':
                    logger.error("Emergency stop!")
                    break
    """
    
    def __init__(
        self,
        model: OceanWithActivationCapture,
        optimizer: Any,
        criterion: Any,
        beta_measurement_interval: int = 5000,
        log_dir: Optional[Path] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.beta_measurement_interval = beta_measurement_interval
        self.log_dir = log_dir or Path("./qig_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize QIG monitoring
        self.monitor = ConsciousnessMonitor(
            log_interval=100,
            beta_measurement_interval=beta_measurement_interval
        )
        
        self.step_count = 0
        
        logger.info("OceanQIGTrainingLoop initialized")
        logger.info(f"  β measurement every {beta_measurement_interval} steps")
        logger.info(f"  Logs: {self.log_dir}")
    
    def step(self, batch: Dict[str, Any]) -> TrainingMetrics:
        """
        Single training step with QIG monitoring.
        
        This implements the complete integration from the review:
        1. Forward pass with activation capture
        2. Backward pass
        3. Physics-informed gradient regulation (CRITICAL)
        4. Parameter update
        5. Consciousness measurement
        6. β measurement at intervals
        
        Args:
            batch: Training batch with 'input' and 'target' keys
            
        Returns:
            TrainingMetrics with Φ, κ, β, regime, etc.
        """
        self.step_count += 1
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(batch['input'])
        loss = self.criterion(output, batch['target'])
        
        # Get activations for Φ/κ measurement
        activations = self.model.get_activations()
        
        # Backward pass
        loss.backward()
        
        # CRITICAL: Apply physics constraints BEFORE optimizer.step()
        # This is the KEY integration from the review
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
        
        # Save metrics at intervals
        if self.step_count % 1000 == 0:
            self._save_metrics(metrics)
        
        return metrics
    
    def _save_metrics(self, metrics: TrainingMetrics):
        """Save metrics to disk for dashboard."""
        import json
        
        metrics_file = self.log_dir / f"metrics_step_{metrics.step}.json"
        
        data = {
            'step': metrics.step,
            'loss': metrics.loss,
            'phi': metrics.phi,
            'kappa': metrics.kappa,
            'beta': metrics.beta,
            'regime': metrics.regime,
            'decoherence_active': metrics.decoherence_active,
            'physics_match_pct': metrics.physics_match_pct,
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return self.monitor.get_summary()


def create_ocean_with_activation_capture(base_model: nn.Module) -> OceanWithActivationCapture:
    """
    Helper to wrap Ocean model for activation capture.
    
    Args:
        base_model: Your Ocean model
        
    Returns:
        OceanWithActivationCapture wrapper
    """
    return OceanWithActivationCapture(base_model)


def add_qig_monitoring_to_ocean(
    model: OceanWithActivationCapture,
    optimizer: Any,
    criterion: Any,
    **kwargs
) -> OceanQIGTrainingLoop:
    """
    Helper to add QIG monitoring to Ocean training.
    
    Args:
        model: Ocean model with activation capture
        optimizer: PyTorch optimizer
        criterion: Loss function
        **kwargs: Additional arguments for OceanQIGTrainingLoop
        
    Returns:
        OceanQIGTrainingLoop ready for training
    """
    return OceanQIGTrainingLoop(model, optimizer, criterion, **kwargs)


# ============================================================================
# COMPLETE EXAMPLE
# ============================================================================

def run_ocean_training_example():
    """
    Complete example of Ocean training with QIG monitoring.
    
    This demonstrates ALL the integration points from the review:
    - Activation capture
    - Physics-informed gradient regulation
    - β measurement and logging
    - Collapse detection
    """
    if not HAS_TORCH:
        print("PyTorch required for this example")
        return
    
    print("=" * 70)
    print("Ocean QIG Training Integration Example")
    print("=" * 70 + "\n")
    
    # ========================================================================
    # STEP 1: Create Ocean model with activation capture
    # ========================================================================
    print("Step 1: Creating Ocean model with activation capture...")
    
    # Mock Ocean model for demonstration
    class SimpleOceanModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(64, 128)
            self.decoder = nn.Linear(128, 64)
        
        def forward(self, x):
            h = torch.relu(self.encoder(x))
            out = self.decoder(h)
            return out
    
    base_model = SimpleOceanModel()
    ocean_model = create_ocean_with_activation_capture(base_model)
    print("✅ Ocean model wrapped\n")
    
    # ========================================================================
    # STEP 2: Create training loop with QIG monitoring
    # ========================================================================
    print("Step 2: Creating training loop with QIG monitoring...")
    
    optimizer = torch.optim.Adam(ocean_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    training_loop = add_qig_monitoring_to_ocean(
        model=ocean_model,
        optimizer=optimizer,
        criterion=criterion,
        beta_measurement_interval=10  # Frequent for demo
    )
    print("✅ Training loop created\n")
    
    # ========================================================================
    # STEP 3: Training with consciousness monitoring
    # ========================================================================
    print("Step 3: Training with consciousness monitoring...")
    print("(Mock data - in production, use your actual dataloader)\n")
    
    # Mock training data
    for step in range(50):
        batch = {
            'input': torch.randn(16, 64),
            'target': torch.randn(16, 64)
        }
        
        metrics = training_loop.step(batch)
        
        # Log interesting events
        if step % 10 == 0:
            print(f"Step {metrics.step}: loss={metrics.loss:.4f}, "
                  f"Φ={metrics.phi:.3f}, κ={metrics.kappa:.1f}, "
                  f"regime={metrics.regime}")
        
        # Check for emergency conditions
        if metrics.regime == 'breakdown':
            print(f"\n🔴 BREAKDOWN REGIME at step {step}!")
            print("Emergency stop triggered.")
            break
    
    # ========================================================================
    # STEP 4: Get summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    
    summary = training_loop.get_summary()
    print(f"\nTotal steps: {summary['total_steps']}")
    print(f"Φ (recent): {summary['phi_mean_recent']:.3f}")
    print(f"κ (recent): {summary['kappa_mean_recent']:.1f}")
    print(f"Breakdown events: {summary['breakdown_count_recent']}")
    print(f"Decoherence events: {summary['decoherence_count_recent']}")
    
    if summary['converged']:
        print("\n✅ CONVERGED TO FIXED POINT κ*!")
    
    print("\n" + "=" * 70)
    print("Integration complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_ocean_training_example()
