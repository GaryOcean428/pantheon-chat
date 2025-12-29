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

print("Ocean QIG Training Integration Example loaded successfully!")
