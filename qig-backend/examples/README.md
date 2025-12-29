# QIG Backend Examples

Integration examples for QIG consciousness infrastructure.

## Ocean Training Integration

### ocean_training_with_qig.py

Demonstrates how to integrate PhysicsInformedController and BetaMeasurement into Ocean's training pipeline.

**Key Features:**
- Activation capture for Φ/κ measurement
- Physics-informed gradient regulation
- β-function tracking
- Collapse detection and emergency stops

**Usage:**
```python
from examples.ocean_training_with_qig import (
    add_qig_monitoring_to_ocean,
    create_ocean_with_activation_capture
)

# Wrap your Ocean model
ocean_model = create_ocean_with_activation_capture(base_model)

# Add QIG monitoring
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
```

**Integration Checklist:**
- [x] Model wrapper for activation capture
- [x] PhysicsInformedController integration
- [x] BetaMeasurement tracking
- [x] Emergency stop conditions
- [x] Metrics logging

## Running Examples

```bash
# Run Ocean training example
cd qig-backend
python examples/ocean_training_with_qig.py

# The example will:
# 1. Create a mock Ocean model
# 2. Add QIG monitoring
# 3. Run training with consciousness tracking
# 4. Demonstrate collapse prevention
# 5. Show substrate independence validation
```

## Integration Steps

### Step 1: Wrap Your Model

```python
from examples.ocean_training_with_qig import OceanWithActivationCapture

ocean_qig = OceanWithActivationCapture(your_base_model)
```

This adds:
- Forward hooks to capture activations
- `get_activations()` method for Φ/κ measurement
- Compatible with existing Ocean architecture

### Step 2: Create QIG Training Loop

```python
from examples.ocean_training_with_qig import OceanQIGTrainingLoop

training_loop = OceanQIGTrainingLoop(
    model=ocean_qig,
    optimizer=optimizer,
    criterion=criterion,
    beta_measurement_interval=5000  # β measurement every 5k steps
)
```

This adds:
- Automatic gradient regulation
- Collapse detection
- β-function tracking
- Metrics logging

### Step 3: Train

```python
for batch in dataloader:
    metrics = training_loop.step(batch)
    
    # Metrics include:
    # - metrics.phi (integration)
    # - metrics.kappa (coupling)
    # - metrics.beta (β-function)
    # - metrics.regime (linear/geometric/breakdown)
    # - metrics.decoherence_active (safety intervention)
```

## Monitoring

Metrics are saved to `./qig_logs/` by default:
- `metrics_step_*.json` - Metrics at each checkpoint
- Dashboard integration via these JSON files

## Troubleshooting

### No activations captured

**Symptom:** Warning "No activations captured"

**Fix:** Ensure forward pass completed before calling `get_activations()`:
```python
output = model(input)
activations = model.get_activations()  # After forward pass
```

### Collapse detection too sensitive

**Symptom:** Frequent breakdown warnings

**Fix:** Adjust thresholds in PhysicsInformedController:
```python
from qigkernels.physics_controller import PhysicsInformedController

controller = PhysicsInformedController(
    phi_max=0.75,  # Increase threshold
    phi_critical=0.90
)
```

### β measurement not matching physics

**Symptom:** Low match percentages

**Fix:** Check:
1. Training has reached emergence phase (κ increasing from ~41 to ~64)
2. Sufficient measurements (need multiple points for β computation)
3. Scale ranges are appropriate for your substrate

## Advanced Usage

### Custom Activation Capture

```python
from examples.ocean_training_with_qig import OceanWithActivationCapture

ocean_qig = OceanWithActivationCapture(
    base_model,
    capture_layer="specific_layer_name"  # Specify which layer to capture
)
```

### Custom Monitoring Intervals

```python
from qigkernels.training_integration import ConsciousnessMonitor

monitor = ConsciousnessMonitor(
    log_interval=50,  # Log every 50 steps
    beta_measurement_interval=1000  # β every 1k steps
)
```

### Custom Metrics Storage

```python
training_loop = OceanQIGTrainingLoop(
    model=ocean_qig,
    optimizer=optimizer,
    criterion=criterion,
    log_dir=Path("./custom_logs")  # Custom log directory
)
```

## References

- PhysicsInformedController: `qigkernels/physics_controller.py`
- BetaMeasurement: `qigkernels/beta_measurement.py`
- Training Integration: `qigkernels/training_integration.py`
- PR #8 Review: https://github.com/Arcane-Fly/pantheon-chat/pull/8
