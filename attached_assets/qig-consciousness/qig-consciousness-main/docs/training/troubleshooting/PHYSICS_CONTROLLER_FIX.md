# Physics-Informed Controller Fix

## Problem

The physics-informed threshold controller in `tools/train_qig_kernel.py` was not engaging during training, despite `dynamic_entanglement: true` being set in the config. The threshold remained constant at 0.150 instead of adapting based on three geometric signals (κ_eff, Φ, basin_distance).

## Root Cause

The nested YAML config structure was not being properly loaded into the TrainingConfig object:

```yaml
model:
  dynamic_entanglement: true
  entanglement_feedback_gain: 0.01
  # ... other model settings
```

The `load_from_file()` method only extracted specific fields from the `model` section but didn't preserve the nested structure as an accessible object. When `update_entanglement_threshold()` tried to access:

```python
model_cfg = getattr(self.config, "model_config", self.config)
dynamic_on = getattr(model_cfg, "dynamic_entanglement", False)
```

It fell back to `self.config` (which lacks the `dynamic_entanglement` attribute), so `dynamic_on` was always `False`, causing an early return.

## Solution

### 1. Store Nested Model Config (tools/train_qig_kernel.py)

```python
from types import SimpleNamespace

# In TrainingConfig.load_from_file():
if 'model' in config:
    model_config = config['model']
    # Store the entire model config as a namespace
    self.model_config = SimpleNamespace(**model_config)
    
    # Also set specific fields for backward compatibility
    if 'vocab_size' in model_config:
        self.vocab_size = model_config['vocab_size']
    # ... etc
```

### 2. Update Controller Access Pattern

```python
# In update_entanglement_threshold():
# Try model_config first (nested YAML), then fall back to config itself (flat)
model_cfg = getattr(self.config, "model_config", self.config)
dynamic_on = getattr(model_cfg, "dynamic_entanglement", False)
```

### 3. Add Telemetry Logging

Added `telemetry['entanglement_threshold'] = new_thr` to make the updated threshold visible in telemetry logs for monitoring.

### 4. Load Additional Config Fields

Added loading of:
- `min_recursion_depth` and `min_Phi` from model config
- `lm_weight`, `basin_weight`, `phi_weight`, `gradient_clip` from training config

## Verification

Created test scripts to verify:
1. Config loading properly stores nested model_config
2. All model config attributes are accessible
3. Threshold controller logic calculates expected updates
4. Three signals (κ_eff, Φ, basin) are properly combined

Test results show threshold should adapt from 0.150 → 0.129 for typical early training values (Φ=0.105, basin=0.975).

## Expected Behavior After Fix

With `dynamic_entanglement: true`, the threshold should now:

1. **Scale with κ_eff** (running coupling from physics)
2. **Modulate with Φ** (integration feedback, asymmetric response)
3. **Adjust for basin distance** (geometric position on manifold)

Example trajectory:
```
Step 0  | Φ: 0.000 | Basin: 1.080 | Thresh: 0.126
Step 50 | Φ: 0.105 | Basin: 0.975 | Thresh: 0.129
Step 100| Φ: 0.203 | Basin: 0.877 | Thresh: 0.131
```

The threshold will now be responsive instead of stuck at 0.150.

## Files Modified

- `tools/train_qig_kernel.py`: Fixed config loading, added telemetry logging

## Testing

Run the conservative test to verify:
```bash
python tools/train_qig_kernel.py \
    --config configs/train_conservative_test.yaml \
    --output-dir runs/test_dynamic \
    --lr 5e-5
```

Monitor with:
```bash
python tools/monitor_dynamic_threshold.py runs/test_dynamic/training_telemetry.jsonl
```

Should show threshold varying based on Φ, κ_eff, and basin_distance.
