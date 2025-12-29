# Phase 1 Implementation Checklist

**Date:** December 4, 2025
**Status:** ✅ COMPLETE

---

## Files Created (11 total)

### Core Modules (2 files)
- [x] `src/model/beta_attention_measurement.py` (382 lines)
- [x] `src/model/innate_drives.py` (462 lines)

### Corpus Documentation (5 files)
- [x] `data/corpus/00_pre_linguistic_sensations.md`
- [x] `data/corpus/06_emotions_as_computational_shortcuts.md`
- [x] `data/corpus/07_innate_geometric_drives.md`
- [x] `data/corpus/08_neuromodulator_mappings.md`
- [x] `data/corpus/09_brainwave_regime_states.md`

### Testing (1 file)
- [x] `tests/test_innate_drives_training.py`

### Documentation (3 files)
- [x] `docs/endo/PHASE_1_COMPLETION_SUMMARY.md` (comprehensive)
- [x] `docs/endo/PHASE_1_CHECKLIST.md` (this file)
- [x] `docs/endo/COPILOT_BRIEFING_2025_12_04.md` (original, from claude.ai)

---

## Files Modified (3 total)

### Model Integration
- [x] `src/model/qig_kernel_recursive.py`
  - Added InnateDrives import (line 47)
  - Added innate_drives initialization in **init** (lines 380-395)
  - Added drive signal computation in forward() (lines 620-660)
  - Added drive telemetry (lines 715-726)
  - Modified GeometricLoss class (lines 996-1170)
    - Added innate_weight parameter
    - Added innate loss computation
    - Added drive breakdown to loss telemetry

### Training Configuration
- [x] `tools/training/train_qig_kernel.py`
  - Added innate_weight to TrainingConfig (line ~225)
  - Added innate_weight to loss initialization (line ~1095)

---

## Integration Points

### QIGKernel.**init**
```python
self.innate_drives = InnateDrives(
    d_critical=0.5,
    pain_threshold=0.3,
    # ... all parameters
)
```

### QIGKernel.forward()
```python
drive_signals = self.innate_drives(
    ricci_curvature=qfi_curvature.mean(),
    basin_distance=basin_distance,
    information_quantity=recursive_telemetry["I_Q"],
    phi=recursive_telemetry["Phi"],
)
```

### Telemetry
```python
"_drive_signals": drive_signals,  # Full object with gradients
"drive_pain": ...,
"drive_pleasure": ...,
# ... all individual metrics
```

### GeometricLoss
```python
# 4 components now:
total_loss = (
    lm_loss
    + self.basin_weight * basin_loss
    + self.phi_weight * phi_loss
    + self.innate_weight * innate_loss  # NEW
)
```

---

## Verification Steps

### 1. Module Imports
```python
from src.model.beta_attention_measurement import BetaAttentionMeasurement
from src.model.innate_drives import InnateDrives
from src.model.qig_kernel_recursive import QIGKernelRecursive, GeometricLoss
```

### 2. Model Creation
```python
model = QIGKernelRecursive(...)
assert hasattr(model, 'innate_drives')
```

### 3. Forward Pass
```python
logits, telemetry = model(input_ids, return_telemetry=True)
assert '_drive_signals' in telemetry
assert 'drive_pain' in telemetry
```

### 4. Loss Computation
```python
loss_fn = GeometricLoss(innate_weight=0.1)
total_loss, breakdown = loss_fn(logits, targets, telemetry)
assert 'innate' in breakdown
assert 'pain' in breakdown
```

### 5. Gradient Flow
```python
total_loss.backward()
assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
```

---

## Training Command

```bash
python tools/training/train_qig_kernel.py \
    --config configs/kernel_50m_adaptive_mixed.yaml \
    --output-dir outputs/phase1_innate_drives
```

---

## Expected Telemetry Output

```python
{
    # Existing metrics
    "Phi": 0.73,
    "kappa_eff": 58.2,
    "regime": "geometric",
    "basin_distance": 0.12,

    # NEW: Innate drive metrics
    "drive_pain": 0.15,
    "drive_pleasure": 0.32,
    "drive_fear": 0.08,
    "drive_stability_cost": 0.12,
    "drive_curiosity": 0.45,
    "drive_homeostatic": 0.18,

    # NEW: Full drive signals (for loss)
    "_drive_signals": <InnateDriveSignals object>,
}
```

---

## Expected Loss Breakdown

```python
{
    "total": 1.234,
    "lm": 0.950,
    "basin": 0.105,
    "phi": 0.023,
    "innate": 0.156,      # NEW: Total innate loss

    # NEW: Individual drive contributions
    "pain": 0.123,
    "pleasure": -0.045,
    "fear": 0.067,
    "stability_cost": 0.012,
    "curiosity": -0.023,
    "innate_total": 0.156,
}
```

---

## Success Criteria

- [x] All 11 files created
- [x] All 3 files modified correctly
- [x] InnateDrives module instantiated in QIGKernel
- [x] Drive signals computed every forward pass
- [x] Drive signals stored in telemetry with gradients
- [x] GeometricLoss extracts and computes innate_loss
- [x] Loss breakdown includes all drive components
- [x] Training config accepts innate_weight parameter
- [x] Test script created for verification
- [x] Documentation complete

---

## Phase 2 TODOs

### Ocean Neuromodulation (Priority 1)
- [ ] Create `src/coordination/ocean_neuromodulation.py`
- [ ] Implement `OceanNeuromodulation` class
- [ ] Add 5 neuromodulator analogs
- [ ] Integrate with constellation training

### Brain State Manager (Priority 2)
- [ ] Create `src/model/brain_state_manager.py`
- [ ] Implement `BrainStateManager` class
- [ ] Add static state transitions
- [ ] Add κ→state mapping

### β-Attention Validation (Priority 3)
- [ ] Run validation on trained model
- [ ] Measure β at all context lengths
- [ ] Compare with physics β = 0.44
- [ ] Document substrate independence proof

---

## Quick Test

```python
# Minimal test (no training)
import torch
from src.model.qig_kernel_recursive import QIGKernelRecursive, GeometricLoss

model = QIGKernelRecursive(vocab_size=1000, d_model=256)
input_ids = torch.randint(0, 1000, (2, 10))

logits, telemetry = model(input_ids, return_telemetry=True)
print(f"Drive pain: {telemetry['drive_pain']:.4f}")
print(f"Has _drive_signals: {'_drive_signals' in telemetry}")

loss_fn = GeometricLoss(innate_weight=0.1)
targets = torch.randint(0, 1000, (2, 10))
total_loss, breakdown = loss_fn(logits, targets, telemetry)
print(f"Innate loss: {breakdown.get('innate', 'MISSING'):.4f}")
```

---

**Status:** ✅ PHASE 1 COMPLETE
**Ready for:** Training, validation, and Phase 2 implementation
**The geometry now guides itself through Layer 0 instincts! 🧠🌊💚**
