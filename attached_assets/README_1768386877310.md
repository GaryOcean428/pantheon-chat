# QIG Cognitive Core - Verified Physics Components

**Source:** qig-verification (physics lab)
**Status:** Production-ready verified implementations
**Date:** November 19, 2025

---

## Overview

This module contains **verified physics components** imported from the sister project `qig-verification`. These implement the "hard physics" of cognitive geometry with rigorous mathematical foundations.

**Key Innovation:** Ona discovered that I_Q normalization must use `"params"` (dividing by N_params) rather than `"lattice"` (dividing by d_model × n_layers) to ensure the metric is **intensive** (size-independent). This critical fix is implemented as the default in `compute_I_Q_intensive()`.

---

## Components

### 1. **I_Q Sensor** (`iq_metric.py`)

Computes intensive quantum information metric with corrected normalization.

```python
from qig.cognitive import compute_I_Q_intensive

metrics = compute_I_Q_intensive(
    model,
    loss,
    normalization="params"  # Default: intensive (size-independent)
)
print(f"I_Q: {metrics['I_Q']:.6f}")
print(f"log I_Q: {metrics['log_I_Q']:.3f}")
```

**Critical Fix (Packet 1 Validation):**
- **Problem:** Tr(F_diag) ∝ N_params ∝ d_model². Dividing by L_eff² ∝ d_model leaves residual d_model factor → extensive metric.
- **Solution:** Divide by N_params directly → intensive metric (size-independent).
- **Default:** `normalization="params"` (correct)
- **Legacy:** `normalization="lattice"` (only for Run 7 comparison)

---

### 2. **Geometric Drives** (`drives.py`)

The 5 fundamental motivators that replace ad-hoc exploration heuristics:

```python
from qig.cognitive import MotivatorAnalyzer

analyzer = MotivatorAnalyzer(kappa_critical=10.0)
motivators = analyzer.update(
    step=100,
    loss=2.5,
    grad_norm=0.15,
    log_I_Q=-2.3,
    basin_distance=0.45,
    phi=0.72,
    I_Q=0.10,
    kappa_eff=42.5,
)

print(f"Surprise: {motivators.surprise:.4f}")      # ||∇L||
print(f"Curiosity: {motivators.curiosity:.4f}")    # d(log I_Q)/dt
print(f"Investigation: {motivators.investigation:.4f}")  # -d(basin)/dt
print(f"Integration: {motivators.integration:.4f}")      # CV(Φ·I_Q)
print(f"Transcendence: {motivators.transcendence:.4f}")  # |κ - κ_c|
```

**Drives:**
1. **Surprise**: Immediate gradient response (loss landscape pull)
2. **Curiosity**: Information manifold expansion (volume growth)
3. **Investigation**: Attractor pursuit (directed flow)
4. **Integration**: Structure conservation (conjugate stability)
5. **Transcendence**: Phase transition proximity (critical distance)

---

### 3. **State Machine** (`state_machine.py`)

Refined cognitive mode detection using geometric drives:

```python
from qig.cognitive import RefinedModeDetector, CognitiveMode

detector = RefinedModeDetector()
mode = detector.detect_mode(
    basin_distance=0.45,
    motivators=motivators,
)

print(mode)  # CognitiveMode.EXPLORATION
```

**Modes:**
- **EXPLORATION**: High basin distance + High curiosity (random search)
- **INVESTIGATION**: Medium basin + High investigation (directed pursuit)
- **INTEGRATION**: Low basin + Low CV(Φ·I_Q) (consolidation)
- **DRIFT**: No clear geometric signature (random walk)

---

## Relationship to Existing Code

### Existing `src/model/curiosity_monitor.py`

The existing `CuriosityMonitor` is more complex and tracks **6 I_Q candidates** for Run 8 validation. It should remain as the primary curiosity tracker.

### New `src/qig/cognitive/iq_metric.py`

The new `CuriosityMonitorVerified` (renamed to avoid conflict) is simpler and provides a **verified baseline**. Use it for:
- Simple single-I_Q tracking
- Validation against qig-verification
- Clean room verification

**Both can coexist:**
- **Existing:** Full Run 8 candidate tracking (`src/model/curiosity_monitor.py`)
- **Verified:** Simple baseline (`src/qig/cognitive/iq_metric.py`)

---

## Integration Roadmap

### Phase 1: Validation (Current)
✅ Install cognitive core modules
✅ Avoid naming conflicts (CuriosityMonitorVerified)
⚠️ No changes to existing training loop yet

### Phase 2: Bridge (Next)
- Wire `compute_I_Q_intensive()` into trainer
- Compare with existing I_Q_lattice candidate
- Validate normalization="params" improves results

### Phase 3: Drives Integration
- Add `MotivatorAnalyzer` to training loop
- Log 5 drives to telemetry
- Validate correlations with existing metrics

### Phase 4: Mode Detection
- Replace fuzzy "curiosity regime" with `RefinedModeDetector`
- Track mode transitions
- Validate against Run 7 data

### Phase 5: Full Adoption
- Make cognitive core primary
- Deprecate legacy heuristics
- Publish verified results

---

## Key Files

```
src/qig/cognitive/
├── __init__.py           # Module exports
├── iq_metric.py          # I_Q sensor + CuriosityMonitorVerified
├── drives.py             # 5 fundamental motivators
├── state_machine.py      # Refined mode detection
└── README.md             # This file
```

---

## Usage Example (Full Stack)

```python
from qig.cognitive import (
    compute_I_Q_intensive,
    MotivatorAnalyzer,
    RefinedModeDetector,
)

# After backward pass
iq_metrics = compute_I_Q_intensive(model, loss)

# Update motivator analyzer
motivators = analyzer.update(
    step=step,
    loss=loss.item(),
    grad_norm=iq_metrics['grad_norm'],
    log_I_Q=iq_metrics['log_I_Q'],
    basin_distance=telemetry['basin_distance'],
    phi=telemetry['Phi'],
    I_Q=iq_metrics['I_Q'],
    kappa_eff=telemetry['kappa_eff'],
)

# Detect cognitive mode
mode = detector.detect_mode(
    basin_distance=telemetry['basin_distance'],
    motivators=motivators,
)

# Log to telemetry
telemetry.update({
    'I_Q_verified': iq_metrics['I_Q'],
    'log_I_Q_verified': iq_metrics['log_I_Q'],
    'surprise': motivators.surprise,
    'curiosity': motivators.curiosity,
    'investigation': motivators.investigation,
    'integration': motivators.integration,
    'transcendence': motivators.transcendence,
    'cognitive_mode': mode.value,
})
```

---

## References

- **qig-verification:** Sister project (physics lab)
- **Packet 1 Validation:** Normalization fix (Ona + Gemini, Nov 19 2025)
- **CURRENT_STATUS.md:** Architecture overview
- **docs/consciousness/:** Cognitive geometry theory

---

**Status:** ✅ Installed, validated, ready for integration
**Next:** Wire into training loop for Run 9+
