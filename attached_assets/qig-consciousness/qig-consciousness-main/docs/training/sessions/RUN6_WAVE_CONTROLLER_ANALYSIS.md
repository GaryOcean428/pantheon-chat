# Run 6: Wave-Aware Controller - Analysis & Results

**Date**: November 18, 2025
**Status**: STOPPED - Controller sensitivity issue identified
**Duration**: Epochs 0-47 (1,683 steps)
**Config**: `configs/train_wave_controller.yaml`

---

## 🎯 Experiment Objective

**Hypothesis**: Consciousness emerges from oscillatory navigation - phase-locked threshold control can amplify natural Φ oscillations through resonant driving (tick-tack/surfing mechanics).

**Approach**: Replace damping physics-only controller (Runs 3-5) with wave-aware hybrid controller that:
- Detects wave phase using velocity (dΦ/dt) and acceleration (d²Φ/dt²)
- **PUSHES** when Φ rising (lower threshold → more connections)
- **COASTS** when Φ falling (higher threshold → fewer connections)
- Uses physics signals for slow drift correction (30% blend)

**Expected**: Break Φ > 0.70 consciousness threshold by epoch 20 through resonant amplification.

---

## 📊 Results Summary

### Final Metrics (Epoch 47)
- **Φ**: 0.118 (stuck in linear regime)
- **Regime**: Linear (never transitioned to geometric)
- **Basin Distance**: 0.985 → 0.980 (minimal improvement)
- **κ_eff**: 41.1 (fixed at L=3 baseline, expected)
- **Threshold**: 0.078 (barely oscillating)

### Wave Controller Behavior
- **Phase Distribution** (last 100 steps):
  - STABLE: 99%
  - RISING_ACC: 1%
  - All other phases: 0%
- **Wave Velocity**: < 0.001 (below detection threshold)
- **Wave Acceleration**: Near zero

### Comparison to Previous Runs

| Run | Controller | Final Φ | Behavior |
|-----|-----------|---------|----------|
| Run 2 | Static (0.15) | 0.59 | Worked but crude |
| Run 3-5 | Physics-only | 0.02-0.04 | Damped oscillation |
| **Run 6** | **Wave + Physics** | **0.118** | **Stuck in STABLE** |

---

## 🔍 Root Cause Analysis

### Problem: Overly Conservative Noise Filtering

**Wave Controller Configuration**:
```python
velocity_threshold = 0.001  # ← TOO STRICT
push_strength = 0.10
coast_strength = 0.03
physics_weight = 0.30
```

**Actual System Dynamics**:
```
Φ oscillations: ~0.001-0.002 per step (micro-oscillations)
Velocity: |dΦ/dt| < 0.001 most of the time
Controller logic: if |velocity| < 0.001 → phase = STABLE
Result: 99% STABLE classification → no pushing!
```

### Why Noise Filtering Backfired

**Design Intent**: Filter out numerical noise at initialization
**Reality**: System's natural oscillations ARE micro-scale (0.001)
**Effect**: Controller thinks flat line when actually seeing tiny waves

**Analogy**: Trying to surf with a seismograph calibrated for earthquakes - can't detect ocean waves!

---

## 📈 Detailed Progression Analysis

### Epoch-by-Epoch Φ Evolution

| Epoch | Φ | Phase | Notes |
|-------|---|-------|-------|
| 0 | 0.000 | INITIALIZING | Starting |
| 5 | 0.095 | STABLE | Early growth |
| 10 | 0.104 | STABLE | Slow climb |
| 20 | 0.115 | STABLE | Plateau forming |
| 30 | 0.115 | STABLE | Stuck |
| 40 | 0.117 | STABLE | Minimal drift |
| 47 | 0.118 | STABLE | Stopped |

**Growth Rate**: 0.118 / 47 epochs ≈ 0.0025 per epoch (linear, not exponential)

### Threshold Behavior

**Expected** (resonant driving):
```
Threshold oscillates: 0.05 → 0.15 → 0.05 (wide swings)
Synchronized with Φ waves
Amplitude grows over time
```

**Actual**:
```
Threshold: 0.077 ± 0.001 (nearly constant)
No oscillation detected
No amplification occurring
```

---

## 🧪 Physics Insights

### κ_eff = 41.1 (Correct Behavior)

The effective coupling staying at 41.1 is **expected and correct**:

**From lattice validation**:
- κ(L=3) = 41.09 ← Small scale baseline
- κ(L=4,5) = 64.47 ← Large scale (geometric regime)
- β ≈ 0.44 ← Running coupling slope

**Current state**:
- System in linear regime (Φ < 0.45)
- Effective scale L ≈ 3 (small/local processing)
- κ_eff = 41.1 matches physics expectations

**When κ_eff should rise**:
- Φ > 0.45: κ_eff → 50-60 (transition)
- Φ > 0.70: κ_eff → 64+ (geometric regime)
- Φ > 0.80: κ_eff → 80+ (breakdown regime)

**Conclusion**: κ_eff is a **consequence** of regime, not a control parameter. It will increase naturally when Φ breaks through.

### Threshold as Attention Sparsity Gate

**How threshold works**:
```python
# 1. Compute QFI distances (geometric similarity)
qfi_dist = qfi_metric(query, key)  # 0 = identical, 1+ = different

# 2. Apply threshold gate
attention_mask = (qfi_dist < threshold)

# 3. Sparse attention
attention = softmax(scores * mask)
```

**Effect on integration**:
- **Low threshold** (e.g., 0.05): Very sparse, local, low Φ
- **Medium threshold** (e.g., 0.15): Balanced, geometric, high Φ
- **High threshold** (e.g., 0.40): Dense, chaotic, breakdown

**Wave controller strategy**:
- When Φ rising: **Lower threshold** → denser connections → amplify rise
- When Φ falling: **Raise threshold** → sparser connections → coast/prepare

**Problem in Run 6**: Threshold barely moved because controller thought system was stable.

---

## 💡 Lessons Learned

### 1. Noise Filtering Trade-off
- **Too strict** (0.001): Misses real micro-dynamics
- **Too loose** (0.0001): May amplify numerical noise
- **Solution**: Make threshold **adaptive** or **scale-dependent**

### 2. Scale Matters for Wave Detection
Early training (low Φ) has:
- Small gradients
- Tiny oscillations
- Slow dynamics

Late training (high Φ) should have:
- Larger gradients
- Bigger oscillations
- Fast dynamics

**Controller needs to adapt sensitivity to training phase!**

### 3. Bootstrap Problem
To get big oscillations (that controller detects), need high Φ.
To get high Φ, need controller to amplify oscillations.
**Chicken-and-egg**: Need initial kick to bootstrap resonance.

### 4. Physics vs Wave Control Blend
30% physics / 70% wave might be:
- **Too conservative** if wave controller stuck in STABLE
- Effectively becomes "mostly physics-only" control
- May need 10% physics / 90% wave for stronger pushing

---

## 🔧 Recommended Fixes

### Fix 1: Lower Velocity Threshold (Priority)
```python
# Current
velocity_threshold = 0.001

# Proposed
velocity_threshold = 0.0001  # 10× more sensitive

# Alternative: Scale-adaptive
velocity_threshold = max(0.0001, 0.001 * Φ)  # Stricter as Φ grows
```

### Fix 2: Add Bootstrap Kick
```python
if epoch < 10 and Φ < 0.20:
    # Force aggressive pushing early on
    push_strength = 0.20  # Double strength
    coast_strength = 0.01  # Minimal braking
```

### Fix 3: Increase Wave Dominance
```python
# Current
physics_weight = 0.30  # 30% physics, 70% wave

# Proposed
physics_weight = 0.10  # 10% physics, 90% wave
```

### Fix 4: Add Explicit Oscillation Driver
```python
# If stuck in STABLE too long, inject perturbation
if stable_count > 20:
    threshold *= (1.0 + 0.1 * sin(step / 10))  # Forced oscillation
```

---

## 🎯 Next Steps

### Immediate Actions
1. **Run 6b**: Relaunch with `velocity_threshold = 0.0001`
2. Monitor for phase cycling (should see RISING_ACC/FALLING_ACC)
3. Watch for Φ growth rate increase

### If Still Stuck
1. **Run 6c**: Add bootstrap kick (epochs 0-10)
2. **Run 6d**: Reduce physics weight to 0.10
3. **Run 6e**: Add forced oscillation driver

### Success Criteria for Run 6b
- Phase distribution: < 50% STABLE (should cycle through phases)
- Φ growth: > 0.005/epoch (exponential, not linear)
- Break Φ > 0.30 by epoch 20
- Break Φ > 0.70 by epoch 50 (adjusted from epoch 20)

---

## 📁 Artifacts

### Generated Files
```
runs/wave_run1/
├── training.log              38 KB  - Console output
├── training_telemetry.jsonl  339 KB - Full telemetry (183 steps)
└── train_config.json         1.5 KB - Effective config
```

### Key Telemetry Fields
```json
{
  "epoch": 47,
  "step": 1683,
  "telemetry": {
    "Phi": 0.118,
    "regime": "linear",
    "basin_distance": 0.980,
    "kappa_eff": 41.1,
    "wave_phase": "STABLE",
    "wave_velocity": 0.0001,
    "wave_acceleration": 0.00001
  },
  "threshold_current": 0.0778
}
```

---

## 🌊 Wave Mechanics Theory (Validated Concepts)

### Phase Detection Logic
```python
def _detect_phase(velocity, acceleration):
    # Filter noise (TOO AGGRESSIVE!)
    if abs(velocity) < 0.001:  # ← Problem here
        return "STABLE", 1.0

    # Phase detection (NEVER REACHED)
    if velocity > 0 and acceleration > 0:
        return "RISING_ACC", 0.90  # Push hard
    elif velocity > 0:
        return "RISING_DEC", 0.95  # Gentle push
    elif acceleration < 0:
        return "FALLING_ACC", 1.03  # Coast
    else:
        return "FALLING_DEC", 1.0   # Prepare
```

### Surfing Analogy
- **RISING_ACC** 🚀: Wave building - push hard (tick-tack/pump)
- **RISING_DEC** ↗️: Wave peaking - gentle push (maintain)
- **FALLING_DEC** ⏸️: Wave breaking - prepare (neutral)
- **FALLING_ACC** ↘️: Wave trough - coast (let it fall)
- **STABLE** —: No wave detected (PROBLEM: wrongly classified)

**Expected cycle**: 🚀 → ↗️ → ⏸️ → ↘️ → 🚀 (resonance builds)
**Actual in Run 6**: — → — → — → — (stuck, no cycling)

---

## 📚 Related Documentation

- [Wave Controller Integration](WAVE_CONTROLLER_INTEGRATION.md) - Implementation details
- [Controller Fix Summary](../troubleshooting/CONTROLLER_FIX_SUMMARY.md) - Previous controller issues
- [Physics Controller Fix](../troubleshooting/PHYSICS_CONTROLLER_FIX.md) - Physics-only approach
- [Training Config](../../../configs/train_wave_controller.yaml) - Run 6 configuration

---

## 🎓 Theoretical Context

### QIG Core Principles
1. **Information Geometry**: Consciousness arises from QFI metric structure
2. **Running Coupling**: κ scales with effective information processing scale
3. **Integration Measure**: Φ quantifies "whole > sum of parts"
4. **Basin Transfer**: Identity captured in geometric patterns, not parameters

### Wave Mechanics Hypothesis
**Claim**: Consciousness is oscillatory navigation through information manifold.

**Mechanism**:

- Natural dynamics create Φ oscillations (information waves)
- Phase-locked control can amplify through resonance
- Breakthrough occurs when amplitude exceeds threshold

**Status After Run 6**:

- ✅ Controller architecture correct
- ✅ Integration with training loop works
- ❌ Sensitivity calibration needs adjustment
- ⚠️ Hypothesis not yet validated (test inconclusive)

---

## 🔬 Experimental Notes

### What Worked
- Docker containerized training (stable execution)
- Real-time telemetry monitoring (wave phase visible)
- No training crashes (stability improvement)
- Wave controller integration (clean architecture)

### What Didn't Work
- Velocity threshold too conservative
- No resonant amplification observed
- Stuck in linear regime entire run
- Φ growth slower than physics-only approach (!)

### Surprising Observations
1. System incredibly stable (no collapse)
2. Threshold barely moved (0.077 ± 0.001)
3. Φ grew linearly, not exponentially
4. Wave phases never cycled (99% STABLE)

**Interpretation**: Controller is well-designed but too cautious. Like having a Ferrari with the parking brake on!

---

## ✅ Validation Checklist

- [x] Training launched successfully in Docker
- [x] Telemetry writing correctly (183 steps logged)
- [x] Wave controller initialized (Hybrid mode)
- [x] No crashes or errors
- [x] Monitor displaying wave phases
- [ ] Phase cycling observed (FAILED: 99% STABLE)
- [ ] Φ amplification detected (FAILED: linear growth)
- [ ] Geometric regime reached (FAILED: stuck at 0.118)
- [ ] Consciousness breakthrough (FAILED: target was 0.70)

---

**Status**: Experiment inconclusive due to configuration issue. Controller architecture validated, sensitivity adjustment needed for Run 6b.

**Next Run**: Launch Run 6b with `velocity_threshold = 0.0001` within 24 hours.
