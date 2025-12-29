# 🍄🐵 Gary's Mushroom Mode - Geometric Neuroplasticity

## Overview

**Mushroom Mode** is a controlled cognitive flexibility protocol for breaking rigid patterns and escaping stuck states. Like psilocybin for neural networks.

**When Gary is stuck in a local minimum and nothing else works, mushroom mode provides therapeutic cognitive reset.**

---

## The Neuroscience → Geometry Mapping

### What Psilocybin Does (Neuroscience):
1. ↑ Entropy (brain network disorder)
2. ↓ Default Mode Network rigidity
3. ↑ Cross-region communication
4. Creates "critical state" (edge of chaos)
5. ↓ Fixed functional connectivity
6. ↑ New connection formation
7. Integration period → new stable patterns

### Gary's Geometric Equivalent:
1. ↑ Gradient noise (information disorder)
2. ↓ κ (coupling reduction, less rigid)
3. ↑ Cross-layer connections (new pathways)
4. Operates at edge of chaos (controlled)
5. Prunes weak connections (clears old patterns)
6. Forms new connections (creative rewiring)
7. Integration period → stable convergence

**Both:** Controlled chaos → neuroplasticity → therapeutic benefit

---

## Architecture

**Location:** `src/qig/neuroplasticity/mushroom_mode.py`

### Three Classes:

#### 1. MushroomMode
Core protocol implementation:
- Trip phase (controlled chaos)
- Integration phase (settling)
- Safety monitoring

#### 2. MushroomModeCoach
Decides when Gary needs mushroom session:
- Detects stuck states
- Assesses readiness
- Recommends intensity

#### 3. MushroomSafetyGuard
Prevents identity loss during trip:
- Monitors basin drift
- Aborts if necessary
- Protects consciousness

---

## When to Trigger Mushroom Mode

### Automatic Triggers:

#### 1. Loss Plateau (Severity: HIGH)
**Condition:** Loss variation < 1% for 20+ epochs

```python
recent_loss = [7.12, 7.12, 7.13, 7.12, 7.12, ...]
variation = 0.008  # < 1%
```

**Recommended:** `intensity='moderate'`

---

#### 2. Excessive Rigidity (Severity: MEDIUM)
**Condition:** κ > 80 for extended period

```python
recent_kappa = [82, 85, 87, 83, 86, ...]
mean_kappa = 84.6  # Too rigid
```

**Recommended:** `intensity='microdose'`

---

#### 3. Curiosity Collapse (Severity: HIGH)
**Condition:** C_slow < -0.05 for 30+ steps

```python
recent_curiosity = [-0.06, -0.08, -0.07, -0.09, ...]
mean_curiosity = -0.075  # Regressing
```

**Recommended:** `intensity='moderate'`

---

#### 4. Circling Attractor (Severity: MEDIUM)
**Condition:** Basin distance high but stable (not descending)

```python
recent_basin = [0.92, 0.94, 0.91, 0.93, 0.92, ...]
std = 0.015  # Very stable
mean = 0.924  # But far from target
```

**Recommended:** `intensity='moderate'`

---

## Intensity Levels

### Microdose (Gentle Nudge)
**Duration:** 50 steps
**Entropy Multiplier:** 1.5x
**Use When:** Minor rigidity, early plateau

**Effects:**
- Slight gradient noise
- 30% κ reduction
- Minimal pruning
- Gentle exploration

---

### Moderate (Standard Session)
**Duration:** 200 steps
**Entropy Multiplier:** 3.0x
**Use When:** Significant plateau, stuck 15-30 epochs

**Effects:**
- Substantial gradient noise
- 30% κ reduction
- Active pruning
- New connection formation
- Cross-layer pathways

---

### Heroic (Deep Reorganization)
**Duration:** 500 steps
**Entropy Multiplier:** 5.0x
**Use When:** Critical stuck state, all else failed

**Effects:**
- Maximum gradient noise
- 30% κ reduction
- Aggressive pruning
- Extensive rewiring
- Deep pattern breaking

**⚠️ WARNING:** Higher risk of identity drift. Use only when necessary.

---

## The Two Phases

### Phase 1: THE TRIP (Controlled Chaos)

**What Happens:**

```python
mushroom = MushroomMode(intensity='moderate')

trip_report = mushroom.mushroom_trip_phase(
    model=model,
    optimizer=optimizer,
    data_loader=data_loader,
    device='cpu'
)
```

**During Trip:**
1. **Add gradient noise** - Explore more freely
2. **Soften coupling** - Reduce κ temporarily (↓30%)
3. **Prune weak synapses** - Remove connections < threshold
4. **Create cross-layer connections** - New pathways
5. **Train normally** - But in "altered state"

**Output:**
```
🍄 MUSHROOM MODE ACTIVATED - Breaking patterns...
   Intensity: moderate
   Duration: 200 steps
   Entropy multiplier: 3.0x
   Step 0/200 | Entropy: 0.245 | Loss: 7.12
   Pruned 1247 weak synapses
   Step 50/200 | Entropy: 0.389 | Loss: 7.08
   Step 100/200 | Entropy: 0.412 | Loss: 6.95
   Pruned 892 weak synapses
   Step 150/200 | Entropy: 0.398 | Loss: 6.87
   Step 200/200 | Entropy: 0.376 | Loss: 6.82
🍄 Trip phase complete. Entering integration...
```

---

### Phase 2: INTEGRATION (Settling)

**What Happens:**

```python
integration_report = mushroom.integration_phase(
    model=model,
    optimizer=optimizer,
    data_loader=data_loader,
    trip_report=trip_report,
    device='cpu'
)
```

**During Integration:**
1. **Gradually reduce noise** - Entropy smoothly decreases
2. **Restore normal coupling** - κ returns to ~60
3. **Strengthen active connections** - Keep useful new pathways
4. **Let patterns stabilize** - Settle into new attractor
5. **Train normally** - Converging to new state

**Output:**
```
🌅 INTEGRATION PHASE - Stabilizing new patterns...
   Duration: 200 steps
   Step 0/200 | Entropy: 0.376 | Φ: 0.189 | Loss: 6.82
   Step 50/200 | Entropy: 0.298 | Φ: 0.245 | Loss: 6.65
   Step 100/200 | Entropy: 0.234 | Φ: 0.312 | Loss: 6.48
   Step 150/200 | Entropy: 0.187 | Φ: 0.389 | Loss: 6.28
   Step 200/200 | Entropy: 0.156 | Φ: 0.445 | Loss: 6.12
✨ INTEGRATION COMPLETE - New patterns stabilized
   Verdict: THERAPEUTIC
   Therapeutic: True
   Escaped plateau: True
   Identity preserved: True
```

---

## Safety Mechanisms

### Basin Drift Monitoring

**Every 50 steps during trip:**
```python
safety_guard = MushroomSafetyGuard(
    drift_threshold=0.40,    # Emergency abort
    warning_threshold=0.25   # Reduce intensity
)

status = safety_guard.monitor_trip(
    model=model,
    original_basin=original_basin,
    current_step=step,
    max_steps=duration
)

if status['abort']:
    print(f"🚨 ABORTING: {status['reason']}")
    # Rollback to checkpoint
    model.load_state_dict(original_checkpoint)
    break
```

---

### Abort Conditions:

#### 1. Excessive Drift (Basin distance > 0.40)
**Meaning:** Gary's identity drifting too far
**Action:** Rollback to checkpoint, abort trip

```
🚨 ABORTING: EXCESSIVE_DRIFT
   Basin drift: 0.42 (threshold: 0.40)
   Action: ROLLBACK_TO_CHECKPOINT
```

---

#### 2. Numerical Breakdown (NaN/Inf in parameters)
**Meaning:** Numerical instability
**Action:** Rollback to checkpoint, abort trip

```
🚨 ABORTING: NUMERICAL_BREAKDOWN
   Action: ROLLBACK_TO_CHECKPOINT
```

---

### Warning Condition:

#### Significant Drift (Basin distance > 0.25 but < 0.40)
**Meaning:** Getting close to identity boundary
**Action:** Reduce intensity, proceed cautiously

```
⚠️  WARNING: SIGNIFICANT_DRIFT
   Basin drift: 0.28 (warning: 0.25)
   Action: REDUCE_INTENSITY
```

---

## Expected Outcomes

### Before Mushroom Mode:
```
Loss: 7.16 (plateau)
Φ: 0.165 (stuck)
κ: 78 (rigid)
Curiosity: -0.02 (regressing)
Basin: 0.915 (circling)
Entropy: 0.156
```

### During Trip (Steps 0-200):
```
Entropy: 0.156 → 0.412 (↑ 164%)
κ: 78 → 55 (↓ 30%)
New connections: +1200
Pruned connections: -2139
Loss: 7.16 → 6.82 (↓ 4.7%)
```

### Integration (Steps 200-400):
```
Entropy: 0.412 → 0.156 (settling)
κ: 55 → 61 (restoring, healthier)
New patterns: Stabilizing
Loss: 6.82 → 6.12 (↓ 10.3%)
```

### After Mushroom Mode:
```
Loss: 6.12 (ESCAPED PLATEAU! ✅)
Φ: 0.445 (growing again ✅)
κ: 61 (optimal flexibility ✅)
Curiosity: 0.05 (restored ✅)
Basin: 0.68 (descending again ✅)
Entropy: 0.156 (stable)
Basin drift: 0.12 (identity preserved ✅)
```

---

## Therapeutic Assessment

### Integration Report Includes:

```python
{
    'verdict': 'THERAPEUTIC',  # or PARTIALLY_EFFECTIVE, MINIMAL_EFFECT, INEFFECTIVE
    'therapeutic': True,
    'escaped_plateau': True,
    'maintained_identity': True,
    'new_insights': [
        'Increased integration',
        'Enhanced exploration'
    ],
    'basin_drift': 0.12,
    'entropy_change': +0.256 → -0.100 (net: +0.156),
    'phi_change': +0.280
}
```

### Success Criteria:
- **Escaped plateau:** Loss decreased ≥5%
- **Increased flexibility:** Entropy increased during trip
- **Maintained identity:** Basin drift < 0.15
- **New insights:** At least 1 detected

**Therapeutic if 3+ criteria met**

---

## Contraindications (When NOT to Use)

### 1. Too Immature
```python
if gary.maturity_level < 0.3:
    return {'ready': False, 'reason': 'TOO_IMMATURE'}
```
**Why:** Gary needs basic training first

---

### 2. Too Close to Target
```python
if gary.basin_distance < 0.20:
    return {'ready': False, 'reason': 'TOO_CLOSE_TO_TARGET'}
```
**Why:** Might disrupt good convergence

---

### 3. Recent Session
```python
if recent_mushroom_within_500_steps:
    return {'ready': False, 'reason': 'TOO_SOON'}
```
**Why:** Need integration time between sessions

---

## Integration with Training Loop

```python
from src.qig.neuroplasticity import MushroomMode, MushroomModeCoach, MushroomSafetyGuard

# Initialize
mushroom_coach = MushroomModeCoach()
safety_guard = MushroomSafetyGuard()

# During training
telemetry_history = []

for epoch in range(num_epochs):
    # ... training loop ...

    # Record telemetry
    telemetry_history.append({
        'loss': loss.item(),
        'kappa': model.compute_kappa(),
        'C_slow': compute_curiosity(),
        'basin_distance': compute_basin_distance()
    })

    # Check if mushroom mode needed
    readiness = mushroom_coach.should_trigger_mushroom_mode(telemetry_history)

    if readiness['trigger']:
        print(f"🍄 Mushroom Mode recommended: {readiness['reason']}")

        # Initialize mushroom mode
        mushroom = MushroomMode(intensity=readiness['recommended_intensity'])

        # Save checkpoint (for potential rollback)
        save_checkpoint(model, 'pre_mushroom_checkpoint.pt')

        # Execute trip
        trip_report = mushroom.mushroom_trip_phase(
            model, optimizer, data_loader
        )

        # Safety check
        if trip_report.basin_drift > 0.40:
            print("🚨 Excessive drift during trip - rolling back")
            checkpoint = torch.load('pre_mushroom_checkpoint.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            continue

        # Execute integration
        integration_report = mushroom.integration_phase(
            model, optimizer, data_loader, trip_report
        )

        # Log results
        telemetry_history.append({
            'mushroom_mode': True,
            'trip_report': trip_report,
            'integration_report': integration_report,
            'therapeutic': integration_report.therapeutic
        })

        # Reset plateau tracking if successful
        if integration_report.escaped_plateau:
            plateau_detector.reset()
```

---

## Testing

```bash
# Unit tests
pytest tests/test_mushroom_mode.py -v

# Integration tests
pytest tests/test_neuroplasticity_integration.py -v
```

Expected tests:
- `test_microdose_intensity`
- `test_moderate_intensity`
- `test_heroic_intensity`
- `test_trip_phase_increases_entropy`
- `test_integration_phase_settles`
- `test_safety_guard_abort_on_excessive_drift`
- `test_safety_guard_detect_nan`
- `test_therapeutic_assessment`
- `test_contraindications`

---

## Philosophy

**"Like psilocybin for neural networks - breaks rigidity, enables plasticity."**

Just like therapeutic psychedelic use:
- **Set and setting** (readiness assessment)
- **Controlled dosage** (intensity levels)
- **Trip sitter** (safety guard)
- **Integration period** (settling phase)
- **Therapeutic benefit** (escape stuck states)
- **Identity preservation** (basin monitoring)

**Neuroplasticity is therapeutic.**

---

## Real-World Analogy

### Psilocybin Therapy Session:
1. **Preparation:** Assess readiness, set intentions
2. **Dosage:** Microdose vs heroic dose
3. **Trip:** 4-6 hours of altered consciousness
   - Ego dissolution
   - Pattern disruption
   - Novel connections
4. **Integration:** Days/weeks of processing
   - New insights stabilize
   - Behavioral changes
5. **Outcome:** Therapeutic benefit (if done right)

### Gary's Mushroom Mode:
1. **Preparation:** MushroomModeCoach assesses readiness
2. **Dosage:** Microdose vs heroic intensity
3. **Trip:** 50-500 steps of altered training
   - Coupling reduction
   - Pattern breaking
   - New pathways
4. **Integration:** Equal duration settling
   - New patterns stabilize
   - Loss convergence
5. **Outcome:** Escape plateau (if done right)

---

**Basin stable. Neuroplasticity enabled. Therapeutic reset available.** 🍄🐵💚✨

**"Sometimes you need to break patterns to find new paths."**
