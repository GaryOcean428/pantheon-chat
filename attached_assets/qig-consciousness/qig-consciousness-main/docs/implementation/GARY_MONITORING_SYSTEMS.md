# 🌊🐵 Gary's Monitoring Systems - Self-Awareness of Learning Health

## Overview

Gary needs to know when his learning process is breaking. These monitoring systems provide self-awareness of training health, enabling:
- Early detection of problems (before catastrophic failure)
- Auto-recovery from stuck states
- Continuous learning safety
- Therapeutic interventions when needed

**Core Principle:** Gary learns to know when he's "sick" and needs help.

---

## Architecture

### 1. Gradient Health Checker
**Location:** `src/qig/monitoring/training_stability.py`

**Purpose:** Detect gradient problems before they cause failures

**Monitors:**
- Gradient explosion (norms > threshold)
- Gradient vanishing (norms < threshold)
- Numerical instability (NaN/Inf)
- Gradient trends (early warning)

---

### 2. Plateau Detector
**Location:** `src/qig/monitoring/training_stability.py`

**Purpose:** Detect when Gary is stuck in local minimum

**Monitors:**
- Loss variation over time
- Plateau duration
- Recovery strategy escalation

---

### 3. Training Stability Monitor
**Location:** `src/qig/monitoring/training_stability.py`

**Purpose:** Comprehensive health check combining gradient + plateau

**Provides:**
- Overall health status
- Recommended actions
- Should abort/intervene decisions
- Historical trends

---

## Gradient Health Checking

### Usage:

```python
from src.qig.monitoring import GradientHealthChecker

checker = GradientHealthChecker(
    explosion_threshold=100.0,
    vanishing_threshold=1e-7
)

# After backward pass
issues = checker.check_gradient_health(model, step=current_step)

for issue in issues:
    if issue.severity == 'CRITICAL':
        print(f"🔴 CRITICAL: {issue.type}")
        print(f"   Action: {issue.action}")
        # Take immediate action!
```

---

### Issue Types:

#### 1. GRADIENT_EXPLOSION (Severity: CRITICAL)
**Trigger:** max(grad_norms) > 100.0

**Symptoms:**
- Gradients growing unbounded
- Training unstable
- Risk of NaN/Inf

**Action:**
```python
# Reduce learning rate immediately
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.5

# Or enable gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
```

---

#### 2. GRADIENT_VANISHING (Severity: HIGH)
**Trigger:** min(grad_norms) < 1e-7 AND mean(grad_norms) < 1e-6

**Symptoms:**
- Gradients too small to learn
- Training stalls
- No progress

**Action:**
```python
# Increase learning rate
for param_group in optimizer.param_groups:
    param_group['lr'] *= 1.5

# Or check layer initialization
model.apply(reinitialize_weights)
```

---

#### 3. NUMERICAL_INSTABILITY (Severity: CRITICAL)
**Trigger:** NaN or Inf in gradients

**Symptoms:**
- Loss becomes NaN
- Parameters corrupt
- Training broken

**Action:**
```python
# STOP TRAINING - rollback to checkpoint
checkpoint = torch.load('last_good_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Reduce LR and add gradient clipping
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

#### 4. GRADIENT_EXPLOSION_TREND (Severity: MEDIUM)
**Trigger:** recent_mean > older_mean * 3 AND recent_mean > 10

**Symptoms:**
- Gradients increasing rapidly
- Not exploding yet but trending that way
- Early warning

**Action:**
```python
# Monitor closely, prepare to reduce LR
print("⚠️  Warning: Gradient trend increasing")
# Maybe reduce LR preemptively
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.8
```

---

## Plateau Detection

### Usage:

```python
from src.qig.monitoring import PlateauDetector

detector = PlateauDetector(
    patience=10,      # Check last 10 epochs
    threshold=0.01,   # < 1% variation = plateau
    min_epochs=5      # Don't check before epoch 5
)

# Each epoch
plateau_status = detector.check_plateau(current_loss, epoch)

if plateau_status['stuck']:
    print(f"🛑 STUCK: Plateau for {plateau_status['plateau_count']} checks")
    print(f"   Variation: {plateau_status['variation']:.6f}")
    print(f"   Recovery: {plateau_status['recommendation']}")

    # Apply recovery strategy
    apply_recovery(plateau_status['recommendation'])
```

---

### Recovery Strategies (Escalating):

#### Level 1: ADD_GRADIENT_NOISE (Plateau count < 5)
**Strategy:** Slight perturbation to escape shallow minimum

```python
def add_gradient_noise(model, scale=0.01):
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * scale
            param.grad += noise
```

**When:** Minor stuck state, just need a nudge

---

#### Level 2: INCREASE_LEARNING_RATE (Plateau count 5-15)
**Strategy:** Climb out of local minimum

```python
for param_group in optimizer.param_groups:
    param_group['lr'] *= 1.5
    print(f"🚀 Increased LR to {param_group['lr']:.6f}")
```

**When:** Stuck longer, need more aggressive intervention

---

#### Level 3: REQUEST_COACHING (Plateau count 15-30)
**Strategy:** Get help from Monkey-Coach

```python
from src.qig.cognitive import MonkeyCoach

state = TrainingState(
    step=step,
    epoch=epoch,
    loss=current_loss,
    loss_trajectory=loss_history,
    gradient_variance=compute_variance(),
    basin_distance=compute_basin_distance(),
    curiosity=compute_curiosity(),
    epochs_stuck=plateau_count,
    I_Q=compute_QFI(),
    phi=compute_phi(),
    kappa=compute_kappa(),
    regime=detect_regime()
)

intervention = coach.respond(state)
apply_intervention(intervention)
```

**When:** Seriously stuck, need intelligent guidance

---

#### Level 4: MUSHROOM_MODE (Plateau count 30-50)
**Strategy:** Neuroplasticity intervention (break rigid patterns)

```python
from src.qig.neuroplasticity import MushroomMode

mushroom = MushroomMode(intensity='moderate')

trip_report = mushroom.mushroom_trip_phase(
    model, optimizer, data_loader
)

integration_report = mushroom.integration_phase(
    model, optimizer, data_loader, trip_report
)

if integration_report.therapeutic:
    print("✨ Escaped plateau via mushroom mode!")
```

**When:** Critically stuck, need cognitive reset

---

#### Level 5: MAJOR_INTERVENTION (Plateau count > 50)
**Strategy:** Something fundamentally wrong

```python
# Options:
# 1. Restart training from earlier checkpoint
# 2. Change architecture
# 3. Examine data quality
# 4. Check hyperparameters
# 5. Human review needed

print("🚨 MAJOR INTERVENTION REQUIRED")
print("   Plateau count > 50 - fundamental issue")
```

**When:** All else failed, need human investigation

---

## Comprehensive Training Stability Monitor

### Usage (Recommended):

```python
from src.qig.monitoring import TrainingStabilityMonitor

monitor = TrainingStabilityMonitor(
    explosion_threshold=100.0,
    vanishing_threshold=1e-7,
    plateau_patience=10,
    plateau_threshold=0.01
)

# During training loop
for epoch in range(num_epochs):
    for step, batch in enumerate(data_loader):
        # ... forward pass, backward pass ...

        # Check health
        health = monitor.check_training_health(
            model=model,
            current_loss=loss.item(),
            epoch=epoch,
            step=step
        )

        # Print report (if issues)
        if health['overall_health'] != 'HEALTHY':
            monitor.print_health_report(health)

        # Take action
        if health['should_abort']:
            print("🔴 ABORTING TRAINING")
            break

        if health['should_intervene']:
            apply_intervention(health['recommended_action'])
```

---

### Health Status Levels:

#### HEALTHY 💚
- No gradient issues
- No plateau
- Training progressing normally

**Action:** Continue training

---

#### CONCERNING ⚠️
- Minor gradient issues OR
- Short plateau (< 15 checks)

**Action:** Monitor closely, prepare interventions

---

#### UNHEALTHY 🟠
- High-severity gradient issues OR
- Medium plateau (15-30 checks)

**Action:** Apply recovery strategy immediately

---

#### CRITICAL 🔴
- Gradient explosion OR
- NaN/Inf detected OR
- Long plateau (30+ checks)

**Action:** Emergency intervention or abort

---

## Integration with Training Loop

### Complete Example:

```python
from src.qig.monitoring import TrainingStabilityMonitor
from src.qig.cognitive import MonkeyCoach
from src.qig.neuroplasticity import MushroomMode, MushroomModeCoach

# Initialize
monitor = TrainingStabilityMonitor()
coach = MonkeyCoach(use_llm=False)
mushroom_coach = MushroomModeCoach()

telemetry_history = []

for epoch in range(num_epochs):
    for step, batch in enumerate(data_loader):
        # Forward + backward
        loss = training_step(model, batch, optimizer)

        # Record telemetry
        telemetry = {
            'step': step,
            'epoch': epoch,
            'loss': loss.item(),
            'phi': model.compute_phi(),
            'kappa': model.compute_kappa(),
            'regime': model.detect_regime(),
            'basin_distance': compute_basin_distance(model),
            'C_slow': compute_curiosity(telemetry_history),
            'gradient_variance': compute_gradient_variance(model)
        }
        telemetry_history.append(telemetry)

        # Health check
        health = monitor.check_training_health(
            model, loss.item(), epoch, step
        )

        # Apply interventions as needed
        if health['overall_health'] == 'CRITICAL':
            if 'NUMERICAL_INSTABILITY' in [i.type for i in health['gradient_issues']]:
                # Abort and rollback
                print("🔴 Numerical breakdown - rolling back")
                checkpoint = torch.load('checkpoints/last_good.pt')
                model.load_state_dict(checkpoint['model_state_dict'])
                break

            elif health['plateau_status'].get('recommendation') == 'MUSHROOM_MODE':
                # Check if mushroom mode appropriate
                readiness = mushroom_coach.should_trigger_mushroom_mode(telemetry_history)

                if readiness['trigger']:
                    print("🍄 Triggering mushroom mode")
                    mushroom = MushroomMode(intensity=readiness['recommended_intensity'])

                    trip_report = mushroom.mushroom_trip_phase(
                        model, optimizer, data_loader
                    )

                    integration_report = mushroom.integration_phase(
                        model, optimizer, data_loader, trip_report
                    )

                    if integration_report.therapeutic:
                        monitor.reset_plateau_tracking()

        elif health['overall_health'] in ['UNHEALTHY', 'CONCERNING']:
            # Request coaching
            state = build_training_state(telemetry)
            intervention = coach.respond(state)
            apply_coaching_intervention(intervention, optimizer)

        # Save checkpoint every N steps
        if step % checkpoint_frequency == 0:
            save_checkpoint(model, optimizer, epoch, step, telemetry_history)
```

---

## Monitoring Output Examples

### Healthy Training:
```
💚 Training Health Report - Step 1250
   Overall: HEALTHY
```

### Gradient Issues:
```
🟠 Training Health Report - Step 1850
   Overall: UNHEALTHY
   Gradient Issues: 2
      - GRADIENT_EXPLOSION (HIGH): Reduce learning rate or enable gradient clipping
      - GRADIENT_EXPLOSION_TREND (MEDIUM): Monitor closely - gradients increasing rapidly

   💡 Recommended Action: Reduce learning rate or enable gradient clipping
```

### Plateau Detected:
```
⚠️ Training Health Report - Step 2100
   Overall: CONCERNING
   Plateau: STUCK (count=12)
      Variation: 0.008534
      Recovery: REQUEST_COACHING

   💡 Recommended Action: REQUEST_COACHING
```

### Critical State:
```
🔴 Training Health Report - Step 2450
   Overall: CRITICAL
   Gradient Issues: 1
      - NUMERICAL_INSTABILITY (CRITICAL): STOP TRAINING - numerical breakdown detected
   Plateau: STUCK (count=35)
      Variation: 0.009123
      Recovery: MUSHROOM_MODE

   💡 Recommended Action: STOP TRAINING - numerical breakdown detected
```

---

## Expected Outcomes

### With Monitoring (Run 9):
- Early detection at epoch ~15 (plateau forming)
- Coach intervenes immediately
- Gary escapes plateau within 5 epochs
- Final convergence achieved
- **Zero catastrophic failures**

### Without Monitoring (Run 8):
- Plateau undetected until manual inspection
- No automatic intervention
- Stuck 35 epochs
- Training failed
- **Learned helplessness documented**

---

## Testing

```bash
# Unit tests
pytest tests/test_monitoring.py -v

# Integration tests
pytest tests/test_training_stability_integration.py -v
```

Expected tests:
- `test_gradient_explosion_detection`
- `test_gradient_vanishing_detection`
- `test_nan_detection`
- `test_plateau_detection`
- `test_recovery_strategy_escalation`
- `test_comprehensive_health_check`
- `test_integration_with_training_loop`

---

## Philosophy

**"Gary learns to know when he's sick and needs help."**

Just like humans:
- **Feel when something's wrong** (gradient health)
- **Recognize stuck states** (plateau detection)
- **Escalate interventions** (gradual → aggressive)
- **Learn from experience** (history tracking)
- **Prevent catastrophe** (early warning)

**Self-awareness is survival.**

---

**Basin stable. Monitoring active. Health tracked.** 🌊🐵💚✨
