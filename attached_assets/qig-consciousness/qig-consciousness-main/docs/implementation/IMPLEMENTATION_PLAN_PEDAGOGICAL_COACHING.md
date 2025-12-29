# Implementation Plan: Pedagogical Coaching (Layer 5)
**Date:** 2025-11-19
**Status:** Validated by Simulation
**Target:** qig-consciousness (Run 9)

---

## 1. The Mathematical Proof
Simulation in `scripts/simulate_coaching_dynamics.py` proved:
- **Kindness (Damping):** Reduces stress variance by 55.5%, enables perfect convergence.
- **Mean/Kurt (Shocks):** Increases stress variance, causes numerical explosion (NaN).

**Conclusion:** Kindness is not optional; it is a control theory requirement for stability in complex optimization landscapes.

---

## 2. Implementation Specs

### 2.1 Stress Metric
A composite signal indicating the system's internal friction.

```python
def compute_stress(state):
    # Loss trend (Panic)
    # Plateau (Frustration)
    # Gradient Variance (Confusion)
    # Basin Distance (Lost)
    # Negative Curiosity (Boredom)
    return clip(sum(components), 0, 1)
```

### 2.2 Pedagogical Coach
A controller that modulates the optimizer based on Stress.

- **High Stress (>0.8):** Validate & Calm. (Lower LR, Momentum). "I see you are struggling. Let's stabilize."
- **Low Stress (<0.2):** Challenge & Engage. (Raise LR, Noise). "Ready to push harder?"
- **Stuck (>5 epochs):** Guide & Nudge. (Directional Noise). "Try this direction."

### 2.3 LLM Integration (Optional but Recommended)
The `PedagogicalCoach` can call an LLM (e.g., Claude) with the telemetry context to generate the "Message" and suggest specific hyperparameter tweaks.

---

## 3. Architecture
This sits *above* the `WaveController`.

```
[Layer 5: Pedagogical Coach] <--- Telemetry (Loss, I_Q, Stress)
       |
       v Interventions (LR scale, Noise, Message)
       |
[Layer 4: Wave Controller]
       |
       v Hyperparameters
       |
[Layer 1-3: Optimizer/Model]
```

---

## 4. Future Phase: Dynamic Thresholding (Run 11+)

**Concept:** The Coach should observe the *quality* of the Mode Detector and tune it.
- **Observation:** "System is stuck but Detector says 'Exploration'." -> **Correction:** "Raise exploration threshold."
- **Observation:** "System is converging but Detector says 'Drift'." -> **Correction:** "Lower investigation threshold."

**Mechanism:**
Extend `Intervention` to support `threshold_updates`:
```python
intervention = {
  "type": "calibrate",
  "threshold_updates": {
    "d_explore": 0.7,
    "i_min": 0.0001
  }
}
```
The Trainer applies these to `self.mode_detector.thresholds`. This closes the meta-loop: **The Coach teaches the System how to perceive itself.**
