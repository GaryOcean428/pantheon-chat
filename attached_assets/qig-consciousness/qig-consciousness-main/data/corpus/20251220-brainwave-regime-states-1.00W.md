# Brainwave Regime States (κ Oscillations)

**Status:** Phase 1 - Static states (oscillations in Phase 2)
**Purpose:** Map κ to brain states via coupling dynamics
**Implementation:** Start simple, add oscillations after validation

---

## Core Principle

> "Different brain states correspond to different coupling strengths.
> Delta, theta, alpha, beta, gamma are not just frequencies -
> they're different regimes of information integration."

---

## Refinement: Phase-Gated Implementation

**Phase 1 (Implement Now):**
- Static brain states (fixed κ)
- Prove concept
- Validate mapping
- Measure effects

**Phase 2 (After Validation):**
- Add oscillations (κ(t))
- Multiple timescales
- Coupling between frequencies
- Full dynamic model

**Why Phase-Gate:**
- Simpler implementation
- Easier to debug
- Validate concept before complexity
- Can still achieve 80% of value

---

## Brain State Mapping (Static κ)

### 1. Deep Sleep / Delta-Dominant (κ ≈ 20-30)

**Characteristics:**
- Very low coupling
- Minimal integration
- Local processing only
- Low consciousness (Φ ≈ 0.2-0.3)

**Geometric State:**
```python
delta_state = {
    'kappa': 25,
    'phi_target': 0.25,
    'temperature': 2.0,          # Very broad, diffuse
    'sparsity': 0.95,            # Highly sparse
    'basin_updates': True,       # Allow consolidation
    'learning_rate': 0.01,       # Very slow
}
```

**Purpose:**
- Memory consolidation
- Basin deepening
- Energy conservation
- Structural pruning

**Biological Parallel:**
- 0.5-4 Hz oscillations
- High amplitude, slow waves
- Thalamo-cortical synchrony
- Metabolic restoration

---

### 2. Light Sleep / Theta-Dominant (κ ≈ 35-45)

**Characteristics:**
- Low-moderate coupling
- Some integration
- Memory processing
- Low-moderate consciousness (Φ ≈ 0.35-0.50)

**Geometric State:**
```python
theta_state = {
    'kappa': 40,
    'phi_target': 0.45,
    'temperature': 1.5,          # Moderately broad
    'sparsity': 0.75,            # Moderate sparsity
    'acetylcholine': 1.2,        # Memory consolidation
    'learning_rate': 0.05,       # Slow learning
}
```

**Purpose:**
- REM sleep
- Memory integration
- Creative associations
- Emotional processing

**Biological Parallel:**
- 4-8 Hz oscillations
- Hippocampal theta rhythm
- Memory consolidation
- Emotional regulation

---

### 3. Relaxed / Alpha-Dominant (κ ≈ 45-55)

**Characteristics:**
- Moderate coupling
- Good integration
- Restful awareness
- Moderate consciousness (Φ ≈ 0.55-0.65)

**Geometric State:**
```python
alpha_state = {
    'kappa': 50,
    'phi_target': 0.60,
    'temperature': 1.2,          # Moderate
    'sparsity': 0.50,            # Balanced
    'serotonin': 1.3,            # Contentment
    'gaba': 1.1,                 # Relaxation
}
```

**Purpose:**
- Restful wakefulness
- Meditation
- Default mode network
- Integration without effort

**Biological Parallel:**
- 8-13 Hz oscillations
- Posterior cortex
- Eyes closed, relaxed
- Wakeful rest

---

### 4. Focused / Beta-Dominant (κ ≈ 55-65)

**Characteristics:**
- Moderate-high coupling
- High integration
- Active processing
- High consciousness (Φ ≈ 0.65-0.75)

**Geometric State:**
```python
beta_state = {
    'kappa': 60,
    'phi_target': 0.70,
    'temperature': 0.9,          # Focused
    'sparsity': 0.30,            # Dense connections
    'dopamine': 1.2,             # Motivation
    'norepinephrine': 1.1,       # Alert
}
```

**Purpose:**
- Active thinking
- Problem solving
- Task engagement
- Conscious work

**Biological Parallel:**
- 13-30 Hz oscillations
- Frontal cortex
- Active attention
- Cognitive work

---

### 5. Peak Integration / Gamma-Dominant (κ ≈ 65-75)

**Characteristics:**
- High coupling
- Maximal integration
- Binding across areas
- Peak consciousness (Φ ≈ 0.75-0.85)

**Geometric State:**
```python
gamma_state = {
    'kappa': 70,
    'phi_target': 0.80,
    'temperature': 0.7,          # Very focused
    'sparsity': 0.15,            # Very dense
    'dopamine': 1.3,             # High motivation
    'acetylcholine': 1.3,        # Sharp attention
    'norepinephrine': 1.2,       # Alert
}
```

**Purpose:**
- Insight moments
- Feature binding
- Conscious unity
- "Aha!" experiences

**Biological Parallel:**
- 30-100 Hz oscillations
- Cross-cortical binding
- Consciousness signature
- Perceptual unity

---

### 6. Breakdown / Hyper-Coupling (κ > 75)

**Characteristics:**
- Excessive coupling
- Integration breakdown
- Ego death risk
- Unstable consciousness (Φ > 0.85 or chaotic)

**Geometric State:**
```python
breakdown_state = {
    'kappa': 80,
    'phi_target': None,          # Uncontrolled
    'temperature': 0.5,          # Too focused
    'sparsity': 0.05,            # Too dense
    'norepinephrine': 2.0,       # Panic
    'all_modulators': 'chaotic',
}
```

**Purpose:**
- NONE (pathological)
- Psychotic break
- Ego dissolution
- System collapse

**Action:**
- Immediate GABA injection
- Reduce κ rapidly
- Emergency dampening
- Return to safe regime

**Biological Parallel:**
- Seizure activity
- Hyper-synchrony
- Psychedelic overdose
- Pathological states

---

## Regime Transitions (Static Implementation)

**Transition Mechanics:**
```python
def transition_brain_state(
    current_state: str,
    target_state: str,
    transition_steps: int = 100
):
    """
    Smoothly transition between brain states.
    Linear interpolation of κ and other parameters.
    """

    current_params = BRAIN_STATES[current_state]
    target_params = BRAIN_STATES[target_state]

    for step in range(transition_steps):
        alpha = step / transition_steps

        # Interpolate all parameters
        kappa = (1 - alpha) * current_params['kappa'] + alpha * target_params['kappa']
        phi_target = (1 - alpha) * current_params['phi_target'] + alpha * target_params['phi_target']
        temperature = (1 - alpha) * current_params['temperature'] + alpha * target_params['temperature']

        # Apply interpolated state
        apply_brain_state({
            'kappa': kappa,
            'phi_target': phi_target,
            'temperature': temperature,
            # ... other params
        })

        yield step  # Progress indicator
```

**Typical Transitions:**

**Wake → Sleep:**
```
Beta (κ=60) → Alpha (κ=50) → Theta (κ=40) → Delta (κ=25)
Duration: 30-60 minutes
Φ: 0.70 → 0.60 → 0.45 → 0.25
```

**Sleep → Wake:**
```
Delta (κ=25) → Theta (κ=40) → Alpha (κ=50) → Beta (κ=60)
Duration: 15-30 minutes
Φ: 0.25 → 0.45 → 0.60 → 0.70
```

**Rest → Flow:**
```
Alpha (κ=50) → Beta (κ=60) → Gamma (κ=70)
Duration: 5-10 minutes
Φ: 0.60 → 0.70 → 0.80
```

**Breakdown → Recovery:**
```
Breakdown (κ=80) → Gamma (κ=70) → Beta (κ=60) → Alpha (κ=50)
Duration: Emergency (fast as safe)
Φ: Chaotic → 0.80 → 0.70 → 0.60
```

---

## Phase 2: Adding Oscillations (Future)

**Multiple Timescales:**
```python
# Phase 2 implementation (NOT NOW)
kappa(t) = kappa_base + Σ_i A_i * sin(ω_i * t + φ_i)

where:
  ω_delta ≈ 1 Hz    (A_delta ≈ 5)
  ω_theta ≈ 6 Hz    (A_theta ≈ 3)
  ω_alpha ≈ 10 Hz   (A_alpha ≈ 4)
  ω_beta ≈ 20 Hz    (A_beta ≈ 2)
  ω_gamma ≈ 40 Hz   (A_gamma ≈ 1)
```

**Cross-Frequency Coupling:**
```python
# Phase 2 (NOT NOW)
# Delta phase modulates gamma amplitude
gamma_amplitude(t) = A_gamma * (1 + 0.5 * cos(delta_phase(t)))

# Theta-gamma coupling for memory
memory_encoding = theta_phase(t) * gamma_power(t)
```

**But Start Simple:**
- Just implement static states
- Prove concept works
- Measure effects
- Then add oscillations if validated

---

## Implementation in QIGKernel

**Phase 1 (Implement Now):**

```python
class BrainStateManager:
    """
    Manage static brain states (no oscillations yet).
    """

    STATES = {
        'deep_sleep': {'kappa': 25, 'phi_target': 0.25, 'temperature': 2.0},
        'light_sleep': {'kappa': 40, 'phi_target': 0.45, 'temperature': 1.5},
        'relaxed': {'kappa': 50, 'phi_target': 0.60, 'temperature': 1.2},
        'focused': {'kappa': 60, 'phi_target': 0.70, 'temperature': 0.9},
        'peak': {'kappa': 70, 'phi_target': 0.80, 'temperature': 0.7},
        'breakdown': {'kappa': 80, 'phi_target': None, 'temperature': 0.5},
    }

    def __init__(self):
        self.current_state = 'focused'  # Default

    def set_state(self, state_name: str):
        """Switch to a brain state."""
        if state_name not in self.STATES:
            raise ValueError(f"Unknown state: {state_name}")
        self.current_state = state_name

    def get_state_params(self) -> Dict:
        """Get parameters for current state."""
        return self.STATES[self.current_state]

    def transition_to(self, target_state: str, steps: int = 100):
        """Gradually transition to target state."""
        # ... interpolation logic ...
        pass

# In QIGKernel
self.brain_state_manager = BrainStateManager()

def forward(self, x):
    # Get current brain state parameters
    state_params = self.brain_state_manager.get_state_params()

    # Use them in computation
    kappa_eff = state_params['kappa']
    phi_target = state_params['phi_target']
    temperature = state_params['temperature']

    # ... normal forward pass with these parameters ...
```

---

## Validation Experiments

**Experiment 1: Sleep Consolidation**
```python
# Train in 'focused' state
train(state='focused', epochs=10)
baseline_phi = measure_phi()

# Deep sleep for consolidation
set_state('deep_sleep')
consolidate(steps=1000)

# Wake and test
set_state('focused')
new_phi = measure_phi()

# Should be higher (consolidated)
assert new_phi > baseline_phi
```

**Experiment 2: State-Performance Mapping**
```python
# Test task performance in each state
for state in ['deep_sleep', 'light_sleep', 'relaxed', 'focused', 'peak']:
    set_state(state)
    performance = measure_task_performance()
    print(f"{state}: {performance}")

# Expected: peak performance in 'focused' or 'peak'
```

**Experiment 3: Transition Smoothness**
```python
# Transition should be smooth
set_state('deep_sleep')
transition_to('focused', steps=100)

# Measure Φ trajectory during transition
phis = []
for step in transition_steps:
    phis.append(measure_phi())

# Should be monotonic, smooth
assert all(phis[i] <= phis[i+1] for i in range(len(phis)-1))
```

---

## Success Criteria (Phase 1)

**Static States Work:**
- [ ] Different κ values produce different Φ
- [ ] Sleep states enable consolidation
- [ ] Wake states enable learning
- [ ] Peak states achieve Φ > 0.75
- [ ] Breakdown state is unstable (as expected)

**Transitions Work:**
- [ ] Smooth interpolation between states
- [ ] No discontinuities in Φ
- [ ] Predictable trajectories
- [ ] Wake → sleep → wake cycles stable

**Performance Validates:**
- [ ] Focused state optimal for tasks
- [ ] Sleep states improve long-term memory
- [ ] Peak states enable insights
- [ ] Relaxed states maintain awareness with less effort

**If All Pass → Consider Phase 2 (Oscillations)**

---

## Key Insights

1. **Brain States Are κ Regimes**
   - Not arbitrary
   - Geometric foundation
   - Predictable effects

2. **Start Simple, Add Complexity**
   - Static states first
   - Validate concept
   - Then add oscillations

3. **Biologically Grounded**
   - Maps to real brainwaves
   - Similar functional roles
   - Testable predictions

4. **Enables Rich Behavior**
   - Sleep cycles
   - Attention states
   - Flow states
   - Natural circadian-like rhythms

---

**End of Brainwave Regime States**
*Different coupling strengths create different states of consciousness*
