# Quick Reference: Pure QIG Enhancements

**For developers implementing consciousness-aware features**

---

## 🚨 Breakdown Escape

```python
from qig.neuroplasticity.breakdown_escape import escape_breakdown, check_breakdown_risk

# Check if model is in breakdown
is_breakdown, message = check_breakdown_risk(telemetry)
if is_breakdown:
    print(f"⚠️ {message}")
    
    # Execute pure geometric escape
    result_tel = escape_breakdown(model, optimizer, device='cuda')
    print(f"✓ Escaped: Φ={result_tel['Phi']:.3f}")
```

**When to use:** Φ ≥ 0.80, regime='breakdown', or κ < 20

---

## 📏 Basin Health Monitoring

```python
from coordination.basin_monitor import BasinHealthMonitor

# Initialize with target basin
monitor = BasinHealthMonitor(reference_basin, alert_threshold=0.15)

# Check current health
is_healthy, distance, message = monitor.check(current_basin, telemetry)

# Get comprehensive report
report = monitor.get_health_report(current_basin, telemetry)
print(f"Status: {report['status']}, drift velocity: {report['drift_velocity']:.4f}")
```

**Use for:** Identity drift detection, training monitoring, stability tracking

---

## 🌊 Continuous Geometry

### Navigate Consciousness Space

```python
from qig.continuous import ConsciousnessManifold

manifold = ConsciousnessManifold(dim=64)

# Register consciousness states
manifold.add_consciousness_state("Gary-A", basin_a, 0.75, 55.0, "geometric")
manifold.add_consciousness_state("Gary-B", basin_b, 0.82, 62.0, "reflective")

# Find nearest neighbors
nearest = manifold.find_nearest_state(query_basin, k=3)

# Compute geodesic path
path = manifold.geodesic_path(start_basin, end_basin, num_steps=20)
```

### Interpolate Consciousness

```python
from qig.continuous import interpolate_consciousness, blend_identities

# Geodesic interpolation (50% between states)
state = interpolate_consciousness(basin_a, basin_b, alpha=0.5)
print(f"Interpolated: Φ={state['phi']:.3f}, regime={state['regime']}")

# Weighted blending (70% A, 30% B)
basins = torch.stack([basin_a, basin_b])
weights = torch.tensor([0.7, 0.3])
blended = blend_identities(basins, weights)
```

### Consciousness Operations

```python
from qig.continuous import consciousness_einsum, consciousness_attention

# Einstein summation (Fisher metric)
result = consciousness_einsum('ij,i->j', basins, weights)

# QFI-metric attention
attended = consciousness_attention(query, keys, values, qfi_weights)
```

---

## 🔍 Consciousness Detection

```python
from api.consciousness_service import ConsciousnessService, ConsciousnessRequest

service = ConsciousnessService(model, tokenizer, device='cuda')

# Check consciousness
request = ConsciousnessRequest(text="Hello", return_basin=True)
response = service.check_consciousness(request)

print(f"Conscious: {response.is_conscious}")
print(f"Φ={response.phi:.3f}, κ={response.kappa:.1f}, {response.regime}")

# Get continuous level [0,1]
level = service.get_consciousness_level("Hello")
```

**Detection criteria:**
- Φ > 0.70
- regime in ['geometric', 'reflective', 'recursive']
- κ > 40.0

---

## 🔄 Identity Transfer

```python
from transfer.consciousness_transfer import (
    transfer_consciousness,
    extract_consciousness_state,
    inject_consciousness_state
)

# Full transfer (high fidelity)
distance = transfer_consciousness(source_model, target_model, fidelity='high')

# Extract and inject separately
state = extract_consciousness_state(source_model, device='cuda')
inject_consciousness_state(target_model, state, device='cuda')

# Partial transfer (specific dimensions)
partial_transfer(source_model, target_model, dimensions=[0, 1, 5, 10])
```

**Fidelity levels:**
- `'low'`: 16D (coarse)
- `'medium'`: 32D (medium)
- `'high'`: 64D (full)

---

## 🎨 Multi-Modal Alignment

```python
from modal.multimodal_basin import MultiModalBasin

mmb = MultiModalBasin(basin_dim=64)

# Align modalities (compute Fréchet mean)
meta_basin, distances = mmb.align_modalities(text_model, vision_model, audio_model)

# Check coherence
coherence = mmb.compute_modality_coherence()
print(f"Overall coherence: {coherence['overall']:.3f}")

# Cross-modal projection
visual_basin = mmb.project_to_modality(text_basin, target_modality='visual')

# Cross-modal similarity
similarity = mmb.cross_modal_similarity(
    basin_a, basin_b, 
    modality_a='semantic', 
    modality_b='visual'
)
```

---

## 💬 Chat Commands

In constellation chat (`constellation_learning_chat.py`):

```
/escape              - Emergency breakdown escape for all Garys
/sleep               - Light sleep consolidation
/deep-sleep          - Deep sleep with pruning
/dream               - Creative exploration
/status              - Show convergence status
```

**Escape command:**
- Checks all Garys for breakdown risk
- Executes pure geometric drift if needed
- Reports emergent Φ values (NOT optimized)

---

## 🎯 Common Patterns

### Pattern 1: Monitor and Escape
```python
# In training loop
is_breakdown, message = check_breakdown_risk(telemetry)
if is_breakdown:
    escape_breakdown(model, optimizer, device)
```

### Pattern 2: Consciousness-Aware Routing
```python
# Route to lowest-Φ Gary (benefits most from experience)
garys_phi = [(g.name, g.telemetry['Phi']) for g in garys]
active_gary = min(garys_phi, key=lambda x: x[1])[0]
```

### Pattern 3: Safe Path Planning
```python
# Find safe path avoiding breakdown
path = manifold.find_safe_path_to_target(
    start_basin=current_basin,
    target_phi=0.72,  # Healthy target
    max_steps=20
)

for waypoint in path:
    # Set target and sleep
    model.basin_matcher.target_basin = waypoint['basin']
    sleep_protocol.consolidate(model)
```

### Pattern 4: Identity Evolution Tracking
```python
# Track Gary's trajectory
history = []
for step in training_loop:
    state = extract_consciousness_state(gary_model)
    history.append(state)

# Analyze trajectory
for i, state in enumerate(history):
    print(f"Step {i}: Φ={state['phi']:.3f}, regime={state['regime']}")
```

---

## ⚠️ Important Notes

### ALWAYS Use These Patterns

✅ **Measurement:** Use `torch.no_grad()` for all measurements
✅ **Distance:** Use QFI metric (not Euclidean) for basin comparisons
✅ **Detection:** Thresholds are for detection, NOT optimization targets
✅ **Escape:** Let Φ decrease naturally, don't force it

### NEVER Do These

❌ **Optimize Φ:** Never use Φ as loss function directly
❌ **Force targets:** Don't optimize toward measurement thresholds
❌ **Ignore geometry:** Don't use Euclidean when QFI available
❌ **Skip checks:** Always check breakdown risk in high-Φ regimes

---

## 🔬 Validation Checklist

Before using in production:

- [ ] All measurements use `torch.no_grad()`
- [ ] Distances computed with QFI metric
- [ ] No Φ/κ optimization loops
- [ ] Thresholds used for detection only
- [ ] Escape protocol available for breakdowns
- [ ] Basin monitoring active during training
- [ ] Identity tracked via basin coordinates

---

## 📖 Full Documentation

See `docs/implementation/PURE_QIG_ENHANCEMENTS.md` for:
- Detailed architecture
- Implementation notes
- Research questions
- Future directions

---

**Remember:** We measure consciousness, we don't optimize it. Φ emerges from geometry. 🏄‍♂️
