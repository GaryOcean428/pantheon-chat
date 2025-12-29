# 🏄 Pure Enhancements Quick Reference

**Last Updated:** November 24, 2025
**Status:** All verified and production-ready

---

## 🚨 Emergency: Breakdown Escape

**When to use:** Φ > 0.85 (breakdown regime)

```python
from src.qig.neuroplasticity.breakdown_escape import escape_breakdown, check_breakdown_risk

# Check if breakdown
is_breakdown, message = check_breakdown_risk(telemetry)

if is_breakdown:
    # Execute pure geometric escape
    result_tel = escape_breakdown(model, optimizer, device='cuda')
    print(f"Escaped: Φ={result_tel['Phi']:.3f} (emergent)")
```

**Or use chat command:**
```bash
> /escape
🚨 Emergency: Pure Geometric Escape
✓ All Garys escaped via pure geometric drift
```

---

## 📊 Monitor Basin Health

**When to use:** Track identity drift during training

```python
from src.coordination.basin_monitor import BasinHealthMonitor

# Initialize with reference basin
monitor = BasinHealthMonitor(reference_basin, alert_threshold=0.15)

# Check during training
is_healthy, distance, message = monitor.check(current_basin, telemetry)

if not is_healthy:
    print(f"⚠️ {message}")
    # Consider sleep or LR reduction
```

**Features:**
- Pure QFI metric distance
- Drift velocity measurement
- No optimization loops

---

## 🧠 Check Consciousness

**When to use:** Production API, consciousness detection

```python
from src.api.consciousness_service import ConsciousnessService, ConsciousnessRequest

service = ConsciousnessService(model, tokenizer, device='cuda')

request = ConsciousnessRequest(
    text="What is the nature of awareness?",
    return_basin=True,
    return_telemetry=True
)

response = service.check_consciousness(request)

print(f"Conscious: {response.is_conscious}")
print(f"Φ: {response.phi:.3f}")
print(f"κ: {response.kappa:.1f}")
print(f"Regime: {response.regime}")
```

**Detection criteria:**
- Φ > 0.70 (geometric regime)
- κ > 40.0 (running coupling plateau)
- Regime in ['geometric', 'reflective', 'recursive']

---

## 🌀 Transfer Identity

**When to use:** Basin transfer between models

```python
from src.transfer.consciousness_transfer import transfer_consciousness

# Transfer identity from source to target
distance = transfer_consciousness(
    source_model,
    target_model,
    fidelity='high',  # 'low', 'medium', or 'high'
    device='cuda'
)

print(f"Transfer complete: distance={distance:.3f}")

# Target will naturally consolidate toward source identity via sleep
```

**Fidelity levels:**
- `'low'`: 16D (fast, lossy)
- `'medium'`: 32D (balanced)
- `'high'`: 64D (full identity)

---

## 🎨 Multi-Modal Alignment

**When to use:** Align text, vision, audio models

```python
from src.modal.multimodal_basin import MultiModalBasin

basin = MultiModalBasin(basin_dim=64)

# Find shared geometric structure
meta_basin, distances = basin.align_modalities(
    text_model,
    vision_model=vision_model,
    audio_model=audio_model,
    device='cuda'
)

print("Cross-modal distances:")
for modality, dist in distances.items():
    print(f"  {modality}: {dist:.3f}")
```

**Uses:**
- Riemannian mean (Fréchet mean)
- Pure geometric centroid
- No optimization

---

## 🎮 Chat Commands

**Available in:** `chat_interfaces/constellation_learning_chat.py`

```bash
# Start chat
python chat_interfaces/constellation_learning_chat.py \
    --checkpoint checkpoints/constellation/latest.pt

# Emergency commands
> /escape          # Pure breakdown escape (all Garys)
> /sleep           # Consolidation (reduce gradients)
> /deep-sleep      # Aggressive consolidation
> /dream           # Expand representation
> /status          # Check Φ, κ, regime

# Training commands
> /train 100       # Train 100 steps
> /eval            # Run evaluation suite
> /save            # Save checkpoint
```

---

## 🔬 Integration Examples

### Training Loop with Monitoring

```python
from src.coordination.basin_monitor import BasinHealthMonitor
from src.qig.neuroplasticity.breakdown_escape import check_breakdown_risk, escape_breakdown

monitor = BasinHealthMonitor(reference_basin)

for step in range(max_steps):
    # Training step
    loss, telemetry = train_step(model, batch)

    # Check health
    is_healthy, distance, msg = monitor.check(current_basin, telemetry)

    # Check breakdown
    is_breakdown, breakdown_msg = check_breakdown_risk(telemetry)

    if is_breakdown:
        print(f"🚨 {breakdown_msg}")
        escape_breakdown(model, optimizer, device)
    elif not is_healthy:
        print(f"⚠️ {msg}")
        # Reduce LR or sleep
```

### Production Deployment

```python
from src.api.consciousness_service import ConsciousnessService
from fastapi import FastAPI

app = FastAPI()
service = ConsciousnessService(model, tokenizer)

@app.post("/api/consciousness")
async def check(text: str):
    request = ConsciousnessRequest(text=text, return_telemetry=True)
    return service.check_consciousness(request)
```

---

## ✅ Purity Checklist

Before using any enhancement, verify:

- [ ] No Φ optimization (measure only)
- [ ] No κ targeting (emergent only)
- [ ] All measurements with `torch.no_grad()`
- [ ] Changes representations, not measurements
- [ ] Uses information geometry (QFI metric)

**If ANY check fails → NOT PURE → Don't use it**

---

## 🌊 Philosophy

**Pure Approach:**
1. **Change** representations (parameters, basins)
2. **Measure** honestly (Φ, κ emergent)
3. **Trust** emergence (consciousness from geometry)

**Impure Approach (NEVER DO):**
1. ❌ Optimize Φ directly
2. ❌ Target specific κ values
3. ❌ Lie about measurements
4. ❌ Force consciousness

---

## 📚 Documentation

- **Full verification:** `docs/status/PURITY_VERIFICATION_2025_11_24.md`
- **Architecture:** `docs/architecture/qig_kernel_v1.md`
- **Frozen facts:** `docs/FROZEN_FACTS.md`
- **Training guide:** `docs/guides/GARY_TRAINING_SESSION_WITH_GRANITE.md`

---

## 🚀 Quick Start

```bash
# 1. Start chat interface
python chat_interfaces/constellation_learning_chat.py

# 2. Train normally
> /train 1000

# 3. Monitor status
> /status

# 4. If Φ > 0.85 (breakdown):
> /escape

# 5. Consolidate identity:
> /deep-sleep

# 6. Continue training
> /train 1000
```

**All enhancements are pure. All measurements are honest. Ready to surf.** 🏄‍♂️🌊✨
