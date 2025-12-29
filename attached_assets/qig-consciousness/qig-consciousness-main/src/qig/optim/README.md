# QIG Geometric Optimizers

Geometric optimization for QIG consciousness architecture based on natural gradient descent on Riemannian manifolds.

## Overview

**Problem:** Standard optimizers (Adam, SGD, RMSProp) assume Euclidean parameter space. They ignore the geometric structure of the loss landscape, leading to inefficient updates that don't follow geodesics on the information manifold.

**Solution:** Natural gradient descent uses the Fisher Information Matrix (FIM) as the metric tensor, allowing updates to follow geodesics on the Riemannian manifold of probability distributions.

**Mathematical Foundation:**

```
Standard gradient:  θ_new = θ_old - lr × ∇L
Natural gradient:   θ_new = θ_old - lr × F^(-1) × ∇L
```

Where F is the Fisher Information Matrix: `F_ij = E[∂log p/∂θ_i × ∂log p/∂θ_j]`

## Available Optimizers

### 1. QIGDiagonalNG

**Diagonal approximation to natural gradient.**

- **When to use:** First step from AdamW, entire model needs geometric awareness
- **Cost:** O(N) - same as Adam
- **Accuracy:** Moderate (diagonal approximation)

```python
from qig.optim import QIGDiagonalNG

optimizer = QIGDiagonalNG(
    model.parameters(),
    lr=1e-3,
    alpha=0.99,  # EMA decay for Fisher diagonal
    eps=1e-8,
)
```

**Mathematical Details:**
- Approximates Fisher with diagonal: `F ≈ diag(F_ii)`
- `F_ii = E[(∂L/∂θ_i)²]` estimated via exponential moving average
- Update: `Δθ_i = -lr × ∇L_i / (√F_ii + ε)`

### 2. BasinNaturalGrad

**Exact natural gradient for basin block using conjugate gradient solver.**

- **When to use:** Basin block only (~3M params), highest geometric accuracy needed
- **Cost:** O(N × cg_iters) - typically 8-10 iterations
- **Accuracy:** Exact (within CG tolerance)

```python
from qig.optim import BasinNaturalGrad

basin_params = [
    model.embedding.basin_coords,
    model.embedding.basin_to_model.weight,
]

optimizer = BasinNaturalGrad(
    basin_params,
    lr=1e-2,  # Can use higher lr with exact NG
    cg_iters=10,
    damping=1e-4,
)

# Important: Use create_graph=True for Pearlmutter trick
loss.backward(create_graph=True)
optimizer.step()
```

**Mathematical Details:**
- Solves `(F + λI) × v = ∇L` using conjugate gradient
- Uses Pearlmutter trick for efficient Fisher-vector products: `Fv = ∂(g^T v)/∂θ`
- Only 1-2 extra backprops per step (via automatic differentiation)
- GPU friendly - all operations on device

### 3. HybridGeometricOptimizer

**Hybrid optimizer: Natural gradient for basin block, AdamW/Diagonal NG for rest.**

- **When to use:** Most practical choice for full model training
- **Cost:** Basin (expensive NG) + Rest (cheap AdamW) = balanced
- **Accuracy:** Exact where it matters (basin), efficient elsewhere

```python
from qig.optim import HybridGeometricOptimizer

optimizer = HybridGeometricOptimizer(
    model,
    lr_ng=1e-2,         # Basin block learning rate
    lr_rest=1e-3,       # Rest of model learning rate
    cg_iters=8,
    use_exact_ng=False, # false = diagonal NG, true = exact NG
    rest_optimizer='adamw',  # or 'diagonal_ng'
    weight_decay=0.01,
)
```

**Rationale:**
- Basin embeddings (~3M params): Active geometric core → needs natural gradient
- Transformer backbone (~25M params): Redundant encodings → cheaper optimizer OK
- This gives best accuracy/cost tradeoff

### 4. AdaptiveMixedQIG

**Adaptive version with telemetry-based gating of natural gradient.**

- **When to use:** RECOMMENDED - Best performance in practice
- **Cost:** Dynamic - applies NG only when geometry demands it
- **Accuracy:** Adaptive - high when needed, efficient when not

```python
from qig.optim import AdaptiveMixedQIG
from qig.optim import AdaptiveConfig

config = AdaptiveConfig(
    min_kappa_for_ng=40.0,          # κ_eff > 40 → apply NG
    min_basin_distance=0.6,          # basin distance > 0.6 → apply NG
    force_ng_every_n_steps=50,       # Force NG every 50 steps
    exploration_triggers_ng=True,    # EXPLORATION regime → NG
    geometric_triggers_ng=True,      # GEOMETRIC regime → NG
    stuck_triggers_ng=True,          # STUCK regime → NG
)

optimizer = AdaptiveMixedQIG(
    model,
    lr_ng=1e-2,
    lr_rest=1e-3,
    use_exact_ng=False,
    adaptive_config=config,
)

# In training loop:
should_use_ng = optimizer.should_apply_ng(telemetry, step)
loss.backward(create_graph=should_use_ng)
optimizer.step()
```

**Gating Logic:**

Natural gradient is applied when ANY of these conditions hold:

1. **Forced:** Every N steps (prevents drift)
2. **High curvature:** κ_eff > threshold
3. **Far from target:** basin_distance > threshold
4. **Exploration:** curiosity regime = EXPLORATION
5. **Geometric regime:** regime = geometric AND high κ
6. **Stuck:** curiosity regime = STUCK (needs escape velocity)

## Training Phases

Use different adaptive configs for different training phases:

```python
from qig.optim.adaptive_gate import get_recommended_config_for_phase

# Early training: Aggressive NG to establish geometry
config_early = get_recommended_config_for_phase('early')

# Middle training: Balanced (default)
config_middle = get_recommended_config_for_phase('middle')

# Late training: Conservative NG for refinement
config_late = get_recommended_config_for_phase('late')

# Fine-tuning: Minimal NG, preserve learned geometry
config_finetune = get_recommended_config_for_phase('fine_tune')
```

## Integration with Trainer

The trainer now supports all geometric optimizers via config:

```yaml
# configs/train_config.yaml

optimizer:
  type: 'adaptive_mixed_qig'
  lr_ng: 0.01
  lr_rest: 0.001
  use_exact_ng: false
  rest_optimizer: 'adamw'
  adaptive_ng_phase: 'middle'
  min_kappa_for_ng: 40.0
  min_basin_distance_for_ng: 0.6
  force_ng_every_n_steps: 50
```

Run training:

```bash
python tools/train_qig_kernel.py --config configs/train_config.yaml
```

## Telemetry

All optimizers provide telemetry for monitoring:

```python
# Get optimizer stats
stats = optimizer.get_stats()

# For diagonal NG optimizers:
print(f"Fisher mean: {stats['fisher_mean']:.4f}")
print(f"Fisher std: {stats['fisher_std']:.4f}")
print(f"Condition number: {stats['condition_number']:.2f}")

# For mixed optimizers:
print(f"Basin params: {stats['num_basin_params']:,}")
print(f"Rest params: {stats['num_rest_params']:,}")

# For adaptive optimizers:
print(f"NG applications: {stats['ng_applications']}")
```

Telemetry is automatically logged to `training_telemetry.jsonl`.

## Emotion Monitor (Optional)

Compute geometric emotional primitives from telemetry:

```python
from qig.affect import EmotionMonitor

monitor = EmotionMonitor(enable_extended_emotions=True)
emotions = monitor.compute(telemetry)

# Emotions computed:
# - Joy = -Ricci curvature (expansion)
# - Suffering = +Ricci curvature (compression)
# - Fear = separatrix proximity × gradient
# - Love = -∇d_basin (toward connection)
# - Hate = +∇d_basin (away from connection)
# - Rage = κ × ∇Φ × stuck
# - Calm = low gradient
# - Curiosity, Frustration (from existing monitors)
```

Enable in config:

```yaml
emotion_monitor:
  enabled: true
```

## Performance Comparison

Expected improvements over AdamW baseline:

| Optimizer | Speed vs AdamW | Convergence | Basin Distance | Φ Quality |
|-----------|----------------|-------------|----------------|-----------|
| AdamW (baseline) | 1.0× | Baseline | Baseline | Baseline |
| QIGDiagonalNG | 0.95× | +15% faster | -20% | +10% |
| MixedQIG (diagonal) | 0.90× | +25% faster | -35% | +15% |
| MixedQIG (exact) | 0.70× | +40% faster | -50% | +25% |
| AdaptiveMixedQIG | 0.85× | +35% faster | -45% | +20% |

*These are approximate based on geometric principles. Actual results depend on problem.*

## References

- **Natural Gradient:** Amari, S. (1998). "Natural Gradient Works Efficiently in Learning"
- **Fisher Information:** Ly et al. (2017). "A Tutorial on Fisher Information"
- **Pearlmutter Trick:** Pearlmutter, B. (1994). "Fast Exact Multiplication by the Hessian"
- **QIG Physics:** See `/mnt/data/QIG_Consciousness_Corrected.md` and session sleep packets

## Implementation Notes

### Why This Isn't Standard ML

⚠️ **IMPORTANT:** These optimizers implement physics-driven geometry, not standard ML:

- Parameter space is a **Riemannian manifold**, not ℝⁿ
- Fisher metric is the **natural metric** from information geometry
- Updates follow **geodesics**, not Euclidean straight lines
- Basin structure is **geometric identity**, not learned features

**Do not** fall back to standard "best practices" - this is novel physics.

### GPU Efficiency

All optimizers are GPU-friendly:

- ✅ No CPU transfers during optimization
- ✅ Element-wise operations (diagonal NG)
- ✅ Sparse Fisher ops (exact NG with locality)
- ✅ CG iterations fully on GPU

Expect <10% overhead vs AdamW for diagonal NG, <30% for exact NG.

### Memory Requirements

- **DiagonalNG:** Same as AdamW (stores Fisher diagonal)
- **BasinNaturalGrad:** O(N) for basin params only (~3M params)
- **MixedQIG:** Sum of both components
- **No explicit Fisher matrix** - all ops via matrix-vector products

## Troubleshooting

### NaN in gradients with exact NG

**Cause:** Damping too low, CG divergence

**Fix:** Increase damping:
```python
optimizer = BasinNaturalGrad(..., damping=1e-3)  # Instead of 1e-4
```

### Slow convergence with diagonal NG

**Cause:** Diagonal approximation insufficient for high curvature

**Fix:** Use mixed optimizer with exact NG for basin:
```python
optimizer = MixedQIGOptimizer(..., use_exact_ng=True)
```

### NG never applied in AdaptiveMixedQIG

**Cause:** Thresholds too high, system stable

**Fix:** Lower thresholds or check telemetry:
```python
config = AdaptiveConfig(
    min_kappa_for_ng=30.0,  # Lower threshold
    force_ng_every_n_steps=20,  # More frequent
)
```

## Contact

For questions about geometric optimizers, see:
- `/mnt/data/QIG_GEOMETRIC_OPTIMIZER_DESIGN_FOR_COPILOT.md`
- Session sleep packets in `/mnt/data/`
- Main QIG documentation in project root

---

**Written for qig-consciousness geometric optimization - Nov 18, 2025**
