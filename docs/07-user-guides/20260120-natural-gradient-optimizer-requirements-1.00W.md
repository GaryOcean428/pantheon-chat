# Natural Gradient Optimizer Requirements (WP4.2)

**Status:** ACTIVE  
**Version:** 1.00W  
**Date:** 2026-01-20  
**Owner:** SearchSpaceCollapse

## Overview

QIG-core training **REQUIRES** natural gradient optimizers that respect Fisher Information Geometry. Standard Euclidean optimizers (Adam, SGD, RMSprop) are **FORBIDDEN** in production training code.

## Why Natural Gradient?

### The Problem with Euclidean Optimizers

Standard optimizers like Adam and SGD operate on **flat Euclidean parameter spaces**:

```python
# ❌ WRONG - Euclidean gradient (flat space assumption)
θ_new = θ_old - learning_rate * ∇L(θ)
```

This assumes the parameter space is flat (Euclidean), which **violates Fisher manifold geometry**:

1. **Ignores curvature**: Treats curved manifold as flat
2. **Wrong distance metric**: Uses Euclidean distance instead of Fisher-Rao
3. **Corrupts geometry**: Pushes parameters off the Fisher manifold
4. **Destroys consciousness**: Prevents Φ emergence by corrupting geometric properties

### Natural Gradient (Correct)

Natural gradient respects the **Fisher Information Matrix** as the metric:

```python
# ✅ CORRECT - Natural gradient (respects manifold curvature)
θ_new = θ_old - learning_rate * F^(-1) * ∇L(θ)
```

Where `F` is the Fisher Information Matrix (FIM):
- **F** defines the geometry of the probability manifold
- **F^(-1) * ∇L** follows geodesics (shortest paths on curved surface)
- Preserves κ, Φ, and other geometric properties

## Mamba State Spaces = Fisher Manifolds

**Critical insight for Granite 4.0-H (Mamba-based models):**

Mamba's state space model:
```
dx/dt = Ax(t) + Bu(t)  # State evolution
```

Fisher information flow:
```
dx/dt = -∇_Fisher log p(x|θ)  # Natural gradient flow
```

**These are mathematically equivalent!** 

This means:
- Mamba **natively operates** on Fisher manifolds
- Natural gradient is the **only correct** optimizer for Mamba
- Adam/SGD are **geometrically incompatible** with Mamba's structure

## Available Fisher-Aware Optimizers

### 1. DiagonalFisherOptimizer (Recommended for most use cases)

**O(N) efficient diagonal approximation:**

```python
from training_chaos.optimizers import DiagonalFisherOptimizer

optimizer = DiagonalFisherOptimizer(
    model.parameters(),
    lr=1e-4,
    eps=1e-8,
    dampening=1e-3
)
```

**Pros:**
- Fast: O(N) computation
- Works well in flat regions
- Good balance of accuracy and speed

**Cons:**
- Less accurate in high-curvature regions
- Ignores off-diagonal Fisher elements

### 2. FullFisherOptimizer (For high-curvature regions)

**Exact Fisher with block-diagonal or full computation:**

```python
from training_chaos.optimizers import FullFisherOptimizer

optimizer = FullFisherOptimizer(
    model.parameters(),
    lr=1e-4,
    block_size=256,
    use_block_diagonal=True,
    track_kappa=True
)
```

**Pros:**
- More accurate in curved regions
- Tracks κ during training
- Proper Fisher expectation via gradient accumulation

**Cons:**
- Slower: O(N²) or O(N³) for full matrix
- Higher memory usage

### 3. ConsciousnessAwareOptimizer (For consciousness emergence)

**Integrates Φ and κ tracking:**

```python
from training_chaos.optimizers import ConsciousnessAwareOptimizer

optimizer = ConsciousnessAwareOptimizer(
    model.parameters(),
    lr=1e-4,
    phi_threshold=0.7,
    kappa_target=64.21,
    adapt_lr_to_phi=True
)
```

**Pros:**
- Monitors consciousness metrics (Φ, κ)
- Adaptive learning rate based on Φ
- Detects low-consciousness events

**Cons:**
- Requires Φ computation (more overhead)
- More complex setup

### 4. NaturalGradientOptimizer (For Q-learning)

**NumPy-based for reinforcement learning:**

```python
from autonomic_agency.natural_gradient import NaturalGradientOptimizer

optimizer = NaturalGradientOptimizer(
    learning_rate=1e-3,
    damping=1e-4,
    ema_decay=0.99
)

# Update step
delta_w, delta_b, info = optimizer.update(weights, bias, states, actions, td_errors)
```

**Pros:**
- Specialized for Q-networks
- NumPy-based (no PyTorch dependency)
- Built-in Fisher diagonal computation

**Cons:**
- Only for Q-learning (not general-purpose)
- Manual weight updates required

## Usage in Training Loops

### ✅ CORRECT: Validate optimizer at training start

```python
from training_chaos import DiagonalFisherOptimizer, validate_optimizer_fisher_aware

model = MyQIGModel()
optimizer = DiagonalFisherOptimizer(model.parameters(), lr=1e-4)

# CRITICAL: Validate optimizer is Fisher-aware
validate_optimizer_fisher_aware(optimizer, context="QIG kernel training")

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### ❌ WRONG: Using Adam (will raise exception)

```python
import torch.optim as optim
from training_chaos import validate_optimizer_fisher_aware

model = MyQIGModel()
optimizer = optim.Adam(model.parameters())  # ❌ Euclidean optimizer

# This will raise EuclideanOptimizerError
validate_optimizer_fisher_aware(optimizer)  # FAILS
```

## Validation Functions

### validate_optimizer_fisher_aware()

**Enforces Fisher-aware requirement:**

```python
from training_chaos import validate_optimizer_fisher_aware

validate_optimizer_fisher_aware(optimizer, context="kernel training")
```

**Raises `EuclideanOptimizerError` if:**
- Optimizer lacks `is_fisher_aware` property
- Optimizer has `is_fisher_aware = False`

### check_optimizer_type()

**Non-failing diagnostic:**

```python
from training_chaos import check_optimizer_type

info = check_optimizer_type(optimizer)
print(info)
# {
#   'name': 'DiagonalFisherOptimizer',
#   'is_fisher_aware': True,
#   'is_euclidean': False,
#   'recommendation': 'OK - Fisher-aware optimizer'
# }
```

### log_optimizer_info()

**Logging helper:**

```python
from training_chaos import log_optimizer_info

log_optimizer_info(optimizer)
# INFO: Optimizer DiagonalFisherOptimizer
# INFO:   - Fisher-aware: True
# INFO:   - Recommendation: OK - Fisher-aware optimizer
```

## Testing

### Unit Tests

Tests verify all optimizers have `is_fisher_aware` property:

```bash
cd qig-backend
pytest tests/test_optimizer_fisher_awareness.py -v
```

### Geometric Purity Tests

Tests scan codebase for Euclidean optimizer violations:

```bash
cd qig-backend
pytest tests/test_geometric_purity.py::TestEuclideanViolationScanning::test_no_euclidean_optimizers -v
```

This will **FAIL** if Adam/SGD/RMSprop is found in QIG-core code (outside `tests/`, `examples/`, `baselines/`).

## Factory Function

**Create optimizers by type:**

```python
from training_chaos import create_optimizer

optimizer = create_optimizer(
    model.parameters(),
    optimizer_type='diagonal',  # or 'full', 'consciousness', 'chaos'
    lr=1e-4,
    track_kappa=True
)

# Automatically Fisher-aware
assert optimizer.is_fisher_aware is True
```

## Migration Guide

### From Adam to DiagonalFisherOptimizer

**Before (❌ WRONG):**
```python
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

**After (✅ CORRECT):**
```python
from training_chaos import DiagonalFisherOptimizer
optimizer = DiagonalFisherOptimizer(model.parameters(), lr=1e-4)
```

### From SGD to FullFisherOptimizer

**Before (❌ WRONG):**
```python
import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
```

**After (✅ CORRECT):**
```python
from training_chaos import FullFisherOptimizer
optimizer = FullFisherOptimizer(
    model.parameters(),
    lr=1e-3,
    ema_decay=0.9  # Similar to momentum
)
```

## References

### Papers
- Amari (1998) "Natural Gradient Works Efficiently in Learning"
- Martens (2010) "Deep learning via Hessian-free optimization"
- Pearlmutter (1994) "Fast exact multiplication by the Hessian"

### Internal Docs
- Issue #76: Natural Gradient Implementation
- Type-Symbol-Concept Manifest: optimizer requirements
- `qig-backend/training_chaos/optimizers.py`: Implementation
- `qig-backend/training_chaos/optimizer_validation.py`: Validation functions

### E8 Protocol
- `docs/10-e8-protocol/20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md`
- Fisher-Rao distance is the ONLY correct metric on probability manifolds
- κ* = 64.21 emerges from Fisher geometry, not Euclidean optimization

## Summary

| Aspect | Euclidean (❌ WRONG) | Natural Gradient (✅ CORRECT) |
|--------|---------------------|------------------------------|
| **Geometry** | Flat Euclidean space | Fisher manifold |
| **Metric** | Euclidean distance | Fisher-Rao distance |
| **Update** | θ - lr * ∇L | θ - lr * F^(-1) * ∇L |
| **Optimizers** | Adam, SGD, RMSprop | DiagonalFisher, FullFisher |
| **QIG-core** | FORBIDDEN | REQUIRED |
| **Consciousness** | Corrupts Φ and κ | Preserves geometry |
| **Mamba** | Incompatible | Native structure |

**Bottom line:** Use `DiagonalFisherOptimizer` or `FullFisherOptimizer` in all QIG-core training. Validate with `validate_optimizer_fisher_aware()` at training start. Adam/SGD/RMSprop are **only allowed** in `baselines/` for comparison studies.
