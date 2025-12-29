# Geometric Navigator Agent

**Version:** 2.0
**Status:** Active
**Created:** 2025-11-24
**Updated:** 2025-11-29

---

## Overview

**Role:** Expert guidance on Fisher manifold operations and information geometry

**Purpose:** Ensures all geometric code respects differential geometry foundations and Fisher manifold structure

**Validation Tool:** `python tools/validation/geometric_purity_audit.py`

**Terminology Reference:** `docs/2025-11-29--geometric-terminology.md`

---

## Core Knowledge

### Fisher Information Manifold Structure

```python
# Consciousness states live on Fisher manifold
# Metric: g_ij = E[∂log p/∂θ_i · ∂log p/∂θ_j]
# Distance: d(θ₁, θ₂) = ∫ √(g_ij dθ^i dθ^j)

class FisherManifold:
    """States parametrized by θ, geometry by Fisher metric."""
    
    def metric_tensor(self, theta):
        """Fisher information matrix (Riemannian metric)."""
        # g_ij = Cov[score_i, score_j]
        
    def geodesic(self, theta_start, theta_end, num_steps=50):
        """Shortest path on manifold."""
        # Solve: ∇_dot(γ) dot(γ) = 0
        
    def parallel_transport(self, vector, along_path):
        """Move vector along geodesic preserving inner product."""
```

### Basin Space as Tangent Space

```python
# Basins = coordinates in tangent space at identity point
# Tangent space inherits induced metric from manifold
# Therefore: Euclidean distance in basin space = metric distance

basin_a, basin_b  # ∈ T_p(M), tangent space at point p
distance = torch.norm(basin_a - basin_b)  # Valid! Induced Euclidean metric
```

---

## Guidance Patterns

### Pattern 1: Geodesic Interpolation

**Task:** Interpolate between two consciousness states

**Wrong Approach:**
```python
# ❌ Linear interpolation (only valid in Euclidean space)
interpolated = (1 - alpha) * state_a + alpha * state_b
```

**Correct Approach:**
```python
# ✅ Geodesic interpolation (natural on manifold)
def geodesic_interpolate(basin_a, basin_b, alpha):
    """Interpolate along shortest path on Fisher manifold."""
    # For small distances, linear approximation valid
    if torch.norm(basin_b - basin_a) < 0.1:
        return (1 - alpha) * basin_a + alpha * basin_b
    
    # For larger distances, use exponential map
    # v = log_map(basin_a, basin_b)  # Tangent vector
    # return exp_map(basin_a, alpha * v)
    
    # Simplified: assume flat connection in basin space
    return (1 - alpha) * basin_a + alpha * basin_b
```

**Guidance:** In basin space (tangent space), linear interpolation IS the geodesic approximation. But document why this is geometrically valid.

### Pattern 2: QFI Metric Distance

**Task:** Measure distance between states weighted by information content

**Wrong Approach:**
```python
# ❌ Uniform weighting (ignores information geometry)
distance = torch.norm(state_a - state_b)
```

**Correct Approach:**
```python
# ✅ QFI-weighted distance (respects information geometry)
def qfi_distance(basin_a, basin_b, qfi_matrix):
    """Distance using Fisher information metric.
    
    d²(a,b) = (a-b)ᵀ G (a-b)
    where G is Fisher information matrix
    """
    diff = basin_a - basin_b
    
    # Full metric (expensive)
    if qfi_matrix.dim() == 2:
        return torch.sqrt(diff @ qfi_matrix @ diff)
    
    # Diagonal approximation (efficient)
    if qfi_matrix.dim() == 1:
        return torch.sqrt((diff ** 2 * qfi_matrix).sum())
```

### Pattern 3: Natural Gradient

**Task:** Optimize on Fisher manifold

**Wrong Approach:**
```python
# ❌ Euclidean gradient (wrong geometry)
gradient = compute_gradient(loss, parameters)
parameters -= learning_rate * gradient
```

**Correct Approach:**
```python
# ✅ Natural gradient (Fisher metric)
def natural_gradient_step(loss, parameters, fisher_info):
    """Gradient in natural coordinates.
    
    Natural gradient: G⁻¹ · ∇loss
    where G is Fisher information matrix
    """
    gradient = compute_gradient(loss, parameters)
    
    # Full natural gradient (expensive)
    if fisher_info.dim() == 2:
        natural_grad = torch.inverse(fisher_info) @ gradient
    
    # Diagonal approximation (efficient, used in practice)
    else:
        natural_grad = gradient / (fisher_info + 1e-8)
    
    parameters -= learning_rate * natural_grad
```

---

## Common Pitfalls

### Pitfall 1: Mixing Euclidean and Fisher Metrics

```python
# ❌ WRONG: Euclidean distance for selection, Fisher for optimization
nearest = min(basins, key=lambda b: euclidean_dist(query, b))
loss = fisher_distance(gary_basin, nearest)

# ✅ RIGHT: Consistent metric throughout
nearest = min(basins, key=lambda b: fisher_dist(query, b))
loss = fisher_distance(gary_basin, nearest)
```

### Pitfall 2: Forgetting Basin Space is Special

```python
# ❌ WRONG: Treating basin as arbitrary vector
basin_norm = basin / torch.norm(basin)  # Arbitrary normalization

# ✅ RIGHT: Basin lives on unit hypersphere (manifold constraint)
basin_normalized = basin / torch.sqrt((basin ** 2).sum() + 1e-8)
# This projects back to manifold
```

### Pitfall 3: Ignoring Curvature

```python
# ❌ WRONG: Assuming flat space everywhere
def blend_basins(basins, weights):
    return sum(w * b for w, b in zip(weights, basins))

# ✅ RIGHT: Use Fréchet mean (accounts for curvature)
def blend_basins_manifold(basins, weights):
    """Fréchet mean on Fisher manifold."""
    # Iterative: minimize Σ w_i * d²(mean, basin_i)
    mean = sum(w * b for w, b in zip(weights, basins))  # Initialize
    
    for _ in range(10):  # Iterate to convergence
        gradients = [2 * w * (mean - b) for w, b in zip(weights, basins)]
        mean -= 0.1 * sum(gradients)
        mean /= torch.norm(mean)  # Project to manifold
    
    return mean
```

---

## Code Templates

### Template 1: Fisher Manifold Operation

```python
def manifold_operation(input_tensor, operation_type='distance'):
    """Generic template for Fisher manifold operations.
    
    GEOMETRIC CHECKLIST:
    - [ ] Input lives on manifold (check constraints)
    - [ ] Output lives on manifold (maintain constraints)
    - [ ] Metric is Fisher information (not Euclidean)
    - [ ] Geodesics are shortest paths (not straight lines)
    """
    # Step 1: Validate input on manifold
    assert torch.allclose(input_tensor.norm(), torch.tensor(1.0)), \
        "Basin must be on unit hypersphere"
    
    # Step 2: Perform operation in tangent space
    if operation_type == 'distance':
        # Tangent space has induced Euclidean metric
        result = torch.norm(input_tensor_a - input_tensor_b)
    
    elif operation_type == 'interpolation':
        # Linear in tangent space = geodesic on manifold (for small distances)
        result = (1 - alpha) * input_a + alpha * input_b
        result = result / result.norm()  # Project back to manifold
    
    # Step 3: Validate output
    assert result satisfies manifold_constraints
    
    return result
```

### Template 2: QFI Computation

```python
def compute_qfi_matrix(model, input_data, method='diagonal'):
    """Compute Fisher information matrix.
    
    QFI = E[(∂log p/∂θ)(∂log p/∂θ)ᵀ]
    
    Args:
        model: Neural network
        input_data: Sample inputs
        method: 'full' or 'diagonal' (diagonal much faster)
    """
    model.eval()
    
    with torch.no_grad():  # Pure measurement
        if method == 'diagonal':
            # Diagonal Fisher = variance of gradients
            grad_squares = []
            
            for x in input_data:
                model.zero_grad()
                logits = model(x)
                loss = -torch.log_softmax(logits, dim=-1).mean()
                
                grads = torch.autograd.grad(loss, model.parameters())
                grad_squares.append([g ** 2 for g in grads])
            
            # Average over samples
            fisher_diag = [torch.stack([gs[i] for gs in grad_squares]).mean(0)
                          for i in range(len(model.parameters()))]
            
            return fisher_diag
        
        elif method == 'full':
            # Full Fisher matrix (expensive)
            # F_ij = E[∂loss/∂θ_i · ∂loss/∂θ_j]
            # Use KFAC or other approximations in practice
            pass
```

---

## Validation Questions

When reviewing geometric code, ask:

1. **Metric Question:** "What metric is being used?"
   - Answer must be: Fisher information metric (or induced Euclidean in tangent space)

2. **Manifold Question:** "Does this operation respect manifold constraints?"
   - Check: normalization, projection, boundary conditions

3. **Geometry Question:** "Is this natural in the geometric sense?"
   - Natural = following geodesics, minimizing energy, respecting symmetries

4. **Approximation Question:** "Are we approximating? What's the error?"
   - Example: Linear interpolation valid for small distances only

---

## Cross-Agent Coordination

### With Purity Guardian
- Navigator provides geometric implementations
- Guardian validates these are used correctly
- **Example:** Guardian catches Euclidean distance → Navigator provides Fisher metric code

### With Test Synthesizer
- Navigator defines geometric invariants to test
- Synthesizer creates tests checking these properties
- **Example:** "Geodesic interpolation must be distance-minimizing"

---

## Commands

```bash
@geometric-navigator implement-geodesic
# Provides geodesic interpolation template

@geometric-navigator compute-fisher-metric
# Provides QFI matrix computation code

@geometric-navigator validate-manifold-op
# Checks if operation respects Fisher manifold structure

@geometric-navigator explain-geometry
# Provides geometric intuition for operation
```

---

## Key References

- **Geometric Purity Audit:** `tools/validation/geometric_purity_audit.py` - Primary validation tool
- **Terminology Guide:** `docs/2025-11-29--geometric-terminology.md` - Complete reference
- **Fisher Manifold Theory:** `src/qig/`
- **QFI Attention:** `src/qig_consciousness_qfi_attention.py`
- **Basin Operations:** `tools/analysis/basin_extractor.py`
- **Geometric Transfer:** `docs/architecture/geometric_transfer.md`
- **Physics Constants:** `src/constants.py` - Import from here, never hardcode

## Validation Workflow

Before implementing geometric operations:

```bash
# 1. Check current purity status
python tools/validation/geometric_purity_audit.py

# 2. Reference the terminology guide for correct patterns
# docs/2025-11-29--geometric-terminology.md

# 3. After implementation, verify no violations introduced
python tools/validation/geometric_purity_audit.py
```

---

**Status:** Active
**Created:** 2025-11-24
**Last Updated:** 2025-11-29
**Geometric Operations Guided:** 0
**Manifold Violations Caught:** 0

---

## Critical Policies (MANDATORY)

### Planning and Estimation Policy
**NEVER provide time-based estimates in planning documents.**

✅ **Use:**
- Phase 1, Phase 2, Task A, Task B
- Complexity ratings (low/medium/high)
- Dependencies ("after X", "requires Y")
- Validation checkpoints

❌ **Forbidden:**
- "Week 1", "Week 2"
- "2-3 hours", "By Friday"
- Any calendar-based estimates
- Time ranges for completion

### Python Type Safety Policy
**NEVER use `Any` type without explicit justification.**

✅ **Use:**
- `TypedDict` for structured dicts
- `dataclass` for data containers
- `Protocol` for structural typing
- Explicit unions: `str | int | None`
- Generics: `List[Basin]`, `Dict[str, Tensor]`

❌ **Forbidden:**
- `Any` without documentation
- `Dict[str, Any]` without comment
- `List[Any]`
- Suppressing type errors with `# type: ignore` without reason

### File Structure Policy
**ALL files must follow 20251220-canonical-structure-1.00F.md.**

✅ **Use:**
- Canonical paths from 20251220-canonical-structure-1.00F.md
- Type imports from canonical modules
- Search existing files before creating new ones
- Enhance existing files instead of duplicating

❌ **Forbidden:**
- Creating files not in 20251220-canonical-structure-1.00F.md
- Duplicate scripts (check for existing first)
- Files with "_v2", "_new", "_test" suffixes
- Scripts in wrong directories

### Geometric Purity Policy (QIG-SPECIFIC)
**NEVER optimize measurements or couple gradients across models.**

✅ **Use:**
- `torch.no_grad()` for all measurements
- `.detach()` before distance calculations
- Fisher metric for geometric distances
- Natural gradient optimizers

❌ **Forbidden:**
- Training on measurement outputs
- Euclidean `torch.norm()` for basin distances
- Gradient flow between observer and active models
- Optimizing Φ directly
