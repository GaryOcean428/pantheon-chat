# Geometric Operations Skill

**Type:** Reusable Component
**Category:** Mathematical Operations
**Used By:** purity-guardian, geometric-navigator, integration-architect

---

## Purpose

Provides templates and validation patterns for Fisher manifold operations, ensuring geometric purity across all QIG implementations.

---

## Core Operations

### 1. Fisher Metric Distance

**Template:**
```python
def fisher_distance(point_a: torch.Tensor, point_b: torch.Tensor,
                    fisher_diag: torch.Tensor) -> torch.Tensor:
    """
    Compute distance on Fisher manifold.

    Args:
        point_a: First point on manifold (basin embedding)
        point_b: Second point on manifold (target basin)
        fisher_diag: Diagonal Fisher Information Matrix

    Returns:
        Geometric distance (scalar)
    """
    diff = point_a - point_b
    # Fisher metric: d²(a,b) = (a-b)ᵀ F (a-b)
    distance_sq = torch.sum(diff * fisher_diag * diff)
    return torch.sqrt(distance_sq + 1e-8)  # Numerical stability
```

**Validation:**
- ✅ Uses Fisher metric (not Euclidean)
- ✅ Includes numerical stability term
- ✅ Returns scalar (not vector)

### 2. Geodesic Interpolation

**Template:**
```python
def geodesic_interpolate(start: torch.Tensor, end: torch.Tensor,
                         alpha: float, fisher_diag: torch.Tensor) -> torch.Tensor:
    """
    Interpolate along geodesic on Fisher manifold.

    Args:
        start: Starting point
        end: Ending point
        alpha: Interpolation parameter [0, 1]
        fisher_diag: Diagonal FIM

    Returns:
        Point on geodesic at alpha
    """
    # For diagonal metric, geodesic is straight line in transformed space
    # x(t) = x₀ + t·F⁻¹·(x₁ - x₀)
    diff = end - start
    velocity = diff / fisher_diag  # Transform by inverse metric
    return start + alpha * velocity
```

**Validation:**
- ✅ Uses inverse Fisher metric
- ✅ Parameter α ∈ [0, 1]
- ✅ Returns point on manifold

### 3. Basin Projection

**Template:**
```python
def project_to_basin(embedding: torch.Tensor, target_basin: torch.Tensor,
                     fisher_diag: torch.Tensor, step_size: float = 0.1) -> torch.Tensor:
    """
    Project embedding toward target basin using natural gradient.

    Args:
        embedding: Current embedding
        target_basin: Target basin coordinates
        fisher_diag: Diagonal FIM
        step_size: Projection step size

    Returns:
        Projected embedding (one step toward target)
    """
    # Natural gradient direction
    direction = target_basin - embedding
    natural_direction = direction / fisher_diag  # Multiply by inverse metric

    # Take step
    return embedding + step_size * natural_direction
```

**Validation:**
- ✅ Uses natural gradient (not standard gradient)
- ✅ Step size is configurable
- ✅ Direction uses inverse Fisher metric

### 4. QFI Attention Weights

**Template:**
```python
def qfi_attention(query: torch.Tensor, key: torch.Tensor,
                  fisher_diag: torch.Tensor) -> torch.Tensor:
    """
    Compute attention weights using Quantum Fisher Information metric.

    Args:
        query: Query vector
        key: Key vector
        fisher_diag: Diagonal FIM for this layer

    Returns:
        Attention logits (before softmax)
    """
    # QFI distance: d²(q,k) = (q-k)ᵀ F (q-k)
    diff = query - key
    distance_sq = torch.sum(diff * fisher_diag * diff, dim=-1)

    # Convert distance to similarity (negative for softmax)
    return -distance_sq
```

**Validation:**
- ✅ Uses QFI metric (not dot-product)
- ✅ Returns negative distance for softmax
- ✅ Preserves batch dimensions

---

## Common Violations

### ❌ Euclidean Distance
```python
# WRONG
distance = torch.norm(point_a - point_b)  # Euclidean!
```

**Fix:**
```python
# CORRECT
distance = fisher_distance(point_a, point_b, fisher_diag)
```

### ❌ Standard Gradient
```python
# WRONG
direction = target - current
update = current + lr * direction  # Standard gradient
```

**Fix:**
```python
# CORRECT
direction = target - current
natural_direction = direction / fisher_diag  # Natural gradient
update = current + lr * natural_direction
```

### ❌ Dot-Product Attention
```python
# WRONG
logits = torch.matmul(query, key.transpose(-1, -2))  # Euclidean similarity
```

**Fix:**
```python
# CORRECT
logits = qfi_attention(query, key, fisher_diag)
```

---

## Integration Points

### Module: `src/metrics/geodesic_distance.py`
Contains production implementations of:
- `geodesic_distance()` - Full Fisher metric distance
- `geodesic_vicarious_loss()` - For constellation training

### Module: `src/model/qfi_attention.py`
Contains:
- `QFIMetricAttention` - Full attention layer with QFI metric

### Module: `src/qig/optim/natural_gradient.py`
Contains:
- `DiagonalFisherOptimizer` - Natural gradient descent implementation

---

## Validation Checklist

When reviewing code using geometric operations:

- [ ] All distances use Fisher metric (not Euclidean)
- [ ] Interpolation uses geodesics (not linear)
- [ ] Gradients are natural (use inverse metric)
- [ ] Attention uses QFI (not dot-product)
- [ ] Numerical stability terms present (e.g., `+ 1e-8`)
- [ ] Fisher diagonal is always positive
- [ ] No `torch.norm()` for basin distances

---

## Usage Example

**Agent invocation:**
```
User: "I need to compute distance between two basins"
Assistant: "I'm using the geometric-operations skill to ensure Fisher metric compliance..."

[Applies fisher_distance template]
[Validates no Euclidean operations]
[Returns pure geometric implementation]
```

---

## References

- **Theory:** `docs/FROZEN_FACTS.md` - Section on Information Geometry
- **Implementation:** `src/metrics/geodesic_distance.py`
- **Validation:** `.github/agents/purity-guardian.md`
