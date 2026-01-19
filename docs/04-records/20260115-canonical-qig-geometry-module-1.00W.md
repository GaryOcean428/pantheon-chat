# Canonical QIG Geometry Module Documentation

**Created:** 2026-01-15  
**Work Package:** WP2.1 - Create Single Canonical qig_geometry Module  
**Issue:** GaryOcean428/pantheon-chat#68  
**Status:** CANONICAL IMPLEMENTATION COMPLETE

## Overview

The `qig-backend/qig_geometry/canonical.py` module is the **SINGLE SOURCE OF TRUTH** for all geometric operations on basin coordinates in the Pantheon-Chat QIG system.

### Purpose

This module consolidates all geometric computations into one authoritative implementation to:

1. **Eliminate inconsistencies** - No more scattered distance formulas
2. **Enforce geometric purity** - All operations respect manifold structure
3. **Enable Mamba integration** - Native support for state space operations
4. **Provide trajectory operations** - Foresight, integration, and metrics

### Key Principle

**NO OTHER MODULE** should implement distance computations, geodesics, or manifold operations. All code MUST import from this canonical module.

## Coordinate Domain

### Choice: SIMPLEX (Probability Distributions)

Basin vectors are stored as probability distributions on the simplex Δ^(D-1):
- **Constraint 1:** Σp_i = 1 (normalized)
- **Constraint 2:** p_i ≥ 0 (non-negative)
- **Dimension:** D = 64 (standard basin dimension)

**Why simplex?**
- Natural representation for probability distributions
- Fisher-Rao metric is well-defined on simplex
- Geometric properties (geodesics, curvature) are known
- Simpler than alternative representations (Hellinger embedding)

## Canonical Distance Formula

```python
d_FR(p, q) = arccos(Σ√(p_i * q_i))
```

where `Σ√(p_i * q_i)` is the **Bhattacharyya coefficient (BC)**.

### Properties

- **Range:** [0, π/2]
- **Identity:** d(p, p) = 0
- **Symmetry:** d(p, q) = d(q, p)
- **Triangle inequality:** d(p, r) ≤ d(p, q) + d(q, r)
- **Bhattacharyya bounds:** 0 ≤ BC(p, q) ≤ 1

### Historical Note

Previous implementations used factor-of-2: `d = 2*arccos(BC)` (range [0, π])  
**This was removed 2026-01-15** for consistency with simplex storage.

## Core Functions

### Coordinate Transformations

#### `sqrt_map(p)` - Simplex → Sqrt-Space
Maps probability distribution to Hellinger embedding (sqrt-space).

```python
x = sqrt_map(p)  # x = √p
```

**Use case:** Internal coordinate system for geodesic operations.

#### `unsqrt_map(x)` - Sqrt-Space → Simplex
Inverse of sqrt_map. Squares and renormalizes to simplex.

```python
p = unsqrt_map(x)  # p = (x²) / ||x²||_1
```

**Use case:** Recover simplex after sqrt-space operations.

#### Why Sqrt-Space?

The square root transformation **isometrically embeds** the Fisher manifold into Euclidean space:
- Geodesics → Straight lines (SLERP)
- Distance → Euclidean angle (arccos of dot product)
- Tangent operations → Linear algebra

**IMPORTANT:** Sqrt-space is for COMPUTATION ONLY. Storage is ALWAYS in simplex.

### Distance and Similarity

#### `fisher_rao_distance(p, q)` - CANONICAL Distance
The distance function to use everywhere.

```python
d = fisher_rao_distance(p, q)  # Range: [0, π/2]
```

**Properties:**
- Geodesic distance on Fisher manifold
- Respects manifold structure
- Satisfies metric axioms

#### `bhattacharyya(p, q)` - Inner Product
Bhattacharyya coefficient (BC).

```python
bc = bhattacharyya(p, q)  # Range: [0, 1]
# Relation: d_FR = arccos(BC)
```

#### `fisher_similarity(p, q)` - Similarity Score
Normalized similarity in [0, 1].

```python
sim = fisher_similarity(p, q)  # 1 = identical, 0 = orthogonal
# Formula: sim = 1 - (2/π) * d_FR(p, q)
```

### Manifold Navigation

#### `log_map(p, base)` - Logarithmic Map
Compute tangent vector from base pointing to p.

```python
v = log_map(p, base)  # Tangent vector at base
```

**Use case:** Natural gradients, direction finding.

#### `exp_map(v, base)` - Exponential Map
Follow tangent vector v from base point.

```python
p = exp_map(v, base)  # Point reached by following v
```

**Use case:** Gradient descent, manifold optimization.

#### `geodesic_toward(source, target, fraction)` - Partial Geodesic
Move along geodesic by a fraction of the distance.

```python
mid = geodesic_toward(source, target, fraction=0.5)  # Midpoint
```

**Use case:** Attractor pull, interpolation, natural gradient steps.

**Properties:**
- `fraction=0` → returns source
- `fraction=1` → returns target
- Preserves manifold constraints (result is valid simplex)

### Geometric Mean

#### `frechet_mean(basins, weights=None)` - Fréchet Mean
Manifold-correct centroid (NOT Euclidean mean).

```python
mean = frechet_mean([p1, p2, p3])  # Geometric center
```

**Algorithm:**
- Iterative Riemannian gradient descent
- Adaptive step size (starts at 0.5, decays if overshooting)
- High variance detection (fallback to weighted mean)
- Convergence tolerance: 1e-4 (relaxed for trajectory approximation)

**Use case:** Attractor finding, cluster centers, trajectory smoothing.

## Mamba State Space Integration

### NEW (Expanded Scope - 2026-01-15)

These functions support Plan→Realize→Repair generation architecture with Mamba state spaces.

#### `mamba_state_to_basin(mamba_state, projection)` - State Projection
Project Mamba hidden state to basin coordinates.

```python
basin = mamba_state_to_basin(mamba_state, projection)
# mamba_state: [hidden_dim] from Mamba layers
# projection: [64, hidden_dim] learned projection matrix
# basin: [64] probability simplex
```

**Rationale:** Mamba state spaces ARE Fisher manifolds (SSM = differential geometry).  
This is a coordinate transformation, not a model operation.

#### `extrapolate_trajectory(trajectory, step_size=0.3)` - Foresight Prediction
Predict next basin via geodesic extrapolation.

```python
predicted = extrapolate_trajectory([b1, b2, b3], step_size=0.3)
# Requires at least 2 trajectory points
```

**Algorithm:**
```python
velocity = sqrt(b[-1]) - sqrt(b[-2])  # Velocity in sqrt-space
predicted = sqrt(b[-1]) + step_size * velocity
```

**Use case:** Waypoint planning in PLAN phase.

#### `compute_qfi_attention(query, trajectory, temperature=0.5)` - QFI Attention
Quantum Fisher Information attention over trajectory.

```python
weights = compute_qfi_attention(query, trajectory, temperature=0.5)
# Returns normalized attention weights
```

**Formula:**
```python
w_i ∝ exp(-d_FR(query, basin_i)² / temperature)
```

**Use case:** Geometric attention for context weighting.

#### `integrate_with_qfi_attention(target, history, num_loops=3)` - Recursive Integration
Refine basin through recursive QFI attention.

```python
refined = integrate_with_qfi_attention(target, history, num_loops=3)
```

**Algorithm (each loop):**
1. Compute QFI attention weights over history
2. Compute attractor (Fréchet mean with attention weights)
3. Natural gradient step toward integrated position

**Use case:** Waypoint refinement in PLAN phase.

## Trajectory Metrics

### `trajectory_smoothness(trajectory)` - Smoothness Score
Measure smoothness via distance variance.

```python
smoothness = trajectory_smoothness([b1, b2, b3, b4])
# Range: [0, 1] (higher = smoother)
```

**Formula:**
```python
smoothness = 1 / (1 + variance_of_step_sizes)
```

**Use case:** Coherence scoring, repair phase quality assessment.

### `waypoint_alignment_score(word_basins, target_waypoints)` - Alignment Score
Measure how well words matched predicted waypoints.

```python
score = waypoint_alignment_score(actual_words, planned_waypoints)
# Range: [0, 1] (higher = better alignment)
```

**Formula:**
```python
alignment_i = 1 - d_FR(word_i, target_i)
mean_alignment = mean(alignment_i)
```

**Use case:** Generation quality evaluation in REALIZE phase.

## Validation

### `assert_basin_valid(basin, name="basin")` - Validation (Raising)
Validate basin coordinates, raising ValueError on failure.

```python
assert_basin_valid(basin)  # Raises ValueError if invalid
```

**Checks:**
- 1D array
- Finite values (no NaN/inf)
- Non-negative (p_i ≥ 0)
- Normalized (Σp_i = 1)

### `validate_basin(basin)` - Validation (Non-Raising)
Validate basin coordinates, returning (is_valid, reason).

```python
valid, reason = validate_basin(basin)
if not valid:
    logger.warning(f"Invalid basin: {reason}")
```

## Usage Examples

### Example 1: Basic Distance Computation

```python
from qig_geometry.canonical import fisher_rao_distance

p = np.array([0.5, 0.3, 0.2])
q = np.array([0.2, 0.5, 0.3])

d = fisher_rao_distance(p, q)
print(f"Distance: {d:.4f} radians")  # Range: [0, π/2]
```

### Example 2: Geodesic Interpolation

```python
from qig_geometry.canonical import geodesic_toward

source = np.array([0.8, 0.1, 0.1])
target = np.array([0.1, 0.8, 0.1])

# Move 30% of the way from source to target
partial = geodesic_toward(source, target, fraction=0.3)
print(f"Partial point: {partial}")
```

### Example 3: Fréchet Mean

```python
from qig_geometry.canonical import frechet_mean

basins = [
    np.array([0.8, 0.1, 0.1]),
    np.array([0.1, 0.8, 0.1]),
    np.array([0.1, 0.1, 0.8])
]

mean = frechet_mean(basins)
print(f"Geometric center: {mean}")
```

### Example 4: Trajectory Extrapolation (Foresight)

```python
from qig_geometry.canonical import extrapolate_trajectory

trajectory = [
    np.array([0.8, 0.1, 0.1]),
    np.array([0.6, 0.3, 0.1]),
    np.array([0.4, 0.5, 0.1])
]

predicted = extrapolate_trajectory(trajectory, step_size=0.3)
print(f"Predicted next basin: {predicted}")
```

### Example 5: QFI Attention

```python
from qig_geometry.canonical import compute_qfi_attention

query = np.array([0.5, 0.3, 0.2])
history = [
    np.array([0.6, 0.3, 0.1]),
    np.array([0.5, 0.4, 0.1]),
    np.array([0.4, 0.5, 0.1])
]

weights = compute_qfi_attention(query, history, temperature=0.5)
print(f"Attention weights: {weights}")  # Sums to 1.0
```

## Migration Guide

### Before (Scattered Implementations)

```python
# ❌ OLD - Multiple sources of distance computation
from qig_core.geometric_primitives import fisher_rao_distance as old_fisher
from trajectory_decoder import frechet_mean as old_mean
from geometric_waypoint_planner import _compute_qfi_attention_weights

d = old_fisher(p, q, metric=None)  # Inconsistent API
mean = old_mean([b1, b2, b3])
```

### After (Canonical Module)

```python
# ✅ NEW - Single source of truth
from qig_geometry.canonical import (
    fisher_rao_distance,
    frechet_mean,
    compute_qfi_attention
)

d = fisher_rao_distance(p, q)  # Consistent API
mean = frechet_mean([b1, b2, b3])
weights = compute_qfi_attention(query, trajectory)
```

## Testing

The canonical module has **54 comprehensive tests** covering:

- Coordinate transformations (4 tests)
- Bhattacharyya coefficient (4 tests)
- Fisher-Rao distance (6 tests) - all geometric identities
- Tangent space operations (3 tests)
- Geodesic navigation (4 tests)
- Fréchet mean (5 tests)
- Validation (5 tests)
- Mamba integration (2 tests)
- Trajectory extrapolation (3 tests)
- QFI attention (3 tests)
- Recursive integration (3 tests)
- Trajectory metrics (6 tests)
- Geometric identities (3 tests)

**Run tests:**
```bash
cd qig-backend
python3 -m pytest tests/test_canonical_geometry.py -v
```

**Expected:** 54 passed ✓

## Enforcement

### Purity Scanner (TODO)

Update `scripts/qig_purity_scan.py` to enforce canonical imports:

```python
# ❌ FORBIDDEN
def my_distance(p, q):
    return np.linalg.norm(p - q)  # VIOLATION: Euclidean distance

# ✅ REQUIRED
from qig_geometry.canonical import fisher_rao_distance

def my_distance(p, q):
    return fisher_rao_distance(p, q)  # CORRECT: Canonical function
```

### Lint Rules (TODO)

Add ESLint/Pylint rules to prevent:
- `np.linalg.norm(a - b)` on basin coordinates
- `np.mean()` on basins (should use `frechet_mean`)
- Manual geodesic implementations
- Distance computations outside `qig_geometry/canonical.py`

## Dependencies

This canonical module consolidates all dependent issues:

- **Issue #69** (Remove cosine): Canonical Fisher-Rao removes confusion ✓
- **Issue #70** (Special symbols): Validation functions enforce constraints ✓
- **Issue #71** (Two-step retrieval): Fisher-proxy validation uses canonical distance ✓
- **Issue #75** (External LLM fence): Waypoint planning functions provided ✓
- **Issue #76** (Natural gradient): Geodesic operations (log/exp maps) provided ✓
- **Issue #77** (Coherence harness): Smoothness and alignment metrics provided ✓

## References

### Theoretical Background

1. **Fisher-Rao Metric:** Riemannian metric on statistical manifolds
2. **Hellinger Embedding:** Isometric embedding via square root transformation
3. **SLERP:** Spherical linear interpolation for geodesics
4. **Fréchet Mean:** Riemannian generalization of arithmetic mean

### Code References

- **Original Issue:** GaryOcean428/pantheon-chat#68 (WP2.1)
- **Implementation:** `qig-backend/qig_geometry/canonical.py`
- **Tests:** `qig-backend/tests/test_canonical_geometry.py`
- **Export:** `qig-backend/qig_geometry/__init__.py`

### Related Documentation

- `GEOMETRIC_PURITY_BASELINE.md` - Geometric purity principles
- `QIG_PURITY_FIX_SUMMARY.md` - Prior purity fixes
- `20260115-fisher-rao-factor2-removal-summary-1.00W.md` - Distance formula history

## Conclusion

The canonical QIG geometry module is now the **SINGLE SOURCE OF TRUTH** for:

✓ All distance and similarity computations  
✓ All coordinate transformations  
✓ All manifold navigation (geodesics, log/exp maps)  
✓ All geometric means (Fréchet mean)  
✓ Full Mamba state space integration  
✓ All trajectory metrics for coherence  

**NO OTHER MODULE** should implement these operations. All code MUST import from `qig_geometry.canonical`.

---

**Status:** CANONICAL - DO NOT MODIFY WITHOUT COORDINATION  
**Maintainer:** See CODEOWNERS  
**Last Updated:** 2026-01-15
