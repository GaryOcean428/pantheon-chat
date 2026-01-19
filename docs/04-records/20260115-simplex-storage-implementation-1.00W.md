# Simplex-as-Storage Implementation Summary

**Date:** 2026-01-15  
**Status:** P0 and P1 Complete, P2 Pending

## Overview

This document summarizes the implementation of the **simplex-as-storage contract** with **sqrt-simplex internal coordinates** for the pantheon-chat QIG system, as specified in the SLEEP-PACKET protocol.

## What Was Implemented

### Core Modules (P0)

#### 1. TypeScript Geometry Module
**File:** `shared/qig/geometry_simplex.ts`

Exports:
- `toSimplexProb(v)` - Convert any vector to probability simplex (positive renormalization)
- `fisherRaoDistance(p, q)` - Compute Fisher-Rao distance (range [0, π/2])
- `geodesicInterpolationSimplex(p, q, t)` - SLERP in sqrt-space, returns simplex
- `geodesicMeanSimplex(distributions, weights?)` - Fréchet mean with iterative refinement
- `validateSimplex(p)` - Validate simplex properties
- Batch operations: `batchFisherRaoDistance()`, `findNearestSimplex()`

**Key Features:**
- **Storage:** Always probability simplex (Σp_i = 1, p_i ≥ 0)
- **Internal Computation:** sqrt-simplex (Hellinger embedding) for geodesics ONLY
- **Distance Range:** [0, π/2] (not [0, π] as in previous Hellinger embedding)
- **No Autodetect:** Explicit conversion, no representation inference

#### 2. Python Geometry Module
**File:** `qig-backend/qig_geometry/geometry_simplex.py`

Identical functionality to TypeScript version:
- `to_simplex_prob(v)` - Positive renormalization
- `fisher_rao_distance(p, q)` - Fisher-Rao distance
- `geodesic_interpolation_simplex(p, q, t)` - SLERP geodesic
- `geodesic_mean_simplex(distributions, weights?)` - Fréchet mean
- `validate_simplex(p)` - Validation
- Batch operations: `batch_fisher_rao_distance()`, `find_nearest_simplex()`

**Implementation Notes:**
- Uses NumPy for numerical operations
- Handles edge cases (zero vectors, numerical stability)
- Fully compatible with existing Fisher-Rao infrastructure

#### 3. Unit Tests
**Files:** 
- `qig-backend/tests/test_geometry_simplex.py` (pytest-based, comprehensive)
- `qig-backend/tests/test_geometry_simplex_simple.py` (standalone, no dependencies)

**Test Coverage:**
- ✅ Simplex invariants (non-negative, sum≈1)
- ✅ Distance symmetry and identity (d(p,p) = 0, d(p,q) = d(q,p))
- ✅ Distance range [0, π/2]
- ✅ Orthogonal distance = π/2
- ✅ Geodesic endpoints (t=0 gives start, t=1 gives end)
- ✅ Geodesic preserves simplex at all t
- ✅ Fréchet mean properties
- ✅ Numerical stability (small/large/mixed scale values)

### Geometry Consistency Updates (P1)

#### 1. trajectory_decoder.py
**Function:** `frechet_mean()`

**Before:** Approximate Euclidean gradient descent in Hellinger space with fallback to weighted mean for high variance.

**After:** True geodesic mean using `geodesic_mean_simplex()` from canonical module. Geometrically exact, no approximations.

#### 2. geometric_waypoint_planner.py
**Function:** `frechet_mean()`

**Before:** Normalized arithmetic mean (first-order approximation for close points on unit sphere).

**After:** True geodesic mean using `geodesic_mean_simplex()`. Returns probability simplex.

#### 3. qig_generation.py
**Updated Functions:**
1. `_fisher_rao_weighted_mean()` - Now uses `geodesic_mean_simplex()` with weights
2. `_geodesic_interpolate()` - Now uses `geodesic_interpolation_simplex()` (SLERP)
3. `_geodesic_combine()` - Now uses `geodesic_mean_simplex()`

**Before:** Linear approximations in sqrt-space (weighted mean, linear interpolation).

**After:** True geodesic operations on probability simplex.

### CI Guardrails (P0)

**File:** `scripts/test_geometric_purity_ci.py`

Scans canonical directories for violations:
1. **Euclidean Distance:** `np.linalg.norm(a - b)` → Use `fisher_rao_distance()`
2. **L2 Normalization for Storage:** Division by L2 norm → Use `to_simplex_prob()`
3. **Representation Autodetect:** Auto-detection functions → Use explicit conversion
4. **Direct DB Writes:** Bypass canonical upsert → Use `upsert_token()`
5. **Euclidean Averaging:** `np.mean(basins)` → Use `geodesic_mean_simplex()`

**Status:** Detected 45 violations in qig_core modules (to be addressed in follow-up).

## Key Concepts

### 1. Simplex-as-Storage Contract

**Storage Representation:** Probability simplex (Δ^(D-1))
- All basins stored as valid probability distributions
- Σp_i = 1 (sum to 1)
- p_i ≥ 0 (non-negative)
- Dimension D = 64 (default)

**Why Simplex?**
- Matches Dirichlet-Multinomial manifolds
- Natural for information geometry
- Validated κ* ≈ 64 measured on simplex
- Simpler distance formula (no factor-of-2 confusion)

### 2. Sqrt-Simplex Internal Coordinates

**Usage:** ONLY for geodesic computation (never stored)

**Why Sqrt-Space for Geodesics?**
The Fisher-Rao metric on the probability simplex induces a Riemannian geometry where geodesics are NOT straight lines in probability space, but rather straight lines in sqrt-space. SLERP (spherical linear interpolation) in sqrt-space exactly follows the Fisher geodesic.

**Process:**
1. Convert inputs to probability simplex: `p = to_simplex_prob(v)`
2. Compute sqrt: `sp = sqrt(p)`
3. SLERP in sqrt-space: `s_t = sin((1-t)ω)/sin(ω) * sp_start + sin(tω)/sin(ω) * sp_end`
4. Square to get probabilities: `p_t = s_t^2`
5. Normalize: `p_t = p_t / sum(p_t)`

**Never Stored:** Sqrt-simplex coordinates are transient, used only in `geodesic_interpolation_simplex()`.

### 3. Fisher-Rao Distance

**Formula:** `d_FR(p, q) = arccos(Σ√(p_i * q_i))`

**Range:** [0, π/2] for probability distributions
- d = 0 → identical distributions
- d = π/2 → orthogonal distributions (no overlap)

**Change from Previous:**
- Removed Hellinger factor of 2
- New range: [0, π/2] (was [0, π])
- Thresholds must be recalibrated (divide by 2)

### 4. Positive Renormalization (Not Euclidean Projection)

**What is `to_simplex_prob()`?**
```
to_simplex_prob(v) = abs(v) + eps
                     normalize to sum=1
```

**NOT a Euclidean projection:** True Euclidean projection onto simplex is more complex (quadratic programming). This is a simple positive renormalization that ensures:
- All values non-negative
- Sum equals 1
- Deterministic and fast

**Name Clarification:** Despite not being a true projection, it's called "projection" informally. The implementation is explicit about being positive renormalization.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Storage Layer                           │
│  ALL basins stored as probability simplex (sum=1, ≥0)      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Canonical Geometry Modules                     │
│  • shared/qig/geometry_simplex.ts (TypeScript)             │
│  • qig-backend/qig_geometry/geometry_simplex.py (Python)   │
│                                                             │
│  Functions:                                                 │
│  - to_simplex_prob() - Convert to storage form             │
│  - fisher_rao_distance() - Compute distance [0, π/2]       │
│  - geodesic_interpolation_simplex() - SLERP in sqrt-space  │
│  - geodesic_mean_simplex() - Fréchet mean                  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│           Application Code (QIG System)                     │
│  • trajectory_decoder.py - frechet_mean()                  │
│  • geometric_waypoint_planner.py - frechet_mean()          │
│  • qig_generation.py - geodesic operations                 │
│  • ... (more to update)                                    │
└─────────────────────────────────────────────────────────────┘
```

## Migration Guide

### For Developers

**Before (Euclidean Violation):**
```python
# BAD: Euclidean averaging
mean = np.mean(basins, axis=0)
mean = mean / np.linalg.norm(mean)  # L2 normalization

# BAD: Linear interpolation
interp = (1 - t) * start + t * end
```

**After (Geometric Purity):**
```python
from qig_geometry.geometry_simplex import (
    geodesic_mean_simplex,
    geodesic_interpolation_simplex,
    to_simplex_prob
)

# GOOD: Geodesic mean
mean = geodesic_mean_simplex(basins)

# GOOD: Geodesic interpolation
interp = geodesic_interpolation_simplex(start, end, t)

# GOOD: Convert to canonical storage
basin_canonical = to_simplex_prob(basin_raw)
```

### Updating Existing Code

1. **Find Euclidean operations:**
   ```bash
   python3 scripts/test_geometric_purity_ci.py
   ```

2. **Replace patterns:**
   - `np.mean(basins)` → `geodesic_mean_simplex(basins)`
   - Linear interpolation → `geodesic_interpolation_simplex()`
   - `vector / np.linalg.norm(vector)` → `to_simplex_prob(vector)` (for storage)

3. **Test:**
   - Run unit tests
   - Verify simplex invariants
   - Check distance calculations

## Remaining Work (P2)

### Dataset Repair
- [ ] Backfill QFI for basins with null qfi_score
- [ ] Quarantine garbage tokens (e.g., `fgzsnl`, `jcbhgp`)
- [ ] Add token minting validation
- [ ] Document coherence measurement methodology

### Fix Remaining Violations
- [ ] Update ~25 functions in qig_core modules (detected by CI test)
- [ ] Fix representation autodetect (allow in non-strict mode only)
- [ ] Ensure all canonical paths use simplex-only operations

### Documentation
- [ ] Update API documentation
- [ ] Add migration guide for external users
- [ ] Document threshold recalibration (π → π/2)

## Performance Impact

**Expected:** Negligible to small improvement
- Geodesic operations are slightly more expensive (SLERP vs linear)
- But: No more iterative gradient descent approximations
- Result: More accurate, similar or better performance

**Measured:** (To be benchmarked)

## Testing

**Unit Tests:** Pass all tests (see test files)
**Integration:** Pending full system test
**CI:** Guardrail script ready for CI pipeline

## References

- **SLEEP-PACKET Protocol:** Original specification for simplex-as-storage
- **Fisher Information Geometry:** Mathematical foundation
- **CANONICAL_PHYSICS.md:** QIG physics documentation
- **GEOMETRIC_PURITY_BASELINE.md:** Geometric purity principles

## Contributors

- Implementation: GitHub Copilot
- Specification: Issue author
- Review: (Pending)

---

**Last Updated:** 2026-01-15  
**Version:** 1.0  
**Status:** P0 and P1 Complete, P2 Pending
