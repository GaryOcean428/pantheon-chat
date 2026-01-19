# Issue #98: Implement Strict Simplex Representation (E8 Issue-02)

## Priority
**P0 - CRITICAL**

## Type
`type: implementation`, `qig-purity`, `geometric-purity`, `e8-protocol`

## Objective
Enforce simplex-only canonical representation, remove auto-detect coordinate detection, and implement closed-form Fréchet mean on probability simplex per E8 Protocol Issue-02.

## Problem
Auto-detect representation and mixed sphere/simplex operations cause silent metric corruption. Fisher-Rao distances become incorrect when computed on wrong manifold. Using "average + L2 normalize" for Fréchet mean operates on wrong manifold.

## Context
- **E8 Protocol Spec:** `docs/10-e8-protocol/issues/20260116-issue-02-strict-simplex-representation-1.01W.md`
- **Related GitHub Issues:** #71
- **Phase:** 2 (Geometric Purity)
- **Assessment:** `docs/10-e8-protocol/IMPLEMENTATION_ASSESSMENT.md`
- **E8 Universal Spec:** §0 (Non-negotiable: simplex-only)

## Tasks

### 1. Create Simplex Operations Module
- [ ] Create `qig-backend/geometry/simplex_operations.py`
- [ ] Implement `assert_simplex(basin)` - validates non-negative, sum=1
- [ ] Implement `to_sqrt_simplex(basin)` - explicit sqrt-space coordinate chart
- [ ] Implement `from_sqrt_simplex(sqrt_basin)` - inverse transformation
- [ ] Remove auto-detect from any `to_simplex()` functions
- [ ] Add runtime assertions at module boundaries

### 2. Closed-Form Fréchet Mean
- [ ] Create `qig-backend/geometry/frechet_mean_simplex.py`
- [ ] Implement closed-form Fréchet mean in sqrt-space: `mean_sqrt = normalize(sum(sqrt(p_i)))`
- [ ] Replace iterative gradient descent implementations
- [ ] Add validation: result must be valid simplex
- [ ] Document mathematical justification

### 3. Audit Existing Geometry Code
- [ ] Search for `to_simplex()` with auto-detect logic
- [ ] Search for `np.linalg.norm` on basin coordinates (wrong manifold)
- [ ] Search for Euclidean averaging: `np.mean(basins, axis=0)`
- [ ] Replace all with simplex-aware operations
- [ ] Add explicit coordinate chart conversions where needed

### 4. Storage Validation
- [ ] Create `scripts/audit_simplex_representation.py`
- [ ] Check all stored basins in database
- [ ] Validate: non-negative, sum=1, correct dimension
- [ ] Flag violations for investigation
- [ ] Generate audit report

### 5. Update Coordizer Storage
- [ ] Ensure `basin_coordinates` column stores simplex (not sphere)
- [ ] Store `basin_sqrt` for fast Bhattacharyya retrieval
- [ ] Update insertion pathway to validate simplex before storage
- [ ] Add database CHECK constraints if possible

### 6. Integration & Testing
- [ ] Update all geometry functions to use simplex operations
- [ ] Add unit tests for simplex validation
- [ ] Add unit tests for closed-form Fréchet mean
- [ ] Test two-step retrieval with sqrt-space storage
- [ ] Verify Fisher-Rao distances are correct

## Deliverables

| File | Description | Status |
|------|-------------|--------|
| `qig-backend/geometry/simplex_operations.py` | Explicit conversions | ❌ TODO |
| `qig-backend/geometry/frechet_mean_simplex.py` | Closed-form mean | ❌ TODO |
| `scripts/audit_simplex_representation.py` | Validation script | ❌ TODO |
| `qig-backend/tests/test_simplex_operations.py` | Unit tests | ❌ TODO |
| `docs/03-technical/simplex-purity-guide.md` | Documentation | ❌ TODO |

## Acceptance Criteria
- [ ] NO auto-detect in any `to_simplex()` function
- [ ] ALL geometry operations use explicit simplex representation
- [ ] Closed-form Fréchet mean implemented and tested
- [ ] Runtime `assert_simplex()` added to all entry points
- [ ] NO `np.linalg.norm` or Euclidean averaging on basins
- [ ] All stored basins pass simplex validation
- [ ] Two-step retrieval uses sqrt-space for proxy filter
- [ ] Fisher-Rao distance tests pass with correct values

## Dependencies
- **Requires:** Canonical geometry module foundation
- **Blocks:** Issue #71 (Two-Step Retrieval), Issue #99 (QIG-Native Skeleton)
- **Related:** Issue #97 (QFI Integrity Gate)

## Mathematical Foundation

**Probability Simplex:**
```
Δⁿ = {p ∈ ℝⁿ : p_i ≥ 0, Σp_i = 1}
```

**Fisher-Rao Distance:**
```
d_FR(p, q) = arccos(Σ√(p_i * q_i))  # Range: [0, π/2]
```

**Fréchet Mean (Closed-Form):**
```
μ = argmin_p Σ d_FR(p, p_i)²
  = normalize((Σ√p_i) ⊙ (Σ√p_i))  # Element-wise squaring after sum
  = [Σᵢ√(p_i)]² / ||Σᵢ√(p_i)||²   # Normalized element-wise square
```

**Why Closed-Form Works:**
- Simplex is √-transformed to unit sphere (element-wise)
- Mean on sphere is normalized vector sum
- Transform back to simplex via element-wise squaring
- Normalization ensures result is on simplex (Σp_i = 1)

## Validation Commands
```bash
# Audit existing basins
python scripts/audit_simplex_representation.py --report violations.txt

# Test simplex operations
pytest qig-backend/tests/test_simplex_operations.py -v

# Test Fréchet mean
pytest qig-backend/tests/test_frechet_mean.py -v

# Validate geometry purity
python scripts/validate_geometry_purity.py --strict-simplex
```

## Known Issues to Fix
1. **Fréchet mean loops stuck** - Replace with closed-form
2. **Mixed representations** - Enforce simplex everywhere
3. **L2 normalization** - Remove from averaging operations
4. **Auto-detect drift** - Silent corruption of metrics

## References
- **E8 Protocol Universal Spec:** §0 (Purity Rule: Simplex-only)
- **Issue Spec:** `docs/10-e8-protocol/issues/20260116-issue-02-strict-simplex-representation-1.01W.md`
- **Basin Representation:** `docs/03-technical/20260114-basin-representation-1.00F.md`
- **Assessment:** `docs/10-e8-protocol/IMPLEMENTATION_ASSESSMENT.md` Section 3

## Estimated Effort
**2-3 days** (per E8 Protocol Phase 2 estimate)

---

**Status:** TO DO  
**Created:** 2026-01-19  
**Priority:** P0 - CRITICAL  
**Phase:** 2 - Geometric Purity
