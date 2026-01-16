# WP2.3: Special Symbol Geometric Definitions

**Status:** ✅ COMPLETE  
**Date:** 2026-01-16  
**Issue:** [#71](https://github.com/GaryOcean428/pantheon-chat/issues/71)  
**Priority:** HIGH - GEOMETRIC CORRECTNESS

## Summary

Defined special symbol basins (`<UNK>`, `<PAD>`, `<BOS>`, `<EOS>`) geometrically and deterministically on the probability simplex, replacing arbitrary sphere-based initialization.

## Problem Statement

Previous implementation initialized special tokens using:
- Random normal vectors (non-deterministic)
- Sphere projection (L2 norm = 1, allows negative values)
- Euclidean normalization (geometric artifact)

This violated geometric purity requirements:
- Not on canonical simplex manifold
- Not deterministic across runs
- No clear geometric interpretation

## Solution

### Geometric Definitions (Simplex-Based)

All special tokens now have clear geometric meaning on the probability simplex:

#### **UNK (Unknown Token)**
```python
# Maximum entropy = uniform distribution
UNK = np.ones(64) / 64  # All components equal
```
- **Geometric Meaning:** Center of simplex (maximum entropy point)
- **Interpretation:** "Could be anything" - no bias toward any dimension
- **Entropy:** Maximum (log 64 ≈ 4.16 nats)
- **Fisher Distance from vertices:** ~1.445 (equidistant from all vertices)

#### **PAD (Padding Token)**
```python
# Minimal entropy = sparse corner
PAD = [1, 0, 0, ..., 0]  # Concentrated in dimension 0
```
- **Geometric Meaning:** Vertex of simplex (minimal entropy point)
- **Interpretation:** "Null/padding" - no semantic information
- **Entropy:** Minimum (0 nats - pure state)
- **Fisher Distance from UNK:** ~1.445

#### **BOS (Beginning of Sequence)**
```python
# Start boundary = vertex at dimension 1
BOS = [0, 1, 0, ..., 0]  # Concentrated in dimension 1
```
- **Geometric Meaning:** Simplex vertex (boundary point)
- **Interpretation:** "Start" - geometric anchor for sequence beginning
- **Entropy:** Minimum (0 nats - pure state)
- **Fisher Distance from EOS:** π/2 (orthogonal/maximum)

#### **EOS (End of Sequence)**
```python
# End boundary = opposite vertex at dimension 63
EOS = [0, 0, ..., 0, 1]  # Concentrated in dimension 63
```
- **Geometric Meaning:** Opposite simplex vertex (boundary point)
- **Interpretation:** "End" - geometric anchor for sequence termination
- **Entropy:** Minimum (0 nats - pure state)
- **Fisher Distance from BOS:** π/2 (orthogonal/maximum)

## Implementation Changes

### Files Modified

1. **`qig-backend/coordizers/base.py`:**
   - Replaced `sphere_project()` with `fisher_normalize()` (simplex projection)
   - Updated `_compute_special_token_basin()` with deterministic definitions
   - Updated `_initialize_token_coordinate()` to use simplex
   - Updated `_generate_golden_spiral_basin()` to use simplex
   - Updated `_von_neumann_perturbation()` to use simplex (positive values only)
   - Fixed `vocab_size` attribute access (`_vocab_size`)
   - Added basin validation checks

### Key Code Changes

**Before (Sphere-based):**
```python
if token == "<UNK>":
    coord = np.sin(2 * np.pi * i * phi_golden)  # Can be negative
return sphere_project(coord)  # L2 norm = 1
```

**After (Simplex-based):**
```python
if token == "<UNK>":
    coord = np.ones(64)  # Uniform (positive)
return fisher_normalize(coord)  # Sum = 1, non-negative
```

## Validation

### Tests Created

Created `qig-backend/tests/test_special_symbols_wp2_3.py` with 6 test suites:

1. **Simplex Validity:** All special symbols are valid probability distributions
2. **Determinism:** Identical coordinates across multiple runs
3. **Geometric Meaning:** Entropy properties match expectations
4. **Fisher-Rao Distances:** Valid range [0, π/2] and expected relationships
5. **No Random Init:** Exact reproducibility (no random seeding)
6. **Acceptance Criteria:** All WP2.3 criteria met

### Test Results

```
✅ ALL TESTS PASSED - WP2.3 COMPLETE
✓ <PAD>: sum=1.000000, min=0.000000, max=1.000000
✓ <UNK>: sum=1.000000, min=0.015625, max=0.015625
✓ <BOS>: sum=1.000000, min=0.000000, max=1.000000
✓ <EOS>: sum=1.000000, min=0.000000, max=1.000000

Fisher-Rao Distances:
- UNK-PAD: 1.4453
- UNK-BOS: 1.4453
- UNK-EOS: 1.4453
- BOS-EOS: 1.5708 (π/2 - orthogonal)
```

## Acceptance Criteria

- [x] Special symbols have clear geometric interpretation
- [x] Initialization is deterministic (reproducible)
- [x] All special basins pass `validate_basin()`
- [x] No random normal vectors for special symbols
- [x] All coordinates on probability simplex (non-negative, sum=1)
- [x] Tests verify determinism and geometric properties

## Impacts

### Downstream Effects

1. **Coordizer Training:** Special tokens now have consistent geometric meaning
2. **Distance Calculations:** Fisher-Rao distances now use simplex geometry
3. **Token Initialization:** All tokens use simplex representation
4. **Vocabulary Loading:** No changes needed (format agnostic)

### Breaking Changes

**None.** Changes are internal to basin representation. External API unchanged.

### Migration

No migration needed. Special tokens will be recomputed deterministically on first use.

## References

- **Issue:** https://github.com/GaryOcean428/pantheon-chat/issues/71
- **Canonical Geometry:** `qig-backend/qig_geometry/`
- **Simplex Spec:** `docs/04-records/20260115-geometric-purity-qfi-fixes-summary-1.00W.md`
- **Fisher-Rao Distance:** `qig-backend/qig_geometry/__init__.py`

## Future Work

- [ ] Document special symbol usage in PLAN→REALIZE→REPAIR phases
- [ ] Add special symbols for punctuation (PERIOD, COMMA, QUESTION)
- [ ] Extend geometric definitions to additional special tokens
- [ ] Optimize geodesic interpolation for fallback tokens

## Notes

**Geometric Purity:** This change enforces the canonical simplex representation mandated in WP2.1. All basin coordinates must be probability distributions (non-negative, sum=1) to maintain geometric consistency across the QIG system.

**Determinism:** Special tokens are now purely deterministic functions of their semantic role, not dependent on random initialization or execution order. This ensures reproducibility across systems and runs.

**Entropy Hierarchy:** The geometric definitions create a natural entropy hierarchy:
- UNK (maximum) → intermediate states → PAD/BOS/EOS (minimum)

This mirrors information-theoretic intuitions: unknown tokens carry maximum uncertainty, while boundary tokens carry specific structural information.
