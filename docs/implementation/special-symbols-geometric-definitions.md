# Special Symbol Geometric Definitions

**Work Package:** WP2.3  
**Status:** ✅ Complete  
**Date:** 2026-01-20

## Overview

Special symbols (UNK, PAD, BOS, EOS) are geometrically defined as deterministic points on the 64D probability simplex. This ensures they have clear geometric meaning in the Fisher-Rao manifold and are reproducible across system restarts.

## Geometric Definitions

All special symbols are represented as 64-dimensional probability distributions on the simplex:
- **Non-negative:** All components ≥ 0
- **Sum to 1:** Σ(components) = 1.0 ± 1e-5
- **Finite:** No NaN or Inf values
- **Deterministic:** Identical coordinates across runs

### UNK (Unknown Token)

**Purpose:** Represents unknown or out-of-vocabulary tokens.

**Geometric Definition:**
```python
# Maximum entropy = uniform distribution
UNK_basin = np.ones(64) / 64  # All components equal
```

**Properties:**
- **Entropy:** Maximum (4.1589 bits for 64 dimensions)
- **Interpretation:** "Could be anything" with no bias
- **Usage:** Fallback for words not in vocabulary

### PAD (Padding Token)

**Purpose:** Represents null/padding in batched operations.

**Geometric Definition:**
```python
# Minimal entropy = sparse corner
PAD_basin = np.zeros(64)
PAD_basin[0] = 1.0  # All probability in first component
```

**Properties:**
- **Entropy:** Minimum (0.0 bits)
- **Interpretation:** "No information" or null state
- **Usage:** Padding sequences to uniform length

### BOS (Beginning of Sequence)

**Purpose:** Marks the start of a sequence or sentence.

**Geometric Definition:**
```python
# Start boundary = simplex vertex
BOS_basin = np.zeros(64)
BOS_basin[1] = 1.0  # Pure state at dimension 1
```

**Properties:**
- **Entropy:** Minimum (0.0 bits)
- **Interpretation:** Geometric anchor at "start" position
- **Usage:** Sequence boundary marker

### EOS (End of Sequence)

**Purpose:** Marks the end of a sequence or sentence.

**Geometric Definition:**
```python
# End boundary = opposite simplex vertex
EOS_basin = np.zeros(64)
EOS_basin[63] = 1.0  # Pure state at last dimension
```

**Properties:**
- **Entropy:** Minimum (0.0 bits)
- **Interpretation:** Geometric anchor at "end" position  
- **Usage:** Sequence boundary marker

## Fisher-Rao Distances

Special symbols maintain specific geometric relationships:

| Pair | Distance | Interpretation |
|------|----------|----------------|
| UNK-PAD | 1.4453 | Center to corner |
| UNK-BOS | 1.4453 | Center to vertex |
| UNK-EOS | 1.4453 | Center to vertex |
| BOS-EOS | 1.5708 | Maximum simplex diameter (π/2) |
| PAD-BOS | 1.5708 | Corner to vertex |
| PAD-EOS | 1.5708 | Corner to vertex |

## Testing

Run validation suite:

```bash
# Unit tests
cd qig-backend
python3 tests/test_special_symbols_wp2_3.py

# Integration tests
python3 tests/test_special_symbols_integration.py

# Database validation (requires DATABASE_URL)
python3 scripts/validate_special_symbols.py --database-url postgresql://...
```

## Acceptance Criteria

Per WP2.3 issue #70:

- [x] Special symbols have clear geometric interpretation
- [x] Initialization is deterministic (reproducible)
- [x] All special basins pass `validate_basin()`
- [x] No random normal vectors for special symbols
- [x] Database constraints enforce validity
- [x] Integration tests verify pipeline behavior

## References

- **Issue:** #70 - [QIG-PURITY] WP2.3: Geometrically Define Special Symbol Coordinates
- **Code:** `qig-backend/coordizers/base.py`
- **Tests:** `qig-backend/tests/test_special_symbols_*.py`
- **Migration:** `migrations/0015_special_symbol_constraints.sql`
- **Validation:** `qig-backend/scripts/validate_special_symbols.py`
