# ISSUE 02: Strict Simplex Representation - Eliminate Auto-Detection

**Priority:** CRITICAL  
**Phase:** 2 (Geometric Purity)  
**Status:** TO DO  
**Blocks:** Fisher-Rao metric correctness, consciousness Φ validity

---

## PROBLEM STATEMENT

### Current State
Mixed representation formats (sphere vs simplex) leak into geometry calculations, causing silent errors and metric distortions.

**Evidence:**
- `to_simplex()` auto-detection silently squares data when detecting L2-normalized inputs
- "Average then L2 normalize" pattern uses sphere geodesics (great circles) instead of simplex geodesics
- Fréchet mean implementation assumes Euclidean space, not Fisher-Rao manifold
- No runtime asserts at module boundaries verifying simplex representation

**Impact:**
- Fisher-Rao distances become incorrect (sphere vs simplex geodesics differ)
- Consciousness metrics (Φ, κ) computed on wrong manifold
- Cannot trust geometric selection or basin coherence
- Silent corruption propagates through pipeline

---

## ROOT CAUSES

### 1. Auto-Detection in `to_simplex()`

**Culprit:** `qig-backend/geometry/canonical_fisher.py`

```python
# CURRENT (BROKEN):
def to_simplex(x: np.ndarray) -> np.ndarray:
    """Convert to simplex, auto-detecting representation."""
    if np.allclose(np.linalg.norm(x), 1.0):
        # SILENT CORRUPTION: Assumes L2 norm → square it
        return x ** 2
    elif np.allclose(x.sum(), 1.0):
        return x
    else:
        # Fallback: L2 normalize
        return x / np.linalg.norm(x)
```

**Problems:**
- Silent squaring changes metric structure (sphere → simplex)
- No indication of which path was taken
- Cannot distinguish intentional L2-normalized simplex from sphere vectors
- Auto-detection creates non-deterministic behavior

---

### 2. Sphere-Based Averaging

**Culprit:** `qig-backend/vocabulary/vocabulary_coordinator.py`

```python
# CURRENT (BROKEN):
def compute_average_basin(basins: List[np.ndarray]) -> np.ndarray:
    """Average multiple basins."""
    avg = np.mean(basins, axis=0)
    avg = avg / np.linalg.norm(avg)  # L2 normalize
    return to_simplex(avg)  # Triggers auto-detect
```

**Problems:**
- Euclidean mean + L2 normalization = sphere geodesic (NOT simplex)
- Should use Fréchet mean on Fisher-Rao manifold
- Introduces systematic bias toward sphere geometry

---

### 3. Missing Fréchet Mean on Simplex

**Current State:** No closed-form Fréchet mean for probability simplex.

**What's Needed:**
- Closed-form mean in sqrt-space (Hellinger distance)
- Equivalent to Fréchet mean on Fisher-Rao manifold for nearby points
- Explicit simplex → sqrt → mean → simplex pipeline

---

### 4. No Module Boundary Checks

**Current State:** No asserts verifying simplex constraints at function entry/exit.

**Missing:**
```python
# NEEDED: Runtime validation
def assert_simplex(x: np.ndarray, name: str = "array"):
    """Assert x is on probability simplex."""
    assert x.ndim == 1, f"{name} must be 1D"
    assert np.all(x >= 0), f"{name} must be non-negative"
    assert np.isclose(x.sum(), 1.0), f"{name} must sum to 1"
```

---

## REQUIRED FIXES

### Fix 1: Remove Auto-Detection from `to_simplex()`

**Update:** `qig-backend/geometry/canonical_fisher.py`

```python
# NEW: Explicit conversion with validation
def to_simplex(x: np.ndarray, validate: bool = True) -> np.ndarray:
    """
    Convert to probability simplex via L1 normalization.
    
    Args:
        x: Non-negative array (no auto-detection)
        validate: If True, raise on negative values
        
    Returns:
        Simplex vector (sums to 1, all non-negative)
        
    Raises:
        ValueError: If x contains negative values or sums to zero
    """
    if validate:
        if np.any(x < 0):
            raise ValueError(f"to_simplex requires non-negative input, got min={x.min()}")
        if np.isclose(x.sum(), 0.0):
            raise ValueError("to_simplex cannot normalize zero vector")
    
    return x / x.sum()


def to_sqrt_simplex(p: np.ndarray) -> np.ndarray:
    """
    Convert simplex vector to sqrt-space (Hellinger coordinates).
    
    Args:
        p: Probability simplex vector
        
    Returns:
        sqrt(p), which embeds simplex into sphere
    """
    assert_simplex(p, "to_sqrt_simplex input")
    return np.sqrt(p)


def from_sqrt_simplex(sqrt_p: np.ndarray) -> np.ndarray:
    """
    Convert sqrt-space vector back to simplex.
    
    Args:
        sqrt_p: sqrt(p) from sqrt-space
        
    Returns:
        p = sqrt_p^2, re-normalized to simplex
    """
    p = sqrt_p ** 2
    return to_simplex(p, validate=False)  # Re-normalize for numerical safety


def assert_simplex(x: np.ndarray, name: str = "array"):
    """Runtime validation of simplex constraints."""
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {x.shape}")
    if np.any(x < 0):
        raise ValueError(f"{name} must be non-negative, got min={x.min()}")
    if not np.isclose(x.sum(), 1.0):
        raise ValueError(f"{name} must sum to 1, got sum={x.sum()}")
```

**Enforcement:**
- ALL functions accepting basins MUST call `assert_simplex()` at entry
- NO auto-detection logic anywhere in codebase
- Explicit `to_sqrt_simplex()` / `from_sqrt_simplex()` for averaging

---

### Fix 2: Implement Closed-Form Simplex Mean

**Create:** `qig-backend/geometry/simplex_mean.py`

```python
"""
Closed-form Fréchet mean on probability simplex.

Uses sqrt-space (Hellinger) representation for averaging, which approximates
the Fréchet mean on the Fisher-Rao manifold for nearby points.
"""

import numpy as np
from .canonical_fisher import assert_simplex, to_sqrt_simplex, from_sqrt_simplex


def simplex_mean(simplices: List[np.ndarray]) -> np.ndarray:
    """
    Compute closed-form mean on probability simplex.
    
    Algorithm:
    1. Transform to sqrt-space: sqrt(p_i)
    2. Average in sqrt-space (Euclidean mean)
    3. Transform back: (mean)^2 → simplex
    
    This is the Fréchet mean for Hellinger distance, which approximates
    the Fisher-Rao Fréchet mean for nearby points.
    
    Args:
        simplices: List of probability simplex vectors
        
    Returns:
        Mean simplex vector
    """
    if not simplices:
        raise ValueError("Cannot compute mean of empty list")
    
    # Validate inputs
    for i, p in enumerate(simplices):
        assert_simplex(p, f"simplex[{i}]")
    
    # Transform to sqrt-space
    sqrt_simplices = [to_sqrt_simplex(p) for p in simplices]
    
    # Average in sqrt-space (Euclidean)
    sqrt_mean = np.mean(sqrt_simplices, axis=0)
    
    # Transform back to simplex
    return from_sqrt_simplex(sqrt_mean)


def weighted_simplex_mean(
    simplices: List[np.ndarray], 
    weights: np.ndarray
) -> np.ndarray:
    """
    Weighted mean on probability simplex.
    
    Args:
        simplices: List of probability simplex vectors
        weights: Non-negative weights (will be normalized)
        
    Returns:
        Weighted mean simplex vector
    """
    if not simplices:
        raise ValueError("Cannot compute mean of empty list")
    if len(simplices) != len(weights):
        raise ValueError("Number of weights must match number of simplices")
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Validate inputs
    for i, p in enumerate(simplices):
        assert_simplex(p, f"simplex[{i}]")
    
    # Transform to sqrt-space
    sqrt_simplices = [to_sqrt_simplex(p) for p in simplices]
    
    # Weighted average in sqrt-space
    sqrt_mean = np.average(sqrt_simplices, axis=0, weights=weights)
    
    # Transform back to simplex
    return from_sqrt_simplex(sqrt_mean)
```

**Usage:**
```python
# Replace all "average basins" logic with:
from qig_backend.geometry.simplex_mean import simplex_mean

avg_basin = simplex_mean([basin1, basin2, basin3])
```

---

### Fix 3: Add Module Boundary Asserts

**Update ALL functions in:**
- `qig-backend/geometry/canonical_fisher.py`
- `qig-backend/geometry/fisher_rao_distance.py`
- `qig-backend/vocabulary/vocabulary_coordinator.py`

**Pattern:**
```python
def quantum_fisher_information(basin: np.ndarray) -> float:
    """Compute QFI for a basin."""
    assert_simplex(basin, "quantum_fisher_information.basin")
    
    # ... computation ...
    
    return qfi_score


def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Fisher-Rao distance between two simplices."""
    assert_simplex(p, "fisher_rao_distance.p")
    assert_simplex(q, "fisher_rao_distance.q")
    
    # ... computation ...
    
    return distance
```

**Enforcement:**
- ALL geometry functions MUST assert simplex at entry
- ALL vocabulary functions returning basins MUST assert at exit
- NO exceptions (even for "performance" reasons)

---

### Fix 4: Replace All Sphere-Based Averaging

**Scan for pattern:**
```bash
# Find all sphere-based normalization patterns
grep -r "np.linalg.norm" qig-backend/ | grep -v "test"
grep -r "/ norm" qig-backend/ | grep -v "test"
```

**Replace:**
```python
# BEFORE (BROKEN):
avg = np.mean(basins, axis=0)
avg = avg / np.linalg.norm(avg)
return to_simplex(avg)

# AFTER (CORRECT):
from qig_backend.geometry.simplex_mean import simplex_mean
return simplex_mean(basins)
```

**Locations:**
- `vocabulary_coordinator.py`: `compute_average_basin()`
- Any other basin averaging logic

---

### Fix 5: Migration Path for Existing Data

**Script:** `scripts/validate_simplex_storage.py`

```python
#!/usr/bin/env python3
"""
Validate that all stored basins are on probability simplex.
"""

import asyncio
import numpy as np
from qig_backend.database import get_db
from qig_backend.geometry.canonical_fisher import assert_simplex

async def validate_storage():
    """Check all basins in database for simplex validity."""
    db = await get_db()
    
    query = """
        SELECT token, basin_embedding
        FROM coordizer_vocabulary
        WHERE basin_embedding IS NOT NULL
    """
    rows = await db.fetch_all(query)
    
    print(f"Validating {len(rows)} basins...")
    
    valid = 0
    invalid = []
    
    for row in rows:
        token = row['token']
        basin = np.array(row['basin_embedding'])
        
        try:
            assert_simplex(basin, token)
            valid += 1
        except (ValueError, AssertionError) as e:
            invalid.append((token, str(e)))
    
    print(f"\nValidation complete:")
    print(f"  Valid: {valid}")
    print(f"  Invalid: {len(invalid)}")
    
    if invalid:
        print("\nFirst 10 invalid basins:")
        for token, error in invalid[:10]:
            print(f"  {token}: {error}")
        
        print("\nRe-normalizing invalid basins...")
        for token, _ in invalid:
            await db.execute("""
                UPDATE coordizer_vocabulary
                SET basin_embedding = basin_embedding / (
                    SELECT SUM(unnest) FROM unnest(basin_embedding)
                )
                WHERE token = %s
            """, (token,))
        
        print("Re-normalization complete.")

if __name__ == "__main__":
    asyncio.run(validate_storage())
```

---

## ACCEPTANCE CRITERIA

### AC1: No Auto-Detection
- [ ] `to_simplex()` removes auto-detection logic
- [ ] `to_simplex()` raises on negative inputs
- [ ] `to_sqrt_simplex()` / `from_sqrt_simplex()` implemented
- [ ] No "L2 norm detect → square" logic anywhere

### AC2: Closed-Form Mean
- [ ] `simplex_mean.py` implemented
- [ ] `simplex_mean()` uses sqrt-space transformation
- [ ] `weighted_simplex_mean()` supports weights
- [ ] All averaging code uses `simplex_mean()`

### AC3: Runtime Asserts
- [ ] `assert_simplex()` implemented
- [ ] ALL geometry functions assert at entry
- [ ] ALL vocabulary functions assert at exit
- [ ] Assertions enabled in production (not stripped)

### AC4: No Sphere Averaging
- [ ] No `np.linalg.norm()` in basin averaging code
- [ ] No "average then L2 normalize" pattern
- [ ] All `compute_average_basin()` uses `simplex_mean()`

### AC5: Data Validated
- [ ] All stored basins pass `assert_simplex()`
- [ ] Invalid basins re-normalized
- [ ] Validation script in `scripts/`

---

## VALIDATION TESTS

```python
# Test 1: to_simplex rejects negatives
def test_to_simplex_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        to_simplex(np.array([1.0, -0.5, 0.5]))

# Test 2: Sqrt-space roundtrip
def test_sqrt_space_roundtrip():
    p = np.array([0.5, 0.3, 0.2])
    sqrt_p = to_sqrt_simplex(p)
    p_recovered = from_sqrt_simplex(sqrt_p)
    
    assert np.allclose(p, p_recovered)
    assert_simplex(p_recovered, "recovered")

# Test 3: Simplex mean preserves simplex
def test_simplex_mean_validity():
    p1 = np.array([0.5, 0.3, 0.2])
    p2 = np.array([0.4, 0.4, 0.2])
    p3 = np.array([0.6, 0.2, 0.2])
    
    mean = simplex_mean([p1, p2, p3])
    assert_simplex(mean, "mean")

# Test 4: Asserts catch violations
def test_assert_simplex_catches_violations():
    # Negative values
    with pytest.raises(ValueError, match="non-negative"):
        assert_simplex(np.array([0.5, -0.1, 0.6]))
    
    # Doesn't sum to 1
    with pytest.raises(ValueError, match="sum to 1"):
        assert_simplex(np.array([0.5, 0.3, 0.1]))
    
    # Wrong dimension
    with pytest.raises(ValueError, match="1D"):
        assert_simplex(np.array([[0.5, 0.5], [0.3, 0.7]]))

# Test 5: No sphere averaging in codebase
def test_no_sphere_averaging():
    """Ensure no L2 normalization in averaging code."""
    import ast
    import inspect
    
    from qig_backend.vocabulary.vocabulary_coordinator import compute_average_basin
    
    source = inspect.getsource(compute_average_basin)
    tree = ast.parse(source)
    
    # Check for np.linalg.norm
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr == "norm":
                pytest.fail("Found np.linalg.norm in compute_average_basin")
```

---

## MIGRATION PATH

### Step 1: Add New Functions (Non-Breaking)
- Implement `to_sqrt_simplex()` / `from_sqrt_simplex()`
- Implement `simplex_mean.py`
- Implement `assert_simplex()`

### Step 2: Add Validation (Breaking for Invalid Data)
- Add asserts to ALL geometry functions
- Run `validate_simplex_storage.py` to fix existing data
- Add pre-commit hook enforcing asserts

### Step 3: Replace Averaging Logic (Breaking Change)
- Replace all `np.mean() + L2 normalize` with `simplex_mean()`
- Remove auto-detection from `to_simplex()`
- Update all call sites to use explicit conversions

### Step 4: Enforce in CI
- Add test for no sphere averaging patterns
- Add test for assert coverage (all geometry functions)
- Block PRs with auto-detection logic

---

## REFERENCES

- **Fisher-Rao Manifold:** `docs/pantheon_e8_upgrade_pack/ULTRA_CONSCIOUSNESS_PROTOCOL_v4_0_UNIVERSAL.md`
- **Canonical Geometry:** `qig-backend/geometry/canonical_fisher.py`
- **Hellinger Distance:** Equivalent to L2 distance in sqrt-space
- **Fréchet Mean:** Minimizer of sum of squared Fisher-Rao distances

---

**Last Updated:** 2026-01-16  
**Estimated Effort:** 3-4 days (includes data validation and averaging replacement)  
**Priority:** CRITICAL - Blocks metric correctness
