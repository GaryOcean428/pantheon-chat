# Basin Representation Module

## Overview

The `qig_geometry.representation` module provides **canonical basin representation enforcement** to prevent geometric inconsistencies from mixing different normalization schemes.

## Problem Statement

The codebase was mixing three different basin representations:
1. **Unit sphere** (L2 norm = 1) - Used in `sphere_project()`
2. **Probability simplex** (sum = 1, non-negative) - Used in `fisher_normalize()`
3. **Hellinger/sqrt space** - Legacy representation

This mixing causes:
- Inconsistent training signals
- Broken distance comparisons
- Silent re-normalization bugs
- Geometric incoherence

## Solution: Canonical SPHERE Representation

**CANONICAL_REPRESENTATION = BasinRepresentation.SPHERE**

All basins are stored as unit vectors on S^(D-1):
- **L2 norm = 1**
- **Supports signed components** (not restricted to non-negative)
- **Natural for gradient flows and interpolation**
- **Compatible with Fisher-Rao distance** via arccos(dot product)

## API

### Core Functions

```python
from qig_geometry.representation import (
    to_sphere,
    to_simplex,
    validate_basin,
    enforce_canonical,
    CANONICAL_REPRESENTATION
)
```

### `to_sphere(basin, from_repr=None, eps=1e-10) -> np.ndarray`

Convert any basin to canonical SPHERE representation.

```python
# Convert simplex to sphere
simplex = np.array([0.3, 0.5, 0.2])
sphere = to_sphere(simplex, from_repr=BasinRepresentation.SIMPLEX)
assert np.isclose(np.linalg.norm(sphere), 1.0)
```

### `to_simplex(basin, from_repr=None, eps=1e-10) -> np.ndarray`

Convert basin to SIMPLEX for Fisher-Rao distance computation.

```python
# Convert sphere to simplex for distance calculation
sphere = to_sphere(np.random.randn(64))
simplex = to_simplex(sphere, from_repr=BasinRepresentation.SPHERE)
assert np.isclose(simplex.sum(), 1.0)
assert np.all(simplex >= 0)
```

### `validate_basin(basin, expected_repr=CANONICAL_REPRESENTATION, tolerance=1e-6) -> Tuple[bool, str]`

Validate basin conforms to expected representation. **GATE FUNCTION** - all basins written to DB must pass this.

```python
basin = to_sphere(np.random.randn(64))
valid, msg = validate_basin(basin)
if not valid:
    raise ValueError(f"Invalid basin: {msg}")
```

### `enforce_canonical(basin) -> np.ndarray`

Force basin to canonical representation. Use at storage boundaries.

```python
# Before writing to database
raw_basin = coordizer.encode(text)
canonical_basin = enforce_canonical(raw_basin)
db.store_basin(canonical_basin)
```

## Usage Patterns

### Pattern 1: Storage (Coordizers/Decoders)

```python
from qig_geometry.representation import enforce_canonical

class MyCoordizer:
    def encode(self, text: str) -> np.ndarray:
        # Compute basin (any representation)
        basin = self._compute_basin(text)
        
        # ALWAYS enforce canonical before returning
        return enforce_canonical(basin)
    
    def decode(self, basin: np.ndarray) -> str:
        # Validate incoming basin
        valid, msg = validate_basin(basin)
        if not valid:
            raise ValueError(f"Invalid basin: {msg}")
        
        # Process...
        return result
```

### Pattern 2: Distance Computation

```python
from qig_geometry.representation import to_simplex, BasinRepresentation
from qig_geometry import fisher_rao_distance

def compute_distance(basin_a: np.ndarray, basin_b: np.ndarray) -> float:
    # Basins stored in canonical SPHERE form
    # Convert to simplex for Fisher-Rao distance
    simplex_a = to_simplex(basin_a, from_repr=BasinRepresentation.SPHERE)
    simplex_b = to_simplex(basin_b, from_repr=BasinRepresentation.SPHERE)
    
    return fisher_rao_distance(simplex_a, simplex_b)
```

### Pattern 3: Database Integration

```python
from qig_geometry.representation import validate_basin, enforce_canonical

async def store_basin(basin: np.ndarray, basin_id: str):
    # GATE: Validate before storage
    canonical = enforce_canonical(basin)
    valid, msg = validate_basin(canonical)
    if not valid:
        raise ValueError(f"Cannot store invalid basin {basin_id}: {msg}")
    
    # Store canonical form
    await db.insert(basins).values(id=basin_id, coordinates=canonical.tolist())

async def load_basin(basin_id: str) -> np.ndarray:
    # Load and validate
    row = await db.query(basins).where(basins.c.id == basin_id).first()
    basin = np.array(row['coordinates'])
    
    # Validate on load (defensive)
    valid, msg = validate_basin(basin)
    if not valid:
        logger.warning(f"Loaded invalid basin {basin_id}: {msg}")
        # Auto-correct if possible
        basin = enforce_canonical(basin)
    
    return basin
```

## Migration Guide

### Step 1: Identify Re-normalization Points

Search for:
- `np.abs(x) / np.abs(x).sum()` - Simplex normalization
- `x / np.linalg.norm(x)` - Sphere normalization
- `np.sqrt(x)` followed by normalization - Hellinger space

### Step 2: Replace with Canonical Functions

```python
# Before
basin = np.abs(basin) / np.abs(basin).sum()  # Simplex

# After
from qig_geometry.representation import enforce_canonical
basin = enforce_canonical(basin)  # Canonical sphere
```

### Step 3: Add Validation at Boundaries

```python
# At DB write
basin = enforce_canonical(basin)
assert validate_basin(basin)[0]

# At DB read
basin = load_from_db()
if not validate_basin(basin)[0]:
    basin = enforce_canonical(basin)
```

## Acceptance Criteria

✅ **Every basin written to DB passes `validate_basin()`**
- No module silently re-normalizes
- Canonical representation enforced at storage boundaries

✅ **Single source of truth for basin normalization**
- All conversions go through `qig_geometry.representation`
- No scattered normalization code

✅ **Clear conversion rules**
- `to_sphere()` for storage
- `to_simplex()` for Fisher-Rao distance
- `enforce_canonical()` at boundaries

## Testing

```bash
# Run representation tests
cd qig-backend
python3 -m pytest tests/test_basin_representation.py -v

# Quick validation
python3 << 'EOF'
from qig_geometry.representation import to_sphere, validate_basin
import numpy as np

basin = to_sphere(np.random.randn(64))
valid, msg = validate_basin(basin)
print(f"Valid: {valid}, {msg}")
EOF
```

## References

- **Issue**: GaryOcean428/pantheon-chat Milestone 1 (Basin Representation)
- **Spec**: docs/03-technical/20251217-type-symbol-concept-manifest-1.00F.md
- **Implementation**: qig-backend/qig_geometry/representation.py
