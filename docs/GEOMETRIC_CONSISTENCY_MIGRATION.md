# Geometric Consistency Migration Guide

**Date**: 2026-01-15  
**Version**: 1.0  
**Status**: CRITICAL - BREAKING CHANGES

## 🚨 Summary of Changes

This migration addresses **geometric chaos** caused by mixed usage of:
1. SPHERE vs SIMPLEX canonical representations
2. Hellinger embedding (factor of 2) vs direct Fisher-Rao
3. Inconsistent distance formulas across the codebase

## Simplex Only below is for reference, do not use sphere. or ay other method. 

### What Changed

| Component | Before | After |
|-----------|--------|-------|
| **Canonical Representation** | SPHERE (L2 norm=1) | SIMPLEX (sum=1, non-negative) |
| **Fisher-Rao Distance** | `2*arccos(BC)` | `arccos(BC)` |
| **Distance Range** | [0, π] | [0, π/2] |
| **Hellinger Embedding** | Used (with factor 2) | Removed |

---

## 📋 Required Code Changes

### 1. Import Updates

**Before:**
```python
from qig_geometry import sphere_project
basin = sphere_project(raw_vector)
```

**After:**
```python
from qig_geometry import fisher_normalize
basin = fisher_normalize(raw_vector)  # Simplex normalization
```

### 2. Distance Calculation

**No API changes** - but distance values will be **different**:

```python
from qig_geometry import fisher_rao_distance

p = fisher_normalize(basin_a)
q = fisher_normalize(basin_b)
d = fisher_rao_distance(p, q)  # Now returns [0, π/2], was [0, π]
```

### 3. Threshold Recalibration

**All distance thresholds must be divided by 2:**

```python
# Before (Hellinger with factor of 2)
SIMILARITY_THRESHOLD = 0.8  # For distance < 0.8
MAX_DISTANCE = 3.0

# After (Direct Fisher-Rao)
SIMILARITY_THRESHOLD = 0.4  # For distance < 0.4 (0.8 / 2)
MAX_DISTANCE = 1.5  # (3.0 / 2, but max possible is π/2 ≈ 1.57)
```

### 4. Validation Updates

**Before:**
```python
from qig_geometry import validate_basin, BasinRepresentation
valid, msg = validate_basin(basin, BasinRepresentation.SPHERE)
```

**After:**
```python
from qig_geometry import validate_basin, BasinRepresentation
valid, msg = validate_basin(basin, BasinRepresentation.SIMPLEX)
# Or use the canonical default:
valid, msg = validate_basin(basin)  # Defaults to SIMPLEX
```

---

## 🔍 Files Requiring Updates

### Critical (Breaks consciousness metrics)

1. **qig-backend/working_memory_bus.py**
   - Line with `2.0 * np.arccos(bc)` → remove factor of 2
   - Update comment about Hellinger embedding

2. **qig-backend/pattern_response_generator.py**
   - Fisher-Rao distance formula
   - Remove Hellinger factor comment

3. **qig-backend/qiggraph/consciousness.py**
   - Fallback Fisher-Rao implementation
   - Remove factor of 2

4. **qig-backend/qig_core/geometric_primitives/geodesic.py**
   - Geodesic length computation
   - Remove factor of 2

5. **qig-backend/qig_core/geometric_primitives/fisher_metric.py**
   - Re-ranking distance calculation
   - Remove factor of 2

### Important (Affect calculations)

6. **qig-backend/training/coherence_evaluator.py**
   - Coherence thresholds need recalibration
   
7. **qig-backend/olympus/hermes_coordinator.py**
   - Fallback distance function

8. **qig-backend/autonomic_agency/state_encoder.py**
   - Basin drift calculation

9. **qig-backend/self_healing/geometric_monitor.py**
   - Basin distance thresholds

10. **qig-backend/qig_geometry.py** (if exists as separate file)
    - Coordinate distance formula

---

## 🎯 Search and Replace Patterns

### Pattern 1: Remove Factor of 2 from Fisher-Rao

**Search:**
```python
2.0 * np.arccos(
```

**Replace:**
```python
np.arccos(
```

**AND update comments:**
```python
# OLD COMMENT
# Factor of 2 for Hellinger embedding

# NEW COMMENT  
# Direct Fisher-Rao distance on simplex
```

### Pattern 2: Replace sphere_project with fisher_normalize

**Search:**
```python
sphere_project(
```

**Replace:**
```python
fisher_normalize(
```

### Pattern 3: Update CANONICAL_REPRESENTATION references

**Search:**
```python
CANONICAL_REPRESENTATION == BasinRepresentation.SPHERE
```

**Replace:**
```python
CANONICAL_REPRESENTATION == BasinRepresentation.SIMPLEX
```

---

## 🧪 Validation Tests

After migration, run these tests:

```python
import numpy as np
from qig_geometry import (
    fisher_rao_distance,
    fisher_normalize,
    CANONICAL_REPRESENTATION,
    BasinRepresentation
)

# Test 1: Verify canonical is SIMPLEX
assert CANONICAL_REPRESENTATION == BasinRepresentation.SIMPLEX

# Test 2: Verify distance range
p = fisher_normalize(np.random.randn(64))
q = fisher_normalize(np.random.randn(64))
d = fisher_rao_distance(p, q)
assert 0 <= d <= np.pi/2, f"Distance out of range: {d}"

# Test 3: Identity distance
d_identity = fisher_rao_distance(p, p)
assert d_identity < 1e-10, f"Identity distance not zero: {d_identity}"

# Test 4: Symmetry
assert np.isclose(
    fisher_rao_distance(p, q),
    fisher_rao_distance(q, p)
), "Distance not symmetric"

# Test 5: Triangle inequality
r = fisher_normalize(np.random.randn(64))
d_pq = fisher_rao_distance(p, q)
d_qr = fisher_rao_distance(q, r)
d_pr = fisher_rao_distance(p, r)
assert d_pr <= d_pq + d_qr + 1e-6, "Triangle inequality violated"
```

---

## ⚠️ Known Impact Areas

### Consciousness Metrics (Critical)

- **Φ (Integration)**: May show different values due to distance changes
- **κ (Coupling)**: Calculations affected by distance formulas
- **Γ (Coherence)**: Basin drift thresholds need recalibration

**Action**: Re-measure baseline consciousness metrics after migration.

### Distance Thresholds (Critical)

All hardcoded distance thresholds in:
- Attention mechanisms
- Basin clustering
- Similarity matching
- Kernel routing

**Action**: Divide all existing thresholds by 2 as starting point, then tune.

### Database Values (Moderate)

Existing basin coordinates in database:
- Should be migrated to simplex representation
- Or conversion applied on read (see next section)

---

## 🔄 Database Migration Strategy

### Option A: Convert on Read (Safer)

```python
def load_basin_from_db(basin_bytes):
    """Load basin with automatic conversion."""
    basin = np.frombuffer(basin_bytes, dtype=np.float32)
    
    # Detect if it's in old SPHERE format
    if np.isclose(np.linalg.norm(basin), 1.0):
        # Convert SPHERE → SIMPLEX
        return fisher_normalize(basin)
    
    # Already in simplex or needs normalization
    return fisher_normalize(basin)
```

### Option B: Batch Migration (More Thorough)

```sql
-- Add temporary column
ALTER TABLE coordizer_vocabulary 
ADD COLUMN basin_coords_simplex vector(64);

-- Python script to convert all basins
-- (Run this offline)
UPDATE coordizer_vocabulary 
SET basin_coords_simplex = convert_to_simplex(basin_coords);

-- Swap columns
ALTER TABLE coordizer_vocabulary DROP COLUMN basin_coords;
ALTER TABLE coordizer_vocabulary 
RENAME COLUMN basin_coords_simplex TO basin_coords;
```

---

## 📊 Rollback Plan

If migration causes critical issues:

1. **Revert branch**: `git revert <commit-sha>`
2. **Restore SPHERE canonical**:
   ```python
   CANONICAL_REPRESENTATION = BasinRepresentation.SPHERE
   ```
3. **Re-add factor of 2**:
   ```python
   return float(2.0 * np.arccos(bc))
   ```
4. **Document reason** in issue tracker

---

## ✅ Migration Checklist

### Pre-Migration

- [ ] Review all files in "Files Requiring Updates" section
- [ ] Create database backup
- [ ] Document current consciousness metric baselines
- [ ] Review all hardcoded distance thresholds

### During Migration

- [ ] Update qig_geometry/representation.py ✅ (Done)
- [ ] Update qig_geometry/__init__.py ✅ (Done)
- [ ] Search for `2.0 * np.arccos` and remove factor of 2
- [ ] Search for `sphere_project` and replace with `fisher_normalize`
- [ ] Update all distance threshold values
- [ ] Run validation tests

### Post-Migration

- [ ] Measure new consciousness metric baselines
- [ ] Compare with pre-migration baselines
- [ ] Tune thresholds based on new metrics
- [ ] Update documentation
- [ ] Monitor production metrics for 48 hours
- [ ] If stable, close migration issues

---

## 📚 Related Issues

- #68: Create Single Canonical qig_geometry Module
- #69: Remove Cosine Similarity from match_coordinates()
- #92: Remove Frequency-Based Stopwords

## 📖 References

- `qig-backend/qig_geometry/representation.py` - Canonical representation
- `qig-backend/qig_geometry/__init__.py` - Distance formulas
- `20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md` - QIG consciousness metrics

---

**Migration Guide Version 1.0** - 2026-01-15
