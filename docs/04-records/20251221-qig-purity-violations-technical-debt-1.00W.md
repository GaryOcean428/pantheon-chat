# QIG Purity Violations - Technical Debt

**Document Type:** Record  
**Status:** Working (1.00W)  
**Date:** 2025-12-21  
**Priority:** Medium (Phase 2 cleanup)

## Executive Summary

QIG purity checker identified **56 violations** of geometric principles in the Python backend. These violations use Euclidean distance operations (`np.linalg.norm`, `np.dot`) on basin coordinates instead of the required Fisher-Rao distance.

**Impact:** Medium - System functions but uses incorrect geometric distance calculations in some paths.

**Recommendation:** Address in Phase 2 after architecture stabilization. Requires systematic refactoring of geometric operations.

## QIG Purity Principles (Immutable)

The following operations are **FORBIDDEN** on basin coordinates:

1. ❌ `np.linalg.norm(basin)` - Euclidean norm
2. ❌ `np.dot(basin1, basin2)` - Euclidean dot product
3. ❌ `torch.norm(basin)` - PyTorch norm
4. ❌ Cosine similarity on basin coordinates
5. ❌ L2 distance on basin coordinates

**Required instead:**
- ✅ `fisher_rao_distance(basin1, basin2)` - Proper geometric distance
- ✅ Bhattacharyya coefficient for similarity
- ✅ Natural gradient descent on manifold

## Violation Summary

### By File (Top Offenders)

| File | Violations | Type |
|------|-----------|------|
| `conversational_kernel.py` | 8 | norm, dot |
| `ocean_qig_core.py` | 4 | norm |
| `qig_tokenizer.py` | 3 | norm, dot |
| `qig_tokenizer_postgresql.py` | 2 | norm |
| `pantheon_kernel_orchestrator.py` | 2 | norm |
| `geometric_kernels.py` | 1 | norm, dot |
| `training_chaos/chaos_kernel.py` | 2 | torch.norm |
| Others | 34 | various |

### By Violation Type

| Operation | Count | Severity |
|-----------|-------|----------|
| `np.linalg.norm(basin)` | 42 | High |
| `np.dot(basin1, basin2)` | 8 | High |
| `torch.norm(basin)` | 2 | Medium |
| Normalization via Euclidean | 4 | High |

## Detailed Violations

### High Priority (Core QIG Modules)

#### `ocean_qig_core.py` (Line 886-887)
```python
# VIOLATION: Euclidean normalization
query_norm = query_basin / (np.linalg.norm(query_basin) + 1e-10)
concept_norm = concept_basin / (np.linalg.norm(concept_basin) + 1e-10)
```

**Fix:**
```python
# Use Fisher-Rao distance instead
similarity = fisher_rao_distance(query_basin, concept_basin)
```

#### `conversational_kernel.py` (Line 262-264)
```python
# VIOLATION: Cosine similarity via Euclidean norm + dot
basin_norm = basin / (np.linalg.norm(basin) + 1e-10)
token_norm = token_basin / (np.linalg.norm(token_basin) + 1e-10)
dot = np.clip(np.dot(basin_norm, token_norm), -1.0, 1.0)
```

**Fix:**
```python
# Use Fisher-Rao distance
similarity = fisher_rao_distance(basin, token_basin)
```

#### `qig_tokenizer.py` (Line 703, 1024)
```python
# VIOLATION 1: Euclidean normalization
return weighted_basin / (np.linalg.norm(weighted_basin) + 1e-8)

# VIOLATION 2: Dot product for alignment
alignment_bonus = float(np.dot(context_basin, token_basin))
```

**Fix:**
```python
# Normalization on manifold
normalized_basin = normalize_on_manifold(weighted_basin)

# Fisher distance for alignment
alignment = 1.0 - fisher_rao_distance(context_basin, token_basin)
```

### Medium Priority (Utility Functions)

#### `geometric_kernels.py` (Line 61)
```python
# VIOLATION: Cosine similarity
dot = np.clip(np.dot(basin1, basin2) / (np.linalg.norm(basin1) * np.linalg.norm(basin2)), -1.0, 1.0)
```

#### `soft_reset.py` (Line 136-137)
```python
# VIOLATION: Euclidean normalization for comparison
a_norm = current_basin_arr / (np.linalg.norm(current_basin_arr) + 1e-10)
b_norm = self.reference_basin / (np.linalg.norm(self.reference_basin) + 1e-10)
```

### Low Priority (Monitoring/Logging)

#### `pantheon_kernel_orchestrator.py` (Line 401, 535)
```python
# VIOLATION: Norm for telemetry
"token_basin_norm": float(np.linalg.norm(token_basin)),
"basin_norm": float(np.linalg.norm(basin)),
```

**Fix:** These can remain for debugging, but add comment:
```python
# NOTE: Euclidean norm for monitoring only - not used in geometric ops
"basin_magnitude": float(np.linalg.norm(basin)),  # Debug only
```

## Recommended Fix Strategy

### Phase 1: Create Geometric Utilities (Week 1)

Create `qig-backend/qig_core/geometric_ops.py`:

```python
"""
Pure Geometric Operations Module
=================================

All geometric operations on basin coordinates MUST use these functions.
NEVER use np.linalg.norm or np.dot directly on basins.
"""

import numpy as np
from typing import Tuple

def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Fisher-Rao distance between probability distributions.
    
    Uses Bhattacharyya coefficient: d_FR = arccos(∑√(p_i * q_i))
    """
    # Ensure normalized (probability distributions)
    p = p / (np.sum(p) + 1e-10)
    q = q / (np.sum(q) + 1e-10)
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, 0.0, 1.0)
    
    # Fisher-Rao distance
    return float(np.arccos(bc))

def normalize_on_manifold(basin: np.ndarray) -> np.ndarray:
    """
    Normalize basin as probability distribution (statistical manifold).
    """
    return basin / (np.sum(basin) + 1e-10)

def basin_similarity(p: np.ndarray, q: np.ndarray) -> float:
    """
    Similarity in [0, 1] based on Fisher-Rao distance.
    """
    distance = fisher_rao_distance(p, q)
    max_distance = np.pi / 2  # Maximum Fisher-Rao distance
    return 1.0 - (distance / max_distance)

def geodesic_interpolation(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    """
    Geodesic interpolation on manifold (for basin smoothing).
    """
    p_norm = normalize_on_manifold(p)
    q_norm = normalize_on_manifold(q)
    
    # Geometric interpolation on simplex
    interpolated = (1 - t) * p_norm + t * q_norm
    return normalize_on_manifold(interpolated)
```

### Phase 2: Systematic Refactoring (Week 2-3)

Replace violations in priority order:
1. Core QIG modules (`ocean_qig_core.py`, `qig_geometry.py`)
2. Kernel implementations (`conversational_kernel.py`, `geometric_kernels.py`)
3. Tokenizers (`qig_tokenizer.py`, `qig_tokenizer_postgresql.py`)
4. Utility modules

### Phase 3: Testing (Week 4)

1. Unit tests for `geometric_ops.py`
2. Integration tests comparing old vs new results
3. Φ/κ validation on test corpus
4. Performance benchmarking

### Phase 4: Documentation (Week 4)

1. Update QIG principles doc
2. Add geometric operations guide
3. Create pre-commit hook to catch violations

## Testing Requirements

Before merging fixes:

1. **Functional:** All existing tests pass
2. **Geometric:** Φ/κ metrics remain in valid ranges
3. **Performance:** <10% slowdown acceptable
4. **Correctness:** Fisher-Rao distances validated against known examples

## Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: qig-purity
      name: QIG Geometric Purity Check
      entry: python tools/qig_purity_check.py
      language: system
      pass_filenames: false
      always_run: true
```

## References

- **QIG Principles:** `docs/03-technical/qig-consciousness/20251208-qig-principles-quantum-geometry-1.00F.md`
- **Fisher Information:** Amari, S. (2016). Information Geometry and Its Applications
- **Purity Checker:** `tools/qig_purity_check.py`

## Acceptance Criteria

This technical debt is resolved when:
- [ ] All 56 violations fixed
- [ ] `qig_purity_check.py` reports 0 violations
- [ ] All tests pass
- [ ] Pre-commit hook enforces purity
- [ ] Documentation updated
- [ ] Performance benchmarks acceptable

---

**Last Updated:** 2025-12-21  
**Owner:** Engineering Team  
**Review Frequency:** Weekly until resolved, then annual
