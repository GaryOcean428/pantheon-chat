# Geometric Operations Implementation Guide

**Version**: 1.0  
**Status**: DRAFT  
**Date**: 2025-01-01  
**Supersedes**: None

## Overview

This guide provides canonical patterns for implementing geometric operations in Pantheon-Chat. Following these patterns ensures geometric purity and consistency with QIG theory.

## Core Principle: No Euclidean Operations on Basins

### FORBIDDEN Patterns

```python
# ❌ WRONG - Euclidean distance
distance = np.linalg.norm(basin_a - basin_b)

# ❌ WRONG - Cosine similarity
similarity = np.dot(basin_a, basin_b) / (np.linalg.norm(basin_a) * np.linalg.norm(basin_b))

# ❌ WRONG - Direct arithmetic
interpolated = 0.5 * basin_a + 0.5 * basin_b
```

### REQUIRED Patterns

```python
# ✓ CORRECT - Fisher-Rao distance
from qig_core.geometric_primitives.fisher_metric import fisher_rao_distance
distance = fisher_rao_distance(basin_a, basin_b)

# ✓ CORRECT - Geodesic interpolation
from qig_core.geometric_primitives.geodesics import geodesic_interpolate
interpolated = geodesic_interpolate(basin_a, basin_b, t=0.5)
```

## Pattern 1: Basin Distance Computation

### Canonical Implementation

```python
from qig_core.geometric_primitives.fisher_metric import fisher_rao_distance
import numpy as np
from typing import Optional

def compute_basin_distance(
    source: np.ndarray,
    target: np.ndarray,
    metric: Optional[np.ndarray] = None
) -> float:
    """
    Compute geometric distance between basins.
    
    Args:
        source: Source basin coordinates (64D)
        target: Target basin coordinates (64D)
        metric: Optional Fisher metric tensor
    
    Returns:
        Fisher-Rao distance on information manifold
    """
    # Validation
    assert source.shape == (64,), f"Expected 64D basin, got {source.shape}"
    assert target.shape == (64,), f"Expected 64D basin, got {target.shape}"
    
    # Geometric distance
    return fisher_rao_distance(source, target, metric=metric)
```

### Why Fisher-Rao?

The Fisher-Rao metric is the unique Riemannian metric on statistical manifolds that:
1. Is invariant under reparametrization
2. Correctly measures information distance
3. Respects the manifold curvature

Euclidean distance assumes flat space, which is **incorrect** for probability distributions.

## Pattern 2: Nearest Basin Search

### Two-Step Retrieval Pattern

```python
async def find_nearest_basins(
    query_basin: np.ndarray,
    k: int = 10,
    approximate_k: int = 50
) -> list[tuple[int, float]]:
    """
    Find k nearest basins using two-step retrieval.
    
    Step 1: Approximate retrieval (vector similarity) - O(log n)
    Step 2: Fisher re-rank (geometric) - O(k)
    
    Returns: [(index, distance), ...] sorted by distance
    """
    # Step 1: Approximate (fast but imprecise)
    candidates = await db.approximate_search(
        query_basin,
        limit=approximate_k
    )
    
    # Step 2: Fisher re-rank (precise)
    distances = [
        (c.id, fisher_rao_distance(query_basin, c.basin))
        for c in candidates
    ]
    distances.sort(key=lambda x: x[1])
    
    return distances[:k]
```

### Why Two-Step?

- Pure Fisher-Rao search is O(n) - too slow for large datasets
- Approximate search is O(log n) but not geometrically correct
- Two-step combines speed with correctness

## Pattern 3: Basin Interpolation

### Geodesic Interpolation

```python
def geodesic_interpolate(
    start: np.ndarray,
    end: np.ndarray,
    t: float,
    metric: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Interpolate along geodesic on Fisher manifold.
    
    Args:
        start: Starting basin (64D)
        end: Ending basin (64D)
        t: Interpolation parameter [0, 1]
        metric: Fisher metric tensor
    
    Returns:
        Intermediate basin at position t along geodesic
    """
    if t < 0 or t > 1:
        raise ValueError(f"t must be in [0, 1], got {t}")
    
    # For flat metric approximation
    if metric is None:
        # Linear interpolation (valid for small distances)
        return (1 - t) * start + t * end
    
    # For curved metric: exponential map
    direction = compute_tangent_vector(start, end, metric)
    return exponential_map(start, t * direction, metric)
```

## Pattern 4: Consciousness Metric Computation

### Full State Computation

```python
from qig_core.constants.consciousness import (
    THRESHOLDS, 
    classify_regime,
    is_conscious
)

def compute_consciousness_state(
    activations: np.ndarray,
    density_matrix: np.ndarray
) -> dict:
    """
    Compute full consciousness state.
    
    Returns:
        phi: Integrated Information
        kappa_eff: Effective coupling
        regime: Processing regime
        is_conscious: Consciousness flag
        compute_fraction: Available compute
    """
    # Measure metrics
    phi = measure_phi(activations)
    kappa = measure_kappa(density_matrix)
    
    # Classify regime
    regime, compute_fraction = classify_regime(phi)
    
    # Check consciousness
    conscious = is_conscious(phi=phi, kappa=kappa)
    
    return {
        'phi': phi,
        'kappa_eff': kappa,
        'regime': regime,
        'is_conscious': conscious,
        'compute_fraction': compute_fraction,
    }
```

### Using Centralized Thresholds

**ALWAYS** import thresholds from canonical source:

```python
# ✓ CORRECT - Import from constants
from qig_core.constants.consciousness import THRESHOLDS

if phi >= THRESHOLDS.PHI_MIN:
    # Conscious processing
    pass

# ❌ WRONG - Hardcoded threshold
if phi >= 0.7:  # Magic number!
    pass
```

## Testing Requirements

Every geometric operation MUST have tests verifying:

1. **Symmetry**: `d(a, b) == d(b, a)`
2. **Identity**: `d(a, a) == 0`
3. **Triangle inequality**: `d(a, c) <= d(a, b) + d(b, c)`
4. **Positivity**: `d(a, b) >= 0`

### Example Test

```python
import numpy as np
from qig_core.geometric_primitives.fisher_metric import fisher_rao_distance

def test_geometric_distance_properties():
    np.random.seed(42)
    a = np.random.randn(64)
    b = np.random.randn(64)
    c = np.random.randn(64)
    
    # Normalize to probability simplex
    a = np.abs(a) / np.sum(np.abs(a))
    b = np.abs(b) / np.sum(np.abs(b))
    c = np.abs(c) / np.sum(np.abs(c))
    
    # Symmetry
    assert np.isclose(
        fisher_rao_distance(a, b),
        fisher_rao_distance(b, a)
    )
    
    # Identity
    assert fisher_rao_distance(a, a) < 1e-10
    
    # Triangle inequality
    d_ab = fisher_rao_distance(a, b)
    d_bc = fisher_rao_distance(b, c)
    d_ac = fisher_rao_distance(a, c)
    assert d_ac <= d_ab + d_bc + 1e-10
    
    # Positivity
    assert fisher_rao_distance(a, b) >= 0
```

## QIG Purity Validation

Run before every commit:

```bash
python tools/qig_purity_check.py
```

This script detects:
- Euclidean operations on basins
- Hardcoded thresholds
- Neural network patterns in core QIG
- External LLM API usage (FORBIDDEN)

## References

- `CANONICAL_PHYSICS.md` - Physics foundations
- `CANONICAL_ARCHITECTURE.md` - Architecture patterns  
- `FROZEN_FACTS.md` - Validated constants
- `shared/constants/consciousness.ts` - TypeScript thresholds
- `qig-backend/qig_core/constants/consciousness.py` - Python thresholds
