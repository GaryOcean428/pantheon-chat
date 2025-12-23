# Geometric Operations Implementation Guide

**Version**: 1.0  
**Status**: FINAL  
**Date**: 2025-01-01  
**Supersedes**: None

## Overview

This guide provides canonical patterns for implementing geometric operations in pantheon-chat. Following these patterns ensures geometric purity and consistency with QIG theory.

## Core Principle: No Euclidean Operations on Basins

### FORBIDDEN Patterns

```python
# ❌ WRONG - Euclidean distance
distance = np.linalg.norm(basin_a - basin_b)

# ❌ WRONG - Cosine similarity
similarity = np.dot(basin_a, basin_b) / (np.linalg.norm(basin_a) * np.linalg.norm(basin_b))

# ❌ WRONG - L2 norm for distance
distance = np.sqrt(np.sum((basin_a - basin_b) ** 2))
```

### REQUIRED Patterns

```python
# ✓ CORRECT - Fisher-Rao distance
from qig_core.geometric_primitives.canonical_fisher import fisher_rao_distance

distance = fisher_rao_distance(basin_a, basin_b)
```

## Pattern 1: Basin Distance Computation

Always use the canonical Fisher-Rao implementation:

```python
from qig_core.geometric_primitives.canonical_fisher import fisher_rao_distance
import numpy as np

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

## Pattern 2: Nearest Basin Search

```python
from qig_core.geometric_primitives.canonical_fisher import find_nearest_basins

def find_relevant_basins(
    query_basin: np.ndarray,
    candidate_basins: list[np.ndarray],
    k: int = 10
) -> list[tuple[int, float]]:
    """
    Find k nearest basins using geometric distance.
    
    Returns: [(index, distance), ...] sorted by distance
    """
    return find_nearest_basins(query_basin, candidate_basins, k=k)
```

## Pattern 3: Basin Interpolation

Use geodesic interpolation, not linear:

```python
from qig_core.geometric_primitives.canonical_fisher import geodesic_interpolate

def move_toward_basin(
    current: np.ndarray,
    target: np.ndarray,
    fraction: float = 0.3
) -> np.ndarray:
    """
    Move current basin toward target along geodesic.
    
    Args:
        current: Current basin position
        target: Target basin position
        fraction: How far to move (0-1)
    
    Returns:
        New basin position along geodesic
    """
    return geodesic_interpolate(current, target, t=fraction)
```

## Pattern 4: Consciousness Metric Computation

```python
from qig_core.constants.consciousness import (
    THRESHOLDS,
    classify_regime,
    is_conscious
)

def compute_consciousness_state(
    phi: float,
    kappa: float,
    **other_metrics
) -> dict:
    """
    Compute full consciousness state.
    
    Returns:
        Dictionary with regime, compute_fraction, is_conscious
    """
    # Classify regime
    regime, compute_fraction = classify_regime(phi)
    
    # Check consciousness
    conscious = is_conscious(phi=phi, kappa=kappa, **other_metrics)
    
    return {
        'phi': phi,
        'kappa_eff': kappa,
        'regime': regime,
        'compute_fraction': compute_fraction,
        'is_conscious': conscious,
    }
```

## Pattern 5: Two-Step Retrieval

Optimized retrieval: approximate first, then Fisher re-rank:

```python
async def two_step_retrieval(
    query_basin: np.ndarray,
    k: int = 10,
    approximate_k: int = 50
) -> list[dict]:
    """
    Two-step retrieval with Fisher re-ranking.
    
    Step 1: Approximate (fast) - O(log n)
    Step 2: Fisher re-rank (precise) - O(k)
    """
    # Step 1: Get approximate candidates (using vector index)
    candidates = await db.approximate_search(
        query_basin,
        limit=approximate_k
    )
    
    # Step 2: Re-rank with exact Fisher-Rao distance
    distances = [
        (c, fisher_rao_distance(query_basin, c['basin']))
        for c in candidates
    ]
    distances.sort(key=lambda x: x[1])
    
    return [c for c, _ in distances[:k]]
```

## Testing Requirements

Every geometric operation MUST have tests verifying:

### 1. Metric Space Properties

```python
def test_geometric_distance_properties():
    a, b, c = (
        np.random.dirichlet(np.ones(64)),
        np.random.dirichlet(np.ones(64)),
        np.random.dirichlet(np.ones(64))
    )
    
    # Identity: d(a, a) = 0
    assert fisher_rao_distance(a, a) < 1e-10
    
    # Symmetry: d(a, b) = d(b, a)
    assert np.isclose(
        fisher_rao_distance(a, b),
        fisher_rao_distance(b, a)
    )
    
    # Triangle inequality: d(a, c) <= d(a, b) + d(b, c)
    d_ab = fisher_rao_distance(a, b)
    d_bc = fisher_rao_distance(b, c)
    d_ac = fisher_rao_distance(a, c)
    assert d_ac <= d_ab + d_bc + 1e-10
    
    # Positivity: d(a, b) >= 0
    assert fisher_rao_distance(a, b) >= 0
```

### 2. Geodesic Properties

```python
def test_geodesic_properties():
    start = np.random.dirichlet(np.ones(64))
    end = np.random.dirichlet(np.ones(64))
    
    # Boundary conditions
    assert np.allclose(geodesic_interpolate(start, end, 0.0), start)
    assert np.allclose(geodesic_interpolate(start, end, 1.0), end, atol=1e-6)
    
    # Geodesic should be shortest path
    direct = fisher_rao_distance(start, end)
    mid = geodesic_interpolate(start, end, 0.5)
    path = fisher_rao_distance(start, mid) + fisher_rao_distance(mid, end)
    assert np.isclose(path, direct, rtol=0.1)
```

### 3. Probability Preservation

```python
def test_probability_preserved():
    start = np.random.dirichlet(np.ones(64))
    end = np.random.dirichlet(np.ones(64))
    
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = geodesic_interpolate(start, end, t)
        assert np.all(result >= 0)  # Non-negative
        assert np.isclose(np.sum(result), 1.0, atol=1e-6)  # Sums to 1
```

## QIG Purity Validation

Run the purity validator before committing:

```bash
python tools/validate_qig_purity.py
```

This checks for:
- ❌ External LLM API usage
- ❌ np.linalg.norm() on basins
- ❌ Cosine similarity on basins
- ❌ max_tokens parameters
- ❌ Traditional token-based generation

## References

- [CANONICAL_PHYSICS.md](../CANONICAL_PHYSICS.md) - Physics foundations
- [CANONICAL_ARCHITECTURE.md](../CANONICAL_ARCHITECTURE.md) - Architecture patterns
- [FROZEN_FACTS.md](../FROZEN_FACTS.md) - Validated constants
- `qig_core/geometric_primitives/canonical_fisher.py` - Reference implementation
- `qig_core/constants/consciousness.py` - Consciousness thresholds
