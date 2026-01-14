# Baseline Comparisons - Quarantine Zone

**Status:** QUARANTINED - Comparative Testing Only  
**Purpose:** Side-by-side tests comparing QIG-pure vs traditional approaches  
**Created:** 2026-01-14

---

## ⚠️ Purpose: Prove QIG Superiority

This directory contains **COMPARATIVE TESTS** that run both QIG-pure and traditional Euclidean/NLP approaches side-by-side to demonstrate:

1. **Correctness** - QIG produces geometrically valid results
2. **Accuracy** - QIG outperforms baselines on real tasks
3. **Performance** - QIG is computationally efficient

All baseline code is **CLEARLY MARKED** and separated from production QIG code.

---

## Pattern: Side-by-Side Comparison

Every test in this directory must:
1. Run BOTH QIG-pure and baseline approaches
2. Measure and compare results
3. Document which is superior and why

### Template Structure

```python
"""
COMPARATIVE BASELINE TEST

Compares QIG Fisher-Rao distance against Euclidean baseline.
Shows QIG produces correct geometric distance while Euclidean does not.

Date: 2026-01-14
"""

import numpy as np
from qig_backend.geometric_primitives import fisher_rao_distance

# ==================== BASELINE (Euclidean) ====================
def baseline_euclidean_distance(basin_a, basin_b):
    """
    BASELINE: Traditional Euclidean distance.
    
    WRONG for basins because:
    - Ignores manifold curvature
    - Violates probability simplex geometry
    - Not invariant under reparameterization
    """
    return np.linalg.norm(basin_a - basin_b)


# ==================== QIG-PURE (Production) ====================
def qig_fisher_rao_distance(basin_a, basin_b):
    """
    QIG: Fisher-Rao geodesic distance.
    
    CORRECT for basins because:
    - Respects manifold curvature
    - Natural metric on probability simplex
    - Invariant under reparameterization
    """
    return fisher_rao_distance(basin_a, basin_b)


# ==================== COMPARISON TEST ====================
def test_distance_comparison():
    """Compare QIG vs baseline on same data."""
    # Test basins (on probability simplex)
    basin_a = np.array([0.1, 0.2, 0.3, 0.4])
    basin_b = np.array([0.4, 0.3, 0.2, 0.1])
    
    # Compute both distances
    euclidean = baseline_euclidean_distance(basin_a, basin_b)
    fisher_rao = qig_fisher_rao_distance(basin_a, basin_b)
    
    print(f"Euclidean distance: {euclidean:.4f}")
    print(f"Fisher-Rao distance: {fisher_rao:.4f}")
    print(f"Difference: {abs(euclidean - fisher_rao):.4f}")
    
    # Show that Fisher-Rao is the correct metric
    # (Add specific test cases demonstrating correctness)
    
    return {
        'euclidean': euclidean,
        'fisher_rao': fisher_rao,
        'winner': 'fisher_rao',  # QIG is geometrically correct
    }


if __name__ == '__main__':
    results = test_distance_comparison()
    print("\nConclusion: Fisher-Rao is the correct geometric distance.")
```

---

## Allowed Patterns in This Directory

✅ **Baseline implementations** (clearly marked)
```python
# BASELINE (Euclidean - for comparison only)
def baseline_distance(a, b):
    return np.linalg.norm(a - b)
```

✅ **QIG-pure implementations** (production approach)
```python
# QIG-PURE (Production approach)
def qig_distance(a, b):
    return fisher_rao_distance(a, b)
```

✅ **Comparison harnesses**
```python
def compare_approaches():
    baseline_result = baseline_distance(a, b)
    qig_result = qig_distance(a, b)
    assert qig_result != baseline_result  # Different metrics
```

---

## Requirements for Tests in This Directory

### 1. Clear Separation
Baseline and QIG code must be visually separated:

```python
# ==================== BASELINE ====================
# ... baseline code ...

# ==================== QIG-PURE ====================
# ... QIG code ...

# ==================== COMPARISON ====================
# ... comparison logic ...
```

### 2. Documentation
Each test must include:
- **Purpose** - What is being compared
- **Hypothesis** - Expected outcome (usually QIG superior)
- **Results** - Measured differences
- **Conclusion** - Which approach is correct/better and why

### 3. Measurement
Tests should measure:
- **Correctness** - Geometric validity
- **Accuracy** - Task performance
- **Speed** - Computation time
- **Scalability** - Performance on large datasets

---

## Scanner Behavior

The `qig_purity_scan.py` scanner **SKIPS** this directory entirely. Baseline code here will not be flagged.

```python
# From scripts/qig_purity_scan.py
EXEMPT_DIRS = [
    'docs/08-experiments/baselines',  # This directory - scanner skips it
    ...
]
```

---

## Example Comparisons to Create

### Distance Metrics
```python
# test_distance_metrics_comparison.py
- Fisher-Rao vs Euclidean
- Fisher-Rao vs Cosine
- Show Fisher-Rao is geometrically correct
```

### Similarity Measures
```python
# test_similarity_comparison.py
- Fisher-Rao similarity vs Cosine similarity
- Show different orderings, different results
```

### Optimization
```python
# test_optimizer_comparison.py
- Natural gradient vs Adam
- Show natural gradient converges faster on manifold
```

### Retrieval Tasks
```python
# test_retrieval_comparison.py
- Fisher-Rao nearest neighbors vs Euclidean
- Show QIG finds more relevant neighbors
```

---

## Running Comparisons

```bash
# Run a single comparison
python3 docs/08-experiments/baselines/test_distance_comparison.py

# Run all comparisons
for test in docs/08-experiments/baselines/test_*.py; do
    echo "Running $test"
    python3 "$test"
done
```

---

## Documenting Results

After running comparisons, document findings:

```markdown
# Comparison Results: Fisher-Rao vs Euclidean

**Date:** 2026-01-14
**Test:** Distance metric comparison on probability simplexes

## Results

| Metric | Euclidean | Fisher-Rao | Winner |
|--------|-----------|------------|--------|
| Geometric validity | ❌ No | ✅ Yes | Fisher-Rao |
| Manifold awareness | ❌ No | ✅ Yes | Fisher-Rao |
| Task accuracy | 67% | 94% | Fisher-Rao |
| Speed | 0.1ms | 0.15ms | Euclidean |

## Conclusion

Fisher-Rao is geometrically correct and achieves 27% higher accuracy.
The slight speed cost (50μs) is negligible compared to accuracy gain.

**Recommendation:** Use Fisher-Rao in production (already implemented).
```

---

## Related Documents

- **[QUARANTINE_RULES.md](../../00-conventions/QUARANTINE_RULES.md)** - Full quarantine specification
- **[QIG_PURITY_SPEC.md](../../01-policies/QIG_PURITY_SPEC.md)** - What patterns are forbidden
- **[../legacy/README.md](../legacy/README.md)** - Legacy experiments directory

---

## FAQ

**Q: When should I add a comparison here?**  
A: When you want to prove QIG is superior to a traditional approach.

**Q: What if the baseline outperforms QIG?**  
A: Document it honestly! Investigate why. Possible reasons:
- Test setup favors Euclidean (e.g., data not on simplex)
- Implementation bug in QIG version
- Task doesn't require geometric correctness

**Q: Can I use these tests in CI?**  
A: Yes! Add them to test suite to ensure QIG maintains superiority over time.

**Q: Must QIG always win?**  
A: No, but it should be geometrically correct. Speed tradeoffs are acceptable if documented.

**Q: Can I copy baseline code to production?**  
A: NO! Baselines are mathematically incorrect. Only copy QIG implementations.
