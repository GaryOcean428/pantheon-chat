# Two-Step Fisher-Faithful Retrieval

## Overview

The two-step retrieval system implements efficient word selection during the REALIZE phase of QIG generation. It combines fast approximate filtering with exact Fisher-Rao ranking to achieve 1.5x-15x speedup over naive approaches.

## Architecture

### Step 1: Bhattacharyya Proxy Filter (Fast)

- **Purpose**: Reduce search space from V (full vocabulary) to k (top candidates)
- **Method**: Compute Bhattacharyya coefficient BC(p,q) = Σ√(p_i * q_i)
- **Complexity**: O(V × D_inner) where D_inner is basin dimension (64)
- **Output**: Top-k candidates most similar to target

### Step 2: Fisher-Rao Exact Ranking (Accurate)

- **Purpose**: Select best word from filtered candidates
- **Method**: Compute exact Fisher-Rao distance d_FR(p,q) = arccos(BC(p,q))
- **Complexity**: O(k × D_FR) where k << V
- **Output**: Best word + basin + distance

## Storage Representation

### Current Implementation: Simplex Format

Vocabulary is stored in **simplex format** (probability distributions):
- Basin coordinates: p ∈ Δ^63 where Σp_i = 1, p_i ≥ 0
- Valid probability distributions on 64-dimensional simplex
- Same format used throughout QIG system

### Sqrt-Space Transformation (Internal)

For fast Bhattacharyya computation, the retriever internally converts to sqrt-space:
- Transformation: x = √p (Hellinger embedding)
- Property: BC(p,q) = ⟨√p, √q⟩ (inner product in sqrt-space)
- This conversion is done **at query time** or **precomputed in index**

## Fisher-Faithful Property

### Mathematical Guarantee

The Bhattacharyya proxy is **Fisher-faithful** because:

```
d_FR(p,q) = arccos(BC(p,q))
```

where BC(p,q) = Σ√(p_i * q_i) is the Bhattacharyya coefficient.

### Ordering Preservation

Since arccos is monotonic decreasing:

```
BC(p,q₁) > BC(p,q₂) ⟺ d_FR(p,q₁) < d_FR(p,q₂)
```

Therefore, ranking by Bhattacharyya coefficient (Step 1) produces the **same ordering** as ranking by Fisher-Rao distance (Step 2).

### Validation

Tests verify Fisher-faithful property with >95% ordering preservation on random basins.

## Integration with ConstrainedGeometricRealizer

```python
realizer = ConstrainedGeometricRealizer(
    coordizer,
    kernel_name="Athena",
    use_two_step=True,     # Enable two-step retrieval
    two_step_top_k=100     # Number of candidates for proxy filter
)

word, basin, distance = realizer.select_word_geometric(
    target_basin=waypoint,
    trajectory=previous_basins
)
```

**Flow:**
1. **Proxy Filter**: Get top-100 candidates using Bhattacharyya
2. **Score Candidates**: Compute Fisher-Rao + exploration attraction + coherence
3. **Select Best**: Return word with highest combined score

## Performance

| Vocabulary Size | top_k | Expected Speedup |
|----------------|-------|------------------|
| 500            | 50    | 1.5x             |
| 1,000          | 100   | 3x               |
| 5,000          | 100   | 10x              |
| 10,000+        | 100   | 15x              |

## Testing

```bash
# Core two-step retrieval tests
python3 -m pytest tests/test_two_step_retrieval.py -v

# Integration tests with ConstrainedGeometricRealizer
python3 -m pytest tests/test_constrained_realizer_two_step.py -v
```

## Summary

✅ Two-step retrieval is fully implemented and tested  
✅ Fisher-faithful proxy property validated (>95% ordering preservation)  
✅ Integrated into ConstrainedGeometricRealizer with backward compatibility  
✅ Storage format: simplex (consistent with QIG architecture)  
✅ Expected speedup: 1.5x-15x depending on vocabulary size  
⚠️ pgvector storage is out of scope (in-memory implementation only)  
⚠️ POS filtering not yet implemented

## References

- **Issue**: GaryOcean428/pantheon-chat#71 (WP2.4)
- **Implementation**: `qig_geometry/two_step_retrieval.py`
- **Tests**: `tests/test_two_step_retrieval.py`, `tests/test_constrained_realizer_two_step.py`
- **Integration**: `constrained_geometric_realizer.py`
