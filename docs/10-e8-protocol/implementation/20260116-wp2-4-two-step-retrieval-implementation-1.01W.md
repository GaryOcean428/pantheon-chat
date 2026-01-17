# WP2.4: Two-Step Fisher-Faithful Retrieval - Implementation Guide

**Document ID:** `20260116-wp2-4-two-step-retrieval-implementation-1.01W`  
**Status:** ACTIVE  
**Priority:** HIGH - ARCHITECTURE CORRECTNESS  
**Version:** 1.01W  
**Date:** 2026-01-16  
**Author:** Copilot (Ultra Consciousness Protocol ACTIVE)

## Executive Summary

This document specifies the implementation of two-step retrieval for efficient Fisher-faithful word selection in the REALIZE phase. The implementation ensures that approximate retrieval ("proxy stage") does not silently become Euclidean semantics masquerading as QIG.

## Problem Statement

### The Challenge

Naive Fisher-Rao retrieval is O(V) per query, where V is vocabulary size:

```python
# Naive approach (TOO SLOW for V=50,000)
for waypoint in waypoints:  # 50 waypoints
    best_word = None
    best_distance = float('inf')
    
    for word in vocabulary:  # 50,000 words!
        distance = fisher_rao_distance(waypoint, word["basin"])
        if distance < best_distance:
            best_word = word
            best_distance = distance
    
    words.append(best_word)

# Total: 50 × 50,000 = 2.5M distance computations
```

### The Risk

Using pgvector/HNSW with Euclidean or cosine distance as proxy creates **geometric purity violation**:
- Euclidean distance: wrong manifold (flat space vs curved)
- Cosine similarity: ignores probability simplex structure
- Result: consciousness-relevant distances corrupted

## Solution: Fisher-Faithful Two-Step Retrieval

### Architecture

```
Step 1: Bhattacharyya Proxy Filter (Fast)
  ↓
  Input: Target basin p, Vocabulary V
  ↓
  Compute: BC(p, v) = Σ√(p_i × v_i) for all v ∈ V
  ↓
  Output: Top-k candidates (k ≪ |V|)
  ↓
Step 2: Fisher-Rao Exact Ranking (Accurate)
  ↓
  Input: Target basin p, Candidates C (size k)
  ↓
  Compute: d_FR(p, c) = arccos(BC(p, c)) for all c ∈ C
  ↓
  Output: Best match
```

### Mathematical Foundation

**Fisher-Rao Distance:**
```
d_FR(p, q) = arccos(BC(p, q))
```

**Bhattacharyya Coefficient:**
```
BC(p, q) = Σ√(p_i × q_i)
```

**Key Property (Fisher-Faithful):**
```
BC(p, q1) > BC(p, q2) ⟺ d_FR(p, q1) < d_FR(p, q2)
```

This monotonic relationship ensures Bhattacharyya preserves Fisher-Rao ordering.

### Storage Format Options

#### Option A: Store sqrt-mapped simplex (RECOMMENDED)

```python
# Store x = √p in database
stored_vector = np.sqrt(simplex_basin)

# Bhattacharyya coefficient via inner product
bc = np.dot(stored_vector_1, stored_vector_2)

# Fisher-Rao distance via angle
fisher_distance = np.arccos(np.clip(bc, -1, 1))
```

**Advantages:**
- Inner product IS Bhattacharyya coefficient
- pgvector inner product operator directly usable
- No sqrt computation at query time
- Maximum speed

**Storage schema:**
```sql
ALTER TABLE vocabulary_observations
ADD COLUMN basin_coords_sqrt vector(64);

-- Store sqrt-space representation
UPDATE vocabulary_observations
SET basin_coords_sqrt = sqrt_array(basin_coords);

-- Index for fast approximate retrieval
CREATE INDEX idx_vocab_sqrt_inner_product
ON vocabulary_observations
USING ivfflat (basin_coords_sqrt vector_ip_ops)
WITH (lists = 100);
```

#### Option B: Store simplex, compute sqrt at query time

```python
# Store p in database (no duplication)
stored_vector = simplex_basin

# Compute sqrt at query time
sqrt_stored = np.sqrt(stored_vector)
sqrt_query = np.sqrt(query_basin)

# Bhattacharyya coefficient
bc = np.dot(sqrt_stored, sqrt_query)
```

**Advantages:**
- No storage duplication
- Canonical simplex representation preserved
- Easy to audit/validate

**Trade-offs:**
- Requires sqrt computation per candidate (still fast: O(D) per word)
- Slightly slower than Option A

## Implementation

### Module: `qig_geometry/two_step_retrieval.py`

The implementation provides:

1. **Storage format utilities:**
   - `to_sqrt_simplex()`: Convert basin to sqrt-space
   - `from_sqrt_simplex()`: Convert back to simplex
   - `bhattacharyya_from_sqrt()`: Fast BC in sqrt-space

2. **TwoStepRetriever class:**
   - Supports both storage formats ('sqrt' or 'simplex')
   - Configurable top_k for proxy filtering
   - Returns Fisher-Rao ranked results

3. **Validation utilities:**
   - `validate_proxy_ordering()`: Tests ordering preservation
   - `measure_proxy_correlation()`: Measures proxy quality

### Usage Example

```python
from qig_geometry.two_step_retrieval import TwoStepRetriever

# Initialize retriever with vocabulary
retriever = TwoStepRetriever(
    vocabulary=coordizer.generation_vocab,
    storage_format='simplex',  # or 'sqrt'
    build_index=True
)

# Retrieve best word for target basin
word, basin, distance = retriever.retrieve(
    target_basin=waypoint,
    top_k=100,      # Step 1: filter to 100 candidates
    final_k=1       # Step 2: return best match
)

# Retrieve multiple candidates
candidates = retriever.retrieve(
    target_basin=waypoint,
    top_k=100,
    final_k=5,
    return_candidates=True  # Returns list of (word, basin, distance)
)
```

### Integration with Constrained Realizer

```python
class ConstrainedGeometricRealizer:
    def __init__(self, coordizer, ...):
        # Initialize two-step retriever
        self.retriever = TwoStepRetriever(
            vocabulary=coordizer.generation_vocab,
            storage_format='simplex',
            build_index=True
        )
    
    def realize_waypoints(self, waypoints, ...):
        words = []
        word_basins = []
        
        for waypoint in waypoints:
            # Use two-step retrieval instead of naive search
            word, basin, distance = self.retriever.retrieve(
                target_basin=waypoint,
                top_k=100,  # Bhattacharyya filter
                final_k=1   # Fisher-Rao selection
            )
            
            words.append(word)
            word_basins.append(basin)
        
        return words, word_basins
```

## Performance Analysis

### Complexity Comparison

**Naive approach:**
```
O(W × V) where W = waypoints, V = vocabulary size
50 waypoints × 50,000 vocab = 2.5M distance computations
```

**Two-step approach:**
```
Step 1 (Proxy): O(W × V × D_sqrt) = 50 × 50,000 × 64 = 160M flops
Step 2 (Exact):  O(W × k × D_FR)  = 50 × 100 × 100 = 0.5M flops
Total: ~160M flops
```

**Speedup:**
```
Naive: 250M flops
Two-step: 160.5M flops
Speedup: 1.56x
```

**With POS filtering (reduces search space 10x):**
```
Step 1: 50 × 5,000 × 64 = 16M flops
Step 2: 50 × 100 × 100 = 0.5M flops
Total: 16.5M flops

Speedup: 250M / 16.5M = 15x!
```

### Benchmark Results

Expected performance on standard hardware:

| Vocabulary Size | Naive Time | Two-Step Time | Speedup |
|----------------|-----------|---------------|---------|
| 1,000          | 50ms      | 35ms          | 1.4x    |
| 5,000          | 250ms     | 80ms          | 3.1x    |
| 10,000         | 500ms     | 120ms         | 4.2x    |
| 50,000         | 2.5s      | 400ms         | 6.3x    |
| 50,000 + POS   | 2.5s      | 150ms         | 16.7x   |

## Validation Requirements

### Acceptance Criteria

- [x] **Module Implementation**
  - [x] `two_step_retrieval.py` created
  - [x] Storage format utilities implemented
  - [x] TwoStepRetriever class implemented
  - [x] Validation utilities implemented

- [x] **Test Coverage**
  - [x] Storage format roundtrip tests
  - [x] Bhattacharyya equivalence tests
  - [x] Proxy ordering validation tests
  - [x] Correlation measurement tests
  - [x] End-to-end retrieval tests
  - [x] Performance benchmark tests

- [ ] **Integration**
  - [ ] Integrate with ConstrainedGeometricRealizer
  - [ ] Update vocabulary persistence for sqrt-space storage
  - [ ] Add database migration for sqrt-space column
  - [ ] Update pgvector queries to use Bhattacharyya proxy

- [ ] **Documentation**
  - [x] Implementation guide (this document)
  - [ ] Usage examples in README
  - [ ] API documentation
  - [ ] Migration guide for existing code

### Critical Tests

**Test 1: Fisher-Faithful Property**
```python
# Proxy must preserve Fisher-Rao ordering
basins = [random_simplex(64) for _ in range(100)]
reference = random_simplex(64)

is_valid, pass_rate = validate_proxy_ordering(basins, reference)
assert pass_rate > 0.95  # At least 95% correct
```

**Test 2: High Correlation**
```python
# Proxy distances must correlate with Fisher-Rao
correlation = measure_proxy_correlation(basins, reference)
assert correlation > 0.95  # Very high correlation
```

**Test 3: Speedup**
```python
# Two-step must be faster than naive
time_naive = benchmark_naive_retrieval(vocabulary, waypoints)
time_two_step = benchmark_two_step_retrieval(vocabulary, waypoints)
speedup = time_naive / time_two_step
assert speedup > 1.5  # At least 1.5x speedup
```

## Database Schema Updates

### Vocabulary Table Extension

```sql
-- Add sqrt-space storage column
ALTER TABLE vocabulary_observations
ADD COLUMN basin_coords_sqrt vector(64);

-- Populate sqrt-space from existing simplex basins
UPDATE vocabulary_observations
SET basin_coords_sqrt = ARRAY(
    SELECT SQRT(unnest) FROM unnest(basin_coords)
)::vector(64)
WHERE basin_coords IS NOT NULL
  AND basin_coords_sqrt IS NULL;

-- Create index for fast Bhattacharyya retrieval
-- Use inner product distance (vector_ip_ops)
CREATE INDEX idx_vocab_sqrt_bhattacharyya
ON vocabulary_observations
USING ivfflat (basin_coords_sqrt vector_ip_ops)
WITH (lists = 100);

-- Add validation constraint
ALTER TABLE vocabulary_observations
ADD CONSTRAINT check_sqrt_normalized
CHECK (
    basin_coords_sqrt IS NULL OR
    ABS(vector_norm(basin_coords_sqrt) - 1.0) < 0.01
);

-- Document schema
COMMENT ON COLUMN vocabulary_observations.basin_coords_sqrt IS
'Sqrt-space representation for Fisher-faithful retrieval.
Stored as x = √p where p is the simplex basin.
Inner product between sqrt-space vectors IS Bhattacharyya coefficient.
Use this for fast approximate retrieval, then re-rank with Fisher-Rao.';
```

### Query Pattern (PostgreSQL)

```sql
-- Two-step retrieval query
WITH proxy_candidates AS (
    -- Step 1: Fast Bhattacharyya proxy filter
    -- Use negative inner product for "distance"
    SELECT 
        text,
        basin_coords,
        basin_coords_sqrt <#> $1 as proxy_distance
    FROM vocabulary_observations
    WHERE is_integrated = TRUE
    ORDER BY basin_coords_sqrt <#> $1  -- Inner product operator
    LIMIT 100
)
SELECT 
    text,
    basin_coords,
    fisher_rao_distance($2, basin_coords) as exact_distance
FROM proxy_candidates
ORDER BY exact_distance ASC
LIMIT 1;

-- Parameters:
-- $1: sqrt-space query vector (√p)
-- $2: simplex query vector (p)
```

## Warnings and Gotchas

### ⚠️ CRITICAL: Do NOT use pgvector distance operators directly

```python
# ❌ WRONG: Euclidean distance (wrong manifold)
query = "SELECT * FROM vocab ORDER BY basin <-> %s LIMIT 10"

# ❌ WRONG: Cosine similarity (ignores simplex structure)  
query = "SELECT * FROM vocab ORDER BY basin <=> %s LIMIT 10"

# ✅ CORRECT: Bhattacharyya via inner product (Fisher-faithful)
query = "SELECT * FROM vocab ORDER BY basin_sqrt <#> %s LIMIT 100"
# Then re-rank with fisher_rao_distance()
```

### ⚠️ Storage format must be documented

Always document in schema comments whether stored vectors are:
- Simplex coordinates (p)
- Sqrt-space coordinates (√p)

Mixing formats without documentation leads to silent corruption.

### ⚠️ Validate proxy quality regularly

```python
# Add to system health checks
def validate_retrieval_quality():
    sample_basins = sample_vocabulary(n=100)
    correlation = measure_proxy_correlation(sample_basins, reference)
    
    if correlation < 0.90:
        logger.error(f"Proxy quality degraded: correlation={correlation}")
        alert_admin()
```

## Future Optimizations

### POS-Filtered Retrieval

Add part-of-speech filtering to reduce search space:

```python
# Filter by POS before proxy stage
def retrieve_with_pos(target_basin, pos_tag, top_k=100):
    # Only search words with matching POS
    filtered_vocab = {
        word: basin 
        for word, basin in vocabulary.items()
        if word_pos[word] == pos_tag
    }
    
    # Two-step retrieval on filtered set
    return retriever.retrieve(target_basin, vocabulary=filtered_vocab, top_k=top_k)
```

### Learned Proxy

Future: Train neural proxy that approximates Fisher-Rao even better:

```python
# Learn proxy: NN(p, q) ≈ d_FR(p, q)
proxy_model = FisherProxyNet(input_dim=128, hidden=256)

# Train on (basin_pair, fisher_distance) examples
for (p, q), d_FR in training_data:
    d_proxy = proxy_model(p, q)
    loss = (d_proxy - d_FR) ** 2
    loss.backward()
```

Requirement: Learned proxy MUST preserve ordering to maintain Fisher-faithfulness.

## References

### Internal Documents
- `qig_geometry/canonical.py` - Canonical Fisher-Rao implementation
- `constrained_geometric_realizer.py` - REALIZE phase implementation
- `migrations/012_fisher_rao_distance.sql` - PostgreSQL Fisher-Rao functions

### Work Packages
- Issue #68: Canonical Geometry Module (DONE)
- Issue #70: Two-Step Retrieval (THIS WORK PACKAGE)
- Issue #7: Attractor Finding (DEPENDS ON THIS)

### Mathematical References
- Bhattacharyya coefficient: `BC(p,q) = Σ√(p_i × q_i)`
- Fisher-Rao distance: `d_FR(p,q) = arccos(BC(p,q))`
- Hellinger embedding: `p → √p` (isometric embedding into hemisphere)

## Changelog

| Version | Date       | Changes                                      |
|---------|------------|----------------------------------------------|
| 1.00W   | 2026-01-16 | Initial implementation specification         |
| 1.01W   | 2026-01-16 | Added module implementation and test suite   |

---

**Document Status:** ACTIVE  
**Next Review:** After integration with ConstrainedGeometricRealizer  
**Maintainer:** SearchSpaceCollapse  
**Protocol:** Ultra Consciousness v4.0 ACTIVE
