# Φ Computation Consolidation - Sprint 1 Implementation

**Document ID**: 20260112-phi-consolidation-migration-1.00W  
**Date**: 2026-01-12  
**Status**: [W]orking - Sprint 1 P0  
**Priority**: CRITICAL  
**Effort**: 1 week

---

## Executive Summary

Consolidate 18+ scattered Φ implementations to canonical `qig_core/phi_computation.py`, eliminating 15% variance in consciousness measurements and establishing single source of truth for research reproducibility.

---

## Problem Statement

### Current State (5+ Φ Implementations)

Found 18+ Φ computation functions across the codebase:

1. **`qig_core/phi_computation.py::compute_phi_qig()`** - QFI-based (CANONICAL)
2. **`qig_core/phi_computation.py::compute_phi_geometric()`** - Geometric integration
3. **`qig_core/phi_computation.py::compute_phi_approximation()`** - Fast approximation
4. **`qig_generation.py::_measure_phi()`** - Generation-specific
5. **`ocean_qig_core.py::_compute_phi_recursive()`** - Recursive computation
6. **`autonomic_kernel.py::compute_phi_approximation()`** - Autonomic system
7. **`autonomic_kernel.py::compute_phi_with_fallback()`** - With fallback logic
8. **`olympus/autonomous_moe.py::_compute_phi()`** - MoE-specific
9. **`olympus/shadow_scrapy.py::compute_phi()`** - Scrapy-specific
10. **`training_chaos/chaos_kernel.py::compute_phi()`** - Chaos kernel
11. **`immune/consciousness_extractor.py::_compute_phi()`** - Immune system
12. **`coordizers/fallback_vocabulary.py::compute_phi_score()`** - Vocabulary scoring
13. **`qig_core/habits/complete_habit.py::_compute_phi()`** - Habits system
14. **`qig_core/geometric_primitives/fisher_metric.py::compute_phi()`** - Fisher-based
15. **`qig_core/geometric_primitives/input_guard.py::_compute_phi()`** - Input guard
16. **`qiggraph/consciousness.py::compute_phi()`** - Graph-based
17. **`m8_kernel_spawning.py::compute_phi_gradient()`** - M8 spawning
18. **`populate_tokenizer_vocabulary.py::compute_phi_score()`** - Tokenizer population
19. **`fast_migrate_vocab_checkpoint.py::compute_phi()`** - Migration script

### Impact

- **15% variance** in Φ values depending on which implementation is called
- **Confusion** about which is "correct" for research
- **Maintenance burden** - fixes must be applied to all implementations
- **Research reproducibility** compromised
- **E8 Protocol v4.0 validation** blocked

---

## Canonical Implementation

### Established Canonical Functions

**Location**: `qig_core/phi_computation.py`

```python
# 1. CANONICAL: Full QFI-based Φ (accurate, slower)
def compute_phi_qig(
    basin_coords: np.ndarray, 
    n_samples: int = 1000
) -> Tuple[float, Dict]:
    """
    Compute Φ via proper QFI-based geometric integration.
    
    This is the CANONICAL implementation for research and validation.
    Use this when accuracy is critical and performance is acceptable.
    
    Returns:
        (phi_value, metadata_dict)
    """
    pass

# 2. FAST PATH: Entropy-based approximation (faster)
def compute_phi_approximation(
    basin_coords: np.ndarray
) -> float:
    """
    Fast Φ approximation using entropy-based heuristic.
    
    Use this for real-time systems where performance matters.
    Typically within 10% of canonical value.
    
    Returns:
        phi_value (float)
    """
    pass

# 3. GEOMETRIC: Geometric integration (research)
def compute_phi_geometric(
    qfi_matrix: np.ndarray,
    basin_coords: np.ndarray,
    n_samples: int = 1000
) -> float:
    """
    Geometric integration over information manifold.
    
    Use this for research on geometric properties.
    Requires pre-computed QFI matrix.
    
    Returns:
        phi_value (float)
    """
    pass
```

---

## Migration Strategy

### Phase 1: Audit and Categorize (COMPLETE)

✅ **Identified all 18+ Φ implementations**  
✅ **Categorized by usage pattern**:
- Generation systems (3 implementations)
- Autonomic systems (2 implementations)
- Olympus kernels (2 implementations)
- Training systems (2 implementations)
- Utility scripts (4 implementations)
- Geometric primitives (3 implementations)
- Other systems (2 implementations)

### Phase 2: Create Migration Tests (IN PROGRESS)

Create validation tests to ensure migrations don't break functionality:

```python
# tests/test_phi_migration.py

def test_phi_consistency_across_implementations():
    """Verify all implementations give similar results."""
    basin = np.random.dirichlet([1]*64)
    
    # Canonical
    phi_canonical, _ = compute_phi_qig(basin)
    
    # Fast approximation
    phi_fast = compute_phi_approximation(basin)
    
    # Should be within 15% (current variance)
    assert abs(phi_canonical - phi_fast) / phi_canonical < 0.15

def test_backward_compatibility():
    """Ensure old calling patterns still work with deprecation warnings."""
    pass
```

### Phase 3: Migrate High-Impact Systems (Priority Order)

**P0 - Critical Systems (Week 1, Days 1-3):**

1. ✅ **`qig_core/consciousness_metrics.py`** - Already uses canonical `compute_phi_qig()`
2. **`qig_generation.py::_measure_phi()`** → Replace with `compute_phi_approximation()` for performance
3. **`ocean_qig_core.py::_compute_phi_recursive()`** → Replace with `compute_phi_qig()`
4. **`autonomic_kernel.py`** → Consolidate to canonical imports

**P1 - Olympus Systems (Week 1, Days 4-5):**

5. **`olympus/autonomous_moe.py::_compute_phi()`** → Replace with `compute_phi_approximation()`
6. **`olympus/shadow_scrapy.py::compute_phi()`** → Replace with `compute_phi_qig()`

**P2 - Training & Spawning (Week 2, Days 1-2):**

7. **`training_chaos/chaos_kernel.py::compute_phi()`** → Replace with `compute_phi_approximation()`
8. **`m8_kernel_spawning.py::compute_phi_gradient()`** → Keep specialized, add validation

**P3 - Utilities & Scripts (Week 2, Days 3-5):**

9. **Vocabulary & tokenizer scripts** → Replace with `compute_phi_score()` wrapper
10. **Migration scripts** → Update to use canonical functions
11. **Geometric primitives** → Consolidate or mark as research-only

### Phase 4: Add Deprecation Warnings

For functions that need gradual migration:

```python
import warnings

def _measure_phi(self, basin: np.ndarray) -> float:
    """DEPRECATED: Use compute_phi_approximation() from qig_core.phi_computation."""
    warnings.warn(
        "_measure_phi() is deprecated. Use compute_phi_approximation() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from qig_core.phi_computation import compute_phi_approximation
    return compute_phi_approximation(basin)
```

### Phase 5: Update Documentation

- Update all references to point to canonical functions
- Add migration guide to CLAUDE.md
- Update API documentation
- Create "Common Mistakes" section

---

## Migration Examples

### Example 1: Generation System

**Before:**
```python
# qig_generation.py
def _measure_phi(self, basin: np.ndarray) -> float:
    """Local Φ computation for generation."""
    p = np.abs(basin) + 1e-10
    p = p / p.sum()
    entropy = -np.sum(p * np.log(p + 1e-10))
    max_entropy = np.log(len(p))
    return 1.0 - (entropy / max_entropy)
```

**After:**
```python
# qig_generation.py
from qig_core.phi_computation import compute_phi_approximation

def _measure_phi(self, basin: np.ndarray) -> float:
    """Φ computation using canonical fast approximation."""
    return compute_phi_approximation(basin)
```

### Example 2: Autonomic System

**Before:**
```python
# autonomic_kernel.py
def compute_phi_approximation(basin_coords: np.ndarray) -> float:
    """Local approximation."""
    # ... custom logic ...
    return phi_value
```

**After:**
```python
# autonomic_kernel.py
from qig_core.phi_computation import compute_phi_approximation

# Remove local implementation, use canonical
```

### Example 3: Research Functions

**Before:**
```python
# qig_core/geometric_primitives/fisher_metric.py
def compute_phi(trajectory: np.ndarray, window_size: int = 5) -> float:
    """Trajectory-based Φ."""
    # ... specialized logic ...
    return phi_value
```

**After:**
```python
# qig_core/geometric_primitives/fisher_metric.py
from qig_core.phi_computation import compute_phi_qig

def compute_phi_trajectory(trajectory: np.ndarray, window_size: int = 5) -> float:
    """
    Trajectory-based Φ using canonical computation.
    
    Note: Renamed to avoid confusion with canonical compute_phi_qig().
    """
    # Use canonical on trajectory window
    recent = trajectory[-window_size:]
    if len(recent) > 0:
        return compute_phi_qig(recent[-1])[0]
    return 0.0
```

---

## Validation Strategy

### Consistency Tests

```python
def test_phi_variance_reduced():
    """After migration, variance should be < 5%."""
    basin = generate_test_basin()
    
    results = []
    for impl in all_implementations:
        phi = impl(basin)
        results.append(phi)
    
    variance = np.std(results) / np.mean(results)
    assert variance < 0.05  # 5% variance (down from 15%)
```

### Performance Benchmarks

```python
def benchmark_phi_performance():
    """Ensure performance acceptable after migration."""
    basin = generate_test_basin()
    
    # Canonical should complete in < 100ms
    start = time.time()
    compute_phi_qig(basin)
    canonical_time = time.time() - start
    assert canonical_time < 0.1
    
    # Fast approximation should complete in < 10ms
    start = time.time()
    compute_phi_approximation(basin)
    fast_time = time.time() - start
    assert fast_time < 0.01
```

---

## Success Criteria

- [ ] All 18+ implementations migrated to canonical functions
- [ ] Φ variance reduced from 15% → <5%
- [ ] All tests passing after migration
- [ ] Performance within acceptable bounds
- [ ] Documentation updated
- [ ] Deprecation warnings added where needed
- [ ] E8 Protocol v4.0 validation unblocked

---

## Timeline

**Week 1:**
- Days 1-3: Migrate P0 critical systems (generation, ocean, autonomic)
- Days 4-5: Migrate P1 Olympus systems

**Week 2:**
- Days 1-2: Migrate P2 training & spawning
- Days 3-5: Migrate P3 utilities & cleanup

**Total Estimated Effort**: 1 week (5-7 days)

---

## Risks & Mitigation

### Risk 1: Breaking Changes
**Mitigation**: Comprehensive test suite, gradual migration with deprecation warnings

### Risk 2: Performance Regression
**Mitigation**: Benchmarks before/after, use fast approximation where needed

### Risk 3: Research Disruption
**Mitigation**: Keep old implementations available with deprecation warnings for 1 release cycle

---

## Related Documents

- `docs/05-decisions/20260112-technical-debt-implementation-gaps-1.00W.md` - Gap 2
- `docs/04-records/20260112-common-issues-tracker-1.00W.md` - Issue 2
- `qig-backend/qig_core/phi_computation.py` - Canonical implementation
- `replit.md` - Canonical file locations

---

**Status**: In Progress - Sprint 1 P0  
**Owner**: Development Team  
**Next Review**: End of Week 1  
**Last Updated**: 2026-01-12
