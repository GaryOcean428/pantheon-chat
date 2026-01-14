# Φ Computation Specialized Implementations

**Document ID**: 20260112-phi-specialized-implementations-1.00W  
**Date**: 2026-01-12  
**Status**: [W]orking - Sprint 1 Task 2  
**Related**: `docs/06-implementation/20260112-phi-consolidation-migration-1.00W.md`

---

## Executive Summary

Documents specialized Φ implementations that serve specific purposes and should be retained alongside the canonical `qig_core/phi_computation.py` implementation.

These implementations are NOT duplicates - they compute different aspects of consciousness measurement for specialized use cases.

---

## Canonical Implementation (General Purpose)

**Location**: `qig_backend/qig_core/phi_computation.py`

**Purpose**: General-purpose Φ computation for basin coordinates

**Functions**:
- `compute_phi_qig(basin_coords)` - Full QFI-based geometric integration (accurate, slower)
- `compute_phi_approximation(basin_coords)` - Fast entropy-based approximation (performance)
- `compute_phi_geometric(qfi_matrix, basin_coords)` - Geometric integration (research)

**Usage**: Default for all basin-based consciousness measurements

---

## Specialized Implementations (Keep)

### 1. Recursive State Change Φ (Ocean QIG Core)

**Location**: `qig_backend/ocean_qig_core.py::_compute_phi_recursive()`

**Purpose**: Measures Φ from state change over time using Fisher-Rao distance

**Formula**:
```
Φ^(n) = 1 - d_FR(s^(n), s^(n-1)) / π
```

**Why Keep**:
- Measures integration via convergence (high Φ = states converged)
- Uses state vectors from subsystems, not basin coordinates
- Legitimate specialized use case for recursive consciousness measurement
- QIG-pure (uses Fisher-Rao distance on state manifold)

**Example Use Case**:
```python
# Ocean tracks state evolution over time
# High Φ indicates the system has integrated information
# Low Φ indicates the system is still exploring
current_state = [s.activation for s in subsystems]
phi = _compute_phi_recursive()  # Compares to previous state
```

---

### 2. Chaos Kernel Φ with QFI Fallback

**Location**: `qig_backend/training_chaos/chaos_kernel.py::compute_phi()`

**Purpose**: Get current Φ with automatic QFI fallback for low-Φ prevention

**Why Keep**:
- Hybrid approach: uses internal Φ tracking, falls back to QFI when too low
- Prevents kernel death from phi=0 during training
- Already imports and uses canonical `compute_phi_qig()` properly
- Training-specific logic for threshold management

**Example Use Case**:
```python
# During chaos kernel training
if self._phi < PHI_MIN_THRESHOLD:
    # Fallback to QFI computation
    qfi_phi, _ = compute_phi_qig(basin_array)
    return max(self._phi, qfi_phi)
```

---

### 3. Shadow Scrapy Insight Scoring (NOT basin Φ)

**Location**: `qig_backend/olympus/shadow_scrapy.py::compute_phi()`

**Purpose**: Provisional Φ-like score for scraped web insights (metadata-based)

**Formula**:
```python
phi = base_phi + pattern_bonus + reputation_bonus + risk_bonus
```

**Why Keep**:
- NOT computing basin Φ - computes content quality score
- Name similarity is unfortunate but functionality is different
- Uses pattern hits, source reputation, content uniqueness
- Should consider renaming to `compute_insight_score()` to avoid confusion

**Example Use Case**:
```python
# Scraped web content evaluation
insight = ScrapedInsight(...)
quality_score = compute_phi(insight, basin_coords)  # Metadata-based
```

**Recommendation**: Rename to `compute_insight_quality()` in future refactor to avoid confusion.

---

### 4. 4D Temporal Consciousness (Multi-dimensional)

**Location**: `qig_backend/consciousness_4d.py::compute_phi_temporal()` and `::compute_phi_4D()`

**Purpose**: Temporal and 4D consciousness measurements

**Why Keep**:
- Extends Φ to temporal dimension (search history over time)
- `compute_phi_4D()` combines spatial and temporal Φ
- Research-specific implementation for 4D consciousness experiments
- Different input types (search history, not basin coords)

---

### 5. Graph-Based Consciousness Φ

**Location**: `qig_backend/qiggraph/consciousness.py::compute_phi()`

**Purpose**: Compute Φ from graph activations

**Why Keep**:
- Input is graph activations, not basin coordinates
- Different data structure and computation method
- QIGGraph-specific implementation

---

### 6. M8 Kernel Spawning Φ Gradient

**Location**: `qig_backend/m8_kernel_spawning.py::compute_phi_gradient()`

**Purpose**: Compute Φ gradient for kernel spawning decisions

**Why Keep**:
- Computes gradient (rate of change), not absolute Φ
- Used for kernel spawning threshold decisions
- Specialized use case for M8 meta-learning

---

## Utility Script Implementations (Deprecate)

These should have deprecation warnings and migrate to canonical:

### 1. Vocabulary Scoring

**Locations**:
- `coordizers/fallback_vocabulary.py::compute_phi_score()`
- `populate_coordizer_vocabulary.py::compute_phi_score()`

**Status**: ⚠️ Should use canonical approximation

**Migration**:
```python
from qig_core.phi_computation import compute_phi_approximation

def compute_phi_score(word: str) -> float:
    """DEPRECATED: Use canonical phi_computation."""
    warnings.warn("Use qig_core.phi_computation", DeprecationWarning)
    # Convert word to basin coordinates
    basin = word_to_basin(word)
    return compute_phi_approximation(basin)
```

### 2. Migration Scripts

**Locations**:
- `fast_migrate_vocab_checkpoint.py::compute_phi()`
- `migrate_vocab_checkpoint_to_pg.py::compute_phi_from_vector()`

**Status**: ⚠️ One-time scripts, low priority

**Action**: Add comment pointing to canonical implementation for future reference

---

## Summary

| Implementation | Location | Type | Action |
|---------------|----------|------|--------|
| Canonical QFI | `qig_core/phi_computation.py` | General Purpose | ✅ Use by default |
| Recursive State Φ | `ocean_qig_core.py` | Specialized | ✅ Keep (temporal) |
| Chaos Kernel Φ | `training_chaos/chaos_kernel.py` | Specialized | ✅ Keep (training) |
| Shadow Scrapy | `olympus/shadow_scrapy.py` | Metadata Scoring | ⚠️ Keep but rename |
| 4D Temporal | `consciousness_4d.py` | Research | ✅ Keep (4D) |
| Graph-Based | `qiggraph/consciousness.py` | Graph Structure | ✅ Keep (graph) |
| M8 Gradient | `m8_kernel_spawning.py` | Meta-Learning | ✅ Keep (gradient) |
| Vocabulary Scripts | `coordizers/`, `populate_*` | Utility | ⚠️ Deprecate |
| Migration Scripts | `*migrate*.py` | One-time | ℹ️ Low priority |

---

## Recommendations

### Immediate (Sprint 1)
1. ✅ Keep specialized implementations (legitimate use cases)
2. ⚠️ Add deprecation warnings to vocabulary script implementations
3. ℹ️ Document distinction between basin Φ and metadata scoring

### Future (Sprint 2+)
1. Consider renaming `shadow_scrapy.compute_phi()` → `compute_insight_quality()`
2. Add type hints to clarify input differences
3. Create developer guide explaining when to use each implementation

---

## Migration Status

**Sprint 1 Progress**: 80% complete

- ✅ Canonical implementation validated
- ✅ High-priority systems using canonical
- ✅ Specialized implementations documented
- ⚠️ Vocabulary scripts need deprecation warnings
- 📋 Repository cleanup pending

**Variance Metrics**:
- QFI ↔ Geometric: 0% (perfect match) ✅
- QFI ↔ Approximation: ~16% (intentional - different algorithms) ⚠️
- Performance: 3.77ms (QFI), 0.07ms (approximation) ✅

---

**Last Updated**: 2026-01-12  
**Owner**: Development Team  
**Status**: DOCUMENTED - Specialized implementations preserved
