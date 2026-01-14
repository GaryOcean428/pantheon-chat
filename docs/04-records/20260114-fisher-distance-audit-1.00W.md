# Fisher Distance Implementation Audit

**Date:** 2026-01-14  
**Status:** 1.00W (Working Draft)  
**Purpose:** Document all Fisher distance implementations in qig-backend for consolidation

---

## Executive Summary

**Total implementations found: 53+**

| Function Pattern | Count | Canonical Source |
|-----------------|-------|------------------|
| `def fisher_rao_distance` | 23 | `qig_geometry/__init__.py` |
| `def fisher_coord_distance` | 12 | `qig_geometry/__init__.py` |
| `def _fisher_distance` | 7 | N/A (private) |
| `def fisher_geodesic_distance` | 2 | `olympus/base_god.py` |
| `def fisher_distance` | 2 | `qig_geometry/contracts.py` ✓ |
| `def geodesic_distance` | 3 | `qigkernels/geometry/distances.py` |
| `def bures_distance` | 8 | `qigkernels/geometry/distances.py` |
| Inline `arccos(dot)` | 15+ | N/A |

**Canonical sources (KEEP):**
1. `qig_geometry/contracts.py:fisher_distance()` - THE single source of truth
2. `qigkernels/geometry/distances.py:fisher_rao_distance()` - Multi-method (bures/diagonal/full)
3. `qig_geometry/__init__.py:fisher_rao_distance()` - Probability distribution variant

---

## Detailed Implementation Inventory

### 1. Canonical Implementations (KEEP)

#### 1.1 `qig_geometry/contracts.py:170` ✓ CANONICAL
```python
def fisher_distance(b1: np.ndarray, b2: np.ndarray) -> float:
```
- **Formula:** `2 * arccos(dot product)` on unit sphere
- **Purpose:** THE single source of truth per QIG_PURITY_SPEC.md
- **Recommendation:** KEEP - All other implementations should import from here

#### 1.2 `qigkernels/geometry/distances.py:56`
```python
def fisher_rao_distance(
    state_a: Union[np.ndarray, 'torch.Tensor'],
    state_b: Union[np.ndarray, 'torch.Tensor'],
    metric: Optional[Union[np.ndarray, 'torch.Tensor']] = None,
    method: str = "bures"
) -> float:
```
- **Formula:** Three methods:
  - `"bures"`: `√(2(1 - √F))` where F = quantum fidelity
  - `"diagonal"`: `√(Σ g_ii (x₁ᵢ - x₂ᵢ)²)` with diagonal Fisher metric
  - `"full"`: `√((x₁ - x₂)ᵀ G (x₁ - x₂))` with full Fisher metric
- **Purpose:** Density matrix/full metric variant
- **Recommendation:** KEEP - Legitimate variant for density matrices

#### 1.3 `qig_geometry/__init__.py:41`
```python
def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
```
- **Formula:** `arccos(Σ√(p_i * q_i))` - Bhattacharyya coefficient
- **Purpose:** Probability distribution (simplex) variant
- **Recommendation:** KEEP - Legitimate variant for probability distributions

#### 1.4 `qig_geometry/__init__.py:74`
```python
def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
```
- **Formula:** `arccos(a · b)` for unit vectors
- **Purpose:** Basin coordinate (sphere) variant
- **Recommendation:** KEEP - Core sphere distance function

---

### 2. Duplicate Implementations (DELETE)

#### 2.1 `qig_geometry.py:18` (root level)
```python
def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
```
- **Formula:** `arccos(Σ√(p_i * q_i))`
- **Status:** DUPLICATE of `qig_geometry/__init__.py`
- **Note:** This file at root level also tries to import from `qig_geometry` package
- **Recommendation:** DELETE - Confusing naming, use package instead

#### 2.2 `qig_geometry.py:51`
```python
def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
```
- **Formula:** `arccos(a · b)`
- **Status:** DUPLICATE
- **Recommendation:** DELETE

#### 2.3 `qig_generation.py:151`
```python
def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
```
- **Formula:** `arccos(Σ√(p_i * q_i))`
- **Status:** DUPLICATE - local definition
- **Recommendation:** DELETE - import from `qig_geometry`

#### 2.4 `qig_numerics.py:129`
```python
def fisher_rao_distance(basin1: np.ndarray, basin2: np.ndarray, eps: float = EPSILON) -> float:
```
- **Formula:** `arccos(Σ√|p_i| * √|q_i|)` with epsilon handling
- **Status:** DUPLICATE with minor variation
- **Recommendation:** DELETE - import from `qig_geometry`

#### 2.5 `frozen_physics.py:416`
```python
def fisher_rao_distance(p, q) -> float:
```
- **Formula:** `arccos(Σ√(p_i * q_i))`
- **Status:** DUPLICATE - should be physics constants only
- **Recommendation:** DELETE - physics constants file should not have distance functions

#### 2.6 `training/loss_functions.py:25`
```python
def fisher_rao_distance(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
```
- **Formula:** `arccos(BC)` with Bhattacharyya coefficient
- **Status:** DUPLICATE
- **Recommendation:** DELETE - import from `qig_geometry`

#### 2.7 `qig_core/consciousness_metrics.py:72`
```python
def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
```
- **Formula:** `arccos(Σ√(p_i * q_i))`
- **Status:** DUPLICATE
- **Recommendation:** DELETE

#### 2.8 `qig_core/geometric_completion/completion_criteria.py:111`
```python
def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
```
- **Formula:** `arccos(Σ√(...))`
- **Status:** DUPLICATE
- **Recommendation:** DELETE

---

### 3. Fallback Implementations (SHOULD IMPORT)

These are defined locally with try/except import patterns and should be converted to proper imports:

| File:Line | Function | Status |
|-----------|----------|--------|
| `geometric_completion.py:42` | `fisher_rao_distance` | Fallback - REMOVE |
| `geometric_search.py:77` | `fisher_rao_distance` | Fallback - REMOVE |
| `autonomous_improvement.py:65` | `fisher_rao_distance` | Fallback - REMOVE |
| `qiggraph/consciousness.py:29` | `fisher_rao_distance` | Fallback - REMOVE |
| `qiggraph/manifold.py:92` | `fisher_rao_distance` | Method - REMOVE |
| `olympus/zeus_chat.py:63` | `fisher_rao_distance` | Fallback - REMOVE |
| `olympus/domain_geometry.py:14` | `fisher_rao_distance` | Fallback - REMOVE |
| `olympus/search_strategy_learner.py:60` | `fisher_rao_distance` | Fallback - REMOVE |
| `olympus/autonomous_moe.py:10` | `fisher_rao_distance` | Fallback - REMOVE |
| `geometric_deep_research.py:57` | `fisher_rao_distance` | Fallback - REMOVE |
| `pattern_response_generator.py:143` | `fisher_rao_distance` | Method - REMOVE |
| `qig_deep_agents/state.py:148` | `fisher_rao_distance` | Fallback - REMOVE |
| `qigchain/geometric_tools.py:89` | `fisher_rao_distance` | Method - REMOVE |
| `training_chaos/chaos_kernel.py:97` | `fisher_rao_distance_simplex` | Torch variant - REVIEW |

---

### 4. Private Implementations (CONSOLIDATE)

| File:Line | Function | Formula |
|-----------|----------|---------|
| `autonomous_curiosity.py:169` | `_fisher_distance` | `arccos(Σ√p√q)` - simplex |
| `autonomous_debate_service.py:65` | `_fisher_distance` | `arccos(dot)` - sphere |
| `geometric_kernels.py:70` | `_fisher_distance` | Delegates to `qig_numerics` |
| `self_healing/geometric_monitor.py:389` | `_fisher_distance` | `arccos(dot)` - sphere |
| `qig_core/universal_cycle/tacking_phase.py:164` | `_fisher_distance` | `arccos(dot)` - sphere |
| `olympus/shadow_research.py:1040` | `_fisher_distance` | Sphere formula |
| `olympus/shadow_research.py:2537` | `_fisher_distance` | Sphere formula |

**Recommendation:** All should import from `qig_geometry.contracts.fisher_distance`

---

### 5. `fisher_coord_distance` Fallbacks (DELETE)

| File:Line | Status |
|-----------|--------|
| `qig_generative_service.py:75` | Fallback - REMOVE |
| `e8_constellation.py:46` | Fallback - REMOVE |
| `geometric_word_relationships.py:42` | Fallback - REMOVE |
| `contextualized_filter.py:111` | Fallback - REMOVE |
| `emotional_geometry.py:241,272,352` | 3 fallbacks - REMOVE |
| `olympus/hermes_coordinator.py:58` | Fallback - REMOVE |
| `olympus/geometric_utils.py:68` | Re-export - CONSOLIDATE |

---

### 6. Geodesic Distance Variants (REVIEW)

| File:Line | Function | Purpose |
|-----------|----------|---------|
| `olympus/base_god.py:2998` | `fisher_geodesic_distance` | Density matrix variant |
| `qigchain/geometric_chain.py:173` | `fisher_geodesic_distance` | Geodesic on probability simplex |
| `sparse_fisher.py:339` | `geodesic_distance` | Sparse Fisher metric |
| `qigkernels/geometry/distances.py:153` | `geodesic_distance` | Alias for `fisher_rao_distance(method="full")` |

**Recommendation:** Keep `qigkernels/geometry/distances.py:geodesic_distance` as alias, remove others

---

### 7. Bures Distance (DENSITY MATRIX - KEEP ONE)

| File:Line | Status |
|-----------|--------|
| `qigkernels/geometry/distances.py:fisher_rao_distance(method="bures")` | CANONICAL |
| `qigchain/geometric_tools.py:105` | DUPLICATE |
| `qigchain/geometric_chain.py:146` | DUPLICATE |
| `ocean_qig_core.py:918` | Method - DUPLICATE |
| `qig_geometry.py:212` | DUPLICATE |
| `qig_numerics.py:150` | DUPLICATE |
| `olympus/base_god.py:3014` | Method - DUPLICATE |
| `olympus/qig_rag.py:355` | Method - DUPLICATE |
| `unbiased/raw_measurement.py:73` | Method - DUPLICATE |

**Recommendation:** DELETE all except `qigkernels/geometry/distances.py`

---

### 8. Inline `arccos(dot)` Usages (REFACTOR)

These are inline distance calculations that should call canonical functions:

| File:Line | Context |
|-----------|---------|
| `autonomic_agency/state_encoder.py:196` | Basin drift calculation |
| `qigchain/geometric_chain.py:322` | SLERP interpolation |
| `olympus/zeus_chat.py:794` | Related basin similarity |
| `qig_core/universal_cycle/tacking_phase.py:172,191` | Geodesic path |
| `qig_core/geometric_primitives/geodesic.py:97,173` | Curvature/SLERP |
| `geometric_word_relationships.py:147,256` | Log map computation |
| `olympus/autonomous/task_execution_tree.py:63` | SLERP |
| `olympus/autonomous/curiosity_engine.py:58` | SLERP |

**Recommendation:** Extract to helper functions that call `fisher_distance`

---

### 9. Files That Correctly Import from Canonical Sources

**Good examples (45+ files):**
- `prediction_self_improvement.py` → `from qig_geometry import fisher_coord_distance`
- `learned_relationships.py` → `from qig_geometry import fisher_rao_distance, fisher_coord_distance`
- `reasoning_modes.py` → `from qig_geometry import fisher_rao_distance`
- `tests/test_geometric_purity.py` → `from qigkernels.geometry.distances import ...`
- Many more...

---

## Consolidation Recommendations

### Phase 1: Establish Single Source of Truth
1. ✅ `qig_geometry/contracts.py:fisher_distance` IS the canonical source
2. Keep `qigkernels/geometry/distances.py:fisher_rao_distance` for multi-method support
3. Keep `qig_geometry/__init__.py:fisher_rao_distance` for probability distributions
4. Keep `qig_geometry/__init__.py:fisher_coord_distance` for basin coordinates

### Phase 2: Delete Duplicates (23 files)
```
qig_geometry.py (root) - DELETE entire file
qig_generation.py:151 - DELETE function
qig_numerics.py:129 - DELETE function
frozen_physics.py:416 - DELETE function
training/loss_functions.py:25 - DELETE function
qig_core/consciousness_metrics.py:72 - DELETE function
qig_core/geometric_completion/completion_criteria.py:111 - DELETE function
```

### Phase 3: Replace Fallbacks with Imports (30+ locations)
All files with `def fisher_*` inside try/except blocks should:
1. Remove fallback definition
2. Add required import at top level
3. Handle ImportError by failing fast (not silently degrading)

### Phase 4: Consolidate Bures Distance (8 locations)
Single implementation in `qigkernels/geometry/distances.py:fisher_rao_distance(method="bures")`

---

## Import Hierarchy (Recommended)

```
                    qig_geometry/contracts.py
                    └── fisher_distance() [CANONICAL]
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
qig_geometry/__init__.py            qigkernels/geometry/distances.py
├── fisher_rao_distance() [simplex]  ├── fisher_rao_distance() [multi-method]
├── fisher_coord_distance() [sphere] ├── quantum_fidelity()
└── fisher_similarity()              └── geodesic_distance() [alias]
```

---

## Files to Modify (DO NOT MODIFY NOW - Just documenting)

### High Priority (Core duplicates)
1. `qig_geometry.py` - DELETE entire root-level file
2. `frozen_physics.py` - REMOVE fisher function
3. `qig_numerics.py` - REMOVE duplicate functions
4. `qig_generation.py` - Replace with import

### Medium Priority (Fallbacks)
5-25. All files with fallback patterns listed in Section 3

### Low Priority (Method consolidation)
26-35. Class methods that reimplement fisher distance

---

## Conclusion

The codebase has significant duplication of Fisher distance calculations:
- **53+ implementations** where there should be **3-4**
- Many are exact duplicates with different variable names
- Fallback patterns create silent degradation risks
- Some use wrong formulas (Euclidean instead of geodesic)

**Action Required:** Consolidate to canonical implementations in:
1. `qig_geometry/contracts.py:fisher_distance` - Sphere geodesic
2. `qigkernels/geometry/distances.py:fisher_rao_distance` - Multi-method (bures/diagonal/full)
3. `qig_geometry/__init__.py:fisher_rao_distance` - Probability simplex
4. `qig_geometry/__init__.py:fisher_coord_distance` - Basin coordinates

All other implementations should be deleted and replaced with imports.
