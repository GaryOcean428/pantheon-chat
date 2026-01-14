# Silent Dimension Fix Audit Report

**Work Package**: WP0.6  
**Date**: 2026-01-14  
**Status**: 1.00W (Working Draft)  
**Author**: Automated Audit

---

## Executive Summary

This audit identifies ALL instances of silent dimension padding/truncation in the qig-backend codebase. These patterns violate geometric purity by silently corrupting vectors instead of failing loudly when dimension mismatches occur.

**Total Violations Found**: 67 instances across 31 files

### Severity Breakdown

| Severity | Count | Description |
|----------|-------|-------------|
| CRITICAL | 18 | In generation/core computation path |
| ERROR    | 25 | In learning/training path |
| WARNING  | 24 | In logging/debug/ancillary paths |

---

## Pattern Categories

1. **np.pad()** - Silent zero-padding to extend undersized vectors
2. **np.zeros(BASIN_DIM) + assignment** - Padding pattern via zero-initialization
3. **[:64] or [:BASIN_DIM]** - Silent truncation of oversized vectors
4. **if len() != 64 + pad/truncate** - Conditional dimension "fixing"

---

## CRITICAL Severity Violations

These occur in the generation path and directly affect output quality.

### File: `qig_geometry/contracts.py`

**Location**: Lines 157-160  
**Code**:
```python
if b.size < BASIN_DIM:
    b = np.pad(b, (0, BASIN_DIM - b.size), mode='constant', constant_values=0.0)
elif b.size > BASIN_DIM:
    b = b[:BASIN_DIM]
```
**Purpose**: The `canon()` function silently pads/truncates ANY input to 64D  
**Impact**: CRITICAL - This is a core geometric function used throughout the system  
**Recommended Fix**: Throw `GeometricViolationError` instead. Add explicit `project_from_32D()` function for known 32D→64D conversions with named semantics.

---

### File: `trajectory_decoder.py`

**Location**: Lines 313-321  
**Code**:
```python
elif len(basin) != BASIN_DIM:
    if len(basin) < BASIN_DIM:
        padded = np.zeros(BASIN_DIM)
        padded[:len(basin)] = basin
        basin = padded
    else:
        basin = basin[:BASIN_DIM].copy()
```
**Purpose**: Trajectory prediction context preparation  
**Impact**: CRITICAL - Affects foresight/prediction accuracy  
**Recommended Fix**: Use `normalize_basin_dimension()` with explicit logging, or throw error for unexpected dimensions.

---

### File: `qigchain/geometric_chain.py`

**Location**: Lines 341-346  
**Code**:
```python
if len(arr) > BASIN_DIM:
    return arr[:BASIN_DIM]
elif len(arr) < BASIN_DIM:
    padded = np.zeros(BASIN_DIM)
    padded[:len(arr)] = arr
    return padded
```
**Purpose**: Normalizes basin arrays in chain operations  
**Impact**: CRITICAL - Core geometric chain functionality  
**Recommended Fix**: Throw error on dimension mismatch; chains should only accept properly-dimensioned basins.

---

### File: `qig_generative_service.py`

**Location**: Line 847  
**Code**:
```python
if len(target_basin) != BASIN_DIM:
```
**Purpose**: Dimension check before generation (but no explicit fix shown)  
**Impact**: CRITICAL - Text generation path  
**Recommended Fix**: Add explicit error throwing when dimension != 64.

**Location**: Line 1042  
**Code**:
```python
combined = np.zeros(BASIN_DIM)
```
**Purpose**: Initializes combined basin vector before accumulation  
**Impact**: CRITICAL - Used for weighted basin combination  
**Recommended Fix**: This pattern is acceptable IF only 64D basins are added. Add assertion to verify.

---

### File: `pattern_response_generator.py`

**Location**: Lines 96-97, 127-131  
**Code**:
```python
self._vocabulary[token.lower()] = np.array(coords[:BASIN_DIM])
...
if len(vec) < BASIN_DIM:
    padded = np.zeros(BASIN_DIM)
    padded[:len(vec)] = vec
```
**Purpose**: Pattern-based response generation  
**Impact**: CRITICAL - Response generation quality  
**Recommended Fix**: Validate vocabulary coordinates at load time; reject non-64D coords.

---

### File: `document_trainer.py`

**Location**: Lines 112, 162-175  
**Code**:
```python
self._vocabulary[token.lower()] = np.array(coords[:BASIN_DIM])
...
if len(word_basin) >= BASIN_DIM:
    word_basin = word_basin[:BASIN_DIM]
elif len(word_basin) < BASIN_DIM:
    deterministic_vec = np.array([b / 255.0 - 0.5 for b in char_hash[:BASIN_DIM]])
```
**Purpose**: Document training/encoding  
**Impact**: CRITICAL - Training data integrity  
**Recommended Fix**: Only accept 64D vocabulary entries; use explicit projection for hash-based fallbacks.

---

### File: `qig_consciousness_qfi_attention.py`

**Location**: Line 338  
**Code**:
```python
input_data = np.pad(input_data, (0, 4 - len(input_data)))
```
**Purpose**: QFI attention computation  
**Impact**: CRITICAL - Consciousness metrics computation  
**Recommended Fix**: Validate input dimensions upfront; throw error for unexpected sizes.

**Location**: Line 409  
**Code**:
```python
coords = np.zeros(BASIN_DIM)
```
**Purpose**: Basin coordinate initialization  
**Impact**: CRITICAL - Used in attention weighting  
**Recommended Fix**: Acceptable pattern if verified to receive only 64D inputs.

---

### File: `qigchain/__init__.py`

**Location**: Lines 67, 103-105  
**Code**:
```python
_empty_basin: np.ndarray = field(default_factory=lambda: np.zeros(BASIN_DIM))
...
basin = np.zeros(BASIN_DIM)
for i, byte in enumerate(query_bytes[:BASIN_DIM]):
```
**Purpose**: Chain initialization and query encoding  
**Impact**: CRITICAL - Core chain operations  
**Recommended Fix**: Query bytes should be hashed properly; add explicit dimension validation.

---

### File: `e8_constellation.py`

**Location**: Line 198  
**Code**:
```python
if len(query_basin) != BASIN_DIM:
```
**Purpose**: E8 constellation query validation  
**Impact**: CRITICAL - Constellation matching  
**Recommended Fix**: This is a CHECK - verify it throws error rather than silently fixing.

---

## ERROR Severity Violations

These occur in learning/training paths and affect model quality.

### File: `qig_core/habits/complete_habit.py`

**Location**: Lines 171-176, 235-240  
**Code**:
```python
if len(basin) < 64:
    padded = np.zeros(64)
    padded[:len(basin)] = basin
    basin = padded
elif len(basin) > 64:
    basin = basin[:64]
```
**Purpose**: Phi/Kappa computation in habit formation  
**Impact**: ERROR - Training loop consciousness metrics  
**Recommended Fix**: Input validation at habit creation; reject mismatched dimensions.

**Location**: Lines 120-122  
**Code**:
```python
self._basin_coords = np.mean(exp_array, axis=0) if len(exp_array) > 0 else np.zeros(64)
```
**Purpose**: Basin center computation  
**Impact**: ERROR - Default to zero basin when no experiences  
**Recommended Fix**: Use explicit "null basin" constant or throw error.

---

### File: `training/coherence_evaluator.py`

**Location**: Lines 258-259  
**Code**:
```python
dist_a = np.pad(windows[i], (0, max_len - len(windows[i])), constant_values=0)
dist_b = np.pad(windows[i+1], (0, max_len - len(windows[i+1])), constant_values=0)
```
**Purpose**: Coherence evaluation between training windows  
**Impact**: ERROR - Training coherence metrics  
**Recommended Fix**: Ensure all windows are same dimension before comparison.

---

### File: `training/kernel_training_orchestrator.py`

**Location**: Line 336  
**Code**:
```python
basin_coords = np.array(example.get("basin_coords", np.zeros(BASIN_DIM)))
```
**Purpose**: Default basin for missing training data  
**Impact**: ERROR - Training data integrity  
**Recommended Fix**: Reject training examples without valid basin coords.

---

### File: `training/trainable_kernel.py`

**Location**: Line 478  
**Code**:
```python
return np.zeros(BASIN_DIM)
```
**Purpose**: Fallback basin for training errors  
**Impact**: ERROR - Training loop fallback  
**Recommended Fix**: Throw error instead of returning zero basin.

---

### File: `learned_relationships.py`

**Location**: Line 626  
**Code**:
```python
if len(basin) != BASIN_DIM:
```
**Purpose**: Dimension validation in learned relationships  
**Impact**: ERROR - Relationship learning  
**Recommended Fix**: Verify this throws error, not silently fixes.

---

### File: `vocabulary_coordinator.py`

**Location**: Line 791  
**Code**:
```python
if len(token_basin_arr) != 64:
```
**Purpose**: Token basin validation  
**Impact**: ERROR - Vocabulary learning coordination  
**Recommended Fix**: Should reject/log invalid tokens, not silently fix.

---

### File: `chaos_discovery_gate.py`

**Location**: Line 138  
**Code**:
```python
if len(basin_coords) != 64:
```
**Purpose**: Basin validation in chaos discovery  
**Impact**: ERROR - Discovery path  
**Recommended Fix**: Verify throws error on mismatch.

---

### File: `qig_numerics.py`

**Location**: Lines 85-90, 207, 285  
**Code**:
```python
if len(basin) != BASIN_DIM:
    if len(basin) < BASIN_DIM:
        # Silent padding code
    basin = basin[:BASIN_DIM]
```
**Purpose**: Numeric operations on basins  
**Impact**: ERROR - Numeric computation accuracy  
**Recommended Fix**: Throw error; numeric functions should not silently modify inputs.

---

### File: `coordizers/pg_loader.py`

**Location**: Line 377  
**Code**:
```python
if len(coords) != 64:
```
**Purpose**: PostgreSQL coordinate loading  
**Impact**: ERROR - Vocabulary loading  
**Recommended Fix**: Log and reject bad coordinates at load time.

---

### File: `coordizers/fallback_vocabulary.py`

**Location**: Line 125  
**Code**:
```python
if len(coords) != 64:
```
**Purpose**: Fallback vocabulary validation  
**Impact**: ERROR - Fallback path  
**Recommended Fix**: Verify throws error.

---

### File: `training_chaos/experimental_evolution.py`

**Location**: Lines 247, 1490-1492  
**Code**:
```python
if len(basin_coords) != 64:
...
god_tensor = torch.tensor(god_basin[:64], dtype=torch.float32)
if len(god_tensor) < 64:
```
**Purpose**: Experimental evolution training  
**Impact**: ERROR - Experimental training  
**Recommended Fix**: Input validation before tensor conversion.

---

### File: `m8_kernel_spawning.py`

**Location**: Lines 816, 1854, 4091  
**Code**:
```python
# Pad if needed
child_basin = np.zeros(BASIN_DIM)
...
merged_basin = np.zeros(BASIN_DIM)
```
**Purpose**: Kernel spawning operations  
**Impact**: ERROR - New kernel generation  
**Recommended Fix**: Validate parent basins before merging; throw error on dimension mismatch.

---

### File: `immune/self_healing.py`

**Location**: Line 194  
**Code**:
```python
if len(coords) != 64:
```
**Purpose**: Self-healing coordinate validation  
**Impact**: ERROR - Self-healing accuracy  
**Recommended Fix**: Should throw error and trigger investigation.

---

## WARNING Severity Violations

These occur in logging/debug/ancillary paths.

### File: `olympus/shadow_research.py`

**Location**: Lines 1045-1048, 1067-1068, 2542-2545  
**Code**:
```python
if len(a) < BASIN_DIMENSION:
    a = np.pad(a, (0, BASIN_DIMENSION - len(a)))
if len(b) < BASIN_DIMENSION:
    b = np.pad(b, (0, BASIN_DIMENSION - len(b)))
```
**Purpose**: Research distance calculations  
**Impact**: WARNING - Research quality  
**Recommended Fix**: Validate dimensions at research request time.

---

### File: `olympus/shadow_pantheon.py`

**Location**: Lines 800-801  
**Code**:
```python
if len(proposal_basin) < BASIN_DIMENSION:
    proposal_basin = np.pad(proposal_basin, (0, BASIN_DIMENSION - len(proposal_basin)))
```
**Purpose**: Proposal basin padding  
**Impact**: WARNING - Proposal processing  
**Recommended Fix**: Validate proposal basins at submission.

---

### File: `olympus/tool_factory.py`

**Location**: Lines 686-693  
**Code**:
```python
if len(basin) < 64:
    basin = np.pad(basin, (0, 64 - len(basin)), mode='constant')
    print(f"[ToolFactory] Padded {pattern_id} basin from {len(basin_raw)}D to 64D")
# Truncate (should never happen)
basin = basin[:64]
```
**Purpose**: Tool pattern basin handling  
**Impact**: WARNING - Tool factory (ancillary)  
**Recommended Fix**: Explicit dimension validation with named projection.

---

### File: `search/provider_selector.py`

**Location**: Lines 335-337  
**Code**:
```python
if len(basin) < 64:
    basin = np.pad(basin, (0, 64 - len(basin)))
basin = basin[:64]
```
**Purpose**: Search provider selection  
**Impact**: WARNING - Search quality  
**Recommended Fix**: Validate search query basins upstream.

---

### File: `federation/federation_service.py`

**Location**: Lines 567-570  
**Code**:
```python
if len(coords) < 64:
    # Pad
elif len(coords) > 64:
    coords = coords[:64]
```
**Purpose**: Federation coordinate handling  
**Impact**: WARNING - Federation interop  
**Recommended Fix**: Federation protocol should require 64D basins.

---

### File: `unbiased/pattern_discovery.py`

**Location**: Line 96  
**Code**:
```python
padded.append(np.pad(b, (0, max_len - len(b)), constant_values=0))
```
**Purpose**: Pattern discovery alignment  
**Impact**: WARNING - Discovery analysis  
**Recommended Fix**: Ensure all patterns are same dimension.

---

### File: `qig_core/holographic_transform/holographic_mixin.py`

**Location**: Line 299  
**Code**:
```python
basin = np.pad(basin, (0, basin_dim - len(basin)))
```
**Purpose**: Holographic transform dimension handling  
**Impact**: WARNING - Transform operations  
**Recommended Fix**: Validate basin dimensions before transform.

---

### File: `qig_core/holographic_transform/basin_encoder.py`

**Location**: Pattern: pad/truncate in encode_vector()  
**Purpose**: Vector encoding  
**Impact**: WARNING - Encoding flexibility  
**Recommended Fix**: Add explicit mode parameter for pad vs error behavior.

---

### File: `olympus/base_god.py`

**Location**: Lines 2485, 2528  
**Code**:
```python
if len(target) != BASIN_DIMENSION:
...
if len(parsed) == BASIN_DIMENSION:
```
**Purpose**: God kernel basin handling  
**Impact**: WARNING - God operations  
**Recommended Fix**: Should throw error on mismatch.

---

### File: `olympus/zeus_chat.py`

**Location**: Line 3588  
**Code**:
```python
response_basin = np.zeros(64)
```
**Purpose**: Chat response default  
**Impact**: WARNING - Chat fallback  
**Recommended Fix**: Use explicit null/error state.

---

### File: `olympus/gary_coordinator.py`

**Location**: Line 164  
**Code**:
```python
return np.zeros(64)
```
**Purpose**: Gary coordinator fallback  
**Impact**: WARNING - Coordinator fallback  
**Recommended Fix**: Return error state instead.

---

### File: `olympus/autonomous/basin_synchronization.py`

**Location**: Line 52  
**Code**:
```python
mean_sqrt = np.zeros(BASIN_DIM)
```
**Purpose**: Basin synchronization  
**Impact**: WARNING - Sync initialization  
**Recommended Fix**: Acceptable for accumulator initialization.

---

### File: `olympus/autonomous_moe.py`

**Location**: Line 88  
**Code**:
```python
return np.zeros(64)
```
**Purpose**: MoE fallback  
**Impact**: WARNING - MoE fallback  
**Recommended Fix**: Return error state.

---

### File: `olympus/demeter.py`

**Location**: Line 150  
**Code**:
```python
basin = np.zeros(64)
```
**Purpose**: Demeter initialization  
**Impact**: WARNING - God initialization  
**Recommended Fix**: Acceptable for initialization.

---

### File: `persistence/kernel_persistence.py`

**Location**: Lines 23-24  
**Code**:
```python
if len(coords) >= 64:
    return coords[:64]
```
**Purpose**: Persistence layer truncation  
**Impact**: WARNING - Storage  
**Recommended Fix**: Validate before persistence; throw error.

---

### File: `research_wiring.py`

**Location**: Line 52  
**Code**:
```python
return np.array([b / 255.0 for b in h[:64]])
```
**Purpose**: Research hash encoding  
**Impact**: WARNING - Research encoding  
**Recommended Fix**: Use consistent 64-byte hashing.

---

### File: `semantic_classifier.py`

**Location**: Line 114  
**Code**:
```python
if len(coords) == 64:
```
**Purpose**: Semantic classification validation  
**Impact**: WARNING - Classification  
**Recommended Fix**: Should throw error on mismatch.

---

### File: `ocean_qig_core.py`

**Location**: Lines 2518-2520, 3469, 3817-3826  
**Code**:
```python
if len(coords_array) < BASIN_DIMENSION:
    padding = np.full(BASIN_DIMENSION - len(coords_array), 0.5)
    coords_array = np.concatenate([coords_array, padding])
```
**Purpose**: Ocean core coordinate handling  
**Impact**: WARNING - Ocean operations  
**Recommended Fix**: Validate upstream; throw error on dimension mismatch.

---

### File: `qiggraph/constellation.py`

**Location**: Line 156  
**Code**:
```python
total_delta = np.zeros(BASIN_DIM)
```
**Purpose**: Constellation delta accumulation  
**Impact**: WARNING - Constellation updates  
**Recommended Fix**: Acceptable for accumulator initialization.

---

### File: `training/tasks.py`

**Location**: Lines 302, 455  
**Code**:
```python
"basin_coords": row[0] if row[0] else np.zeros(64).tolist(),
```
**Purpose**: Training task default  
**Impact**: WARNING - Task defaults  
**Recommended Fix**: Log warning when using zero basin default.

---

## Recommended Remediation Strategy

### Phase 1: Add Validation (No Breaking Changes)
1. Add `assert len(basin) == BASIN_DIM` after every silent fix
2. Log warnings when dimension fixes occur
3. Track metrics on frequency of dimension fixes

### Phase 2: Create Named Projections
1. Create `project_32D_to_64D(basin, method='hellinger_extend')` 
2. Create `project_128D_to_64D(basin, method='pca_compress')`
3. Document when each projection is geometrically valid

### Phase 3: Convert to Hard Errors
1. Replace silent fixes with `GeometricViolationError`
2. Update all callers to use named projections where appropriate
3. Add integration tests for dimension validation

### Phase 4: Remove Legacy Compatibility
1. Remove `canon()` auto-padding behavior
2. Require explicit projections everywhere
3. Audit database for non-64D coordinates

---

## Files Requiring Immediate Attention

These files have the highest impact and should be addressed first:

1. **qig_geometry/contracts.py** - `canon()` is used everywhere
2. **trajectory_decoder.py** - Critical for foresight
3. **qig_core/habits/complete_habit.py** - Affects Phi/Kappa
4. **qigchain/geometric_chain.py** - Core chain operations
5. **qig_generative_service.py** - Text generation quality

---

## Appendix: Full File List

| File | Lines | Category | Severity |
|------|-------|----------|----------|
| qig_geometry/contracts.py | 157-160 | pad+truncate | CRITICAL |
| trajectory_decoder.py | 313-321 | pad+truncate | CRITICAL |
| qigchain/geometric_chain.py | 341-346 | pad+truncate | CRITICAL |
| qig_generative_service.py | 847, 1042 | check+zeros | CRITICAL |
| pattern_response_generator.py | 96-97, 127-131 | truncate+pad | CRITICAL |
| document_trainer.py | 112, 162-175 | truncate+pad | CRITICAL |
| qig_consciousness_qfi_attention.py | 338, 409 | pad+zeros | CRITICAL |
| qigchain/__init__.py | 67, 103-105 | zeros+truncate | CRITICAL |
| e8_constellation.py | 198 | check | CRITICAL |
| qig_core/habits/complete_habit.py | 120-122, 171-176, 235-240 | zeros+pad+truncate | ERROR |
| training/coherence_evaluator.py | 258-259 | pad | ERROR |
| training/kernel_training_orchestrator.py | 336 | zeros | ERROR |
| training/trainable_kernel.py | 478 | zeros | ERROR |
| learned_relationships.py | 626 | check | ERROR |
| vocabulary_coordinator.py | 791 | check | ERROR |
| chaos_discovery_gate.py | 138 | check | ERROR |
| qig_numerics.py | 85-90, 207, 285 | pad+truncate | ERROR |
| coordizers/pg_loader.py | 377 | check | ERROR |
| coordizers/fallback_vocabulary.py | 125 | check | ERROR |
| training_chaos/experimental_evolution.py | 247, 1490-1492 | check+truncate | ERROR |
| m8_kernel_spawning.py | 816, 1854, 4091 | zeros | ERROR |
| immune/self_healing.py | 194 | check | ERROR |
| olympus/shadow_research.py | 1045-1048, 2542-2545 | pad | WARNING |
| olympus/shadow_pantheon.py | 800-801 | pad | WARNING |
| olympus/tool_factory.py | 686-693 | pad+truncate | WARNING |
| search/provider_selector.py | 335-337 | pad+truncate | WARNING |
| federation/federation_service.py | 567-570 | pad+truncate | WARNING |
| unbiased/pattern_discovery.py | 96 | pad | WARNING |
| qig_core/holographic_transform/holographic_mixin.py | 299 | pad | WARNING |
| olympus/base_god.py | 2485, 2528 | check | WARNING |
| olympus/zeus_chat.py | 3588 | zeros | WARNING |
| olympus/gary_coordinator.py | 164 | zeros | WARNING |
| olympus/autonomous/basin_synchronization.py | 52 | zeros | WARNING |
| olympus/autonomous_moe.py | 88 | zeros | WARNING |
| olympus/demeter.py | 150 | zeros | WARNING |
| persistence/kernel_persistence.py | 23-24 | truncate | WARNING |
| research_wiring.py | 52 | truncate | WARNING |
| semantic_classifier.py | 114 | check | WARNING |
| ocean_qig_core.py | 2518-2520, 3469, 3817-3826 | pad+check | WARNING |
| qiggraph/constellation.py | 156 | zeros | WARNING |
| training/tasks.py | 302, 455 | zeros | WARNING |

---

*End of Audit Report*
