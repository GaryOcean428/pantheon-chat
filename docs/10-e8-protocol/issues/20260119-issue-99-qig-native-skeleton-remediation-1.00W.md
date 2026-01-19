# Issue #99: Implement QIG-Native Skeleton (E8 Issue-03)

## Priority
**P1 - HIGH**

## Type
`type: implementation`, `qig-purity`, `geometric-self-sufficiency`, `e8-protocol`

## Objective
Replace external NLP dependencies (spacy, nltk, LLMs) with internal geometric token_role system and Fisher-Rao trajectory prediction for generation structure per E8 Protocol Issue-03.

## Problem
Current generation pipeline relies on external tools for structure extraction:
- spacy/nltk for POS tagging
- External LLM calls for skeleton generation
- Template fallbacks when tools unavailable

This breaks geometric purity and prevents self-sufficiency.

## Context
- **E8 Protocol Spec:** `docs/10-e8-protocol/issues/20260116-issue-03-qig-native-skeleton-1.01W.md`
- **Related GitHub Issues:** #92, #90
- **Phase:** 3 (Coherence Architecture - Geometric Self-Sufficiency)
- **Assessment:** `docs/10-e8-protocol/IMPLEMENTATION_ASSESSMENT.md`

## Tasks

### 1. Token Role Learner (Geometric Clustering)
- [ ] Create `qig-backend/generation/token_role_learner.py`
- [ ] Implement `derive_token_role(word)` using Fisher-Rao neighborhood clustering
- [ ] Cluster vocabulary by geometric similarity (FR distance < threshold)
- [ ] Assign role labels: function, content, transition, anchor, etc.
- [ ] Store in `coordizer_vocabulary.token_role` column
- [ ] Backfill existing vocabulary

### 2. Foresight Predictor (Trajectory Regression)
- [ ] Create `qig-backend/generation/foresight_predictor.py`
- [ ] Implement `predict_next_basin(trajectory)` using geodesic extrapolation
- [ ] Use last 3-5 basins to compute velocity vector
- [ ] Project forward along geodesic on probability simplex
- [ ] Return predicted basin coordinates (not word)

### 3. Unified Generation Pipeline
- [ ] Create `qig-backend/generation/unified_pipeline.py`
- [ ] Integrate: token_role skeleton + foresight prediction + geometric selection
- [ ] Remove dependency on external POS taggers
- [ ] Remove external LLM calls for structure
- [ ] Implement geometric backoff (not template fallback)

### 4. Remove External Dependencies
- [ ] Audit codebase for spacy imports
- [ ] Audit codebase for nltk imports
- [ ] Audit codebase for external LLM calls in generation
- [ ] Replace or remove each dependency
- [ ] Update requirements.txt to remove unneeded packages

### 5. QIG Purity Mode Enforcement
- [ ] Create `qig-backend/purity/enforce.py`
- [ ] Implement `@require_qig_purity` decorator
- [ ] Check `QIG_PURITY_MODE` environment variable
- [ ] Block external calls when mode enabled
- [ ] Add validation at CI level

### 6. Integration & Testing
- [ ] Add unit tests for token_role derivation
- [ ] Add unit tests for foresight prediction
- [ ] Add integration test for unified pipeline
- [ ] Test generation in `QIG_PURITY_MODE=true`
- [ ] Verify quality matches or exceeds external NLP

## Deliverables

| File | Description | Status |
|------|-------------|--------|
| `qig-backend/generation/token_role_learner.py` | Geometric role derivation | ❌ TODO |
| `qig-backend/generation/foresight_predictor.py` | Trajectory prediction | ❌ TODO |
| `qig-backend/generation/unified_pipeline.py` | Integrated generation | ❌ TODO |
| `qig-backend/purity/enforce.py` | QIG_PURITY_MODE checks | ❌ TODO |
| `qig-backend/tests/test_qig_generation.py` | Generation tests | ❌ TODO |
| `docs/03-technical/qig-native-generation.md` | Documentation | ❌ TODO |

## Acceptance Criteria
- [ ] Geometric `token_role` derived and backfilled for all vocabulary
- [ ] Skeleton generation uses roles (not POS tags)
- [ ] Foresight prediction implemented using Fisher-Rao trajectory
- [ ] NO spacy imports in generation code
- [ ] NO nltk imports in generation code
- [ ] NO external LLM calls in QIG_PURITY_MODE
- [ ] Unified pipeline produces coherent output without external deps
- [ ] Tests pass in QIG_PURITY_MODE
- [ ] Generation quality validated against baseline

## Dependencies
- **Requires:** Issue #97 (QFI Integrity), Issue #98 (Strict Simplex)
- **Blocks:** Full E8 Protocol compliance
- **Related:** Issue #90 (Complete QIG-Pure Generation Architecture)

## Token Role Categories

**Proposed Geometric Roles:**
- **FUNCTION**: High-frequency, low-dimension basins (connectives, determiners)
- **CONTENT**: Low-frequency, high-dimension basins (nouns, verbs)
- **TRANSITION**: Bridging words between semantic clusters
- **ANCHOR**: High curvature words (negation, modality)
- **MODIFIER**: Words that adjust trajectory direction

**Derivation Method:**
1. Compute Fisher-Rao distance matrix for vocabulary
2. Cluster by distance threshold (e.g., 0.3)
3. Analyze cluster properties (size, density, connectivity)
4. Assign role based on cluster characteristics

## Foresight Prediction Algorithm

```python
def predict_next_basin(trajectory: List[np.ndarray]) -> np.ndarray:
    """
    Predict next basin via geodesic extrapolation.
    
    Uses last 3-5 basins to compute velocity, projects forward.
    """
    if len(trajectory) == 0:
        raise ValueError("Cannot predict from empty trajectory")
    
    if len(trajectory) < 2:
        # No velocity available
        return trajectory[-1]  # Stay at current basin
    
    # Compute velocity in sqrt-space (tangent space of unit sphere)
    sqrt_curr = np.sqrt(trajectory[-1])
    sqrt_prev = np.sqrt(trajectory[-2])
    velocity = sqrt_curr - sqrt_prev
    
    # Project forward
    sqrt_next = sqrt_curr + velocity
    
    # Check for zero-length vector (degenerate case)
    norm_next = np.linalg.norm(sqrt_next)
    if norm_next == 0:
        return trajectory[-1]  # Stay at current basin
    
    # Renormalize to unit sphere
    sqrt_next = sqrt_next / norm_next
    
    # Transform back to simplex
    next_basin = sqrt_next ** 2
    
    return next_basin
```

## Validation Commands
```bash
# Test token role derivation
python qig-backend/generation/token_role_learner.py --test

# Test foresight prediction
pytest qig-backend/tests/test_foresight.py -v

# Test unified pipeline
python qig-backend/generation/unified_pipeline.py --input "test query" --verbose

# Test in QIG purity mode
QIG_PURITY_MODE=true python qig-backend/tests/test_qig_generation.py

# Audit external dependencies
grep -r "import spacy" qig-backend/
grep -r "import nltk" qig-backend/
grep -r "openai.ChatCompletion" qig-backend/generation/
```

## References
- **E8 Protocol Universal Spec:** §0 (Purity Rule: No external NLP)
- **Issue Spec:** `docs/10-e8-protocol/issues/20260116-issue-03-qig-native-skeleton-1.01W.md`
- **Generation Architecture:** Issue #90 (Plan→Realize→Repair)
- **Assessment:** `docs/10-e8-protocol/IMPLEMENTATION_ASSESSMENT.md` Section 4

## Estimated Effort
**3-4 days** (per E8 Protocol Phase 3 estimate)

---

**Status:** TO DO  
**Created:** 2026-01-19  
**Priority:** P1 - HIGH  
**Phase:** 3 - Coherence Architecture
