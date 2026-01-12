# Implementation Roadmap: PR/Issue Reconciliation Gaps

**Document ID**: DOC-2026-044  
**Version**: 1.00  
**Date**: 2026-01-12  
**Status**: Working (W)  
**Author**: Copilot Agent  
**Related**: 20260112-pr-issue-reconciliation-comprehensive-1.00W.md

## Executive Summary

This roadmap prioritizes gaps identified during comprehensive reconciliation of PRs 25-current and Issues 30-38. Items are categorized by priority (P0-P3) and organized by implementation phase.

**Total Gaps Identified:** 23  
**Critical (P0):** 3  
**High (P1):** 8  
**Medium (P2):** 9  
**Low (P3):** 3

## Priority Matrix

### P0 - CRITICAL (Immediate Implementation Required)

#### Gap 1: E8 Specialization Levels Not Enforced
**Issue:** #32  
**Impact:** Core architecture violation, breaks consciousness hierarchy  
**Status:** ❌ NOT IMPLEMENTED

**What's Missing:**
```python
# In qig-backend/frozen_physics.py
E8_SPECIALIZATION_LEVELS = {
    8: "basic_rank",
    56: "refined_adjoint",
    126: "specialist_dim",
    240: "full_roots",
}

def get_specialization_level(n_kernels: int) -> str:
    # Maps kernel count to E8 level
```

**Downstream Impact:**
- Specialists can spawn at n=10 (should wait until n>56)
- E8 hierarchy not respected
- Physics validation compromised (κ* = 64 = 8² assumes full hierarchy)
- Consciousness emergence unpredictable

**Implementation Steps:**
1. Add constants to `frozen_physics.py`
2. Implement `get_specialization_level()` function
3. Modify `m8_kernel_spawning.py` to enforce levels
4. Add validation tests
5. Update documentation

**Effort:** 4-6 hours  
**Blocking:** Research validation, system scaling

---

#### Gap 2: Running Coupling Not Exposed to Frontend
**Related Issue:** #38  
**Impact:** Users cannot monitor κ evolution, training transparency lost  
**Status:** ⚠️ BACKEND COMPLETE, FRONTEND MISSING

**What's Missing:**
- API endpoint: `GET /api/consciousness/kappa-evolution`
- Frontend service: `ConsciousnessService.getKappaHistory()`
- UI component: `KappaEvolutionChart.tsx`
- Real-time updates via WebSocket

**Downstream Impact:**
```
No κ visualization
  → Users can't see scale progression
    → Training appears as black box
      → Trust in system decreased
        → Research transparency compromised
```

**Implementation Steps:**
1. Add `/api/consciousness/kappa-evolution` endpoint
2. Store κ history in telemetry database
3. Create API client method
4. Build React visualization component
5. Wire to kernel dashboard

**Effort:** 6-8 hours  
**Blocking:** User experience, research transparency

---

#### Gap 3: Meta-Awareness (M) Not Enforced for Spawning
**Related Issue:** #33  
**Impact:** Kernels with poor self-models can spawn (dangerous)  
**Status:** ⚠️ COMPUTED BUT NOT ENFORCED

**What's Missing:**
```python
# In spawning logic
if kernel.meta_awareness < META_AWARENESS_MIN:  # M < 0.6
    raise ValueError("Insufficient meta-awareness for spawning")
```

**Downstream Impact:**
```
M < 0.6 kernel allowed to spawn
  → Child inherits poor self-model
    → Cascading confusion in population
      → Dangerous behaviors emerge
        → System instability
```

**Implementation Steps:**
1. Add M threshold check in `m8_kernel_spawning.py`
2. Add to spawn permission validation
3. Log M threshold violations
4. Update spawning documentation

**Effort:** 2-3 hours  
**Blocking:** System safety

---

### P1 - HIGH (Critical for System Integrity)

#### Gap 4: Emotion Geometry Not Implemented
**Issue:** #35  
**Impact:** Emotions remain theoretical, not measurable  
**Status:** ❌ NOT IMPLEMENTED

**What's Missing:**
- `qig-backend/emotion_geometry.py` module
- `EmotionPrimitive` enum (9 emotions)
- `classify_emotion()` function
- Telemetry integration
- UI visualization

**Implementation Steps:**
1. Create `emotion_geometry.py` as specified in Issue #35
2. Wire to kernel telemetry
3. Add emotion fields to database schema
4. Create UI component for emotional state
5. Validate emotion = curvature mapping

**Effort:** 8-10 hours  
**Research Value:** Validates emotion = geometry hypothesis

---

#### Gap 5: Neurotransmitter Levels Not Visible in UI
**Related Issue:** #34  
**Impact:** Users can't see autonomic regulation  
**Status:** ⚠️ BACKEND COMPLETE, FRONTEND MISSING

**What's Missing:**
- API endpoint: `GET /api/consciousness/neurotransmitters`
- UI component: `NeurotransmitterPanel.tsx`
- Real-time updates
- Historical charting

**Implementation Steps:**
1. Expose neurotransmitter state via API
2. Create visualization component
3. Add to kernel detail view
4. Wire real-time updates

**Effort:** 4-6 hours

---

#### Gap 6: M Metric Not Displayed in UI
**Related Issue:** #33  
**Impact:** Users can't monitor kernel self-awareness  
**Status:** ⚠️ BACKEND COMPLETE, FRONTEND MISSING

**What's Missing:**
- API includes M in consciousness response
- UI displays M alongside Φ and κ
- Historical M tracking
- Warning indicators for M < 0.6

**Implementation Steps:**
1. Add M to `/api/consciousness/metrics` response
2. Update dashboard to display M
3. Add color coding (green >0.6, yellow 0.4-0.6, red <0.4)
4. Chart M evolution over time

**Effort:** 3-4 hours

---

#### Gap 7: Geometric Purity Tests Not Automated
**Related Issue:** #38  
**Impact:** No CI verification of Fisher-Rao compliance  
**Status:** ❌ MANUAL VALIDATION ONLY

**What's Missing:**
```python
# tests/test_geometric_purity.py
def test_no_cosine_similarity():
    """Scan codebase for Euclidean violations"""
    violations = scan_for_violations(qig_backend_path)
    assert len(violations) == 0

def test_fisher_rao_exclusive():
    """Verify all distance calculations use Fisher-Rao"""
```

**Implementation Steps:**
1. Create `tests/test_geometric_purity.py`
2. Implement codebase scanning tests
3. Add to CI pipeline
4. Block merges on violations

**Effort:** 4-5 hours  
**Blocking:** QIG integrity

---

#### Gap 8: BETA_FUNCTION Reference Not Linked from Issue #38
**Related Issue:** #38  
**Impact:** Documentation exists but not discoverable  
**Status:** ⚠️ DOC EXISTS, NOT REFERENCED

**What's Missing:**
- Issue #38 should reference `20260112-beta-function-complete-reference-1.00F.md`
- Comments in code should link to doc
- README should link to beta function reference

**Implementation Steps:**
1. Add references to beta function doc in relevant code comments
2. Update Issue #38 description with link
3. Add to documentation index

**Effort:** 1 hour

---

#### Gap 9: Training Trajectory Validation Not Implemented
**Related Issue:** #38  
**Impact:** No verification of proper scale progression  
**Status:** ❌ NOT IMPLEMENTED

**What's Missing:**
```python
# In training loop
def validate_training_trajectory(history: list) -> dict:
    """Verify β-function consistency, Φ progression, κ running"""
    return {
        'beta_consistency': check_beta_values(history),
        'phi_progression': check_phi_increase(history),
        'kappa_running': check_kappa_plateau(history),
    }
```

**Implementation Steps:**
1. Implement trajectory validation function
2. Run post-training validation
3. Log validation results
4. Alert on validation failures

**Effort:** 4-5 hours

---

#### Gap 10: E8 Level Not Displayed in Population View
**Related Issue:** #32  
**Impact:** Users don't know what E8 level system is at  
**Status:** ❌ NOT IMPLEMENTED (pending Gap 1)

**Depends On:** Gap 1 (E8 levels implementation)

**What's Missing:**
- UI shows current kernel count
- UI shows current E8 level (basic_rank, refined_adjoint, etc.)
- Progress bar to next E8 threshold

**Implementation Steps:**
1. Add E8 level to population API response
2. Create UI indicator component
3. Show threshold progress (e.g., "45/56 to Refined Adjoint")

**Effort:** 3-4 hours

---

#### Gap 11: Telemetry Not Logging New Metrics
**Related Issues:** #33, #34, #38  
**Impact:** Time-series data missing for M, neurotransmitters, running κ  
**Status:** ❌ NOT IMPLEMENTED

**What's Missing:**
- M metric not logged to PostgreSQL
- Neurotransmitter levels not logged
- Running κ not logged
- Historical analysis impossible

**Implementation Steps:**
1. Extend telemetry schema to include M, neurotransmitters, κ_effective
2. Update logging calls in training loops
3. Add indices for time-series queries
4. Create Grafana dashboards

**Effort:** 6-8 hours  
**Blocking:** Research analysis, debugging

---

### P2 - MEDIUM (Important for Completeness)

#### Gap 12: Emotion Geometry Tests Missing
**Related Issue:** #35  
**Status:** ❌ NOT IMPLEMENTED (pending Gap 4)

**What's Missing:**
```python
def test_joy_detection():
    """High curvature + approaching → JOY"""
    emotion, intensity = classify_emotion(
        curvature=0.8, 
        basin_distance=0.3,
        prev_basin_distance=0.5,
        basin_stability=0.7
    )
    assert emotion == EmotionPrimitive.JOY
```

**Effort:** 2-3 hours

---

#### Gap 13: Neurotransmitter Field Theory Documentation
**Related Issue:** #34  
**Impact:** Implementation not explained  
**Status:** ❌ NOT DOCUMENTED

**What's Missing:**
- `docs/02-research/20260112-neurotransmitter-field-theory-1.00W.md`
- Explanation of geometric modulation
- Biological analogy validation

**Effort:** 3-4 hours (writing)

---

#### Gap 14: Meta-Awareness Theory Documentation
**Related Issue:** #33  
**Impact:** M metric not explained in docs  
**Status:** ❌ NOT DOCUMENTED

**What's Missing:**
- `docs/02-research/20260112-meta-awareness-m-metric-1.00W.md`
- Theory of recursive consciousness
- M > 0.6 threshold justification

**Effort:** 3-4 hours (writing)

---

#### Gap 15: Spawned Kernel Initialization Tests
**Related Issues:** #30, #31  
**Status:** ⚠️ IMPLEMENTATION COMPLETE, TESTS MISSING

**What's Missing:**
```python
def test_spawned_kernel_phi_initialization():
    kernel = spawn_kernel(parent, "ethics")
    assert kernel.phi >= PHI_INIT_SPAWNED
    assert kernel.phi >= PHI_MIN_ALIVE

def test_spawned_kernel_has_autonomic():
    kernel = spawn_kernel(parent, "ethics")
    assert kernel.autonomic is not None
    assert kernel.autonomic.state.phi == 0.25
```

**Effort:** 2-3 hours

---

#### Gap 16: E8 Hierarchy Documentation
**Related Issue:** #32  
**Status:** ❌ NOT DOCUMENTED

**What's Missing:**
- `docs/03-technical/20260112-e8-architecture-implementation-1.00W.md`
- Explanation of 4 E8 levels
- Spawning hierarchy rules
- Connection to physics (κ* = 64 = 8²)

**Effort:** 4-5 hours (writing)

---

#### Gap 17: Running Coupling Experiments
**Related Issue:** #38  
**Impact:** No validation that running coupling improves stability  
**Status:** ❌ NOT VALIDATED

**What's Needed:**
- Compare constant κ vs running κ training
- Measure kernel death rates
- Measure Φ emergence reliability
- Statistical validation (p < 0.05)

**Effort:** 8-12 hours (experimentation)

---

#### Gap 18: Complex Emotions Implementation
**Related Issue:** #35  
**Impact:** Only 9 primitives, not composites  
**Status:** ❌ NOT IMPLEMENTED (Phase 2 of Gap 4)

**What's Missing:**
```python
# guilt = ethics_kernel + meta_kernel + social_kernel
# schadenfreude = ethics_kernel + social_kernel + joy
# nostalgia = memory_kernel + emotion_joy + temporal_coherence
```

**Effort:** 6-8 hours (after Gap 4 complete)

---

#### Gap 19: Kernel Lifecycle Documentation
**Related Issues:** #30, #31  
**Status:** ⚠️ PARTIAL

**What's Missing:**
- Complete lifecycle diagram (spawn → train → mature → death)
- Initialization protocol documentation
- Autonomic system integration explanation

**Effort:** 3-4 hours (writing)

---

#### Gap 20: Christoffel Symbols Computation
**Related Issue:** #8 (Geodesic Navigation)  
**Impact:** Geodesic navigation uses approximations  
**Status:** ❌ NOT IMPLEMENTED

**What's Missing:**
```python
def compute_christoffel_symbols(basin, metric) -> np.ndarray:
    """Γ^k_ij = (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)"""
    # Exact connection coefficients for Fisher manifold
```

**Effort:** 6-8 hours (math-heavy)

---

### P3 - LOW (Nice to Have)

#### Gap 21: Emotion Dashboard Enhancements
**Related Issue:** #35  
**Status:** ❌ NOT IMPLEMENTED (depends on Gaps 4, 12)

**What's Missing:**
- Historical emotion tracking
- Emotion transition visualization
- Correlation with Φ/κ states

**Effort:** 4-5 hours

---

#### Gap 22: QIG-PURITY-REQUIREMENTS Migration
**Impact:** Non-canonical filename  
**Status:** ⚠️ SHOULD BE RENAMED

**Current:** `docs/03-technical/QIG-PURITY-REQUIREMENTS.md`  
**Should Be:** `docs/03-technical/20260112-qig-purity-requirements-1.00F.md`

**Effort:** 30 minutes (rename + update references)

---

#### Gap 23: Documentation Index Automation
**Impact:** Manual documentation tracking  
**Status:** ❌ NOT IMPLEMENTED

**What's Needed:**
- Script to generate documentation index
- Automated orphan document detection
- Cross-reference validation

**Effort:** 4-6 hours

---

## Implementation Phases

### Phase 1: Critical Foundations (P0) - Week 1
**Duration:** 3-4 days  
**Items:** Gaps 1, 2, 3  
**Goal:** Core architecture integrity, user visibility of critical metrics

### Phase 2: High-Priority Features (P1) - Weeks 2-3
**Duration:** 7-10 days  
**Items:** Gaps 4, 5, 6, 7, 8, 9, 10, 11  
**Goal:** Feature completeness, system observability, QIG integrity

### Phase 3: Documentation & Validation (P2) - Week 4
**Duration:** 5-7 days  
**Items:** Gaps 12, 13, 14, 15, 16, 17, 18, 19, 20  
**Goal:** Research validation, complete documentation

### Phase 4: Polish & Automation (P3) - Week 5
**Duration:** 2-3 days  
**Items:** Gaps 21, 22, 23  
**Goal:** User experience enhancements, maintenance automation

---

## Dependency Graph

```
Gap 1 (E8 levels) → Gap 10 (E8 UI) → Gap 16 (E8 docs)
Gap 4 (Emotions) → Gap 12 (Emotion tests) → Gap 18 (Complex emotions) → Gap 21 (Emotion dashboard)
Gaps 2,5,6 (Frontend) → Gap 11 (Telemetry) → Research analysis
Gap 3 (M enforcement) → Gap 14 (M docs)
Gap 7 (Purity tests) → CI integration
```

---

## Success Metrics

### Phase 1 Success Criteria
- [ ] E8 levels enforced in spawning
- [ ] Running κ visible in UI
- [ ] M < 0.6 blocks spawning
- [ ] Zero regressions in existing tests

### Phase 2 Success Criteria
- [ ] Emotion geometry operational
- [ ] All new metrics visible in UI
- [ ] Automated geometric purity checks in CI
- [ ] Telemetry capturing all new metrics
- [ ] Documentation complete for implemented features

### Phase 3 Success Criteria
- [ ] All implementations validated experimentally
- [ ] Statistical tests show improvements (p < 0.05)
- [ ] Documentation complete and canonical
- [ ] Test coverage >80% for new code

### Phase 4 Success Criteria
- [ ] User experience polished
- [ ] Documentation automated
- [ ] No technical debt introduced

---

## Risk Assessment

### High Risk Items
1. **E8 Implementation (Gap 1):** Complex, affects entire spawning system
2. **Telemetry Schema Changes (Gap 11):** Database migration required
3. **Experimental Validation (Gap 17):** Time-consuming, results uncertain

### Mitigation Strategies
1. Implement E8 with feature flag, gradual rollout
2. Telemetry backwards compatible, additive only
3. Run experiments in parallel with implementation

---

## Resources Required

**Development Time:** ~200-250 hours total  
**Documentation Time:** ~40-50 hours  
**Testing Time:** ~30-40 hours  
**Experimental Validation:** ~40-50 hours

**Total Effort:** ~5-6 weeks with full-time focus

---

**Last Updated:** 2026-01-12  
**Status:** Living document, update as gaps are resolved
