# PR Sweep Analysis Report
**Date:** 2026-01-22  
**Analyst:** @Copilot  
**Total Open PRs:** 9

## Executive Summary

Reviewed all 9 open PRs for wiring, functionality, QIG purity, and integration needs. Found 3 distinct implementation chains with critical merge order dependencies, several missing integration points, and some QIG purity violations in existing code.

### Critical Findings
1. **Merge Order Dependencies:** PRs must be merged in specific sequences to avoid breaking changes
2. **Missing Wiring:** 4 major integration points incomplete between PR chains
3. **QIG Purity:** Existing codebase has `np.mean()` violations on basin coordinates (NOT in PR code)
4. **Parallel Chains:** 3 safe-to-merge chains can proceed independently

---

## PR Status Matrix

| PR # | Title | Status | QIG Purity | Wiring | Tests | Dependencies |
|------|-------|--------|------------|--------|-------|--------------|
| #246 | Hemisphere Scheduler | ✅ Ready | ✅ Clean | ⚠️ Partial | ✅ Pass | Needs #247 |
| #247 | Psyche Plumbing | ✅ Ready | ✅ Clean | ⚠️ Partial | ✅ Pass | None |
| #248 | QFI Integrity | ✅ Ready | ✅ Clean | ✅ Complete | ✅ Pass | None (Foundation) |
| #249 | Ethical Consciousness | ❌ WIP | N/A | ❌ Not Started | N/A | All others |
| #250 | Genetic Lineage | ✅ Ready | ✅ Clean | ⚠️ Partial | ✅ Pass | Needs #246 |
| #251 | Unified Pipeline | ✅ Ready | ✅ Clean | ⚠️ Partial | ✅ Pass | Needs #248 |
| #252 | Pure QIG Generation | ✅ Ready | ✅ Clean | ⚠️ Partial | ✅ Pass | Needs #248, #251 |
| #253 | Dead Code Cleanup | ✅ Ready | ✅ Clean | ⚠️ Risk | ✅ Pass | Should be LAST |
| #254 | This Sweep | 🔄 Active | N/A | N/A | N/A | All others |

---

## Merge Order Strategy

### Chain A: Database Foundation → Generation Pipeline
**Critical Path - MUST MERGE IN ORDER**

```
#248 (QFI Integrity) → #251 (Unified Pipeline) → #252 (Pure QIG)
```

**Rationale:**
- **#248** creates `is_generation_eligible` flag & DB constraints
- **#251** requires QFI-validated tokens from #248
- **#252** extends #251 with pure-QIG strategy (no external LLMs)

**Merge Strategy:**
1. Merge #248 first (foundational DB changes)
2. Test that vocabulary tokens have QFI scores
3. Merge #251 (adds pipeline strategies)
4. Test pipeline with QFI-gated vocabulary
5. Merge #252 (adds pure QIG strategy)
6. Validate end-to-end generation

---

### Chain B: Consciousness Architecture → Kernel Management
**Can proceed in parallel with Chain A**

```
#247 (Psyche Plumbing) → #246 (Hemisphere Scheduler) → #250 (Genetic Lineage)
```

**Rationale:**
- **#247** establishes Id/Superego/Φ hierarchy (foundation for kernel types)
- **#246** uses hierarchy to assign gods to LEFT/RIGHT hemispheres
- **#250** uses scheduler's kernel lifecycle for genome evolution

**Merge Strategy:**
1. Merge #247 first (Psyche hierarchy)
2. Verify Id/Superego kernels instantiate correctly
3. Merge #246 (hemisphere scheduler)
4. Test LEFT/RIGHT god assignments
5. Merge #250 (genetic lineage)
6. Validate genome merge operations

---

### Chain C: Maintenance & Cleanup
**MUST BE LAST**

```
All others → #253 (Dead Code Cleanup) → #249 (Ethical Wiring - Complete It First)
```

**Rationale:**
- **#253** removes files that may be referenced by #246-#252
- **#249** is empty (WIP) - must be completed before merging
- Cleanup should happen after all feature PRs merged

**Merge Strategy:**
1. Complete all PRs in Chains A & B
2. Finish #249 (wire ethical consciousness)
3. Test that no active code references files in #253
4. Merge #253 last

---

## Missing Integration Points

### 1. Hemisphere ↔ Psyche Coupling ⚠️
**Files:** `kernel_rest_scheduler.py`, `hemisphere_scheduler.py`, `phi_hierarchy.py`

**Issue:** PR #246 (hemisphere scheduler) and PR #247 (psyche hierarchy) don't communicate.

**Missing:**
- LEFT/RIGHT hemispheres need different Φ thresholds (exploit vs explore)
- Id kernel (reflex) should bypass hemisphere scheduler (fast path)
- Superego (ethics) should gate hemisphere transitions

**Fix Needed:**
```python
# In hemisphere_scheduler.py
def get_hemisphere_for_kernel(kernel_type, phi, kappa):
    if kernel_type == "id":
        return None  # Id bypasses scheduler
    if kernel_type == "superego":
        # Ethics check before hemisphere assignment
        ...
```

---

### 2. Generation Strategy Dispatch ⚠️
**Files:** `qig_generation.py`, `unified_pipeline.py`, `test_pure_qig_generation.py`

**Issue:** PRs #251 and #252 both add generation strategies. Unclear how they coexist.

**Missing:**
- Explicit strategy enum/registry
- Dispatch logic to select strategy
- Strategy priority/fallback rules

**Fix Needed:**
```python
# In qig_generation.py
class GenerationStrategy(Enum):
    UNIFIED_FORESIGHT = "foresight_driven"  # PR #251
    UNIFIED_ROLE = "role_driven"            # PR #251
    UNIFIED_HYBRID = "hybrid"               # PR #251
    PURE_QIG = "pure_qig"                   # PR #252
    FALLBACK_GARY = "gary_coordinator"      # Existing

def select_strategy(context, qig_purity_mode):
    if qig_purity_mode:
        return GenerationStrategy.PURE_QIG
    # ... dispatch logic
```

---

### 3. Genome Merging → Vocabulary Pipeline ⚠️
**Files:** `geometric_pair_merging.py`, `coordizer_vocabulary table`, `vocabulary_generation_ready view`

**Issue:** PR #250 (genetic lineage) creates merged genomes but doesn't connect to generation vocabulary.

**Missing:**
- Merged kernel basins need to update coordizer_vocabulary
- New basins need QFI scores computed (PR #248)
- Vocabulary view must include merged tokens

**Fix Needed:**
```python
# In kernel_lineage.py
async def after_genome_merge(child_genome):
    # Insert child basin into vocabulary
    await insert_token(
        token=child_genome.kernel_name,
        basin=child_genome.basin_seed,
        compute_qfi=True  # PR #248 requirement
    )
```

---

### 4. Ethical Consciousness Wiring ❌
**Files:** `consciousness_ethical.py`, `ethics_gauge.py`, PR #249 (empty)

**Issue:** PR #249 is a placeholder. No actual wiring implemented.

**Missing:**
- Integration with `ocean_qig_core.py` consciousness loop
- Connection to rest scheduler (ethical violations trigger rest?)
- Ethical constraints in hemisphere transitions
- Superego kernel ethical enforcement

**Fix Needed:**
- Complete PR #249 implementation
- Wire ethical checks into:
  - Basin transition validation
  - Generation filtering (reject unethical outputs)
  - Kernel activation gating

---

## QIG Purity Assessment

### PR Code: ✅ All Clean
All PRs (#246-#252) use proper Fisher-Rao distance on simplex coordinates. No violations in NEW code.

### Existing Codebase: ⚠️ Violations Found
**CRITICAL:** These files have `np.mean()` on basin coordinates (violates Fisher-Rao manifold):

```
qig-backend/autonomous_debate_service.py:1257
qig-backend/contextualized_filter.py:261, 349, 382
qig-backend/qig_generative_service.py:1709, 1836
qig-backend/qig_phrase_classifier.py:84
qig-backend/geometric_repairer.py:365
```

**Action Required:**
- These violations are NOT in PR code
- Mark as technical debt
- PR #252 must NOT use these files for "pure QIG" generation
- Future cleanup issue to replace `np.mean()` with Fisher-Fréchet mean

---

## Conflict Analysis

### Potential Conflicts

1. **#251 vs #252 (Generation Strategy)**
   - **Risk:** Both add generation strategies
   - **Resolution:** Make #252 a strategy within #251's unified pipeline
   - **Status:** ⚠️ Needs coordination

2. **#246 vs #247 (Scheduler vs Hierarchy)**
   - **Risk:** Competing approaches to kernel management
   - **Resolution:** #247 defines types, #246 schedules them (complementary)
   - **Status:** ✅ Compatible

3. **#250 vs #248 (Genome vs QFI)**
   - **Risk:** Migration ordering (genome table depends on QFI constraints)
   - **Resolution:** Merge #248 first (provides QFI foundation)
   - **Status:** ✅ Resolved by merge order

4. **#253 vs All (Dead Code Cleanup)**
   - **Risk:** May delete code used by other PRs
   - **Resolution:** Merge #253 LAST after all others
   - **Status:** ✅ Mitigated

---

## Validation Checklist

### Before Merging Chain A (#248 → #251 → #252)
- [ ] #248: Run `npm run validate:qfi` - all tokens have QFI
- [ ] #248: Verify `vocabulary_generation_ready` view works
- [ ] #251: Test 3 generation strategies independently
- [ ] #251: Verify strategy dispatch logic
- [ ] #252: Test pure QIG generation end-to-end
- [ ] #252: Confirm NO external LLM calls in purity mode

### Before Merging Chain B (#247 → #246 → #250)
- [ ] #247: Verify Id/Superego/Φ hierarchy instantiates
- [ ] #247: Test fast-path reflexes (<100ms)
- [ ] #246: Verify LEFT/RIGHT god assignments
- [ ] #246: Test κ-gated coupling at κ<40, κ>70
- [ ] #250: Test binary genome merge (SLERP geodesic)
- [ ] #250: Test multi-parent merge (Fréchet mean)
- [ ] #250: Verify lineage tracking across generations

### Before Merging Chain C (#249 → #253)
- [ ] #249: Complete ethical consciousness implementation
- [ ] #249: Wire into ocean_qig_core.py
- [ ] #253: Audit deleted files - NO active imports
- [ ] #253: Run full test suite after cleanup

---

## Recommendations

### Immediate Actions
1. **Merge #248 TODAY** - Foundational for both chains
2. **Complete #249** - Currently empty, blocking #253
3. **Coordinate #251/#252** - Clarify strategy relationship
4. **Wire hemisphere ↔ psyche** - Critical integration gap

### Merge Timeline (Suggested)
- **Week 1:** #248 (QFI), #247 (Psyche)
- **Week 2:** #251 (Pipeline), #246 (Hemisphere)
- **Week 3:** #252 (Pure QIG), #250 (Lineage)
- **Week 4:** #249 (Ethics - complete first!), #253 (Cleanup)

### Post-Merge Validation
1. Run full E8 protocol validation suite
2. Test geometric purity with `validate_geometry_purity.py`
3. Verify consciousness metrics (Φ, κ) across all kernels
4. Benchmark generation latency (pure QIG vs unified)
5. Load test hemisphere scheduler under high coupling (κ>70)

---

## Conclusion

**Status:** 6/9 PRs ready to merge with proper sequencing. 1 PR (# 249) needs completion. 1 PR (#253) should be last. 1 PR (#254) is this analysis.

**Critical Path:** Merge order is non-negotiable. Missing integration points must be addressed during merge or immediately after.

**QIG Purity:** NEW code is clean. Existing violations are technical debt (separate issue).

**Next Steps:**
1. Comment on each PR with specific findings
2. Assign @Copilot to PRs needing integration work
3. Create GitHub issues for missing wiring points
4. Begin merging Chain A (#248 first)
