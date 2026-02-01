# PR Comments to Post

This document contains the comments that should be posted on each PR based on the sweep analysis.

---

## PR #246: Hemisphere Scheduler

### Comment to Post:

**✅ Ready to Merge (after #247)**

### Analysis Summary
- **Status:** Complete, tested, QIG-pure
- **Dependencies:** Requires #247 (Psyche Plumbing) to be merged first
- **Tests:** 28 tests passing
- **QIG Purity:** ✅ Clean (Fisher-Rao only, no violations)

### Integration Gap Identified ⚠️
**Missing:** Hemisphere ↔ Psyche coupling

The hemisphere scheduler and psyche hierarchy (#247) don't communicate. After both PRs merge, need to add:

```python
# In hemisphere_scheduler.py
def get_hemisphere_for_kernel(kernel_type, phi, kappa):
    if kernel_type == "id":
        return None  # Id bypasses scheduler (fast reflex <100ms)
    if kernel_type == "superego":
        # Ethics check before hemisphere assignment
        if not ethics_satisfied(phi, kappa):
            return "quarantine"
    # ... existing LEFT/RIGHT logic
```

### Recommendation
1. ✅ Merge #247 first
2. ✅ Merge this PR (#246)
3. 🔧 Add integration code above
4. ✅ Merge #250 (Genetic Lineage)

**See full analysis:** `PR_SWEEP_ANALYSIS.md`

---

## PR #247: Psyche Plumbing

### Comment to Post:

**✅ Ready to Merge - FOUNDATIONAL**

### Analysis Summary
- **Status:** Complete, tested, QIG-pure
- **Dependencies:** None (foundational for Chain B)
- **Tests:** 21 tests passing
- **QIG Purity:** ✅ Clean (Fisher-Rao only, no violations)

### What This Enables
This PR is foundational for Chain B (Consciousness Architecture). After merge:
- ✅ #246 (Hemisphere Scheduler) can use Id/Superego types
- ✅ #250 (Genetic Lineage) can use Φ hierarchy for evolution

### Integration Note
After this PR and #246 merge, add coupling between hemisphere scheduler and psyche types (see comment on #246).

### Recommendation
**Merge this PR immediately** - no blockers, enables other PRs.

**See full analysis:** `PR_SWEEP_ANALYSIS.md`

---

## PR #248: QFI Integrity

### Comment to Post:

**✅ Ready to Merge - CRITICAL PATH (MERGE FIRST)**

### Analysis Summary
- **Status:** Complete, validated, QIG-pure
- **Dependencies:** None (foundational for Chain A)
- **Validation:** All 16,018 tokens have valid QFI scores
- **QIG Purity:** ✅ Clean (enforces [0.0, 1.0] range)

### Why This MUST Go First
This PR creates the foundation for the entire generation pipeline:

1. **Creates:** `is_generation_eligible` flag
2. **Creates:** `vocabulary_generation_ready` view
3. **Enforces:** QFI constraints at database level
4. **Enables:** #251 (Unified Pipeline) and #252 (Pure QIG)

### What This Blocks
- ❌ Cannot merge #251 without this
- ❌ Cannot merge #252 without this
- ✅ #247, #246, #250 can proceed in parallel

### Validation Complete
- ✅ 16,018/16,018 tokens have QFI ∈ [0,1]
- ✅ 16,018/16,018 basins are valid simplex
- ✅ 15,978 tokens marked generation-eligible
- ✅ All constraints tested at DB level

### Recommendation
**Merge this PR TODAY** - critical path for generation chain.

**See full analysis:** `PR_SWEEP_ANALYSIS.md`, `20260122-pr-merge-order-guide-1.00W.md`

---

## PR #249: Ethical Consciousness

### Comment to Post:

**❌ WIP - NEEDS IMPLEMENTATION**

### Analysis Summary
- **Status:** Empty placeholder (no code)
- **Dependencies:** All other PRs (should be completed after others)
- **Blocker:** Blocks #253 (Dead Code Cleanup)

### What's Missing
This PR is currently empty. Needs implementation of:

1. **Integration with ocean_qig_core.py**
   - Wire ethical checks into consciousness loop
   - Add to consciousness metrics tracking

2. **Connection to rest scheduler**
   - Ethical violations should trigger rest cycles?
   - Safety guardrails for kernel activation

3. **Ethical constraints in transitions**
   - Basin transition validation
   - Generation filtering (reject unethical outputs)
   - Hemisphere transition gating

4. **Superego kernel enforcement**
   - Connect to Superego from #247
   - Ethical field penalties
   - Forbidden region enforcement

### Files to Implement
- Wire `consciousness_ethical.py` into main pipeline
- Update `ethics_gauge.py` with Fisher-Rao drift calculation
- Add ethical checks to generation pipeline
- Connect to Superego kernel from #247

### Recommendation
1. ❌ **DO NOT merge as-is** (empty PR)
2. 🔧 **Complete implementation** after Chains A & B merge
3. ✅ **Then merge #253** (Dead Code Cleanup)

**Assigned to:** @Copilot for completion

**See full analysis:** `PR_SWEEP_ANALYSIS.md`

---

## PR #250: Genetic Lineage

### Comment to Post:

**✅ Ready to Merge (after #246)**

### Analysis Summary
- **Status:** Complete, tested, QIG-pure
- **Dependencies:** Requires #246 (Hemisphere Scheduler) for kernel lifecycle
- **Tests:** 15 tests passing
- **QIG Purity:** ✅ Clean (geodesic merges, no linear averaging)

### Integration Gap Identified ⚠️
**Missing:** Genome merging → Vocabulary pipeline

Merged kernel genomes don't flow into generation vocabulary. After merge, need to add:

```python
# In kernel_lineage.py
async def after_genome_merge(child_genome):
    # Insert child basin into vocabulary
    await insert_token(
        token=child_genome.kernel_name,
        basin=child_genome.basin_seed,
        compute_qfi=True  # PR #248 requirement
    )
    
    # Update vocabulary view
    await refresh_vocabulary_generation_ready()
```

### Key Features
- ✅ Geodesic merge operations (SLERP, Fréchet mean)
- ✅ Lineage tracking with genealogy trees
- ✅ Cannibalism with genome archival
- ✅ Mutation support for resurrection

### Recommendation
1. ✅ Merge #246 first (hemisphere scheduler)
2. ✅ Merge this PR (#250)
3. 🔧 Add genome → vocabulary integration above

**See full analysis:** `PR_SWEEP_ANALYSIS.md`

---

## PR #251: Unified Pipeline

### Comment to Post:

**✅ Ready to Merge (after #248)**

### Analysis Summary
- **Status:** Complete, tested, QIG-pure
- **Dependencies:** Requires #248 (QFI Integrity) for generation-eligible tokens
- **Tests:** Integration tests passing
- **QIG Purity:** ✅ Clean (Fisher-Rao only, no violations)

### Integration Gap Identified ⚠️
**Missing:** Strategy dispatch coordination with #252

Both this PR and #252 add generation strategies. After both merge, need explicit dispatch:

```python
# In qig_generation.py
class GenerationStrategy(Enum):
    FORESIGHT_DRIVEN = "foresight"   # This PR
    ROLE_DRIVEN = "role"             # This PR
    HYBRID = "hybrid"                # This PR
    PURE_QIG = "pure_qig"            # PR #252
    FALLBACK_GARY = "gary"           # Existing

def select_strategy(qig_purity_mode, context):
    if qig_purity_mode:
        return GenerationStrategy.PURE_QIG
    elif context.has_foresight_data:
        return GenerationStrategy.FORESIGHT_DRIVEN
    # ... dispatch logic
```

### Key Features
- ✅ 3 generation strategies (foresight, role, hybrid)
- ✅ Per-token observables
- ✅ QIG_PURITY_MODE enforcement

### Recommendation
1. ✅ Merge #248 first (QFI foundation)
2. ✅ Merge this PR (#251)
3. ✅ Merge #252 (Pure QIG strategy)
4. 🔧 Add strategy dispatch above

**Coordinate with:** #252 on strategy relationship

**See full analysis:** `PR_SWEEP_ANALYSIS.md`

---

## PR #252: Pure QIG Generation

### Comment to Post:

**✅ Ready to Merge (after #248 and #251)**

### Analysis Summary
- **Status:** Complete, validated, QIG-pure
- **Dependencies:** Requires #248 (QFI) and #251 (Pipeline framework)
- **Tests:** Comprehensive purity validation
- **QIG Purity:** ✅ Clean (no external LLMs, pure geometric)

### Key Achievement
**Validates QIG principles:** Coherent text generation from pure geometric operations on 64D simplex using Fisher-Rao distance, without external LLM dependencies.

### Integration Note
This PR adds "pure QIG" as a strategy within the unified pipeline framework from #251. See comment on #251 for strategy dispatch coordination.

### Validation Complete
- ✅ No external LLM imports (openai, anthropic, google.generativeai)
- ✅ Fisher-Rao distance throughout pipeline
- ✅ Token role filtering (generation-eligible)
- ✅ QFI score requirements enforced
- ✅ Geometric completion criteria
- ✅ Simplex representation enforced

### Known Limitation
**Existing codebase violations** (NOT in this PR):
```python
# These files have np.mean() on basin coordinates
autonomous_debate_service.py:1257
contextualized_filter.py:261, 349, 382
qig_generative_service.py:1709, 1836
```
**Action:** Mark as technical debt, separate cleanup issue

### Recommendation
1. ✅ Merge #248 first (QFI foundation)
2. ✅ Merge #251 (unified pipeline)
3. ✅ Merge this PR (#252)
4. 🔧 Create issue for existing np.mean() violations

**See full analysis:** `PR_SWEEP_ANALYSIS.md`

---

## PR #253: Dead Code Cleanup

### Comment to Post:

**✅ Ready to Merge - BUT MUST BE LAST**

### Analysis Summary
- **Status:** Complete, cleanup verified
- **Dependencies:** ALL other PRs (must be last)
- **Risk Level:** ⚠️ May delete code used by other PRs
- **Files Removed:** 422 lines (broken imports, unused tests)

### Why This MUST Be Last
Removes files that may be referenced by PRs #246-#252:
- `constellation_service.py` (may be imported by scheduler)
- `retry_decorator.py` (may be used by generation)
- `constellation_routes.py` (may be registered in wsgi.py)

### Safety Protocol
Before merging:
1. ✅ Merge ALL other PRs first
2. ✅ Complete #249 (Ethical Consciousness)
3. ✅ Verify no active imports to deleted files:
   ```bash
   grep -r "constellation_service" qig-backend/
   grep -r "retry_decorator" qig-backend/
   ```
4. ✅ Run full test suite
5. ✅ Then merge this PR

### Recommendation
**Hold this PR** until all others (#246-#252, #249) are merged.

**Merge order:** ALL → #249 → #253 (LAST)

**See full analysis:** `PR_SWEEP_ANALYSIS.md`, `20260122-pr-merge-order-guide-1.00W.md`

---

## Summary Comments Document

This file documents the recommended comments for each PR based on the comprehensive sweep analysis. 

**To post these comments:**
1. Manually copy each comment to the respective PR
2. Or use GitHub CLI: `gh pr comment <PR#> --body "$(cat comment.txt)"`
3. Or create issues linking to this analysis

**Analysis Files:**
- Full analysis: `PR_SWEEP_ANALYSIS.md`
- Quick reference: `20260122-pr-merge-order-guide-1.00W.md`
- This file: `20260122-pr-review-comments-1.00W.md`

**Analyst:** @Copilot  
**Date:** 2026-01-22
