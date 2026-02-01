# PR Merge Order - Quick Reference

## Visual Dependency Map

```
┌─────────────────────────────────────────────────────────────┐
│                    CHAIN A: GENERATION                      │
│  (Database Foundation → Generation Pipeline)                │
└─────────────────────────────────────────────────────────────┘

    #248 QFI Integrity
    ├── Creates: is_generation_eligible flag
    ├── Creates: vocabulary_generation_ready view  
    ├── Creates: QFI constraints [0.0, 1.0]
    └── Status: ✅ READY TO MERGE
         │
         ▼
    #251 Unified Pipeline
    ├── Depends: QFI-validated tokens from #248
    ├── Adds: 3 generation strategies
    └── Status: ✅ READY (after #248)
         │
         ▼
    #252 Pure QIG Generation
    ├── Depends: Pipeline framework from #251
    ├── Adds: Pure geometric strategy (no LLMs)
    └── Status: ✅ READY (after #251)


┌─────────────────────────────────────────────────────────────┐
│              CHAIN B: CONSCIOUSNESS                         │
│  (Psyche Hierarchy → Kernel Management)                     │
└─────────────────────────────────────────────────────────────┘

    #247 Psyche Plumbing
    ├── Creates: Id/Superego/Φ hierarchy
    ├── Creates: 3-level consciousness model
    ├── Creates: Reflex system (<100ms)
    └── Status: ✅ READY TO MERGE
         │
         ▼
    #246 Hemisphere Scheduler
    ├── Depends: Kernel types from #247
    ├── Adds: LEFT/RIGHT hemispheres
    ├── Adds: κ-gated coupling
    └── Status: ✅ READY (after #247)
         │
         ▼
    #250 Genetic Lineage
    ├── Depends: Kernel lifecycle from #246
    ├── Adds: Genome schema
    ├── Adds: Geodesic merge operations
    └── Status: ✅ READY (after #246)


┌─────────────────────────────────────────────────────────────┐
│              CHAIN C: MAINTENANCE                           │
│  (Complete WIP → Cleanup Dead Code)                        │
└─────────────────────────────────────────────────────────────┘

    ALL CHAINS COMPLETE
         │
         ▼
    #249 Ethical Consciousness  ⚠️ MUST COMPLETE FIRST
    ├── Status: ❌ EMPTY WIP
    ├── Needs: Implementation
    ├── Needs: Wiring to ocean_qig_core
    └── Blocks: #253
         │
         ▼
    #253 Dead Code Cleanup
    ├── Removes: Broken imports
    ├── Removes: Unused test files
    └── Status: ✅ READY (MUST BE LAST)


┌─────────────────────────────────────────────────────────────┐
│              META PR                                        │
└─────────────────────────────────────────────────────────────┘

    #254 This PR Sweep
    └── Delivers: Analysis, merge strategy, integration plan
```

---

## Critical Rules

### 🔴 NEVER VIOLATE THESE

1. **#248 MUST GO FIRST** (all generation PRs depend on it)
2. **#247 MUST GO BEFORE #246** (hemisphere needs psyche types)
3. **#246 MUST GO BEFORE #250** (lineage needs lifecycle)
4. **#253 MUST BE LAST** (may delete code used by others)
5. **#249 MUST BE COMPLETED** (currently empty placeholder)

### 🟡 SAFE TO MERGE IN PARALLEL

- Chain A and Chain B are independent
- Can merge #248 + #247 simultaneously
- Can merge #251 + #246 simultaneously (after their dependencies)

### 🟢 SUGGESTED ORDER (Optimal)

```
Day 1:  #248 + #247  (both foundational)
Day 2:  #251 + #246  (both ready after day 1)
Day 3:  #252 + #250  (capstone features)
Day 4:  Complete #249 (WIP)
Day 5:  #253 (cleanup)
```

---

## Integration Work Needed

### After #246 + #247 Merge
**Hemisphere ↔ Psyche Coupling**

Add to `hemisphere_scheduler.py`:
```python
def get_hemisphere_for_kernel(kernel_type, phi, kappa):
    if kernel_type == "id":
        return None  # Id bypasses scheduler (fast reflex)
    if kernel_type == "superego":
        # Ethics check before assignment
        if not ethics_satisfied(phi, kappa):
            return "quarantine"
    # ... rest of logic
```

### After #251 + #252 Merge
**Generation Strategy Dispatch**

Add to `qig_generation.py`:
```python
class GenerationStrategy(Enum):
    FORESIGHT_DRIVEN = "foresight"   # #251
    ROLE_DRIVEN = "role"             # #251
    HYBRID = "hybrid"                # #251
    PURE_QIG = "pure_qig"            # #252
    FALLBACK_GARY = "gary"           # Existing

def select_strategy(qig_purity_mode, context):
    if qig_purity_mode:
        return GenerationStrategy.PURE_QIG
    # ... dispatch logic
```

### After #248 + #250 Merge
**Genome → Vocabulary Pipeline**

Add to `kernel_lineage.py`:
```python
async def after_genome_merge(child_genome):
    await insert_token(
        token=child_genome.kernel_name,
        basin=child_genome.basin_seed,
        compute_qfi=True  # #248 requirement
    )
```

### After ALL Merges
**Ethical Consciousness Wiring (#249)**

Implement:
1. Integration with ocean_qig_core.py
2. Connection to rest scheduler
3. Ethical constraints in transitions
4. Superego kernel ethical enforcement

---

## Validation Checklist

### Before Merging ANY PR
- [ ] QIG purity check passes (`validate_geometry_purity.py`)
- [ ] All tests pass
- [ ] No merge conflicts
- [ ] Dependencies merged first

### After Merging #248
- [ ] Run `npm run validate:qfi`
- [ ] Verify all tokens have QFI scores
- [ ] Test `vocabulary_generation_ready` view

### After Merging #247
- [ ] Verify Id/Superego kernels instantiate
- [ ] Test <100ms reflex latency
- [ ] Test Φ hierarchy (reported/internal/autonomic)

### After Merging #251
- [ ] Test all 3 generation strategies
- [ ] Verify QFI-gated vocabulary works
- [ ] Test strategy dispatch logic

### After Merging #246
- [ ] Verify LEFT/RIGHT god assignments
- [ ] Test κ-gated coupling (κ<40, κ>70)
- [ ] Test hemisphere tacking

### After Merging #252
- [ ] Test pure QIG generation end-to-end
- [ ] Confirm NO external LLM calls
- [ ] Validate geometric completion criteria

### After Merging #250
- [ ] Test binary genome merge (SLERP)
- [ ] Test multi-parent merge (Fréchet mean)
- [ ] Verify lineage tracking

### After Merging #253
- [ ] Run full test suite
- [ ] Verify no broken imports
- [ ] Check that deleted files weren't referenced

---

## Contact

**PR Analysis by:** @Copilot  
**Full Report:** `20260122-pr-sweep-analysis-1.00W.md`  
**Date:** 2026-01-22
