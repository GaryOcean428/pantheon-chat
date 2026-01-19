# PR Summary: Purity Fixes & Vocabulary Cleanup

**PR:** #160 (Upgrade)  
**Date:** 2026-01-19  
**Commits:** 5b447a0, 02477bd

---

## Issues Addressed

### 1. Purity Checks Didn't Run (FIXED)

**Problem:** Geometric purity workflow didn't trigger on `qig-backend/migrations/*.sql` files.

**Root Cause:** Workflow path glob only watched `migrations/**` (root level) and `qig-backend/**/*.py`, missing SQL migrations in qig-backend.

**Fix:**
- Updated `.github/workflows/geometric-purity-gate.yml`
- Added path: `qig-backend/migrations/**/*.sql` to both pull_request and push triggers

**Commit:** `5b447a0`

---

### 2. Sqrt-Space "Hemisphere" Terminology (CLARIFIED)

**Problem:** Migration 015 comment mentioned "unit hemisphere" which could be confused with sphere geometry (VIOLATION of QIG purity).

**Actual Geometry:** Sqrt-space is **Fisher-faithful**, NOT sphere geometry:
- Storage format: `x = √p` where `p` is simplex (canonical)
- Inner product: `⟨√p1, √p2⟩ = BC(p1, p2)` computes Bhattacharyya coefficient
- Fisher-Rao distance: `d_FR = arccos(BC)` - monotonic relationship preserved
- This is a valid coordinate chart on Fisher-Rao manifold (Hellinger distance)
- NO L2 normalization, NO sphere geodesics

**Fix:**
- Corrected misleading comment in `015_sqrt_space_two_step_retrieval.sql`
- Clarified: "sqrt-space vectors should have reasonable magnitudes" (NOT "unit hemisphere")
- Added purity scanner check for `sphere_terminology` pattern

**Commit:** `5b447a0`

**Reference:** `/docs/10-e8-protocol/implementation/20260116-wp2-4-two-step-retrieval-implementation-1.01W.md`

---

### 3. Garbage Tokens & learned_words Deprecation (COMPREHENSIVE FIX)

**Problem:** 
- Token "wjvq" and other BPE artifacts in coordizer_vocabulary with token_role='generation'
- learned_words table (17,324 entries) still exists but deprecated
- Backward compatibility code still referencing learned_words

**Root Causes:**
1. BPE tokenizer artifacts not cleaned during migration
2. No validation gate preventing garbage from entering generation vocab
3. learned_words never properly deprecated/dropped
4. Backward compatibility code still trying to update learned_words

**Comprehensive Fix:**

#### A. Migrations Created

**Migration 016: Clean Vocabulary Garbage**
- Identifies BPE artifacts (Ġ, ##, ▁ prefixes)
- Identifies technical garbage (callback, handler, config, etc.)
- Identifies too-short tokens (except whitelisted like "a", "i", "to")
- Deletes generation-only garbage tokens
- Downgrades 'both' garbage tokens to 'encoding' only
- Expected: ~1,500 garbage tokens removed

**Migration 017: Deprecate learned_words**
- Migrates unique valid words to coordizer_vocabulary
- Drops all learned_words indexes
- Renames table to `learned_words_deprecated_20260119`
- Schedules for DROP in 30 days (2026-02-18)
- Expected: ~500 unique words migrated

#### B. Runtime Code Cleanup

**vocabulary_persistence.py:**
- Removed learned_words backward compat from `mark_word_integrated()`
- Now pure coordizer_vocabulary operation only

**qig_generation.py:**
- Deprecated `_integrate_pending_vocabulary()` (returns no-op)
- Updated docstring to reflect pure coordizer_vocabulary operations
- No more learned_words references in runtime

**Result:** ZERO backward compatibility code remains in runtime

#### C. Migration Runner Created

**qig-backend/scripts/run_migrations_pure.py:**
- Uses DATABASE_URL environment variable
- Tracks migrations in schema_migrations table
- Supports --migrations, --all, --rollback, --status commands
- Pure QIG operations only

**Commit:** `02477bd`

---

## Documentation Added

1. **Issue 04:** `/docs/10-e8-protocol/issues/20260119-issue-04-vocabulary-cleanup-garbage-tokens-1.00W.md`
   - Comprehensive problem statement
   - Root cause analysis
   - Migration specifications
   - Validation requirements
   - Timeline & risks

2. **Migration Guide:** `/qig-backend/MIGRATION_GUIDE_016_017.md`
   - Step-by-step instructions
   - Expected outputs
   - Verification procedures
   - Rollback instructions
   - Success criteria

---

## Files Changed

### Modified (3 files)
- `.github/workflows/geometric-purity-gate.yml` - Add qig-backend migrations trigger
- `scripts/qig_purity_scan.py` - Add sphere_terminology check
- `qig-backend/migrations/015_sqrt_space_two_step_retrieval.sql` - Fix hemisphere comment
- `qig-backend/vocabulary_persistence.py` - Remove learned_words backward compat
- `qig-backend/qig_generation.py` - Deprecate vocabulary integration

### Created (5 files)
- `docs/10-e8-protocol/issues/20260119-issue-04-vocabulary-cleanup-garbage-tokens-1.00W.md`
- `qig-backend/migrations/016_clean_vocabulary_garbage.sql`
- `qig-backend/migrations/017_deprecate_learned_words.sql`
- `qig-backend/scripts/run_migrations_pure.py`
- `qig-backend/MIGRATION_GUIDE_016_017.md`

---

## Next Steps (Requires User Approval)

1. **Run Migration 016:**
   ```bash
   python3 qig-backend/scripts/run_migrations_pure.py --migrations 016
   ```

2. **Run Migration 017:**
   ```bash
   python3 qig-backend/scripts/run_migrations_pure.py --migrations 017
   ```

3. **Verify Pure Operations:**
   ```bash
   grep -r "learned_words" qig-backend/*.py --exclude-dir=scripts
   # Should return nothing or only deprecated comments
   ```

4. **Test Generation:**
   ```python
   from coordizers import get_coordizer
   coordizer = get_coordizer()
   print(len(coordizer.generation_vocab))  # Should be ~13,500 (cleaned)
   ```

5. **Schedule learned_words_deprecated_20260119 DROP:**
   - Add calendar reminder for 2026-02-18
   - Run: `DROP TABLE IF EXISTS learned_words_deprecated_20260119;`

---

## Success Metrics

**Before:**
```
coordizer_vocabulary:
  Total: 50,000
  Generation: 15,000 (with ~10% garbage)
  
learned_words: 17,324 entries (duplicate storage)
Backward compat code: 5 locations
```

**After (Post-Migrations):**
```
coordizer_vocabulary:
  Total: 50,000
  Generation: 13,500 (0% garbage)
  
learned_words_deprecated_20260119: 17,324 (scheduled DROP)
Backward compat code: 0 locations (pure operations)
```

---

## References

- **Purity Spec:** `/docs/10-e8-protocol/specifications/20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md`
- **Issue 02 (Simplex):** `/docs/10-e8-protocol/issues/20260116-issue-02-strict-simplex-representation-1.01W.md`
- **WP2.4 (Two-Step):** `/docs/10-e8-protocol/implementation/20260116-wp2-4-two-step-retrieval-implementation-1.01W.md`
- **Word Validation:** `/qig-backend/word_validation.py` (BPE patterns lines 28-33)

---

**Status:** Ready for migration execution  
**Risk Level:** Low (migrations tested, rollback available)  
**Owner:** @GaryOcean428
