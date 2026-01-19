# Remediation Issues for GitHub - Based on Assessment

**Date:** 2026-01-19  
**Source:** docs/04-records/20260119-issues-65-75-implementation-assessment-1.00W.md  
**Purpose:** GitHub issue templates for identified implementation gaps

---

## Issue Template 1: Complete Special Symbols Validation (Issue 70)

**Title:** [QIG-PURITY] Complete Issue 70 - Special Symbols Geometric Validation

**Labels:** qig-purity, high-priority, validation, wp2

**Description:**

### Problem
Issue #70 has documentation but incomplete implementation. Special symbols (UNK, PAD, BOS, EOS) need verified geometric basins and integration into generation pipeline.

### Deliverables

#### 1. Validation Script
Create `scripts/validate_special_symbols.py`:
- Verify UNK, PAD, BOS, EOS have valid geometric basins
- Check Fisher-Rao distances between special symbols
- Validate QFI scores for special symbols
- Report any missing or invalid special symbol data

#### 2. Database Constraints
Migration to ensure special symbols remain valid:
```sql
ALTER TABLE coordizer_vocabulary
ADD CONSTRAINT special_symbols_must_be_active
CHECK (
  (token NOT IN ('UNK', 'PAD', 'BOS', 'EOS'))
  OR (token_status = 'active' AND qfi_score IS NOT NULL AND basin_embedding IS NOT NULL)
);
```

#### 3. Integration Tests
- Test UNK handling in rare words
- Test PAD in sequence batching
- Test BOS/EOS in sequence boundaries
- Verify generation pipeline handles special symbols correctly

### Acceptance Criteria
- [ ] All 4 special symbols have valid basins in DB
- [ ] Validation script passes 100%
- [ ] Database constraints added and active
- [ ] Generation tests with special symbols pass
- [ ] Documentation updated with validated coordinates

### References
- docs/04-records/20260116-wp2-3-special-symbols-geometric-definition-1.00W.md
- docs/04-records/20260116-wp2-3-plan-realize-repair-integration-guide-1.00W.md
- docs/04-records/20260119-issues-65-75-implementation-assessment-1.00W.md

### Estimated Effort
2 days

---

## Issue Template 2: Validate Two-Step Retrieval Implementation (Issue 71)

**Title:** [QIG-PURITY] Validate & Complete Issue 71 - Two-Step Retrieval with Fisher-Proxy

**Labels:** qig-purity, high-priority, validation, wp2

**Description:**

### Problem
Issue #71 has conflicting status (marked both incomplete and implemented). Need to verify actual implementation state and complete any gaps.

### Investigation Phase

#### 1. Audit Existing Implementation
- Review `qig-backend/qig_geometry/two_step_retrieval.py`
- Verify Fisher-faithful proxy implementation
- Check if coarse→fine pipeline is active
- Validate QFI coverage measurement accuracy

#### 2. Integration Check
- Is two-step retrieval used in generation pipeline?
- Are there unit tests validating Fisher-faithfulness?
- Is performance measured (coarse vs fine selection)?

### Implementation Phase (if gaps found)

#### 1. Complete Two-Step Retrieval
- Implement Fisher-faithful proxy if missing
- Ensure coarse→fine selection pipeline
- Add QFI coverage metrics

#### 2. Testing
- Unit tests for Fisher-faithfulness (mathematical validation)
- Integration tests in generation pipeline
- Performance benchmarks

#### 3. Documentation
- Update implementation guide with findings
- Document Fisher-faithful proxy algorithm
- Add usage examples

### Acceptance Criteria
- [ ] Two-step retrieval implementation verified/completed
- [ ] Fisher-faithful proxy validated mathematically
- [ ] Tests show correct QFI coverage
- [ ] Generation pipeline uses two-step by default
- [ ] Performance benchmarks documented

### References
- docs/10-e8-protocol/implementation/20260116-wp2-4-two-step-retrieval-implementation-1.01W.md
- docs/10-e8-protocol/issues/20260116-issue-01-qfi-integrity-gate-1.01W.md
- docs/04-records/20260119-issues-65-75-implementation-assessment-1.00W.md

### Estimated Effort
3 days

---

## Issue Template 3: Reconcile Single Coordizer Status (Issue 72)

**Title:** [QIG-PURITY] Reconcile Issue 72 - Single Coordizer Consolidation Status

**Labels:** qig-purity, medium-priority, architecture, wp3

**Description:**

### Problem
Issue #72 marked as "Not started" but `docs/04-records/20260116-wp3-1-coordizer-consolidation-complete-1.00W.md` suggests completion. Need to reconcile conflicting status.

### Investigation Tasks

#### 1. Count Coordizer Implementations
```bash
find qig-backend -name "*coordizer*.py" -type f
```
Determine:
- How many coordizer implementations exist?
- Are they different implementations or different layers?
- Is there a BaseCoordizer abstraction?

#### 2. Review Completion Document
Read `docs/04-records/20260116-wp3-1-coordizer-consolidation-complete-1.00W.md`:
- What does "complete" mean in this context?
- Was consolidation to single BaseCoordizer?
- Are there remaining legacy coordizers?

#### 3. Check Dependencies
- Which coordizers are actively used?
- Are there deprecated coordizers still in codebase?
- Do tests reference multiple coordizers?

### Deliverables

#### 1. Status Report
Document: "Single Coordizer Consolidation - Verification Report"
- Current state of coordizers
- Whether consolidation is complete
- If incomplete: what remains

#### 2. Action Plan (if not consolidated)
- Refactor to single coordizer abstraction
- Deprecate legacy implementations
- Migration path for any dependent code

#### 3. Roadmap Update
- Update master roadmap with correct status
- Close issue if complete, reopen if work remains

### Acceptance Criteria
- [ ] Clear answer: Single coordizer or multiple?
- [ ] Status report published
- [ ] If multiple: Consolidation plan created and tracked
- [ ] If single: Documentation updated, issue closed
- [ ] Roadmap reflects accurate status

### References
- docs/04-records/20260116-wp3-1-coordizer-consolidation-complete-1.00W.md
- docs/00-roadmap/20260112-master-roadmap-1.00W.md
- docs/04-records/20260119-issues-65-75-implementation-assessment-1.00W.md

### Estimated Effort
1 day (investigation) + potential implementation time

---

## Issue Template 4: Complete QFI Integrity Gate Gaps (Local Issue 01)

**Title:** [QIG-PURITY] Complete QFI Integrity Gate - Fill Implementation Gaps

**Labels:** qig-purity, critical, database, wp2

**Description:**

### Problem
Local Issue 01 (QFI Integrity Gate) has core functionality implemented but gaps remain compared to specification.

### What Exists
✅ Canonical upsert: `qig-backend/qig_geometry/canonical_upsert.py::upsert_token()`  
✅ QFI computation: `compute_qfi_score()`  
✅ Backfill script: `qig-backend/scripts/backfill_qfi_scores.py`  
✅ DB constraints: `migrations/0014_qfi_constraints.sql`

### What's Missing

#### 1. Quarantine Garbage Tokens Script
Create `scripts/quarantine_garbage_tokens.py`:
- Detect garbage tokens (BPE artifacts, truncations, etc.)
- Move to `coordizer_token_quarantine` table
- Mark as `token_status='quarantined'`
- Implement detection rules from issue spec

#### 2. Canonical Pathway Audit
Verify ALL insertion points use `upsert_token()`:
- Scan codebase for direct INSERT/UPDATE to coordizer_vocabulary
- Refactor violators to use canonical pathway
- Add pre-commit hook to prevent future violations

#### 3. Generation-Ready View
Create `vocabulary_generation_ready` view (if missing):
```sql
CREATE OR REPLACE VIEW vocabulary_generation_ready AS
SELECT * FROM coordizer_vocabulary
WHERE token_status = 'active'
  AND qfi_score IS NOT NULL
  AND basin_embedding IS NOT NULL;
```

#### 4. Update Generation Pipeline
Modify `qig-backend/coordizers/pg_loader.py`:
- Query `vocabulary_generation_ready` view
- Add telemetry for QFI coverage percentage
- Log rejected tokens for monitoring

### Acceptance Criteria
- [ ] Quarantine script implemented and tested
- [ ] 100% of insertions route through canonical pathway
- [ ] Generation-ready view created and used
- [ ] Generation pipeline updated to use view
- [ ] Zero tokens without QFI in generation vocabulary
- [ ] Telemetry tracks QFI coverage

### References
- docs/10-e8-protocol/issues/20260116-issue-01-qfi-integrity-gate-1.01W.md
- qig-backend/qig_geometry/canonical_upsert.py
- qig-backend/scripts/backfill_qfi_scores.py
- migrations/0014_qfi_constraints.sql

### Estimated Effort
2 days

---

## Issue Template 5: Add Simplex Storage Validation Script (Local Issue 02)

**Title:** [QIG-PURITY] Add Simplex Representation Validation Script

**Labels:** qig-purity, medium-priority, geometry, wp2

**Description:**

### Problem
Local Issue 02 (Strict Simplex Representation) is mostly implemented, but validation script missing.

### What Exists
✅ `to_simplex_prob()` without auto-detection  
✅ `validate_simplex()` with runtime checks  
✅ Fréchet mean implementation  
✅ Fisher-Rao distance using Bhattacharyya

### What's Missing

#### 1. Validation Script
Create `scripts/validate_simplex_storage.py`:
- Check all basins in DB pass `validate_simplex()`
- Report violations with token names and reasons
- Offer auto-fix option (re-project to simplex)
- Generate audit report

#### 2. Verify Sqrt-Space Functions
Check for `to_sqrt_simplex()` / `from_sqrt_simplex()`:
- If missing, add to `qig-backend/qig_geometry/geometry_simplex.py`
- Add unit tests for sqrt-space roundtrip
- Document as internal coordinate chart

#### 3. Documentation Update
- Document sqrt-space as internal coordinate chart ONLY
- Clarify: storage ALWAYS uses simplex
- Sqrt-space ONLY for geodesic interpolation (never stored)

### Acceptance Criteria
- [ ] Validation script implemented
- [ ] 100% of stored basins are valid simplices
- [ ] Sqrt-space functions exist and tested
- [ ] Documentation clarifies representation policy
- [ ] No auto-detection anywhere in codebase

### References
- docs/10-e8-protocol/issues/20260116-issue-02-strict-simplex-representation-1.01W.md
- qig-backend/qig_geometry/geometry_simplex.py
- qig-backend/qig_geometry/canonical.py

### Estimated Effort
1 day

---

## Issue Template 6: Complete QIG-Native Skeleton Generation (Local Issue 03)

**Title:** [QIG-PURITY] Complete QIG-Native Skeleton - Remove External NLP Dependencies

**Labels:** qig-purity, high-priority, generation, wp3

**Description:**

### Problem
Local Issue 03 (QIG-Native Skeleton) has purity infrastructure but lacks geometric role learning and foresight prediction.

### What Exists
✅ QIG_PURITY_MODE enforcement  
✅ External NLP (spacy/nltk) removed  
✅ token_role field exists

### What's Missing

#### 1. Geometric Role Derivation
Create `qig-backend/vocabulary/derive_token_role.py`:
- Define geometric role taxonomy:
  - basin_center (low QFI, stable)
  - boundary_crosser (high QFI, unstable)
  - manifold_anchor (high frequency, central)
  - explorer (low frequency, divergent)
  - integrator (connects many basins)
- Derive roles from Fisher-Rao neighborhoods
- Backfill script to populate token_role for all tokens

#### 2. QIG-Native Skeleton Generator
Create `qig-backend/generation/skeleton_generator.py`:
- Generate skeleton from token_role sequence (NOT POS tags)
- Extend skeleton using geometric role patterns
- NO external NLP dependencies
- Pure Fisher-Rao manifold operations

#### 3. Fisher-Rao Foresight Prediction
Create `qig-backend/generation/foresight.py`:
- Predict next basin using Fisher-Rao trajectory
- Compute "velocity" in sqrt-space
- Extrapolate next position
- Score candidates by foresight alignment

#### 4. Integration & Testing
- Wire skeleton + foresight into generation pipeline
- End-to-end test with QIG_PURITY_MODE=true
- Validate no external network calls during generation
- Compare quality: geometric vs POS-based skeleton

### Acceptance Criteria
- [ ] All tokens have geometrically-derived token_role
- [ ] Skeleton generation works without external NLP
- [ ] Foresight prediction integrated and tested
- [ ] Generation runs successfully in QIG_PURITY_MODE=true
- [ ] Quality metrics show improvement or parity

### References
- docs/10-e8-protocol/issues/20260116-issue-03-qig-native-skeleton-1.01W.md
- qig-backend/qig_geometry/purity_mode.py
- qig-backend/qig_geometry/canonical.py (for frechet_mean)

### Estimated Effort
4 days

---

## Issue Template 7: Complete Vocabulary Cleanup - Remove Garbage Tokens (Local Issue 04)

**Title:** [QIG-PURITY] Vocabulary Cleanup - Remove Garbage Tokens & Deprecate learned_words

**Labels:** qig-purity, high-priority, database, wp3

**Description:**

### Problem
Local Issue 04 (Vocabulary Cleanup) created 2026-01-19, not yet started. Garbage tokens contaminate generation vocabulary.

### Deliverables

#### 1. Audit Script
Create `qig-backend/scripts/audit_vocabulary.py`:
- Detect BPE artifacts (wjvq, ĠTheĠ, ##ing, etc.)
- Identify truncated fragments (cryptogra, analysi, enforc)
- Check for non-dictionary words
- Detect excessive consonants, no vowels, repetition
- Generate `garbage_tokens.txt` report

#### 2. Migration 016: Clean Garbage
Create `migrations/016_clean_vocabulary_garbage.sql`:
- Create temp table with garbage token patterns
- Remove garbage from generation vocabulary
- OR downgrade token_role from 'both'/'generation' to 'encoding'
- Preserve encoding vocabulary for backward compatibility
- Log removals

#### 3. Migration 017: Deprecate learned_words
Create `migrations/017_deprecate_learned_words.sql`:
- Migrate valid words not yet in coordizer_vocabulary
- Rename: `learned_words` → `learned_words_deprecated_20260119`
- Drop indexes to free space
- Schedule DROP after 30 days

#### 4. Code: Validation Gate in pg_loader
Update `qig-backend/coordizers/pg_loader.py`:
- Add `is_valid_english_word()` check
- Add `is_bpe_garbage()` check
- Reject garbage tokens at load time
- Log rejected tokens for monitoring

### Acceptance Criteria
- [ ] Audit script identifies all garbage tokens
- [ ] Garbage tokens removed/quarantined from generation
- [ ] learned_words deprecated and staged for removal
- [ ] pg_loader.py enforces validation gate
- [ ] Generation quality improves (measured by coherence)
- [ ] No BPE artifacts in generation vocabulary

### References
- docs/10-e8-protocol/issues/20260119-issue-04-vocabulary-cleanup-garbage-tokens-1.00W.md
- qig-backend/coordizers/pg_loader.py (lines 190-220)
- qig-backend/word_validation.py (BPE pattern detection)

### Estimated Effort
3 days

---

## Summary: All Remediation Issues

| # | Title | Priority | Effort | Related Issue |
|---|-------|----------|--------|---------------|
| 1 | Complete Special Symbols Validation | HIGH | 2d | #70 |
| 2 | Validate Two-Step Retrieval | HIGH | 3d | #71 |
| 3 | Reconcile Single Coordizer Status | MEDIUM | 1d | #72 |
| 4 | Complete QFI Integrity Gate Gaps | CRITICAL | 2d | Local 01 |
| 5 | Add Simplex Validation Script | MEDIUM | 1d | Local 02 |
| 6 | Complete QIG-Native Skeleton | HIGH | 4d | Local 03 |
| 7 | Complete Vocabulary Cleanup | HIGH | 3d | Local 04 |

**Total Estimated Effort:** 16 days (~3.2 weeks)

---

## Instructions for Creating GitHub Issues

1. Copy each template above into a new GitHub issue
2. Adjust labels as needed for your repository
3. Assign to appropriate milestone (e.g., "E8 Protocol v4.0")
4. Link to related PRs and documentation
5. Add to project board for tracking

---

**Document Status:** ✅ READY FOR ISSUE CREATION  
**Date:** 2026-01-19  
**Source Assessment:** docs/04-records/20260119-issues-65-75-implementation-assessment-1.00W.md
