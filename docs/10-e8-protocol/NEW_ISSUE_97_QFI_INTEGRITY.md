# Issue #97: Implement QFI Integrity Gate (E8 Issue-01)

## Priority
**P0 - CRITICAL**

## Type
`type: implementation`, `qig-purity`, `e8-protocol`

## Objective
Implement canonical token insertion pathway with QFI computation, backfill missing QFI scores, and enforce database integrity constraints per E8 Protocol Issue-01.

## Problem
Large fraction of vocabulary tokens missing `qfi_score`, invalidating geometric selection and consciousness metrics. No canonical insertion pathway leads to inconsistent QFI application across the vocabulary database.

## Context
- **E8 Protocol Spec:** `docs/10-e8-protocol/issues/20260116-issue-01-qfi-integrity-gate-1.01W.md`
- **Related GitHub Issues:** #70, #71, #72
- **Phase:** 2 (Vocabulary + Database Integrity)
- **Assessment:** `docs/10-e8-protocol/IMPLEMENTATION_ASSESSMENT.md`

## Tasks

### 1. Create Canonical Token Insertion Pathway
- [ ] Create `qig-backend/vocabulary/insert_token.py` with QFI computation
- [ ] Implement `canonical_insert_token(word, basin_coords, ...)` function
- [ ] Compute QFI score before insertion (required for generation eligibility)
- [ ] Add validation: non-negative simplex, sum=1, dimension=64
- [ ] Return structured result with insertion status

### 2. Fix Existing Vocabulary Modules
- [ ] Update `qig-backend/vocabulary/learned_relationships.py` to use `insert_token()`
- [ ] Update `qig-backend/vocabulary/vocabulary_coordinator.py` to use `insert_token()`
- [ ] Remove any direct INSERT statements into `coordizer_vocabulary`
- [ ] Add deprecation warnings for old insertion methods

### 3. Database Integrity Constraints
- [ ] Create migration `migrations/018_qfi_integrity_gate.sql` (next available sequential number)
- [ ] Add `is_generation_eligible` computed column based on QFI presence
- [ ] Add CHECK constraint: `qfi_score IS NOT NULL OR is_generation_eligible = FALSE`
- [ ] Create `vocabulary_generation_ready` view filtering by QFI
- [ ] Update generation queries to use view or explicit QFI filter

### 4. QFI Backfill Script
- [ ] Create `qig-backend/scripts/backfill_qfi.py`
- [ ] Identify all tokens with NULL qfi_score
- [ ] Compute QFI for each token from basin coordinates
- [ ] Update records with computed QFI
- [ ] Generate report: total processed, successful, failed
- [ ] Add dry-run mode for validation

### 5. Garbage Token Cleanup
- [ ] Create `qig-backend/scripts/quarantine_garbage_tokens.py`
- [ ] Detect BPE artifacts (e.g., "##ing", "@@", byte-pair fragments)
- [ ] Detect non-words (random character sequences)
- [ ] Move to `coordizer_vocabulary_quarantine` table
- [ ] Generate quarantine report with reasoning
- [ ] Allow manual review and restoration

### 6. Integration & Testing
- [ ] Update coordizer to load only generation-eligible tokens
- [ ] Add unit tests for `insert_token()` validation
- [ ] Add integration test for backfill script
- [ ] Test generation pipeline with QFI-filtered vocabulary
- [ ] Verify no performance degradation

## Deliverables

| File | Description | Status |
|------|-------------|--------|
| `qig-backend/vocabulary/insert_token.py` | Canonical insertion with QFI | ❌ TODO |
| `qig-backend/scripts/backfill_qfi.py` | Backfill missing QFI scores | ❌ TODO |
| `qig-backend/scripts/quarantine_garbage_tokens.py` | Garbage cleanup | ❌ TODO |
| `migrations/018_qfi_integrity_gate.sql` | DB constraints | ❌ TODO |
| `qig-backend/tests/test_insert_token.py` | Unit tests | ❌ TODO |

## Acceptance Criteria
- [ ] ALL tokens in `coordizer_vocabulary` have `qfi_score` OR are marked not generation-eligible
- [ ] ZERO direct INSERT statements into vocabulary tables (all use `insert_token()`)
- [ ] Backfill script successfully processes all NULL QFI tokens
- [ ] Garbage tokens quarantined with clear reasoning
- [ ] Generation queries explicitly filter by QFI or use `vocabulary_generation_ready` view
- [ ] Tests pass for canonical insertion pathway
- [ ] CI validates QFI coverage on every commit

## Dependencies
- **Requires:** Canonical geometry module with QFI computation
- **Blocks:** Issue #99 (QIG-Native Skeleton), Issue #100 (Vocabulary Cleanup)
- **Related:** Issue #72 (Single Coordizer Implementation)

## Validation Commands
```bash
# Check QFI coverage
python qig-backend/scripts/check_qfi_coverage.py

# Backfill missing QFI (dry run)
python qig-backend/scripts/backfill_qfi.py --dry-run

# Quarantine garbage tokens (dry run)
python qig-backend/scripts/quarantine_garbage_tokens.py --dry-run --report

# Validate vocabulary integrity
python qig-backend/scripts/validate_vocabulary_integrity.py

# Run unit tests
pytest qig-backend/tests/test_insert_token.py -v
```

## References
- **E8 Protocol Universal Spec:** §3 (Correctness Risks - Token Integrity)
- **Issue Spec:** `docs/10-e8-protocol/issues/20260116-issue-01-qfi-integrity-gate-1.01W.md`
- **Assessment:** `docs/10-e8-protocol/IMPLEMENTATION_ASSESSMENT.md` Section 2

## Estimated Effort
**2-3 days** (per E8 Protocol Phase 2 estimate)

---

**Status:** TO DO  
**Created:** 2026-01-19  
**Priority:** P0 - CRITICAL  
**Phase:** 2 - Core Integrity
