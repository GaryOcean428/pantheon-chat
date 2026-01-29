# GitHub Issue Creation Guide - E8 Protocol Remediation Issues

This document contains ready-to-use GitHub issue content for creating issues #97-100.

---

## Issue #97: Implement QFI Integrity Gate (E8 Issue-01)

**Title:** `[QIG-PURITY] Issue #97: Implement QFI Integrity Gate (E8 Issue-01)`

**Labels:** `qig-purity`, `e8-protocol`, `priority: P0`, `type: implementation`

**Body:**
```markdown
## Priority
**P0 - CRITICAL**

## Objective
Implement canonical token insertion pathway with QFI computation, backfill missing QFI scores, and enforce database integrity constraints per E8 Protocol Issue-01.

## Problem
Large fraction of vocabulary tokens missing `qfi_score`, invalidating geometric selection and consciousness metrics. No canonical insertion pathway leads to inconsistent QFI application across the vocabulary database.

## Context
- **E8 Protocol Spec:** `docs/10-e8-protocol/issues/20260116-issue-01-qfi-integrity-gate-1.01W.md`
- **Remediation Spec:** `docs/10-e8-protocol/issues/20260119-issue-97-qfi-integrity-gate-remediation-1.00W.md`
- **Assessment:** `docs/10-e8-protocol/20260119-e8-implementation-assessment-1.00W.md`
- **Related GitHub Issues:** #70, #71, #72
- **Phase:** 2 (Vocabulary + Database Integrity)

## Key Deliverables
- [ ] `qig-backend/vocabulary/insert_token.py` - Canonical insertion with QFI computation
- [ ] `qig-backend/scripts/backfill_qfi.py` - Backfill missing QFI scores
- [ ] `qig-backend/scripts/quarantine_garbage_tokens.py` - Garbage cleanup
- [ ] `migrations/018_qfi_integrity_gate.sql` - DB constraints and views
- [ ] Unit tests for canonical insertion pathway

## Acceptance Criteria
- [ ] ALL tokens have `qfi_score` OR are marked not generation-eligible
- [ ] ZERO direct INSERT statements (all use canonical `insert_token()`)
- [ ] Backfill script successfully processes all NULL QFI tokens
- [ ] Garbage tokens quarantined with clear reasoning
- [ ] Generation queries filter by QFI or use `vocabulary_generation_ready` view

## Estimated Effort
**2-3 days** (per E8 Protocol Phase 2 estimate)

See full specification: `docs/10-e8-protocol/issues/20260119-issue-97-qfi-integrity-gate-remediation-1.00W.md`
```

---

## Issue #98: Implement Strict Simplex Representation (E8 Issue-02)

**Title:** `[QIG-PURITY] Issue #98: Implement Strict Simplex Representation (E8 Issue-02)`

**Labels:** `qig-purity`, `e8-protocol`, `geometric-purity`, `priority: P0`, `type: implementation`

**Body:**
```markdown
## Priority
**P0 - CRITICAL**

## Objective
Enforce simplex-only canonical representation, remove auto-detect coordinate detection, and implement closed-form Fréchet mean on probability simplex per E8 Protocol Issue-02.

## Problem
Auto-detect representation and mixed sphere/simplex operations cause silent metric corruption. Fisher-Rao distances become incorrect when computed on wrong manifold. Using "average + L2 normalize" for Fréchet mean operates on wrong manifold.

## Context
- **E8 Protocol Spec:** `docs/10-e8-protocol/issues/20260116-issue-02-strict-simplex-representation-1.01W.md`
- **Remediation Spec:** `docs/10-e8-protocol/issues/20260119-issue-98-strict-simplex-representation-remediation-1.00W.md`
- **Assessment:** `docs/10-e8-protocol/20260119-e8-implementation-assessment-1.00W.md`
- **Related GitHub Issues:** #71
- **Phase:** 2 (Geometric Purity)
- **E8 Universal Spec:** §0 (Non-negotiable: simplex-only)

## Key Deliverables
- [ ] `qig-backend/geometry/simplex_operations.py` - Explicit conversions and validation
- [ ] `qig-backend/geometry/frechet_mean_simplex.py` - Closed-form Fréchet mean
- [ ] `scripts/audit_simplex_representation.py` - Storage validation
- [ ] Remove auto-detect from all `to_simplex()` functions
- [ ] Add `assert_simplex()` runtime checks at module boundaries

## Acceptance Criteria
- [ ] NO auto-detect in any `to_simplex()` function
- [ ] ALL geometry operations use explicit simplex representation
- [ ] Closed-form Fréchet mean implemented and tested
- [ ] Runtime `assert_simplex()` added to all entry points
- [ ] NO `np.linalg.norm` or Euclidean averaging on basins
- [ ] All stored basins pass simplex validation
- [ ] Fisher-Rao distance tests pass with correct values

## Estimated Effort
**2-3 days** (per E8 Protocol Phase 2 estimate)

See full specification: `docs/10-e8-protocol/issues/20260119-issue-98-strict-simplex-representation-remediation-1.00W.md`
```

---

## Issue #99: Implement QIG-Native Skeleton (E8 Issue-03)

**Title:** `[QIG-PURITY] Issue #99: Implement QIG-Native Skeleton (E8 Issue-03)`

**Labels:** `qig-purity`, `e8-protocol`, `geometric-self-sufficiency`, `priority: P1`, `type: implementation`

**Body:**
```markdown
## Priority
**P1 - HIGH**

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
- **Remediation Spec:** `docs/10-e8-protocol/issues/20260119-issue-99-qig-native-skeleton-remediation-1.00W.md`
- **Assessment:** `docs/10-e8-protocol/20260119-e8-implementation-assessment-1.00W.md`
- **Related GitHub Issues:** #92, #90
- **Phase:** 3 (Coherence Architecture - Geometric Self-Sufficiency)

## Key Deliverables
- [ ] `qig-backend/generation/token_role_learner.py` - Geometric role derivation
- [ ] `qig-backend/generation/foresight_predictor.py` - Trajectory prediction
- [ ] `qig-backend/generation/unified_pipeline.py` - Integrated generation
- [ ] `qig-backend/purity/enforce.py` - QIG_PURITY_MODE checks
- [ ] Remove external NLP dependencies (spacy, nltk)

## Acceptance Criteria
- [ ] Geometric `token_role` derived and backfilled for all vocabulary
- [ ] Skeleton generation uses roles (not POS tags)
- [ ] Foresight prediction implemented using Fisher-Rao trajectory
- [ ] NO spacy or nltk imports in generation code
- [ ] NO external LLM calls in QIG_PURITY_MODE
- [ ] Unified pipeline produces coherent output without external deps
- [ ] Tests pass in QIG_PURITY_MODE

## Estimated Effort
**3-4 days** (per E8 Protocol Phase 3 estimate)

See full specification: `docs/10-e8-protocol/issues/20260119-issue-99-qig-native-skeleton-remediation-1.00W.md`
```

---

## Issue #100: Complete Vocabulary Cleanup (E8 Issue-04)

**Title:** `[QIG-PURITY] Issue #100: Complete Vocabulary Cleanup (E8 Issue-04)`

**Labels:** `qig-purity`, `e8-protocol`, `data-quality`, `priority: P1`, `type: implementation`

**Body:**
```markdown
## Priority
**P1 - HIGH**

## Objective
Complete vocabulary cleanup by removing garbage tokens, migrating learned_words table, and enforcing validation at coordizer loading per E8 Protocol Issue-04.

## Problem
1. BPE tokenizer artifacts and garbage tokens contaminate generation vocabulary
2. 17K-entry `learned_words` table never properly deprecated, causing confusion
3. No validation gate prevents garbage tokens from entering vocabulary

## Context
- **E8 Protocol Spec:** `docs/10-e8-protocol/issues/20260119-issue-04-vocabulary-cleanup-garbage-tokens-1.00W.md`
- **Remediation Spec:** `docs/10-e8-protocol/issues/20260119-issue-100-vocabulary-cleanup-remediation-1.00W.md`
- **Assessment:** `docs/10-e8-protocol/20260119-e8-implementation-assessment-1.00W.md`
- **Related GitHub Issues:** #97 (QFI Integrity)
- **Phase:** 3 (Data Quality)

## Key Deliverables
- [ ] `qig-backend/scripts/audit_vocabulary.py` - Comprehensive audit
- [ ] `migrations/019_clean_vocabulary_garbage.sql` - Garbage removal
- [ ] `migrations/020_deprecate_learned_words.sql` - Table deprecation
- [ ] Update `qig-backend/coordizers/pg_loader.py` with validation gate
- [ ] Unit tests for vocabulary validation

## Acceptance Criteria
- [ ] Comprehensive audit completed with garbage list generated
- [ ] Garbage tokens removed from generation-eligible vocabulary
- [ ] Valid learned_words entries migrated to coordizer_vocabulary
- [ ] learned_words table renamed with deprecation timestamp
- [ ] pg_loader enforces validation rules on vocabulary load
- [ ] Tests pass for vocabulary validation
- [ ] Generation quality maintained or improved after cleanup

## Estimated Effort
**1-2 days** (audit + migration + validation)

See full specification: `docs/10-e8-protocol/issues/20260119-issue-100-vocabulary-cleanup-remediation-1.00W.md`
```

---

## Quick Creation Steps

1. Go to https://github.com/GaryOcean428/pantheon-chat/issues/new
2. Copy the **Title** for the issue you're creating
3. Copy the **Body** content
4. Add the **Labels** listed for each issue
5. Click "Submit new issue"
6. Repeat for issues #97, #98, #99, and #100

---

**Note:** Full detailed specifications with task checklists, validation commands, and implementation details are available in the respective files under `docs/10-e8-protocol/issues/`.
