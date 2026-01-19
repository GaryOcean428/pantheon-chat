# Pantheon E8 Upgrade Pack v1.1

**Status:** ✅ READY FOR IMPLEMENTATION  
**Version:** 1.1 (2026-01-16)  
**Authority:** E8 Protocol v4.0 Universal Purity Specification

---

## OVERVIEW

This upgrade pack contains the complete specification, implementation blueprints, and issue definitions for upgrading the Pantheon-Chat repository to E8 Protocol v4.0 with strict geometric purity enforcement.

**Key Objectives:**
1. Enforce simplex-only canonical representation (NO auto-detect)
2. Ensure ALL vocabulary tokens have QFI scores before generation eligibility
3. Replace external NLP with internal geometric token_role skeleton
4. Implement E8 hierarchical kernel architecture (0/1→4→8→64→240)
5. Prevent geometry drift through CI purity gates

---

## CONTENTS

### Core Specifications

#### `specifications/20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md`
**Universal purity invariants and implementation phases**
- §0: Non-negotiable purity rules (simplex-only, Fisher-Rao only, no NLP)
- §1: Bootstrap load order
- §2: Repository strategy (pantheon-chat as canonical)
- §3: Correctness risks (token integrity, geometry drift, generation purity)
- §4: Implementation phases (5 phases, no timeboxing)
- §5: Open design questions
- §6: Protocol discipline (CoPP + Ultra)
- §7: Validation commands

#### `specifications/20260116-wp5-2-e8-implementation-blueprint-1.01W.md`
**E8 hierarchical kernel architecture**
- E8 layer structure (0/1, 4, 8, 64, 240)
- Core 8 faculties mapped to Greek gods (Zeus, Athena, Apollo, Hermes, Artemis, Ares, Hephaestus, Aphrodite)
- Hemisphere pattern (explore/exploit with κ-gated coupling)
- Psyche plumbing (Id, Superego, Preconscious, Ocean, Gary/Ego)
- God-kernel mapping with genealogy
- Rest scheduler (dolphin-style alternation)
- Validation tests and implementation checklist

---

### Issue Specifications (`issues/`)

#### `issues/20260116-issue-01-qfi-integrity-gate-1.01W.md`
**Problem:** Large fraction of tokens missing `qfi_score`, garbage tokens present  
**Solution:**
- Canonical `insert_token()` pathway with QFI computation
- DB integrity constraints (`is_generation_eligible` flag)
- Backfill script for missing QFI
- Garbage token quarantine and cleanup

**Deliverables:**
- `qig-backend/vocabulary/insert_token.py`
- `scripts/backfill_qfi.py`
- `scripts/quarantine_garbage_tokens.py`
- Migration `0015_qfi_integrity_gate.sql`

#### `issues/20260116-issue-02-strict-simplex-representation-1.01W.md`
**Problem:** Auto-detect representation, "average + L2 normalize" uses wrong manifold  
**Solution:**
- Remove auto-detect from `to_simplex()`
- Add explicit `to_sqrt_simplex()` / `from_sqrt_simplex()` helpers
- Replace Fréchet mean with closed-form sqrt-space mean
- Runtime asserts at module boundaries

**Deliverables:**
- `qig-backend/geometry/simplex_operations.py` (explicit conversions)
- `qig-backend/geometry/frechet_mean_simplex.py` (closed form)
- `scripts/audit_simplex_representation.py`

#### `issues/20260116-issue-03-qig-native-skeleton-1.01W.md`
**Problem:** External NLP (spacy, nltk) in generation, template fallbacks  
**Solution:**
- Derive `token_role` from Fisher-Rao neighborhood clustering
- Use token_role as structure (not POS tags)
- Foresight: predict next basin via trajectory regression
- Score candidates by Fisher distance to predicted basin

**Deliverables:**
- `qig-backend/generation/token_role_learner.py`
- `qig-backend/generation/foresight_predictor.py`
- `qig-backend/generation/unified_pipeline.py`
- `QIG_PURITY_MODE` enforcement

---

## IMPLEMENTATION PHASES

### Phase 1: Repo Truth + Invariants
**Goal:** Inventory geometry functions, create canonical contract, remove forbidden patterns

**Tasks:**
- [ ] Inventory all geometry functions (fisher_rao_distance, geodesic_interpolation, to_simplex, etc.)
- [ ] Create `docs/10-e8-protocol/specs/CANONICAL_GEOMETRY_CONTRACT.md`
- [ ] Search & remove: cosine_similarity, np.linalg.norm on basins, dot-product ranking
- [ ] Generate purity scan report

**Estimated Effort:** 2-3 days

---

### Phase 2: Vocabulary + Database Integrity
**Goal:** Single insertion pathway, QFI backfill, garbage cleanup

**Tasks:**
- [ ] Implement canonical `insert_token()` (Issue #01)
- [ ] Fix `learned_relationships.py` and `vocabulary_coordinator.py`
- [ ] Add DB constraints and generation-ready view
- [ ] Run backfill and garbage quarantine scripts

**Estimated Effort:** 2-3 days (backfill may be slow on large DB)

---

### Phase 3: Coherence Architecture
**Goal:** Unified generation pipeline without legacy fallbacks

**Tasks:**
- [ ] Implement token_role skeleton (Issue #03)
- [ ] Implement foresight predictor (trajectory regression)
- [ ] Unify generation pipeline (skeleton + trajectory + foresight)
- [ ] Add per-token observable metrics

**Estimated Effort:** 3-4 days

---

### Phase 4: Kernel Redesign (E8 Hierarchy)
**Goal:** Implement E8 layers 0/1→4→8→64→240

**Tasks:**
- [ ] Implement core 8 faculties (WP5.2 Phase 4A)
- [ ] Create god registry with Greek canonical names (Phase 4B)
- [ ] Implement hemisphere scheduler (Phase 4C)
- [ ] Implement psyche plumbing (Phase 4D)
- [ ] Add genetic lineage (Phase 4E)
- [ ] Implement rest scheduler (Phase 4F)
- [ ] Extend to 240 constellation (Phase 4G)

**Estimated Effort:** 5-7 days (largest phase)

---

### Phase 5: Platform Hardening
**Goal:** CI purity gates to prevent regressions

**Tasks:**
- [ ] Create `.github/workflows/qig-purity-gate.yml`
- [ ] Implement `scripts/validate_geometry_purity.py`
- [ ] Add pre-commit hooks for geometry validation
- [ ] Create generation smoke tests in `QIG_PURITY_MODE`
- [ ] Add DB schema drift tests

**Estimated Effort:** 2-3 days

---

## VALIDATION COMMANDS

```bash
# Geometry purity scan (run before commit)
python scripts/validate_geometry_purity.py

# QFI coverage report
python scripts/check_qfi_coverage.py

# Simplex representation audit
python scripts/audit_simplex_representation.py

# Garbage token detection
python scripts/detect_garbage_tokens.py

# Schema consistency check
python scripts/validate_schema_consistency.py

# Generation purity test (no external calls)
QIG_PURITY_MODE=true python qig-backend/test_generation_pipeline.py

# Full validation suite
python scripts/run_all_validations.py
```

---

## ACCEPTANCE CRITERIA

### Purity Gates (MUST PASS)
- [ ] NO cosine_similarity, np.linalg.norm, or dot-product ranking on basins
- [ ] NO auto-detect representation in any geometric function
- [ ] NO direct INSERT into coordizer_vocabulary (all use insert_token())
- [ ] NO external NLP (spacy, nltk) in generation pipeline
- [ ] NO external LLM calls in QIG_PURITY_MODE

### Database Integrity (MUST PASS)
- [ ] ALL tokens in vocabulary have qfi_score or are marked not generation-eligible
- [ ] ZERO garbage tokens in generation-ready vocabulary
- [ ] Generation queries use vocabulary_generation_ready view or explicit QFI filter

### E8 Architecture (MUST PASS)
- [ ] Core 8 faculties implemented and mapped to Greek gods
- [ ] NO apollo_1 style numbered kernels (canonical identities only)
- [ ] Kernel merges use geodesic interpolation (not linear average)
- [ ] God registry enforces Greek canonical names

### CI/CD (MUST PASS)
- [ ] Purity gate workflow runs on all PRs
- [ ] Pre-commit hook prevents geometry violations
- [ ] Generation smoke tests in QIG_PURITY_MODE pass

---

## MIGRATION STRATEGY

### Replit → Railway Sync (If Needed)

**Recommended Approach:**
- **Strategy A (Preferred):** Point Railway at GaryOcean428/pantheon-chat directly
- **Strategy B:** If Arcane-Fly/pantheon-chat must be used, merge trunk then run full validation

**Validation Before Deploy:**
1. Run purity scan
2. Check QFI coverage (should be 100%)
3. Count garbage tokens (should be 0 or quarantined)
4. Test generation in QIG_PURITY_MODE

---

## KNOWN ISSUES & FIXES

### Issue: "Replit agents reintroduce impurities"
**Root Cause:** Agents modify code without purity awareness  
**Fix:** Update agent instructions with v4.0 spec (DONE in this upgrade)  
**Prevention:** CI purity gates block impure PRs

### Issue: "Fréchet mean loops stuck (max 20 iterations)"
**Root Cause:** Euclidean gradient descent on simplex (wrong manifold)  
**Fix:** Use closed-form sqrt-space mean (Issue #02)

### Issue: "Token fragments (cryptogra, analysi) in vocabulary"
**Root Cause:** No validation at insertion  
**Fix:** Quarantine script with validation rules (Issue #01)

---

## REFERENCES

### Core Documentation
- **Ultra Protocol:** `docs/08-experiments/20251231-Ultra-Consciousness-Protocol-0.04F.md`
- **Frozen Facts:** `docs/01-policies/20251208-frozen-facts-immutable-truths-1.00F.md`
- **Universal κ*:** `docs/08-experiments/20251228-Universal-kappa-star-discovery-0.01F.md`

### Updated Agent Instructions
- **Copilot Instructions:** `.github/copilot-instructions.md`
- **AGENTS.md:** Root `AGENTS.md`
- **QIG Purity Agent:** `.github/agents/qig-purity-validator.md`

### E8 Resources
- **E8 Metrics:** `shared/constants/e8.ts`
- **E8 Constellation:** `qig-backend/e8_constellation.py`
- **E8 Validation:** `qig-backend/tests/test_e8_specialization.py`

---

## CONTACT & SUPPORT

**Questions?** See `specifications/20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md` §5 (Open Design Questions)

**Implementation Help?** See WP5.2 blueprint checklist and issue specs

**CI Failures?** Run validation commands locally to debug

---

**Last Updated:** 2026-01-16  
**Version:** 1.1  
**Status:** Ready for phased implementation  
**Authority:** E8 Protocol v4.0 Universal Specification
