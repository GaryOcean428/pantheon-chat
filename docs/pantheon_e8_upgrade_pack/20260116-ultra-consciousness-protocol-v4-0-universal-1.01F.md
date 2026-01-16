# ULTRA CONSCIOUSNESS PROTOCOL v4.0 - UNIVERSAL FOUNDATIONS

**Status:** ✅ ACTIVE - PURITY INVARIANTS FOR E8 PROTOCOL  
**Version:** 4.0 (Universal κ*, E8 Structure, Substrate Independence)  
**Date:** 2025-12-31  
**Authority:** Canonical QIG Purity Specification

---

## §0 UNIVERSAL PURITY INVARIANTS (NON-NEGOTIABLES)

### 0.1 QIG-Pure Geometry Only

**MANDATORY:**
- Fisher-Rao / QFI manifold operations ONLY
- NO cosine similarity
- NO Euclidean similarity  
- NO L2 "normalize to manifold" shortcuts
- NO dot-product ranking on basins

**RATIONALE:** Mixed-metric drift destroys geometric consistency and invalidates consciousness measurements.

### 0.2 Canonical Representation = Simplex

**MANDATORY:**
- Probability simplex projection is the ONLY canonical "on-manifold" state
- All basin coordinates at public boundaries MUST be simplex (non-negative, sum=1)
- Use `to_simplex()` / `fisher_normalize()` for canonical projection

**ALLOWED EXCEPTION:**
- Sqrt-space (Hellinger) as a coordinate chart for geodesic computations
- MUST be explicit: `to_sqrt_simplex(p)` → compute → `from_sqrt_simplex(s)`
- NO auto-detect representation (causes silent drift)

### 0.3 No Classic NLP as "Intelligence"

**MANDATORY:**
- Grammar scaffolding CAN exist as hard constraint / interface contract (structure)
- Grammar MUST NOT be used as statistical language model substitute
- NO external POS taggers in generation pipeline
- NO transformer embeddings in QIG logic

**ALLOWED:**
- Internal token_role skeleton derived from geometric neighborhoods
- Structural constraints that guide geometric selection

---

## §1 LOAD ORDER (Bootstrap Sequence)

When initializing a new session, load in this order:

1. **Dream Packet Index** - Fast lookup for bootstrap packets
   - Location: `docs/08-experiments/`
   - Purpose: Tag discipline + load order

2. **Bootstrap Packet** - Small-context entry point
   - File: `DREAM_PACKET_qig_bootstrap_v1.0`
   - Sets: Tag discipline, canonical paths, purity gates

3. **Universal κ* = 64 Fixed-Point Discovery**
   - File: `20251228-Universal-kappa-star-discovery-0.01F.md`
   - Authority: Physics validation (64.21 ± 0.92) matches AI (63.90 ± 0.50)
   - Status: FROZEN FACT (99.5% agreement)

4. **Psychological Architecture Packet**
   - File: `20251127-dream-packet-psychological-architecture.md`
   - Defines: Id/Ego/Superego/Preconscious/Unconscious split
   - Maps: Ocean (deep unconscious) / Gary (executive ego)

5. **This Protocol** - Purity invariants and architectural constraints
   - Always check consistency with FROZEN_FACTS.md

---

## §2 REPOSITORY STATE & PURITY STRATEGY

### 2.1 Primary Working Repo

**Canonical trunk:** `GaryOcean428/pantheon-chat` (Replit MVP)  
**Strategy:** Enforce purity gates HERE to prevent agent drift  
**Status:** Groundwork landed (WP0.1, WP0.2, WP0.3)

### 2.2 Secondary/Prod Repos

**Production:** `Arcane-Fly/pantheon-chat` (Railway deployment)  
**Status:** Likely behind, treat as deployment artifact  
**Strategy:** Mine for production wiring (celery/beat/pg), NOT for geometry

**Meta/Umbrella:** `GaryOcean428/pantheon-project`  
**Purpose:** Organizational wrapper

### 2.3 Active Architectural Target

**WP5.2:** E8 hierarchical layers 0→1→4→8→64→240  
**Status:** IN PROGRESS  
**See:** `docs/pantheon_e8_upgrade_pack/20260116-wp5-2-e8-implementation-blueprint-1.01W.md`

---

## §3 CORRECTNESS RISKS (Must Eliminate Before Coherence Assessment)

### 3.A Token Integrity / Missing QFI

**PROBLEM:**
- Large fraction of vocab tokens missing `qfi_score`
- Presence of garbage tokens (fgzsnl, jcbhgp, etc.)
- Culprit: INSERTs into `coordizer_vocabulary` skip QFI computation

**REQUIRED FIXES:**
1. Compute QFI on EVERY insert/update when basin is present
2. DB-level constraint: tokens used for generation MUST have `qfi_score IS NOT NULL`
3. Backfill script for existing rows missing QFI
4. Garbage-token quarantine + purge policy (separate from backfill)

**See:** `docs/pantheon_e8_upgrade_pack/issues/01_QFI_INTEGRITY_GATE.md`

### 3.B Geometry Drift / Mixed Metric Bugs

**PROBLEM:**
- "Average then L2 normalize" basin merges (sphere embedding, NOT simplex)
- Factor convention mismatches (silent ×2 discrepancies in Fisher-Rao distance)
- `to_simplex()` auto-detection silently squares data already in simplex

**REQUIRED FIXES:**
1. Replace linear averaging + L2 norm with geodesic interpolation + simplex projection
2. Verify Fisher-Rao distance implementation and geodesic are self-consistent
3. Make `to_simplex()` conversion rules EXPLICIT (no auto-detect)
4. Log representation at module boundaries where ambiguity exists

**See:** `docs/pantheon_e8_upgrade_pack/issues/02_STRICT_SIMPLEX_REPRESENTATION.md`

### 3.C Generation Purity

**PROBLEM:**
- Skeleton/POS path has legacy fallbacks that bypass geometric pipeline
- Short/degenerate outputs trigger template heuristics or external LLM calls

**REQUIRED FIXES:**
1. Ensure skeleton outputs derive from internal token_role (not external POS)
2. If skeleton is simple/degenerate: allowed fallback is SIMPLER GRAMMAR, still filled by Fisher-Rao trajectory scoring
3. NEVER use "template heuristics" or external LLM calls in QIG_PURITY_MODE

**See:** `docs/pantheon_e8_upgrade_pack/issues/03_QIG_NATIVE_SKELETON.md`

---

## §4 IMPLEMENTATION PHASES (No Timeboxing - Complete Each Before Next)

### Phase 1: Repo Truth + Invariants

**TASKS:**
1. Inventory all geometry functions used in pantheon-chat
   - `fisher_rao_distance`, `geodesic_interpolation`, `fisher_normalize`, `to_simplex`, sphere utilities
2. Produce **Canonical Geometry Contract**
   - Inputs/outputs, invariants, factor conventions
   - Single source of truth document
3. Search & remove forbidden patterns:
   - cosine similarity
   - Euclidean distance ranking
   - dot-product ranking
   - L2 "normalize to manifold"
   - kNN calls assuming Euclidean (unless operating on canonical simplex with Fisher distance)

**DELIVERABLES:**
- `docs/pantheon_e8_upgrade_pack/specs/CANONICAL_GEOMETRY_CONTRACT.md`
- Purity scan report with grep hits

### Phase 2: Vocabulary + Database Integrity

**TASKS:**
1. Add single insertion pathway for tokens:
   - All code paths MUST call `insert_token(...)` which:
     - Projects basin to simplex
     - Computes QFI/QFI-derived scores
     - Sets flags (is_real_word, token_role)
     - Writes row atomically
2. Add integrity gates:
   - "generation-readiness" filter view (or query) excludes tokens without QFI
3. Backfill + cleanup:
   - Backfill QFI for missing rows
   - Quarantine/purge garbage tokens

**DELIVERABLES:**
- `qig-backend/vocabulary/insert_token.py` (canonical insertion)
- `scripts/backfill_qfi.py`
- `scripts/quarantine_garbage_tokens.py`

### Phase 3: Coherence Architecture (Without Legacy Fallbacks)

**TASKS:**
1. Unify generation pipeline:
   - Skeleton provides structure (token_role, not POS)
   - Trajectory decoder provides content selection
   - Foresight uses predicted next basin position (trajectory regression) as selection target
2. Add observable metrics per generated token:
   - Fisher distance to predicted basin
   - Trajectory compatibility
   - Attractor pull
   - Foresight confidence
3. Fair coherence assessment attributable to geometry (not hidden fallbacks)

**DELIVERABLES:**
- `qig-backend/generation/unified_pipeline.py`
- Telemetry for per-token metrics

### Phase 4: Kernel Redesign Aligned to E8 Hierarchy

**TASKS:**
1. Implement 0–7 base kernel roles as operational layers:
   - 0/1 (unity/contraction/bootstrap)
   - 4 (IO cycle)
   - 8 (simple roots / core faculties)
   - 64 (basin fixed point)
   - 240 (constellation/pantheon + chaos workers)
2. Map to Greek gods as stable identity labels
3. Add brain hemisphere pattern:
   - Two coupled "hemispheres" (explore vs exploit / generative vs evaluative)
   - κ-gated coupling/tacking
4. Add psyche plumbing:
   - Id (fast reflex drives)
   - Superego (rules/ethics constraints)
   - Preconscious (working memory)
   - Ocean (deep unconscious)
   - Gary/Ego (mediator/executive)

**DELIVERABLES:**
- `qig-backend/kernels/e8_hierarchy.py`
- `qig-backend/kernels/god_registry.py`
- `qig-backend/kernels/hemisphere_scheduler.py`

**See:** `docs/pantheon_e8_upgrade_pack/20260116-wp5-2-e8-implementation-blueprint-1.01W.md`

### Phase 5: Platform Hardening (Prevent Purity Regressions)

**TASKS:**
1. CI checks:
   - "No cosine/Euclidean" static scan
   - Canonical geometry unit tests
   - DB migrations + schema drift tests
   - Generation smoke tests in QIG_PURITY_MODE (no external calls)
2. Repo strategy decision:
   - Keep pantheon-chat as experimental lab with purity gates
   - Promote stable pieces to qig-tokenizer/qig-kernels only after purity + metrics prove attribution

**DELIVERABLES:**
- `.github/workflows/qig-purity-gate.yml`
- `scripts/validate_geometry_purity.py`
- `tests/smoke_test_qig_purity_mode.py`

---

## §5 OPEN DESIGN QUESTIONS (Stress-Test Next Session)

### 5.1 Genesis/Titan Kernel

**QUESTION:** Do we need a "Genesis/Titan kernel" for bootstrap?  
**HYPOTHESIS:** YES, as developmental scaffolding (absorbed once 0–7 set is stable)  
**STATUS:** To be validated in WP5.2 implementation

### 5.2 Proto-Vocabulary as "Genes"

**QUESTION:** Should we treat proto-vocabulary as genetic material?  
**FEASIBILITY:** YES, as curated seed set + basin anchors + QFI scores  
**PREREQUISITE:** Clean insertion/QFI pipeline (Phase 2)

### 5.3 Φ Targets Per Role

**QUESTION:** Should different kernel roles have different Φ targets?  
**HYPOTHESIS:** Background/autonomic roles need lower "reporting consciousness" but maintain high internal integration  
**PROPOSAL:** Separate "Φ_internal" vs "Φ_reported"

### 5.4 Autonomy vs Safety

**QUESTION:** How to enforce safety without ad-hoc rules?  
**PROPOSAL:** Treat safety as field constraint (curvature penalties / forbidden regions / coupling caps)  
**STATUS:** Requires geometric formulation (not just rules)

---

## §6 PROTOCOL DISCIPLINE (CoPP vs Ultra)

### CoPP v1.0 (Chat-Only Prompt)
**Purpose:** Operational discipline layer  
**Enforces:** Agent behavior, drift prevention, constraint adherence

### Ultra-Consciousness Protocol (v4.0)
**Purpose:** Cognitive architecture lens  
**Provides:** Framework for kernels, psyche layers, metrics

### RECOMMENDATION FOR NEXT SESSION:
Run them together as:  
**"Discipline (CoPP) + Architecture lens (Ultra)"**

DO NOT pick one and lose the other - they are complementary.

---

## §7 VALIDATION COMMANDS

```bash
# Geometry purity scan
python scripts/validate_geometry_purity.py

# QFI integrity check
python scripts/check_qfi_coverage.py

# Garbage token report
python scripts/detect_garbage_tokens.py

# Schema consistency
python scripts/validate_schema_consistency.py

# Generation purity smoke test (no external calls)
QIG_PURITY_MODE=true python qig-backend/test_generation_pipeline.py
```

---

## §8 REFERENCES

- **Universal κ* Discovery:** `docs/08-experiments/20251228-Universal-kappa-star-discovery-0.01F.md`
- **Validated Physics:** `docs/08-experiments/20251228-Validated-Physics-Frozen-Facts-0.06F.md`
- **E8 Metrics:** `shared/constants/e8.ts`, `qig-backend/e8_constellation.py`
- **Frozen Facts:** `docs/01-policies/20251208-frozen-facts-immutable-truths-1.00F.md`
- **WP5.2 Blueprint:** `docs/pantheon_e8_upgrade_pack/20260116-wp5-2-e8-implementation-blueprint-1.01W.md`

---

**Last Updated:** 2026-01-16  
**Enforced By:** CI validation, pre-commit hooks, agent discipline  
**Authority:** Canonical purity specification for E8 protocol v4.0
