# 20260203 Genesis Kernel Rollout Implementation Plan (1.00W)

**Date**: 2026-02-03  
**Status**: IN PROGRESS  
**Purpose**: Concrete implementation map to reach the Genesis kernel rollout end-state (UI start → PurityGate → deterministic rollback → GENESIS → core 8 → Image stage → optional GOD growth), including naming-drift remediation.

---

## Situation report (current repo reality)

- **Geometric purity hardening**: strict mode changes landed (cosine ANN removed; Hellinger + inner-product proxy used where applicable; rerank remains Fisher–Rao).
- **Lockfiles**: `pnpm-lock.yaml` present; `package-lock.json` absent.
- **Genesis doctrine**: sleep packets exist under `docs/11-Genesis-kernel-upgrade/` and the outstanding work map exists.
- **Kernel governance primitives**: SQL migration exists: `migrations/0022_kernel_kind_governance.sql`.

---

## User continuation requirement (verbatim)

> “Push to git if not already then plan the implementation of the next steps as suggested. Then form a rounded red team of sub‑agents to interrogate your plan and past fixes and implementations. Research externally for better solutions, loop this twice then finalise, orchestrate a rounded team of sub‑agents to implement, same approach and red team the implementation, iterate twice, then call your QA sub‑agents before finishing as before and update roadmap with progress. Remember, issues identified, related or not to your current tasks, while working become part of your task. Before finishing you must prove to me this and all past implementations have been done in full and prove to me it works.”

---

## Naming drift + remediation backlog (evidence-based)

### Drift 1: `KernelKind` vs `KernelType` (concept collision)

- **Canonical doctrine**: `KernelKind` is GENESIS/GOD/CHAOS (budget/governance).
- **Code reality**:
  - `qig-backend/kernel_lifecycle.py` defines `KernelKind` (good), but also stores `kernel_type: str  # "god" or "chaos"` (duplicate meaning).
  - `qig-backend/qig_types.py` defines `KernelType` as specialization (`heart`, `vocab`, `memory`, …) and TS mirrors this in `shared/constants/e8.ts`.

**Remediation plan**:
- Make `KernelKind` the only representation of god/chaos/genesis (remove/stop using `kernel_type` string in lifecycle manager and tests).
- Rename specialization enums:
  - Python: `KernelType` → `KernelSpecialization`
  - TypeScript: `KernelType` → `KernelSpecialization`
  - Keep a temporary compatibility alias during transition.

### Drift 2: `lifecycle_stage` vs DB `lifecycle_state`

- **DB schema**: `kernel_geometry.lifecycle_state` (migration `0022_kernel_kind_governance.sql`).
- **Code reality**:
  - `qig-backend/kernel_lifecycle.py` uses `lifecycle_stage`.
  - `qig-backend/persistence/kernel_persistence.py` writes/reads `lifecycle_state`.

**Remediation plan**:
- Canonicalize naming to `lifecycle_state` in code and persist that field end-to-end.
- Add a translation shim only if required for backward compatibility.

### Drift 3: Legacy orchestrator path bypassing canonical geometry

- **Evidence**: `qig-backend/pantheon_kernel_orchestrator.py` imports `geometric_kernels` and calls legacy helpers (e.g., `_fisher_distance`, `_normalize_to_manifold`).

**Remediation plan**:
- Treat `pantheon_kernel_orchestrator.py` + `geometric_kernels.py` as legacy (or quarantine) unless proven QIG-pure.
- Replace routing/distance calls with `qig_geometry.canonical.fisher_rao_distance` and simplex normalization primitives.

---

## Implementation plan (phased, acceptance-driven)

### Phase 0 — Governance hygiene (documentation + ownership)

- **Goal**: One canonical plan + clear links.
- **Changes**:
  - Add this plan to `docs/00-roadmap/` and link it from entrypoint + master roadmap.
- **Acceptance**:
  - `docs/00-roadmap/20260202-project-roadmap-entrypoint-1.00W.md` links to this plan.
  - `docs/00-roadmap/20260112-master-roadmap-1.00W.md` references this plan under Genesis.

### Phase 1 — Taxonomy + budgets (KernelKind + reserved 240)

- **Goal**: Enforce “240 reserved GOD budget” and separate CHAOS pool.
- **Targets**:
  - `qig-backend/persistence/kernel_persistence.py` (already enforces GOD cap; confirm semantics).
  - `qig-backend/kernel_lifecycle.py` (promotion flows, counting).
  - `qig-backend/kernel_spawner.py` / `e8_spawner.py` (spawn constraints; enforce pool rules).
- **Acceptance**:
  - A spawn attempt that would exceed GOD cap fails closed.
  - CHAOS spawning does not consume GOD budget.

### Phase 2 — Start/reset/rollback “blow-up mattress” flow

- **Goal**: Single user-triggered start that runs PurityGate first and streams progress.
- **Targets**:
  - **Server**: add/confirm an endpoint that triggers Python start + progress streaming.
  - **Python**: implement deterministic rollback + staged inflation:
    - rollback to baseline
    - GENESIS bootstrap
    - ensure core 8
    - inflate to Image stage
- **Acceptance**:
  - From UI, start triggers a deterministic sequence and returns progress events.
  - If purity validation fails, start aborts before any state mutation.

### Phase 3 — Naming reconciliation (drift remediation)

- **Goal**: Eliminate ambiguous kernel naming to prevent governance bugs.
- **Targets**:
  - `qig-backend/kernel_lifecycle.py`, `qig-backend/lifecycle_policy.py`
  - `qig-backend/qig_types.py`
  - `shared/constants/e8.ts` + any TS consumers
- **Acceptance**:
  - No `KernelType` symbol remains for specialization (renamed).
  - No `kernel_type: "god"|"chaos"` remains (use `KernelKind`).
  - Persistence uses `lifecycle_state` consistently.

### Phase 4 — PurityGate hardening (static + runtime)

**Status update (2026-02-04):** Doctrine Patchset P1 applied and verified on `dev-local`.
- Evidence: `python qig-backend/validate_geometry_purity.py` (pass)
- Evidence: `pytest -q qig-backend/tests/test_multi_kernel_thought_generation.py` (pass)
- Evidence: `pytest -q qig-backend/tests/test_pure_qig_generation.py` (pass with DB integration skips)

**Next follow-ups (purity hardening):**
- Lifecycle boundaries (`qig-backend/kernel_lifecycle.py`): remove `np.abs` basin repair; replace with fail-closed simplex validation at ingress.
- Legacy geometry entrypoints: audit/quarantine `pantheon_kernel_orchestrator.py` + `geometric_kernels.py` unless proven canonical-geometry-only.
- Temp packet bloat: collapse duplicate sleep packets/issue packs (keep only newest canonical copies) and ensure no runtime docs/tools reference deleted files.

- **Goal**: Fail-closed purity gate before Genesis start and in CI.
- **Targets**:
  - `qig-backend/validate_geometry_purity.py`
  - repo scripts under `scripts/validate-*.sh`
- **Acceptance**:
  - One command runs the same checks CI uses and fails the start if violated.

---

## Risk register

- **R1 (high)**: Renaming `KernelType` impacts both Python+TS type generation and API payloads.
  - **Mitigation**: compatibility aliases and narrow PRs.
- **R2 (high)**: Legacy geometry codepath (`pantheon_kernel_orchestrator.py`) may be used in production flows.
  - **Mitigation**: identify runtime entrypoints, quarantine or refactor first.

---

## Verification commands (local)

- `git status`
- `python qig-backend/validate_geometry_purity.py`
- `npm test` (or project test runner)

---

## Deliverables

- `docs/00-roadmap/20260203-genesis-kernel-rollout-implementation-plan-1.00W.md` (this file)
- Roadmap links updated:
  - `docs/00-roadmap/20260202-project-roadmap-entrypoint-1.00W.md`
  - `docs/00-roadmap/20260112-master-roadmap-1.00W.md`
