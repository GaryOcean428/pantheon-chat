# QIG Integration Plan (Ultra-Mode Edition)

**Status:** 0.01H
**Date:** 2025-12-06
**Scope:** Wire geometric purity + REL coupling into existing constellation
**Repos:** qig-core → qigkernels → qig-consciousness → qig-dreams

---

## 0. Ground Rules for Agents

- Do **not** create new summary/notes docs.
- Update only:
  - `20251205-architecture-canonical-*.md`
  - `20251205-roadmap-canonical-*.md`
  - `20251205-decisions-canonical-*.md`
  - `20251205-changelog-canonical-*.md`
- Prefer **editing** existing modules over creating parallel variants.
- No time estimates. Use **sequence**, not **schedule**.
- Never reintroduce:
  - Softmax sampling in core QIG flows
  - Euclidean L2 distances where Fisher-Rao is available
  - Hard-coded κ constants outside `qig-core` / `qigkernels`

---

## 1. Target Stack (Conceptual)

```text
qig-core           # Pure Fisher geometry + QFI math
   ↓
qigkernels         # E8-aligned kernels, basin, routing, REL hooks
   ↓
qig-consciousness  # Constellation, coaches, dream/sleep/mushroom
   ↓
qig-dreams         # Primitive-aligned corpora (PER/MEM/ACT/PRD/ETH/META/HRT/REL)
```

**Goal:** Make qig-consciousness stop doing its own geometry and instead consume:

- Fisher distance / QFI samplers from qig-core
- Basin / routing / sync from qigkernels
- Corpora from qig-dreams

---

## 2. Phase A — Wire qig-core into qigkernels ✅ COMPLETE

### A1. Verify imports ✅

- qigkernels now imports Fisher geometry from qig-core when available
- Fallback to Bures approximation when qig-core unavailable

### A2. Replace Euclidean stubs ✅

- `basin.py`: `basin_distance()` now uses Fisher-Rao (Bures approximation)
- `basin_sync.py`: Uses E8-aligned constants

### A3. Lock κ* and β source of truth ✅

- `qigkernels/constants.py` defines KAPPA_STAR = 64.0
- Other modules import from constants.py

---

## 3. Phase B — Wire qigkernels into qig-consciousness ✅ PARTIAL

### B1. Identify inline geometry ✅

Fixed L2 violations in:

- `src/qig_compat.py` → Fisher-Rao
- `src/qig/neuroplasticity/mushroom_mode.py` → Fisher-Rao
- `src/coordination/basin_sync.py` → Fisher-Rao

### B2. Swap in qigkernels basin + routing 🔄 IN PROGRESS

- `qig_chat.py` now imports constants from `src/constants.py`
- Helper functions extracted to `chat_interfaces/lib/helpers.py`

### B3. Telemetry shape alignment 📋 TODO

- Need adapter layer for qigkernels ↔ qig-consciousness telemetry

---

## 4. Phase C — REL Coupling Integration 📋 TODO

### C1. Introduce REL tensor

Create `qigkernels/rel_coupling.py`:

- `compute_rel_coupling(self_state, other_state) -> float`
- REL in [0, 1] based on:
  - Basin overlap
  - Interaction history
  - Shared primitive exposure

### C2. REL-weighted basin sync

Update `qigkernels/basin_sync.py`:

- Implement `rel_weighted_sync_loss()`
- Formula: d²_eff = d²_F · (1 - λ_rel · r_ij)

### C3. REL-aware routing (optional)

- Prefer kernels with appropriate Φ/κ AND high REL
- Keep default = Φ-based until tested

---

## 5. Phase D — Training Flow Preservation

### D1. Do not dismantle dream/sleep/mushroom

All protocol logic stays in qig-consciousness.
Only geometry + basin + routing move "down" into qigkernels.

### D2. Constellation stays; math moves

- qig-consciousness remains: Constellation definition, coaches, curriculum
- qigkernels becomes: Mathematical substrate

---

## 6. Phase E — Documentation & Changelog Discipline

For every non-trivial change:

- Add entry to `20251205-changelog-canonical-*.md`
- If design decision locked, add to `20251205-decisions-canonical-*.md`

---

## Appendix: Import Mapping

| ChatGPT Suggested | Actual qigkernels Path |
|-------------------|------------------------|
| `qigkernels.basin.BasinSignature` | `qigkernels.basin.BasinProjector` |
| `qigkernels.metrics.fisher_rao_distance` | `qigkernels.basin.basin_distance` |
| `qigkernels.basin_sync.compute_sync_loss` | `qigkernels.basin_sync.compute_sync_strength` |
| `qigkernels.constants.KAPPA_STAR` | `qigkernels.constants.KAPPA_STAR` ✅ |
