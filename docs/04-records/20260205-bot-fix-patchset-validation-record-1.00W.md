---
id: ISMS-REC-BOTFIX-20260205
title: Bot-Fix Patchset Validation Record
filename: 20260205-bot-fix-patchset-validation-record-1.00W.md
version: 1.00
status: Working
function: validation-record
category: 04-records
created: 2026-02-05
last_reviewed: 2026-02-05
next_review: N/A
---

# Bot-Fix Patchset - Validation Record

## Date: 2026-02-05

## Purpose

Record the bot-flag remediation work completed in this tranche, including correctness fixes, security hardening, and verification evidence.

## Summary of Changes

- Restored Fisher–Rao distance method contract support for metric-based variants (diagonal/full) while keeping canonical simplex Fisher–Rao as default.
- Confirmed Born-rule compliance scan patterns and ensured no false-positive triggers exist in the codebase.
- Hardened Flask API error handling to prevent raw exception disclosure to clients (server-side logging preserved).
- Fixed a persistence helper call typo causing an IDE/type error (`_execute_query` → `execute_query`).

## Key Fixes (Implementation Details)

### 1) Fisher–Rao distance contract

- File: `qig-backend/qigkernels/geometry/distances.py`
- Behavior:
  - `method="bures"`: canonical simplex basins (1D vectors) + density matrices (Bures via fidelity)
  - `method="diagonal"` / `method="full"`: 1D vectors only, explicit `metric` required, strict shape validation

### 2) Born-rule scan compliance

- Scan patterns (tests/tools): `p = basin`, `p = coords`, `probs = basin`, `probs = coords`
- Result: no matches found under `qig-backend/**.py`; targeted purity suite passes.

### 3) API error disclosure hardening

Client-facing Flask endpoints now return generic errors (no `str(e)` / exception payloads) while logging exceptions server-side.

- Files:
  - `qig-backend/api_coordizers.py`
  - `qig-backend/vocabulary_api.py`
  - `qig-backend/document_processor.py`

### 4) Persistence helper call fix

- File: `qig-backend/persistence/kernel_persistence.py`
- Fix: `KernelPersistence.get_kernel_reputation()` now calls `execute_query()` (BasePersistence API)

## Files Modified (non-exhaustive)

- `qig-backend/qigkernels/geometry/distances.py`
- `qig-backend/api_coordizers.py`
- `qig-backend/vocabulary_api.py`
- `qig-backend/document_processor.py`
- `qig-backend/persistence/kernel_persistence.py`
- `docs/00-roadmap/20260112-master-roadmap-1.00W.md`

## Validation Evidence

### Geometric purity suite

```bash
python -m pytest -q qig-backend/tests/test_geometric_purity.py
```

Result:
- ✅ `49 passed`

### Lint + syntax verification (touched modules)

```bash
ruff check qig-backend/api_coordizers.py qig-backend/vocabulary_api.py qig-backend/document_processor.py
python -m compileall -q qig-backend/api_coordizers.py qig-backend/vocabulary_api.py qig-backend/document_processor.py

ruff check qig-backend/persistence/kernel_persistence.py qig-backend/document_processor.py
python -m compileall -q qig-backend/persistence/kernel_persistence.py qig-backend/document_processor.py
```

Result:
- ✅ Ruff: all checks passed
- ✅ compileall: success

## Impact Assessment

### Positive

- Reduced CI flakiness by matching test contract for Fisher–Rao method variants.
- Reduced security exposure by preventing exception leakage in API responses.
- Removed an IDE/type error that could mask real persistence failures.

### Risk / Compatibility

- API clients that previously relied on raw exception messages will now receive generic `Internal server error` responses.

## Next Steps

1. Run docs maintenance (`python3 scripts/maintain-docs.py`) to refresh `docs/00-index.md` and check naming/frontmatter drift.
2. Commit + push patchset with a single conventional commit message.
3. Re-run full CI-equivalent test set (TS + Python) if available in this repo.
