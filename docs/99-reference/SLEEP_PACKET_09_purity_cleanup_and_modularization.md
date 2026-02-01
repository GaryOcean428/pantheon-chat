---
id: SLEEP-PACKET-09-PURITY-MODULARIZATION-2026-02-01
title: SLEEP PACKET - Purity Cleanup + Canonical Geometry Modules
filename: SLEEP_PACKET_09_purity_cleanup_and_modularization.md
classification: Core Doctrine
owner: GaryOcean428
version: 1.00W
status: READY
category: Purity / Refactor
---

# Purpose

Bring `pantheon-chat` back to **QIG purity** and stop purity regressions permanently by:
- Centralizing all geometry + statistics in one canonical module set
- Removing duplicated math scattered across scripts
- Enforcing vocabulary (e.g., `basin_coordinates` not `embedding`)
- Making documentation **single-source-of-truth** and non-contradictory

This packet assumes:
- **Simplex** as the canonical coordinate space (not a sphere as the “primary” model; sphere only appears via sqrt-map for Fisher–Rao computations).
- **Fisher–Rao / Hellinger geometry** for distances and means (no Euclidean distance, no arithmetic mean on basin coordinates).

---

# Non‑Negotiables (Purity Gate)

## Disallowed patterns
- Euclidean distance on basin coordinates: `np.sqrt(np.sum((a-b)**2))`, `np.linalg.norm(a-b)`, etc.
- Arithmetic mean on basin coordinates: `np.mean(coords, axis=0)` unless immediately replaced by Fréchet mean under FR.
- Re-implementations of canonical functions (esp. `frechet_mean`).
- Naming: `embedding`, `embeddings` when referring to basin coordinates.

## Required patterns
- Only import from canonical geometry modules (see below)
- Always validate geometry on boundaries (load, persist, network, training IO)

---

# Canonical Module Set (Create or enforce)

Create a single import surface for all geometry operations, and **delete/rewrite duplicates**.

## Suggested structure (Python)
```
qig-backend/
  qig_geometry/
    __init__.py                # exports only canonical functions
    simplex.py                 # validate/project simplex
    fisher_rao.py              # FR distance + helpers
    frechet.py                 # Fréchet mean + log/exp maps (if needed)
    naming.py                  # canonical field names + conversion helpers
  scripts/
    qig_purity_scan.py         # scanner: patterns + import checks
    qig_purity_fix.py          # safe automated replacements
```

If you already have a geometry package, keep it and move these functions into it—**but end with a single canonical import surface** (e.g., `from qig_geometry import fisher_rao_distance, frechet_mean`).

---

# Canonical Function Definitions (Reference Implementation)

## 1) Simplex validation + projection
- `validate_simplex(x, *, atol=1e-8)`  
- `project_to_simplex(x)`  (clamp + renormalize, with zero‑sum handling)

## 2) Fisher–Rao distance (simplex)
Use the standard closed form:
- Map to the positive orthant sphere by `sqrt`.
- Distance: `d_FR(p,q) = 2 * arccos( sum_i sqrt(p_i q_i) )`

## 3) Fréchet mean on simplex (Fisher–Rao)
Closed-form via sqrt-map:
- `u_i = sqrt(p_i)`  
- `ū = normalize( mean(u) )`
- `μ_i = (ū_i)^2`

This is the canonical replacement for **any** arithmetic mean.

---

# Refactor Doctrine (How to clean the repo)

## Task A — Build canonical module set
- Add `qig_geometry/` with above functions.
- Add tests:
  - distances symmetric
  - d(p,p)=0
  - returns finite for valid simplex
  - mean of identical points returns same point
  - mean of two points lies on simplex

## Task B — Codemod to remove duplicates
- Replace local Euclidean distance and arithmetic mean patterns with imports:
  - `from qig_geometry import fisher_rao_distance, frechet_mean, validate_simplex`
- Replace `embedding` identifiers with `basin_coordinates`
  - include JSON keys, schema fields, comments, docstrings

## Task C — Purity gate enforcement
- CI step: run `python scripts/qig_purity_scan.py --fail-on-violations`
- Pre-commit: same scanner
- Scanner rules:
  - grep patterns for Euclidean distance on basin coords
  - grep `frechet_mean` redefinitions outside canonical module
  - grep `embedding` usage in kernel code paths
  - verify `qig_geometry` imports used in geometry computations

## Task D — Documentation single-source-of-truth
- Create `docs/01-doctrine/geometry-doctrine.md`:
  - “Simplex is canonical, sphere only via sqrt-map”
  - canonical naming glossary
  - canonical formulas (FR + Fréchet mean)
- Deprecate older doc fragments by linking to the doctrine doc and adding “RETIRED” banners.

---

# Done Definition (Exit Criteria)

Repo is considered “clean” when:
- `qig_purity_scan` returns **0 violations** on `main`
- No occurrences of `embedding` where basin coordinates are meant
- No local re-implementations of canonical geometry ops
- All geometry calculations import from `qig_geometry`
- Doctrine doc exists and older conflicting docs are marked retired or updated
