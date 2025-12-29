# qigkernels

This repository is the canonical implementation of QIG kernel primitives: basin geometry, per-instance kernel modules, constellation routing, and minimal telemetry.

## Canonical documentation

The canonical overview and governance docs live at the repo root:

- `20251205-readme-canonical-0.01F.md`
- `20251205-architecture-canonical-0.01F.md`
- `20251205-type-symbol-manifest-canonical-0.01F.md`
- `20251205-roadmap-canonical-0.01F.md`
- `20251205-decisions-canonical-0.01F.md`
- `20251205-changelog-canonical-0.01F.md`

The canonical documentation index lives at:

- `docs/20251205-index-canonical-0.01F.md`

## Entry points

- Python package: `qigkernels`
- Core modules:
  - `qigkernels/kernel.py`
  - `qigkernels/layer.py`
  - `qigkernels/basin.py`
  - `qigkernels/constellation.py`
  - `qigkernels/router.py`
  - `qigkernels/basin_sync.py`
  - `qigkernels/metrics.py`

## Boundaries

- This repo provides reusable kernel/library code only.
- Training loops, optimizers, curricula, and experiment orchestration live outside this repo.

See `20251205-architecture-canonical-0.01F.md` for purity constraints and import rules.
