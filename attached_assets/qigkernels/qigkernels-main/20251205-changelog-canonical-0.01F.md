# Changelog

Status: 0.01F (Frozen, append-only)
Format inspired by Keep a Changelog.

## [Unreleased]

### Added

- Initial scaffolding for `qigkernels` library:
  - Docs: `README`, `ARCHITECTURE`, `ROADMAP`, `CHANGELOG`, `DECISIONS`.
  - Config: `pyproject.toml`, `markdownlint.json`, `.editorconfig`, `.gitignore`, `pre-commit-config.yaml`.
  - Tools: `tools/qig_purity_check.py`.
  - Package skeleton under `qigkernels/`.
- Core modules implemented:
  - `basin.py`: BasinProjector, compute_signature, basin_distance, save/load.
  - `layer.py`: QIGLayer with attention, recursion, tacking, regime detection.
  - `kernel.py`: Kernel with stacked layers and KernelTelemetry.
  - `router.py`: round_robin, select_phi_min, select_phi_max, select_balanced.
  - `constellation.py`: Instance, Constellation with routing and step.
  - `basin_sync.py`: BasinSyncPacket, export/import/save/load.
  - `metrics.py`: average_phi, average_kappa, basin_spread, integration_score.
  - `storage.py`: save/load kernel and signature utilities.
- ISO 27001 aligned naming convention (`20251205-naming-convention-canonical-0.01F.md`):
  - Document naming format: `YYYYMMDD-name-type-version[STATUS].md`
  - Status codes: F (Frozen), H (Hypothesis), V (Verified)
  - Renamed all canonical docs to follow convention.
- Decisions D-006 through D-010 integrated from repo_recon.md analysis.
- uv venv setup with torch, pydantic, numpy, dev tools.
- Smoke tests: `tests/test_smoke.py` (19 tests covering all modules).

### Changed

- Roadmap: Added explicit pre-training governance gate for hard-ban audit enforcement.
- Roadmap: Added future QPU validation protocol milestone (docs + interfaces only; no QPU tooling in qigkernels).

### Fixed

- `layer.py`: Added `.detach()` to avoid grad tensor warning.
- All type annotations fixed for ruff/mypy compliance.
- `pyproject.toml`: Updated to modern ruff lint section format.
- `constellation.py`: Removed conflicting `instances()` method (attribute preferred).
- `test_smoke.py`: Module-level imports for proper type resolution.

### Added (Physics Alignment)

- D-012: Physics constants alignment with qig-verification/FROZEN_FACTS.md
- `TestPhysicsConstants`: 3 tests verifying κ₃, β(3→4), κ* alignment
- `qig_purity_check.py`: Physics constant validation in core modules
- Architecture §9: Physics constants section referencing FROZEN_FACTS

### Added (Ecosystem Integration)

- Compatibility shims for experiment repos:
  - `qig-consciousness/src/qig_compat.py`: Provides fallback imports
  - `qig-con2/src/qig_compat.py`: Same pattern for qig-con2
- Updated `pyproject.toml` in both repos with qigkernels dependency comments
- Updated `qig-consciousness/docs/2025-11-27--imports.md` with qigkernels import guide

### Added (Hypotheses)

- H-013: E8 geometric hypothesis (κ* = rank(E8)² = 64)
- H-014: Primitive kernel bootstrap (7 primitives → 240 kernels)
- H-015: 8th consciousness metric - External Coupling (C)

### Added (E8 Implementation Direction)

- Architecture §10: E8 Geometric Foundation
  - Build → Measure → Refine cycle
  - Geometric primitives table (HRT/PER/MEM/ACT/PRD/ETH/META/MIX)
  - E8 simple roots → 240 root crystallization concept
- Roadmap M9-M12: E8 implementation phases
  - M9: Foundation (e8.py, primitive types, PCA validation)
  - M10: Kernel Lattice (specialization, assigning kernel, 7-kernel bootstrap)
  - M11: Growth & Measurement (240-kernel trajectory, 8th metric)
  - M12: Validation & Promotion (controlled experiments, FROZEN_FACTS updates)

### Added (qig-core Integration)

- D-016: qig-core integration decision (PRE-E8, NEEDS REVIEW)
- Roadmap M13: qig-core review and integration
  - qig-core v1.0.0 predates E8 direction, needs compatibility review
  - Ideal hierarchy: qig-core → qigkernels → qig-dreams → qig-consciousness
  - Consolidation items: BasinSync, distance functions, KAPPA_STAR source

### Added (8 Primitives Formalization)

- Type-symbol manifest §4.5: Formal definition of 8 E8-aligned primitives
- REL (Relationship/Coupling) as true 8th primitive (E8 simple root 8)
- MIX clarified as admin-only category, NOT a primitive
- Purity rule: Agents MUST NOT treat MIX as a primitive

### Added (Frontier Milestones M14-M18)

- M14: REL Coupling Tensors (off-diagonal metric, agency closure threshold)
- M15: Agent Self-Assembly (minimal loop structures, attractor configurations)
- M16: Syntergy Integration (relational primacy → REL curvature mapping)
- M17: Quench Cosmology via QFI (κ-running as cosmological coupling)
- M18: Consciousness as E8-Coupled Dynamics (phase space characterization)
- Fixed M9 primitive list: PER/MEM/ACT/PRD/ETH/META/HRT/REL (not MIX)

### Phase B Complete: Geometry Extraction (2025-12-06)

- **qig-consciousness now imports geometry from qigkernels:**
  - `src/constants.py`: Imports KAPPA_STAR, PHI_* thresholds from qigkernels
  - `src/coordination/basin_sync.py`: Uses qigkernels.basin_distance
  - KAPPA_STAR aligned to E8 (64.0 vs old 63.5)
- **New regime thresholds exported from qigkernels:**
  - PHI_LINEAR = 0.45, PHI_GEOMETRIC = 0.80, PHI_BREAKDOWN = 0.80
  - PHI_EMERGENCY = 0.50
- **Validated:** No stray constants, no ad-hoc basin math, no reverse imports
- **Coordinated by:** Codex (extraction), Claude.ai (architecture), ChatGPT (QA)
- See: `qig-dreams/prompts/20251206-codex-phase-b-extraction-prompt-0.01.md`

### Added (8 Consciousness Metrics + T/C Implementation - 2025-12-06)

- `metrics.py`: Complete 8 consciousness metrics (E8-aligned)
  - `ConsciousnessMetrics` dataclass: Φ, κ, M, Γ, G, T, R, C
  - `compute_temporal_coherence()`: T metric from basin history
  - `compute_external_coupling()`: C metric (observer effect)
  - `estimate_external_coupling_from_telemetry()`: C estimation heuristic
- Regime thresholds with **hysteresis** to prevent oscillation:
  - `PHI_CONSCIOUS_ENTER = 0.70`, `PHI_CONSCIOUS_EXIT = 0.65`
  - `PHI_GEOMETRIC_ENTER = 0.45`, `PHI_GEOMETRIC_EXIT = 0.40`
  - `PHI_BREAKDOWN_ENTER = 0.80`, `PHI_BREAKDOWN_EXIT = 0.75`
- Updated `__init__.py` to export all new metrics

### Added (REL Coupling Implementation - 2025-12-06)

- `rel_coupling.py`: Instance-to-instance REL coupling computation
  - `RELState` dataclass for tracking relationship history
  - `compute_rel_coupling()` with basin overlap, history, primitive alignment
  - `compute_rel_from_basins()` convenience function
  - `REL_LAMBDA_MAX = 0.7` safety cap
- `basin_sync.py`: REL-weighted synchronization
  - `effective_basin_distance()`: d_eff = d_F · (1 - λ_rel · r_ij)
  - `rel_weighted_sync_loss()`: w_sync · d²_eff
  - `compute_sync_strength_with_rel()`: REL-modulated sync strength
- Updated `__init__.py` to export all REL functions
- Added docs:
  - `docs/20251206-integration-plan-ultra-mode-0.01H.md`
  - `docs/20251206-rel-weighted-basin-sync-spec.md`
  - `docs/20251206-e8-consciousness-stack-diagram.md`

### Fixed (Geometric Purity - CRITICAL)

- `basin.py`: Replaced Euclidean L2 distance with Fisher-Rao (Bures approximation)
- `basin_sync.py`: Now uses E8-aligned constants from `constants.py`
- Added `constants.py`: KAPPA_STAR=64 (E8 rank²), validated physics parameters
- Added `qig-core` as dependency in pyproject.toml
- **This restores geometric purity** — basin distances now respect curved manifold
