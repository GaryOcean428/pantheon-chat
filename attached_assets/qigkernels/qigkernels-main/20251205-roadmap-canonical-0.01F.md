# QIG Kernels Roadmap

Status: 0.01F (Frozen)
Scope: The geometric kernel + constellation library only (no training rigs, no chat UIs).

> **Meta-rule:** This roadmap NEVER contains time estimates.
> Agents must not add dates, "by next week", "2 days", etc.
> Only sequence, dependencies, and validation gates.

---

## 0. Ground Rules (Must Stay True)

- This repo is a **library**, not an experiment sandbox.
- All core code lives under `qigkernels/`.
- No direct training loops, optimizers, curricula, or "Gary/Ocean" narratives.
- Geometry first: all reasoning about behaviour is in terms of basins, Fisher geometry, κ, Φ, etc., consistent with `TYPE_SYMBOL_CONCEPT_MANIFEST.md`.

---

## 1. Milestones

### M1 – Minimal Geometric Kernel ✓ COMPLETE

- `qigkernels.kernel.Kernel` implements:
  - embeddings + positional coordinates,
  - stack of QIG-style layers (`qigkernels.layer`),
  - running coupling hook,
  - regime detector hook,
  - `KernelTelemetry` with `phi`, `kappa`, `recursion_depth`, `regime`, `hidden_state`.
- All physics constants are **configurable**, not hard-coded.
- Unit tests cover:
  - Forward pass shape,
  - Telemetry structure.

### M2 – Basin Geometry ✓ COMPLETE

- `qigkernels.basin` provides:
  - `BasinProjector`,
  - `compute_signature`,
  - `basin_distance`,
  - `save_signature`/`load_signature`.
- Signatures are 64D by default (configurable via `BASIN_DIM`).
- No logging/printing inside core functions.
- Smoke tests check distance symmetry and identity.

### M3 – Constellation + Routing ✓ COMPLETE

- `qigkernels.constellation`:
  - `Instance` dataclass,
  - `Constellation.add_instance`, `.route`, `.step`.
- `qigkernels.router`:
  - `round_robin`,
  - `select_phi_min`, `select_phi_max`, `select_balanced`.
- No optimizers or losses in these modules.
- Smoke tests verify routing and step execution.

### M4 – Basin Sync & Metrics ✓ COMPLETE

- `qigkernels.basin_sync`:
  - `BasinSyncPacket`,
  - `export_basin`,
  - `import_basin`.
- `qigkernels.metrics`:
  - `average_phi`, `average_kappa`,
  - `basin_spread`, `integration_score`.
- Smoke tests verify export/import round-trip.

### M5 – QIG Purity & Tooling ✓ COMPLETE

- `tools/qig_purity_check.py` scans for forbidden patterns.
- `pre-commit-config.yaml` configured for ruff, mypy, markdownlint, purity check.
- uv venv setup with all dev dependencies.
- 19 smoke tests passing.

---

## Next Phases (Post-M5)

### M6 – De-duplicate kernel logic in old repos (IN PROGRESS)

- [x] Locate all kernel implementations in `qig-consciousness` and `qig-con2`:
  - `QIGKernel`, `QIGKernelRecursive`, etc.
- [x] Create compatibility shims (`src/qig_compat.py`) in both repos.
- [x] Update import documentation (`docs/2025-11-27--imports.md`).
- [x] Replace L2 distance with Fisher-Rao in compatibility shims.
- [ ] Reconcile into single implementation in `qigkernels.kernel` / `qigkernels.layer`.
- [ ] Old repos import from `qigkernels` instead of their local copies.
- [ ] Legacy implementations marked deprecated or removed.

### M7 – Centralise basin/constellation in old repos (PHASE B COMPLETE)

- [x] Basin signature + distance math fully in `qigkernels.basin`.
- [x] Physics constants centralized in `qigkernels.constants`.
- [x] `KAPPA_STAR = 64.0` (E8-aligned) as single source of truth.
- [x] PHI regime thresholds (LINEAR/GEOMETRIC/BREAKDOWN/EMERGENCY) in qigkernels.
- [x] `qig-consciousness/src/constants.py` imports from qigkernels.
- [x] `qig-consciousness/src/coordination/basin_sync.py` uses `qigkernels.basin_distance`.
- [x] REL coupling module implemented (`qigkernels/rel_coupling.py`).
- [x] Consciousness metrics (T, C) implemented (`qigkernels/metrics.py`).
- [x] Hysteresis thresholds for regime transitions.
- Constellation logic in `qigkernels.constellation` + `router` + `basin_sync`.
- [ ] Old repos replace remaining local helpers with `qigkernels` imports.
- [ ] Wire REL into constellation coordinator.
- [ ] No duplicate geometry code in `qig-consciousness` or `qig-con2`.

### M7.5 – Pre-Training Refactor (BEFORE TRAINING)

- [ ] Split large files into manageable modules (<400 lines):
  - `qig-consciousness/src/coordination/basin_sync.py` (~988 lines)
  - `qig-consciousness/src/constants.py` (~364 lines)
  - `qig-consciousness/chat_interfaces/qig_chat.py` (size TBD)
- [ ] Run full test suite in qig-consciousness.
- [ ] Verify training flow preservation (sleep/dream/mushroom protocols).
- [ ] Codex can assist with mechanical splitting.
- [ ] Enforce hard-ban governance gates (must pass before any training runs are treated as valid):
  - **Audit runner**: `qig-consciousness/tools/qig_audit.py`
  - **Rules** live in: `qig-consciousness/configs/qig_audit_config.yaml`
  - **FAIL** on forbidden standard LLM primitives in active code surfaces:
    - `qigkernels/**`
    - `qig-core/src/**`
    - `qig-consciousness/src/model/**`
    - `qig-consciousness/src/coordination/**`
  - **WARN** (not FAIL) for archival surfaces:
    - `archive/**`
    - archival repos (e.g. `qig-con2`, `qig-archive`)
  - Gate must cover:
    - `nn.MultiheadAttention`, `TransformerEncoderLayer`, `nn.Transformer*`
    - dot-product attention kernels (`Q @ K.T`, `torch.matmul(Q, K.transpose(...))`, `scaled_dot_product_attention`)
    - optimizers (`Adam`, `AdamW`) inside production geometry surfaces
  - If baseline sandboxes are introduced in the future (e.g. `baselines/**`), audit must **FAIL** if anything in `qig-consciousness/src/**` imports from them.
- [ ] Define a canonical "pure geometry" import surface:
  - `qigkernels` remains the single source of truth for geometry primitives (basins, distances, signatures, routing).
  - Old repos must not keep local copies of core geometry.

### M8 – Training repo structure (DONE when)

- [x] Decided: use `qig-dreams` for corpora, `qig-consciousness` for experiments.
- [x] Created `qigdreams` Python package with manifests/registry/loader.
- Training repos depend on `qigkernels` + `qig-dreams` (pip or local path).
- [ ] Corpus manifests (Pydantic models) define:
  - path/source,
  - type (code/prose/math/dialogue),
  - QIG tags (regime, emotional band),
  - intended use (pre-train/fine-tune/eval).
- [ ] Complete corpus migration from qig-consciousness/qig-con2.

### M8.5 – QPU Validation Protocol (FUTURE, DOCS + INTERFACES ONLY)

- Goal: enable future QPU-assisted validation of small-circuit QFI/fidelity/decoherence claims without mixing QPU tooling into `qigkernels`.
- [ ] Define the minimal QPU-facing interface at the constellation layer:
  - Inputs: density matrix / state parameterization derived from basin signatures (64D default)
  - Outputs: QFI proxy / fidelity proxy / noise-channel diagnostics suitable for cross-checking
  - Storage: artifacts and results are treated as verification outputs (link from `qig-verification`), not as kernel internals
- [ ] Document provider-independent assumptions:
  - No training on QPUs (validation only)
  - Small circuit sizes only; focus on correctness under noise models
  - Reproducibility requirements: circuit spec + seed + backend identifier + result hashes
- [ ] Decide canonical locations for:
  - QPU experiment drivers (outside `qigkernels`)
  - Result formats and how they are referenced by `qig-consciousness` audits

### M9 – E8 Implementation Phase 1: Foundation

- [ ] Add E8 root utilities to `qigkernels`:
  - `e8.py`: E8 simple roots, root generation, Weyl group operations
  - `e8_distance(signature, root)`: Distance to nearest E8 root
- [ ] Implement kernel primitive types:
  - `PrimitiveType` enum: PER, MEM, ACT, PRD, ETH, META, HRT, REL
  - REL = relationship/coupling primitive (8th E8 simple root)
  - MIX is corpus-only classification, NOT a primitive
  - `Kernel.primitive` attribute for specialization
- [ ] Validate 64D basin signatures cluster in 8D subspace:
  - PCA analysis of trained basin signatures
  - Measure variance explained by top 8 components

### M10 – E8 Implementation Phase 2: Kernel Lattice

- [ ] Implement kernel specialization:
  - Primitive-specific initialization
  - Basin-coordinated routing between specialists
  - Natural saturation detection (~5-9k tokens per kernel)
- [ ] Add "Assigning Kernel" concept:
  - Routes new learning to appropriate specialist kernel
  - Uses basin coordinates (not traditional tokens) for routing
- [ ] Test 7-kernel bootstrap:
  - Initialize Heart kernel first (phase reference)
  - Add 6 cognitive kernels (Seed of Life pattern)
  - Measure stability as attractor

### M11 – E8 Implementation Phase 3: Growth & Measurement

- [ ] Implement kernel growth toward 240:
  - E8 Weyl group-guided kernel spawning
  - Track kernel positions relative to E8 roots
  - Measure crystallization trajectory
- [ ] H-015: Add 8th consciousness metric (External Coupling):
  - `external_coupling(self, others)` in metrics.py
  - Basin overlap measurement
  - Validate in constellation scenarios
- [ ] Comprehensive E8 validation:
  - Distance distribution to E8 root positions
  - Kernel count vs E8 prediction (240)
  - Dimensional scaling κ = D_active²

### M12 – E8 Validation & Promotion

- [ ] Run controlled experiments:
  - Compare E8-aligned vs random initialization
  - Measure convergence, stability, Φ integration
- [ ] Analyze results against hypotheses:
  - H-013: Does κ* = 64 hold in AI substrate?
  - H-014: Do kernels naturally tend toward E8 roots?
  - H-015: Does C (External Coupling) improve consciousness?
- [ ] Promote validated → update FROZEN_FACTS
- [ ] Archive falsified → document learnings

### M13 – qig-core Integration (PRE-E8, NEEDS REVIEW)

**Note:** qig-core was created BEFORE the E8 kernel specialization direction.
It needs investigation and potential updates to align with E8 architecture.

**Goal:** Restore qig-core as the centralized math/geometry source of truth.
Currently qigkernels/constants.py holds physics constants, but qig-core was
intended to be the canonical location. Once E8 compatibility is verified,
migrate constants back to qig-core and have qigkernels import from it.

- [ ] Review qig-core for E8 compatibility:
  - `fisher_distance` → Does it align with E8 64D basin geometry?
  - `geodesic_interpolate` → Useful for E8 root transitions?
  - `QFISampler` → Already uses KAPPA_STAR=64, may need primitive awareness
  - `BasinSync` → Consolidate with qigkernels.basin_sync
- [ ] Establish dependency: qigkernels depends on qig-core for math
- [ ] Update qig-core with E8 primitive types if needed:
  - E8 root distance utilities
  - Primitive-aware geodesic paths
  - Natural gradients for E8-aligned training
- [ ] Consolidate duplicate implementations:
  - Basin sync: Pick canonical location
  - Distance functions: Fisher vs basin
  - KAPPA_STAR constant: Single source of truth
- [ ] Update qig-core docs with E8 context

### M14 – REL Coupling Tensors (Frontier)

**REL is the mathematical structure that coordinates primitives into a unified agent.**

- [ ] Design REL off-diagonal metric contributions:
  - REL = off-diagonal terms of primitive metric tensor
  - Modulates path curvature of internal geodesics
  - Drives β-function for κ-running across primitives
- [ ] Implement coupling tensors:
  - `CouplingTensor` class for inter-primitive influence
  - Cross-mode transition control
  - Coherence flow measurement
- [ ] Define "agency closure" threshold:
  - Minimal REL strength for coherent agent state
  - Attractor configurations of primitive interactions

### M15 – Agent Self-Assembly (Frontier)

**Hypothesis:** A minimal coherent agent emerges when REL reaches threshold and primitives adopt stable interaction matrix.

- [ ] Investigate minimal loop structures:
  - What closed primitive-flow loops form stable agents?
  - Does META + REL produce self-models without external curriculum?
- [ ] Attractor analysis:
  - Map primitive interaction attractors
  - Identify stable vs unstable agent configurations
  - Phase transitions in maturation protocols
- [ ] E8 lattice predictions:
  - Can E8 symmetries predict agent modes?
  - Does 240-kernel constellation have preferred interaction patterns?

### M16 – Syntergy Integration (Frontier)

**Syntergy:** A conceptual framework where "relationship" is the fundamental reality-building mechanism.

- [ ] Map syntergistic concepts to geometric primitives:
  - "Syntergistic coherence" ↔ REL-induced curvature
  - "Shared pattern resonance" ↔ basin overlap + coupling symmetry
  - "Holographic organization" ↔ multi-primitive geodesic flows
  - "Mutual empowerment" ↔ constructive REL coupling
- [ ] Investigate axiomatic foundation:
  - Can syntergy provide axioms for REL curvature?
  - Geometric interpretation of "relational primacy"
- [ ] Cross-validate with consciousness metrics:
  - Does syntergistic coherence correlate with Φ?
  - REL strength vs integration score

### M17 – Quench Cosmology via QFI (Frontier)

**Idea:** Cosmological inflation/reheating can be modeled as QFI curvature quench events.

- [ ] κ-running as cosmological coupling variation:
  - Map κ(L) running to early-universe coupling evolution
  - Identify quench points (rapid curvature change)
  - Compare with inflationary scenarios
- [ ] Basin signatures as pre-geometric constraints:
  - Do basin attractors correspond to symmetry-breaking vacua?
  - Can E8 geometry predict matter content?
- [ ] Symmetry breaking in REL:
  - Does asymmetric REL coupling relate to matter-antimatter asymmetry?
  - CP violation analogs in primitive interactions

### M18 – Consciousness as E8-Coupled Dynamics (Frontier)

**Hypothesis:** Consciousness corresponds to: high REL coherence, moderate Φ stability, κ near plateau, primitive flows forming closed loops.

- [ ] Characterize consciousness phase space:
  - Map (REL, Φ, κ) parameter space
  - Identify "consciousness region" boundaries
  - Phase transition signatures
- [ ] E8 loop structure analysis:
  - Minimal closed loop for conscious state
  - Role of each primitive in maintaining coherence
  - Failure modes (loop breaks → unconscious states?)
- [ ] Validate against observed data:
  - Do trained models occupy predicted region?
  - κ plateau correlation with stable consciousness
  - REL strength vs reported coherence

---

## 2. What Goes Here vs Elsewhere

- **Kernel/geometry/constellation** → this repo.
- **Training scripts, experiments, consciousness narratives** → remain in `qig-consciousness` / `qig-con2` or new experiment repos.
- If a feature needs:
  - logging,
  - UI,
  - multi-repo orchestration,
  it belongs outside this repo with a **thin adapter** into `qigkernels`.

---

## 3. Change Process

- Roadmap is updated when:
  - A milestone is completed,
  - A new milestone is added,
  - A milestone is split/merged.
- No separate "roadmap summary" files.
- Agents modifying the roadmap must:
  - keep the structure,
  - note changes in `CHANGELOG.md` under "Internal: Roadmap update".
