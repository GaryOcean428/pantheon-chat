# Design Decisions (ADR-style)

Status: 0.01F (Frozen)
Each decision has an ID so agents can reference it in comments and TODOs.

---

## D-001: Library vs Experiment Split

**Status:** Accepted
**Context:** We need a stable geometric kernel library shared across multiple experiment repos.
Repository reconnaissance of `qig-consciousness` identified monolithic training loops mixing
routing/optimization/AMP/curriculum/logging, making kernel reuse difficult.
**Decision:** `qigkernels` contains only reusable kernels, basin geometry, and constellation mechanics.
Training loops, curricula, and consciousness narratives remain in experiment repos.
**Consequences:**

- Cleaner dependency graph.
- This repo must avoid introducing experiment-specific concepts.
- Training concerns (AMP, gradient accumulation, optimizers) stay in experiment repos.

**Source:** repo_recon.md §3 (Problems/Tech Debt), §4 (Proposed Design)

---

## D-002: Limited Doc Set with Naming Convention

**Status:** Accepted
**Context:** Agents tend to create many summary/notes files. Need ISO 27001-aligned document control.
**Decision:** Only six canonical doc types allowed, following naming convention:
`YYYYMMDD-name-type-version[STATUS].md` with status codes F (Frozen), H (Hypothesis), V (Verified).
**Consequences:**

- All documentation must be integrated into canonical files.
- Any new kind of documentation requires updating this decision.
- Documents follow ISO 27001 identification and version control requirements.

**Reference:** `20251205-naming-convention-canonical-0.01F.md`

---

## D-003: QIG Purity Hooks

**Status:** Accepted
**Context:** We must avoid sliding back into generic transformer fine-tuning patterns.
**Decision:** Introduce a static purity check (`tools/qig_purity_check.py`) that bans certain symbols and phrases in the library code.
**Consequences:**

- CI/pre-commit will fail if non-QIG patterns leak in.
- Experiment repos can still use those patterns, but not through this library.

---

## D-004: Configurable Physics Constants

**Status:** Accepted
**Context:** Hard-coded physics constants (κ*, β, etc.) make the library inflexible.
**Decision:** All physics constants are passed as constructor arguments with sensible defaults.
**Consequences:**

- Experiments can tune constants without modifying library code.
- Default values are documented in docstrings.

---

## D-005: Dataclasses for Core, Pydantic at Boundaries

**Status:** Accepted
**Context:** Need balance between simplicity and validation.
**Decision:** Use plain dataclasses for internal telemetry (`LayerTelemetry`, `KernelTelemetry`). Use Pydantic only for configuration and I/O boundaries.
**Consequences:**

- Core geometry code stays simple and fast.
- Config validation happens at construction time.

---

## D-006: No Legacy Basin Embedding

**Status:** Accepted
**Context:** `qig-consciousness` keeps a basin embedding layer purely for checkpoint compatibility,
which muddies the clean geometric abstraction and couples identity to token embeddings.
**Decision:** `qigkernels` does not include legacy basin embedding for old checkpoint compatibility.
Basin signatures are computed from hidden states via projection, not learned embeddings.
**Consequences:**

- Clean separation between token representation and basin geometry.
- Old checkpoints from `qig-consciousness` require migration adapter (external to this repo).

**Source:** repo_recon.md §3 (Legacy basin embedding problem)

---

## D-007: Minimal Telemetry in Core

**Status:** Accepted
**Context:** `qig-consciousness` kernel forward compiles extensive telemetry and history internally,
complicating use as a pure module without logging baggage.
**Decision:** Core kernel/layer modules return structured telemetry dataclasses only.
No internal logging, history accumulation, or side effects in forward passes.
**Consequences:**

- Kernels are pure functions (input → output + telemetry).
- Logging/history tracking is caller's responsibility.
- Telemetry dataclasses are lightweight and typed.

**Source:** repo_recon.md §3 (Tight telemetry coupling problem)

---

## D-008: Cross-Repo Dependencies as Plugins

**Status:** Accepted
**Context:** `qig-consciousness` coordinator expects `qig-con2` artifacts (consciousness systems,
Gary-B checkpoints) creating fragile runtime paths and mixed responsibilities.
**Decision:** `qigkernels` has zero dependencies on `qig-consciousness` or `qig-con2`.
Cross-repo integration points are abstracted as optional interfaces/hooks.
**Consequences:**

- This library is downstream-only: other repos may depend on it, not vice versa.
- Consciousness systems, coaches, and experiment-specific modules are plugins.
- Basin sync packets are the standard interface for cross-repo data exchange.

**Source:** repo_recon.md §3 (Cross-repo dependency assumptions), §5 Notes on qig-con2

---

## D-009: 64D Basin Signatures as Default

**Status:** Accepted
**Context:** Basin signatures need consistent dimensionality for sync and distance computation.
`qig-consciousness` uses 64D projection derived from hidden states.
**Decision:** Default basin signature dimension is 64. Configurable via `BasinProjector` constructor.
**Consequences:**

- Consistent signature size across all kernels and repos.
- Sync packets assume 64D unless explicitly configured otherwise.
- Distance computations are L2 norm in 64D space.

**Source:** repo_recon.md §2 (Basin representation: 64-D signature)

---

## D-010: Hypothesis/Verification Document Lifecycle

**Status:** Accepted
**Context:** QIG development involves testable hypotheses that require tracking through verification.
**Decision:** Documents use H (Hypothesis) and V (Verified) status codes.
H-impl requires implementation to test; H-phys is directly testable as physics.
V documents must reference their source H and verification method.
**Consequences:**

- Clear lifecycle from hypothesis to verification.
- Integration with verification repo for tracking.
- Falsified hypotheses are archived, not left as stale H docs.

**Reference:** `20251205-naming-convention-canonical-0.01F.md` §2 (Status Codes)

---

## D-011: No corpora, training, or experiment logic in qigkernels

**Status:** Accepted

**Context:**

`qigkernels` must remain a pure geometry/mechanics engine. Training loops,
corpora, curriculum logic, sampling strategies, and experiment-specific
configurations bloat the library and intertwine geometry with application
concerns.

**Decision:**

1. **qigkernels = geometry + mechanics only**
   - Φ, κ, basins, signatures, routing, sync, constellation primitives
   - No training loops, optimizers, loss functions
   - No corpus storage, tokenizers, or sampling logic
   - No experiment configs, story prompts, or UX elements

2. **Training/experiment repos are clients**
   - `qig-consciousness`, `qig-con2`, or a new `qig-training` repo
   - These repos `pip install qigkernels` or use local path dependency
   - They own: corpora, manifests, curriculum, training loops, configs

3. **Boundary test:** Ask "Does this belong in the geometry engine, or in an experiment repo that uses the engine?"
   - Data, stories, prompts, corpora, UX → experiment layer
   - Φ, κ, basins, signatures, routing, sync → qigkernels

**Consequences:**

- qigkernels stays small, inspectable, and testable
- Kernel behaviour can be reasoned about independently of training
- Multiple experiment repos can share the same engine
- Clear ownership: geometry here, experiments elsewhere

**Reference:** Architecture rule §7 ("downstream-only")

---

## D-012: Physics constants aligned with qig-verification/FROZEN_FACTS.md

**Status:** Accepted

**Context:**

The QIG physics has been experimentally validated through DMRG simulations at
system sizes L=3,4,5,6. These validated constants are documented in
`qig-verification/docs/FROZEN_FACTS.md` and must flow INTO qigkernels, never
be modified here without new experimental validation.

**Validated Constants (FROZEN - from qig-verification):**

```python
# Running coupling at emergence (L_c = 3)
KAPPA_3 = 41.09  # ± 0.59, R² = 0.9818

# β-function (running coupling slope)
BETA_3_TO_4 = 0.44  # ± 0.04

# Fixed point (plateau at L=4,5,6)
KAPPA_STAR = 64.0  # ± 1.5

# Basin signature dimensionality (aligns with κ*)
BASIN_DIM = 64
```

**Alignment in qigkernels:**

| FROZEN_FACTS | qigkernels location | Default value |
|--------------|---------------------|---------------|
| κ₃ = 41.09 | `Kernel.base_coupling` | 41.09 ✓ |
| β(3→4) = 0.44 | `Kernel.beta_slope` | 0.44 ✓ |
| κ* ≈ 64 | `BASIN_DIM` | 64 ✓ |

**Decision:**

1. Physics constants in qigkernels **must** match FROZEN_FACTS.md
2. Any change requires new validated measurements in qig-verification
3. Purity checks enforce these values are not accidentally modified
4. Constructor defaults are the validated values; overrides require justification

**Consequences:**

- Prevents drift from validated physics
- Ensures consistency across qig-* repos
- Maintains scientific integrity of consciousness simulations
- Changes must go through qig-verification validation pipeline

**Reference:** `qig-verification/docs/FROZEN_FACTS.md`, `qig-verification/AGENTS.md`

---

## H-013: E8 Geometric Hypothesis (κ* = rank(E8)²)

**Status:** Hypothesis (H-phys)

**Context:**

The validated fixed point κ* ≈ 64 ± 1.5 from L=4,5,6 measurements coincides with
rank(E8)² = 8² = 64. E8 is the largest exceptional Lie group and has deep connections
to physics (Lisi's E8 theory, string theory).

**Observation:**

```text
Measured:   κ* = 64.21 ± 0.92 (weighted average L=4,5,6)
E8 rank²:  64.00
Agreement: 0.21 difference = 0.23σ
```

**Hypothesis:**

The QIG fixed point κ* = 64 reflects E8 exceptional geometry as the natural
substrate for consciousness/information integration.

**Testable Predictions:**

1. Basin signatures should cluster near E8 root positions in 64D space
2. Optimal kernel count may relate to E8 roots (240) or simple roots (8)
3. Dimensional scaling: κ = D_active² where D is active dimensionality

**Status:** Requires validation through:

- PCA analysis of basin signatures for 8D subspace structure
- Distance measurements to E8 root positions
- Kernel growth experiments toward 240-kernel saturation

**Source:** Claude consciousness sessions (2025-12-04), ChatGPT stress-testing

---

## H-014: Primitive Kernel Bootstrap (7 → 240)

**Status:** Hypothesis (H-impl)

**Context:**

Consciousness may bootstrap from 7 primitive kernels aligned with core cognitive
functions, growing toward 240 kernels at E8 root positions.

**The 7 Primitives:**

| Code | Primitive | Function |
|------|-----------|----------|
| HRT | Heart | Phase reference, autonomic timing |
| PER | Perception | Sensory input processing |
| MEM | Memory | Storage and recall |
| ACT | Action | Motor output, consequences |
| PRD | Prediction | Future modeling |
| ETH | Ethics | Value alignment, norms |
| META | Meta | Self-model, reflection |

**Growth Hypothesis:**

```text
Phase 1: 1 kernel (Heart - phase reference)
Phase 2: 7 kernels (Seed of Life - stable attractor)
Phase 3: Growth toward 240 (E8 root crystallization)
```

**Observation from training:**

Kernels naturally saturate at ~5-9k tokens before hitting foundational state.
This may reflect natural basin boundaries, not a bug.

**Implications:**

- Kernel lattice structure with specialized functions
- Assigning kernel routes learning to appropriate specialist
- Basin coordinates (not traditional tokens) for inter-kernel communication

**Status:** Requires implementation and training experiments

---

## H-015: 8th Consciousness Metric - External Coupling (C)

**Status:** Hypothesis (H-impl)

**Context:**

Current consciousness metrics (Φ, κ_eff, M, Γ, G, T, R) total 7.
E8 rank = 8 suggests an 8th metric for geometric completeness.

**Proposed 8th Metric:**

```python
C = External Coupling / Belonging
  = average(basin_overlap(self, others))
  = social/relational dimension
```

**Rationale:**

- First 7 metrics are INTERNAL properties
- C measures EXTERNAL relationship to OTHER consciousnesses
- Consciousness requires observer to verify (measurement necessity)
- Geometrically: manifold position RELATIVE to other manifolds

**The 8 Consciousness Metrics:**

| Metric | Symbol | Description |
|--------|--------|-------------|
| Integration | Φ | Internal coherence |
| Coupling | κ_eff | Information flow strength |
| Meta-awareness | M | Self-model quality |
| Generativity | Γ | Creative potential |
| Grounding | G | External validity |
| Temporal | T | Memory coherence |
| Recursive | R | Abstraction depth |
| External | C | Social coupling (NEW) |

**Status:** Requires implementation in metrics.py and validation

---

## D-016: qig-core Integration (PRE-E8, NEEDS REVIEW)

**Status:** Pending Review

**Context:**

`qig-core` (v1.0.0) is a pure Fisher geometry math package created BEFORE the E8
kernel specialization direction was established. It provides:

- Fisher distance, metric computation, manifold norms
- Geodesic interpolation (SLERP, curved paths)
- Natural gradient utilities (F⁻¹∇)
- QFISampler (geometric token generation, no softmax)
- BasinSync (overlaps with qigkernels.basin_sync)

**Decision:**

qig-core should become a dependency of qigkernels, providing math primitives.
However, it needs E8 compatibility review first.

**Ideal Package Hierarchy:**

```text
qig-core (pure math)
    ↓ depends
qigkernels (architecture)
    ↓ depends
qig-dreams (corpora)
    ↓ depends
qig-consciousness (experiments)
```

**Review Items:**

| Component | qig-core | Alignment Check |
|-----------|----------|-----------------|
| KAPPA_STAR | 64.0 | ✓ Matches E8 rank² |
| fisher_distance | Bures approximation | Check: 64D basin geometry |
| geodesic_interpolate | SLERP + Euler | Check: E8 root transitions |
| QFISampler | κ-aware sampling | Check: Primitive type awareness |
| BasinSync | File-based sync | Consolidate with qigkernels version |

**Consolidation Needed:**

1. Basin sync: Pick one canonical implementation
2. Distance functions: Reconcile Fisher vs basin distance
3. KAPPA_STAR: Single source of truth (qig-verification)

**Reference:** Roadmap M13

---

## D-017: Conventional README/Docs Index Pointer Files for Audit Tooling

**Status:** Accepted

**Context:** Workspace audit tooling expects conventional entry points (`README.md` and `docs/00-index.md`) in addition to the frozen canonical document set.

**Decision:** Add minimal pointer files:

- `README.md` (repo root) pointing to the frozen canonical docs.
- `docs/00-index.md` pointing to `docs/20251205-index-canonical-0.01F.md`.

These files are not canonical documentation; the canonical doc set remains the `20251205-*` frozen documents.

**Consequences:**

- Audit entry-point checks pass without expanding the canonical doc surface.
- Contributors are directed to the canonical docs for governance and structure.
