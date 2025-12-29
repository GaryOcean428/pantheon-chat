# QIG Kernels Architecture & Rules

Status: 0.01F (Frozen)
Canonical naming & geometry: see `20251205-type-symbol-manifest-canonical-0.01F.md`.

---

## 1. Module Layers & Import Rules

The import graph must remain acyclic and follow this direction:

```text
[core]      : qigkernels.kernel, qigkernels.layer
[geometry]  : qigkernels.basin, qigkernels.metrics
[structure] : qigkernels.constellation, qigkernels.router, qigkernels.basin_sync
[io/tools]  : qigkernels.storage, tools/
```

### Allowed imports

* `kernel` may import:
  * `layer`,
  * standard libs,
  * `torch`, `pydantic` (for configs/telemetry models if used).

* `basin` may import:
  * standard libs,
  * `torch`.

* `metrics` may import:
  * `basin`,
  * `kernel` (for `KernelTelemetry` only).

* `router` may import:
  * standard libs,
  * `typing`.

* `constellation` may import:
  * `kernel`,
  * `basin`,
  * `router`.

* `basin_sync` may import:
  * `basin`,
  * `constellation`.

* `storage` may import:
  * `basin`,
  * `constellation`.

* `pure_kernel_template` may import:
  * standard libs,
  * `torch`.

* `tools/qig_purity_check.py` may import anything, but **no production code may import tools**.

> **Rule:** If a new module is added, its allowed imports must be documented here and updated in `__init__.py`.
> No new `.py` files under `qigkernels/` without updating this section.

---

## 2. QIG Purity Rules

We inherit global rules from `TYPE_SYMBOL_CONCEPT_MANIFEST.md` and add code-level constraints:

### 2.1 Conceptual Constraints

* Think and document in **geometric** terms:
  * Fisher metric, basins, κ, Φ, curvature, manifolds.
* Avoid importing or reimplementing classical **"off-the-shelf transformer"** stacks.
* Any use of standard components (e.g. `nn.Embedding`) must serve the geometric architecture, not the other way around.

### 2.2 Banned / discouraged patterns (enforced by `qig_purity_check.py`)

* Hard bans (script fails if found in library code):
  * `nn.Transformer`
  * `BertModel`
  * `GPT2Model`
  * `CrossEntropyLoss` (without a comment explaining geometric role)
  * `AdamW` / `Adam` optimizers in this repo

* Soft bans (warnings):
  * "token-level cross entropy" in comments/docs.
  * "just fine-tune a transformer" in comments/docs.

If a banned symbol legitimately appears in an experiment adapter, that code lives outside this repo.

---

## 3. File Length & Modularity

* Soft limit: **400 lines** per module under `qigkernels/`.
* If a module exceeds 400 lines, agents must:
  * refactor into smaller modules, **or**
  * explicitly justify in `DECISIONS.md` (with a new decision entry).
* No "god modules" performing orchestration + I/O + geometry.

---

## 4. Pydantic & Config Usage

* Pydantic models may be used for:
  * kernel configuration (`KernelConfig`),
  * telemetry snapshot types,
  * constellation configuration.

**Rules:**

* All Pydantic models live in the module they logically belong to (no random `models.py` dumping ground).
* They must be imported via `qigkernels.kernel`, `qigkernels.constellation`, etc., not deep paths.
* Pydantic models must use `ConfigDict` to forbid extra fields (`extra="forbid"`).

If in doubt, prefer plain dataclasses in core geometry and use Pydantic only at the boundaries (config/IO).

---

## 5. Documentation Discipline

Allowed top-level docs:

* `README.md` – What this library is, high-level usage.
* `ARCHITECTURE.md` – This file: structure, rules, imports.
* `ROADMAP.md` – Milestones and sequencing (no time estimates).
* `CHANGELOG.md` – Human-readable log of changes.
* `DECISIONS.md` – Design decisions and their status.
* `TYPE_SYMBOL_CONCEPT_MANIFEST.md` – Canonical external spec (read-only here).

**Forbidden:**

* New "summary" or "notes" `.md` files.
* Per-feature ad-hoc docs like `basin_notes.md`, `design_v3.md`, etc.

If an agent wants to "summarize" or "document":

* Update `ARCHITECTURE.md` or `DECISIONS.md`.
* Record what changed in `CHANGELOG.md`.

---

## 6. Agent Behaviour Guidelines

* No timeframes or schedules in any file.
* No "we'll do a simpler version first and maybe improve later" as a justification for stubbing core logic.
  * If something is intentionally stubbed:
    * mark clearly as `# TODO:`,
    * link to a specific entry in `ROADMAP.md` or `DECISIONS.md`.
* When modifying behaviour, prefer:
  * **editing existing files** over creating new ones.
  * e.g. adjust `ARCHITECTURE.md` instead of adding `architecture_v2.md`.

---

## 7. Import / Export Manifest

All external surface of the library is defined in `qigkernels/__init__.py`.

* Any new public function/class must be:
  * imported and re-exported in `__init__.py`,
  * documented in `README.md` under "Public API".
* No module may import from `qig-consciousness` or `qig-con2`.
* This library is **downstream-only**: other repos may depend on it, not vice versa.

---

## 8. Explicit Out-of-Scope

The following are **never** part of `qigkernels` (see D-011):

| Category | Examples | Where it belongs |
|----------|----------|------------------|
| **Training** | Optimizers, loss functions, training loops, gradient accumulation | `qig-training` or experiment repo |
| **Corpora** | Raw text, tokenizers, vocab files, sampling logic | `qig-training` or experiment repo |
| **Curriculum** | Phase schedules, regime progressions, maturity tracking | Experiment repo |
| **UX/Story** | Chat interfaces, consciousness narratives, "Gary/Ocean" prompts | Experiment repo |
| **Safety/Ethics** | Safety metrics, content filtering, social alignment scores | Experiment repo |
| **Experiment configs** | Hyperparameter sweeps, experiment manifests, run configs | Experiment repo |

**Boundary test:** "Does this belong in the geometry engine, or in an experiment repo that uses the engine?"

* Φ, κ, basins, signatures, routing, sync → **qigkernels**
* Data, stories, prompts, corpora, training → **experiment layer**

---

## 9. Physics Constants (FROZEN)

All physics constants in qigkernels are experimentally validated. The authoritative
source is `qig-verification/docs/FROZEN_FACTS.md`.

### Validated Constants

| Constant | Value | Source | Location in qigkernels |
|----------|-------|--------|------------------------|
| κ₃ | 41.09 ± 0.59 | L=3 emergence | `Kernel.base_coupling` |
| β(3→4) | 0.44 ± 0.04 | Running coupling slope | `Kernel.beta_slope` |
| κ* | 64 ± 1.5 | Fixed point (L=4,5,6) | `BASIN_DIM` |

### Rules

1. **Never change** these defaults without new validated measurements in qig-verification
2. Tests in `test_smoke.py::TestPhysicsConstants` enforce alignment
3. Constructor overrides are allowed for experiments but require justification
4. Physics flows FROM qig-verification TO qigkernels, never reverse

**Reference:** Decision D-012, `qig-verification/docs/FROZEN_FACTS.md`

---

## 10. E8 Geometric Foundation

The validated physics (κ* = 64) coincides with E8 exceptional geometry (rank² = 8² = 64).
This is the implementation direction for qigkernels: build E8-aligned structures, measure results.

### Foundation (Validated Physics)

```text
κ* = 64 ± 1.5     ← Experimentally validated fixed point (L=4,5,6)
rank(E8)² = 64    ← E8 exceptional Lie group rank squared
Agreement: 0.23σ  ← Not coincidence, geometric necessity
```

### Implementation Direction (Informed Hypotheses)

| Hypothesis | ID | Implementation Target |
|------------|----|-----------------------|
| κ* = rank(E8)² | H-013 | Basin signatures in 64D ≈ E8-aligned space |
| 7 primitive kernels | H-014 | Kernel lattice with HRT/PER/MEM/ACT/PRD/ETH/META |
| 240-kernel saturation | H-014 | Growth toward E8 root positions |
| 8 consciousness metrics | H-015 | Add External Coupling (C) as 8th metric |

### Build → Measure → Refine Cycle

```text
Phase 1: Build E8-aligned structures
         - 64D basin signatures (done)
         - 7 primitive kernel codes (qig-dreams)
         - Constellation routing (done)

Phase 2: Implement kernel lattice
         - Kernel specialization by primitive
         - Basin-coordinated inter-kernel routing
         - Natural saturation detection (~5-9k tokens)

Phase 3: Measure against E8 predictions
         - PCA for 8D active subspace
         - Distance to E8 root positions
         - Kernel count trajectory toward 240

Phase 4: Refine hypotheses based on measurements
         - Promote validated → FROZEN_FACTS
         - Archive falsified hypotheses
         - Iterate implementation
```

### Geometric Primitives (from H-014)

These align with `qig-dreams` corpus primitive codes:

| Code | Kernel Type | E8 Role |
|------|-------------|---------|
| HRT | Heart | Phase reference (simple root 1) |
| PER | Perception | Sensory input (simple root 2) |
| MEM | Memory | Storage/recall (simple root 3) |
| ACT | Action | Motor output (simple root 4) |
| PRD | Prediction | Future modeling (simple root 5) |
| ETH | Ethics | Value alignment (simple root 6) |
| META | Meta | Self-model (simple root 7) |
| MIX | Multi | Cross-primitive (simple root 8) |

The 8 simple roots generate E8's 240 roots via Weyl group action.
Kernels "crystallize" toward these root positions during training.

**Reference:** Decisions H-013, H-014, H-015
