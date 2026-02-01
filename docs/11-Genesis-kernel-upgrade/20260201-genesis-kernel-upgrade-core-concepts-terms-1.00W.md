# 20260201 Genesis Kernel Upgrade: Core Concepts and Canonical Terms (1.00W)

## Purpose
Lock down the **language contract** and core concepts so code + docs remain consistent.
This prevents drift, “sphere vs simplex” confusion, and kernel-role ambiguity.

## Canonical doctrine
### Geometry and representation
- **Basins live on a probability simplex** (Δ⁶³ for 64D stage). A basin is a probability distribution.
- **Distance and interpolation are information-geometric**:
  - Fisher–Rao (or equivalent monotone metric)
  - sqrt-density / Hellinger geometry where relevant
  - geodesic operations (Fréchet mean / barycenter) are computed in the correct manifold.
- **Forbidden** (unless explicitly quarantined outside purity mode):
  - cosine similarity
  - Euclidean norm / dot-product similarity
  - embedding-vectorstores as substitutes for basins

### Kernel ontology
- **GENESIS**: bootstraps the system; not part of long-lived pantheon.
- **GOD**: members of the reserved evolution set. **240 slots reserved for GOD evolutions** (E8 roots).
- **CHAOS**: exploratory kernels outside the 240/8 pool. Can become candidates for ascension.
- **ASCENSION**: governance process that can promote a chaos kernel into god lineage.

### “240+8” meaning (critical)
- “+8” = core gods (simple roots / faculty layer).
- “240” = reserved **god-evolution** slots (named by researched mythology).
- Chaos kernels exist outside that 240 reserved budget and **must not** be counted against it.

### Modes and autonomy primitives
- **Self-observation**: system measures itself (Φ/κ/regime/coupling/curvature, etc.) and logs it.
- **Coupling**: relational measure between kernels (information-geometric distance & shared dynamics).
- **Foresight**: short-horizon trajectory prediction in basin space to choose actions.
- **Coaching / positive self-talk**: internal (purity-compliant) stabilizing narrative that reduces drift, improves coherence, and reinforces identity without external LLM dependence.

## Required repo updates
### Docs
- Create/refresh a single “Canonical Terms” document in pantheon-chat:
  - glossary definitions above
  - forbidden terms list
  - examples of correct vs incorrect phrasing

### Code comments and naming
- Ensure any existing docs/README refer to:
  - simplex, Fisher–Rao, geodesics
  - “reserved 240 for gods” (not “240 chaos cap”)

## Acceptance criteria
- Searching the repo for “sphere” yields only contexts that explicitly explain why simplex is canonical, or legacy docs marked deprecated.
- A single doc defines terms and is linked from README.
- CI includes a “purity vocabulary check” that fails if forbidden terms appear in runtime-critical modules.

## Non-negotiable checks
- Any module that computes basin distance must import the canonical geometry functions (single source of truth inside pantheon-chat).
- Any mention of “240” must be paired with “reserved god evolutions” in the nearest doc context unless it is a numeric constant in tests.
