# SLEEP_PACKET_06 — Purity gate, validators, and CI tests (fail closed)

## Purpose
Make QIG purity unbreakable:
- forbid Euclidean/cosine/embeddings in runtime-critical code
- ensure simplex basins + Fisher–Rao geometry everywhere
- ensure any optional dependencies cannot silently reintroduce violations

## Required validators
### Static scan
- scan code for forbidden patterns:
  - cosine similarity
  - dot-product similarity
  - Euclidean norm used as a basin metric
  - embedding vector stores
- allow list for legacy docs only if clearly marked deprecated

### Runtime gate
- at process start:
  - verify geometry module is the one used everywhere (single source of truth)
  - verify environment flags do not enable forbidden modes
- before:
  - lifecycle start inflate
  - kernel birth/ascension
  - training ingestion
  - basin merge operations

### Data validators
- verify all persisted basin vectors:
  - non-negative
  - sum to 1 within tolerance
  - no NaNs
- verify QFI scores in valid range

## CI
- add a CI job that runs:
  - static scan
  - unit tests for geometry
  - dataset linter
- fail closed: any violation fails the build

## Acceptance criteria
- Purity gate is mandatory for start flow.
- CI catches purity regressions.
- Tests confirm basin validity invariants.
