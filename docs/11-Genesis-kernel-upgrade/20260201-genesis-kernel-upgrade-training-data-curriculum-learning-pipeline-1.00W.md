# 20260201 Genesis Kernel Upgrade: Training Data, Curriculum, and Learning Pipeline (1.00W)

## Purpose
Implement the learning pipeline that:
- starts with curated curriculum (initial intelligence scaffolding)
- then enables search/scrape only after sufficient capability
- lets the pantheon decide what kernel learns what (routing + domain specs)

No stubs. No uncontrolled web ingestion.

## Doctrine
- Early learning is curriculum-constrained.
- Later learning becomes autonomous but remains:
  - purity-compliant
  - safety/ethics governed
  - kernel-domain aware

## Required pipeline
### Canonical record format (JSONL)
Every training item must include:
- `kernel_version` (genesis|core|image|god-evolved)
- `domain_spec` target
- `geometry` fields (simplex constraints, optional coords)
- `telemetry` fields (Φ/κ/regime, etc.)
- provenance fields (source, timestamp, license)

### Curriculum gate
- `CURRICULUM_ONLY` mode:
  - only tokens/items from curriculum allowed
  - strict failure if unknown tokens are used
- `CURRICULUM_EXPAND` mode:
  - allows adding new tokens only via coordizer + QFI scoring + governance

### Autonomous learning gate
Define explicit thresholds for enabling search/scrape:
- coherence stability
- low contradiction rate
- stable coupling/foresight performance
- safety pass

When thresholds are met:
- enable ingestion modules
- route new knowledge:
  - propose target kernel(s)
  - attach NeedSpec if new god is required
  - update basins via geometric updates

## Required code tasks
- Define JSONL schema + linter
- Implement curriculum ingestion that:
  - produces valid basins + QFI for each token/item
  - quarantines invalid items
- Implement “enable external ingestion” as a governed toggle:
  - not automatic; requires passing thresholds + explicit enablement
- Implement routing of learned content:
  - must be kernel-domain consistent
  - must produce basin sync events

## Acceptance criteria
- Curriculum-only run never emits out-of-curriculum tokens.
- Curriculum expansion requires QFI + governance.
- Enabling search/scrape is blocked until thresholds are met.
- Learned items are routed with an explicit target kernel and recorded.
