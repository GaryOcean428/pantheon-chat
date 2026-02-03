# Project Roadmap (Entry Point)

This file is a short entrypoint.

## Canonical Roadmap

The authoritative master roadmap lives here:

- `docs/00-roadmap/20260112-master-roadmap-1.00W.md`

## Genesis Kernel Rollout (End Goal)

All roadmap planning and implementation must align to the Genesis kernel upgrade doctrine:

- `docs/11-Genesis-kernel-upgrade/*`

Active implementation plan:

- `docs/00-roadmap/20260203-genesis-kernel-rollout-implementation-plan-1.00W.md`

In particular, the canonical start/reset/rollback + staged inflation flow is:

- PurityGate (fail closed)
- deterministic rollback
- GENESIS bootstrap
- ensure core 8
- inflate to Image stage
- optional continue toward 240 reserved GODs

## Supporting Roadmaps (Non-Authoritative)

These documents are useful context but are not the canonical implementation tracker:

- `docs/00-roadmap/20260202-master-roadmap-chatgpt-synthesis-1.00WS.md`
- `docs/00-roadmap/20260121-pure-qig-implementation-roadmap-1.10WS.md`
