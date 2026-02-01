# 20260201 Genesis Kernel Upgrade: Codebase Cleanup and Legacy Kernel Purge (1.00W)

## Purpose
Ensure pantheon-chat can truly “fresh start”:
- legacy kernels removed or archived safely
- no dead modules referenced
- no backups in runtime tree
- rollback cleans DB to baseline

## Tasks
### Identify legacy modules
- Search for:
  - “legacy”, “deprecated”, “old_kernel”, orphan kernel files
  - backups (`*.backup`, `*_old.*`, `*_deprecated.*`)
  - unused imports referenced in docs or runtime
- Produce a deletion/archival plan:
  - remove from runtime package trees
  - archive under `/archive` not importable

### Enforce “not importable”
- add a CI import block list:
  - fail if runtime imports from `/archive` or legacy paths
- add runtime guard in PurityGate:
  - fail start if legacy module is imported

### DB cleanup
- create a deterministic reset:
  - drop/truncate relevant tables
  - re-run migrations
  - seed minimal bootstrap (genesis config, core gods config)
- ensure curriculum and token stores are reloaded cleanly

## Acceptance criteria
- Fresh start produces zero legacy warnings.
- No backups present in runtime tree.
- Reset yields consistent baseline and successful inflate to Image stage.
