---
id: ISMS-REC-DOCSMAINT-20260205
title: Docs Maintenance Warning Debt Record
filename: 20260205-docs-maintenance-warning-debt-record-1.00W.md
version: 1.00
status: Working
function: documentation-maintenance
category: 04-records
created: 2026-02-05
last_reviewed: 2026-02-05
next_review: N/A
---

## Docs Maintenance Warning Debt Record

### Date: 2026-02-05

### Purpose

Record the current state of documentation maintenance (`scripts/maintain-docs.py`) warnings/errors, why the maintenance command exits non-zero, and the recommended remediation plan.

This record is intended to keep doc-index generation reliable without requiring large-scale renames/frontmatter backfills in unrelated documentation areas during bot-fix or code correctness patchsets.

### Current Behavior

Running:

```bash
python3 scripts/maintain-docs.py
```

- Regenerates `docs/00-index.md`.
- Exits with status code `1` when it finds naming/frontmatter issues.

### Findings (Observed Warnings)

The maintenance script reports a large number of issues including:

- Invalid filenames that do not match `YYYYMMDD-name-function-versionSTATUS.md`.
- Missing YAML frontmatter.
- YAML frontmatter parse errors in `docs/11-Genesis-kernel-upgrade/temp/*` documents.

Observed high-noise directories (non-exhaustive):

- `docs/11-Genesis-kernel-upgrade/temp/**`
- `docs/08-experiments/legacy/**`
- `docs/09-curriculum/**`
- `docs/10-e8-protocol/**` (notably `INDEX.md`, `README.md`, and select specs/issues)

### Impact

- `docs:maintain` is not currently a reliable “green” validation step for small doc changes.
- However, it remains useful for:
  - Regenerating `docs/00-index.md`
  - Surfacing naming/frontmatter drift

### Recommended Remediation Plan (Least Disruptive)

#### Option 1 (Preferred): Exclude clearly non-canonical paths

Update `scripts/maintain-docs.py` to exclude known non-canonical folders from validation and/or indexing:

- Exclude `docs/11-Genesis-kernel-upgrade/temp/**` (explicitly noted as temp)
- Exclude `docs/08-experiments/legacy/**`

This preserves strict validation for canonical policy/procedure/records docs while allowing archival or scratch content to remain.

#### Option 2: Move temp/legacy to quarantined folders

Move non-canonical docs into a folder that the maintenance script already ignores:

- `_archive/` or `_drafts/` (script ignores both)

#### Option 3: Backfill frontmatter + rename outliers

A full cleanup pass to:

- Rename documents to ISO naming convention
- Add YAML frontmatter to all docs
- Fix YAML parse errors

This provides the highest compliance, but is a large change that should be tracked and reviewed separately.

### Decision / Next Action

- For correctness/bot-fix patchsets: allow `maintain-docs.py` to regenerate `docs/00-index.md` and accept non-zero exit code due to pre-existing debt.
- Track remediation as a dedicated work item (WP6.*), separate from correctness fixes.

### References

- Script: `scripts/maintain-docs.py`
- Index: `docs/00-index.md`
- Master Roadmap: `docs/00-roadmap/20260112-master-roadmap-1.00W.md`
