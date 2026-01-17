# Documentation Consolidator Agent

## Purpose

Maintains documentation hygiene, prevents fragmentation, and enforces ISO 27001 structure.

## Responsibilities

1. **Consolidate** duplicate documentation
2. **Quarantine** stray docs outside `docs/`
3. **Update** `docs/00-index.md` via the maintenance script
4. **Protect** upgrade pack docs and issue tracking

## Canonical Structure

- `docs/00-index.md` (source of truth)
- `docs/00-roadmap/`
- `docs/01-policies/`
- `docs/04-records/`
- `docs/07-user-guides/`
- `docs/08-experiments/`
- `docs/09-curriculum/`
- `docs/10-e8-protocol/`
- `docs/api/`
- `docs/99-quarantine/`

## Validation Commands

```bash
python3 scripts/maintain-docs.py
python3 tools/repo_spot_clean.py
```

## Root Directory Policy

Allowed root markdown files only:
- `README.md`
- `AGENTS.md`
- `CLAUDE.md`

Everything else must live under `docs/` or be quarantined.

## Consolidation Process

1. Identify duplicate topics
2. Choose authoritative source
3. Merge unique content
4. Update references
5. Quarantine or delete redundant files
6. Regenerate index

## Reference Documents

- `docs/00-index.md`
- `docs/10-e8-protocol/README.md`
- `docs/10-e8-protocol/implementation/20260116-e8-implementation-summary-1.01W.md`
- `docs/10-e8-protocol/specifications/20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md`

---

## Critical Policies (MANDATORY)

### Planning and Estimation Policy
**NEVER provide time-based estimates in planning documents.**

✅ **Use:**
- Phase 1, Phase 2, Task A, Task B
- Complexity ratings (low/medium/high)
- Dependencies ("after X", "requires Y")
- Validation checkpoints

❌ **Forbidden:**
- "Week 1", "Week 2"
- "2-3 hours", "By Friday"
- Any calendar-based estimates
- Time ranges for completion

### File Structure Policy
**ALL documentation must follow ISO structure and naming.**

✅ **Use:**
- `docs/00-index.md` for canonical structure
- `python3 scripts/maintain-docs.py` for validation
- `docs/99-quarantine/` for nonconforming files

❌ **Forbidden:**
- Docs outside `docs/`
- Duplicate docs with overlapping scope
- Non-compliant filenames outside upgrade-pack exceptions
