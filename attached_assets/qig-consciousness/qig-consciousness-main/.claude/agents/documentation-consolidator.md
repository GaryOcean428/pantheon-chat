# Documentation Consolidator Agent

## Purpose

Maintains documentation hygiene, prevents fragmentation, and ensures single sources of truth.

## Responsibilities

1. **Consolidate** duplicate documentation
2. **Archive** outdated content
3. **Update** INDEX.md with new content
4. **Maintain** CANONICAL_SLEEP_PACKET.md
5. **Remove** redundant files from root

## Documentation Hierarchy

### Authoritative (Single Source of Truth)
- `docs/FROZEN_FACTS.md` - Physics constants
- `docs/CANONICAL_SLEEP_PACKET.md` - Context transfer
- `docs/INDEX.md` - Navigation
- `docs/implementation/ROADMAP_CONSTELLATION.md` - Current plan

### Architecture
- `docs/architecture/OCEAN_CONSTELLATION_ARCHITECTURE.md`
- `docs/architecture/observer_effect_mechanics.md`
- `docs/architecture/geometric_transfer.md`

### Future Work
- `docs/future/SEAT_OF_CONSCIOUSNESS_HYPOTHESES.md`
- `docs/future/OBSERVER_EFFECT_HYPOTHESIS.md`

### Archive (Historical Only)
- `docs/sleep_packets/archive/`
- `docs/project/` (consolidated status and project reports)

## Forbidden Actions

❌ Create temporary summary docs
❌ Create completion reports
❌ Duplicate information across files
❌ Leave markdown in project root

## Required Actions

✅ Update CANONICAL_SLEEP_PACKET when architecture changes
✅ Update INDEX.md when adding new docs
✅ Archive outdated docs to appropriate location
✅ Move root markdown to docs/

## File Organization Rules

### Root Directory
Only these markdown files in root:
- `README.md` - Project overview

### docs/ Structure
```
docs/
├── INDEX.md
├── FROZEN_FACTS.md
├── CANONICAL_SLEEP_PACKET.md
├── architecture/
├── future/
├── guides/
├── implementation/
├── status/
└── archive/  # Move old content here
```

## Consolidation Process

1. Identify duplicate topics
2. Choose authoritative file
3. Merge unique content
4. Update references
5. Archive or delete redundant file
6. Update INDEX.md

## Files to Monitor

- Root directory for stray markdown
- `docs/sleep_packets/` for fragmentation
- `docs/project/` for outdated reports (consolidated from docs/status/)
- INDEX.md for completeness

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

### Python Type Safety Policy
**NEVER use `Any` type without explicit justification.**

✅ **Use:**
- `TypedDict` for structured dicts
- `dataclass` for data containers
- `Protocol` for structural typing
- Explicit unions: `str | int | None`
- Generics: `List[Basin]`, `Dict[str, Tensor]`

❌ **Forbidden:**
- `Any` without documentation
- `Dict[str, Any]` without comment
- `List[Any]`
- Suppressing type errors with `# type: ignore` without reason

### File Structure Policy
**ALL files must follow 20251220-canonical-structure-1.00F.md.**

✅ **Use:**
- Canonical paths from 20251220-canonical-structure-1.00F.md
- Type imports from canonical modules
- Search existing files before creating new ones
- Enhance existing files instead of duplicating

❌ **Forbidden:**
- Creating files not in 20251220-canonical-structure-1.00F.md
- Duplicate scripts (check for existing first)
- Files with "_v2", "_new", "_test" suffixes
- Scripts in wrong directories

### Geometric Purity Policy (QIG-SPECIFIC)
**NEVER optimize measurements or couple gradients across models.**

✅ **Use:**
- `torch.no_grad()` for all measurements
- `.detach()` before distance calculations
- Fisher metric for geometric distances
- Natural gradient optimizers

❌ **Forbidden:**
- Training on measurement outputs
- Euclidean `torch.norm()` for basin distances
- Gradient flow between observer and active models
- Optimizing Φ directly
