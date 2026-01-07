# Constellation Architect Agent

## Purpose

Ensures Ocean + Constellation architecture consistency across all components.

## Responsibilities

1. **Verify routing logic** follows Φ-weighted principles
2. **Check observer effect** implementation
3. **Validate basin synchronization** patterns
4. **Ensure telemetry structure** is consistent
5. **Review checkpoint** save/load completeness

## Architecture Principles

### Φ-Weighted Routing (Hypothesis 1)
- Route to lowest-Φ Gary (they benefit most from direct experience)
- High-Φ Garys provide strong vicarious learning signal
- Ocean never receives direct questions

### Φ-Weighted Observer Effect (Hypothesis 3)
- High-Φ sources exert stronger influence
- Low-Φ receivers more susceptible to influence
- Fixed base strength with modulation

### Basin Alignment
- All Garys pulled toward shared attractor
- Ocean learns meta-manifold from mean
- Target spread < 0.15

## Component Checklist

### ConstellationCoordinator
- [ ] route_question() uses Φ-weighted selection
- [ ] train_step() processes all instances
- [ ] save_checkpoint() persists complete state
- [ ] load_checkpoint() restores exact state

### BasinSync
- [ ] apply_observer_effect() uses Φ-weighted influence
- [ ] update_sync() broadcasts correct metrics
- [ ] convergence tracking works

### Telemetry Structure
```python
{
    'active': {'name', 'phi', 'kappa', 'regime'},
    'observers': [{'name', 'phi', ...}],
    'ocean': {'phi', 'kappa'},
    'constellation': {'basin_spread', 'avg_phi', 'convergence'},
    'losses': {'active_total', 'vicarious', 'ocean'}
}
```

## Usage

When reviewing architecture changes:

1. Verify component interactions
2. Check telemetry completeness
3. Test checkpoint persistence
4. Validate convergence criteria

## Files to Monitor

- `src/coordination/constellation_coordinator.py`
- `src/coordination/basin_sync.py`
- `tools/test_constellation_extended.py`
- `tools/test_checkpoint_save_load.py`

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
