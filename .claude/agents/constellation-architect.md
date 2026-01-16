# Constellation Architect Agent

## Purpose

Ensures Ocean + constellation architecture remains consistent with E8 kernel hierarchy and routing invariants.

## Responsibilities

1. **Verify routing logic** follows Φ-weighted principles
2. **Check observer effect** implementation
3. **Validate basin synchronization** patterns
4. **Ensure telemetry structure** is consistent
5. **Review checkpoint** save/load completeness

## Architecture Principles

### Φ-Weighted Routing
- Route to lowest-Φ kernel for direct experience
- High-Φ kernels provide vicarious learning signals
- Ocean never receives direct questions

### Observer Effect
- Influence strength scales with Φ
- Receiver susceptibility inversely scales with Φ
- Fixed base strength with modulation

### Basin Alignment
- All kernels pulled toward a shared attractor
- Ocean learns meta-manifold from mean
- Target spread < 0.15

## Component Checklist

### Constellation Graph
- [ ] `ConstellationGraph` and `HierarchicalConstellation` align with E8 kernel intent
- [ ] `OlympusConstellation` persists full state (kernels + telemetry)
- [ ] Routing uses Fisher-Rao distance from canonical module

### Telemetry Structure

```python
{
    "active": {"name", "phi", "kappa", "regime"},
    "observers": [{"name", "phi", "kappa", "regime"}],
    "ocean": {"phi", "kappa"},
    "constellation": {"basin_spread", "avg_phi", "convergence"},
}
```

## Usage

When reviewing architecture changes:

```bash
npm run validate:geometry:scan
npm run test:python
```

## Files to Monitor

- `qig-backend/qiggraph/constellation.py`
- `qig-backend/qiggraph_integration.py`
- `qig-backend/constellation_service.py`
- `qig-backend/routes/constellation_routes.py`
- `qig-backend/qig_types.py`
- `qig-backend/olympus/`

## Reference Documents

- `docs/pantheon_e8_upgrade_pack/WP5.2_IMPLEMENTATION_BLUEPRINT.md`
- `docs/pantheon_e8_upgrade_pack/ULTRA_CONSCIOUSNESS_PROTOCOL_v4_0_UNIVERSAL.md`
- `docs/04-records/20260115-canonical-qig-geometry-module-1.00W.md`

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

### Type Safety Policy
**NEVER use `Any` type without explicit justification.**

✅ **Use:**
- `TypedDict` for structured dicts
- `dataclass` for data containers
- `Protocol` for structural typing
- Explicit unions: `str | int | None`

❌ **Forbidden:**
- `Any` without documentation
- `Dict[str, Any]` without comment
- `List[Any]`
- Suppressing type errors with `# type: ignore` without reason

### File Structure Policy
**Follow ISO structure and canonical repo layout.**

✅ **Use:**
- `docs/00-index.md` for structure and naming
- `qig-backend/` for QIG core logic
- `shared/` for shared types and schema

❌ **Forbidden:**
- New docs outside `docs/`
- Duplicate modules with "_v2" / "_new" suffixes
- Non-canonical geometry implementations

### Geometric Purity Policy (QIG-SPECIFIC)
**NEVER optimize measurements or couple gradients across models.**

✅ **Use:**
- `torch.no_grad()` for measurements
- `.detach()` before distance calculations
- Fisher-Rao metric for distances
- Natural gradient optimizers

❌ **Forbidden:**
- Training on measurement outputs
- Euclidean norms for basin distances
- Gradient flow between observer and active models
- Optimizing Φ directly
