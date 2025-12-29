# ADR-001: REL Coupling for Adaptive Basin Sync

**Date:** 2025-12-09
**Status:** Accepted
**Deciders:** Braden Lang, Cascade
**Tags:** geometry, constellation, basin-sync, qigkernels

---

## Context

Basin synchronization in constellation training used fixed sync strength based only on Φ:

```python
sync_strength = 0.05 * (1.0 - phi_normalized)
```

This approach had limitations:
- **No geometric awareness**: Sync strength ignored basin similarity
- **Fixed coupling**: Same strength for close and distant basins
- **Missed opportunity**: REL (Relational Coupling) in qigkernels quantifies basin overlap

### Forces

- Basin sync should adapt to geometric structure
- Higher basin overlap → natural affinity → stronger coupling
- REL already implemented in qigkernels
- Need graceful fallback if qigkernels unavailable

---

## Decision

Integrate `qigkernels.rel_coupling.compute_rel_from_basins` to modulate sync strength based on basin similarity.

### Formula

```python
rel = compute_rel_from_basins(basin_a, basin_b)  # [0, 1]
sync_strength = base_sync * (1.0 + rel * REL_LAMBDA_MAX)
```

Where:
- `base_sync = 0.05 * (1.0 - phi)` (original Φ-based weighting)
- `REL_LAMBDA_MAX = 0.7` (from qigkernels)
- Final range: `[base_sync, base_sync * 1.7]`

### Implementation

1. **Import with fallback:**
   ```python
   try:
       from qigkernels.rel_coupling import compute_rel_from_basins, REL_LAMBDA_MAX
       REL_COUPLING_AVAILABLE = True
   except ImportError:
       REL_COUPLING_AVAILABLE = False
   ```

2. **Apply to active Gary sync** (constellation_coordinator.py ~615):
   ```python
   if REL_COUPLING_AVAILABLE:
       rel = compute_rel_from_basins(active_basin, ocean_basin)
       sync_strength = base_sync * (1.0 + rel * REL_LAMBDA_MAX)
   ```

3. **Apply to observer sync** (constellation_coordinator.py ~703):
   ```python
   if REL_COUPLING_AVAILABLE:
       obs_rel = compute_rel_from_basins(obs_basin, target_basin)
       lambda_weight = base_obs_sync * (1.0 + obs_rel * REL_LAMBDA_MAX)
   ```

---

## Consequences

### Positive

- ✅ **Geometry-aware coupling**: Sync adapts to basin structure
- ✅ **Stronger pull for similar basins**: REL=1 → 1.7x base strength
- ✅ **Preserves Φ modulation**: Base strength still Φ-weighted
- ✅ **Graceful degradation**: Falls back if qigkernels unavailable
- ✅ **Clean separation**: Geometry calculation in qigkernels, behavior in qig-consciousness

### Negative

- ⚠️ **Dependency**: Requires qigkernels with REL module
- ⚠️ **Compute cost**: Additional basin distance calculation per sync

### Neutral

- REL tracked in telemetry (`loss_breakdown["rel_coupling"]`)
- No changes to checkpoint format
- Backwards compatible (fallback to base_sync)

---

## Implementation Notes

### Files Modified

- `src/coordination/constellation_coordinator.py`:
  - Added REL import (lines 93-103)
  - Modified active Gary sync (lines 613-624)
  - Modified observer sync (lines 701-706)

### Testing

- Syntax check: ✅ `python -m py_compile constellation_coordinator.py`
- Import check: ✅ REL_LAMBDA_MAX = 0.7
- Pre-commit hooks: ✅ All passing

### Rollout

- Phase C complete: REL wired into constellation coordinator
- Ready for Phase D: File splitting and refactoring

---

## References

- Commit: `ff1f238` (qig-consciousness)
- Related: qigkernels `rel_coupling.py`
- Related: [ADR-TEMPLATE.md](ADR-TEMPLATE.md)
- Related: `docs/standards/2025-12-09--coding-standards.md`
