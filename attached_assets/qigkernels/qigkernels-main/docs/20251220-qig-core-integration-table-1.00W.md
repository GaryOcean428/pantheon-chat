# qig-core → qigkernels Integration Table

**Status:** CRITICAL FIXES COMPLETE
**Priority:** M13 on roadmap

## Identified Replacements

| qigkernels function | File | Issue | qig-core equivalent | Action |
|---------------------|------|-------|---------------------|--------|
| `basin_distance()` | basin.py:75 | Was L2 (Euclidean!) | `qig_core.fisher.fisher_distance` | DONE |
| `kappa_star=63.62` | basin_sync.py:127 | Was hardcoded | `qig_core.constants.KAPPA_STAR` | DONE |
| `compute_sync_strength()` | basin_sync.py:121 | Ad-hoc sync logic | Consider `qig_core.coordination.BasinSync` | DONE |
| - | - | No geodesic interpolation | `qig_core.geodesic.geodesic_interpolate` | ADD |
| - | - | No natural gradient | `qig_core.fisher.natural_gradient_step` | ADD |

## Critical Issue: Euclidean Distance

```python
# basin.py (FIXED - NOW USES FISHER-RAO)
def basin_distance(a: Tensor, b: Tensor, use_fisher: bool = True) -> Tensor:
    """Compute Fisher-Rao distance between basin signatures."""
    # Uses Bures approximation: d² = 2(1 - cos_sim)
    # Falls back to qig_core.fisher.fisher_distance if available
    ...  # GEOMETRIC PURITY RESTORED
```

## KAPPA_STAR Consolidation

Now uses `qigkernels/constants.py`:

- `KAPPA_STAR = 64.0` (E8 rank² = 8²)
- `KAPPA_3 = 41.09` (validated 3-layer)
- `KAPPA_PLATEAU = 63.62` (experimental plateau)

## Functions to Add from qig-core

1. **geodesic_interpolate** - For smooth basin transitions
2. **natural_gradient_step** - For Fisher-aware optimization
3. **QFISampler** - For geometric token sampling

## Integration Steps

1. [x] Add `qig-core` as dependency in pyproject.toml ✓
2. [x] Replace `basin_distance` with Fisher-Rao (Bures) ✓
3. [x] Create `constants.py` with KAPPA_STAR ✓
4. [x] Update `compute_sync_strength` to use E8 constants ✓
5. [ ] Add geodesic utilities where needed
6. [ ] Run tests to verify equivalence

## Notes

- qig-core is PRE-E8, may need updates for primitive awareness
- Fisher distance may need REL coupling tensor integration later
- Keep backward compatibility during transition
