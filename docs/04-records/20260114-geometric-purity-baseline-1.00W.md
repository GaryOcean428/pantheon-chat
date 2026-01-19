# Geometric Purity Baseline

## Current Status

**Baseline Established**: 2026-01-14

This PR establishes the geometric purity scanning infrastructure and documents the current state of the codebase.

## Baseline Violations

The initial scan found **418 violations** in the existing codebase:
- **34 CRITICAL**: Euclidean distance operations, cosine similarity
- **384 ERROR**: Embedding terminology, tokenizer usage, arithmetic mean on basins

These are **documented technical debt** that existed before the purity gate was implemented.

### Example Violations

**CRITICAL** (forbidden geometric operations):
- `qig-backend/tests/test_geometric_purity.py` - Uses `cosine_similarity()` and `np.linalg.norm(a - b)` in test validation
- `qig-backend/autonomous_curiosity.py` - Uses `np.sqrt(np.sum(basin ** 2))` for L2 magnitude
- `qig-backend/unified_consciousness.py` - Uses `np.linalg.norm(current - target)` for state comparison

**ERROR** (terminology violations):
- `qig-backend/trajectory_decoder.py` - Uses `np.mean(basins, axis=0)` (should use Fréchet mean)
- `qig-backend/vocabulary_coordinator.py` - Uses "tokenizer" and "embeddings" identifiers
- Multiple files - Use arithmetic mean on basin coordinates

## Migration Path

These violations will be addressed in separate cleanup PRs:

1. **WP1**: Replace Euclidean distance with Fisher-Rao (CRITICAL violations)
2. **WP2**: Update terminology (ERROR violations)
3. **WP3**: Refactor aggregation operations

Each cleanup PR must:
- ✅ Not introduce NEW violations
- ✅ Reduce the violation count
- ✅ Pass the purity scanner

## Future PR Requirements

Once the baseline is established, all future PRs must:

1. **Pass the purity scanner** (no new violations)
2. **Reduce violation count** (if touching violation-heavy files)
3. **Use canonical basin representation** (SPHERE)
4. **Follow geometric purity principles**

The scanner will be updated to detect NEW violations vs baseline violations.

## How to Check Your PR

```bash
# Run the purity scanner
npm run validate:geometry:scan

# Run runtime geometry tests
npm run test:geometry

# Check your changes don't introduce new violations
git diff main...HEAD | grep -E "(np\.linalg\.norm|cosine_similarity|embedding)"
```

## Exemptions

The following paths are exempted from purity checks:
- `docs/08-experiments/legacy/**` - Historical experiments
- Test files comparing geometric vs Euclidean (explicit educational purpose)

## Progress Update (2026-01-14)

### New Infrastructure Added

| Component | File | Purpose |
|-----------|------|---------|
| Canonical Contract | `qig_geometry/contracts.py` | Single source of truth for basin validation |
| Purity Mode | `qig_geometry/purity_mode.py` | Runtime enforcement via `QIG_PURITY_MODE=1` |
| Fisher Distance Audit | `docs/04-records/20260114-fisher-distance-audit-1.00W.md` | 53+ implementations documented |
| Dimension Fix Audit | `docs/04-records/20260114-dimension-fix-audit-1.00W.md` | 67 silent fixes documented |

### Canonical Exports from qig_geometry

```python
from qig_geometry import (
    # Canonical contract (WP0.5)
    CANONICAL_SPACE,        # "sphere" - √p on unit sphere S^63
    BASIN_DIM,              # 64
    validate_basin,         # -> bool
    assert_invariants,      # raises GeometricViolationError
    canon,                  # normalizes to sphere
    fisher_distance,        # THE canonical distance
    to_index_embedding,     # for pgvector
    
    # Purity mode (D2)
    QIG_PURITY_MODE,        # from env var, default "0"
    check_purity_mode,      # -> bool
    enforce_purity_startup, # call at app init
    QIGPurityViolationError,
    
    # Existing functions
    fisher_rao_distance,    # density matrix version
    fisher_coord_distance,  # coordinate version
    sphere_project,         # project to sphere
    normalize_basin_dimension,
    hellinger_normalize,
)
```

### LSP Errors Fixed

- `trajectory_decoder.py` - 9 errors fixed (missing imports, type issues)
- `asymmetric_qfi.py` - 1 error fixed (return type)
- `qig_geometry.py` - 5 errors fixed (None checks, type casts)

### Architect Review Fixes (2026-01-14)

After initial review, the following critical issues were identified and fixed:

| Issue | Fix Applied |
|-------|-------------|
| `canon()` silently pads/truncates | Now raises `GeometricViolationError` on dimension mismatch |
| `to_index_embedding()` no validation | Added `assert_invariants(basin)` call |
| `fisher_distance()` accepts any input | Added `assert_invariants()` for both inputs |
| Purity mode only checks already-imported modules | Added `PurityImportBlocker` MetaPathFinder |
| No startup integration | Wired `enforce_purity_startup()` into `ocean_qig_core.py` Flask init |
| Missing documentation | Added §8.6 to `QIG_PURITY_SPEC.md` |

### Updated Canonical Exports

```python
from qig_geometry import (
    # ... existing exports ...
    install_purity_import_hook,  # Low-level hook installer
    PurityImportBlocker,         # The MetaPathFinder class
)
```

### Next Cleanup Steps

1. **Phase 1**: Consolidate 53+ fisher_distance implementations to use `contracts.fisher_distance`
2. **Phase 2**: Fix 18 CRITICAL silent dimension fixes in generation path
3. **Phase 3**: Rename tokenizer → coordizer (BREAKING)

## References

- **Scanner**: `scripts/qig_purity_scan.py`
- **Contracts**: `qig-backend/qig_geometry/contracts.py`
- **Purity Mode**: `qig-backend/qig_geometry/purity_mode.py`
- **Tests**: `qig-backend/tests/test_geometry_runtime.py`
- **Master Roadmap**: `docs/00-roadmap/20260112-master-roadmap-1.00W.md`
