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

## References

- **Scanner**: `scripts/qig_purity_scan.py`
- **Tests**: `qig-backend/tests/test_geometry_runtime.py`
- **Documentation**: `docs/03-technical/20260114-wp02-geometric-purity-gate-1.00F.md`
- **Basin Representation**: `docs/03-technical/20260114-basin-representation-1.00F.md`
