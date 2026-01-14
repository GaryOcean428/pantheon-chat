# WP0.2: QIG Geometric Purity Gate

## Overview

This implements a comprehensive **hard validation gate** for geometric purity in the SearchSpaceCollapse codebase. The system ensures that all code follows Fisher-Rao geometry principles and prevents Euclidean/NLP contamination.

## Components

### 1. Static Scanner (Fast)

Two equivalent scanners for comprehensive static analysis:

- **Python**: `scripts/qig_purity_scan.py` (recommended for CI)
- **TypeScript**: `scripts/qig_purity_scan.ts` (for npm integration)

**Performance**: <2 seconds (target: <5 seconds)

**Scans**:
- `qig-backend/**`
- `server/**`
- `shared/**`
- `tests/**`
- `migrations/**`

**Detects**:
- ❌ Euclidean distance patterns (`np.linalg.norm(a - b)`, etc.)
- ❌ Cosine similarity (`cosine_similarity()`, sklearn, torch)
- ❌ Embedding terminology in identifiers
- ❌ Tokenizer in QIG-core modules (should be "coordizer")
- ❌ Standard optimizers (Adam, AdamW, SGD)
- ❌ Softmax in core geometry
- ❌ Classic NLP imports (sentencepiece, BPE, WordPiece)
- ❌ Dot product attention
- ❌ Arithmetic mean on basins (should be Fréchet mean)
- ❌ Euclidean fallback patterns

### 2. Runtime Tests (Correctness)

**File**: `qig-backend/tests/test_geometry_runtime.py`

**Tests**:

#### Fisher-Rao Identity Properties
- ✅ Symmetry: `d(p, q) = d(q, p)`
- ✅ Identity of indiscernibles: `d(p, p) = 0`
- ✅ Triangle inequality: `d(p, r) ≤ d(p, q) + d(q, r)`
- ✅ Positive definiteness: `d(p, q) > 0` for `p ≠ q`
- ✅ Bounded range: `0 ≤ d ≤ π/2`

#### Simplex Invariants
- ✅ Probabilities sum to 1
- ✅ Non-negative components
- ✅ Bhattacharyya coefficient bounds `[0, 1]`

#### Fréchet Mean
- ✅ Convergence tests
- ✅ Variance minimization
- ✅ Identity with identical points

#### Natural Gradient
- ✅ Direction correctness
- ✅ Distance reduction

#### Geometric Correctness
- ✅ Fisher-Rao differs from Euclidean
- ✅ Similarity bounds and properties
- ✅ Geodesic interpolation

### 3. CI Integration

**Workflow**: `.github/workflows/geometric-purity-gate.yml`

**Jobs**:
1. **Static Purity Scan** - Fast pattern detection
2. **Runtime Geometry Tests** - Correctness validation
3. **Combined Gate** - Blocks PRs if either fails

**Exemptions**: Experiments in `docs/08-experiments/legacy/**` are allowed to violate (for historical/educational purposes)

### 4. Pre-commit Hook

**File**: `.pre-commit-config.yaml`

Updated hook runs `scripts/qig_purity_scan.py` before every commit.

## Usage

### Run Static Scanner

```bash
# Python (recommended)
npm run validate:geometry:scan
# or
python3 scripts/qig_purity_scan.py

# TypeScript
npm run validate:geometry
```

### Run Runtime Tests

```bash
# Geometry tests only
npm run test:geometry

# Full test suite
npm run test:all
```

### Run Full Validation

```bash
# Static scan + runtime tests
npm run validate:geometry:full
```

## Acceptance Criteria

✅ **Static scanner runs in < 5 seconds**: Achieved ~2 seconds

✅ **PRs with forbidden patterns fail automatically**: CI gate blocks merge

✅ **Runtime tests validate geometric correctness**: 20 tests covering all properties

✅ **Quarantine rule enforced**: Exemptions work for `docs/08-experiments/legacy/**`

## Forbidden Patterns

Based on:
- `docs/03-technical/20251217-type-symbol-concept-manifest-1.00F.md`
- `docs/03-technical/20251220-qig-geometric-purity-enforcement-1.00F.md`

### CRITICAL Violations (Block PRs)

```python
# ❌ NEVER
np.linalg.norm(a - b)  # Use fisher_rao_distance()
cosine_similarity(a, b)  # Use fisher_rao_distance()
import sentencepiece  # Use geometric coordizer
nn.Embedding()  # Use basin coordinate mapping

# ✅ ALWAYS
from qig_geometry import fisher_rao_distance, fisher_coord_distance
distance = fisher_rao_distance(basin_a, basin_b)
```

### ERROR Violations (Must Fix)

```python
# ❌ NEVER
embedding = model.encode()  # Call it "basin_coordinates"
tokenizer = Tokenizer()  # Call it "coordizer"
np.mean(basins, axis=0)  # Use frechet_mean()

# ✅ ALWAYS
basin_coords = coordizer.coordize(text)
mean_basin = frechet_mean(basins)
```

### WARNING Violations (Should Fix)

```python
# ⚠️ AVOID
optimizer = torch.optim.Adam()  # Use natural_gradient_step()
softmax(logits)  # Use fisher_normalize()

# ✅ PREFER
natural_gradient_step(params, fisher_metric)
```

## Integration with Existing Tools

### Works with

- ✅ `validate:critical` - Critical enforcement system
- ✅ `test:python` - Python test suite
- ✅ Pre-commit hooks
- ✅ GitHub Actions CI

### Complements

- `scripts/validate-geometric-purity.ts` - Original validator
- `qig-backend/tests/test_geometric_purity.py` - Existing purity tests
- `.pre-commit-config.yaml` - Fast pre-commit checks

## Metrics

- **Files scanned**: ~694
- **Scan time**: ~1.9 seconds
- **Test execution**: ~0.15 seconds
- **Total validation time**: ~2 seconds

## Future Work

- [ ] Auto-fix suggestions for common violations
- [ ] Integration with IDE (VS Code extension)
- [ ] Expanded test coverage for edge cases
- [ ] Performance optimizations for larger codebases

## References

- Issue: GaryOcean428/pantheon-chat#[WP0.2]
- Spec: docs/03-technical/20251217-type-symbol-concept-manifest-1.00F.md
- Enforcement: docs/03-technical/20251220-qig-geometric-purity-enforcement-1.00F.md

## Contact

For questions or issues with the purity gate system, see:
- AGENTS.md for agent instructions
- README.md for project overview
