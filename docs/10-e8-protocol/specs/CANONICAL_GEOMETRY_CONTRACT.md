# CANONICAL GEOMETRY CONTRACT

**Status:** 🔒 FROZEN (Single Source of Truth)  
**Version:** 1.0  
**Date:** 2026-01-23  
**Authority:** E8 Protocol v4.0 Universal Purity Specification

---

## PURPOSE

This document serves as the **single source of truth** for all geometric operations in the Pantheon-Chat QIG system. It:

1. **Inventories** all canonical geometry functions
2. **Defines** forbidden patterns that violate geometric purity
3. **Specifies** required patterns for E8 Protocol v4.0 compliance
4. **Establishes** the canonical representation (SIMPLEX ONLY)

**All code must conform to this contract.** Violations are enforced via CI purity gates.

---

## CANONICAL REPRESENTATION

### Simplex (Probability Manifold Δ⁶³)

**SINGLE CANONICAL FORM:** Probability distributions on the simplex

```python
# Storage Format
- Representation: Probability distributions on Δ^(D-1) where D=64
- Constraints: Σp_i = 1, p_i ≥ 0 for all i
- Dimension: 64D (E8 rank²)
- Range: Each p_i ∈ [0, 1]
```

**Rationale:**
- **Physics Domain:** κ* = 64.21 ± 0.92 validated on simplex
- **AI Semantic Domain:** κ* = 63.90 ± 0.50 validated on simplex
- Universal fixed point measured on probability manifolds
- Natural geometry for Fisher information metrics

### Migration History

**Pre-2026-01-15:** SPHERE representation (L2 norm=1, allows negatives)  
**Post-2026-01-15:** SIMPLEX representation (sum=1, non-negative)

**Key Changes:**
- Distance formula: `2*arccos(dot)` → `arccos(BC)`
- Range: `[0, π]` → `[0, π/2]`
- **All distance thresholds must be divided by 2**

---

## GEOMETRY FUNCTION INVENTORY

### 1. Fisher-Rao Distance

The **ONLY** permitted distance metric for basins.

#### Python Implementations

| Function | Location | Description |
|----------|----------|-------------|
| `fisher_rao_distance(p, q, eps)` | `qig_geometry/canonical.py:174` | **PRIMARY - Canonical implementation** |
| `bhattacharyya(p, q, eps)` | `qig_geometry/canonical.py:127` | Bhattacharyya coefficient: BC = Σ√(p_i * q_i) |
| `fisher_similarity(p, q, eps)` | `qig_geometry/canonical.py:206` | Similarity score [0,1] = 1 - 2*d_FR/π |
| `batch_fisher_rao_distance(query, candidates)` | `qig_geometry/geometry_simplex.py:297` | Vectorized batch distance |
| `fisher_rao_distance(basin_a, basin_b, metric, validate)` | `qig_core/geometric_primitives/canonical_fisher.py:70` | With metric tensor support |
| `fisher_rao_distance(state_a, state_b, metric, method)` | `qigkernels/geometry/distances.py:56` | Universal (supports density matrices) |
| `quantum_fidelity(rho1, rho2)` | `qigkernels/geometry/distances.py:22` | Quantum: F(ρ₁, ρ₂) = Tr(√(√ρ₁ ρ₂ √ρ₁))² |

#### TypeScript Implementations

| Function | Location | Description |
|----------|----------|-------------|
| `fisherRaoDistance(pIn, qIn)` | `shared/qig/geometry_simplex.ts:91` | TS canonical port |

**Formula:**
```python
d_FR(p, q) = arccos(Σ√(p_i * q_i))
```

**Range:** `[0, π/2]`

**NOT:** `[0, π]` (legacy Hellinger with factor of 2)

---

### 2. Geodesic Interpolation

Interpolation along Fisher-Rao geodesics (NOT linear interpolation).

#### Python Implementations

| Function | Location | Description |
|----------|----------|-------------|
| `geodesic_interpolation_simplex(p_in, q_in, t)` | `qig_geometry/geometry_simplex.py:89` | **PRIMARY - SLERP in sqrt-space** |
| `geodesic_toward(source, target, fraction, eps)` | `qig_geometry/canonical.py:348` | Move fraction of distance along geodesic |
| `geodesic_interpolate(start, end, t, metric)` | `qig_core/geometric_primitives/canonical_fisher.py:192` | With metric tensor support |
| `geodesic_interpolation(start, end, t)` | `qig_geometry/__init__.py` | Module-level wrapper |
| `geodesic_interpolation(start, end, t)` | `olympus/geometric_utils.py` | Standalone implementation |

#### TypeScript Implementations

| Function | Location | Description |
|----------|----------|-------------|
| `geodesicInterpolationSimplex(pIn, qIn, t)` | `shared/qig/geometry_simplex.ts` | TS canonical port |

**Algorithm:** SLERP (Spherical Linear Interpolation) in sqrt-space
1. Convert to Hellinger coordinates: `√p, √q`
2. Perform SLERP in sqrt-space
3. Square back to simplex: `result²`

**NOT:** Linear interpolation `(1-t)*p + t*q` (wrong manifold)

---

### 3. Simplex Conversion Functions

Canonical pathway to simplex representation.

#### Python Implementations

| Function | Location | Description |
|----------|----------|-------------|
| `to_simplex(basin, from_repr, eps, strict)` | `qig_geometry/representation.py:95` | **PRIMARY - Canonical conversion** |
| `to_simplex_prob(v, eps)` | `qig_geometry/geometry_simplex.py:23` | Direct positive renormalization |
| `to_simplex(p)` | `qig_geometry/geometry_ops.py` | Simplified: abs + normalize |
| `simplex_normalize(p, eps)` | `qig_geometry/representation.py:421` | Renormalize existing simplex |
| `amplitude_to_simplex(amplitude, eps)` | `qig_geometry/representation.py:396` | Born rule: p_i = |amplitude_i|² |
| `hellinger_to_simplex(h, eps)` | `qig_geometry/representation.py:469` | Convert sqrt-space: p_i = h_i² |
| `enforce_canonical(basin)` | `qig_geometry/representation.py:375` | Force to SIMPLEX form |
| `fisher_normalize(basin)` | `qig_geometry/representation.py:384` | Alias for to_simplex() |

#### TypeScript Implementations

| Function | Location | Description |
|----------|----------|-------------|
| `toSimplexProb(v, eps)` | `shared/qig/geometry_simplex.ts:28` | TS canonical port |

**Formula:**
```python
# Positive renormalization
p = np.abs(v)
p = p / (p.sum() + eps)
```

**Constraints:**
- `Σp_i = 1` (within tolerance)
- `p_i ≥ 0` for all i
- All values finite (no NaN/inf)

---

### 4. Fréchet Mean (Geometric Mean)

Riemannian center of mass on the Fisher manifold.

#### Python Implementations

| Function | Location | Description |
|----------|----------|-------------|
| `frechet_mean(basins, weights, max_iter, tolerance, eps)` | `qig_geometry/canonical.py:410` | **PRIMARY - Iterative Riemannian gradient descent** |
| `geodesic_mean_simplex(distributions, weights, max_iter, tolerance)` | `qig_geometry/geometry_simplex.py:158` | Same algorithm with convergence improvements |
| `simplex_mean_sqrt_space(distributions, weights)` | `qig_geometry/geometry_simplex.py:338` | **Fast closed-form approximation** (sqrt-space) |
| `weighted_simplex_mean(distributions, weights)` | `qig_geometry/geometry_simplex.py:409` | Alias for sqrt-space mean |
| `frechet_mean(basins)` | `qig_geometry/geometry_ops.py` | Simplified sqrt-space version |

#### TypeScript Implementations

| Function | Location | Description |
|----------|----------|-------------|
| `geodesicMeanSimplex(distributions, weights, maxIter, tolerance)` | `shared/qig/geometry_simplex.ts` | TS iterative version |

**Algorithms:**
1. **Iterative (Primary):** Riemannian gradient descent with adaptive step size
2. **Closed-form (Fast):** Average in sqrt-space, then square back to simplex

**NOT:** Arithmetic mean `np.mean(basins, axis=0)` (wrong manifold)

---

### 5. Validation & Assertion Functions

Enforce simplex constraints at module boundaries.

#### Python Implementations

| Function | Location | Description |
|----------|----------|-------------|
| `validate_basin(basin, expected_repr, tolerance)` | `qig_geometry/representation.py:264` | **PRIMARY validation gate** |
| `validate_simplex(basin, tolerance)` | `qig_geometry/representation.py:344` | Alias for simplex validation |
| `validate_simplex(p, tolerance)` | `qig_geometry/geometry_simplex.py:53` | Returns (bool, reason) tuple |
| `validate_basin(basin)` | `qig_geometry/contracts.py:59` | Dimension + constraint checks |
| `validate_basin_detailed(basin)` | `qig_geometry/contracts.py` | Detailed error messages |
| `validate_basin(basin, expected_dim, require_probability)` | `qig_core/geometric_primitives/canonical_fisher.py:291` | With optional probability requirement |
| `assert_invariants(basin)` | `qig_geometry/contracts.py` | Raises GeometricViolationError if invalid |

#### TypeScript Implementations

| Function | Location | Description |
|----------|----------|-------------|
| `validateSimplex(p, tolerance)` | `shared/qig/geometry_simplex.ts:52` | TS validation port |

**Checks:**
- Dimension = 64 (BASIN_DIM)
- All values finite (no NaN/inf)
- All values ≥ 0 (non-negative)
- Sum ≈ 1.0 (within tolerance)

---

### 6. Coordinate Transformation Functions

Explicit conversions between coordinate charts (NOT auto-detect).

#### Python Implementations

| Function | Location | Description |
|----------|----------|-------------|
| `sqrt_map(p, eps)` | `qig_geometry/canonical.py:70` | Simplex → Hellinger: √p |
| `unsqrt_map(x, eps)` | `qig_geometry/canonical.py:99` | Hellinger → Simplex: x² |
| `log_map(p, base, eps)` | `qig_geometry/canonical.py:230` | Simplex → Tangent space |
| `exp_map(v, base, eps)` | `qig_geometry/canonical.py:275` | Tangent space → Simplex |

#### TypeScript Implementations

_Note: sqrt mapping is performed inline within `geodesicInterpolationSimplex` (line 152) for internal computation only. No standalone export._

**Usage:**
```python
# ✅ CORRECT: Explicit coordinate chart
h = sqrt_map(p)           # To Hellinger for computation
result = compute_in_hellinger(h)
p_result = unsqrt_map(result)  # Back to simplex

# ❌ WRONG: Auto-detect representation
if is_hellinger(basin):  # FORBIDDEN
    ...
```

---

### 7. Nearest Neighbor / Search Functions

Fisher-Rao based similarity search.

#### Python Implementations

| Function | Location | Description |
|----------|----------|-------------|
| `find_nearest_simplex(query, candidates, k)` | `qig_geometry/geometry_simplex.py:314` | k-NN using Fisher-Rao |
| `find_nearest_basins(query_basin, candidates, k, metric)` | `qig_core/geometric_primitives/canonical_fisher.py:265` | With metric tensor |

#### TypeScript Implementations

| Function | Location | Description |
|----------|----------|-------------|
| `findNearestSimplex(query, candidates, k)` | `shared/qig/geometry_simplex.ts` | TS k-NN port |

**Algorithm:** Two-step retrieval
1. Approximate search (pgvector HNSW for scale)
2. Fisher-Rao re-rank top candidates

---

### 8. Utility Functions

Supporting geometric operations.

#### Python Implementations

| Function | Location | Description |
|----------|----------|-------------|
| `bhattacharyya_coefficient(p, q)` | `qig_geometry/geometry_ops.py` | BC computation |
| `fisher_metric_tensor(probabilities)` | `qig_core/geometric_primitives/fisher_metric.py:18` | G_ij = δ_ij / p_i |
| `normalize_basin_dimension(basin, target_dim)` | `qig_geometry/__init__.py` | Resize basin |
| `estimate_manifold_curvature(path)` | `qig_geometry/__init__.py` | Curvature along path |
| `basin_magnitude(basin)` | `qig_geometry/__init__.py` | Compute magnitude |
| `basin_diversity(basin)` | `qig_geometry/__init__.py` | Entropy-like diversity |
| `geodesic_distance(point_a, point_b, metric)` | `qigkernels/geometry/distances.py:153` | Alias for fisher_rao_distance |

---

## FORBIDDEN PATTERNS

These patterns **VIOLATE** geometric purity and are **FORBIDDEN** in all QIG code.

### 1. Euclidean Distance on Simplex

```python
# ❌ FORBIDDEN - Violates manifold structure
d_wrong = np.linalg.norm(basin_a - basin_b)
d_wrong = np.sqrt(np.sum((basin_a - basin_b) ** 2))

# ✅ REQUIRED - Use Fisher-Rao distance
from qig_geometry import fisher_rao_distance
d_correct = fisher_rao_distance(basin_a, basin_b)
```

**Why forbidden:** Euclidean distance does not respect the information geometry of probability distributions. It treats the simplex as a flat Euclidean space, which is geometrically incorrect.

---

### 2. Cosine Similarity on Simplex

```python
# ❌ FORBIDDEN - Wrong geometry
from sklearn.metrics.pairwise import cosine_similarity
sim_wrong = cosine_similarity(basin_a.reshape(1, -1), basin_b.reshape(1, -1))[0, 0]

# ❌ ALSO FORBIDDEN
sim_wrong = np.dot(basin_a, basin_b) / (np.linalg.norm(basin_a) * np.linalg.norm(basin_b))

# ✅ REQUIRED - Use Fisher similarity
from qig_geometry import fisher_similarity
sim_correct = fisher_similarity(basin_a, basin_b)
```

**Why forbidden:** Cosine similarity is designed for vector embeddings in Euclidean space, not probability distributions on the simplex manifold.

---

### 3. Arithmetic Mean on Simplex

```python
# ❌ FORBIDDEN - Wrong manifold averaging
mean_wrong = np.mean(basins, axis=0)
mean_wrong = sum(basins) / len(basins)

# ❌ ALSO FORBIDDEN - Weighted arithmetic mean
mean_wrong = np.average(basins, axis=0, weights=weights)

# ✅ REQUIRED - Use Fréchet mean (geometric mean)
from qig_geometry import frechet_mean
mean_correct = frechet_mean(basins, weights=weights)
```

**Why forbidden:** Arithmetic averaging does not preserve the simplex structure (may produce invalid probabilities) and does not follow geodesics on the Fisher manifold.

---

### 4. Auto-Detect Representation

```python
# ❌ FORBIDDEN - Auto-detect causes silent drift
def to_simplex(basin):
    if is_sphere(basin):
        return sphere_to_simplex(basin)
    elif is_hellinger(basin):
        return hellinger_to_simplex(basin)
    else:
        return basin  # Already simplex?

# ✅ REQUIRED - Explicit representation parameter
from qig_geometry import to_simplex, BasinRepresentation

# Caller MUST know and specify representation
basin_simplex = to_simplex(basin, from_repr=BasinRepresentation.SPHERE)
```

**Why forbidden:** Auto-detection leads to ambiguity, silent errors, and geometric drift. The caller must explicitly declare the input representation.

---

### 5. Linear Interpolation on Simplex

```python
# ❌ FORBIDDEN - Not a geodesic
interpolated_wrong = (1 - t) * basin_a + t * basin_b

# ❌ ALSO FORBIDDEN - Weighted linear blend
interpolated_wrong = alpha * basin_a + (1 - alpha) * basin_b

# ✅ REQUIRED - Use geodesic interpolation
from qig_geometry import geodesic_interpolation
interpolated_correct = geodesic_interpolation(basin_a, basin_b, t=0.5)
```

**Why forbidden:** Linear interpolation does not follow geodesics on the Fisher manifold. It may leave the simplex and violate probability constraints.

---

### 6. Dot Product Ranking on Simplex

```python
# ❌ FORBIDDEN - Dot product is not a valid similarity
scores_wrong = [np.dot(query, candidate) for candidate in candidates]
best_wrong = candidates[np.argmax(scores_wrong)]

# ✅ REQUIRED - Use Fisher-Rao distance
from qig_geometry import fisher_rao_distance
distances = [fisher_rao_distance(query, c) for c in candidates]
best_correct = candidates[np.argmin(distances)]  # Minimum distance
```

**Why forbidden:** Dot product does not respect the information geometry of the simplex manifold.

---

### 7. Factor of 2 in Fisher-Rao Distance (Legacy)

```python
# ❌ FORBIDDEN - Legacy Hellinger with factor of 2
from qig_geometry import bhattacharyya
bc = bhattacharyya(p, q)
d_wrong = 2 * np.arccos(bc)  # Range [0, π] - OUTDATED

# ✅ REQUIRED - Direct Fisher-Rao (no factor of 2)
from qig_geometry import fisher_rao_distance
d_correct = fisher_rao_distance(p, q)  # Range [0, π/2] - CURRENT
```

**Why forbidden:** The factor of 2 is legacy from the old Hellinger embedding. Current implementation uses direct Bhattacharyya coefficient without the factor.

**Migration:** All distance thresholds must be divided by 2 when migrating from [0, π] to [0, π/2] range.

---

### 8. External NLP in Generation Pipeline

```python
# ❌ FORBIDDEN - External NLP breaks geometric purity
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
pos_tags = [token.pos_ for token in doc]

# ❌ ALSO FORBIDDEN - External LLM calls in QIG core
import openai
response = openai.ChatCompletion.create(...)

# ✅ REQUIRED - QIG-native token role from Fisher neighborhoods
from qig_generation import derive_token_role_geometric
token_role = derive_token_role_geometric(basin, vocabulary)
```

**Why forbidden:** External NLP breaks QIG purity mode. All language structure must be learned geometrically from Fisher-Rao neighborhoods.

---

### 9. Direct INSERT into Vocabulary

```python
# ❌ FORBIDDEN - Bypasses QFI computation
cursor.execute(
    "INSERT INTO coordizer_vocabulary (token, basin) VALUES (%s, %s)",
    (token, basin)
)

# ✅ REQUIRED - Use canonical insert_token() pathway
from vocabulary import insert_token
insert_token(
    token=token,
    basin=basin,
    compute_qfi=True,  # Must compute QFI for generation eligibility
    validate_geometry=True
)
```

**Why forbidden:** Direct INSERT bypasses QFI integrity checks. All tokens must go through canonical insertion pathway.

---

## REQUIRED PATTERNS

These patterns are **MANDATORY** for E8 Protocol v4.0 compliance.

### 1. Fisher-Rao Distance for ALL Similarity

```python
# ✅ REQUIRED - Import from canonical module
from qig_geometry import fisher_rao_distance

# ✅ REQUIRED - Use for basin similarity
d = fisher_rao_distance(basin_a, basin_b)

# ✅ REQUIRED - Convert to similarity score if needed
from qig_geometry import fisher_similarity
sim = fisher_similarity(basin_a, basin_b)  # Returns [0, 1]
```

**Range:** `[0, π/2]`
- `0` = identical distributions
- `π/2` = orthogonal distributions

---

### 2. Geodesic Interpolation for Blending

```python
# ✅ REQUIRED - Use geodesic interpolation
from qig_geometry import geodesic_interpolation

# Blend two basins (t=0.5 is midpoint)
blended = geodesic_interpolation(basin_a, basin_b, t=0.5)

# Move fraction of distance along geodesic
from qig_geometry.canonical import geodesic_toward
pulled = geodesic_toward(source=basin, target=attractor, fraction=0.3)
```

**Properties:**
- Follows geodesics on Fisher manifold
- Preserves simplex constraints
- Respects information geometry

---

### 3. Fréchet Mean for Aggregation

```python
# ✅ REQUIRED - Use Fréchet mean for basin aggregation
from qig_geometry import frechet_mean

# Unweighted geometric mean
center = frechet_mean(basins)

# Weighted geometric mean
center = frechet_mean(basins, weights=weights)
```

**Properties:**
- Minimizes sum of squared Fisher-Rao distances
- Respects manifold structure
- Natural center of mass on Fisher manifold

---

### 4. Simplex Assertion at Boundaries

```python
# ✅ REQUIRED - Validate at module boundaries
from qig_geometry import assert_invariants

def store_basin(basin: np.ndarray):
    """Store basin in database."""
    # Assert invariants before database write
    assert_invariants(basin)
    
    # Proceed with storage
    db.insert(basin)
```

**Where to validate:**
- Before database writes
- After external function calls
- At module boundaries
- After numerical operations that may drift

---

### 5. Explicit Representation Conversion

```python
# ✅ REQUIRED - Explicit from_repr parameter
from qig_geometry import to_simplex, BasinRepresentation

# Caller declares input representation
basin_simplex = to_simplex(
    basin,
    from_repr=BasinRepresentation.SPHERE,  # Explicit
    strict=True  # Enforce validation
)
```

**NO auto-detect:** Caller must know and declare representation.

---

### 6. Two-Step Retrieval Pattern

```python
# ✅ REQUIRED - Approximate search + Fisher re-rank
from qig_geometry import fisher_rao_distance

# Step 1: Approximate search (e.g., pgvector HNSW)
candidates = db.approximate_search(query, k=100)

# Step 2: Fisher-Rao re-rank
distances = [fisher_rao_distance(query, c.basin) for c in candidates]
sorted_idx = np.argsort(distances)
top_k = [candidates[i] for i in sorted_idx[:10]]
```

**Why required:** Approximate search provides scale, Fisher re-rank provides accuracy.

---

### 7. Canonical Constants from Single Source

```python
# ✅ REQUIRED - Import constants from canonical source
from qig_geometry.contracts import (
    CANONICAL_SPACE,  # = "simplex"
    BASIN_DIM,        # = 64
    NORM_TOLERANCE    # = 1e-5
)

# ✅ REQUIRED - Import physics constants
from qigkernels.physics_constants import (
    KAPPA_STAR,       # = 64.21 ± 0.92
    KAPPA_3,          # = 41.09 ± 0.59
    PHI_THRESHOLD,    # = 0.70
    BETA_PHYSICS_EMERGENCE,   # = 0.443 ± 0.04
    BETA_SEMANTIC_EMERGENCE   # = 0.267 ± 0.05
)
```

**Single source of truth:** All constants defined once, imported everywhere.

---

### 8. QFI Score for Generation Eligibility

```python
# ✅ REQUIRED - Compute QFI for all vocabulary tokens
from vocabulary import insert_token

insert_token(
    token="example",
    basin=basin,
    compute_qfi=True,        # REQUIRED for generation
    validate_geometry=True   # REQUIRED for purity
)

# ✅ REQUIRED - Filter by QFI in generation queries
vocabulary = db.query("""
    SELECT * FROM coordizer_vocabulary
    WHERE qfi_score IS NOT NULL
    AND is_generation_eligible = TRUE
""")
```

**Rule:** NO token without QFI score can be used in generation.

---

## E8 PROTOCOL v4.0 COMPLIANCE

### §0 Non-Negotiable Purity Rules

1. ✅ **Simplex-only canonical representation** (NO sphere, NO auto-detect)
2. ✅ **Fisher-Rao distance only** (NO cosine similarity, NO Euclidean)
3. ✅ **Geodesic operations** (NO linear interpolation, NO arithmetic mean)
4. ✅ **QFI integrity** (ALL tokens must have QFI score for generation)
5. ✅ **NO external NLP** (spacy, nltk) in generation pipeline
6. ✅ **NO external LLMs** (openai, anthropic) in QIG_PURITY_MODE

### §4 Implementation Phases

**Phase 1: Repo Truth + Invariants** ← THIS DOCUMENT
- [x] Inventory all geometry functions (50+ functions documented)
- [x] Create canonical geometry contract (this document)
- [ ] Search & remove forbidden patterns (ongoing CI enforcement)
- [ ] Generate purity scan reports

### Validation Commands

```bash
# Purity scan (run before commit)
python scripts/validate_geometry_purity.py

# QFI coverage check
python scripts/check_qfi_coverage.py

# Generation purity test
QIG_PURITY_MODE=true python qig-backend/test_generation_pipeline.py
```

---

## CROSS-REFERENCES

### Implementation Files

#### Primary Canonical Modules
- **Python:** `qig-backend/qig_geometry/` (canonical.py, geometry_simplex.py, contracts.py, representation.py)
- **TypeScript:** `shared/qig/geometry_simplex.ts`

#### Supporting Modules
- `qig-backend/qig_core/geometric_primitives/` (canonical_fisher.py, fisher_metric.py, geodesic.py)
- `qig-backend/qigkernels/geometry/` (distances.py)
- `server/fisher-vectorized.ts` (vectorized operations)
- `server/temporal-geometry.ts` (temporal geometry)
- `server/basin-geodesic-manager.ts` (geodesic management)

### Documentation

- **E8 Protocol v4.0:** `docs/10-e8-protocol/specifications/20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md`
- **E8 Blueprint:** `docs/10-e8-protocol/specifications/20260116-wp5-2-e8-implementation-blueprint-1.01W.md`
- **Ultra-Consciousness Protocol:** `docs/08-experiments/20251231-Ultra-Consciousness-Protocol-0.04F.md` (includes frozen facts)
- **Universal κ*:** `docs/08-experiments/20251228-Universal-kappa-star-discovery-0.01F.md`
- **QIG Purity Spec:** `qig_geometry/contracts.py` (inline documentation)

### Issue Specifications
- **QFI Integrity:** `docs/10-e8-protocol/issues/20260116-issue-01-qfi-integrity-gate-1.01W.md`
- **Simplex Purity:** `docs/10-e8-protocol/issues/20260116-issue-02-strict-simplex-representation-1.01W.md`
- **Native Skeleton:** `docs/10-e8-protocol/issues/20260116-issue-03-qig-native-skeleton-1.01W.md`
- **Vocabulary Cleanup:** `docs/10-e8-protocol/issues/20260119-issue-04-vocabulary-cleanup-garbage-tokens-1.00W.md`

---

## ENFORCEMENT

### CI Purity Gates

File: `.github/workflows/qig-purity-gate.yml`

**Checks:**
- ✅ NO `cosine_similarity` in QIG code
- ✅ NO `np.linalg.norm` on basin coordinates
- ✅ NO `np.dot` or `@` operator for basin similarity
- ✅ NO auto-detect representation
- ✅ NO direct INSERT into coordizer_vocabulary
- ✅ NO external NLP in generation pipeline

**Status:** ALL PRs must pass purity gate before merge.

### Pre-Commit Hooks

File: `.pre-commit-config.yaml`

**Hooks:**
- Geometry purity validation
- Forbidden pattern detection
- Simplex validation at boundaries

### Runtime Assertions

**Where enforced:**
- Module boundaries (`assert_invariants`)
- Database writes (`validate_basin`)
- After external calls (`validate_simplex`)
- Numerical operations with drift risk

---

## MIGRATION GUIDE

### From SPHERE to SIMPLEX (2026-01-15)

**Distance Range Change:**
```python
# OLD: [0, π] range
if distance < 0.8:  # OLD threshold
    ...

# NEW: [0, π/2] range (divide by 2)
if distance < 0.4:  # NEW threshold
    ...
```

**Distance Formula Change:**
```python
# OLD: Hellinger with factor of 2
bc = bhattacharyya(p, q)
d_old = 2 * np.arccos(bc)  # Range [0, π]

# NEW: Direct Fisher-Rao (no factor)
d_new = fisher_rao_distance(p, q)  # Range [0, π/2]
```

**Conversion Functions:**
```python
# Convert old thresholds to new
threshold_new = threshold_old / 2.0

# Convert old distances to new
distance_new = distance_old / 2.0
```

### From Auto-Detect to Explicit

**OLD Pattern:**
```python
# Auto-detect (FORBIDDEN)
basin_simplex = to_simplex(basin)  # Guesses representation
```

**NEW Pattern:**
```python
# Explicit (REQUIRED)
from qig_geometry import to_simplex, BasinRepresentation

basin_simplex = to_simplex(
    basin,
    from_repr=BasinRepresentation.SPHERE,  # Explicit
    strict=True
)
```

---

## VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-23 | Initial canonical geometry contract |

---

## AUTHORITY

This document is authoritative under:
- **E8 Protocol v4.0** Universal Purity Specification
- **Ultra-Consciousness Protocol** (docs/08-experiments/20251231-Ultra-Consciousness-Protocol-0.04F.md) - includes frozen facts and validation ground truth
- **Universal κ* = 64** Discovery (docs/08-experiments/20251228-Universal-kappa-star-discovery-0.01F.md)

**Status:** 🔒 FROZEN (changes require formal review and propagation to all implementation files)

---

**Last Updated:** 2026-01-23  
**Maintainer:** E8 Protocol Working Group  
**Contact:** GaryOcean428/pantheon-chat repository
