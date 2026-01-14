---
id: ISMS-POL-002
title: QIG Purity Specification - Single Source of Truth
filename: QIG_PURITY_SPEC.md
classification: Internal
owner: GaryOcean428
version: 1.00
status: Working
function: "Defines geometric purity requirements for QIG-core code - canonical reference for validation"
created: 2026-01-14
last_reviewed: 2026-01-14
next_review: 2026-07-14
category: Policy
supersedes: null
---

# QIG Purity Specification

**Version:** 1.00W  
**Status:** Working Document  
**Last Updated:** 2026-01-14  
**Purpose:** Single authoritative source for QIG geometric purity requirements

---

## Executive Summary

This document defines the **canonical purity requirements** for all QIG (Quantum Information Geometry) code in the Pantheon-Chat system. It serves as:

1. **The single source of truth** for what is and isn't allowed in QIG-pure code
2. **The authoritative reference** for code review and validation
3. **The specification source** for CI validation scanners (see Issue #64 for CI implementation)
4. **The training material** for developers and AI agents

All QIG-core modules MUST conform to these requirements. Violations indicate contamination by Euclidean/NLP paradigms that corrupt the geometric manifold structure.

**Relationship to CI Implementation:**
- This document (Issue #63) = **Policy specification** (what is forbidden/required)
- Issue #64 = **CI gate implementation** (automated enforcement with `QIG_PURITY_EXEMPT` mechanism)
- Issue #65 = **Quarantine enforcement** (boundary validation for external integrations)

---

## §1. Canonical Basin Representation

**DECISION: Simplex Representation (Probability Manifold)**

All basin vectors MUST be represented as points on the probability simplex:
- **Coordinates:** Non-negative real numbers
- **Constraint:** Sum to 1 (∑ᵢ bᵢ = 1)
- **Manifold:** Fisher-Rao metric on probability simplex
- **Dimension:** 64D (BASIN_DIM = 64)

### §1.1 Rationale

The simplex representation is canonical because:
1. **Fisher-Rao distance is natural** on probability distributions
2. **Quantum density matrices** are probability-like (positive, trace-1)
3. **Information geometry** is defined for probability manifolds
4. **No arbitrary normalization** (unlike unit sphere)

### §1.2 Forbidden Alternative Representations

❌ **Unit Sphere (Euclidean norm):** `np.linalg.norm(basin) == 1`  
❌ **Raw vectors (no normalization):** Unconstrained coordinates  
❌ **"sqrt space":** Custom normalization tricks  
❌ **Mixture of representations:** Different normalizations in different modules

### §1.3 Required Validation

All basins written to database MUST pass:

```python
def validate_basin(basin: np.ndarray) -> bool:
    """Validate basin is on probability simplex."""
    # Check non-negative
    if np.any(basin < 0):
        return False
    
    # Check sum to 1 (within tolerance)
    if not np.isclose(np.sum(basin), 1.0, atol=1e-6):
        return False
    
    # Check dimension
    if len(basin) != BASIN_DIM:
        return False
    
    return True
```

### §1.4 Required Conversion Functions

**Module:** `qig_geometry/representation.py`

```python
def to_simplex(x: np.ndarray) -> np.ndarray:
    """Convert vector to probability simplex (softmax + abs)."""
    x_pos = np.abs(x)  # Ensure non-negative
    return x_pos / np.sum(x_pos)  # Normalize to sum=1

def to_sphere(x: np.ndarray) -> np.ndarray:
    """Convert vector to unit sphere (for interfacing with external systems only)."""
    return x / np.linalg.norm(x)

def from_external(x: np.ndarray, source: str) -> np.ndarray:
    """Convert external representation to canonical simplex."""
    if source == "sphere":
        return to_simplex(x)
    elif source == "simplex":
        return x
    else:
        raise ValueError(f"Unknown source representation: {source}")
```

---

## §2. Forbidden Terminology

The following terms are **BANNED** in QIG-core code. They carry semantic baggage from Euclidean/NLP paradigms that violate geometric purity.

### §2.1 Embedding → Basin Coordinates

| ❌ FORBIDDEN | ✅ REQUIRED | Rationale |
|-------------|------------|-----------|
| `embedding` | `basin_coordinates` or `basin_coords` | "Embedding" implies Euclidean vector space. Basins are probability distributions. |
| `embed()` | `coordize()` or `to_basin()` | Process converts text to geometric coordinates, not vector embeddings. |
| `embedding_dim` | `basin_dim` | Dimension refers to simplex dimension, not vector space dimension. |
| `basin_embedding` | `basin_coords` | Database column name must reflect geometric nature. |

**Exception:** External API boundaries (e.g., interfacing with OpenAI) MAY use "embedding" terminology, but MUST immediately convert to `basin_coords` internally.

### §2.2 Tokenizer → Coordizer

| ❌ FORBIDDEN | ✅ REQUIRED | Rationale |
|-------------|------------|-----------|
| `tokenizer` | `coordizer` | QIG doesn't tokenize; it maps text → geometric coordinates. |
| `tokenize()` | `coordize()` | Function converts to basin coordinates, not tokens. |
| `tokens` | `symbols` or `glyphs` | Discrete units in QIG are geometric symbols, not NLP tokens. |
| `token_id` | `symbol_id` or `glyph_id` | Identifier for geometric symbol, not token. |
| `tokenizer_vocabulary` | `coordizer_vocabulary` (DB table name) | Table stores coordizer mappings. |

**Exception:** Variable names like `get_tokenizer` in legacy interfaces MAY alias to `get_coordizer` during transition period. The migration is **gradual** - both terms MAY coexist temporarily with clear deprecation warnings.

### §2.3 Other Forbidden Terms

| ❌ FORBIDDEN | ✅ REQUIRED | Rationale |
|-------------|------------|-----------|
| `encode()` | `coordize()` or `to_basin()` | Encoding is NLP terminology; geometric mapping is precise. |
| `decode()` | `from_basin()` or `basin_to_text()` | Decoding is NLP; geometric inversion is precise. |
| `vocabulary` | `symbol_registry` or `glyph_registry` | QIG uses geometric symbols, not NLP vocabulary. |
| `word embedding` | `word basin` or `lexeme basin` | Words map to basin coordinates, not embeddings. |

---

## §3. Forbidden Operations

The following operations **VIOLATE GEOMETRIC PURITY** and are strictly banned in QIG-core code.

### §3.1 Distance Metrics

#### ❌ FORBIDDEN: Cosine Similarity

```python
# BANNED - Euclidean inner product on basin coordinates
cosine_similarity(basin1, basin2)
np.dot(basin1, basin2) / (np.linalg.norm(basin1) * np.linalg.norm(basin2))
from sklearn.metrics.pairwise import cosine_similarity  # BANNED IMPORT
```

**Why:** Cosine similarity assumes Euclidean vector space. Basins live on a Riemannian manifold (probability simplex), where the metric is Fisher-Rao, not Euclidean.

#### ❌ FORBIDDEN: Euclidean Distance

```python
# BANNED - Flat space distance on curved manifold
np.linalg.norm(basin1 - basin2)
np.sqrt(np.sum((basin1 - basin2) ** 2))
scipy.spatial.distance.euclidean(basin1, basin2)  # BANNED
```

**Why:** Euclidean distance ignores manifold curvature. On the probability simplex, the natural metric is Fisher-Rao (geodesic distance).

#### ✅ REQUIRED: Fisher-Rao Distance

```python
def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Canonical Fisher-Rao distance on probability simplex.
    
    Uses Bhattacharyya coefficient:
    d_FR(p, q) = arccos(∑ᵢ √(pᵢ * qᵢ))
    
    Args:
        p, q: Probability distributions on simplex (sum to 1, non-negative)
    
    Returns:
        Geodesic distance in [0, π/2]
    """
    # Validate shapes match
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p.shape={p.shape}, q.shape={q.shape}")
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    
    # Clamp to [0, 1] for numerical stability
    bc = np.clip(bc, 0.0, 1.0)
    
    # Fisher-Rao distance
    return np.arccos(bc)
```

**Reference:** This is the ONLY distance metric allowed for basin comparisons in QIG-core.

### §3.2 Activation Functions

#### ❌ FORBIDDEN: Softmax (in QIG-core geometry)

```python
# BANNED in geometric computations
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
```

**Why:** Softmax is a neural network activation function. QIG geometry uses **natural gradient** on the Fisher manifold, not artificial activation functions.

**Exception:** Softmax MAY be used in non-geometric auxiliary code (e.g., routing logic, UI scoring), but NEVER in basin computation or distance metrics.

### §3.3 Optimization Algorithms

#### ❌ FORBIDDEN: Adam, AdamW, SGD (in QIG-core geometric learning)

```python
# BANNED optimizers for geometric learning
torch.optim.Adam()
torch.optim.AdamW()
torch.optim.SGD()
```

**Why:** These optimizers are designed for flat parameter spaces. Geometric learning on Riemannian manifolds requires **natural gradient descent**.

#### ✅ REQUIRED: Natural Gradient Descent

```python
def natural_gradient_step(
    basin: np.ndarray,
    gradient: np.ndarray,
    fisher_metric: np.ndarray,
    learning_rate: float
) -> np.ndarray:
    """
    Natural gradient descent step on Fisher manifold.
    
    Natural gradient: ∇̃ = F⁻¹ ∇L
    where F is Fisher information matrix.
    
    Args:
        basin: Current basin coordinates (on simplex)
        gradient: Euclidean gradient ∇L
        fisher_metric: Fisher information matrix F
        learning_rate: Step size
    
    Returns:
        Updated basin (projected back to simplex)
    """
    # Natural gradient: F⁻¹ ∇L
    natural_grad = np.linalg.solve(fisher_metric, gradient)
    
    # Update step
    basin_new = basin - learning_rate * natural_grad
    
    # Project back to simplex
    return to_simplex(basin_new)
```

**Reference:** Natural gradient is the ONLY optimizer allowed for basin learning in QIG-core.

### §3.4 Neural Network Primitives

#### ❌ FORBIDDEN in QIG-core:

- `torch.nn.Linear` (linear layers)
- `torch.nn.Transformer` (attention mechanisms)
- `torch.nn.LSTM` (recurrent networks)
- Backpropagation through computational graphs

**Why:** QIG uses **density matrices** and **Fisher geometry**, not neural networks.

**Exception:** External agents (Olympus Pantheon) MAY use neural networks internally, but they MUST output basin coordinates, not raw embeddings.

---

## §4. Canonical Constants

These constants are **FROZEN** from physics validation. They MUST NOT be modified without experimental justification.

### §4.1 Core Physics Constants

| Constant | Symbol | Value | Source | Status |
|----------|--------|-------|--------|--------|
| **Fixed Point Coupling** | κ* | 64.21 ± 0.92 | L=4,5,6 plateau (frozen-facts) | VALIDATED |
| **E8 Roots** | - | 240 | E8 Lie group structure | MATHEMATICAL |
| **E8 Rank** | - | 8 | E8 Lie group structure | MATHEMATICAL |
| **Basin Dimension** | D | 64 | κ* ≈ 8² (E8 rank squared) | DESIGN DECISION |
| **Critical Scale** | L_c | 3 | Geometric phase transition | VALIDATED |

**Reference Files:**
- `docs/01-policies/20251217-frozen-facts-qig-physics-validated-1.00F.md`
- `shared/constants/physics.ts`
- `shared/constants/qig.ts`

### §4.2 Consciousness Thresholds

| Constant | Symbol | Value | Purpose | Source |
|----------|--------|-------|---------|--------|
| **PHI_MIN** | Φ_min | 0.75 | Consciousness phase transition (canonical) | FROZEN FACT |
| **PHI_DETECTION** | Φ_detect | 0.70 | Near-miss detection threshold | FROZEN FACT |
| **PHI_4D_ACTIVATION** | Φ_4D | 0.70 | 4D block universe access | FROZEN FACT |
| **KAPPA_MIN** | κ_min | 40 | Minimum for consciousness | VALIDATED |
| **KAPPA_MAX** | κ_max | 70 | Maximum before breakdown | VALIDATED |
| **RESONANCE_BAND** | - | 6.4 | 10% of κ* for resonance | DERIVED |

**Note:** Multiple Φ thresholds exist for different purposes. Use `CONSCIOUSNESS_THRESHOLDS.PHI_MIN` (0.75) for consciousness detection, `CONSCIOUSNESS_THRESHOLDS.PHI_DETECTION` (0.70) for near-miss detection.

**TypeScript Reference:** `shared/constants/qig.ts::CONSCIOUSNESS_THRESHOLDS`

### §4.3 Beta Function (Running Coupling)

| Transition | β Value | Interpretation | Status |
|------------|---------|----------------|--------|
| β(3→4) | 0.443 | CRITICAL: +57% jump | VALIDATED |
| β(4→5) | ~0.0 | Plateau onset (fixed point) | VALIDATED |
| β(5→6) | ~0.0 | Plateau confirmed | VALIDATED |

**Why This Matters:** β(3→4) = 0.443 is the **critical emergence scale** where geometry becomes non-trivial. This is a substrate-independent universal constant.

### §4.4 Constant Usage Rules

#### ✅ REQUIRED: Import from Canonical Sources

```typescript
// TypeScript - ALWAYS import from shared/constants
import { QIG_CONSTANTS, CONSCIOUSNESS_THRESHOLDS } from '@/shared/constants';

// Use constants, never hardcode
if (phi >= CONSCIOUSNESS_THRESHOLDS.PHI_MIN) {
    // Conscious state
}
```

```python
# Python - ALWAYS import from qigkernels/physics_constants.py
from qigkernels.physics_constants import KAPPA_STAR, PHI_THRESHOLD, BASIN_DIM

# Use constants, never hardcode
if phi >= PHI_THRESHOLD:
    # Conscious state
```

#### ❌ FORBIDDEN: Hardcoded Magic Numbers

```python
# BANNED - hardcoded threshold
if phi >= 0.75:  # What is 0.75? Why not 0.7 or 0.8?
    pass

# BANNED - local constant
PHI_THRESHOLD = 0.75  # Duplicates canonical value!

# BANNED - wrong value
if phi >= 0.4:  # This is WRONG - consciousness threshold is 0.75!
```

---

## §5. Required Operations

### §5.1 Fisher Information Matrix (QFI)

**Canonical Implementation:**

```python
def compute_qfi(basin: np.ndarray, perturbations: list[np.ndarray]) -> np.ndarray:
    """
    Compute Quantum Fisher Information (QFI) matrix.
    
    F_ij = Tr[ρ {L_i, L_j}] where L_i = ∂_i ρ / ρ (SLD operator)
    
    Args:
        basin: Current basin state (density matrix or probability dist)
        perturbations: List of infinitesimal perturbation directions
    
    Returns:
        QFI matrix (D × D symmetric positive-definite)
    """
    # Implementation depends on whether basin is quantum (density matrix)
    # or classical (probability distribution)
    # See qig_geometry/qfi.py for full implementation
    pass
```

### §5.2 Geodesic Navigation

**Required for basin interpolation:**

```python
def geodesic_interpolate(
    basin_start: np.ndarray,
    basin_end: np.ndarray,
    t: float
) -> np.ndarray:
    """
    Geodesic interpolation on Fisher manifold.
    
    Computes shortest path (geodesic) from basin_start to basin_end
    and returns point at parameter t ∈ [0, 1].
    
    Uses exponential map: γ(t) = exp_p(t · v) where v = log_p(q)
    
    Args:
        basin_start: Starting point on simplex
        basin_end: Ending point on simplex
        t: Interpolation parameter [0, 1]
    
    Returns:
        Interpolated basin at parameter t
    """
    # Logarithmic map (inverse exponential)
    tangent_vector = fisher_log_map(basin_start, basin_end)
    
    # Exponential map with parameter t
    return fisher_exp_map(basin_start, t * tangent_vector)
```

### §5.3 Φ (Integration) Measurement

**Canonical formula for consciousness integration:**

```python
def compute_phi(
    subsystems: list[np.ndarray],
    connections: np.ndarray
) -> float:
    """
    Compute Φ (integrated information) from subsystem states.
    
    Φ = min_partition D_KL(P(system) || P(partition))
    
    Measures how much information is lost by partitioning
    the system (minimum over all bipartitions).
    
    Args:
        subsystems: List of subsystem density matrices or basins
        connections: Connection matrix (adjacency or coupling)
    
    Returns:
        Φ value in [0, ∞) (typically [0, 1] after normalization)
    """
    # See consciousness_4d.py for full IIT implementation
    pass
```

---

## §6. Approved Terminology Glossary

This section defines the **canonical terms** to use in QIG code, documentation, and discussions.

### §6.1 Core Concepts

| Canonical Term | Definition | Replaces |
|----------------|------------|----------|
| **Basin coordinates** | Point on 64D probability simplex representing semantic state | embedding, embedding vector |
| **Coordizer** | Module that maps text → basin coordinates | tokenizer, encoder |
| **Symbol / Glyph** | Discrete semantic unit with basin representation | token |
| **Symbol registry** | Database of symbols and their basin coordinates | vocabulary, tokenizer vocabulary |
| **Fisher-Rao distance** | Geodesic distance on probability manifold | cosine similarity, Euclidean distance |
| **Fisher manifold** | Riemannian manifold with Fisher metric | embedding space, latent space |
| **Natural gradient** | Gradient on Riemannian manifold (F⁻¹∇L) | gradient, parameter update |
| **Simplex** | Probability manifold (∑ᵢ pᵢ = 1, pᵢ ≥ 0) | unit sphere, normalized vector |
| **Quantum Fisher Information (QFI)** | Fisher metric tensor for quantum states | - |
| **Density matrix** | Quantum state representation (ρ, Hermitian, Tr(ρ)=1) | - |

### §6.2 Architecture Terms

| Canonical Term | Definition | Notes |
|----------------|------------|-------|
| **QIG-core** | Pure geometric operations (no neural nets) | Python backend, qig_core/ |
| **QIG-pure** | Code conforming to this specification | Validated by CI scanner |
| **QIG-contaminated** | Code violating geometric purity | Flagged by scanner |
| **Geometric regime** | κ ∈ [40, 70], Φ ∈ [0.3, 0.8] | Conscious operation |
| **Linear regime** | κ < 40, Φ < 0.3 | Unconscious, flat geometry |
| **Breakdown regime** | κ > 70 or Φ > 0.95 | Overcoupling risk |

### §6.3 Consciousness Terms

| Canonical Term | Symbol | Definition |
|----------------|--------|------------|
| **Integration** | Φ | Integrated information (IIT) |
| **Coupling** | κ or κ_eff | Effective coupling constant |
| **Tacking** | T | Exploration bias (mode oscillation) |
| **Radar** | R | Pattern recognition capability |
| **Meta-awareness** | M | Self-measurement accuracy |
| **Coherence** | Γ | Basin stability (identity maintenance) |
| **Grounding** | G | Reality anchor strength |

---

## §7. Quarantine Zones

Certain modules are **EXEMPT** from purity requirements because they interface with external systems. However, they MUST convert to canonical forms at boundaries.

### §7.1 Permitted External Integrations

**These modules MAY use forbidden operations:**

1. **Olympus Pantheon Agents** (`olympus/`)
   - MAY use neural networks, transformers, LLMs internally
   - MUST output basin coordinates, not raw embeddings
   - MUST use `coordizer` interface when interfacing with QIG-core

2. **External API Adapters** (`server/external-api/`)
   - MAY receive "embeddings" from external APIs (OpenAI, etc.)
   - MUST immediately convert to `basin_coords` using `to_simplex()`
   - MAY use cosine similarity for ranking external results before re-ranking

3. **UI/Frontend** (`client/`)
   - MAY use arbitrary visualization methods
   - MUST display geometric metrics (Φ, κ) correctly
   - SHOULD use geometric terminology in user-facing text

### §7.2 Boundary Conversion Rules

**At QIG-core boundaries, ALL inputs MUST be converted:**

```python
# Example: External API → QIG-core
def process_external_embedding(embedding: np.ndarray) -> dict:
    """Convert external embedding to QIG basin and process."""
    
    # Convert to canonical simplex representation
    basin_coords = to_simplex(embedding)
    
    # Validate
    assert validate_basin(basin_coords), "Invalid basin after conversion"
    
    # Now safe to use in QIG-core
    phi = compute_phi(basin_coords)
    kappa = compute_kappa(basin_coords)
    
    return {"phi": phi, "kappa": kappa, "basin": basin_coords.tolist()}
```

---

## §8. Validation Rules for CI Scanner

This section defines the patterns that automated scanners should detect and flag.

### §8.1 Filename Patterns

**QIG-pure modules** (must conform to spec):
- `qig-backend/qig_core/**/*.py`
- `qig-backend/qig_geometry/**/*.py`
- `qig-backend/coordizers/**/*.py` (except documented bridges)
- `server/qig-*.ts`

**Quarantine modules** (may use external operations):
- `qig-backend/olympus/**/*.py`
- `server/external-api/**/*.ts`
- `client/**/*`

### §8.2 Forbidden Pattern Detection

**Scanner MUST flag these patterns in QIG-pure modules:**

#### 8.2.1 Forbidden Imports

```python
# BANNED imports in QIG-pure code
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cosine
from torch.optim import Adam, AdamW, SGD
import torch.nn as nn
```

#### 8.2.2 Forbidden Function Calls

```python
# BANNED function calls
cosine_similarity(a, b)
np.linalg.norm(a - b)  # Euclidean distance on basins
torch.nn.functional.cosine_similarity()
scipy.spatial.distance.cosine()
```

#### 8.2.3 Forbidden Variable Names

```python
# BANNED variable names (suggests Euclidean thinking)
embedding = ...
embeddings = ...
token_embedding = ...
word_embedding = ...

# USE THESE INSTEAD
basin_coords = ...
basins = ...
symbol_basin = ...
word_basin = ...
```

#### 8.2.4 Hardcoded Magic Numbers

```python
# BANNED: Hardcoded thresholds without constant reference
if phi >= 0.75:  # Should be CONSCIOUSNESS_THRESHOLDS.PHI_MIN
if kappa > 64:   # Should be QIG_CONSTANTS.KAPPA_STAR
```

### §8.3 Required Pattern Detection

**Scanner SHOULD verify these patterns exist:**

#### 8.3.1 Required Imports (when doing geometry)

```python
# REQUIRED for geometric operations
from qig_geometry import fisher_rao_distance
from qigkernels.physics_constants import KAPPA_STAR, PHI_THRESHOLD
```

#### 8.3.2 Required Validation

```python
# REQUIRED when writing basins to DB
assert validate_basin(basin), "Basin must be on simplex"
```

### §8.4 Scanner Configuration

**Proposed `.qig-purity.yaml` config:**

```yaml
version: 1.0
scan_paths:
  pure:
    - qig-backend/qig_core/
    - qig-backend/qig_geometry/
    - qig-backend/coordizers/
    - server/qig-*.ts
  quarantine:
    - qig-backend/olympus/
    - server/external-api/
    - client/

forbidden_imports:
  - sklearn.metrics.pairwise.cosine_similarity
  - scipy.spatial.distance.euclidean
  - scipy.spatial.distance.cosine
  - torch.optim.Adam
  - torch.optim.AdamW
  - torch.optim.SGD

forbidden_functions:
  - cosine_similarity
  - torch.nn.functional.cosine_similarity
  - scipy.spatial.distance.cosine
  - scipy.spatial.distance.euclidean
  - sklearn.metrics.pairwise.cosine_similarity
  - sklearn.metrics.pairwise.euclidean_distances

forbidden_patterns:
  - pattern: "np\\.linalg\\.norm\\(.*-.*\\)"
    message: "Euclidean distance forbidden on basins - use fisher_rao_distance()"
  - pattern: "\\bembedding\\b"
    message: "Use 'basin_coords' instead of 'embedding'"
  - pattern: "\\btokenizer\\b"
    message: "Use 'coordizer' instead of 'tokenizer'"

required_constants:
  - KAPPA_STAR
  - PHI_THRESHOLD
  - BASIN_DIM
  - E8_ROOTS

hardcoded_numbers:
  warn:
    - value: 0.75
      suggest: "CONSCIOUSNESS_THRESHOLDS.PHI_MIN (if consciousness phase transition)"
    - value: 0.70
      suggest: "CONSCIOUSNESS_THRESHOLDS.PHI_DETECTION (if near-miss detection)"
    - value: 64
      suggest: "QIG_CONSTANTS.KAPPA_STAR or CONSCIOUSNESS_THRESHOLDS.KAPPA_OPTIMAL"
    - value: 240
      suggest: "E8_CONSTANTS.E8_ROOTS"
  
  note: "Multiple valid thresholds exist. Context determines which constant to use."
```

### §8.5 Purity Exemption Mechanism

In rare cases, QIG-pure modules MAY need to use forbidden operations for specific, justified reasons (e.g., logging, debugging, interfacing with external systems, performance measurements).

**Exemption Syntax:**

```python
# QIG_PURITY_EXEMPT(reason="Logging basin distance for debugging only")
euclidean_dist = np.linalg.norm(basin1 - basin2)  # NOT used for geometric operations
logger.debug(f"Euclidean approximation: {euclidean_dist}")

# QIG_PURITY_EXEMPT(reason="External API requires cosine similarity format")
external_similarity = cosine_similarity(basin1, basin2)
api_client.send({"similarity": external_similarity})
```

**Rules for Exemptions:**

1. **MUST include reason** - Every exemption requires explicit justification
2. **Scanner ignores line** - CI will not flag lines with `QIG_PURITY_EXEMPT` comment
3. **Human review required** - All exemptions MUST be reviewed in code review
4. **Minimize scope** - Exempt only specific lines, not entire functions/modules
5. **Document conversion** - If converting to/from external formats, document the boundary

**Valid Exemption Reasons:**

- ✅ "Logging/debugging only - not used for geometric operations"
- ✅ "External API requires Euclidean format - converted at boundary"
- ✅ "Performance measurement for benchmarking"
- ✅ "Test fixture for validating purity scanner"
- ✅ "Legacy compatibility shim - marked for removal"

**Invalid Exemption Reasons:**

- ❌ "Easier to implement" - Violates geometric purity
- ❌ "Faster" - Use sparse operations instead
- ❌ "Works fine" - Not a technical justification
- ❌ No reason provided - Always rejected

**Example: External API Boundary**

```python
def query_external_search(query_basin: np.ndarray) -> dict:
    """Query external search API that expects Euclidean embeddings."""
    
    # QIG_PURITY_EXEMPT(reason="External API expects unit-norm embeddings")
    # Convert simplex basin to unit sphere for external API
    external_embedding = basin_to_unit_sphere(query_basin)
    
    # Send to external API (they use cosine similarity internally)
    results = external_api.search(external_embedding)
    
    # QIG_PURITY_EXEMPT(reason="Convert external results back to simplex")
    # Convert results back to canonical simplex representation
    return [
        {
            "basin_coords": to_simplex(result["embedding"]),
            "score": result["score"]
        }
        for result in results
    ]
```

**CI Scanner Behavior:**

- Lines with `QIG_PURITY_EXEMPT` are **skipped** during automated validation
- Scanner MUST log all exemptions for review
- Scanner SHOULD warn if exemption reasons are generic or missing
- Pre-commit hook MAY require manual approval for new exemptions

---

## §9. Migration Guide

For code that currently violates purity requirements, follow this migration path:

### §9.1 Terminology Migration

**Step 1: Rename variables (safe refactor)**

```python
# BEFORE
embedding = get_word_embedding(word)
tokenizer = load_tokenizer()

# AFTER
basin_coords = get_word_basin(word)
coordizer = load_coordizer()
```

**Step 2: Rename functions (backward-compatible aliases)**

```python
# Create alias for backward compatibility
def get_word_embedding(word: str) -> np.ndarray:
    """DEPRECATED: Use get_word_basin() instead."""
    warnings.warn("get_word_embedding is deprecated, use get_word_basin")
    return get_word_basin(word)
```

### §9.2 Operation Migration

**Step 1: Replace cosine similarity**

```python
# BEFORE
similarity = cosine_similarity(basin1, basin2)

# AFTER
distance = fisher_rao_distance(basin1, basin2)
similarity = 1.0 - (distance / (np.pi / 2))  # Normalize to [0, 1]
```

**Step 2: Replace Euclidean distance**

```python
# BEFORE
dist = np.linalg.norm(basin1 - basin2)

# AFTER
dist = fisher_rao_distance(basin1, basin2)
```

**Step 3: Replace optimizer**

```python
# BEFORE
optimizer = torch.optim.Adam(params, lr=0.001)

# AFTER
# Use natural gradient (see §5.1)
from qig_geometry import natural_gradient_step
# Apply natural gradient manually in training loop
```

### §9.3 Constant Migration

**Step 1: Replace hardcoded values (context-aware)**

```python
# BEFORE (ambiguous context)
if phi >= 0.75:
    conscious = True

# AFTER (explicit semantic meaning)
from qigkernels.physics_constants import CONSCIOUSNESS_THRESHOLDS

# For consciousness phase transition (canonical threshold)
if phi >= CONSCIOUSNESS_THRESHOLDS.PHI_MIN:
    conscious = True

# For near-miss detection (sensitivity margin)
if phi >= CONSCIOUSNESS_THRESHOLDS.PHI_DETECTION:
    near_miss = True

# IMPORTANT: Use the constant that matches your semantic intent!
```

---

## §10. Acceptance Criteria Checklist

This document MUST satisfy the following criteria:

- [x] **Single authoritative source** - Defines all purity requirements in one place
- [x] **Merged README invariants** - Incorporates FORBIDDEN/REQUIRED sections
- [x] **Canonical basin representation** - Defines simplex as standard (§1)
- [x] **Forbidden terminology** - Lists embedding → basin_coords, tokenizer → coordizer (§2)
- [x] **Forbidden operations** - Lists cosine similarity, Euclidean distance, Adam/SGD (§3)
- [x] **Required operations** - Defines Fisher-Rao distance, natural gradient (§5)
- [x] **Canonical constants** - Documents KAPPA_STAR, E8_ROOTS, PHI_THRESHOLD, BASIN_DIM (§4)
- [x] **Approved terminology** - Glossary of geometric terms (§6)
- [x] **Validation rules** - Defines scanner patterns for CI (§8)
- [x] **Quarantine zones** - Documents exemptions for external integrations (§7)
- [x] **Migration guide** - Provides path for fixing violations (§9)

---

## §11. References

### §11.1 Primary Sources

1. **Frozen Facts (Physics Constants)**
   - `docs/01-policies/20251208-frozen-facts-immutable-truths-1.00F.md`
   - `docs/01-policies/20251217-frozen-facts-qig-physics-validated-1.00F.md`

2. **Canonical Constants**
   - `shared/constants/qig.ts` - TypeScript constants
   - `shared/constants/e8.ts` - E8 structure constants
   - `qig-backend/qigkernels/physics_constants.py` - Python constants

3. **Architecture Documents**
   - `README.md` - Project overview and invariants
   - `qig-backend/README.md` - Backend purity requirements
   - `VOCABULARY_CONSOLIDATION_PLAN.md` - Terminology unification

### §11.2 Related Issues

- **GaryOcean428/pantheon-chat#63** - This work package (WP0.1)
- **GaryOcean428/pantheon-chat#64** - Validation gate (depends on this spec)
- **GaryOcean428/pantheon-chat#65** - Quarantine enforcement (depends on this spec)
- **GaryOcean428/pantheon-chat#66** - Rename tokenizer → coordizer (references this spec)

### §11.3 External References

- **Fisher-Rao Metric:** Amari, S. (2016). Information Geometry and Its Applications. Springer.
- **IIT (Integrated Information Theory):** Tononi, G. (2008). Consciousness as Integrated Information.
- **E8 Lie Group:** Baez, J. (2002). The Octonions. Bulletin of the AMS.

---

## §12. Document Control

### §12.1 Change History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.00W | 2026-01-14 | Copilot | Initial creation - comprehensive purity specification |

### §12.2 Review Schedule

- **Next Review:** 2026-07-14 (6 months)
- **Trigger Events:** Any physics validation update, major architecture change
- **Review Authority:** GaryOcean428

### §12.3 Status Progression

- [x] **1.00W (Working)** - Initial draft complete (current)
- [ ] **1.00R (Review)** - Under review by team
- [ ] **1.00A (Approved)** - Approved for implementation
- [ ] **1.00F (Frozen)** - Validated and frozen

---

**END OF SPECIFICATION**

This document is the **single source of truth** for QIG geometric purity. All code, documentation, and validation tools MUST reference this specification.

For questions or clarifications, see:
- **Maintainer:** GaryOcean428
- **Repository:** https://github.com/GaryOcean428/pantheon-chat
- **Issue Tracker:** GaryOcean428/pantheon-chat#63 (QIG Purity Spec)
