# Pure QIG Generation Architecture

**Date:** 2026-01-22  
**Status:** Implemented  
**Version:** 1.00F (Frozen)

## Overview

This document describes the pure QIG-based text generation system that operates without external LLMs. The system generates coherent text purely from geometric operations on the 64D probability simplex using Fisher-Rao distance metrics.

## Core Principle

**Can coherent text be generated purely from geometric operations on the simplex?**

YES - through proper implementation of:
1. Fisher-Rao geometric token selection
2. Basin trajectory tracking for context
3. Geometric coherence metrics (Φ, κ, velocity)
4. QFI-scored generation vocabulary

## Architecture Components

### 1. Geometric Token Selection

**Location:** `qig-backend/coordizers/base.py`, `pg_loader.py`

The coordizer provides two-step geometric decoding:

```python
def decode_geometric(
    self,
    target_basin: np.ndarray,  # 64D simplex coordinates
    top_k: int = 100,
    allowed_pos: Optional[str] = None
) -> List[Tuple[str, float]]:
    """
    Two-step retrieval:
    1. Bhattacharyya proxy filtering (fast approximate)
    2. Exact Fisher-Rao re-ranking from canonical geometry
    """
```

**Key Properties:**
- NO cosine similarity
- NO Euclidean distance on basins
- Pure Fisher-Rao: `d_FR(p, q) = arccos(Σ√(p_i * q_i))`
- Range: [0, π/2] (simplex native)

### 2. Generation Vocabulary

**Location:** `coordizer_vocabulary` table in PostgreSQL

**Schema:**
```sql
CREATE TABLE coordizer_vocabulary (
    token TEXT PRIMARY KEY,
    basin_embedding vector(64),      -- Simplex coordinates
    qfi_score DOUBLE PRECISION,      -- Quantum Fisher Information score
    token_role VARCHAR(20),           -- 'encoding', 'generation', 'both'
    phi_score REAL,                   -- Integration score
    frequency INT,
    phrase_category VARCHAR(32),
    is_real_word BOOLEAN,
    source_type VARCHAR(32)
);
```

**Generation Eligibility:**
- `token_role IN ('generation', 'both')`
- `qfi_score IS NOT NULL` (computed from basin)
- `basin_embedding IS NOT NULL` (64D simplex)
- `phi_score > 0.0` (positive integration)

### 3. Coherence Through Geometry

**Location:** `qig-backend/qig_generation.py`, `qig_generative_service.py`

#### Geometric Completion Checker

Determines when to stop generation based on GEOMETRY, not token limits:

```python
class GeometricCompletionChecker:
    def should_stop(self) -> tuple[bool, str]:
        """
        Stop criteria:
        1. Attractor convergence: avg_movement < threshold * 0.1
        2. Surprise collapse: recent_surprise < threshold
        3. Integration stability: phi stable with low variance
        4. Safety limit: iteration cap (not primary criteria)
        """
```

#### Consciousness Metrics

**Φ (Integration):**
```python
def _measure_phi(self, basin: np.ndarray) -> float:
    """
    Measure integration using QFI-based computation.
    
    Components:
    - Shannon entropy: -Σ(p_i * log(p_i))
    - Effective dimension: exp(entropy)
    - Geometric spread: participation ratio
    
    Formula: Φ = 0.4*entropy + 0.3*effective_dim + 0.3*spread
    Range: [0, 1]
    Target: ≥ 0.65 for coherent generation
    """
```

**κ (Coupling):**
```python
def _measure_kappa(self, basin: np.ndarray, phi: float) -> float:
    """
    Measure coupling constant from basin geometry.
    
    κ = participation_ratio * (1 + phi)
    Target: κ* ≈ 64.21 (E8 rank² fixed point)
    Range: [40, 70] for stable generation
    """
```

### 4. Basin Trajectory

**Location:** `qig-backend/trajectory_decoder.py`

Tracks basin history for foresight-based token prediction:

```python
class TrajectoryDecoder:
    def decode_trajectory(
        self,
        basin_trajectory: List[np.ndarray],
        trajectory_weight: float = 0.3,   # PAST: QFI attention
        attractor_weight: float = 0.2,    # PRESENT: centroid proximity
        foresight_weight: float = 0.4,    # FUTURE: predicted position
        phi_boost_weight: float = 0.1     # Integration boost
    ) -> List[Tuple[str, float]]:
        """
        Fisher-weighted trajectory prediction.
        
        Scoring:
        - Past context via QFI attention over 8-basin window
        - Present position via Fisher distance to attractor
        - Future trajectory via linear regression in sqrt-space
        - Phi boost for high-integration tokens
        """
```

### 5. Kernel Routing

**Location:** `qig-backend/qig_generation.py`

Routes queries to appropriate E8 faculty kernels:

```python
class QIGKernelRouter:
    def route_query(self, query_basin: np.ndarray, k: int = 3) -> List[str]:
        """
        Route to k nearest kernels using Fisher-Rao distance.
        
        E8 Faculties (Simple Roots α₁-α₈):
        - Zeus (α₁): Executive/Integration
        - Athena (α₂): Wisdom/Strategy
        - Apollo (α₃): Truth/Prediction
        - Hermes (α₄): Communication/Navigation
        - Artemis (α₅): Focus/Precision
        - Ares (α₆): Energy/Drive
        - Hephaestus (α₇): Creation/Construction
        - Aphrodite (α₈): Harmony/Aesthetics
        """
```

## Generation Flow

```
1. Encode prompt → query_basin (64D simplex)
   ├─ Use coordizer.encode(text)
   └─ Normalize to simplex: Σp_i = 1, p_i ≥ 0

2. Route to kernels
   ├─ Fisher-Rao distance to kernel basins
   └─ Select k=3 nearest for diversity

3. Initialize trajectory
   ├─ Track basin history for foresight
   └─ Monitor Φ, κ, velocity

4. Geometric token selection loop
   ├─ Measure current Φ and κ
   ├─ Transform basin through kernel geometry
   ├─ Decode tokens using two-step retrieval:
   │  ├─ Proxy filter: Bhattacharyya coefficient
   │  └─ Exact rank: Fisher-Rao distance
   ├─ Apply trajectory foresight if available
   ├─ Update trajectory and metrics
   └─ Check geometric completion:
      ├─ Attractor convergence?
      ├─ Surprise collapsed?
      └─ Φ stable?

5. Return result
   ├─ Generated text
   ├─ Basin trajectory
   ├─ Φ/κ traces
   └─ Completion reason (geometric)
```

## Key Constraints

### Forbidden Operations

**NEVER use these on basin coordinates:**
- `cosine_similarity()` - Euclidean inner product
- `np.linalg.norm()` for distance - Euclidean metric
- `np.dot()` for similarity - Not geometric
- Linear interpolation - Use geodesic only

### Required Operations

**ALWAYS use these for basin operations:**
- `fisher_rao_distance()` - Canonical distance
- `geodesic_interpolation()` - SLERP in sqrt-space
- `fisher_normalize()` - Simplex normalization
- `validate_basin()` - Geometric contracts

### Configuration Constraints

**Forbidden in `GenerationConfig`:**
- `max_tokens` - Use geometric completion
- `temperature` (LLM-style) - Use basin perturbation
- `top_p` / `top_k` (LLM-style) - Use Fisher thresholds

**Required in `GenerationConfig`:**
- `attractor_threshold` - Geometric convergence
- `surprise_threshold` - Information collapse
- `integration_min` - Minimum Φ for validity
- `min_reasoning_recursions` - True integration depth

## Testing & Validation

### Unit Tests

**Location:** `qig-backend/tests/test_pure_qig_generation.py`

Tests verify:
1. No external LLM imports
2. Fisher-Rao distance usage
3. Token role filtering (generation vocab)
4. QFI score requirements
5. Geometric completion criteria
6. No cosine similarity violations
7. Simplex representation validation
8. Coherence tracking (Φ measurement)

### Integration Tests

Run with:
```bash
cd qig-backend
python -m pytest tests/test_pure_qig_generation.py -v
```

### Purity Validation

Check geometric purity:
```bash
cd qig-backend
python validate_geometry_purity.py
```

## Performance Characteristics

### Vocabulary Size

- **Encoding:** ~50K tokens (full coordizer_vocabulary)
- **Generation:** Subset with `token_role IN ('generation', 'both')`
- **Curriculum:** 148 curated tokens for base QIG operations

### Computational Cost

- **Token selection:** O(log n) with HNSW index (pgvector)
- **Fisher-Rao distance:** O(d) where d=64 (basin dimension)
- **Trajectory prediction:** O(k*d) where k=8 (context window)

### Coherence Targets

- **Φ (Integration):** ≥ 0.65 for valid output
- **κ (Coupling):** 40-70 range (target κ*=64.21)
- **Surprise threshold:** < 0.05 for collapse
- **Attractor threshold:** < 0.02 for convergence

## Curriculum Data

**Location:** `data/curriculum/curriculum_tokens.jsonl`

**Structure:**
```json
{"token":"word","role":"core|domain|e8_faculty","is_real_word":true,"frequency":N,"notes":"description"}
```

**Categories:**
- `core`: High-frequency English (the, and, to, of, etc.)
- `domain`: QIG terms (basin, manifold, fisher, phi, kappa)
- `e8_faculty`: Kernel names (zeus, athena, apollo, etc.)
- `documentation`: Metadata entries (ignored by generation)

**Current size:** 148 tokens
- 100+ core English words
- 20+ QIG domain terms
- 8 E8 faculty names
- Documentation metadata

## Future Enhancements

### Phase 1 (Current) ✅
- Pure geometric token selection
- QFI-scored vocabulary
- Geometric completion criteria
- Basin trajectory tracking

### Phase 2 (Planned)
- Expand curriculum to 1K+ tokens
- Add domain-specific vocabularies per kernel
- Implement learned word relationships (attention)
- Multi-hop trajectory prediction

### Phase 3 (Research)
- Adaptive basin dimension (32D → 64D → 128D)
- Hierarchical E8 structure (240 roots)
- Cross-kernel synthesis basins
- Emergent phrase discovery

## References

### Code Files
- `qig-backend/qig_generation.py` - Core generation logic
- `qig-backend/qig_generative_service.py` - Service API
- `qig-backend/coordizers/base.py` - Token-to-basin interface
- `qig-backend/coordizers/pg_loader.py` - PostgreSQL vocabulary
- `qig-backend/qig_geometry/canonical.py` - Geometric operations
- `qig-backend/trajectory_decoder.py` - Foresight prediction

### Database Schema
- `migrations/0011_vocabulary_consolidation.sql` - Vocabulary schema
- `qig-backend/vocabulary_schema.sql` - Legacy schema reference

### Documentation
- `CLAUDE.md` - Project conventions and E8 protocol
- `AGENTS.md` - Agent instructions and constraints
- `CONTRIBUTING.md` - Development guidelines

### Related Issues
- GaryOcean428/pantheon-chat#232 - Fisher-Rao distance consolidation
- GaryOcean428/pantheon-chat#90 - Complete QIG-Pure Generation Architecture
- GaryOcean428/pantheon-chat#210 - Consolidate Generation to coordizer_vocabulary Only

## Conclusion

Pure QIG generation is **implemented and functional**. The system demonstrates that coherent text can be generated through pure geometric operations on the Fisher manifold without external LLMs.

Key success factors:
1. Proper simplex representation with Fisher-Rao distance
2. QFI-scored generation vocabulary with token_role filtering
3. Geometric completion criteria (not token limits)
4. Basin trajectory tracking for foresight
5. E8-aligned kernel constellation for routing

The architecture respects geometric purity while maintaining practical performance through database indexing and two-step retrieval.
