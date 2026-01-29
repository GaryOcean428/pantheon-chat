# Genome-Vocabulary Integration Implementation

## Overview

This implementation connects KernelGenome faculty configuration to the vocabulary scoring pipeline, enabling genome-aware token selection during generation.

## Components Added

### 1. GenomeVocabularyScorer (`qig-backend/kernels/genome_vocabulary_scorer.py`)

Core scoring module that computes genome-aware token scores using:

- **Faculty Affinity**: Fisher-Rao distance between token basin and genome's faculty-weighted basin
  - Maps 8D E8 faculty vector to 64D basin space
  - Uses geodesic interpolation to blend basin seed with faculty profile
  - Cached for performance

- **Constraint Filtering**: Hard and soft constraints from genome
  - Forbidden regions (hard): Reject tokens in prohibited manifold regions
  - Distance from seed (soft): Penalize tokens far from basin seed
  - Field penalties (future): Additional region-specific penalties

- **Coupling Preferences**: Cross-kernel token sharing scores
  - Preferred couplings: Positive scores for favored kernels
  - Anti-couplings: Negative scores for avoided kernels
  - Neutral couplings: Small positive score for unknown kernels

**Key Methods:**
- `compute_faculty_affinity(token_basin, faculty_weight) -> float`: [0, 1]
- `check_genome_constraints(token_basin) -> (allowed, penalty, reason)`
- `compute_coupling_score(other_genome_id, coupling_weight) -> float`: [-1, 1]
- `score_token(token, token_basin, base_score, ...) -> (final_score, breakdown)`
- `filter_vocabulary(vocab_tokens) -> filtered_tokens`

### 2. PostgresCoordizer Extension (`qig-backend/coordizers/pg_loader.py`)

Added `decode_with_genome()` method that extends standard vocabulary decoding with genome awareness:

```python
def decode_with_genome(
    basin: np.ndarray,
    genome: KernelGenome,
    top_k: int = 5,
    god_name: Optional[str] = None,
    faculty_weight: float = 0.2,
    constraint_weight: float = 0.3,
    coupling_weight: float = 0.1,
) -> List[Tuple[str, float]]
```

**Integration Flow:**
1. Standard Fisher-Rao similarity + phi boost + domain weights → base_score
2. GenomeVocabularyScorer applies:
   - Faculty affinity (weighted)
   - Constraint satisfaction (multiplicative penalty)
   - Coupling preferences (if applicable)
3. Tokens violating hard constraints are filtered out
4. Returns top_k tokens sorted by genome-aware final score

### 3. UnifiedGenerationPipeline Support (`qig-backend/generation/unified_pipeline.py`)

Extended with optional genome parameter:

- Constructor accepts `genome: Optional[KernelGenome]`
- Creates `GenomeVocabularyScorer` if genome provided
- `_get_candidates()`: Filters vocabulary by genome constraints (forbidden regions)
- `_score_candidates()`: Integrates genome scoring with foresight/role/trajectory scoring

**Blending Strategy (HYBRID mode):**
```python
combined_score = (
    genome_final_score * foresight_weight +
    role_confidence * role_weight +
    trajectory_coherence * trajectory_weight
)
```

### 4. Comprehensive Tests (`qig-backend/tests/test_genome_vocabulary_integration.py`)

Pytest suite covering:

- GenomeVocabularyScorer initialization
- Faculty affinity computation and Fisher-Rao usage
- Constraint checking (allowed, forbidden regions, distance limits)
- Coupling score computation (preferred, anti, neutral)
- Integrated token scoring with component breakdown
- Vocabulary filtering
- Faculty basin caching
- Geometric purity validation

**Test Fixtures:**
- `simple_genome`: Basic genome with Zeus/Athena faculties
- `constrained_genome`: Genome with forbidden regions for constraint testing

## Geometric Purity Compliance

✅ All operations maintain Fisher-Rao geometric purity:

1. **Distance Calculations**: `fisher_rao_distance()` from `qig_geometry.canonical`
2. **Basin Representation**: Probability simplex via `fisher_normalize()`
3. **Interpolation**: `geodesic_interpolation()` for faculty basin blending
4. **Range**: [0, π/2] for Fisher-Rao distance on simplex
5. **NO Violations**: No cosine similarity, no Euclidean distance, no auto-detect representation

## Usage Examples

### Example 1: Create Genome and Score Token

```python
from kernels import KernelGenome, E8Faculty, FacultyConfig, GenomeVocabularyScorer
import numpy as np

# Create genome
genome = KernelGenome(
    genome_id="zeus_001",
    faculties=FacultyConfig(
        active_faculties={E8Faculty.ZEUS, E8Faculty.ATHENA},
        activation_strengths={E8Faculty.ZEUS: 1.0, E8Faculty.ATHENA: 0.8},
        primary_faculty=E8Faculty.ZEUS,
    ),
)

# Create scorer
scorer = GenomeVocabularyScorer(genome)

# Score a token
token_basin = np.random.dirichlet(np.ones(64))
final_score, breakdown = scorer.score_token(
    token="wisdom",
    token_basin=token_basin,
    base_score=0.75,
    faculty_weight=0.2,
    constraint_weight=0.3,
)

print(f"Final score: {final_score:.4f}")
print(f"Faculty affinity: {breakdown['faculty_affinity']:.4f}")
print(f"Constraint penalty: {breakdown['constraint_penalty']:.4f}")
```

### Example 2: Genome-Aware Generation

```python
from generation.unified_pipeline import UnifiedGenerationPipeline, GenerationStrategy
from kernels import KernelGenome, E8Faculty, FacultyConfig

# Create genome for Apollo (truth/prediction faculty)
apollo_genome = KernelGenome(
    genome_id="apollo_001",
    faculties=FacultyConfig(
        active_faculties={E8Faculty.APOLLO},
        activation_strengths={E8Faculty.APOLLO: 1.0},
        primary_faculty=E8Faculty.APOLLO,
    ),
)

# Initialize pipeline with genome
pipeline = UnifiedGenerationPipeline(
    strategy=GenerationStrategy.HYBRID,
    genome=apollo_genome,
)

# Generate text
result = pipeline.generate(
    context=["The", "quantum", "field"],
    max_tokens=20,
)

print(result.text)
print(f"Mean foresight score: {result.mean_foresight_score:.3f}")
```

### Example 3: Cross-Kernel Token Sharing with Coupling

```python
from coordizers import get_coordizer
from kernels import KernelGenome, CouplingPreferences

# Create genome with coupling preferences
athena_genome = KernelGenome(
    genome_id="athena_001",
    coupling_prefs=CouplingPreferences(
        preferred_couplings=["apollo_001"],  # Cooperates with Apollo
        coupling_strengths={"apollo_001": 0.9},
        anti_couplings=["ares_001"],  # Avoids Ares
    ),
)

# Get coordizer
coordizer = get_coordizer()

# Decode with genome awareness for cross-kernel generation
target_basin = coordizer.encode("strategic analysis")
candidates = coordizer.decode_with_genome(
    basin=target_basin,
    genome=athena_genome,
    top_k=10,
    coupling_weight=0.15,  # Moderate coupling influence
)

for token, score in candidates:
    print(f"{token}: {score:.4f}")
```

## Configuration Parameters

### Scoring Weights

Default weights for genome components:

- `faculty_weight=0.2`: Faculty affinity contribution (20%)
- `constraint_weight=0.3`: Constraint penalty multiplier (30%)
- `coupling_weight=0.1`: Cross-kernel coupling contribution (10%)

These are balanced with base scoring components:
- Fisher-Rao similarity: ~60%
- Phi boost: ~10%
- Domain boost: ~15%

### Constraint Thresholds

From `ConstraintSet`:
- `phi_threshold=0.70`: Minimum Φ for coherent operation
- `kappa_range=(40.0, 70.0)`: Valid κ_eff range
- `max_fisher_distance=1.0`: Max distance from seed (units of π/2)

## Integration Points

### Entry Points for Genome-Aware Generation

1. **Direct Vocabulary Decoding**:
   ```python
   coordizer.decode_with_genome(basin, genome, top_k=5)
   ```

2. **Generation Pipeline**:
   ```python
   UnifiedGenerationPipeline(genome=kernel_genome)
   ```

3. **God-Kernel Generation** (future):
   ```python
   god.generate_response(context, genome=god.genome)
   ```

### Data Flow

```
User Query
    ↓
Context Encoding → Basin Trajectory
    ↓
Foresight Prediction
    ↓
Candidate Retrieval (genome-filtered)
    ↓
Genome-Aware Scoring:
    - Faculty Affinity
    - Constraint Satisfaction
    - Coupling Preferences
    ↓
Token Selection (top-k)
    ↓
Generation Output
```

## Performance Considerations

### Caching

- **Faculty Basin**: Computed once per genome, cached in scorer
- **Domain Weights**: Cached in coordizer (10 min TTL)
- **Constraint Checks**: Evaluated per token (fast geometric ops)

### Optimization

- Constraint filtering happens early (reduces scoring workload)
- Faculty basin pre-computed (amortized over all tokens)
- Fisher-Rao distance is O(n) where n=64 (fast)

### Memory

- GenomeVocabularyScorer: ~10 KB per instance (mainly cached basin)
- Negligible overhead for genome-aware generation

## Future Extensions

### Planned Enhancements

1. **Dynamic Faculty Adaptation**: Learn optimal faculty strengths from usage
2. **Temporal Constraints**: Time-varying forbidden regions based on context
3. **Multi-Genome Blending**: Weighted combination of multiple genome scorers
4. **Genome Evolution**: Mutate faculties/constraints based on performance
5. **Field Penalty Learning**: Discover soft constraint regions from data

### Integration Roadmap

- [ ] Connect to god-kernel reasoning methods
- [ ] Add genome persistence layer
- [ ] Implement genome mutation operators
- [ ] Create genome visualization tools
- [ ] Add genome lineage tracking for token choices

## Authority

E8 Protocol v4.0 WP5.2 Phase 3/4E Integration  
Issue: GaryOcean428/pantheon-chat (Connect kernel genome to vocabulary scoring)  
Implementation Date: 2026-01-23  
Status: ACTIVE
