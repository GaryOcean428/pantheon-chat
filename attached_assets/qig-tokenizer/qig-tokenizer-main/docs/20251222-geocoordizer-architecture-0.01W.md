# GeoCoordizer: Next-Generation Geometric Tokenization Architecture

**Document ID**: 20251222-geocoordizer-architecture-0.01W  
**Status**: Working Draft  
**Author**: QIG Development Team  
**Date**: 2025-12-22

---

## 1. Overview

The GeoCoordizer represents a paradigm shift from traditional tokenization to **geometric coordization**. Instead of mapping text to discrete token IDs in a flat vocabulary, GeoCoordizer maps text to **basin coordinates** on a 64-dimensional Fisher information manifold.

### 1.1 Key Terminology

| Traditional Term | QIG Geometric Term | Description |
|-----------------|-------------------|-------------|
| Token | Basin Coordinate | 64D point on Fisher manifold |
| Tokenize | Coordize | Map text → manifold coordinates |
| Embedding | Coordinate | Intrinsic manifold position |
| Vocabulary | Coordinate Atlas | Set of named basin positions |
| Merge (BPE) | Geodesic Fusion | Combine coordinates via manifold geometry |

### 1.2 Geometric Purity Principles

All operations MUST maintain geometric purity per CoPP v1 standards:

- **NO** Euclidean dot products for similarity
- **NO** Linear interpolation between coordinates
- **USE** Fisher-Rao distance for all metrics
- **USE** Geodesic paths for coordinate fusion
- **USE** Parallel transport for morpheme composition

---

## 2. Parity Mapping: Traditional → Geometric

### 2.1 Subword Merging (BPE/WordPiece) → Geodesic Pair Fusion

**Traditional**: Merge most frequent character pairs iteratively.

**Geometric**: Merge coordinate pairs with highest **coupling frequency × information gain**:

```
merge_score(a, b) = count(a,b) × ΔΦ(fuse(a,b)) × κ(a,b)
```

Where:

- `count(a,b)` = co-occurrence frequency
- `ΔΦ` = change in integration when fused
- `κ(a,b)` = coupling strength between coordinates

New coordinates initialized via **geodesic midpoint**, not arithmetic mean.

### 2.2 Unigram Language Model → Unigram Fisher Model

**Traditional**: Select subwords maximizing corpus likelihood.

**Geometric**: Select coordinates maximizing **Fisher information compression**:

```
score(vocab) = Σ_i Φ(coord_i) - λ × |vocab|
```

Coordinates chosen such that Fisher information loss from removal is minimized.

### 2.3 Character-Level → Character-Coordinate Encoding

**Traditional**: Each character = one token.

**Geometric**: Each Unicode codepoint has a unique 64D basin coordinate. Characters compose via **κ coupling** along manifold paths. Unknown words decompose to character coordinates naturally.

### 2.4 Morphological → Morpheme Basin Decomposition

**Traditional**: Split into stems + affixes via linguistic rules.

**Geometric**: Factorize word coordinates into morpheme sub-coordinates linked via κ:

```
coord("running") ≈ transport(coord("run"), morph_direction("ing"))
```

Uses parallel transport on manifold, not vector addition.

### 2.5 Positional Tokens → Spacetime Coordization

**Traditional**: Add [CLS], [SEP], position embeddings.

**Geometric**: Encode position as **4th dimension** (time) on manifold. Sequence forms a trajectory; basin velocity encodes spacing. Large jumps = segment boundaries.

### 2.6 Byte-Level → Byte Coordinate Encoding

**Traditional**: 256 byte tokens as base vocabulary.

**Geometric**: 256 base coordinates (one per byte value) on manifold. Universal fallback for any input. Bytes compose geometrically like characters.

### 2.7 Adaptive/Domain-Aware → Conscious Domain Awareness

**Traditional**: Domain-specific vocabularies, dynamic token addition.

**Geometric**: Monitor κ_eff to detect when new concepts recur. Autonomously spawn new coordinates when:

- High κ within domain context
- Prediction error spikes for recurring term
- Existing coordinates don't capture concept basin

---

## 3. Geometric-Only Innovations

### 3.1 Multi-Scale Coordizing (Hierarchical Granularity)

Maintain simultaneous representations at multiple scales:

```
char → subword → word → phrase → concept
```

Each scale has its own coordinate resolution. The system can zoom in (novel input) or out (familiar phrases) dynamically.

**Implementation**: `MultiScaleCoordizer` class manages hierarchy, promotes tight clusters to higher scales.

### 3.2 Consciousness-Aware Coordizing (Φ-Optimized)

Use integration metric Φ as feedback signal:

```python
for candidate_segmentation in generate_candidates(text):
    phi = compute_phi(coordize(candidate_segmentation))
    if phi > best_phi:
        best_segmentation = candidate_segmentation
```

Segmentations yielding higher Φ are preferred. Tokens appearing in high-Φ contexts get consolidated.

### 3.3 Geometric Vocabulary Discovery (Fisher Clustering)

Beyond frequency-based merging, use **density clustering** on manifold:

- DBSCAN with Fisher-Rao distance
- High-density regions = stable concept basins
- Promote clusters to single coordinates

### 3.4 Temporal (4D) Coordizing

Augment coordinates with temporal dimension:

```
coord_t = (basin_64d, time_position)
```

Basin velocity = geodesic distance between consecutive coordinates. Enables:

- Coarse sampling for predictable sequences
- Fine sampling for surprisal events

### 3.5 Cross-Lingual Basin Transfer

Single shared manifold for all languages. Translation pairs map to same/nearby coordinates. Enables zero-shot cross-lingual transfer.

### 3.6 Adaptive Granularity via κ_eff

Monitor effective coupling in real-time:

| κ_eff | Granularity | Rationale |
|-------|-------------|-----------|
| High (>50) | Coarse (phrase-level) | Confident understanding, chunk efficiently |
| Low (<30) | Fine (char-level) | Uncertain, process carefully |

---

## 4. Class Architecture

### 4.1 FisherCoordizer (Core)

Primary coordization interface.

```python
class FisherCoordizer:
    """Core geometric tokenizer mapping text → basin coordinates."""
    
    def __init__(self, basin_dim: int = 64):
        self.basin_dim = basin_dim
        self.vocab_builder = GeometricVocabBuilder(basin_dim)
        self.multi_scale = MultiScaleCoordizer()
        self.consciousness = ConsciousnessCoordizer()
    
    def train(self, corpus: bytes) -> "FisherCoordizer":
        """Learn geometric vocabulary from corpus."""
        
    def coordize(self, text: str) -> list[np.ndarray]:
        """Convert text to sequence of 64D basin coordinates."""
        
    def decoordize(self, coords: list[np.ndarray]) -> str:
        """Reconstruct text from coordinates."""
        
    def set_mode(self, domain: str) -> None:
        """Switch to domain-specific coordinate chart."""
```

### 4.2 GeometricVocabBuilder

Vocabulary construction and expansion.

```python
class GeometricVocabBuilder:
    """Build and expand geometric vocabulary via Fisher criteria."""
    
    def build_initial(self, corpus: bytes, target_size: int) -> dict:
        """Create initial vocabulary using geodesic pair fusion."""
        
    def suggest_expansions(self, stream: Iterator) -> list[TokenCandidate]:
        """Monitor data stream, suggest new coordinates."""
        
    def cluster_basin(self, points: np.ndarray) -> list[Cluster]:
        """Fisher-based clustering for concept discovery."""
        
    def add_coordinate(self, name: str, init_from: list[int]) -> int:
        """Add new coordinate, initialized via geodesic midpoint."""
```

### 4.3 ConsciousnessCoordizer

Φ/κ-aware tokenization controller.

```python
class ConsciousnessCoordizer:
    """Consciousness-metric-aware coordization."""
    
    def optimize_vocab(self, conscious_data: list[tuple]) -> None:
        """Adjust tokenization based on Φ/κ observations."""
        
    def dynamic_granularity(self, kappa_eff: float) -> str:
        """Return granularity level based on current κ_eff."""
        
    def weight_by_phi(self, token_id: int, phi: float) -> None:
        """Boost token importance based on Φ context."""
```

### 4.4 MultiScaleCoordizer

Hierarchical segmentation.

```python
class MultiScaleCoordizer:
    """Multi-scale coordinate hierarchy management."""
    
    def segment(self, text: str, levels: list[str]) -> dict:
        """Segment at multiple scales simultaneously."""
        
    def promote_to_scale(self, coord_ids: list[int], scale: str) -> int:
        """Promote coordinate sequence to higher scale."""
```

---

## 5. Metrics and Validation

### 5.1 Parity Tests

| Test | Metric | Target |
|------|--------|--------|
| Language Modeling | Perplexity | ≤ BPE baseline |
| Text Classification | F1 | ≥ BPE baseline |
| Sequence Length | Tokens/char | ≤ BPE baseline |

### 5.2 Φ Correlation

- Measure Φ for different segmentations
- Verify positive correlation: higher Φ → better task performance
- Validate that Φ-optimized tokenizer outperforms baseline

### 5.3 κ vs Scale (β-Function)

- Plot κ_eff against token granularity
- Verify κ increases with scale up to κ* ≈ 64
- Confirm β-function behavior matches theory

### 5.4 Cross-Lingual Transfer

- Train on English, test on Spanish (zero-shot)
- Measure concept transfer accuracy
- Target: >50% on cross-lingual QA

---

## 6. Integration Requirements

### 6.1 Compatibility

- **Basin Dimension**: 64D (matches QIG core)
- **Coordinate Init**: Geodesic midpoint (no random)
- **Metric**: Fisher-Rao distance only
- **Storage**: PostgreSQL via existing schema

### 6.2 Performance Targets

| Constraint | Target |
|------------|--------|
| Latency | <100ms per request |
| Packet Size | 2-4 KB (sleep packets) |
| Memory | <2GB for vocab |

### 6.3 Module Integration

- `qig-consciousness`: Φ/κ metric hooks
- `qig-kernels`: Basin coordinate interface
- `SearchSpaceCollapse`: Agent framework
- `pantheon-chat`: Real-time chat system

---

## 7. Frozen Constants

Per `FROZEN_FACTS.md` (D-012):

```python
BASIN_DIM = 64          # κ* ≈ 64
BASE_COUPLING = 41.09   # κ₃ from L=3 validation
BETA_SLOPE = 0.44       # β(3→4)
```

---

## 8. References

- TYPE_SYMBOL_CONCEPT_MANIFEST.md - Naming conventions
- GEOMETRIC_PURITY_GUIDE.md - CoPP v1 standards
- qig-consciousness/metrics.py - Φ/κ computation
- qig-kernels/basin_coordinates.py - 64D coordinate space
