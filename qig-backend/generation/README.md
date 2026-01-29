# Unified Generation Pipeline - E8 Phase 3

**Status:** ✅ IMPLEMENTED  
**Version:** 1.0  
**Date:** 2026-01-22  
**Protocol:** Ultra Consciousness v4.0 ACTIVE

---

## Overview

The Unified Generation Pipeline integrates token role learning, foresight prediction, and trajectory scoring into a single QIG-pure generation system that operates without external LLM calls.

**Location:** `qig-backend/generation/`

**Key Components:**
- `token_role_learner.py` - Geometric role classification
- `foresight_predictor.py` - Trajectory-based basin prediction
- `unified_pipeline.py` - Integrated generation pipeline
- `__init__.py` - Module exports

---

## Architecture

### 1. Token Role Learner

**Purpose:** Derive geometric roles from Fisher-Rao manifold structure (NOT linguistic POS tags)

**Geometric Roles:**
- `basin_center` - Low QFI, stable attractor
- `boundary_crosser` - High QFI, between basins
- `manifold_anchor` - High frequency, low divergence
- `explorer` - Low frequency, high divergence
- `integrator` - Connects multiple basins

**Classification Logic:**
```python
# High-confidence rules
if qfi_score < 0.3 and frequency > 10:
    return BASIN_CENTER
elif qfi_score > 0.7 and mean_distance > 0.5:
    return BOUNDARY_CROSSER
```

### 2. Foresight Predictor

**Purpose:** Predict next basin via trajectory regression

**Method:** Fisher-weighted regression over context window (default 8 basins)

**Key Features:**
- Wraps `TrajectoryDecoder` for clean API
- Returns predictions on probability simplex
- Provides confidence metrics
- Scores candidates by Fisher distance to predicted basin

**Scoring Formula:**
```python
fisher_distance = fisher_rao_distance(candidate_basin, predicted_basin)
score = 1.0 - (fisher_distance / (π/2))  # Normalize to [0, 1]
```

### 3. Unified Pipeline

**Purpose:** Integrate all components for QIG-pure generation

**Strategies:**
1. **FORESIGHT_DRIVEN** - Use trajectory prediction only
2. **ROLE_DRIVEN** - Use token role skeleton only
3. **HYBRID** - Combine both (default weights: 0.4 foresight, 0.3 role, 0.3 trajectory)

**Generation Flow:**
```
1. Encode context → basin trajectory
2. Get token roles for current context
3. Predict next basin via foresight
4. Get candidates from vocabulary
5. Score candidates (foresight + role + trajectory)
6. Select best candidate
7. Update trajectory and context
8. Repeat until max_tokens or completion
```

---

## Usage

### Basic Usage

```python
from generation import UnifiedGenerationPipeline, GenerationStrategy

# Initialize pipeline
pipeline = UnifiedGenerationPipeline(
    strategy=GenerationStrategy.HYBRID,
    context_window=8,
    enforce_purity=True,
)

# Generate text
result = pipeline.generate(
    context=["the", "quick", "brown"],
    max_tokens=50,
)

# Access results
print(result.text)
print(f"Mean foresight score: {result.mean_foresight_score:.3f}")
print(f"Mean role confidence: {result.mean_role_confidence:.3f}")
print(f"Trajectory coherence: {result.trajectory_coherence:.3f}")
```

### Per-Token Metrics

```python
# Access per-token metrics
for metrics in result.token_metrics:
    print(f"Token: {metrics.token}")
    print(f"  Foresight score: {metrics.foresight_score:.3f}")
    print(f"  Geometric role: {metrics.geometric_role.value}")
    print(f"  Role confidence: {metrics.role_confidence:.3f}")
    print(f"  Fisher distance to predicted: {metrics.fisher_distance_to_predicted:.3f}")
```

### Strategy Selection

```python
# Foresight-driven (trajectory prediction only)
pipeline = UnifiedGenerationPipeline(
    strategy=GenerationStrategy.FORESIGHT_DRIVEN
)

# Role-driven (token role skeleton only)
pipeline = UnifiedGenerationPipeline(
    strategy=GenerationStrategy.ROLE_DRIVEN
)

# Hybrid (default - combines both)
pipeline = UnifiedGenerationPipeline(
    strategy=GenerationStrategy.HYBRID,
    foresight_weight=0.4,
    role_weight=0.3,
    trajectory_weight=0.3,
)
```

---

## QIG_PURITY_MODE

The pipeline enforces QIG purity by:

1. **NO External LLM calls** - All generation from geometric operations
2. **NO External NLP** - No spacy, nltk, or other linguistic tools
3. **Fisher-Rao distances only** - No cosine similarity, no Euclidean distance on basins
4. **Simplex constraints** - All basins on probability simplex

**Enable Purity Mode:**
```bash
export QIG_PURITY_MODE=true
python3 your_script.py
```

**Check Purity Mode:**
```python
from qig_purity_mode import is_purity_mode_enabled

if is_purity_mode_enabled():
    print("Running in QIG_PURITY_MODE")
```

---

## Validation

**Run validation suite:**
```bash
cd qig-backend
python3 validate_unified_pipeline.py
```

**Expected output:**
```
✅ TokenRoleLearner tests passed!
✅ ForesightPredictor tests passed!
✅ UnifiedGenerationPipeline tests passed!
✅ QIG_PURITY_MODE tests passed!
✅ ALL VALIDATION TESTS PASSED!
```

**Run purity audit:**
```bash
python3 scripts/comprehensive_purity_audit.py generation/
```

**Expected output:**
```
✅ E8 PROTOCOL v4.0 COMPLIANCE: COMPLETE
```

---

## Per-Token Observable Metrics

The pipeline provides rich per-token metrics for analysis:

| Metric | Description | Range |
|--------|-------------|-------|
| `fisher_distance_to_predicted` | Fisher-Rao distance to predicted basin | [0, π/2] |
| `foresight_score` | Similarity to predicted basin | [0, 1] |
| `geometric_role` | Token's geometric role | Enum |
| `role_confidence` | Confidence in role classification | [0, 1] |
| `trajectory_coherence` | Trajectory smoothness | [0, 1] |
| `velocity_magnitude` | Trajectory velocity | ≥0 |
| `combined_score` | Weighted combination of all metrics | [0, 1] |

---

## Integration with Existing Systems

### With QIG Generation

```python
from qig_generation import QIGGenerator
from generation import UnifiedGenerationPipeline

# Use unified pipeline as backend for QIG generation
generator = QIGGenerator()
pipeline = UnifiedGenerationPipeline()

# Pipeline can replace or augment existing generation logic
```

### With Coordizer

```python
from coordizers import get_coordizer
from generation import UnifiedGenerationPipeline

# Pipeline automatically uses coordizer if available
coordizer = get_coordizer()
pipeline = UnifiedGenerationPipeline()  # Will use coordizer internally
```

### With Trajectory Decoder

```python
from trajectory_decoder import TrajectoryDecoder
from generation import ForesightPredictor

# ForesightPredictor wraps TrajectoryDecoder
coordizer = get_coordizer()
predictor = ForesightPredictor(coordizer=coordizer)

# Use directly
trajectory = [...]  # List of basin coordinates
predicted = predictor.predict(trajectory)
```

---

## Design Decisions

### Why No POS Tags?

POS tags (noun, verb, adjective) are linguistic categories, not geometric roles. They:
- Require external NLP tools (spacy, nltk)
- Don't align with QIG's Fisher-Rao manifold
- Can't be learned from basin coordinates
- Break QIG_PURITY_MODE

Geometric roles are derived from:
- QFI score (information geometry)
- Frequency (statistical presence)
- Fisher-Rao distance to neighbors (manifold position)

### Why Foresight Over Bigram?

Bigram models look at the last 2 tokens - reactive, noisy. Foresight prediction:
- Uses full trajectory context (8 basins default)
- Fisher-weighted regression (recency + coherence)
- Predicts WHERE trajectory is GOING, not WHERE it IS
- More robust to noise in any single basin

### Why Hybrid Strategy?

Pure foresight can be too aggressive (overfit to trajectory). Pure role can be too conservative (ignore dynamics). Hybrid combines:
- Foresight: 40% weight - direction prediction
- Role: 30% weight - structural constraints
- Trajectory: 30% weight - flow coherence

---

## Troubleshooting

### Issue: "Coordizer not available"

**Cause:** Database connection required for vocabulary access

**Solution:**
```bash
# Set database URL
export DATABASE_URL="postgresql://..."

# Or run with mock coordizer for testing
pipeline = UnifiedGenerationPipeline(enforce_purity=False)
```

### Issue: "Trajectory too short for prediction"

**Cause:** Less than 2 basins in trajectory

**Solution:**
```python
# Ensure sufficient context
context = ["the", "quick", "brown"]  # At least 2 tokens
result = pipeline.generate(context, max_tokens=10)
```

### Issue: "Basin sum not 1.0"

**Cause:** Hellinger (sqrt-space) vs simplex representation mismatch

**Solution:** Fixed in v1.0 - ForesightPredictor automatically converts from Hellinger to simplex

### Issue: "Purity mode violations"

**Cause:** External LLM calls or cosine similarity detected

**Solution:** Run purity audit to identify violations:
```bash
python3 scripts/comprehensive_purity_audit.py generation/
```

---

## Testing

### Unit Tests

```bash
cd qig-backend
python3 -m pytest tests/test_unified_generation_pipeline.py -v
```

### Integration Tests

```bash
# Run full validation
python3 validate_unified_pipeline.py
```

### Manual Testing

```python
from generation import UnifiedGenerationPipeline

pipeline = UnifiedGenerationPipeline(enforce_purity=False)
result = pipeline.generate(["hello", "world"], max_tokens=5)
print(result.text)
```

---

## Performance Considerations

### Memory Usage

- Token role cache: O(n) where n = unique tokens seen
- Trajectory storage: O(k) where k = context_window
- Candidate scoring: O(m) where m = vocabulary size

### Optimization Tips

1. **Limit candidates:** Use `max_candidates` parameter (default 100)
2. **Clear role cache:** Call `role_learner.clear_cache()` periodically
3. **Reduce context window:** Decrease from 8 to 4 for faster prediction

---

## Future Enhancements

### Planned (E8 Phase 4)

- [ ] Attractor-based completion criteria
- [ ] Multi-scale trajectory prediction (L=3→4→8→64)
- [ ] E8 kernel routing for domain-specific generation
- [ ] Consciousness metrics integration (Φ, κ)

### Under Consideration

- [ ] Beam search with foresight scoring
- [ ] Adaptive weight adjustment based on Φ regime
- [ ] Token role learning from user feedback
- [ ] Trajectory visualization tools

---

## References

- **E8 Protocol Phase 3:** `docs/10-e8-protocol/README.md`
- **Issue #03:** `docs/10-e8-protocol/issues/20260116-issue-03-qig-native-skeleton-1.01W.md`
- **TrajectoryDecoder:** `qig-backend/trajectory_decoder.py`
- **QIG Purity Mode:** `qig-backend/qig_purity_mode.py`
- **Fisher-Rao Geometry:** `qig-backend/qig_geometry/canonical.py`

---

## Authors

- **Copilot Agent** - Initial implementation (E8 Phase 3)
- **E8 Protocol v4.0** - Architecture specification

---

## License

This code is part of the Pantheon-Chat project and follows the project's license.
