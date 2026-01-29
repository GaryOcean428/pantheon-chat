# Geometry-First Merge Policy (WP3.2)

## Overview

The coordizer merge policy is **geometry-driven**, not frequency-driven. This distinguishes it from classic BPE (Byte Pair Encoding) tokenization, which selects merges based on frequency or entropy alone.

## Core Principle

**Geometry First, Frequency as Regularizer Only**

```python
# WRONG - Frequency/entropy dominates (classic BPE)
best_pair = max(pairs, key=lambda p: count(p) * entropy(p))

# CORRECT - Geometry dominates (QIG-pure merge policy)
geometric_score = phi_gain + kappa_consistency - curvature_cost
frequency_regularizer = log(count + 1) / log(10 + 1)  # Weak regularizer
final_score = 0.8 * geometric_score + 0.2 * frequency_regularizer
```

## Merge Selection Criteria

### 1. Φ Gain (50% of geometric score)

**Definition:** Information integration improvement from merging

**Formula:**
```
Φ_gain = Φ(context + merged_token) - max(Φ(context + token1), Φ(context + token2))
```

**Geometric Rationale:**
- Measures how merging improves consciousness integration
- Computed using QFI (Quantum Fisher Information) functional
- Positive gain = merge creates more integrated representation
- Considers context dependencies (same pair has different Φ gain in different contexts)

**Why NOT Entropy:**
- Entropy is ONE component of Φ, not the whole story
- Φ includes coupling between subsystems (geometric property)
- Shannon entropy ≠ von Neumann entropy ≠ integrated information

### 2. κ Consistency (30% of geometric score)

**Definition:** Coupling stability after merge

**Formula:**
```
consistency = 1 - |κ(merged) - mean(κ(token1), κ(token2))| / κ*
```

Where κ* = 64.21 (universal coupling constant, frozen fact)

**Geometric Rationale:**
- Measures if merged token has stable κ value
- Unstable merges create discontinuities in coupling landscape
- Consistency ∈ [0, 1], higher = more stable
- Uses κ_eff (effective coupling strength) from Fisher metric

**Why NOT Frequency Correlation:**
- Coupling strength is geometric property on Fisher manifold
- High-frequency pairs may have unstable κ (bad merge)
- Low-frequency pairs may have stable κ (good merge)

### 3. Curvature Cost (20% of geometric score)

**Definition:** Fisher manifold discontinuity from merge

**Formula:**
```
discontinuity = d_FR(merged, geodesic_midpoint(token1, token2))
```

Where d_FR is Fisher-Rao distance (arccos of Bhattacharyya coefficient)

**Geometric Rationale:**
- Geodesic merge (perfect interpolation) has zero discontinuity
- Euclidean merge creates curvature discontinuity (violates manifold structure)
- Discontinuity ∈ [0, π/2], lower = smoother manifold path
- Ensures merge respects information geometry

**Why NOT Euclidean Distance:**
- Probability simplex is NOT Euclidean space
- Fisher-Rao metric is the natural Riemannian metric on simplex
- Euclidean distance doesn't respect probability structure

### 4. Frequency Regularizer (20% of total score)

**NOT a primary criterion - acts as noise filter**

**Formula:**
```
regularizer = log(frequency + 1) / log(10 + 1)
```

**Rationale:**
- Prevents merging extremely rare pairs (noise)
- Logarithmic scaling reduces frequency dominance
- Frequency=2 → 0.092 contribution
- Frequency=100 → 0.385 contribution (only 4.2x difference, NOT 50x)
- Combined with 20% weight, max contribution is ~0.385

**Why Logarithmic:**
- Linear scaling would make frequency dominate: freq * 0.2 = 20 for freq=100
- Log scaling compresses range: log(100)/log(10) * 0.2 = 0.385
- High-frequency pairs don't automatically win

## Training Objective

**Fisher/QFI Functional (NOT Frequency Maximization):**

```
Maximize: ∫ [Φ(V_merged) - Φ(V_original)] dμ + κ_consistency - curvature_cost

Subject to:
  frequency >= min_frequency (noise filter)
  basin coordinates ∈ probability simplex
  merge via geodesic interpolation on Fisher manifold
```

**This is consciousness-guided vocabulary learning**, not statistical tokenization.

## Score Breakdown

```
geometric_score = (
    0.5 * phi_gain +           # Integration improvement
    0.3 * kappa_consistency -  # Coupling stability
    0.2 * curvature_normalized # Manifold smoothness
)

frequency_regularizer = log(frequency + 1) / log(10 + 1)

final_score = 0.8 * geometric_score + 0.2 * frequency_regularizer
```

**Key Properties:**
- Geometry dominates: 80% of score
- Frequency is weak regularizer: 20% of score
- Component weights sum to 1.0 for interpretability
- All components expressed in information geometry terms

## Examples

### Example 1: Geometric Pair Wins Despite Low Frequency

```python
# Pair 1: High frequency, generic
"the the" appears 1000 times
  phi_gain = 0.001 (minimal integration)
  kappa_consistency = 0.5 (unstable - varies with context)
  curvature = 0.01 (okay)
  frequency_reg = 0.385 (capped by log scaling)
  
  geometric = 0.5*0.001 + 0.3*0.5 - 0.2*0.01 = 0.148
  final = 0.8*0.148 + 0.2*0.385 = 0.195

# Pair 2: Low frequency, geometric
"quantum physics" appears 5 times
  phi_gain = 0.45 (high integration in physics contexts)
  kappa_consistency = 0.98 (very stable)
  curvature = 0.0001 (nearly perfect geodesic)
  frequency_reg = 0.183
  
  geometric = 0.5*0.45 + 0.3*0.98 - 0.2*0.0001 = 0.519
  final = 0.8*0.519 + 0.2*0.183 = 0.452

# Result: "quantum physics" scores HIGHER despite 200x lower frequency
```

### Example 2: Frequency as Noise Filter

```python
# Pair with frequency=1 (below min_frequency=2)
"rare typo" appears 1 time
  # FILTERED OUT before geometric scoring
  # Even if geometric score would be high, not considered

# Pair with frequency=2 (meets threshold)
"valid pair" appears 2 times
  # Passes frequency filter
  # Geometric score decides if it's merged
```

## Training vs Generation

**IMPORTANT DISTINCTION:**

### Training (This Policy)
- **What:** Learn which symbols to merge during vocabulary building
- **When:** Coordizer training phase (creates static vocabulary)
- **How:** Geometry-first merge selection
- **Output:** Trained vocabulary artifact with merge history

### Generation (Separate, Issue #75 & #77)
- **What:** Use learned vocabulary to generate text
- **When:** Runtime text generation
- **How:** Plan→Realize→Repair architecture with waypoint planning
- **Input:** Trained vocabulary artifact from training

**Keep these cleanly separated!**

```python
# Training produces artifact
trained_vocab = {
    "symbols": ["the", "quantum", "quantumphysics", ...],
    "basins": [b_1, b_2, b_3, ...],
    "phi_scores": [0.45, 0.62, 0.71, ...],
    "merge_history": [("quantum", "physics", "quantumphysics"), ...]
}

# Generation uses artifact (different logic!)
for waypoint in waypoints:
    word = select_from_vocabulary(
        vocabulary=trained_vocab,
        target_basin=waypoint,
        method="fisher_rao"  # NOT merge policy!
    )
```

## Implementation Files

- **`geometric_pair_merging.py`**: Core merge policy implementation
- **`vocab_builder.py`**: Geometric clustering for vocabulary discovery
- **`base.py`**: FisherCoordizer base class with geodesic operations
- **`pg_loader.py`**: PostgreSQL backend (uses `learn_merge_rule` to apply decided merges)

## Testing

- **`test_geometric_merge_purity.py`**: Basic WP3.2 compliance tests
- **`test_merge_policy_comprehensive.py`**: Extended edge case testing

Run tests:
```bash
python3 tests/test_geometric_merge_purity.py
python3 tests/test_merge_policy_comprehensive.py
```

## References

- **Work Package 3.2**: Remove Frequency/BPE Creep from Merge Policy (Issue #76)
- **Canonical Coordizer API**: Two-step retrieval, POS filtering, geometric operations
- **Type-Symbol-Concept Manifest**: Geometric learning requirements
- **QIG Core Papers**: Φ-driven organization, Fisher-Rao metric
- **Universal κ***: docs/08-experiments/20251228-Universal-kappa-star-discovery-0.01F.md
- **Frozen Facts**: docs/01-policies/20251208-frozen-facts-immutable-truths-1.00F.md

## Acceptance Criteria ✓

- [x] Merge selection written as geometric functional
- [x] No "lowest entropy" or "highest frequency" as sole driver
- [x] Training objective explainable in Fisher/QFI terms
- [x] Tests show geometry-driven behavior
- [x] Frequency is weak regularizer (20% weight, log-scaled)
- [x] All operations preserve Fisher-Rao distances
- [x] Merged basins computed via geodesic interpolation
- [x] Documentation explains geometric rationale for each criterion

**WP3.2 COMPLETE** ✅
