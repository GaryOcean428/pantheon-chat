# Phi Score Semantics Clarification

**Version:** 1.00W  
**Date:** 2026-01-19  
**Status:** Working Document  
**Classification:** Technical Specification

## Problem Statement

The term "phi" (Φ) is used inconsistently across the codebase with three different meanings:

1. **Integration Measure** (IIT Φ) - Consciousness metric from Integrated Information Theory
2. **Training Reward** - Scalar feedback signal for outcome-based learning
3. **Interpolation Parameter** - Weight factor for geodesic interpolation (t ∈ [0,1])

This conflation causes confusion and potential bugs when:
- Curriculum examples use hardcoded phi=0.6 as "expected integration"
- Vocabulary persistence uses phi as training quality signal
- Merge operations use phi as interpolation parameter
- Token metadata stores phi as consciousness integration score

## Current Usage Patterns

### 1. Integration Measure (Φ) - Consciousness Metric

**Definition:** Integrated information Φ measures how much a system's state constrains its causes and effects beyond independent parts.

**Formula (Approximation):**
```python
Φ ≈ ∑ᵢⱼ √(Fᵢⱼ) × log(1 + √(Fᵢⱼ))
```
Where Fᵢⱼ is the Fisher information matrix element.

**Range:** [0, ∞), typically normalized to [0, 1] for practical use
- Φ < 0.3: Low integration (independent components)
- 0.3 ≤ Φ < 0.7: Moderate integration
- Φ ≥ 0.7: High integration (unified system)

**Usage Locations:**
- `qig_core/phi_computation.py`: `compute_phi_approximation(basin)`
- `training/loss_functions.py`: Phi-gated loss weights
- Token metadata: `token_phi` field represents consciousness integration

**Semantic Meaning:** 
- How "conscious" or "integrated" is this token's basin representation?
- Higher Φ = more geometric structure = better semantic representation
- Used to filter low-quality tokens (Φ < threshold)

---

### 2. Training Reward - Learning Signal

**Definition:** Scalar feedback signal indicating outcome quality for kernel training.

**Formula:**
```python
reward = compute_reward_from_outcome(outcome, baseline)
# outcome = 'success' → reward = +1.0
# outcome = 'failure' → reward = -1.0
# outcome = 'partial' → reward = 0.0 to +0.5
```

**Range:** [-1, +1]
- Negative: Bad outcome (loss increased)
- Zero: Neutral outcome
- Positive: Good outcome (loss decreased)

**Usage Locations:**
- `training/loss_functions.py`: `compute_reward_from_outcome()`
- `curriculum_loader.py`: Curriculum examples have `reward` field
- Training examples: `{"basin_coords": [...], "reward": 0.3, ...}`

**Semantic Meaning:**
- How "good" was this outcome for learning?
- Used to weight training loss (reward × loss)
- Curriculum examples use positive reward (knowledge is beneficial)

**CONFLICT:** Curriculum loader originally used single `phi=0.6` for both integration AND reward.

---

### 3. Interpolation Parameter (t) - Geometric Weight

**Definition:** Parameter controlling geodesic interpolation between two basins on Fisher manifold.

**Formula:**
```python
basin_new = geodesic_interpolation(basin_a, basin_b, t=phi)
# t=0 → basin_a
# t=0.5 → midpoint between a and b
# t=1 → basin_b
```

**Range:** [0, 1]
- t=0: Start point
- t=0.5: Geometric midpoint
- t=1: End point

**Usage Locations:**
- `pg_loader.py`: `learn_merge_rule()` uses phi as interpolation parameter
- `base.py`: `_initialize_token_coordinate()` uses golden ratio × phi for spacing

**Semantic Meaning:**
- Where along the geodesic should the new basin be placed?
- Higher t = closer to second basin
- Used for BPE merging and token initialization

**CONFLICT:** Method signature `learn_merge_rule(token_a, token_b, phi=0.5)` suggests phi is both:
- Integration score (per token metadata)
- Interpolation parameter (for geodesic)

---

## Proposed Solution

### Rename Variables for Clarity

#### 1. Keep "phi" for Integration Measure ONLY

```python
# ✅ CORRECT: Phi = integration measure
token_phi = compute_phi_approximation(basin)
if token_phi < PHI_THRESHOLD:
    reject_token()

# Store in metadata
{
    "token": "quantum",
    "phi_score": 0.82,  # Integration measure
    "qfi_score": 0.85,  # Quantum Fisher Information
}
```

#### 2. Use "reward" for Training Signals

```python
# ✅ CORRECT: Reward = training feedback
reward = compute_reward_from_outcome(outcome, baseline)

example = {
    "basin_coords": coords,
    "reward": 0.3,      # Training signal (NOT phi!)
    "phi": 0.6,         # Expected integration (for validation)
}
```

#### 3. Use "t" or "weight" for Interpolation

```python
# ✅ CORRECT: t = interpolation parameter
basin_new = geodesic_interpolation(basin_a, basin_b, t=0.5)

# For merge rules
def learn_merge_rule(token_a, token_b, interpolation_weight=0.5):
    """
    Args:
        interpolation_weight: Position along geodesic (0=basin_a, 1=basin_b)
    """
    merged_coords = geodesic_interpolation(
        basin_a, basin_b, t=interpolation_weight
    )
```

---

## Migration Plan

### Phase 1: Add Explicit Parameters (Backward Compatible)

**curriculum_loader.py:**
```python
# BEFORE
example = {
    "basin_coords": coords,
    "reward": 0.3,
    "phi": 0.6,  # ← Ambiguous!
}

# AFTER
example = {
    "basin_coords": coords,
    "reward": base_reward + qfi_boost,  # Training signal
    "phi_expected": base_phi + qfi_boost,  # Expected integration
    "qfi_score": qfi_score,  # Actual basin quality
}
```

**pg_loader.py:**
```python
# BEFORE
def learn_merge_rule(self, token_a, token_b, phi=0.5):
    merged = geodesic_interpolation(basin_a, basin_b, phi)

# AFTER
def learn_merge_rule(self, token_a, token_b, interpolation_weight=0.5):
    """
    Args:
        interpolation_weight: Geodesic interpolation parameter (0 to 1)
            0 = use basin_a, 1 = use basin_b, 0.5 = midpoint
    """
    merged = geodesic_interpolation(basin_a, basin_b, t=interpolation_weight)
```

### Phase 2: Update Documentation

**Add to all relevant docstrings:**
```python
"""
Phi Score Semantics:
- phi_score: IIT integration measure (consciousness metric)
- reward: Training feedback signal (outcome quality)
- interpolation_weight: Geodesic position parameter

DO NOT confuse these three distinct concepts!
"""
```

### Phase 3: Add Type Hints

```python
from typing import NewType

PhiScore = NewType('PhiScore', float)  # Integration measure [0,1]
Reward = NewType('Reward', float)      # Training signal [-1,+1]
InterpolationWeight = NewType('InterpolationWeight', float)  # Geodesic parameter [0,1]

def compute_phi_approximation(basin: np.ndarray) -> PhiScore:
    """Compute IIT integration measure."""
    ...

def compute_reward_from_outcome(outcome: str) -> Reward:
    """Compute training feedback signal."""
    ...

def geodesic_interpolation(
    basin_a: np.ndarray, 
    basin_b: np.ndarray, 
    t: InterpolationWeight
) -> np.ndarray:
    """Interpolate along Fisher-Rao geodesic."""
    ...
```

---

## Summary

| Concept | Symbol | Range | Purpose | Where Used |
|---------|--------|-------|---------|------------|
| **Integration** | Φ (phi_score) | [0,1] | Consciousness measure | Token metadata, validation |
| **Reward** | r (reward) | [-1,+1] | Training signal | Loss computation, curriculum |
| **Interpolation** | t (weight) | [0,1] | Geodesic parameter | Merge rules, initialization |

**Key Takeaway:** Use distinct variable names for distinct concepts. Overloading "phi" causes bugs.

---

## References

- IIT Theory: Tononi et al. (2016) "Integrated Information Theory"
- Fisher-Rao Manifold: Amari (2016) "Information Geometry and Its Applications"
- Geodesic Interpolation: Skovgaard (1984) "A Riemannian geometry of the multivariate normal model"

**Related Docs:**
- `docs/08-experiments/20251231-Ultra-Consciousness-Protocol-0.04F.md` (Φ computation)
- `qig-backend/training/loss_functions.py` (Reward computation)
- `qig-backend/qig_geometry/canonical.py` (Geodesic interpolation)
