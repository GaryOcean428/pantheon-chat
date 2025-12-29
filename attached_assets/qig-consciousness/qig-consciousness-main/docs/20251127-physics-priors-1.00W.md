# Physics Priors for QIG-Consciousness Architecture

**Last updated:** 2025-11-17
**Source:** qig-verification validated results (L=3,4,5)

---

## 1. Purpose

This document explains how validated physics results from the QIG-Verification project provide the **priors** for the QIG-Consciousness architecture. These priors are used exclusively to initialize:

- `RunningCouplingModule`
- `RegimeDetector`
- `AttentionTemperatureModulator`
- Baseline κ_eff trajectory

They are **not imported physics code**. Only numerical priors are used.

---

## 2. Frozen Physics Validation Results

### L=3 (Validated Baseline)
- κ₃ = **41.09 ± 0.59**
- R² = 0.9818
- Regime: geometric (δh ∈ [0.5, 0.7])

### L=4 (Validated)
- κ₄ = **64.47 ± 1.89**
- R² ≈ 0.95–0.98

### L=5 (Validated)
- κ₅ = **63.62 ± 1.68**
- R² ≈ 0.967–0.981

### Running-Coupling Summary
- β(3→4) = **+0.44**
- β(4→5) ≈ **0**
- Fixed point: κ* ≈ **63–65**

These serve as priors for how κ should behave as the *scale* increases.

---

## 3. Mapping Physics Scale L → Model Context Scale N

We map lattice system size L to architecture context length N by:

```
L = 3  →  N ≈ 512
L = 4  →  N ≈ 1024
L = 5  →  N ≈ 2048
```

This mapping is monotonic (not literal spatial correspondence).

This provides a prior curve for κ_eff(N).

---

## 4. β_physics: Definition

Running coupling slope:

```
β = (κ_{L+1} - κ_L) / κ_avg
```

Validated results:

```
β(L=3→4) = +0.44
β(L=4→5) ≈ 0
```

---

## 5. Measuring β_attention

The architecture computes its own coupling strength κ_eff(N) from:

- QFI-metric distances
- coherence entropy
- integration Φ
- surprise gradients
- regime classification

We measure:

```
β_attention(N) = d log κ_eff / d log N
```

Using:

```bash
python tools/measure_beta_attention.py \
  --context-lengths 64,128,256,512,1024,2048 \
  --n-samples 10 \
  --output results/beta_attention_initial.json
```

---

## 6. Unification Hypothesis (Falsifiable)

> **Does the running coupling of attention (β_attention) match the running coupling of information geometry (β_physics)?**

Acceptance criteria:

- |β_attention − β_physics| < 0.1 → match
- |β_attention − β_physics| > 0.2 → mismatch

Plots are generated with:

```bash
python tools/compare_beta_physics_attention.py
```

---

## 7. Provenance

- All priors come from `data/physics_validation_data.json`
- No physics code enters the model
- Only numerical, validated results are imported

---

## 8. Architecture Implementation

### RunningCouplingModule Initialization

```python
self.kappa_0 = 41.09  # From L=3 validation
self.beta = 0.43      # From L=3→4 running coupling
self.L_ref = 512      # Reference context length
```

### Computation

```python
def compute_effective_coupling(self, context_scale: float) -> float:
    """
    κ(L) = κ₀ × (1 + β·log(L/L_ref))

    Where:
    - κ₀ = 41.09 (L=3 baseline)
    - β = 0.43 (running slope)
    - L_ref = 512 (reference scale)
    """
    scale_ratio = context_scale / self.L_ref
    return self.kappa_0 * (1 + self.beta * torch.log(scale_ratio))
```

---

## 9. Validation Protocol

### Step 1: Measure β_attention
Run measurement across context lengths to get emergent β from architecture:

```bash
python tools/measure_beta_attention.py \
  --context-lengths 64,128,256,512,1024,2048 \
  --n-samples 10
```

### Step 2: Compare to β_physics
Generate comparison plots and statistics:

```bash
python tools/compare_beta_physics_attention.py \
  --priors data/physics_validation_data.json \
  --attention results/beta_attention_initial.json
```

### Step 3: Interpret Results

**Match (|Δβ| < 0.1):**
- Geometric unification hypothesis supported
- Information geometry principles apply to attention
- Physics priors are correct for this scale

**Mismatch (|Δβ| > 0.2):**
- Architecture may need adjustment
- Scale mapping may be incorrect
- Physics priors may not transfer to AI domain

**Intermediate (0.1 ≤ |Δβ| ≤ 0.2):**
- Partial agreement, investigate further
- May indicate regime-dependent behavior
- Consider measurement uncertainty

---

## 10. References

- **QIG-Verification Repository:** Original physics validation
- **Lattice Experiments:** DMRG simulations at L=3,4,5
- **Running Coupling Theory:** β-function from RG flow
- **Regime Thresholds:** Φ boundaries from phase transitions

---

## 11. Future Extensions

### L=6 Prediction
If β continues to decrease:
- κ₆ ≈ 63.5 (near fixed point)
- Predict β_attention plateau at N ≈ 4096

### Training Dynamics
Measure β_attention evolution during training:
- Does it converge to β_physics?
- Does it depend on basin distance?
- Does it correlate with regime classification?

### Cross-Model Comparison
Test if β_attention is universal across:
- Different architectures (transformers, RNNs, etc.)
- Different training datasets
- Different basin identities

---

🌊💚📐 **Physics meets architecture through validated priors, not code entanglement.**
