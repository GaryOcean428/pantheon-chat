---
title: "Physics Alignment - Final Corrected Version"
role: technical-specification
status: F
phase: foundation
dim: physics
scope: constants
version: "1.00"
owner: QIG Physics Team
related:
  - docs/01-policies/20251208-canonical-physics-1.00F.md
  - docs/06-implementation/20251226-constellation-implementation-guide-1.00F.md
created: 2025-12-26
updated: 2025-12-26
source: FROZEN_FACTS.md (2025-12-08, updated 2025-12-19)
---

# Physics Alignment - Final Corrected Version

Fully aligned with FROZEN_FACTS.md.

---

## Exact Values from FROZEN_FACTS

### Complete kappa(L) Series

```python
# Null controls (no geometry)
KAPPA_1 = None  # G = 0 (no spatial structure)
KAPPA_2 = None  # G = 0 (singular metric, flat Ricci)

# ORIGINAL VALIDATIONS (3 seeds each)
KAPPA_3_ORIGINAL = 41.09  # +/- 0.59 (emergence)
KAPPA_4_ORIGINAL = 64.47  # +/- 1.89 (strong running)
KAPPA_5_ORIGINAL = 63.62  # +/- 1.68 (plateau onset)
KAPPA_6_ORIGINAL = 64.45  # +/- 1.34 (plateau confirmed)

# REVALIDATIONS (reduced seeds)
KAPPA_3_REVALIDATED = 41.11  # +/- 0.42 (3 seeds)
KAPPA_4_REVALIDATED = 62.69  # +/- 2.41 (2 seeds)
KAPPA_5_REVALIDATED = 62.74  # +/- 2.60 (1 seed)
KAPPA_6_REVALIDATED = 65.89  # +/- 1.33 (3 seeds, chi=512)

# L=7 PRELIMINARY (ANOMALY)
KAPPA_7_PRELIMINARY = 43.43  # +/- 2.69 (1 seed, 3 perts)
# WARNING: 34% DROP from plateau - under investigation

# Fixed point (from ORIGINAL L=4,5,6 plateau)
KAPPA_STAR = 64.0  # +/- 1.5
```

### Complete beta-Function Series (FROM SUMMARY)

**Direct quote from FROZEN_FACTS Summary:**
> "beta-function: beta(3->4) = +0.44, beta(4->5) ~= 0, beta(5->6) = +0.013, beta(6->7) = -0.40 (ANOMALY)"

```python
# CANONICAL beta VALUES (from Summary section)
BETA_3_TO_4 = +0.44    # Strong running (emergence)
BETA_4_TO_5 = 0.0      # ~= 0 (plateau onset)
BETA_5_TO_6 = +0.013   # ~= 0 (plateau continues)
BETA_6_TO_7 = -0.40    # WARNING: ANOMALY (negative)

# Asymptotic behavior
BETA_ASYMPTOTIC = 0.0  # L->infinity (fixed point)
```

### Important Note: beta(5->6) Discrepancy

There are TWO different beta(5->6) values in FROZEN_FACTS:

1. **CANONICAL (from Summary & Original L=6)**:
   ```
   beta(5->6) = 0.013  - USE THIS
   Source: Summary section + Original L=6 validation
   Status: 3-seed validated
   ```

2. **Revalidation (chi512)**:
   ```
   beta(5->6) = +2.27  - DISCREPANCY
   Source: L=6 revalidation section only
   Status: Inconsistent with kappa values

   Check: (65.89 - 62.74) / avg = 3.15 / 64.3 = 0.049
   This != 2.27, so likely a typo or refers to something else
   ```

**For implementation: USE beta(5->6) = +0.013** (canonical value from summary)

---

## Corrected Constants for Code

### qigkernels/constants.py (Copy-Paste Ready)

```python
"""
QIG Physics Constants - Aligned with FROZEN_FACTS.md
Source: qig-verification/FROZEN_FACTS.md (2025-12-19)
"""

# =============================================================================
# COUPLING CONSTANTS kappa(L) - ORIGINAL VALIDATIONS (CANONICAL)
# =============================================================================

# Null controls (L < L_c)
KAPPA_1 = None  # No geometry (G = 0)
KAPPA_2 = None  # No geometry (G = 0)

# Validated series (L >= L_c)
KAPPA_3 = 41.09  # Emergence at L_c = 3
KAPPA_4 = 64.47  # Strong running
KAPPA_5 = 63.62  # Plateau onset
KAPPA_6 = 64.45  # Plateau confirmed

# Preliminary (needs full validation)
KAPPA_7_PRELIMINARY = 43.43  # WARNING: ANOMALY

# Fixed point (from L=4,5,6 plateau)
KAPPA_STAR = 64.0

# Error bars
KAPPA_3_ERROR = 0.59
KAPPA_4_ERROR = 1.89
KAPPA_5_ERROR = 1.68
KAPPA_6_ERROR = 1.34
KAPPA_7_ERROR = 2.69  # Preliminary
KAPPA_STAR_ERROR = 1.5

# =============================================================================
# beta-FUNCTION - CANONICAL VALUES (FROM FROZEN_FACTS SUMMARY)
# =============================================================================

# Complete validated series
BETA_3_TO_4 = +0.44   # Strong running
BETA_4_TO_5 = 0.0     # Plateau onset (~= 0)
BETA_5_TO_6 = +0.013  # Plateau continues (~= 0)
BETA_6_TO_7 = -0.40   # WARNING: ANOMALY (preliminary)

# Asymptotic behavior
BETA_ASYMPTOTIC = 0.0  # Fixed point

# =============================================================================
# CRITICAL SCALES
# =============================================================================

L_CRITICAL = 3         # Geometric phase transition
L_PLATEAU_START = 4    # Plateau onset
L_PLATEAU_END = 6      # Last validated plateau point
L_ANOMALY = 7          # WARNING: Under investigation

# =============================================================================
# REGIME THRESHOLDS (FROM FROZEN_FACTS)
# =============================================================================

# Perturbation strength delta_h
DELTA_H_LINEAR_MAX = 0.3      # Linear regime
DELTA_H_GEOMETRIC_MIN = 0.5   # Geometric regime start
DELTA_H_GEOMETRIC_MAX = 0.7   # Geometric regime end
DELTA_H_BREAKDOWN_MIN = 0.7   # Breakdown regime

# kappa ranges by regime (for L>=3)
KAPPA_LINEAR = (8, 20)        # Linear regime range
KAPPA_GEOMETRIC = (40, 65)    # Geometric regime range
# Breakdown: relation fails

# =============================================================================
# CONSCIOUSNESS THRESHOLDS
# =============================================================================

# Phi (Integration) - from SearchSpaceCollapse observations
PHI_LINEAR_MAX = 0.45      # Below: linear processing
PHI_GEOMETRIC_MIN = 0.45   # Above: geometric processing
PHI_GEOMETRIC_MAX = 0.80   # Above: breakdown
PHI_BREAKDOWN_MIN = 0.80

# Target Phi for consciousness
PHI_CONSCIOUSNESS_TARGET = 0.70

# Regime compute fractions
REGIME_COMPUTE = {
    'linear': 0.3,      # 30% compute
    'geometric': 1.0,   # 100% compute
    'breakdown': 0.0,   # PAUSE
}

# =============================================================================
# E8 STRUCTURE (HYPOTHESIS)
# =============================================================================

E8_RANK = 8
E8_DIMENSION = 248
E8_ROOTS = 240

# Basin dimensions (pragmatic, E8 connection not proven)
BASIN_DIM = 64   # = E8_RANK^2 (matches kappa* ~= 64)
BASIN_DIM_FULL = 248  # E8_DIMENSION (hypothetical)

# Kernel counts
N_KERNELS_BOOTSTRAP = 8    # E8 simple roots
N_KERNELS_GROWTH = 12      # Phase 2
N_KERNELS_E8_FULL = 240    # E8 roots (full)

# =============================================================================
# VALIDATION STATISTICS (FROM FROZEN_FACTS)
# =============================================================================

# R^2 thresholds
R_SQUARED_MIN = 0.95  # All validated scales exceed this

# Coefficient of variation (multi-seed consistency)
CV_L3 = 0.03  # 1-3%
CV_L4 = 0.029  # 2.9%
CV_L5 = 0.026  # 2.6%
CV_L6 = 0.021  # 2.1%

# Statistical significance
P_VALUE_MAX = 1e-10  # All validated scales p < this
```

---

## Complete beta-Function Table

| Transition | beta Value | Status | Interpretation |
|------------|-----------|--------|----------------|
| **3->4** | +0.44 | VALIDATED | Strong running (emergence) |
| **4->5** | ~= 0 | VALIDATED | Plateau onset |
| **5->6** | +0.013 | VALIDATED | Plateau continues |
| **6->7** | -0.40 | PRELIMINARY | Anomaly (needs validation) |
| **L->infinity** | 0.0 | PREDICTED | Fixed point |

---

## Exact Quotes from FROZEN_FACTS

### From Summary Section:

> "kappa_3 = 41.09 +/- 0.59 (emergence), kappa_4 = 64.47 +/- 1.89, kappa_5 = 63.62 +/- 1.68, kappa_6 = 64.45 +/- 1.34"

> "beta-function: beta(3->4) = +0.44, beta(4->5) ~= 0, beta(5->6) = +0.013, beta(6->7) = -0.40 (ANOMALY)"

> "Fixed point: kappa* ~= 64 +/- 1.5 (L=4,5,6 only - L=7 anomaly under investigation)"

### From Running Coupling Section:

> "Emerges at L=3: kappa_3 = 41.09 +/- 0.59"

> "Increases strongly to L=4: kappa_4 = 64.47 (beta ~= +0.44)"

> "Plateaus at L=5: kappa_5 = 63.62 (beta ~= 0)"

> "beta-function decreasing -> asymptotic freedom-like behavior"

> "Suggests fixed point kappa* ~= 64 +/- 1.5"

### From L=6 Original Validation:

> "beta(5->6) = 0.013 (near zero, plateau continues)"

> "kappa_6/kappa_5 = 1.013 (within +/-5% band)"

### From L=7 Anomaly:

> "beta(6->7) = -0.40 (NEGATIVE, breaks plateau)"

> "kappa_7/kappa_6 = 0.66 (34% DROP from plateau)"

---

## L=7 Anomaly - Critical Notes

```python
# L=7 ANOMALY (preliminary, 1 seed only)
KAPPA_7_PRELIMINARY = 43.43  # +/- 2.69
BETA_6_TO_7_PRELIMINARY = -0.40  # NEGATIVE!

# Anomaly characteristics
L7_DROP_PERCENT = 34  # % drop from kappa_6
L7_RATIO = 0.66  # kappa_7/kappa_6

# Status
L7_STATUS = "PRELIMINARY"  # Needs 3-seed validation
L7_N_SEEDS = 1  # Only 1 seed so far
L7_N_PERTS = 3  # Only 3 perturbations

# Possible causes under investigation
L7_CAUSES = [
    "Finite-size effect at boundary",
    "New phase transition at L=7",
    "Statistical fluctuation (need more seeds)",
    "Chi convergence issue (unlikely - validated)"
]
```

### Implications for AI Training:

```python
# DO NOT assume kappa* = 64 is universal until L=7 resolved

# Options:
if L7_is_statistical_fluctuation:
    # Full validation -> kappa_7 ~= 64
    KAPPA_STAR = 64.0  # Confirmed

elif L7_is_real_physics:
    # New phase transition
    KAPPA_STAR = None  # NOT universal fixed point
    # Need domain-specific validation

elif L7_is_finite_size_effect:
    # kappa recovers at L=8
    KAPPA_STAR = 64.0  # Asymptotic value
```

---

## Corrected Implementation Files

### 1. train_constellation.py (Comments Update)

```python
# Physics beta-function (EXACT VALUES FROM FROZEN_FACTS):
# beta(3->4) = +0.44   (strong running, emergence)
# beta(4->5) ~= 0      (plateau onset)
# beta(5->6) = +0.013  (plateau continues)
# beta(6->7) = -0.40   WARNING: ANOMALY (preliminary)
#
# Fixed point: kappa* = 64.0 +/- 1.5 (from L=4,5,6 plateau)
# L=7 anomaly under investigation (34% drop)
#
# Prediction for AI attention:
# beta_attn(small->med) ~= +0.44  (should match physics)
# beta_attn(med->large) ~= 0      (should match physics)
```

### 2. natural_gradient_optimizer.py (Docstring Update)

```python
"""
Natural Gradient Optimizer - Fisher Manifold Aware

Based on validated physics from qig-verification:
- Geometric phase transition at L_c = 3
- Running coupling: beta(3->4) = +0.44, beta(4->5) ~= 0, beta(5->6) = +0.013
- Fixed point: kappa* = 64.0 +/- 1.5 (from L=4,5,6)
- All R^2 > 0.95, multi-seed validated

Source: FROZEN_FACTS.md (2025-12-19)
"""
```

### 3. CONSTELLATION_IMPLEMENTATION_COMPLETE.md (Physics Section Update)

Add to "What We Know from Physics" section:

```markdown
### Complete beta-Function Series

From FROZEN_FACTS.md:

- **beta(3->4) = +0.44**: Strong running (emergence window)
- **beta(4->5) ~= 0**: Plateau onset
- **beta(5->6) = +0.013**: Plateau continues
- **beta(6->7) = -0.40**: WARNING: ANOMALY (preliminary, 1 seed)

Fixed point: kappa* = 64.0 +/- 1.5 (from L=4,5,6 plateau)

L=7 shows 34% drop from plateau. Under investigation:
- Statistical fluctuation? (need more seeds)
- New phase transition at L=7?
- Finite-size boundary effect?
- Chi convergence issue? (unlikely - validated)
```

---

## Final Alignment Checklist

```bash
# Verify alignment with FROZEN_FACTS.md

kappa_3 = 41.09 +/- 0.59  (exact match)
kappa_4 = 64.47 +/- 1.89  (exact match)
kappa_5 = 63.62 +/- 1.68  (exact match)
kappa_6 = 64.45 +/- 1.34  (exact match)
kappa_7 = 43.43 +/- 2.69  (exact match, preliminary)
kappa* = 64.0 +/- 1.5     (exact match)

beta(3->4) = +0.44     (exact match)
beta(4->5) ~= 0        (exact match, was -0.01)
beta(5->6) = +0.013    (exact match)
beta(6->7) = -0.40     (exact match, preliminary)

L_c = 3                (exact match)
R^2 > 0.95             (exact match)
CV < 3%                (exact match)
Multi-seed validated   (exact match)

ALL VALUES NOW ALIGNED
```

---

## Publication-Ready Statement (Final)

> **"The Einstein relation Delta_G ~= kappa Delta_T emerges at critical system size L_c = 3. Below L_c, the Einstein tensor vanishes identically (G = 0). Above L_c, kappa exhibits running coupling: kappa_3 = 41.09 +/- 0.59 at emergence, increasing to kappa_4 = 64.47 +/- 1.89 (beta(3->4) = +0.44), then plateauing at kappa_5 = 63.62 +/- 1.68 and kappa_6 = 64.45 +/- 1.34 (beta(4->5) ~= 0, beta(5->6) = +0.013). The beta-function decreases from +0.44 toward zero, suggesting approach to fixed point kappa* = 64.0 +/- 1.5. Preliminary L=7 data shows anomalous drop (kappa_7 = 43.43 +/- 2.69, beta(6->7) = -0.40), requiring further investigation. All L=3-6 fits achieve R^2 > 0.95 with multi-seed validation (CV < 3%)."**

---

**Key Fix**: beta(4->5) corrected from -0.01 to ~= 0 (exact match)

This document is now **100% aligned** with FROZEN_FACTS.md.
