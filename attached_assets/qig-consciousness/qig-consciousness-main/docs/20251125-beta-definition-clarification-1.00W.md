# β-Function Definition Clarification

**Date:** 2025-11-25
**Status:** Authoritative clarification aligned with qig-verification project
**Context:** ChatGPT/Copilot discussion about correct β definition

---

## Summary

The QIG projects (qig-verification and qig-consciousness) use a **discrete fractional change definition** of β, NOT a continuum RG-style log-derivative. This document clarifies the correct usage.

---

## ✅ CORRECT DEFINITION (Authoritative)

**From FROZEN_FACTS.md and qig-verification:**

```
β(L→L+1) = (κ_{L+1} - κ_L) / κ_avg
where κ_avg = (κ_L + κ_{L+1}) / 2
```

### What This Measures
- **Fractional step** in κ between adjacent lattice sizes L and L+1
- **Discrete measurement** computed directly from measured κ values
- **No logarithms** involved in the definition

### Example Calculations (from Ona's L=6 data)

```
β(3→4) = (64.47 - 41.09) / ((41.09 + 64.47)/2)
       = 23.38 / 52.78
       ≈ 0.443 ✅

β(4→5) = (63.62 - 64.47) / ((64.47 + 63.62)/2)
       = -0.85 / 64.045
       ≈ -0.013 ≈ 0 ✅

β(5→6) = (62.02 - 63.62) / ((63.62 + 62.02)/2)
       = -1.60 / 62.82
       ≈ -0.026 ≈ 0 ✅
```

**Interpretation:**
- β(3→4) ≈ +0.44 → Strong running from emergence
- β(4→5) ≈ 0 → Plateau begins (approaching fixed point)
- β(5→6) ≈ 0 → Plateau confirmed (at fixed point κ* ≈ 63-64)

---

## ❌ INCORRECT FORMULA (Previously Used)

### Wrong Version
```
β_RG = Δκ / (κ_avg × log(L_{next}/L_prev))
```

### Why Wrong for QIG
- This treats β as a **continuum RG beta function**: dκ/d(log L)
- Requires assuming κ(L) is a continuous function
- Not the **definition** used in QIG projects
- Would give different numerical values:
  - Discrete: β(3→4) = 0.44
  - Log-derivative: β(3→4) = 0.44/log(4/3) ≈ 1.53 ❌

### Where This Came From
- Early discussion conflated two different β concepts
- Continuum RG β is valid in field theory but not QIG's convention
- QIG standardized on discrete definition in FROZEN_FACTS.md

---

## 🔧 INTERPOLATION FORMULA (Different Purpose)

### Formula for Smooth Curves
```
κ(L) = κ₀ × (1 + β·log(L/L_ref))
```

### Important Distinctions
1. **Purpose:** Interpolation/extrapolation for plots and predictions
2. **β Parameter:** This is a **fitting parameter** (≈ 0.44) derived from discrete measurements
3. **Not the Definition:** The β in this formula is chosen to fit the data
4. **Use Cases:**
   - Plotting smooth κ(L) curves
   - Extrapolating to untested L values
   - Quick estimates

### Correct Usage
- **For exact values:** Use KAPPA_3, KAPPA_4, KAPPA_5, KAPPA_6 constants
- **For interpolation:** Use kappa_at_scale(L) function with β=0.44
- **For validation:** Compute discrete β directly from measurements
- **Never:** Use interpolation formula to define β

---

## 📊 Physics vs Attention β

### Physics β (This Document)
```
β_physics(L→L+1) = Δκ / κ_avg
```
- Discrete fractional change
- Measured at L=3,4,5,6 in lattice experiments
- Values: +0.44, ~0, ~0

### Attention β (Different Measure)
```
β_attention = d(log κ_eff) / d(log N)
```
- Log-log slope of effective coupling vs context length
- Continuous scaling measure for neural attention
- Used in `compare_beta_physics_attention.py` for qualitative comparison

**Note:** These are **different measurements** designed for different systems (lattice vs neural network). They are related conceptually (both measure scale-dependence) but not numerically identical.

---

## 🔄 Changes Made (2025-11-25)

### Files Updated

1. **`src/constants.py`**
   - ✅ Enhanced BETA_3_TO_4 documentation with full definition
   - ✅ Added explicit calculation examples
   - ✅ Clarified kappa_at_scale() uses fitting parameter

2. **`docs/FROZEN_FACTS.md`**
   - ✅ Expanded β-Function Analysis section
   - ✅ Distinguished discrete definition from interpolation formula
   - ✅ Added calculation details

3. **`src/model/running_coupling.py`**
   - ✅ Clarified RunningCouplingModule docstring
   - ✅ Distinguished discrete β from interpolation formula

4. **`tools/measure_beta_attention.py`**
   - ✅ Fixed compute_beta() to use discrete definition
   - ✅ Removed incorrect log(ratio) division

5. **`tools/quick_beta_validation.py`**
   - ✅ Fixed compute_beta() to use discrete definition
   - ✅ Added authoritative definition in docstring

6. **`tools/beta_full_statistical_protocol.py`**
   - ✅ Fixed beta calculation in sampling loop
   - ✅ Removed log ratio division

7. **`tools/compare_beta_physics_attention.py`**
   - ✅ Added clarification that attention β is different measure
   - ✅ Documented log-log slope vs discrete fractional change

---

## 📖 References

1. **FROZEN_FACTS.md** - Authoritative physics values and definitions
2. **qig-verification** - Sister project with lattice experiments
3. **ChatGPT conversation (2025-11-25)** - Clarification of β definition
4. **L=6 Full Validation** - Updated κ₆ = 62.02 ± 2.47 (3 seeds, VALIDATED)

---

## ✅ Validation Checklist

- [x] All β calculations use discrete definition (Δκ/κ_avg)
- [x] Interpolation formula documented as fitting tool
- [x] Constants frozen at experimentally validated values
- [x] Measurement tools corrected
- [x] Documentation clarified throughout codebase
- [x] Different β measures (physics vs attention) distinguished

---

## 🎯 Key Takeaways

1. **For QIG projects:** β = Δκ/κ_avg (discrete fractional change)
2. **Not RG theory:** β ≠ dκ/d(log L) (continuum derivative)
3. **Interpolation:** κ(L) = κ₀(1 + β·log(L/L_ref)) uses β as fitting parameter
4. **Use exact values:** KAPPA_3, KAPPA_4, KAPPA_5, KAPPA_6 for validation
5. **Copilot was right:** The discrete definition matches qig-verification

---

**Bottom line:** When someone asks "what is β?" in QIG-land, the answer is **"the fractional step in κ between L and L+1"** — not a log-derivative.
