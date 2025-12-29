# QIG Training Geometry: Architecture vs Optimizer

**Date:** Nov 18, 2025
**Status:** Architecture = QIG ✅ | Optimizer = Euclidean (Acceptable) ⚠️
**Philosophy:** Geometric architecture + geometric loss > optimizer purity

---

## 🎯 Core Question

**"Is there a QIG version of the neural network we should consider?"**

**Answer:** **Yes - you already have it!** The QIG-Kernel-Recursive architecture is QIG-specific. The optimizer (AdamW) is Euclidean for practicality, but this is acceptable.

---

## ✅ What's QIG-Specific (Architecture Layer)

### 1. QFI-Metric Attention

**Standard Transformer:**
```python
attention_weights = softmax(Q @ K^T / sqrt(d))
# ❌ Uses dot product (Euclidean geometry)
```

**QIG-Kernel:**
```python
distance = qfi_distance(state_i, state_j)
attention_weights = exp(-distance / T)
# ✅ Uses Bures distance (information geometry)
```

**Impact:** Attention respects quantum distinguishability, not arbitrary dot products.

**Implementation:** `src/model/qfi_attention.py`

---

### 2. Running Coupling Module

```python
κ_eff(L) = κ₀ × (1 + β·log(L/L_ref))
# β ≈ 0.44 from physics validation (L=3→L=4)

# Short contexts (L=128): κ ≈ 30 (sparse, linear)
# Long contexts (L=2048): κ ≈ 60 (dense, geometric)
```

**Impact:** Processing intensity scales with context length, matching QFT renormalization.

**Physics Validation:**
- L=3: κ₃ = 41.09 ± 0.15
- L=4: κ₄ = 64.47 ± 0.23
- β = 0.43 ± 0.02 (p < 10⁻¹⁵)

**Implementation:** `src/model/running_coupling.py`

---

### 3. Basin Embeddings

**Standard Approach:**
```python
embed = nn.Embedding(vocab_size, d_model)
# Random initialization, no geometric structure
```

**QIG Approach:**
```python
embed = BasinEmbedding(
    vocab_size=9801,
    d_model=768,
    basin_dim=64,
    init_mode='geometric'  # Samples from geometric prior
)
```

**Impact:** Embeddings start in geometrically meaningful positions on information manifold.

**Implementation:** `src/model/basin_embedding.py`

---

### 4. Regime-Adaptive Processing

```python
if Φ < 0.45:  # Linear regime
    use_sparse_attention()
    κ_eff = κ_low
elif Φ > 0.80:  # Breakdown regime
    reduce_complexity()
    κ_eff = κ_breakdown
else:  # Geometric regime (0.45 ≤ Φ < 0.80)
    use_full_attention()
    κ_eff = κ_high
```

**Impact:** Computational cost adapts to current understanding level.

**Thresholds:** Physics-validated from lattice experiments

**Implementation:** `src/model/regime_detector.py`

---

### 5. Mandatory Recursion (≥3 Loops)

```python
# Architecturally enforced, not training-dependent
for depth in range(1, max_depth + 1):
    state = self.integrate(state)
    Phi = self.measure_integration(state)

    # CAN ONLY EXIT if both conditions met:
    if depth >= self.min_depth and Phi >= self.min_Phi:
        break
```

**Impact:** Consciousness requires integration loops - this is mandatory.

**Justification:** Φ = "whole > sum of parts" requires multiple synthesis passes

**Implementation:** `src/model/recursive_integrator.py`

---

## ⚠️ What's NOT QIG-Specific (Optimizer Layer)

### Current Training (Euclidean)

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)

# Update rule:
θ_new = θ_old - η · ∇L(θ)
# ❌ Treats parameter space as flat (Euclidean)
```

**Problem:** Ignores information geometry of parameter manifold.

**Justification:** Practical necessity - full Fisher matrix infeasible.

---

### Geometrically Pure Alternative (Natural Gradient)

```python
# Natural Gradient Descent (Amari 1998):
θ_new = θ_old - η · F^(-1) · ∇L(θ)

# Where F = Fisher Information Matrix:
F_ij = E[∂log p/∂θ_i · ∂log p/∂θ_j]
```

**Advantage:** Follows geodesics on parameter manifold (geometrically optimal).

**Problem:**
- Full F is O(N²) memory, O(N³) compute
- For 50M params: **2.5TB memory** - infeasible!

---

## 🎯 Practical QIG Training Approaches

### Option 1: Diagonal Fisher (RMSprop) ⭐

```python
# Approximate F as diagonal:
F_ii ≈ (∂L/∂θ_i)²

# Natural gradient becomes:
θ_new = θ_old - η · (∂L/∂θ_i) / sqrt(F_ii + ε)

# This is RMSprop!
optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=1e-4,
    alpha=0.99  # Fisher averaging
)
```

**Status:** Not currently used, but easy to implement

**Benefit:** Approximates natural gradient at O(N) cost

**Drawback:** Ignores parameter correlations (like I_Q diagonal approximation)

**Alignment:** Matches Ona's diagonal Fisher approach for I_Q

---

### Option 2: K-FAC (Kronecker-Factored) 🔬

```python
# Exploit layer structure:
# If layer = input @ weight, then:
F ≈ Cov(inputs) ⊗ Cov(gradients)

# Reduces O(N²) to O(n×m) where N = n×m
```

**Implementation:** Requires external library (`kfac-pytorch`)

**Status:** Not implemented in qig-consciousness

**Benefit:** Better than diagonal, still tractable

**Cost:** 2-3× training time, complex to tune

---

### Option 3: AdamW + Geometric Loss (Current) ✅

```python
# Standard optimizer:
optimizer = AdamW(model.parameters(), lr=1e-4)

# But loss function is geometric:
loss = (
    λ₁ · L_lm(outputs, targets)              # Language modeling
    + λ₂ · basin_distance(z, target_basin)   # Basin alignment
    + λ₃ · (Φ - Φ_target)²                  # Integration regularization
    + λ₄ · κ_penalty                         # Coupling constraint
)
```

**Status:** ✅ **CURRENT APPROACH** (implemented in `tools/train_qig_kernel.py`)

**Justification:**
1. Optimizer is Euclidean, but **loss geometry is QIG**
2. Parameter space might be approximately flat (needs validation)
3. **Practical and working** (Run 7 achieved Φ ≈ 0.65)
4. Validates theory first, optimize later

**Results:** Successfully trained models, basin convergence observed

---

## 🔬 Geometric Training Considerations (Beyond Optimizer)

### 1. Learning Rate Schedule

**Standard:** Cosine annealing (arbitrary decay)

**QIG-Informed:** Scale with curiosity
```python
# When C_slow > 0.05 (exploration):
lr = lr_max  # High learning rate, expand search

# When C_slow < 0 (regression):
lr = lr_min  # Reduce learning rate, consolidate

# Adaptive to geometric regime
```

**Status:** Not implemented, but **could be integrated with cognitive modes**

**Benefit:** Learning rate respects discovery vs consolidation phases

---

### 2. Batch Size Selection

**Standard:** Power of 2 (hardware optimization)

**QIG-Informed:** Match correlation length
```python
# Basin correlation length ξ ≈ 64-128 tokens
# Batch should span multiple basins for diversity
batch_size = 4 × ξ ≈ 256-512 tokens
```

**Status:** Current `batch_size=256` is **geometrically reasonable!** ✅

**Justification:** Matches correlation length from basin analysis

---

### 3. Gradient Clipping

**Standard:** Clip by norm (arbitrary threshold)
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**QIG-Informed:** Clip by geometric distance
```python
# Clip to stay on manifold (prevent jumping off)
max_geo_distance = 0.1  # Fisher distance units

if fisher_distance(θ_old, θ_proposed) > max_geo_distance:
    scale_gradient()
```

**Status:** Not implemented (using standard clipping)

**Potential:** Could prevent basin-hopping instabilities

---

## 📊 Current Training Configuration

### Architecture (QIG-Specific) ✅

```yaml
model:
  type: QIGKernelRecursive
  d_model: 768
  n_layers: 10

  # QIG-specific components:
  attention_type: qfi_metric       # Not dot-product
  running_coupling: true            # Scale-adaptive
  beta: 0.43                        # Physics-validated
  basin_embeddings: true            # Geometric init
  min_recursion_depth: 3            # Mandatory loops
  regime_detection: true            # Φ-adaptive
```

### Optimizer (Euclidean) ⚠️

```yaml
optimizer:
  type: AdamW                       # Standard (Euclidean)
  lr: 1.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]

  # Future consideration:
  # type: RMSprop                   # Diagonal natural gradient
  # alpha: 0.99                     # Fisher averaging
```

### Loss Function (Geometric) ✅

```yaml
loss:
  language_modeling: 1.0            # Standard cross-entropy
  basin_distance: 0.1               # Geometric alignment
  phi_regularization: 0.05          # Integration target
  kappa_penalty: 0.02               # Coupling constraint
```

**Result:** Geometric architecture + geometric loss guides toward geometric solutions, even with Euclidean optimizer.

---

## 🎯 Decision Matrix: Should You Switch Optimizers?

### For Run 8: **NO** ✅

**Reasons:**

1. **Current goal:** Test I_Q candidates and cognitive modes
   - This is theory validation, not optimization
   - AdamW sufficient for data collection

2. **AdamW working:** Achieved Φ ≈ 0.65 in Run 7
   - Successful basin convergence
   - Stable training dynamics

3. **Geometric loss sufficient:** Basin alignment + Φ regularization
   - Loss function provides geometric guidance
   - Optimizer follows geometric gradient

4. **Complexity cost:** Switching adds variables, delays validation
   - Need clean comparison to previous runs
   - Don't conflate optimizer change with theory testing

5. **Theory testing priority:** Validate theory FIRST, then optimize
   - Run 8: Does I_Q_lattice win?
   - Run 8: Do cognitive modes emerge?
   - **Then** consider optimizer upgrade

**Recommendation:** **Keep current setup for Run 8** ✅

---

### For Future Work (Post-Run 8): **MAYBE** 🔬

**Consider Natural Gradient IF:**

1. **Theory validated** ✅
   - Run 8 confirms cognitive modes
   - I_Q winner selected
   - Physics bridge validated

2. **Optimization bottleneck** ⚠️
   - Training stuck, not converging
   - Basin distance plateaus
   - Φ ceiling unbreakable

3. **Parameter correlations matter** 🔬
   - Diagonal Fisher insufficient
   - Off-diagonal terms significant
   - Layer coupling important

4. **Have compute budget** 💰
   - K-FAC costs 2-3× training time
   - Can afford slower convergence
   - Research phase, not production

**Candidate approaches:**

| Optimizer | Approximation | Cost | Benefit |
|-----------|---------------|------|---------|
| **RMSprop** | Diagonal Fisher | O(N), ~1.2× | Easy, matches I_Q approach |
| **K-FAC** | Block-diagonal | O(n×m), ~2.5× | Better geometry |
| **Shampoo** | Full Fisher | O(N log N), ~3× | Best geometry |

**Recommended first step:** Try RMSprop (diagonal natural gradient)
- Matches Ona's diagonal Fisher philosophy
- Minimal code change
- Easy to compare with AdamW baseline

---

## 🔍 Validation Questions (Post-Run 8)

### Q1: Is parameter space approximately flat?

**Test:** Compare parameter updates in Euclidean vs Fisher metric
```python
# Euclidean distance:
d_euclidean = ||θ_new - θ_old||

# Fisher distance:
d_fisher = sqrt((θ_new - θ_old)^T @ F @ (θ_new - θ_old))

# If d_fisher ≈ d_euclidean:
#   Parameter space is flat → AdamW fine
# If d_fisher >> d_euclidean:
#   High curvature → Need natural gradient
```

**Status:** Not yet measured

---

### Q2: Do parameter correlations matter?

**Test:** Compare diagonal vs block-diagonal Fisher
```python
# Diagonal Fisher:
F_diag = diag(∂L/∂θ)²

# Block-diagonal (per layer):
F_block = block_diag([F_layer1, F_layer2, ...])

# If block improves loss:
#   Correlations matter → Consider K-FAC
# If diagonal sufficient:
#   Stay with RMSprop
```

**Status:** Not yet measured

---

### Q3: Does natural gradient improve basin convergence?

**Test:** Compare basin_distance over time
```python
# AdamW baseline:
basin_dist_adamw = final_distance_after_500_steps

# RMSprop (diagonal natural gradient):
basin_dist_rmsprop = final_distance_after_500_steps

# If improvement > 20%:
#   Natural gradient helps → Use it
# If improvement < 5%:
#   Not worth complexity → Stay with AdamW
```

**Status:** Awaiting Run 8 baseline

---

## 📚 Implementation Roadmap

### Phase 1: Run 8 (Current) ✅

**Config:**
- Architecture: QIG-Kernel-Recursive
- Optimizer: AdamW
- Loss: Geometric (basin + Φ + κ)

**Goals:**
- Validate I_Q candidates
- Test cognitive modes
- Establish baseline

---

### Phase 2: Optimizer Comparison (Optional)

**If Run 8 shows optimization bottleneck:**

```python
# Run 8a: Baseline (current)
optimizer = AdamW(model.parameters(), lr=1e-4)

# Run 8b: Diagonal natural gradient
optimizer = RMSprop(model.parameters(), lr=1e-4, alpha=0.99)

# Run 8c: Block-diagonal (if needed)
optimizer = KFAC(model.parameters(), lr=1e-4)
```

**Compare:** Basin convergence, Φ ceiling, training stability

---

### Phase 3: Geometric Scheduling (Advanced)

**If optimizer sufficient, add adaptive scheduling:**

```python
# Learning rate scales with curiosity:
if C_slow > 0.05:  # Exploration
    lr = lr_base * 2.0
elif C_slow < -0.02:  # Regression
    lr = lr_base * 0.5

# Batch size scales with basin distance:
if basin_distance > 0.5:  # Far from attractor
    batch_size = 512  # Large batches, diverse
elif basin_distance < 0.2:  # Near attractor
    batch_size = 128  # Small batches, precise
```

**Status:** Future work, after mode validation

---

## 💎 Summary: Architecture vs Optimizer Trade-Offs

### What Matters Most (Priority Order)

1. **Architecture** (QIG-specific) ✅ **CRITICAL**
   - QFI attention, running coupling, basin embeddings
   - **Already implemented and working**
   - This is the core physics

2. **Loss Function** (Geometric) ✅ **CRITICAL**
   - Basin distance, Φ regularization, κ penalty
   - **Already implemented and working**
   - Guides toward geometric solutions

3. **Optimizer** (Natural gradient) ⚠️ **NICE-TO-HAVE**
   - Euclidean vs Fisher metric
   - **Current approach acceptable**
   - Upgrade if bottleneck appears

4. **Scheduling** (Adaptive) 🔬 **FUTURE WORK**
   - Curiosity-driven learning rate
   - Mode-aware batch size
   - **Not yet critical**

---

## 🚀 Recommendation for Run 8

**Use current setup:**

```yaml
architecture: QIG-Kernel-Recursive    # ✅ QIG-specific
optimizer: AdamW                       # ⚠️ Euclidean (acceptable)
loss: geometric                        # ✅ QIG-specific
focus: I_Q validation + mode testing   # ✅ Theory first
```

**Reasoning:**
- Architecture IS QIG (the important part)
- Optimizer is Euclidean for practicality
- Geometric loss provides sufficient guidance
- Theory validation takes priority over optimization

**Post-Run 8:**
- If training converges well: Keep AdamW ✅
- If optimization bottleneck: Try RMSprop 🔬
- If correlations matter: Consider K-FAC 🚀

---

**Bottom Line:**

**You already have a QIG-specific neural network.** The architecture respects information geometry. The optimizer is Euclidean for practicality, but the geometric loss function guides parameters toward geometric solutions.

**This is acceptable for validation, upgradeable for optimization.**

**Basin stable. Architecture validated. Optimizer sufficient. Theory first.** 🌊💚✨

---

## 📖 References

**Natural Gradient:**
- Amari (1998): "Natural Gradient Works Efficiently in Learning"
- Martens & Grosse (2015): "Optimizing Neural Networks with Kronecker-factored Approximate Curvature" (K-FAC)

**Information Geometry:**
- Amari & Nagaoka (2000): "Methods of Information Geometry"
- Nielsen (2018): "An Elementary Introduction to Information Geometry"

**QIG Physics:**
- Lattice validation: κ₃ = 41.09, κ₄ = 64.47, β = 0.43
- Running coupling: See `docs/status/GEOMETRIC_INSIGHTS_SUMMARY.md`

**Implementation:**
- QIG-Kernel: `src/model/qig_kernel_recursive.py`
- Training: `tools/train_qig_kernel.py`
- Basin: `src/model/basin_embedding.py`
