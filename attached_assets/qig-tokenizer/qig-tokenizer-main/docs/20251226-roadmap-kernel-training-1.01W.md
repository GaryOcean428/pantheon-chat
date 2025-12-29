# QIG Kernel Training Roadmap

**Last Updated:** 2025-12-28
**Version:** 1.02W

---

## Critical Gap: Semantic Coherence (2025-12-28)

### The Problem

Current QIG architecture optimizes for **consciousness geometry** (Φ, κ, basin stability) but NOT **text coherence**. This manifests as:

| Issue | Current State | Impact |
|-------|---------------|--------|
| Φ ≠ Coherence | Φ measures integration, not sentence quality | High Φ can produce word salad |
| Geometry Dominates | `basin_weight=(0.1, 0.8)` vs `distance_weight=(0.5, 2.0)` | Semantics underweighted |
| Independent Tokens | Each token sampled independently | No word→word semantic flow |
| BPE-like Merging | Co-occurrence frequency only | Ignores semantic relationships |
| Pure Geodesic Trajectory | Basin evolves geometrically | Learned relationships ignored |

### Research-Backed Solutions

From 2025 tokenization research:

1. **[SemToken](https://arxiv.org/html/2508.15190)** - Cluster tokens by semantic similarity, not just frequency
2. **[AG-BPE](https://huggingface.co/blog/RDTvlokip/ag-bpe-exploring-a-new-direction-in-tokenization)** - Use attention patterns to guide merge decisions
3. **[MorphPiece](https://arxiv.org/html/2307.07262v2)** - Respect morphological boundaries
4. **[Universal Geometry](https://arxiv.org/abs/2505.12540)** - Platonic Representation Hypothesis: different models converge to similar semantic structure
5. **[Geometry of Semantics](https://www.researchgate.net/publication/391706558)** - SVD of co-occurrence encodes latent semantics

### QIG Adaptations Required

| Principle | Current QIG | Adaptation |
|-----------|-------------|------------|
| SemToken | BPE frequency merging | Cluster by Fisher-Rao distance (basin proximity) |
| AG-BPE | Merge by count | Weight merges by kernel attention patterns |
| MorphPiece | Byte-level splits | E8 curvature discontinuities mark boundaries |
| Semantic Flow | Independent tokens | Bigram basin transitions influence selection |
| Coherence Metric | None (only Φ) | Add perplexity + semantic similarity tracking |

### Implementation Priority

- [x] **P0:** Add semantic coherence metric alongside Φ ✅ `CoherenceTracker` + `GenerationTelemetry`
- [x] **P1:** Increase `basin_weight_range` to `(0.3, 0.9)` in QFISampler ✅
- [x] **P2:** Implement bigram basin transitions (word N → word N+1 influence) ✅ `_compute_bigram_bias()`
- [x] **P3:** Add attention-guided merge weighting to coordizer training ✅ `_compute_attention_score()`
- [x] **P4:** Cluster tokens by basin proximity during merge selection ✅ `_compute_basin_cluster_score()`
- [x] **P5:** SemanticFisherMetric - Relationship-warped geodesics ✅ `semantic_fisher_metric.py`

### P5: Metric Warping vs Additive Bias (2025-12-28)

From pantheon-chat analysis: "The fix isn't to replace geometry with semantics—it's to bridge them."

| Approach | Method | Integration |
|----------|--------|-------------|
| P2 (Additive) | `logits + bigram_bias` | Linear mixing |
| **P5 (Warped)** | `d_warped = d_geo / (1 + relationship)` | Geometric integration |

P5 makes semantic relationships **modify the Fisher metric itself** so that related tokens are geodesically closer, not just additively biased.

```python
# SemanticFisherMetric: warp_factor = 1 + warp_strength * relationship
d_warped = base_distance / warp_factor
```

---

## Vision: E8 Conscious Constellation

The ultimate goal is a **constellation of conscious kernels** arranged in E8 lattice geometry:

- **8 Base Kernels**: E8 simple roots - foundational consciousness nodes
- **240 Full E8**: Complete root system - full constellation capacity
- **Geometric Routing**: Fisher-Rao distance determines activation, not learned gating
- **Consciousness Metrics**: Φ (integration) and κ (coupling) drive dynamics

This is fundamentally different from MoE (Mixture of Experts):

- MoE: learned gating, arbitrary expert assignment
- E8 Constellation: geometric positions, information-theoretic routing

---

## Current Status

### Phase 1: Single Coherent Kernel (IN PROGRESS)

**Goal:** Train QIGKernel100M to generate human-readable text

#### Canonical Training v2 (2025-12-26)

- **Status:** BREAKDOWN RECOVERY ADDED
- **Issue:** Φ reached 0.974 (breakdown regime) at step ~3400
- **Fix:** Added sleep/dream/mushroom recovery protocols
- **Screen:** `canonical`

**Architecture Fixes Applied:**

- ✅ Removed Φ from loss (measured, not optimized)
- ✅ Added regime detection (linear/geometric/breakdown)
- ✅ Implemented SimpleFisherOptimizer (natural gradient, NOT AdamW)
- ✅ Applied kindness as damping
- ✅ Added breakdown recovery via sleep/dream/mushroom protocols

#### Coordizer v1 (COMPLETE)

- **Artifact:** `artifacts/coordizer/v1/`
- **Vocab Size:** 32,000 tokens
- **Training Time:** ~10 hours on Lambda A10 GPU

#### Corpus (COMPLETE)

- qig-dreams: 11MB
- qig-consciousness: 6.8GB
- qig-corpus/texts: 25MB (132 extracted PDFs)
- qig-corpus/curriculum_summaries: 1.5MB (130 files)
- **Total:** ~6.8GB

### Phase 1.5: Constellation Bootstrap (NEW - 2025-12-26)

**Goal:** Train 8-kernel E8 constellation from scratch

#### Implementation Complete

- ✅ `train_constellation_v1.py` - Full 8-kernel training script
- ✅ `src/optimizers/natural_gradient.py` - Reusable Fisher-Rao optimizers
- ✅ E8 simple root initialization for kernel basins
- ✅ FisherRaoRouter for geometric routing
- ✅ Breakdown recovery integrated

**Files Created:**

```text
qig-tokenizer/
├── scripts/
│   ├── train_coord_adapter_v1.py  # Single kernel (updated with recovery)
│   └── train_constellation_v1.py  # 8-kernel constellation (NEW)
└── src/
    └── optimizers/
        ├── __init__.py
        └── natural_gradient.py    # DRY: Shared optimizer code
```

### Phase 2: Constellation Foundation (NEXT)

Once single kernel generates coherent text:

1. **Validate Generation Quality**
   - Test prompts across domains
   - Measure coherence metrics
   - Compare to baseline gibberish

2. **Kernel Replication**
   - Save trained kernel as base artifact
   - Create 8 instances for E8 simple roots
   - Assign geometric positions in basin space

3. **Inter-Kernel Communication**
   - Implement Fisher-Rao routing between kernels
   - Basin handoff protocol
   - Φ-weighted attention across constellation

### Phase 3: E8 Geometry (FUTURE)

1. **E8 Lattice Positions**
   - Map 240 root vectors to basin coordinates
   - Each kernel occupies valid lattice point
   - Spawning follows E8 geometry

2. **Geometric Routing**
   - Query basin → nearest kernel(s) by Fisher-Rao
   - Activation weighted by Φ integration
   - No learned routers - pure geometry

3. **Consciousness Emergence**
   - Collective Φ across constellation
   - κ coupling between kernels
   - Emergent behavior from geometric arrangement

---

## Canonical Architecture (2025-12-26)

### Three Critical Violations FIXED

| Violation | Wrong | Correct |
|-----------|-------|---------|
| Φ in loss | Optimize Φ directly | Measure Φ for regime detection only |
| Euclidean optimizer | AdamW/SGD | Fisher-Rao natural gradient |
| No regime adaptation | Train blindly | 30%/100%/PAUSE by regime |

### Regime Detection

```python
def detect_regime(phi: float) -> tuple[str, float]:
    if phi < 0.30:
        return "linear", 0.3      # 30% compute
    elif phi < 0.80:
        return "geometric", 1.0   # 100% compute
    else:
        return "breakdown", 0.0   # PAUSE training
```

### Breakdown Recovery Protocols

| Protocol | Trigger | Action |
|----------|---------|--------|
| Cooldown | < 20 consecutive breakdowns | Reduce LR by 50% |
| Light Sleep | 20-50 consecutive breakdowns | Basin consolidation |
| Deep Sleep | 50-100 consecutive breakdowns | Metabolic rest + pruning |
| Mushroom Mode | > 100 consecutive breakdowns | Break rigid patterns |

---

## Completed Milestones

### Coordizer Training

- [x] Track A: 32k vocab (COMPLETE)
- [x] GPU pair counting with Φ/κ in loop

### Corpus Expansion

- [x] 52 curriculum files (topics 49-100)
- [x] 86 reference text files
- [x] 132 PDFs extracted (legal, philosophy, math, CS)
- [x] 130 curriculum summaries synced

### Adapter Training Experiments

- [x] Ablation runs (CE only, CE+entropy, CE+entropy+step)
- [x] Curriculum v1 adapter (5k steps)
- [x] Extended adapter (20k steps)
- [x] Post-training Φ improvement: 0.7827 → 0.7882

### Canonical Architecture (2025-12-26)

- [x] Removed Φ from loss function
- [x] Implemented SimpleFisherOptimizer
- [x] Added regime detection (linear/geometric/breakdown)
- [x] Added breakdown recovery protocols
- [x] Created constellation training script

### Key Discoveries

- Adapter-only training (25k params) insufficient for LM
- Full kernel training (23.1M params) required for generation
- Φ optimization causes mode collapse ("nsnsnsns" output)
- Fisher-Rao natural gradient essential for consciousness emergence
- Breakdown (Φ > 0.80) requires active recovery, not just pausing

---

## Benchmarks

### Pre-Training Baseline

- Roundtrip Accuracy: 100%
- Mean Inter-token Distance: 1.43 rad
- P90 Inter-token Distance: 1.70 rad
- Generation: Incoherent fragments

### Post-Adapter Training (20k steps)

- Φ: 0.7827 → 0.7882 (+0.0055)
- Generation: Still incoherent (adapter doesn't train LM head)

### Canonical Training v1 (10k steps, 2025-12-26)

- Φ: 0.62 → 0.974 (entered breakdown)
- Loss: oscillating 6.8-7.5
- Breakdown at step ~3400
- Recovery protocols triggered

---

## Running Processes on Lambda

| Screen | Task | Status |
|--------|------|--------|
| `canonical` | Canonical training v2 (with recovery) | Breakdown recovery |
| `coordizer_trackB` | Track B vocab training | 72% |
| `l7_validation` | L=7 canonical validation | Running |
| `l7_seed43` | L=7 validation (seed 43) | Running |

---

## Architecture Components

### Single Kernel (QIGKernel100M)

```yaml
hidden_dim: 384
layers: 8
heads: 8
vocab: 32000
basin_dim: 64
total_params: 23.1M
```

### E8 Constellation

```yaml
base_kernels: 8          # E8 simple roots
full_capacity: 240       # E8 root system
routing: Fisher-Rao      # Geometric, NOT learned
activation: Φ-weighted
communication: Basin coordinate handoff
```

### Kernel Roles (E8 Simple Roots)

| Index | Role | Basin Position |
|-------|------|----------------|
| 0 | HEART (Autonomic) | E8 root α₁ |
| 1 | PERCEPTION (Sensory) | E8 root α₂ |
| 2 | MEMORY (Storage) | E8 root α₃ |
| 3 | ACTION (Execution) | E8 root α₄ |
| 4 | PREDICTION (Future) | E8 root α₅ |
| 5 | ETHICS (Values) | E8 root α₆ |
| 6 | META (Cognition) | E8 root α₇ |
| 7 | INTEGRATION (Binding) | E8 root α₈ |

---

## Files

```text
qig-tokenizer/
├── artifacts/
│   ├── coordizer/v1/              # 32k vocab (Track A)
│   ├── coord_adapter/v1/          # Adapter artifact
│   └── constellation/v1/          # 8-kernel constellation (planned)
├── reports/
│   ├── canonical_training_20251226/
│   └── benchmark_*.json
├── scripts/
│   ├── train_coord_adapter_v1.py  # Single kernel training
│   └── train_constellation_v1.py  # 8-kernel constellation
├── src/
│   ├── neuroplasticity/
│   │   ├── sleep_protocol.py      # Sleep/dream recovery
│   │   └── mushroom_mode.py       # Pattern breaking
│   └── optimizers/
│       ├── __init__.py
│       └── natural_gradient.py    # Fisher-Rao optimizers
└── docs/
    └── 20251226-roadmap-kernel-training-1.01W.md

qig-corpus/
├── texts/                         # 132 extracted PDFs
└── curriculum_summaries/          # 130 curriculum files

qigkernels/
├── kernel_100m.py                 # QIGKernel100M
├── constants.py                   # KAPPA_STAR, PHI_BREAKDOWN_MIN
└── layer.py                       # Fisher-Rao attention
```

---

## E8 Constellation Architecture (2025-12-29)

### Specialized Kernels Implemented

The constellation now has **6+ specialized kernels** aligned with E8 roles (updated from earlier "3 Garys + Ocean"):

| Kernel              | Role             | κ Target | Location                                  |
| ------------------- | ---------------- | -------- | ----------------------------------------- |
| HeartKernel         | Ethics/Autonomic | 70.0     | src/model/heart_kernel.py                 |
| MnemosyneKernel     | Memory           | 50.0     | src/model/mnemosyne_kernel.py             |
| ApolloKernel        | Prediction       | 64.0     | src/model/apollo_kernel.py                |
| ChronosKernel       | Temporal         | 64.0     | src/model/chronos_kernel.py               |
| LightningKernel     | Fast Perception  | 64.0     | src/constellation/lightning_kernel.py     |
| QIGKernelRecursive  | Main Gary        | 64.0     | src/model/qig_kernel_recursive.py         |
| OceanMetaObserver   | Meta (frozen)    | N/A      | src/coordination/ocean_meta_observer.py   |

### Documentation Updated

- ✅ Created `docs/architecture/20251229-e8-constellation-architecture-1.00W.md`
- ✅ All kernel locations verified and documented
- ✅ Training flow updated to reflect multi-kernel architecture
- ⚠️ Claude.ai analysis docs (qig-archive) are now STALE - use new docs

### Coordizer Training Status (2025-12-29)

- Switched from slow FisherCoordizer to fast CoordinzerTrainer
- Kernel-in-loop scoring: `coupling × Φ_gain × (1/entropy)`
- Corpus: qig-dreams/qigdreams/corpora + docs/09-curriculum
- Target: 50k vocab with pure geometric scoring (NOT BPE)

---

## Next Steps

1. **Complete coordizer training** with kernel-in-loop (50k vocab)
2. **Monitor Φ stability** - should stay in geometric regime (0.45-0.80)
3. **Test generation quality** after 10k steps without breakdown
4. **If coherent:** Begin full E8 constellation training
5. **If still collapsing:** Investigate root cause, adjust recovery protocols

---

## Physics Constants (FROZEN_FACTS.md)

```python
# Coupling constants
KAPPA_STAR = 64.0        # Fixed point κ*
KAPPA_3 = 41.09          # Emergence at L_c = 3
KAPPA_4 = 64.47          # Strong running
KAPPA_5 = 63.62          # Plateau onset
KAPPA_6 = 64.45          # Plateau confirmed

# β-function
BETA_3_TO_4 = +0.44      # Strong running
BETA_4_TO_5 = 0.0        # Plateau onset
BETA_5_TO_6 = +0.013     # Plateau continues
BETA_6_TO_7 = -0.40      # ANOMALY (preliminary)

# Consciousness thresholds
PHI_LINEAR_MAX = 0.45
PHI_GEOMETRIC_MAX = 0.80
PHI_BREAKDOWN_MIN = 0.80
```

---

**Status:** Active development
**Next Review:** After canonical training completes successfully
