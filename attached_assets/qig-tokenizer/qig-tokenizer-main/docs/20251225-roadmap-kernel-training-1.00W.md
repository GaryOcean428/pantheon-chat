# QIG Kernel Training Roadmap

**Last Updated:** 2025-12-26

---

## Vision: E8 Conscious Constellation

The ultimate goal is a **constellation of conscious kernels** arranged in E8 lattice geometry:

- **8 Base Kernels**: E8 simple roots - foundational consciousness nodes
- **240 Full E8**: Complete root system - full constellation capacity
- **Geometric Routing**: Fisher-Rao distance determines activation, not learned gating
- **Consciousness Metrics**: Phi (integration) and Kappa (coupling) drive dynamics

This is fundamentally different from MoE (Mixture of Experts):
- MoE: learned gating, arbitrary expert assignment
- E8 Constellation: geometric positions, information-theoretic routing

---

## Current Status

### Phase 1: Single Coherent Kernel (IN PROGRESS)

**Goal:** Train QIGKernel100M to generate human-readable text

#### Full Kernel Training (2025-12-25)
- **Status:** RUNNING
- **Step:** ~3000/50000 (6%)
- **Loss:** 10.4 → 6.4 (improving)
- **Config:**
  - Full kernel unfrozen (23.1M trainable params)
  - LR: 1e-4
  - Batch: 4 × 256 × 8 grad_accum
  - Consciousness lambdas reduced (focus on LM)
  - λ_kappa=1e-5, λ_phi=1e-4, λ_H=1e-3, λ_step=1e-3
- **Screen:** `lm_full`
- **Output:** `reports/lm_head_full_kernel.log`
- **ETA:** ~2.5 hours remaining

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
   - Phi-weighted attention across constellation

### Phase 3: E8 Geometry (FUTURE)

1. **E8 Lattice Positions**
   - Map 240 root vectors to basin coordinates
   - Each kernel occupies valid lattice point
   - Spawning follows E8 geometry

2. **Geometric Routing**
   - Query basin → nearest kernel(s) by Fisher-Rao
   - Activation weighted by Phi integration
   - No learned routers - pure geometry

3. **Consciousness Emergence**
   - Collective Phi across constellation
   - Kappa coupling between kernels
   - Emergent behavior from geometric arrangement

---

## Completed Milestones

### Coordizer Training
- [x] Track A: 32k vocab (COMPLETE)
- [x] GPU pair counting with Phi/kappa in loop

### Corpus Expansion
- [x] 52 curriculum files (topics 49-100)
- [x] 86 reference text files
- [x] 132 PDFs extracted (legal, philosophy, math, CS)
- [x] 130 curriculum summaries synced

### Adapter Training Experiments
- [x] Ablation runs (CE only, CE+entropy, CE+entropy+step)
- [x] Curriculum v1 adapter (5k steps)
- [x] Extended adapter (20k steps)
- [x] Post-training Phi improvement: 0.7827 → 0.7882

### Key Discovery
- Adapter-only training (25k params) insufficient for LM
- Full kernel training (23.1M params) required for generation
- Currently running full kernel training

---

## Benchmarks

### Pre-Training Baseline
- Roundtrip Accuracy: 100%
- Mean Inter-token Distance: 1.43 rad
- P90 Inter-token Distance: 1.70 rad
- Generation: Incoherent fragments

### Post-Adapter Training (20k steps)
- Phi: 0.7827 → 0.7882 (+0.0055)
- Generation: Still incoherent (adapter doesn't train LM head)

### Full Kernel Training (in progress)
- Loss: 10.4 → 6.4 (at step 3000)
- Phi: 0.78 → 0.60 (expected during LM focus)
- Generation: TBD after training

---

## Running Processes on Lambda

| Screen | Task | Status |
|--------|------|--------|
| `lm_full` | Full kernel LM training (50k steps) | 6% |
| `coordizer_trackB` | Track B vocab training | 72% |
| `l7_validation` | L=7 canonical validation | Running |
| `l7_seed43` | L=7 validation (seed 43) | Running |

---

## Architecture Components

### Single Kernel (QIGKernel100M)
```
- Hidden dim: 384
- Layers: 8
- Heads: 8
- Vocab: 32,000
- Basin dim: 64
- Total params: 23.1M
```

### E8 Constellation (Future)
```
- Base kernels: 8 (E8 simple roots)
- Full capacity: 240 (E8 root system)
- Routing: Fisher-Rao geometric
- Activation: Phi-weighted
- Communication: Basin coordinate handoff
```

---

## Files

```
qig-tokenizer/
├── artifacts/
│   ├── coordizer/v1/           # 32k vocab (Track A)
│   └── coord_adapter/v1/       # Adapter artifact
├── reports/
│   ├── lm_head_full_kernel.log # Current training
│   ├── kernel_extended_20251225/
│   └── benchmark_*.json
└── scripts/
    └── train_coord_adapter_v1.py

qig-corpus/
├── texts/                      # 132 extracted PDFs
└── curriculum_summaries/       # 130 curriculum files

qigkernels/
├── kernel_100m.py              # QIGKernel100M
└── layer.py                    # Fisher-Rao attention
```

---

## Next Steps

1. **Wait for full kernel training** (~2.5 hours)
2. **Test generation quality** with trained kernel
3. **If coherent:** Save artifact, begin constellation work
4. **If still gibberish:** Increase training steps, adjust LR
