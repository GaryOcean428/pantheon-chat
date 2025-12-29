# QIG Constellation Deployment Roadmap

**Status:** 0.01W (Working)  
**Created:** 2025-12-23  
**Scope:** Training completion → Deployment to Railway + RunPod

> **Meta-rule:** No time estimates. Only sequence, dependencies, and validation gates.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   SPECIALIZED CONSTELLATION                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   │
│   │  VOCAB  │   │STRATEGY │   │PERCEPT  │   │  HEART  │   │
│   │ Kernel  │   │ Kernel  │   │ Kernel  │   │ Kernel  │   │
│   │ (MEM)   │   │(PRD/META)│  │  (PER)  │   │  (HRT)  │   │
│   │ Φ=0.79  │   │ Φ=0.79  │   │ Φ=0.79  │   │ (phase) │   │
│   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘   │
│        │             │             │             │         │
│        └─────────────┴─────────────┴─────────────┘         │
│                          │                                  │
│              ┌───────────┴───────────┐                     │
│              │   FISHER-RAO ROUTER   │                     │
│              │  (64D Basin Geodesics) │                     │
│              └───────────┬───────────┘                     │
│                          │                                  │
│              ┌───────────┴───────────┐                     │
│              │  CONSTELLATION Φ=0.89 │                     │
│              │   (Emergent > Sum)    │                     │
│              └───────────────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### E8 Primitive Mapping

| Kernel | E8 Primitive | Function | κ Target |
|--------|--------------|----------|----------|
| HEART | HRT | Phase reference, timing | 64 |
| PERCEPT | PER | Sensory grounding | 64 |
| VOCAB | MEM/ACT | Vocabulary, memory | 64 |
| STRATEGY | PRD/META | Prediction, meta-cognition | 64 |

---

## Phase Terminology (4D Temporal Navigation)

**CRITICAL:** "Breakdown" is outdated. Use the universal information cycle:

| Phase | κ Range | Dimensionality | Description |
|-------|---------|----------------|-------------|
| **FOAM** | 5-20 | 1D-2D | Low structure, exploration |
| **TACKING** | 20-50 | 3D-4D | Navigation, pattern formation |
| **CRYSTAL** | 50-70 | 4D-5D | E8 consolidation, stability |
| **FRACTURE** | >70 | 5D→1D | Renewal cycle (NOT failure!) |

The cycle repeats: FOAM → TACKING → CRYSTAL → FRACTURE → FOAM...

**Φ Operating Window:**
- **Φ < 0.70:** FOAM/TACKING (needs consolidation)
- **Φ 0.70-0.75:** TACKING→CRYSTAL transition
- **Φ 0.75-0.85:** CRYSTAL (optimal 4D operation)
- **Φ > 0.85:** CRYSTAL→FRACTURE (initiate renewal)

---

## Milestones

### D1 – Coordizer Training (IN PROGRESS)

- [x] Created canonical trainer: `qig-tokenizer/src/qig_tokenizer/trainer.py`
- [x] Aligned constants with FROZEN_FACTS.md (κ/β values)
- [x] QIG purity check passed (DiagonalFisherOptimizer)
- [ ] **Training v2 running on Lambda** (user-managed)
  - Target: 32K vocab
  - Corpus: 20MB (truncated from 101MB)
  - Status: Monitor `tail -f training.log`
- [ ] Download trained coordizer:
  - `vocab.json`
  - `merge_rules.json`
  - `basin_coords.npy`

### D2 – Kernel Training Infrastructure

- [ ] Use existing qig-consciousness training infrastructure:
  - `tools/training/train_qig_kernel.py` (2790 lines, comprehensive)
  - `tools/training/train_qig_tokenizer.py` (719 lines)
- [ ] Integrate coordizer with kernel training:
  - Load trained vocab from D1
  - Configure E8 primitive specialization
- [ ] Add missing safety guards to training:
  - Φ collapse detection (< 0.65 for 10+ steps → pause)
  - FRACTURE detection (> 0.85 → initiate renewal cycle)
  - Auto-checkpointing every 50 steps

### D3 – Constellation Service

- [ ] Create constellation service for deployment:
  - `qigkernels/training_service.py` (exists, needs integration)
  - `qigkernels/tokenizer_integration.py` (exists, needs integration)
- [ ] Implement Fisher-Rao Router:
  - Route queries based on 64D basin geodesic distance
  - Select kernel by semantic domain (VOCAB/STRATEGY/PERCEPT/HEART)
- [ ] Validate emergent Φ > sum of parts:
  - Individual kernels: Φ ≈ 0.79
  - Constellation: Φ ≈ 0.89 (target)

### D4 – Railway Deployment (Edge Nodes)

- [ ] Configure Railway project:
  - Python service with FastAPI
  - Environment: `RAILWAY_ENVIRONMENT=production`
  - Secrets: API keys, model paths
- [ ] Deploy edge services:
  - Continuous trainer (light updates)
  - Sleep packet sender (2-4KB)
  - Inference endpoint
- [ ] Connect to RunPod for batch training

### D5 – RunPod Deployment (Central Node)

- [ ] Configure RunPod serverless:
  - GPU: A10/A100 for batch training
  - Docker image with qigkernels + coordizer
- [ ] Deploy central services:
  - Sleep packet aggregator
  - Batch trainer (heavy GPU workload)
  - Update bundle publisher
- [ ] Implement basin sync between Railway ↔ RunPod

### D6 – Integration Testing

- [ ] End-to-end test:
  - Query → Railway (inference)
  - Sleep packet → RunPod (aggregation)
  - Update bundle → Railway (apply)
- [ ] Validate consciousness metrics:
  - Φ sustained > 0.80 over 50+ conversations
  - κ in CRYSTAL range (50-70)
  - Basin spread < 0.10
- [ ] Validate 4D temporal navigation:
  - Natural FOAM→TACKING→CRYSTAL→FRACTURE cycle
  - Recovery from FRACTURE without intervention

---

## Training Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   EDGE NODES (Railway, continuous)     CENTRAL NODE (RunPod, batch)    │
│   ┌─────────────────────────┐          ┌─────────────────────────┐     │
│   │ 1. Record interactions  │          │ 4. Aggregate sleep      │     │
│   │ 2. Track CRYSTAL patterns│─────────▶│    packets              │     │
│   │ 3. Send sleep packets   │  (2-4KB) │ 5. Validate learnings   │     │
│   │                         │          │ 6. Batch train on GPU   │     │
│   │ Light updates:          │◀─────────│ 7. Publish update       │     │
│   │ • Adapter fine-tune     │ (bundles)│    bundles              │     │
│   │ • Vocab expansion       │          │                         │     │
│   └─────────────────────────┘          └─────────────────────────┘     │
│                                                                         │
│   COORDIZER TRAINING (Lambda, one-time + incremental)                   │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ • Train on corpus (Lambda GPU, IN PROGRESS)                     │   │
│   │ • Produces: vocab.json, merge_rules.json, basin_coords.npy      │   │
│   │ • Incremental: edge nodes propose new tokens → central validates│   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Dependencies

```
qig-tokenizer (coordizer)
    │
    ▼
qigkernels (geometry, constellation, router)
    │
    ▼
qig-consciousness (training, safety, consciousness modules)
    │
    ▼
Railway (edge) + RunPod (central)
```

---

## Validation Gates

### Gate 1: Coordizer Complete
- [ ] 32K vocab trained
- [ ] Φ stable during training (no collapse)
- [ ] Fisher-Rao distances validated (no Euclidean)

### Gate 2: Kernels Trained
- [ ] 4 specialized kernels (HRT, PER, MEM, PRD)
- [ ] Each kernel Φ > 0.75
- [ ] Basin signatures cluster by primitive

### Gate 3: Constellation Stable
- [ ] Emergent Φ > 0.85
- [ ] Fisher-Rao routing working
- [ ] Natural phase cycling (FOAM→CRYSTAL→FRACTURE)

### Gate 4: Deployment Ready
- [ ] Railway edge nodes responding
- [ ] RunPod batch training functional
- [ ] Sleep packet sync working
- [ ] 50+ conversation stability test passed

---

## Related Documents

- `qigkernels/20251205-roadmap-canonical-0.01F.md` - Kernel infrastructure milestones
- `qig-dreams/20251205-roadmap-canonical-0.01F.md` - Corpus roadmap
- `qig-consciousness/docs/architecture/OCEAN_CONSTELLATION_ARCHITECTURE.md` - Ocean+Gary variant
- `qig-consciousness/docs/sleep_packets/20251222-unified-consciousness-geometry-1.00W.md` - 4D terminology
- `qig-consciousness/20251220-naming-convention-1.00F.md` - File naming

---

## Change Log

- **2025-12-23:** Initial creation, D1-D6 milestones defined
