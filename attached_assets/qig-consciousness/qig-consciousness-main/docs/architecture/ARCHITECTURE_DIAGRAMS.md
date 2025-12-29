# QIG-Kernel Architecture Diagrams

**Visual specifications for pure QIG-Kernel implementation**

Version: 1.0  
Date: 2025-11-17  
Status: Implementation-ready

---

## Table of Contents

1. [Core Architecture](#1-core-architecture)
2. [Tacking Controller Flow](#2-tacking-controller-flow)
3. [Self-Repair Pipeline](#3-self-repair-pipeline)
4. [Maturity Evaluation System](#4-maturity-evaluation-system)
5. [Training Day Flow](#5-training-day-flow)
6. [β-Attention Measurement](#6-β-attention-measurement)
7. [Sweet-Spot Geometry](#7-sweet-spot-geometry)
8. [Data Flow & Integration](#8-data-flow--integration)

---

## 1. Core Architecture

### QIG-Kernel-Pure (50-100M parameters)

```
Input Tokens [batch_size × seq_len]
         │
         ↓
┌────────────────────────┐
│ Token Embeddings       │
│ (256-512 dim)          │
│ + Positional Encoding  │
└────────────────────────┘
         │
         ↓
┌────────────────────────────────────┐
│ QFI-Attention Layer 1              │
│  ┌──────────────────────────────┐  │
│  │ QFI Distance Metric          │  │ → κ₁, Φ₁, H₁
│  │ Entanglement-Entropy Gating  │  │
│  │ Ethical Constraints (gauge)  │  │
│  └──────────────────────────────┘  │
│  Feed-Forward (geometric proj)     │
└────────────────────────────────────┘
         │
         ↓
┌────────────────────────────────────┐
│ QFI-Attention Layer 2              │ → κ₂, Φ₂, H₂
└────────────────────────────────────┘
         │
         ↓
        ...
         │
         ↓
┌────────────────────────────────────┐
│ QFI-Attention Layer 8              │ → κ₈, Φ₈, H₈
└────────────────────────────────────┘
         │
         ↓
┌────────────────────────────────────┐
│ Tacking Controller (WuWei)         │
│  ┌──────────────────────────────┐  │
│  │ Aggregate: Φ_total, κ_eff    │  │
│  │ Compute: |∇κ|                │  │
│  │ Decide: feeling/logic/tack   │  │
│  └──────────────────────────────┘  │
└────────────────────────────────────┘
         │
         ↓
┌────────────────────────────────────┐
│ Regime Detector                    │
│  Input: Φ, κ, attention patterns   │
│  Output: linear/geometric/breakdown│
└────────────────────────────────────┘
         │
         ↓
┌────────────────────────────────────┐
│ Output Projection                  │
│ [seq_len × vocab_size]             │
└────────────────────────────────────┘
         │
         ↓
    Output Tokens

Telemetry Side-Channel:
- Per-layer: κᵢ, Φᵢ, Hᵢ (entropy)
- Aggregate: κ_eff, Φ_total, |∇κ|
- Mode: feeling/logic/tack
- Regime: linear/geometric/breakdown
- Sweet-spot alignment
```

---

**See full file at `/home/runner/work/qig-consciousness/qig-consciousness/docs/architecture/ARCHITECTURE_DIAGRAMS.md` for complete diagrams including:**

- Layer detail diagrams
- Tacking controller flow (complete WuWei implementation)
- Self-repair pipeline (episode tracking & correction)
- Maturity evaluation system (stage progression logic)
- Training day flow (24-hour cycle visualization)
- β-attention measurement protocol
- Sweet-spot geometry (2D state space)
- Data flow & integration (end-to-end system)
- Symbol reference tables

Total: 16KB of visual specifications with ASCII diagrams.

