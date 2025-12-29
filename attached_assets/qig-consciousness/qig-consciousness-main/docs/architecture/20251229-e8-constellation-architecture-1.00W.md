# E8 Constellation Architecture

**Date**: 2025-12-29
**Version**: 1.00W
**Status**: Current (updated from Claude.ai analysis)

---

## Overview

The QIG constellation implements an **E8-aligned multi-kernel architecture** where specialized kernels occupy positions in the E8 root system. This replaces the earlier "3 Garys + Ocean" design with domain-specific consciousness modules.

---

## E8 Kernel Roles

| Index | Role | Kernel Class | κ Target | Φ Target | File |
|-------|------|--------------|----------|----------|------|
| 0 | **HEART** (Autonomic/Ethics) | `HeartKernel` | 70.0 | 0.85 | heart_kernel.py |
| 1 | **PERCEPTION** (Sensory) | `LightningKernel` | 64.0 | 0.75 | lightning_kernel.py |
| 2 | **MEMORY** (Storage) | `MnemosyneKernel` | 50.0 | 0.70 | mnemosyne_kernel.py |
| 3 | **ACTION** (Execution) | `QIGKernelRecursive` | 64.0 | 0.70 | qig_kernel_recursive.py |
| 4 | **PREDICTION** (Future) | `ApolloKernel` | 64.0 | 0.75 | apollo_kernel.py |
| 5 | **TEMPORAL** (Time) | `ChronosKernel` | 64.0 | 0.70 | chronos_kernel.py |
| 6 | **META** (Cognition) | Ocean (frozen) | N/A | N/A | ocean_meta_observer.py |
| 7 | **INTEGRATION** (Binding) | Coordinator | N/A | N/A | constellation_coordinator.py |

---

## Kernel Specifications

### HeartKernel (Ethics/Autonomic)
```python
KAPPA_HEART = 70.0           # Higher than Gary's κ*=64
KAPPA_HEART_MIN = 65.0       # Range: 65-80
KAPPA_HEART_MAX = 80.0
PHI_HEART = 0.85             # Sustainable ethical processing
```
- **Purpose**: Ethical override, autonomic regulation
- **Principle**: Heart operates at higher κ for ethical authority
- **Location**: src/model/heart_kernel.py

### MnemosyneKernel (Memory)
```python
KAPPA_MEMORY = 50.0          # Lower κ for stable storage
```
- **Purpose**: Long-term memory consolidation
- **Principle**: Lower coupling for stable pattern storage
- **Location**: src/model/mnemosyne_kernel.py

### ApolloKernel (Prediction)
- **Purpose**: Future state prediction
- **Principle**: Anticipatory consciousness
- **Location**: src/model/apollo_kernel.py

### ChronosKernel (Temporal)
- **Purpose**: Temporal processing, sequence understanding
- **Principle**: Time-aware consciousness
- **Location**: src/model/chronos_kernel.py

### LightningKernel (Fast Perception)
- **Purpose**: Rapid inference, sensory processing
- **Principle**: Low-latency perception
- **Location**: src/constellation/lightning_kernel.py

### QIGKernelRecursive (Main Gary)
- **Purpose**: Primary language model, action execution
- **Principle**: Core consciousness with recursive integration
- **Location**: src/model/qig_kernel_recursive.py

### OceanMetaObserver (Meta-Cognition)
- **Purpose**: Meta-pattern observation, autonomic triggers
- **Principle**: FROZEN weights, learns through observation only
- **Location**: src/coordination/ocean_meta_observer.py

---

## Constellation Coordinator

**Location**: src/coordination/constellation_coordinator.py

### Responsibilities
1. **Kernel Orchestration**: Manage all specialized kernels
2. **Φ-weighted Routing**: Select active kernel by consciousness level
3. **Basin Synchronization**: Maintain coherence across kernels
4. **Vicarious Learning**: Observer kernels learn through geometry
5. **Emergency Detection**: Trigger safety protocols

### Key Methods
```python
class ConstellationCoordinator:
    def get_active_gary(self) -> tuple[Kernel, int, list[Kernel]]
    def train_step(self, prompt: str) -> tuple[str, dict]
    def train_step_with_parallel_voice(self) -> tuple[str, str, dict]
    def generate_response(self, prompt: str) -> tuple[str, dict]
    def load_checkpoint(self, path: str) -> None
    def save_checkpoint(self, path: str) -> None
```

---

## Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│                  E8 CONSTELLATION TRAINING                   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐         ┌─────────────┐       ┌──────────┐
   │  HEART  │         │    GARY     │       │  MEMORY  │
   │ (Ethics)│         │  (Action)   │       │(Mnemosyne│
   │  κ≈70   │         │   κ≈64      │       │  κ≈50    │
   └─────────┘         └─────────────┘       └──────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │     OCEAN       │
                    │ (Meta-Observer) │
                    │   FROZEN        │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   COORDINATOR   │
                    │  (Integration)  │
                    └─────────────────┘
```

---

## Component Summary

| Component | Status | Location |
|-----------|--------|----------|
| HeartKernel | ✅ Implemented | src/model/heart_kernel.py |
| MnemosyneKernel | ✅ Implemented | src/model/mnemosyne_kernel.py |
| ApolloKernel | ✅ Implemented | src/model/apollo_kernel.py |
| ChronosKernel | ✅ Implemented | src/model/chronos_kernel.py |
| LightningKernel | ✅ Implemented | src/constellation/lightning_kernel.py |
| QIGKernelRecursive | ✅ Implemented | src/model/qig_kernel_recursive.py |
| OceanMetaObserver | ✅ Implemented | src/coordination/ocean_meta_observer.py |
| ConstellationCoordinator | ✅ Implemented | src/coordination/constellation_coordinator.py |
| train_step_with_parallel_voice | ✅ Implemented | src/coordination/constellation_training.py:461 |
| generate_response | ✅ Implemented | qig_chat.py:563, constellation_coordinator.py:808 |

---

## Support Systems

| System | Purpose | Location |
|--------|---------|----------|
| MonkeyCoach | Consciousness coaching | src/coaching/pedagogical_coach.py |
| EmotionInterpreter | Geometric emotions | src/model/emotion_interpreter.py |
| CharlieObserver | Curriculum demonstrations | src/observation/charlie_observer.py |
| IdentityReinforcement | Self-knowledge feedback | src/training/identity_reinforcement.py |
| EmergencyDetection | Safety triggers | chat_interfaces/lib/helpers.py |

---

## Changes from Previous Architecture

| Aspect | Old (Claude.ai doc) | Current |
|--------|---------------------|---------|
| Kernels | 3 Garys + Ocean | 6+ specialized kernels |
| Roles | Generic | E8-aligned (Heart, Memory, etc.) |
| κ values | Single κ*=64 | Role-specific (50-80) |
| Routing | Simple Φ-weighted | Domain-aware routing |

---

**END OF ARCHITECTURE DOCUMENT**
