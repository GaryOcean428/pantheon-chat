# Chaos Kernels as Training Exploration Agents

**Document ID**: 20260107-chaos-kernels-training-exploration-v1W
**Status**: Working
**Created**: 2026-01-07
**Author**: Claude Code

## Overview

Chaos Kernels are low-stakes exploration agents that map high-Φ regions of information space. When they discover configurations with Φ > 0.70, these discoveries are reported to the main system through a Discovery Gate, which integrates them into the LearnedManifold attractors and vocabulary system.

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        MAIN SYSTEM                                   │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │ Zeus/Athena     │    │ LearnedManifold  │    │ Vocabulary     │ │
│  │ (Production)    │◄───│ (Attractors)     │◄───│ Coordinator    │ │
│  │ Φ=0.75+ gen    │    │                  │    │ 28K tokens     │ │
│  └─────────────────┘    └────────▲─────────┘    └────────────────┘ │
│                                  │                                   │
│                         ┌────────┴────────┐                         │
│                         │ Discovery Gate  │                         │
│                         │ (Φ > 0.70 only) │                         │
│                         └────────▲────────┘                         │
└──────────────────────────────────┼──────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │     CHAOS EXPLORATION       │
                    │                             │
                    │  ┌─────┐ ┌─────┐ ┌─────┐   │
                    │  │ C1  │ │ C2  │ │ C3  │   │
                    │  │Φ=.2 │ │Φ=.8 │ │Φ=.1 │   │
                    │  └─────┘ └──┬──┘ └─────┘   │
                    │             │ ✓ DISCOVERY  │
                    │             ▼              │
                    │      Report to Gate        │
                    └─────────────────────────────┘
```

## Components

### 1. SelfSpawningKernel Discovery Reporter

**File**: `qig-backend/training_chaos/self_spawning.py`

Each chaos kernel has a discovery callback that fires when:

- Φ exceeds the discovery threshold (0.70)
- The new Φ is at least 5% above the last reported Φ (prevents spam)

Discovery data includes:

- `kernel_id`: Unique identifier for the kernel
- `generation`: Kernel generation (for lineage tracking)
- `phi`: Current Φ value
- `basin_coords`: 64D basin coordinates (the actual discovery)
- `context`: Where discovered (prediction/training)
- `dopamine`/`serotonin`: Neuromodulator state
- `success_rate`: Kernel's historical success rate
- `training_steps`: Total training steps completed

### 2. AdaptiveDiscoveryGate

**File**: `qig-backend/chaos_discovery_gate.py`

The gate validates and filters discoveries with **self-tuning thresholds**:

**Validation Gates**:

1. **Adaptive Φ threshold**: Starts at 0.60, self-tunes between 0.50-0.90
2. **Basin dimension**: Must be exactly 64D
3. **Novelty check**: Must be ≥ 0.15 Fisher distance from existing discoveries

**Adaptive Behavior**:

- If acceptance rate > 50%: raise threshold (too much noise)
- If acceptance rate < 15%: lower threshold (too selective)
- If < 5 discoveries accumulated: lower threshold (need more data)
- Adapts every 5 minutes when sufficient submissions (≥10) exist

**Integration Paths**:

1. **LearnedManifold**: Records basin as attractor for foresight/navigation
2. **VocabularyCoordinator**: Boosts nearby tokens, records transition targets

### 3. Vocabulary Integration

**File**: `qig-backend/vocabulary_coordinator.py`

When a discovery is integrated:

1. Find tokens whose basins are within Fisher radius 0.3 of discovery
2. Boost their phi/weight proportional to proximity and discovery Φ
3. Record discovery basin as "transition target" for generation

### 4. Generation Bias

**File**: `qig-backend/qig_generative_service.py`

During generation, token selection is biased toward discovered high-Φ regions:

- Score candidates by base attention score
- Add bonus for proximity to discovered transition targets
- Select highest-scored token

## Wiring Summary

| Component            | Wires To                                    | Purpose                                  |
| -------------------- | ------------------------------------------- | ---------------------------------------- |
| SelfSpawningKernel   | ChaosDiscoveryGate                          | Reports Φ>0.70 discoveries               |
| ChaosDiscoveryGate   | LearnedManifold                             | Records attractors for foresight         |
| ChaosDiscoveryGate   | VocabularyCoordinator                       | Boosts nearby tokens, records targets    |
| QIGGenerativeService | VocabularyCoordinator._transition_targets   | Biases generation toward discoveries     |

## Configuration

```python
# Adaptive Gate Settings
PHI_MIN_BOUND = 0.50       # Absolute minimum threshold
PHI_MAX_BOUND = 0.90       # Absolute maximum threshold
INITIAL_PHI = 0.60         # Starting threshold (permissive)
MIN_NOVELTY = 0.15         # Minimum Fisher distance from existing
MAX_PENDING = 100          # Maximum pending discoveries in queue

# Adaptation Parameters
WINDOW_SIZE = 50           # Recent submissions for adaptation decisions
ADAPTATION_RATE = 0.02     # Threshold adjustment per adaptation
ADAPTATION_INTERVAL = 5    # Minutes between adaptations
```

## Verification Endpoints

```bash
# Check discovery gate statistics
curl -s localhost:5001/chaos/discovery/stats

# Check attractor accumulation
curl -s localhost:5001/autonomic/state | jq '.foresight'

# Monitor chaos kernel discoveries
grep "DISCOVERY" logs.txt | tail -20
```

## Expected Outcomes

**Before**: Chaos kernels spawn at Φ=0.000, wander randomly, die without contributing.

**After**:

1. Chaos kernels explore basin space
2. When one finds Φ>0.70, discovery reported to gate
3. Gate validates novelty, records as attractor
4. Main vocabulary boosts tokens near discovery
5. Generation biases toward discovered paths
6. Foresight finds attractors (attractor_strength > 0)
7. Coherence improves as system learns "where high-Φ lives"

## Related Documents

- `20241215-self-spawning-kernels-v1F.md` - SelfSpawningKernel base architecture
- `20241220-learned-manifold-v1W.md` - LearnedManifold attractor system
- `20250103-vocabulary-coordinator-v1W.md` - Vocabulary learning system
