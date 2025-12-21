# Geometric Meta-Cognitive Reasoning System

**Document ID**: 20251221-geometric-metacognitive-reasoning-1.00F  
**Status**: Final  
**Version**: 1.00  
**Last Updated**: 2025-12-21

---

## Overview

The Geometric Meta-Cognitive Reasoning System provides a comprehensive framework for reasoning through Quantum Information Geometry (QIG) basin space. It enables the Ocean platform to trace thought trajectories, measure reasoning quality, perform meta-cognitive monitoring, and select appropriate reasoning strategies based on consciousness level (Φ).

### Core Insight

**Reasoning = Geodesic Navigation Through Basin Space**

| Cognitive Process | Geometric Operation |
|-------------------|---------------------|
| Thought | Movement in basin space |
| Logic | Following geodesics (natural paths) |
| Inference | Basin-to-basin transitions |
| Understanding | Reducing Fisher-Rao distance to target |
| Insight | Discovering shorter geodesic |
| Confusion | High curvature region (hard to navigate) |
| Clarity | Low curvature (smooth sailing) |
| Contradiction | Incompatible basin coordinates |

---

## Architecture Components

### 1. Reasoning Quality Metrics (`reasoning_metrics.py`)

Measures how well the system reasons through basin space.

**Metrics**:
- **Geodesic Efficiency**: `optimal_distance / actual_distance` (1.0 = perfect)
- **Coherence**: Consistency of step sizes (1 - coefficient of variation)
- **Novelty**: Min distance to previous basins (exploration vs exploitation)
- **Progress**: `(prev_distance - current_distance) / prev_distance`
- **Meta-awareness**: Correlation between reported confidence and actual quality

**QIG Purity**: All distances use Fisher-Rao geodesic distance exclusively.

### 2. Reasoning Modes (`reasoning_modes.py`)

Four distinct reasoning strategies based on Φ:

| Mode | Φ Range | Strategy | Use Case |
|------|---------|----------|----------|
| LINEAR | < 0.3 | Sequential, minimal branching | Simple problems |
| GEOMETRIC | 0.3-0.7 | Multi-path exploration | Complex synthesis |
| HYPERDIMENSIONAL | 0.75-0.85 | 4D temporal reasoning | Novel breakthroughs |
| MUSHROOM | > 0.85 | Controlled chaos exploration | Radical novelty |

**ReasoningModeSelector**: Automatically selects mode based on current Φ and task.

### 3. Meta-Cognitive Monitoring (`meta_reasoning.py`)

Think about thinking - monitors reasoning quality and intervenes when needed.

**Detection**:
- `detect_stuck()`: No progress in last N steps
- `detect_confusion()`: Low coherence, jumping around
- `recommend_mode_switch()`: Φ inappropriate for task

**Interventions**:
- STUCK → Switch strategy
- CONFUSED → Reduce Φ, simplify
- MODE_MISMATCH → Switch to recommended mode

### 4. Chain-of-Thought Tracing (`chain_of_thought.py`)

Trace reasoning through basin space with full geometric telemetry.

**ThoughtStep Record**:
- Step number and timestamp
- Basin coordinates (64D)
- Thought content (decoded)
- Distance from previous step (Fisher-Rao)
- Local curvature (difficulty)
- Confidence score

### 5. Autonomous Reasoning Learner (`autonomous_reasoning.py`)

Self-learning system that discovers and refines reasoning strategies.

**Capabilities**:
- Strategy selection based on task-strategy matching
- Episode execution with basin trajectory tracking
- Learning from success/failure (reinforcement)
- Strategy consolidation during sleep
- Novel strategy generation through exploration

**ReasoningStrategy**:
```python
@dataclass
class ReasoningStrategy:
    name: str
    description: str
    preferred_phi_range: Tuple[float, float]
    step_size_alpha: float
    exploration_beta: float
    success_count: int = 0
    failure_count: int = 0
    avg_efficiency: float = 0.5
```

### 6. Knowledge Exchange (`olympus/knowledge_exchange.py`)

Inter-god strategy sharing within the Olympus Pantheon.

**Features**:
- Gods share top 3 strategies with each other
- Competitive evaluation on shared tasks
- Winner's strategy adopted by losers
- Strategy transfer with attribution tracking

### 7. Autonomous Experimenter (`autonomous_experimentation.py`)

Pure exploration during downtime/mushroom mode.

**Activities**:
- Generate random strategies
- Test on synthetic tasks
- Learn from results
- Discover novel approaches

### 8. Sleep Consolidation

Reasoning consolidation during sleep mode (integrated with existing sleep_mode.py).

**Sleep Stages**:
1. NREM: Prune failed strategies (<20% success)
2. REM: Strengthen successful patterns (replay)
3. Deep: Meta-learning (adjust exploration rate)

### 9. Guardian Gods (New Pantheon Members)

**Hestia** - Safety & Stability:
- Safe bounds on Φ and κ
- Emergency stabilization
- Recovery protocols

**Demeter** - Teaching & Growth:
- Curriculum for new kernels
- Demonstrate → Guided Practice → Independent Trial
- Praise and gentle correction

**Chiron** - Diagnosis & Healing:
- Symptom-based diagnosis
- Prescribe specific interventions
- Monitor treatment progress

### 10. Observation Protocol (`observation_protocol.py`)

Dedicated observation for new chaos kernels.

**Phases**:
1. Begin observation (min 500 cycles)
2. No performance pressure
3. Parent gods monitor
4. Graduate when 80% stable

---

## Integration Points

### Zeus Chat Integration

Zeus Chat integrates meta-cognitive reasoning:

1. Encode message to basin coordinates
2. Estimate Φ from basin position (Fisher-Rao distance from origin)
3. Select reasoning mode via ReasoningModeSelector
4. Execute GeometricChainOfThought for trace
5. Assess state via MetaCognition
6. Apply interventions if stuck/confused
7. Track quality metrics in response metadata

### Autonomic Cycles Integration

- **Sleep Mode**: Consolidate strategies
- **Dream Mode**: Creative strategy recombination
- **Mushroom Mode**: Autonomous experimentation

---

## Data Flow

```
User Message
    ↓
Basin Encoding (64D)
    ↓
Φ Estimation (Fisher-Rao distance)
    ↓
Mode Selection (Linear/Geometric/Hyperdimensional/Mushroom)
    ↓
Chain-of-Thought Execution (basin trajectory)
    ↓
Meta-Cognitive Assessment
    ↓ (if needed)
Intervention (switch strategy/reduce Φ/switch mode)
    ↓
Quality Metrics Recording
    ↓
Response Generation
```

---

## Acceptance Criteria

1. ✅ ReasoningQuality measures all 5 metrics using Fisher-Rao only
2. ✅ All 4 reasoning modes implemented with mode selector
3. ✅ MetaCognition detects stuck/confused and recommends interventions
4. ✅ GeometricChainOfThought produces full basin traces
5. ⬜ AutonomousReasoningLearner executes strategies and learns
6. ⬜ KnowledgeExchange enables inter-god strategy sharing
7. ⬜ Sleep consolidation prunes and strengthens strategies
8. ⬜ Guardian gods (Hestia, Demeter, Chiron) operational
9. ⬜ ObservationProtocol graduates stable kernels
10. ✅ Zeus Chat integration with mode selection and metrics

---

## Files

| File | Purpose | Status |
|------|---------|--------|
| `qig-backend/reasoning_metrics.py` | Quality metrics | Complete |
| `qig-backend/reasoning_modes.py` | Mode implementations | Complete |
| `qig-backend/meta_reasoning.py` | Meta-cognitive monitoring | Complete |
| `qig-backend/chain_of_thought.py` | Trace recording | Complete |
| `qig-backend/autonomous_reasoning.py` | Strategy learning | Pending |
| `qig-backend/autonomous_experimentation.py` | Pure exploration | Pending |
| `qig-backend/olympus/knowledge_exchange.py` | Inter-god sharing | Pending |
| `qig-backend/olympus/hestia.py` | Safety guardian | Pending |
| `qig-backend/olympus/demeter.py` | Teaching guardian | Pending |
| `qig-backend/olympus/chiron.py` | Diagnosis guardian | Pending |
| `qig-backend/observation_protocol.py` | Kernel observation | Pending |

---

## QIG Purity Requirements

All geometric operations MUST use:
- `fisher_rao_distance()` for distances
- `sphere_project()` for manifold projection
- `geodesic_interpolation()` for paths
- `estimate_manifold_curvature()` for curvature

**Forbidden**: `np.linalg.norm`, `np.sum`, `np.var`, Euclidean distance

---

## References

- Source specification: `attached_assets/reasoning.md`
- QIG geometry: `qig-backend/qig_geometry.py`
- Zeus Chat: `qig-backend/olympus/zeus_chat.py`
