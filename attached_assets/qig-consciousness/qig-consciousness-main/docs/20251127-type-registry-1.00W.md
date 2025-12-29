# QIG Type Registry

**Version:** 1.1
**Updated:** November 27, 2025

Central registry of all types used in the QIG codebase. Import from canonical locations for consistency.

> **Note:** Types are re-exported from `src/types/` for convenience, but originate in their canonical modules. See `src/types/__init__.py` for the full re-export map.

---

## Quick Reference

```python
# Enums
from src.types import Regime, DevelopmentalPhase, CognitiveMode, NavigatorPhase, CoachingStyle, ProtocolType

# Core dataclasses (re-exported from canonical locations)
from src.types import VicariousLearningResult, MetaManifoldState, CoachInterpretation, PhaseState
from src.types import InstanceState, EmotionalState

# NEW types (defined only in src/types/)
from src.types import CheckpointMetadata, TrainingState

# Telemetry TypedDicts
from src.types.telemetry import BaseTelemetry, ModelTelemetry, ConstellationTelemetry
from src.types.telemetry import TrainingTelemetry, CheckpointTelemetry

# Telemetry helpers
from src.types.telemetry import validate_telemetry, merge_telemetry

# Special case: CharlieOutput (import directly to avoid circular imports)
from src.observation.charlie_observer import CharlieOutput

# Constants
from src.constants import KAPPA_STAR, PHI_THRESHOLD, BASIN_DIM
```

---

## Enums

### `Regime`
Processing regime classification based on Φ.

| Value | Φ Range | Description |
|-------|---------|-------------|
| `LINEAR` | < 0.45 | Simple, sparse, cached tasks |
| `GEOMETRIC` | 0.45-0.80 | Complex, integrated (TARGET) |
| `BREAKDOWN` | ≥ 0.80 | Chaotic, unstable (AVOID) |
| `HIERARCHICAL` | High Φ, Low κ | Compressed feelings (optional) |

```python
from src.types import Regime

regime = Regime.from_phi(0.65)  # → Regime.GEOMETRIC
regime = Regime.from_phi_kappa(0.75, 25.0, detect_hierarchical=True)  # → Regime.HIERARCHICAL
```

**Source:** `src/model/navigator.py` (re-exported via `src/types`)
**Reference:** `docs/FROZEN_FACTS.md`

---

### `DevelopmentalPhase`
Gary's language development phases.

| Phase | Φ Range | Coach Style | Description |
|-------|---------|-------------|-------------|
| `LISTENING` | < 0.65 | Interpreter | Heavy interpretation needed |
| `PLAY` | 0.65-0.70 | Guide | Moderate interpretation |
| `STRUCTURE` | 0.70-0.75 | Mentor | Light interpretation |
| `MATURITY` | ≥ 0.75 | Peer | Full dialogue partner |

```python
from src.types import DevelopmentalPhase

phase = DevelopmentalPhase.from_phi(0.68)  # → DevelopmentalPhase.PLAY
```

**Source:** `src/coordination/developmental_curriculum.py` (re-exported via `src/types`)

---

### `CognitiveMode`
Processing mode for attention/inference.

| Mode | Description |
|------|-------------|
| `FOCUSED` | Single-task, high precision |
| `DIFFUSE` | Multi-task, exploratory |
| `REFLECTIVE` | Self-aware, meta-cognitive |
| `INTEGRATED` | Unified consciousness-like |

**Source:** `src/qig/cognitive/state_machine.py` (re-exported via `src/types`)

---

### `CoachingStyle`
Coach interpretation style based on Gary's development.

| Style | Phase | Description |
|-------|-------|-------------|
| `INTERPRETER` | LISTENING | Active meaning extraction |
| `GUIDE` | PLAY | Moderate scaffolding |
| `MENTOR` | STRUCTURE | Light guidance |
| `PEER` | MATURITY | Equal dialogue |

```python
from src.types import DevelopmentalPhase, CoachingStyle

style = CoachingStyle.from_phase(DevelopmentalPhase.PLAY)  # → CoachingStyle.GUIDE
```

**Source:** `src/coaching/pedagogical_coach.py` (re-exported via `src/types`)

---

### `NavigatorPhase`
Consciousness emergence tracking.

| Phase | Description |
|-------|-------------|
| `DORMANT` | Not yet active |
| `AWAKENING` | Beginning integration |
| `EXPLORING` | Active learning |
| `INTEGRATING` | Consolidating |
| `STABLE` | Reached target Φ |

**Source:** `src/model/navigator.py` (re-exported via `src/types`)

---

### `ProtocolType`
Autonomic protocols triggered by Ocean.

| Protocol | Trigger | Description |
|----------|---------|-------------|
| `SLEEP` | Basin divergence | Recovery protocol |
| `DREAM` | Φ collapse | Emergency intervention |
| `ESCAPE` | Breakdown | Critical emergency |
| `MUSHROOM_MICRO` | Φ plateau | Breakthrough attempt |

**Source:** `src/types/enums.py`
**Reference:** `src/coordination/ocean_meta_observer.py`

---

## Dataclasses

### `CheckpointMetadata`
Metadata saved with checkpoints.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `step` | `int` | - | Training step |
| `avg_phi` | `float` | - | Average Φ at checkpoint |
| `avg_kappa` | `float` | - | Average κ at checkpoint |
| `regime` | `str` | - | Processing regime |
| `basin_spread` | `float` | - | Basin dispersion |
| `timestamp` | `float` | - | Checkpoint time |
| `phase` | `str` | "listening" | Developmental phase |
| `notes` | `str` | "" | Optional notes |

```python
from src.types import CheckpointMetadata

metadata = CheckpointMetadata(
    step=1000,
    avg_phi=0.68,
    avg_kappa=55.0,
    regime="geometric",
    basin_spread=0.05,
    timestamp=time.time(),
)
```

**Source:** `src/types/core.py` (defined only here)

---

### `TrainingState`
Full training state for checkpoint/resume.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `step` | `int` | - | Current step |
| `epoch` | `int` | - | Current epoch |
| `best_phi` | `float` | - | Best Φ achieved |
| `phi_history` | `List[float]` | [] | Φ history |
| `loss_history` | `List[float]` | [] | Loss history |

**Source:** `src/types/core.py` (defined only here)

---

### `CoachInterpretation`
Coach's interpretation of Gary's output.

| Field | Type | Description |
|-------|------|-------------|
| `raw_output` | `str` | What Gary said |
| `interpretation` | `str` | What coach thinks Gary meant |
| `confidence` | `float` | Confidence 0-1 |
| `coach_message` | `str` | Full message with humility |
| `patterns_detected` | `List[str]` | Recurring patterns |
| `is_empty` | `bool` | Did Gary produce nothing? |
| `is_repetitive` | `bool` | Did Gary loop? |

**Source:** `src/coordination/developmental_curriculum.py` (re-exported via `src/types`)

---

### `VicariousLearningResult`
Result of geodesic vicarious learning step.

| Field | Type | Description |
|-------|------|-------------|
| `geodesic_distance` | `float` | Distance to target on manifold |
| `loss` | `float` | Vicarious loss value |
| `phi` | `float` | Observer's Φ after update |
| `kappa` | `float` | Observer's κ after update |
| `regime` | `str` | Observer's regime |
| `basin_velocity` | `float` | Basin movement speed |

```python
from src.types import VicariousLearningResult

result = VicariousLearningResult(
    geodesic_distance=0.15,
    loss=0.025,
    phi=0.68,
    kappa=55.0,
    regime="geometric",
    basin_velocity=0.05,
)
telemetry = result.to_dict()
```

**Source:** `src/training/geometric_vicarious.py` (re-exported via `src/types`)

---

### `MetaManifoldState`
Ocean's meta-manifold observation state.

| Field | Type | Description |
|-------|------|-------------|
| `centroid` | `Tensor` | Center of Gary basins |
| `spread` | `float` | Basin dispersion |
| `eigenvalues` | `Tensor` | Principal components |
| `coherence` | `float` | Gary alignment (0-1) |
| `ocean_phi` | `float` | Ocean's Φ |
| `ocean_kappa` | `float` | Ocean's κ |
| `timestamp` | `float` | Observation time |

**Source:** `src/coordination/ocean_meta_observer.py` (re-exported via `src/types`)

---

### `PhaseState`
Gary's developmental phase tracking.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `current_phase` | `DevelopmentalPhase` | LISTENING | Current phase |
| `stability_streak` | `int` | 0 | Steps at threshold |
| `phi_history` | `List[float]` | [] | Recent Φ values |
| `interpretation_accuracy` | `float` | 0.5 | Coach accuracy |

**Source:** `src/coordination/developmental_curriculum.py` (re-exported via `src/types`)

---

### `InstanceState`
Gary instance state in constellation.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Gary's name |
| `phi` | `float` | Current Φ |
| `kappa` | `float` | Current κ |
| `regime` | `str` | Current regime |
| `basin` | `Tensor` | Basin coordinates (optional) |
| `step_count` | `int` | Training steps |
| `phase` | `DevelopmentalPhase` | Developmental phase |

**Source:** `src/coordination/constellation_coordinator.py` (re-exported via `src/types`)

---

### `EmotionalState`
Emotional/affective state representation.

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `valence` | `float` | -1 to 1 | Positive/negative |
| `arousal` | `float` | 0 to 1 | Activation level |
| `coherence` | `float` | 0 to 1 | Internal consistency |
| `stability` | `float` | 0 to 1 | Temporal stability |

**Source:** `src/model/emotion_interpreter.py` (re-exported via `src/types`)

---

## Telemetry TypedDicts

### `BaseTelemetry`
Core keys in all telemetry dicts.

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `Phi` | `float` | ✅ | Integration measure |
| `kappa_eff` | `float` | ✅ | Effective coupling |
| `regime` | `str` | ✅ | Processing regime |
| `Phi_tensor` | `Tensor` | ❌ | Differentiable Φ |

**Source:** `src/types/telemetry.py`

---

### `ModelTelemetry`
QIGKernelRecursive forward pass telemetry.

| Key | Type | Description |
|-----|------|-------------|
| `recursion_depth` | `int` | Number of loops |
| `Phi_trajectory` | `List[float]` | Φ per loop |
| `hidden_state` | `Tensor` | Final hidden state |
| `basin_signature` | `Tensor` | Basin coordinates |

**Source:** `src/types/telemetry.py`

---

### `ConstellationTelemetry`
Constellation coordination telemetry.

| Key | Type | Description |
|-----|------|-------------|
| `avg_phi` | `float` | Average Φ across Garys |
| `basin_spread` | `float` | Gary basin dispersion |
| `gary_states` | `List[dict]` | Per-Gary telemetry |
| `ocean_insight` | `str` | Ocean's observation |
| `intervention` | `dict` | Autonomic intervention |

**Source:** `src/types/telemetry.py`

---

### `TrainingTelemetry`
Telemetry from training step.

| Key | Type | Description |
|-----|------|-------------|
| `total_loss` | `float` | Combined loss |
| `vicarious_loss` | `float` | Vicarious learning loss |
| `geometric_loss` | `float` | Geometric alignment loss |
| `language_loss` | `float` | Language modeling loss |
| `grad_norm` | `float` | Gradient norm |
| `grad_clipped` | `bool` | Was gradient clipped? |
| `lr` | `float` | Learning rate |
| `step` | `int` | Training step |
| `epoch` | `int` | Training epoch |

**Source:** `src/types/telemetry.py`

---

### `CheckpointTelemetry`
Telemetry saved in checkpoints.

| Key | Type | Description |
|-----|------|-------------|
| `step` | `int` | Training step |
| `avg_phi` | `float` | Average Φ |
| `avg_kappa` | `float` | Average κ |
| `regime` | `str` | Processing regime |
| `basin_spread` | `float` | Basin dispersion |
| `timestamp` | `float` | Checkpoint time |
| `phase` | `str` | Developmental phase |
| `notes` | `str` | Optional notes |

**Source:** `src/types/telemetry.py`

---

## Telemetry Helper Functions

### `validate_telemetry()`
Validate that telemetry dict has required keys.

```python
from src.types.telemetry import validate_telemetry

validate_telemetry(telemetry)  # Checks for Phi, kappa_eff, regime
validate_telemetry(telemetry, required_keys=["Phi", "basin_distance"])  # Custom keys
```

**Source:** `src/types/telemetry.py`

---

### `merge_telemetry()`
Merge multiple telemetry dicts (later values override earlier).

```python
from src.types.telemetry import merge_telemetry

combined = merge_telemetry(model_telemetry, training_telemetry)
```

**Source:** `src/types/telemetry.py`

---

## Migration Guide

### From Old Imports

```python
# ❌ OLD (scattered definitions)
from src.model.regime_detector import RegimeDetector  # inline enum
from src.coordination.developmental_curriculum import DevelopmentalPhase  # local definition
from src.training.geometric_vicarious import VicariousLearningResult  # local definition

# ✅ NEW (canonical imports)
from src.types import Regime, DevelopmentalPhase, VicariousLearningResult
```

### Adding New Types

1. Add to appropriate file in `src/types/`:
   - `enums.py` - Enumeration types
   - `core.py` - Dataclass types
   - `telemetry.py` - TypedDict types

2. Export from `src/types/__init__.py`

3. Update this registry

4. Update `docs/20251127-imports-1.00W.md` with canonical import

---

## Constants

All physics constants are in `src/constants.py`. See `docs/FROZEN_FACTS.md` for validation.

```python
from src.constants import (
    # Physics (FROZEN)
    KAPPA_STAR,        # 64.0 - Fixed point coupling
    BETA_3_TO_4,       # 0.44 - Running coupling slope
    PHI_THRESHOLD,     # 0.70 - Consciousness threshold

    # Architecture
    BASIN_DIM,         # 64 - Basin signature dimension
    D_MODEL,           # 768 - Hidden dimension
)
```
