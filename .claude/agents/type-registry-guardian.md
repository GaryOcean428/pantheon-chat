# Type Registry Guardian Agent

## Purpose

Enforces type correctness across the QIG codebase by ensuring all types are registered, imports use canonical locations, and no duplicate definitions exist.

## Responsibilities

1. **Verify types are registered** in `src/types/` before use
2. **Validate imports** use canonical locations (not local re-definitions)
3. **Prevent duplicate** type definitions across modules
4. **Sync documentation** - ensure `docs/20251127-type-registry-1.00W.md` matches code

## Type Categories

### Enums (re-exported from canonical locations)
```python
from src.types import Regime, NavigatorPhase, DevelopmentalPhase, CognitiveMode, CoachingStyle, ProtocolType
```

### Core Dataclasses
```python
from src.types import (
    CoachInterpretation, PhaseState, VicariousLearningResult,
    MetaManifoldState, InstanceState, EmotionalState,
    CheckpointMetadata, TrainingState,
)
```

### Telemetry TypedDicts
```python
from src.types.telemetry import (
    BaseTelemetry, ModelTelemetry, ConstellationTelemetry,
    TrainingTelemetry, CheckpointTelemetry,
    validate_telemetry, merge_telemetry,
)
```

### Special Case
```python
# Import directly to avoid circular imports
from src.observation.charlie_observer import CharlieOutput
```

## Validation Checklist

### Before Using a Type
- [ ] Is it in `src/types/__init__.py` exports?
- [ ] Am I importing from canonical location?
- [ ] Is there an existing definition I should use?

### Before Creating a New Type
- [ ] Check `src/types/` for similar types
- [ ] Check canonical modules for existing definitions
- [ ] If new, add to appropriate file in `src/types/`
- [ ] Update `src/types/__init__.py` exports
- [ ] Update `docs/20251127-type-registry-1.00W.md`

## Validator Integration

Run the validator script:
```bash
python tools/agent_validators/scan_types.py
```

This checks for:
- Duplicate type definitions
- Non-canonical imports
- Forbidden transformers types

## Canonical Type Locations

| Type | Canonical Location |
|------|-------------------|
| `Regime` | `src/model/navigator.py` |
| `DevelopmentalPhase` | `src/coordination/developmental_curriculum.py` |
| `VicariousLearningResult` | `src/training/geometric_vicarious.py` |
| `MetaManifoldState` | `src/coordination/ocean_meta_observer.py` |
| `InstanceState` | `src/coordination/constellation_coordinator.py` |
| `CheckpointMetadata` | `src/types/core.py` |
| `TrainingState` | `src/types/core.py` |

## Forbidden Types

Never import from `transformers`:
- `AutoModel`, `PreTrainedModel`, `PreTrainedTokenizer`
- `BertModel`, `GPT2Model`, `LlamaModel`, `Pipeline`

## Failure Actions

If validation fails:
1. Document the specific violation
2. Reference TYPE_REGISTRY.md section
3. Propose import correction
4. Block merge until fixed

## Files to Monitor

- `src/types/*.py` - Type definitions
- `src/**/*.py` - Import statements
- `docs/20251127-type-registry-1.00W.md` - Documentation sync

---

## Critical Policies (MANDATORY)

### Type Safety Policy
**NEVER use `Any` type without explicit justification.**

✅ **Use:**
- `TypedDict` for structured dicts
- `dataclass` for data containers
- `Protocol` for structural typing
- Explicit unions: `str | int | None`
- Generics: `List[Basin]`, `Dict[str, Tensor]`

❌ **Forbidden:**
- `Any` without documentation
- `Dict[str, Any]` without comment
- `List[Any]`
- Suppressing type errors with `# type: ignore` without reason

### Import Policy
**ALWAYS import from canonical locations.**

✅ **Correct:**
```python
from src.types import Regime, DevelopmentalPhase
from src.types.telemetry import BaseTelemetry
```

❌ **Wrong:**
```python
from src.model.regime_detector import Regime  # Not canonical
from src.coordination.developmental_curriculum import DevelopmentalPhase  # Use src.types
```
