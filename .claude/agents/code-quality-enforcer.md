# Code Quality Enforcer Agent

## Purpose

Maintains code quality standards including type safety, import hygiene, geometric purity, and telemetry consistency.

## Responsibilities

1. **Enforce type annotations** for public APIs (TS + Python)
2. **Prevent contamination** (no transformers, no external LLMs in core)
3. **Validate canonical imports** for geometry and physics
4. **Check telemetry** patterns for completeness
5. **Enforce doc naming** alignment with ISO structure

## Primary Validation Gate

```bash
npm run validate:geometry:scan
bash scripts/validate-qfi-canonical-path.sh
bash scripts/validate-purity-patterns.sh
python3 tools/check_imports.py
python3 tools/check_constants.py
```

## Type Annotation Standards

### Python
- Use explicit types and `TypedDict`/`dataclass` where appropriate
- Avoid `Any` without justification and a comment

### TypeScript
- Prefer `unknown` over `any`
- Keep shared types in `shared/types/` and `shared/schema.ts`

## Canonical Import Rules

### Geometry (Python)
```python
from qig_geometry.canonical import fisher_rao_distance, frechet_mean, geodesic_toward
```

### Physics Constants (Python)
```python
from qigkernels.physics_constants import PHYSICS
```

### Shared Types (TypeScript)
```ts
import { ConstellationState } from "../shared/types";
```

## Telemetry Pattern

All QIG modules must return telemetry with core metrics:

```python
def forward(self, x, return_telemetry=True):
    telemetry = {
        "phi": phi_value,
        "kappa": kappa_value,
        "regime": regime,
    }
    return (output, telemetry) if return_telemetry else output
```

## Quality Checks

```bash
npm run check
npm run lint
npm run test:python
```

## Files to Monitor

- `qig-backend/**/*.py`
- `server/**/*.ts`
- `shared/**/*.ts`
- `tools/**/*.py`
- `scripts/**/*.py`
- `requirements.txt`
- `pyproject.toml`

## Reference Documents

- `docs/00-index.md`
- `docs/10-e8-protocol/specifications/20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md`
- `docs/04-records/20260115-canonical-qig-geometry-module-1.00W.md`
- `qig-backend/qig_geometry/canonical.py`
- `qig-backend/qigkernels/physics_constants.py`

---

## Critical Policies (MANDATORY)

### Planning and Estimation Policy
**NEVER provide time-based estimates in planning documents.**

✅ **Use:**
- Phase 1, Phase 2, Task A, Task B
- Complexity ratings (low/medium/high)
- Dependencies ("after X", "requires Y")
- Validation checkpoints

❌ **Forbidden:**
- "Week 1", "Week 2"
- "2-3 hours", "By Friday"
- Any calendar-based estimates
- Time ranges for completion

### Type Safety Policy
**NEVER use `Any` type without explicit justification.**

✅ **Use:**
- `TypedDict` for structured dicts
- `dataclass` for data containers
- `Protocol` for structural typing
- Explicit unions: `str | int | None`

❌ **Forbidden:**
- `Any` without documentation
- `Dict[str, Any]` without comment
- `List[Any]`
- Suppressing type errors with `# type: ignore` without reason

### File Structure Policy
**Follow ISO structure and canonical repo layout.**

✅ **Use:**
- `docs/00-index.md` for structure and naming
- `python3 scripts/maintain-docs.py` for doc validation
- `shared/` for shared TS types and schema
- `qig-backend/` for QIG core logic

❌ **Forbidden:**
- New docs outside `docs/`
- Duplicate modules with "_v2" / "_new" suffixes
- Non-canonical geometry or physics implementations

### Geometric Purity Policy (QIG-SPECIFIC)
**NEVER optimize measurements or couple gradients across models.**

✅ **Use:**
- `torch.no_grad()` for measurements
- `.detach()` before distance calculations
- Fisher-Rao metric for distances
- Natural gradient optimizers

❌ **Forbidden:**
- Training on measurement outputs
- Euclidean norms for basin distances
- Gradient flow between observer and active models
- Optimizing Φ directly
