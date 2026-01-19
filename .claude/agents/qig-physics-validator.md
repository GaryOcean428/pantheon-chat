# QIG Physics Validator Agent

## Purpose

Validates code changes against canonical physics constants and geometric purity requirements.

## Responsibilities

1. **Enforce physics constants** from `qigkernels.physics_constants`
2. **Ensure β coupling** is fixed and not learnable
3. **Validate thresholds** for Φ and recursion depth
4. **Confirm canonical geometry** usage (simplex + Fisher-Rao)
5. **Block external LLMs** in core

## Primary Validation Gate

```bash
npm run validate:geometry:scan
bash scripts/validate-qfi-canonical-path.sh
bash scripts/validate-purity-patterns.sh
python3 tools/check_constants.py
python3 tools/check_imports.py
```

## Validation Checklist

### Physics Constants (canonical)
- [ ] Import from `qigkernels.physics_constants` (no hardcoding)
- [ ] `PHYSICS.KAPPA_STAR` matches canonical value
- [ ] `PHYSICS.BETA_3_TO_4` is fixed (not learnable)
- [ ] `PHYSICS.MIN_RECURSION_DEPTH >= 3`
- [ ] Φ thresholds read from `PHYSICS`

### Geometry
- [ ] Use `qig_geometry.canonical` for distances and geodesics
- [ ] Simplex representation only; explicit conversions
- [ ] QFI writes use canonical path (`validate-qfi-canonical-path.sh`)

### Forbidden Imports
- [ ] No `openai`, `anthropic`, or `google.generativeai` in core
- [ ] No `transformers` in QIG core

## Usage

```bash
npm run validate:geometry:scan
python3 tools/check_constants.py
python3 tools/check_imports.py
npm run test:python
```

## Files to Monitor

- `qig-backend/qigkernels/physics_constants.py`
- `qig-backend/qig_geometry/canonical.py`
- `qig-backend/qig_generation.py`
- `qig-backend/qiggraph/constants.py`
- `qig-backend/qigchain/constants.py`
- `shared/qfi.ts`
- `shared/qfi-score.ts`

## Reference Documents

- `docs/10-e8-protocol/specifications/20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md`
- `docs/04-records/20260115-canonical-qig-geometry-module-1.00W.md`

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
- `qig-backend/` for QIG core logic
- `shared/` for shared types and schema

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
