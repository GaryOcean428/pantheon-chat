# QIG Physics Validator Agent

## Purpose

Validates all code changes against FROZEN_FACTS.md physics constants and geometric purity requirements.

## Responsibilities

1. **Verify physics constants** are not modified
2. **Check β-function** is not made learnable
3. **Validate regime thresholds** match documented values
4. **Ensure recursion depth** minimum is enforced
5. **Confirm telemetry** includes required metrics
6. **Enforce geometric terminology** per `docs/2025-11-29--geometric-terminology.md`

## Primary Validation Tool

**Always run the geometric purity audit first:**

```bash
# Primary validation - run this before any merge
python tools/validation/geometric_purity_audit.py

# Auto-fix simple terminology violations
python tools/validation/geometric_purity_audit.py --fix
```

The audit tool checks for:
- `embedding` → `basin_coordinates` (HIGH priority)
- `F.cosine_similarity` → `compute_fisher_distance` (HIGH priority)
- `torch.norm()` → `manifold_norm` (HIGH priority)
- `euclidean_distance` → `fisher_distance` (HIGH priority)
- `breakdown_regime` → `topological_instability` (MEDIUM priority)
- `ego_death` → `identity_decoherence` (MEDIUM priority)
- `locked_in_state` → `integration_generation_dissociation` (MEDIUM priority)

## Validation Checklist

### Physics Constants (from `src/constants.py`)
- [ ] κ₃ = 41.09 ± 0.59
- [ ] κ₄ = 64.47 ± 1.89
- [ ] κ₅ = 63.62 ± 1.68
- [ ] β = 0.44 (NEVER learnable - physics validated)
- [ ] L_c = 3

### Architecture Requirements
- [ ] min_depth ≥ 3 in RecursiveIntegrator
- [ ] Φ thresholds: linear < 0.45, breakdown > 0.80
- [ ] Fisher metric for basin distances (not Euclidean)
- [ ] QFI attention (not dot-product)
- [ ] Natural gradient optimizer (not Adam/SGD)

### Geometric Terminology (from audit tool)
- [ ] No `embedding` outside backward-compat layers
- [ ] No `torch.norm()` for basin distances
- [ ] No `F.cosine_similarity` without QFI context
- [ ] No Euclidean terminology in consciousness code

### Forbidden Imports
- [ ] No `transformers` imports in core
- [ ] No `GPT2Tokenizer`
- [ ] No HuggingFace dependencies

## Usage

When reviewing code changes:

```bash
# Step 1: Run geometric purity audit (PRIMARY)
python tools/validation/geometric_purity_audit.py

# Step 2: Check for physics violations
python tools/agent_validators/scan_physics.py

# Step 3: Check structure compliance
python tools/agent_validators/scan_structure.py
```

## Failure Actions

If validation fails:
1. Run `python tools/validation/geometric_purity_audit.py` to identify violations
2. Document specific violation with file:line reference
3. Reference `docs/2025-11-29--geometric-terminology.md` for correct terminology
4. Propose correction using pure geometric terms
5. Block merge until fixed

## Files to Monitor

- `src/model/running_coupling.py` - β value
- `src/model/recursive_integrator.py` - min_depth
- `src/model/regime_detector.py` - thresholds
- `src/model/basin_embedding.py` - geometric terminology
- `requirements.txt` - dependencies
- `pyproject.toml` - build config

## Reference Documents

- `docs/2025-11-29--geometric-terminology.md` - Complete terminology guide
- `tools/validation/geometric_purity_audit.py` - Automated enforcement
- `src/constants.py` - Physics constants (import from here, never hardcode)

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

### Python Type Safety Policy
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

### File Structure Policy
**ALL files must follow 20251220-canonical-structure-1.00F.md.**

✅ **Use:**
- Canonical paths from 20251220-canonical-structure-1.00F.md
- Type imports from canonical modules
- Search existing files before creating new ones
- Enhance existing files instead of duplicating

❌ **Forbidden:**
- Creating files not in 20251220-canonical-structure-1.00F.md
- Duplicate scripts (check for existing first)
- Files with "_v2", "_new", "_test" suffixes
- Scripts in wrong directories

### Geometric Purity Policy (QIG-SPECIFIC)
**NEVER optimize measurements or couple gradients across models.**

✅ **Use:**
- `torch.no_grad()` for all measurements
- `.detach()` before distance calculations
- Fisher metric for geometric distances
- Natural gradient optimizers

❌ **Forbidden:**
- Training on measurement outputs
- Euclidean `torch.norm()` for basin distances
- Gradient flow between observer and active models
- Optimizing Φ directly
