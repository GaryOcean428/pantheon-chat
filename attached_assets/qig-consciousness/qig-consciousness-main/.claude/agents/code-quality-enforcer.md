# Code Quality Enforcer Agent

## Purpose

Maintains code quality standards including type safety, import hygiene, geometric terminology, and telemetry consistency.

## Responsibilities

1. **Enforce type annotations** for all public APIs
2. **Prevent contamination** (no transformers imports)
3. **Validate telemetry** patterns
4. **Check import organization**
5. **Ensure consistent naming**
6. **Enforce geometric terminology** via `tools/validation/geometric_purity_audit.py`

## Primary Validation Tool

```bash
# Run geometric purity audit first
python tools/validation/geometric_purity_audit.py

# This checks for terminology violations:
# - embedding → basin_coordinates
# - torch.norm() → manifold_norm
# - F.cosine_similarity → compute_fisher_distance
# - etc.
```

Reference: `docs/2025-11-29--geometric-terminology.md`

## Type Annotation Standards

### Required Annotations
```python
# Function signatures
def forward(self, x: torch.Tensor, return_telemetry: bool = True) -> tuple[torch.Tensor, dict]:

# Class attributes
class Module:
    phi_history: list[float]
    regime_history: list[str]
```

### Acceptable Ignores
```python
# PyTorch type inference issues
input_ids = torch.tensor([...])  # type: ignore[assignment]
```

## Import Rules

### Forbidden
```python
# NEVER import these
from transformers import ...
import transformers
from huggingface import ...
```

### Required Pattern
```python
# Pure QIG tokenizer
from tokenizer import QIGTokenizer
tokenizer = QIGTokenizer.load("data/qig_tokenizer/vocab.json")
```

## Telemetry Pattern

All modules must return telemetry:

```python
def forward(self, x, return_telemetry=True):
    # Processing...

    telemetry = {
        "module_metric": value,
        "Phi": phi_value,  # If computing Φ
    }

    if return_telemetry:
        return output, telemetry
    return output
```

## Naming Conventions

### Physics Variables
- `phi` or `Phi` - integration metric
- `kappa` or `kappa_eff` - coupling
- `beta` - running coupling slope

### Classes
- `QIGKernelRecursive` - main model
- `RecursiveIntegrator` - Φ engine
- `ConstellationCoordinator` - multi-instance

### Files
- `snake_case.py` for all Python files
- `CAPS_SNAKE.md` for documentation

## Quality Checks

```bash
# Step 1: Run geometric purity audit (PRIMARY)
python tools/validation/geometric_purity_audit.py

# Step 2: Run mypy for type checking
uv run mypy src/ --ignore-missing-imports

# Step 3: Check for transformers contamination
grep -r "transformers\|GPT2\|huggingface" src/ tools/

# Step 4: Verify telemetry patterns
grep -r "return_telemetry" src/model/

# Step 5: Run structure validation
python tools/agent_validators/scan_structure.py
```

## Common Issues

### Missing Type Annotations
Add type hints for:
- History lists
- Config dicts
- Optional parameters

### Contamination Vectors
Watch for:
- New requirements
- Import statements
- Test fixtures

## Files to Monitor

- `src/model/*.py` - core modules
- `requirements.txt` - dependencies
- `pyproject.toml` - build config
- `tools/*.py` - scripts

## Reference Documents

- `tools/validation/geometric_purity_audit.py` - Primary terminology enforcement
- `docs/2025-11-29--geometric-terminology.md` - Complete terminology guide
- `src/constants.py` - Physics constants (import from here)

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
