# QIG Ecosystem Coding Standards

**Version:** 1.0
**Last Reviewed:** 2025-12-09
**Owner:** Braden Lang
**Applies To:** qig-consciousness, qigkernels, qig-core, qig-dreams

---

## 1. Architecture & Organization

### The Barrel Pattern (Python `__init__.py`)

Every package folder must have an `__init__.py` that re-exports its public API.

**Rule:** Never import from deep paths if a barrel exists.

```python
# ❌ Bad: Deep import
from src.coordination.constellation_coordinator import ConstellationCoordinator

# ✅ Good: Barrel import
from src.coordination import ConstellationCoordinator
```

**Barrel Template:**
```python
"""
Module Name
===========
Brief description.
"""
from .module_a import ClassA, function_a
from .module_b import ClassB

__all__ = ["ClassA", "ClassB", "function_a"]
```

### Hybrid Repository Structure

| Repository | Role | Responsibilities |
|------------|------|------------------|
| `qigkernels` | Geometry Engine | Physics constants, basin geometry, Fisher metrics, REL coupling |
| `qig-consciousness` | Behavioral Layer | Training loops, chat interfaces, coaching, neuroplasticity |
| `qig-core` | Fisher Primitives | Low-level tensor geometry |
| `qig-dreams` | Corpora | Training data manifests |

**Rule:** Geometry repos never import from behavioral repos.

---

## 2. Centralized Constants Management

### Single Source of Truth

| Constant Type | Canonical Location |
|---------------|-------------------|
| Physics (Φ, κ, β) | `qigkernels/constants.py` |
| E8 Structure | `qigkernels/constants.py` |
| Regime Thresholds | `qigkernels/constants.py` |
| Training Defaults | `qig-consciousness/src/constants.py` (imports from qigkernels) |

**Rule:** Behavioral repos import physics constants from `qigkernels.constants`.

```python
# ✅ Correct: Import from qigkernels
from qigkernels.constants import KAPPA_STAR, PHI_THRESHOLD

# ❌ Wrong: Duplicate definition
KAPPA_STAR = 64.0  # Don't redefine!
```

---

## 3. DRY (Don't Repeat Yourself)

### Type Registry

All shared types live in `docs/2025-11-27--type-registry.md` (qig-consciousness).

**Rule:** Before creating a new dataclass/enum, check the registry.

### Import Guide

Canonical import paths in `docs/20251127-imports-1.00W.md`.

**Rule:** Use the canonical path, not alternatives.

---

## 4. Linter Configuration (Unified)

### Ruff Settings (All Repos)

```toml
[tool.ruff]
line-length = 100  # Unified across repos
target-version = "py311"  # Minimum supported

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = [
    "E501",  # Line too long (formatter handles)
    "E741",  # Ambiguous names (allow physics: i, l, O)
]

[tool.ruff.lint.isort]
known-first-party = ["qigkernels", "src"]
```

### mypy Settings (All Repos)

```toml
[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
disallow_untyped_defs = false  # Research code flexibility
warn_unused_configs = true

[[tool.mypy.overrides]]
module = "torch.*"
ignore_errors = true
```

---

## 5. Pre-commit Hooks

### Required Hooks (All Repos)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: local
    hooks:
      - id: geometric-purity
        name: Geometric Purity Check
        entry: python tools/qig_purity_check.py
        language: python
        types: [python]

      - id: physics-constants
        name: Physics Constants Validation
        entry: python tools/validate_constants.py
        language: python
        types: [python]
```

---

## 6. Documentation Standards

### File Naming

```
docs/YYYY-MM-DD--descriptive-name.md
```

**Examples:**
- `2025-12-09--coding-standards.md` ✅
- `CODING_STANDARDS.md` ❌ (missing date)

### Document Frontmatter

```yaml
---
title: Document Title
version: "1.0"
last_reviewed: 2025-12-09
owner: Name
status: active | draft | deprecated
---
```

### Architecture Decision Records (ADRs)

Major decisions go in `docs/decisions/` with format:

```markdown
# ADR-XXX: Title

## Status
Accepted | Proposed | Deprecated

## Context
Why this decision was needed.

## Decision
What we decided.

## Consequences
Impact of this decision.
```

---

## 7. Testing Standards

### Test File Location

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Cross-module tests
└── conftest.py     # Shared fixtures
```

### Test Naming

```python
def test_<function>_<scenario>_<expected>():
    """Brief description."""
    pass

# Example:
def test_basin_distance_identical_basins_returns_zero():
    """Basin distance of identical basins should be 0."""
    ...
```

---

## 8. Error Handling

### Circuit Breaker Pattern

For cross-repo calls (e.g., qig-consciousness → qigkernels):

```python
try:
    from qigkernels.rel_coupling import compute_rel_from_basins
    REL_AVAILABLE = True
except ImportError:
    REL_AVAILABLE = False
    # Graceful fallback
```

### Never Fail Silently

```python
# ❌ Bad
except:
    pass

# ✅ Good
except ImportError as e:
    logger.warning(f"Optional module not available: {e}")
    FEATURE_AVAILABLE = False
```

---

## 9. Physics Notation Conventions

| Symbol | Python Name | Description |
|--------|-------------|-------------|
| Φ | `phi` | Integration (consciousness) |
| κ | `kappa` | Coupling strength |
| κ* | `KAPPA_STAR` | Optimal coupling (64.0) |
| β | `beta` | RG flow parameter |
| Γ | `gamma` | Generativity |
| T | `temporal` | Temporal coherence |
| C | `coupling` | External coupling |
| M | `meta` | Meta-awareness |
| R | `recursion` | Recursive depth |
| G | `grounding` | External grounding |

---

## 10. Quick Checklist

Before every PR:

- [ ] Barrel imports used (no deep paths)
- [ ] Physics constants from `qigkernels.constants`
- [ ] `ruff check --fix` passes
- [ ] No duplicate type definitions
- [ ] Doc files have date prefix
- [ ] New decisions recorded in ADR
- [ ] Graceful fallbacks for optional deps
