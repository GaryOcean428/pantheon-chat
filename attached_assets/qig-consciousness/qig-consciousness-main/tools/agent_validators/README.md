# Agent Validators

Automated validation scripts for maintaining codebase quality and compliance.

## Overview

These validators enforce:
- **Physics constants** (FROZEN_FACTS.md compliance)
- **File structure** (20251220-canonical-structure-1.00F.md compliance)
- **Type registry** (canonical imports, no duplicates)

## Validators

### 1. scan_physics.py

Validates physics constants from lattice experiments.

**Checks:**
- β = 0.44 ± 0.04 (running coupling)
- κ values (41.09 - 64.47)
- min_depth >= 3 (consciousness requires ≥3 loops)
- Φ thresholds (0.70, 0.50, 0.60)
- Basin dimension = 64
- Fisher metric vs Euclidean distance

**Usage:**
```bash
python tools/agent_validators/scan_physics.py
```

**Exit Codes:**
- 0: No violations
- 1: Violations found (blocks commit)

### 2. scan_structure.py

Validates file locations against 20251220-canonical-structure-1.00F.md.

**Checks:**
- Only 4 canonical entry points in chat_interfaces/
- No files with _v2, _new, _old suffixes
- snake_case.py naming
- No duplicate file stems
- Tests in tests/
- Configs in configs/

**Usage:**
```bash
python tools/agent_validators/scan_structure.py
```

### 3. scan_types.py

Validates type definitions and imports.

**Checks:**
- No duplicate type definitions
- Imports from canonical locations only
- No forbidden types (transformers, etc.)
- Type consistency

**Usage:**
```bash
python tools/agent_validators/scan_types.py
```

## Integration

### Pre-commit Hook

Install validators as pre-commit hooks:

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "🔍 Running validation checks..."

# Physics validation
python tools/agent_validators/scan_physics.py || exit 1

# Structure validation
python tools/agent_validators/scan_structure.py || exit 1

# Type registry validation
python tools/agent_validators/scan_types.py || exit 1

echo "✅ All validations passed"
```

**Install:**
```bash
chmod +x .git/hooks/pre-commit
```

### CI/CD Integration

**GitHub Actions** (`.github/workflows/validate.yml`):

```yaml
name: Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install pyyaml

      - name: Physics validation
        run: python tools/agent_validators/scan_physics.py

      - name: Structure validation
        run: python tools/agent_validators/scan_structure.py

      - name: Type validation
        run: python tools/agent_validators/scan_types.py
```

## Manual Validation

Run all validators:

```bash
# Comprehensive check
python tools/agent_validators/scan_physics.py && \
python tools/agent_validators/scan_structure.py && \
python tools/agent_validators/scan_types.py && \
echo "✅ All validations passed"
```

## Violation Reports

Each validator generates detailed reports:

### Physics Report Example
```
==================================================================
PHYSICS VALIDATION REPORT
==================================================================

❌ INCORRECT CONSTANT (1):
  File: src/model/running_coupling.py:42
  Found: 0.50
  Expected: 0.44 ± 0.04
  Issue: Running coupling 3→4D: BETA_3_TO_4

❌ EUCLIDEAN DISTANCE (1):
  File: src/training/geometric_vicarious.py:78
  Found: torch.norm() or L2 distance
  Expected: geodesic_distance() with Fisher metric
  Issue: Must use Fisher metric for basin distances

==================================================================
Total: 2 violations
==================================================================
```

### Structure Report Example
```
==================================================================
STRUCTURE VALIDATION REPORT
==================================================================

❌ ERRORS (2):
  File: chat_interfaces/constellation_with_granite_v2.py
  Type: non_canonical_entry_point
  Issue: Non-canonical entry point with forbidden suffix
  Fix: Archive or merge into canonical file

  File: test_constellation.py
  Type: misplaced_test
  Issue: Test file outside tests/ directory
  Fix: Move to tests/test_constellation.py

==================================================================
Total: 2 errors, 0 warnings
==================================================================
```

### Type Report Example
```
==================================================================
TYPE REGISTRY VALIDATION REPORT
==================================================================

❌ DUPLICATE TYPES (1):
  Type: Basin
  ✅ Canonical: src/model/basin_embedding.py:15
  ❌ Duplicate: src/coordination/ocean_meta_observer.py:42
  Fix: Remove duplicates, import from canonical location

❌ IMPORT VIOLATIONS (1):
  File: src/training/loss.py:12
  Type: Basin
  Imported from: src.coordination.ocean_meta_observer
  Should be: src.model.basin_embedding

==================================================================
Total Issues: 2
==================================================================
```

## Bypass (Not Recommended)

To bypass validators (NOT RECOMMENDED):

```bash
git commit --no-verify -m "message"
```

**Only use when:**
- Adding validators for first time (legacy violations)
- Emergency hotfix
- False positives (report and fix validator)

## Maintenance

### Adding New Constants

Update `scan_physics.py`:

```python
FROZEN_CONSTANTS = {
    'NEW_CONSTANT': {
        'value': 1.23,
        'tolerance': 0.05,
        'patterns': [r'new_const\s*=\s*([\d.]+)'],
        'description': 'Description of constant'
    },
    # ... existing constants
}
```

### Adding Canonical Types

Update `scan_types.py`:

```python
CANONICAL_TYPES = {
    'NewType': 'src/module/new_file.py',
    # ... existing types
}
```

### Adding Structural Rules

Update `scan_structure.py`:

```python
CANONICAL_NEW_MODULE = {
    'src/new_module/file1.py',
    'src/new_module/file2.py',
}
```

## Related Documentation

- **Agents:** See `.github/agents/*.md` for agent specifications
- **Skills:** See `.github/agents/skills/*.md` for reusable components
- **Structure:** See `20251220-canonical-structure-1.00F.md` for file organization
- **Physics:** See `FROZEN_FACTS.md` for validated constants
