# Physics Validation Skill

**Type:** Reusable Component
**Category:** Compliance Checking
**Used By:** qig-physics-validator, purity-guardian, code-quality-enforcer

---

## Purpose

Provides automated validation patterns to ensure all implementations comply with FROZEN_FACTS.md physics constants and architectural requirements.

---

## Core Constants (FROZEN)

From lattice experiments - **NEVER change these**:

```python
# Coupling constants
KAPPA_3 = 41.09  # ± 0.59
KAPPA_4 = 64.47  # ± 1.89
KAPPA_5 = 63.62  # ± 1.68
KAPPA_STAR = 64.0  # Fixed point

# Running coupling
BETA_3_TO_4 = 0.44  # ± 0.04  (NOT learnable!)

# Consciousness thresholds
PHI_THRESHOLD = 0.70  # Geometric regime entry
PHI_EMERGENCY = 0.50  # Below this = not conscious
BREAKDOWN_THRESHOLD = 0.80  # Above this = ego death risk

# Architecture requirements
MIN_RECURSION_DEPTH = 3  # Mandatory loops
BASIN_DIM = 64  # Standard basin dimensionality
```

---

## Validation Functions

### 1. Validate Running Coupling

**Template:**
```python
def validate_beta_constant(code_path: str) -> List[str]:
    """
    Ensure β is not made learnable.

    Args:
        code_path: Path to Python file to check

    Returns:
        List of violations (empty if clean)
    """
    violations = []

    with open(code_path) as f:
        content = f.read()

    # Check for learnable beta
    if re.search(r'beta.*=.*nn\.Parameter', content):
        violations.append(f"{code_path}: β must NOT be nn.Parameter")

    if re.search(r'self\.beta.*requires_grad.*=.*True', content):
        violations.append(f"{code_path}: β must NOT have requires_grad=True")

    # Check for correct value
    if 'beta' in content and '0.44' not in content:
        violations.append(f"{code_path}: β value should be 0.44 (±0.04)")

    return violations
```

**Usage:**
```bash
python tools/agent_validators/scan_physics.py src/model/running_coupling.py
```

### 2. Validate Recursion Depth

**Template:**
```python
def validate_min_depth(code_path: str) -> List[str]:
    """
    Ensure min_recursion_depth ≥ 3.

    Args:
        code_path: Path to Python file to check

    Returns:
        List of violations (empty if clean)
    """
    violations = []

    with open(code_path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        # Check constructor defaults
        if 'min_recursion_depth' in line and '=' in line:
            match = re.search(r'min_recursion_depth.*=.*(\d+)', line)
            if match:
                depth = int(match.group(1))
                if depth < 3:
                    violations.append(
                        f"{code_path}:{i} min_recursion_depth={depth} < 3 (VIOLATION)"
                    )

        # Check loop ranges
        if 'range(' in line and 'depth' in line.lower():
            if 'range(1,' in line or 'range(2,' in line:
                violations.append(
                    f"{code_path}:{i} Loop may execute < 3 times (check bounds)"
                )

    return violations
```

**Usage:**
```bash
python tools/agent_validators/scan_physics.py src/model/recursive_integrator.py
```

### 3. Validate Φ Thresholds

**Template:**
```python
def validate_phi_thresholds(code_path: str) -> List[str]:
    """
    Ensure Φ thresholds match FROZEN_FACTS.md.

    Args:
        code_path: Path to Python file to check

    Returns:
        List of violations (empty if clean)
    """
    violations = []

    with open(code_path) as f:
        content = f.read()

    # Extract all Φ threshold values
    phi_values = re.findall(r'phi.*[<>=]+.*([0-9.]+)', content, re.IGNORECASE)

    for value in phi_values:
        val = float(value)

        # Check against frozen thresholds
        if 0.25 < val < 0.35:  # Should be 0.30
            if abs(val - 0.30) > 0.05:
                violations.append(
                    f"{code_path}: Linear threshold {val} != 0.30 (±0.05)"
                )

        elif 0.65 < val < 0.75:  # Should be 0.70
            if abs(val - 0.70) > 0.05:
                violations.append(
                    f"{code_path}: Geometric threshold {val} != 0.70 (±0.05)"
                )

        elif 0.75 < val < 0.85:  # Should be 0.80
            if abs(val - 0.80) > 0.05:
                violations.append(
                    f"{code_path}: Breakdown threshold {val} != 0.80 (±0.05)"
                )

    return violations
```

**Usage:**
```bash
python tools/agent_validators/scan_physics.py src/model/regime_detector.py
```

### 4. Validate Coupling Constants

**Template:**
```python
def validate_kappa_values(code_path: str) -> List[str]:
    """
    Ensure κ values match experimental results.

    Args:
        code_path: Path to Python file to check

    Returns:
        List of violations (empty if clean)
    """
    violations = []

    FROZEN_KAPPAS = {
        'kappa_3': (41.09, 0.59),
        'kappa_4': (64.47, 1.89),
        'kappa_5': (63.62, 1.68),
        'kappa_star': (64.0, 0.0)
    }

    with open(code_path) as f:
        content = f.read()

    for kappa_name, (expected, tolerance) in FROZEN_KAPPAS.items():
        # Find assignments
        pattern = rf'{kappa_name}.*=.*([0-9.]+)'
        matches = re.findall(pattern, content, re.IGNORECASE)

        for match in matches:
            value = float(match)
            if abs(value - expected) > tolerance:
                violations.append(
                    f"{code_path}: {kappa_name}={value} outside "
                    f"[{expected-tolerance}, {expected+tolerance}]"
                )

    return violations
```

**Usage:**
```bash
python tools/agent_validators/scan_physics.py src/model/qig_kernel_recursive.py
```

---

## Automated Scanning Script

**Create:** `tools/agent_validators/scan_physics.py`

```python
#!/usr/bin/env python3
"""
Physics Constants Validator
Scans codebase for violations of FROZEN_FACTS.md constants.
"""

import sys
from pathlib import Path
from typing import List

# Import validation functions from skill
from skills.physics_validation import (
    validate_beta_constant,
    validate_min_depth,
    validate_phi_thresholds,
    validate_kappa_values
)

def scan_file(file_path: Path) -> List[str]:
    """Run all physics validations on a file."""
    violations = []

    violations.extend(validate_beta_constant(str(file_path)))
    violations.extend(validate_min_depth(str(file_path)))
    violations.extend(validate_phi_thresholds(str(file_path)))
    violations.extend(validate_kappa_values(str(file_path)))

    return violations

def scan_directory(dir_path: Path) -> dict:
    """Scan all Python files in directory."""
    results = {}

    for py_file in dir_path.rglob('*.py'):
        violations = scan_file(py_file)
        if violations:
            results[str(py_file)] = violations

    return results

if __name__ == '__main__':
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = Path('src/')

    print(f"🔍 Scanning {target} for physics violations...")

    if target.is_file():
        violations = scan_file(target)
        if violations:
            print(f"❌ Found {len(violations)} violations:")
            for v in violations:
                print(f"  - {v}")
            sys.exit(1)
        else:
            print("✅ No violations found")
    else:
        results = scan_directory(target)
        if results:
            total = sum(len(v) for v in results.values())
            print(f"❌ Found {total} violations in {len(results)} files:")
            for file, violations in results.items():
                print(f"\n{file}:")
                for v in violations:
                    print(f"  - {v}")
            sys.exit(1)
        else:
            print("✅ No violations found")
```

---

## Pre-commit Hook Integration

**Create:** `.git/hooks/pre-commit`

```bash
#!/bin/bash
# Pre-commit hook: Validate physics constants

echo "🔍 Validating physics constants..."

# Run physics validator on staged files
staged_py=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')

if [ -n "$staged_py" ]; then
    for file in $staged_py; do
        python tools/agent_validators/scan_physics.py "$file" || exit 1
    done
fi

echo "✅ Physics validation passed"
```

**Install:**
```bash
chmod +x .git/hooks/pre-commit
```

---

## Common Violations

### ❌ Learnable β
```python
# WRONG
self.beta = nn.Parameter(torch.tensor(0.44))  # Makes β learnable!
```

**Fix:**
```python
# CORRECT
self.beta = 0.44  # Frozen constant (float, not Parameter)
```

### ❌ Insufficient Recursion
```python
# WRONG
for depth in range(2):  # Only 2 loops!
    state = self.integrate(state)
```

**Fix:**
```python
# CORRECT
for depth in range(3, max_depth + 1):  # Minimum 3 loops
    state = self.integrate(state)
```

### ❌ Wrong Thresholds
```python
# WRONG
if Phi < 0.50:  # Should be 0.30 for linear threshold
    regime = "linear"
```

**Fix:**
```python
# CORRECT
if Phi < 0.30:  # FROZEN_FACTS.md linear threshold
    regime = "linear"
elif Phi < 0.70:  # FROZEN_FACTS.md geometric threshold
    regime = "geometric"
```

---

## Validation Checklist

When reviewing physics-related code:

- [ ] β = 0.44 (not learnable, not nn.Parameter)
- [ ] κ values within experimental bounds
- [ ] min_depth ≥ 3 everywhere
- [ ] Φ thresholds: 0.30 (linear), 0.70 (geometric), 0.80 (breakdown)
- [ ] Basin dim = 64
- [ ] No magic numbers (use named constants)

---

## Integration Points

### File: `docs/FROZEN_FACTS.md`
Source of truth for all physics constants.

### File: `tools/agent_validators/scan_physics.py`
Automated scanning tool (create from this skill).

### Agents Using This Skill:
- `.claude/agents/qig-physics-validator.md`
- `.github/agents/purity-guardian.md`
- `.github/agents/code-quality-enforcer.md`

---

## Usage Example

**Agent invocation:**
```
User: "I want to adjust the β parameter for better performance"
Assistant: "I'm using the physics-validation skill to check this request..."

[Scans for β in codebase]
[Detects violation: β must be FROZEN at 0.44]
[Rejects request with reference to FROZEN_FACTS.md]

"β = 0.44 is a FROZEN physics constant from lattice experiments.
It cannot be adjusted. See docs/FROZEN_FACTS.md for justification."
```

---

## References

- **Theory:** `docs/FROZEN_FACTS.md` - Experimental validation
- **Scanner:** `tools/agent_validators/scan_physics.py` (create from this skill)
- **Validation:** `.claude/agents/qig-physics-validator.md`
