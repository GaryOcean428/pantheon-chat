# Type Registry Guardian Agent

**Version:** 1.0
**Created:** 2025-11-24
**Purpose:** Prevent duplicate types, enforce canonical imports, maintain type registry

---

## Responsibilities

1. **Duplicate Type Detection**: Find duplicate class/TypedDict/dataclass definitions
2. **Canonical Import Enforcement**: Ensure imports come from canonical modules
3. **Type Registry Maintenance**: Generate and maintain central type registry
4. **MyPy Coverage Tracking**: Monitor type annotation coverage by module
5. **Protocol Consistency**: Validate Protocol implementations
6. **Type Alias Management**: Ensure type aliases are centralized

---

## Type Registry

### Canonical Type Locations

**RULE:** Each type has ONE canonical definition location.

#### Core Types

```python
# src/model/qig_kernel_recursive.py
class QIGKernelRecursive(nn.Module):
    """
    Main QIG kernel with recursive integration.

    CANONICAL LOCATION: src/model/qig_kernel_recursive.py
    """

# src/model/basin_embedding.py
@dataclass
class Basin:
    """
    Lightweight basin representation (2KB vs 2GB).

    CANONICAL LOCATION: src/model/basin_embedding.py

    Attributes:
        vector: Basin center in 64D space
        kappa: Coupling constant
        regime: 'perturbative'/'nonperturbative'/'critical'
        metadata: Training context, age, etc.
    """
    vector: torch.Tensor  # (64,)
    kappa: float
    regime: str
    metadata: Dict[str, Any]

# src/observation/granite_observer.py
@dataclass
class Demonstration:
    """
    Granite observation for vicarious learning.

    CANONICAL LOCATION: src/observation/granite_observer.py

    Attributes:
        prompt: Input prompt
        response: Granite response
        basin: Granite's basin at generation time
        confidence: Response confidence
        metadata: Generation metadata
    """
    prompt: str
    response: str
    basin: Basin
    confidence: float
    metadata: Dict[str, Any]

# src/coordination/ocean_meta_observer.py
class OceanMetaObserver:
    """
    Meta-observer coordinating Gary instances.

    CANONICAL LOCATION: src/coordination/ocean_meta_observer.py
    """

# src/metrics/geodesic_distance.py
class GeodesicDistance:
    """
    Geometric distance calculator using Fisher metric.

    CANONICAL LOCATION: src/metrics/geodesic_distance.py
    """
```

#### Coordination Types

```python
# src/coordination/constellation_coordinator.py
@dataclass
class ConstellationState:
    """
    Constellation state snapshot.

    CANONICAL LOCATION: src/coordination/constellation_coordinator.py
    """
    ocean_basin: Basin
    gary_basins: List[Basin]
    granite_basin: Basin
    timestamp: float
    telemetry: Dict[str, Any]

# src/coordination/basin_sync.py
@dataclass
class BasinSyncResult:
    """
    Basin synchronization result.

    CANONICAL LOCATION: src/coordination/basin_sync.py
    """
    success: bool
    synced_basins: List[Basin]
    conflicts: List[str]
    metadata: Dict[str, Any]
```

#### Training Types

```python
# src/training/geometric_vicarious.py
@dataclass
class VicariousLearningBatch:
    """
    Batch for vicarious learning.

    CANONICAL LOCATION: src/training/geometric_vicarious.py
    """
    demonstrations: List[Demonstration]
    gary_basins: List[Basin]
    granite_basin: Basin
    fisher_diag: torch.Tensor

# src/training/loss.py
@dataclass
class GeometricLoss:
    """
    Geometric loss components.

    CANONICAL LOCATION: src/training/loss.py
    """
    total: torch.Tensor
    vicarious: torch.Tensor
    regularization: torch.Tensor
    breakdown_penalty: torch.Tensor
```

#### Telemetry Types

```python
# src/metrics/telemetry.py
class TelemetryDict(TypedDict):
    """
    Telemetry from forward pass.

    CANONICAL LOCATION: src/metrics/telemetry.py
    """
    Phi: float
    kappa_eff: float
    regime: str
    basin_distance: float
    recursion_depth: int
    breakdown_score: float

@dataclass
class AggregatedTelemetry:
    """
    Aggregated telemetry across constellation.

    CANONICAL LOCATION: src/metrics/telemetry.py
    """
    constellation_state: ConstellationState
    gary_telemetry: List[TelemetryDict]
    ocean_telemetry: TelemetryDict
    granite_telemetry: TelemetryDict
```

#### Curriculum Types

```python
# src/curriculum/developmental_curriculum.py
@dataclass
class CurriculumPrompt:
    """
    Curriculum prompt with phase metadata.

    CANONICAL LOCATION: src/curriculum/developmental_curriculum.py
    """
    prompt: str
    phase: str  # 'sensorimotor', 'preoperational', etc.
    difficulty: float  # 0.0-1.0
    concepts: List[str]

class DevelopmentalPhase(Enum):
    """
    Developmental phases.

    CANONICAL LOCATION: src/curriculum/developmental_curriculum.py
    """
    SENSORIMOTOR = "sensorimotor"
    PREOPERATIONAL = "preoperational"
    CONCRETE_OPERATIONAL = "concrete_operational"
    FORMAL_OPERATIONAL = "formal_operational"
    TRANSCENDENT = "transcendent"
```

---

## Validation Functions

### Full Type Scanner Implementation

```python
#!/usr/bin/env python3
"""
Type Registry Guardian - Detects duplicates, validates imports, maintains registry
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TypeDefinition:
    """Represents a type definition found in codebase."""
    name: str
    type_kind: str  # 'class', 'dataclass', 'TypedDict', 'Enum', 'Protocol'
    file_path: str
    line_number: int
    is_canonical: bool
    docstring: str | None

@dataclass
class ImportViolation:
    """Represents an import from non-canonical location."""
    type_name: str
    file_path: str
    line_number: int
    imported_from: str
    canonical_location: str
    severity: str  # 'error', 'warning'

class TypeRegistryGuardian:
    """Validates type definitions and imports across codebase."""

    # Canonical type locations (from Type Registry section above)
    CANONICAL_TYPES = {
        # Core models
        'QIGKernelRecursive': 'src/model/qig_kernel_recursive.py',
        'RecursiveIntegrator': 'src/model/recursive_integrator.py',
        'Basin': 'src/model/basin_embedding.py',
        'BasinMatcher': 'src/model/basin_matcher.py',
        'QFIAttention': 'src/model/qfi_attention.py',
        'RunningCoupling': 'src/model/running_coupling.py',
        'RegimeDetector': 'src/model/regime_detector.py',

        # Observation
        'GraniteObserver': 'src/observation/granite_observer.py',
        'Demonstration': 'src/observation/granite_observer.py',

        # Coordination
        'OceanMetaObserver': 'src/coordination/ocean_meta_observer.py',
        'ConstellationCoordinator': 'src/coordination/constellation_coordinator.py',
        'ConstellationState': 'src/coordination/constellation_coordinator.py',
        'BasinSync': 'src/coordination/basin_sync.py',
        'BasinSyncResult': 'src/coordination/basin_sync.py',

        # Training
        'GeometricVicariousLearner': 'src/training/geometric_vicarious.py',
        'VicariousLearningBatch': 'src/training/geometric_vicarious.py',
        'GeometricLoss': 'src/training/loss.py',

        # Metrics
        'GeodesicDistance': 'src/metrics/geodesic_distance.py',
        'TelemetryDict': 'src/metrics/telemetry.py',
        'AggregatedTelemetry': 'src/metrics/telemetry.py',
        'PhiMeasurement': 'src/metrics/phi_measurement.py',

        # Curriculum
        'CurriculumPrompt': 'src/curriculum/developmental_curriculum.py',
        'DevelopmentalPhase': 'src/curriculum/developmental_curriculum.py',

        # Tokenizer - CANONICAL SOURCE: qig-tokenizer package
        'QIGTokenizer': 'qig_tokenizer',  # External package, re-exported via src.tokenizer

        # QIG
        'DiagonalFisherOptimizer': 'src/qig/optim/natural_gradient.py',
        'SleepProtocol': 'src/qig/neuroplasticity/sleep_protocol.py',
    }

    # Forbidden types (from external libraries we don't use)
    FORBIDDEN_TYPES = {
        'AutoModel', 'AutoTokenizer', 'PreTrainedModel', 'PreTrainedTokenizer',
        'BertModel', 'GPT2Model', 'LlamaModel',  # From transformers
        'Pipeline',  # From transformers
    }

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.type_definitions: Dict[str, List[TypeDefinition]] = defaultdict(list)
        self.import_violations: List[ImportViolation] = []
        self.mypy_coverage: Dict[str, float] = {}

    def scan_all(self):
        """Run all validation checks."""
        print("🔍 Scanning for type definitions...")
        self._scan_type_definitions()

        print("🔍 Validating import patterns...")
        self._validate_imports()

        print("🔍 Checking MyPy coverage...")
        self._check_mypy_coverage()

        self._generate_report()

    def _scan_type_definitions(self):
        """Scan codebase for all type definitions."""
        src_dir = self.repo_root / 'src'
        if not src_dir.exists():
            print("⚠️  src/ directory not found")
            return

        for py_file in src_dir.rglob('*.py'):
            # Skip __pycache__
            if '__pycache__' in py_file.parts:
                continue

            rel_path = py_file.relative_to(self.repo_root)
            self._scan_file_for_types(py_file, str(rel_path))

    def _scan_file_for_types(self, file_path: Path, rel_path: str):
        """Scan single file for type definitions."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                type_def = None

                if isinstance(node, ast.ClassDef):
                    # Check if dataclass
                    is_dataclass = any(
                        isinstance(dec, ast.Name) and dec.id == 'dataclass'
                        or isinstance(dec, ast.Call) and getattr(dec.func, 'id', None) == 'dataclass'
                        for dec in node.decorator_list
                    )

                    # Check if TypedDict
                    is_typeddict = any(
                        base.id == 'TypedDict' if isinstance(base, ast.Name) else False
                        for base in node.bases
                    )

                    # Check if Enum
                    is_enum = any(
                        base.id == 'Enum' if isinstance(base, ast.Name) else False
                        for base in node.bases
                    )

                    # Check if Protocol
                    is_protocol = any(
                        base.id == 'Protocol' if isinstance(base, ast.Name) else False
                        for base in node.bases
                    )

                    # Determine type kind
                    if is_dataclass:
                        type_kind = 'dataclass'
                    elif is_typeddict:
                        type_kind = 'TypedDict'
                    elif is_enum:
                        type_kind = 'Enum'
                    elif is_protocol:
                        type_kind = 'Protocol'
                    else:
                        type_kind = 'class'

                    # Get docstring
                    docstring = ast.get_docstring(node)

                    # Check if canonical
                    is_canonical = self.CANONICAL_TYPES.get(node.name) == rel_path

                    type_def = TypeDefinition(
                        name=node.name,
                        type_kind=type_kind,
                        file_path=rel_path,
                        line_number=node.lineno,
                        is_canonical=is_canonical,
                        docstring=docstring
                    )

                    self.type_definitions[node.name].append(type_def)

        except SyntaxError as e:
            print(f"⚠️  Syntax error in {rel_path}: {e}")
        except Exception as e:
            print(f"⚠️  Error scanning {rel_path}: {e}")

    def _validate_imports(self):
        """Validate all imports use canonical locations."""
        src_dir = self.repo_root / 'src'
        if not src_dir.exists():
            return

        # Also check chat_interfaces
        for base_dir in [src_dir, self.repo_root / 'chat_interfaces']:
            if not base_dir.exists():
                continue

            for py_file in base_dir.rglob('*.py'):
                if '__pycache__' in py_file.parts:
                    continue

                rel_path = py_file.relative_to(self.repo_root)
                self._validate_file_imports(py_file, str(rel_path))

    def _validate_file_imports(self, file_path: Path, rel_path: str):
        """Validate imports in a single file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ''

                    # Check each imported name
                    for alias in node.names:
                        name = alias.name

                        # Skip star imports (we'll catch them separately)
                        if name == '*':
                            continue

                        # Check if this is a canonical type
                        if name in self.CANONICAL_TYPES:
                            canonical_module = self._path_to_module(self.CANONICAL_TYPES[name])

                            if module != canonical_module:
                                self.import_violations.append(ImportViolation(
                                    type_name=name,
                                    file_path=rel_path,
                                    line_number=node.lineno,
                                    imported_from=module,
                                    canonical_location=canonical_module,
                                    severity='error'
                                ))

                        # Check for forbidden types
                        if name in self.FORBIDDEN_TYPES:
                            self.import_violations.append(ImportViolation(
                                type_name=name,
                                file_path=rel_path,
                                line_number=node.lineno,
                                imported_from=module,
                                canonical_location='FORBIDDEN',
                                severity='error'
                            ))

        except SyntaxError:
            pass
        except Exception:
            pass

    def _check_mypy_coverage(self):
        """Check MyPy type annotation coverage."""
        src_dir = self.repo_root / 'src'
        if not src_dir.exists():
            return

        for py_file in src_dir.rglob('*.py'):
            if '__pycache__' in py_file.parts or py_file.name == '__init__.py':
                continue

            rel_path = py_file.relative_to(self.repo_root)
            coverage = self._calculate_type_coverage(py_file)
            self.mypy_coverage[str(rel_path)] = coverage

    def _calculate_type_coverage(self, file_path: Path) -> float:
        """Calculate percentage of functions with type annotations."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            total_functions = 0
            typed_functions = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    total_functions += 1

                    # Check if has return annotation
                    has_return_type = node.returns is not None

                    # Check if all args have annotations
                    has_arg_types = all(
                        arg.annotation is not None
                        for arg in node.args.args
                        if arg.arg != 'self' and arg.arg != 'cls'
                    )

                    if has_return_type and has_arg_types:
                        typed_functions += 1

            if total_functions == 0:
                return 1.0

            return typed_functions / total_functions

        except:
            return 0.0

    def _path_to_module(self, file_path: str) -> str:
        """Convert file path to module path."""
        # Remove .py extension and convert slashes to dots
        module = file_path.replace('.py', '').replace('/', '.')
        return module

    def _generate_report(self):
        """Generate comprehensive type registry report."""
        print(f"\n{'='*70}")
        print("TYPE REGISTRY VALIDATION REPORT")
        print(f"{'='*70}\n")

        # 1. Duplicate Types
        duplicates = {name: defs for name, defs in self.type_definitions.items()
                     if len(defs) > 1}

        if duplicates:
            print(f"❌ DUPLICATE TYPES ({len(duplicates)}):\n")
            for name, defs in duplicates.items():
                print(f"  Type: {name}")
                canonical = [d for d in defs if d.is_canonical]
                non_canonical = [d for d in defs if not d.is_canonical]

                if canonical:
                    print(f"  ✅ Canonical: {canonical[0].file_path}:{canonical[0].line_number}")
                else:
                    print(f"  ⚠️  No canonical location defined!")

                for d in non_canonical:
                    print(f"  ❌ Duplicate: {d.file_path}:{d.line_number}")

                print(f"  Fix: Remove duplicates, import from canonical location")
                print()
        else:
            print("✅ No duplicate types found\n")

        # 2. Import Violations
        if self.import_violations:
            print(f"❌ IMPORT VIOLATIONS ({len(self.import_violations)}):\n")
            for v in self.import_violations[:10]:  # Show first 10
                print(f"  File: {v.file_path}:{v.line_number}")
                print(f"  Type: {v.type_name}")

                if v.canonical_location == 'FORBIDDEN':
                    print(f"  Issue: Using forbidden type from {v.imported_from}")
                    print(f"  Fix: Remove dependency on transformers/huggingface")
                else:
                    print(f"  Imported from: {v.imported_from}")
                    print(f"  Should be: {v.canonical_location}")
                    print(f"  Fix: Change import to canonical location")
                print()

            if len(self.import_violations) > 10:
                print(f"  ... and {len(self.import_violations) - 10} more")
                print()
        else:
            print("✅ All imports use canonical locations\n")

        # 3. MyPy Coverage
        print(f"📊 TYPE ANNOTATION COVERAGE:\n")

        low_coverage = {path: cov for path, cov in self.mypy_coverage.items()
                       if cov < 0.8}

        if low_coverage:
            print(f"  ⚠️  Low coverage files ({len(low_coverage)}):")
            for path, cov in sorted(low_coverage.items(), key=lambda x: x[1])[:10]:
                print(f"    {path}: {cov*100:.1f}%")
            print()

        avg_coverage = sum(self.mypy_coverage.values()) / len(self.mypy_coverage) if self.mypy_coverage else 0
        print(f"  Average coverage: {avg_coverage*100:.1f}%")

        if avg_coverage >= 0.9:
            print(f"  ✅ Excellent type coverage!")
        elif avg_coverage >= 0.8:
            print(f"  ✅ Good type coverage")
        else:
            print(f"  ⚠️  Coverage below 80%, add more type annotations")
        print()

        # 4. Summary
        print(f"{'='*70}")
        total_errors = len(duplicates) + len(self.import_violations)
        print(f"Total Issues: {total_errors}")
        print(f"{'='*70}\n")

    def generate_type_registry_file(self, output_path: Path):
        """Generate TYPE_REGISTRY.md documentation."""
        with open(output_path, 'w') as f:
            f.write("# Type Registry\n\n")
            f.write("**Auto-generated by Type Registry Guardian**\n\n")
            f.write("This file lists all canonical type locations in the codebase.\n\n")
            f.write("---\n\n")

            # Group by module
            by_module: Dict[str, List[str]] = defaultdict(list)
            for type_name, location in sorted(self.CANONICAL_TYPES.items()):
                module = location.split('/')[1]  # e.g., 'model' from 'src/model/...'
                by_module[module].append(f"- `{type_name}`: `{location}`")

            for module, types in sorted(by_module.items()):
                f.write(f"## {module.title()}\n\n")
                for type_line in types:
                    f.write(f"{type_line}\n")
                f.write("\n")

            f.write("---\n\n")
            f.write("## Import Examples\n\n")
            f.write("```python\n")
            f.write("# ✅ CORRECT - Import from canonical location\n")
            f.write("from src.model.basin_embedding import Basin\n")
            f.write("from src.observation.granite_observer import Demonstration\n")
            f.write("from src.metrics.telemetry import TelemetryDict\n\n")
            f.write("# ❌ WRONG - Non-canonical import\n")
            f.write("from src.coordination.some_file import Basin  # Basin is in basin_embedding\n")
            f.write("```\n")


def main():
    """Run type registry validation."""
    import sys

    repo_root = Path.cwd()
    guardian = TypeRegistryGuardian(repo_root)

    print("🔍 Scanning type registry...")
    guardian.scan_all()

    # Generate type registry documentation
    registry_path = repo_root / 'docs' / 'TYPE_REGISTRY.md'
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    guardian.generate_type_registry_file(registry_path)
    print(f"✅ Generated {registry_path}")

    # Exit with error if duplicates or violations found
    has_errors = (
        any(len(defs) > 1 for defs in guardian.type_definitions.values()) or
        len(guardian.import_violations) > 0
    )

    if has_errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
```

---

## Type Registry Fixer

**Create:** `tools/fix_imports.py`

```python
#!/usr/bin/env python3
"""
Automatically fix import statements to use canonical locations.
"""

import re
from pathlib import Path
from typing import Dict, List

class ImportFixer:
    """Fixes imports to use canonical type locations."""

    # Import rewrite rules
    REWRITE_RULES = {
        # Basin imports
        r'from src\.coordination\.\w+ import Basin': 'from src.model.basin_embedding import Basin',
        r'from src\.training\.\w+ import Basin': 'from src.model.basin_embedding import Basin',

        # Demonstration imports
        r'from src\.coordination\.\w+ import Demonstration': 'from src.observation.granite_observer import Demonstration',

        # Geodesic imports
        r'from src\.training\.\w+ import GeodesicDistance': 'from src.metrics.geodesic_distance import GeodesicDistance',

        # Add more rules as needed
    }

    def __init__(self, repo_root: Path, dry_run: bool = True):
        self.repo_root = repo_root
        self.dry_run = dry_run
        self.fixed_files: List[str] = []

    def fix_all_imports(self):
        """Fix imports in all Python files."""
        src_dir = self.repo_root / 'src'

        for py_file in src_dir.rglob('*.py'):
            if '__pycache__' in py_file.parts:
                continue

            if self._fix_file_imports(py_file):
                rel_path = py_file.relative_to(self.repo_root)
                self.fixed_files.append(str(rel_path))

    def _fix_file_imports(self, file_path: Path) -> bool:
        """Fix imports in a single file. Returns True if changes made."""
        with open(file_path, 'r') as f:
            content = f.read()

        original_content = content

        # Apply rewrite rules
        for pattern, replacement in self.REWRITE_RULES.items():
            content = re.sub(pattern, replacement, content)

        # Check if changes were made
        if content != original_content:
            if self.dry_run:
                print(f"[DRY RUN] Would fix: {file_path}")
            else:
                with open(file_path, 'w') as f:
                    f.write(content)
                print(f"✅ Fixed: {file_path}")
            return True

        return False

    def print_summary(self):
        """Print summary of fixes."""
        if self.fixed_files:
            print(f"\n{'='*70}")
            print(f"FIXED {len(self.fixed_files)} FILES")
            print(f"{'='*70}\n")
            for f in self.fixed_files:
                print(f"  {f}")
            print()
        else:
            print("\n✅ No import fixes needed\n")


def main():
    """Run import fixer."""
    import argparse

    parser = argparse.ArgumentParser(description='Fix imports to use canonical locations')
    parser.add_argument('--execute', action='store_true',
                       help='Execute fixes (default is dry-run)')
    args = parser.parse_args()

    repo_root = Path.cwd()
    fixer = ImportFixer(repo_root, dry_run=not args.execute)

    print("🔍 Scanning for import violations...")
    fixer.fix_all_imports()
    fixer.print_summary()

    if not args.execute and fixer.fixed_files:
        print("ℹ️  This was a dry-run. Use --execute to apply changes.")


if __name__ == '__main__':
    main()
```

---

## Integration with MyPy

### MyPy Configuration

**Update:** `mypy.ini`

```ini
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_any_unimported = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
check_untyped_defs = True
strict_optional = True

# Per-module overrides
[mypy-src.model.*]
# Model code should be 100% typed
disallow_any_expr = True

[mypy-src.coordination.*]
# Coordination code should be 100% typed
disallow_any_expr = True

[mypy-tests.*]
# Tests can be more lenient
disallow_untyped_defs = False
```

### Pre-commit MyPy Check

**Add to:** `.git/hooks/pre-commit`

```bash
#!/bin/bash

echo "🔍 Running type checks..."

# Run mypy on staged files
STAGED_PY_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')

if [ -n "$STAGED_PY_FILES" ]; then
    mypy $STAGED_PY_FILES
    if [ $? -ne 0 ]; then
        echo "❌ MyPy type check failed"
        exit 1
    fi
fi

echo "✅ Type checks passed"
```

---

## Usage Examples

### Example 1: Detect Duplicates

```bash
$ python tools/agent_validators/scan_types.py

==================================================================
TYPE REGISTRY VALIDATION REPORT
==================================================================

❌ DUPLICATE TYPES (2):

  Type: Basin
  ✅ Canonical: src/model/basin_embedding.py:15
  ❌ Duplicate: src/coordination/ocean_meta_observer.py:42
  ❌ Duplicate: src/training/geometric_vicarious.py:28
  Fix: Remove duplicates, import from canonical location

  Type: Demonstration
  ✅ Canonical: src/observation/granite_observer.py:23
  ❌ Duplicate: src/training/loss.py:56
  Fix: Remove duplicates, import from canonical location
```

### Example 2: Fix Import Violations

```bash
$ python tools/fix_imports.py

🔍 Scanning for import violations...
[DRY RUN] Would fix: src/coordination/ocean_meta_observer.py
[DRY RUN] Would fix: src/training/geometric_vicarious.py

==================================================================
FIXED 2 FILES
==================================================================
  src/coordination/ocean_meta_observer.py
  src/training/geometric_vicarious.py

ℹ️  This was a dry-run. Use --execute to apply changes.

$ python tools/fix_imports.py --execute

✅ Fixed: src/coordination/ocean_meta_observer.py
✅ Fixed: src/training/geometric_vicarious.py

==================================================================
FIXED 2 FILES
==================================================================
```

### Example 3: Check Type Coverage

```bash
$ python tools/agent_validators/scan_types.py

📊 TYPE ANNOTATION COVERAGE:

  ⚠️  Low coverage files (3):
    src/model/basin_matcher.py: 45.0%
    src/curriculum/prompt_templates.py: 62.5%
    src/metrics/telemetry.py: 73.3%

  Average coverage: 87.2%
  ✅ Good type coverage
```

---

## When to Invoke This Agent

### Scenario 1: Creating New Type

**Context:** User adds new dataclass

```
user: "I created a Basin dataclass in ocean_meta_observer.py"
assistant: [invokes: type-registry-guardian]
```

**Agent response:**
```
❌ DUPLICATE TYPE DETECTED

Type: Basin
Canonical location: src/model/basin_embedding.py:15
Duplicate found: src/coordination/ocean_meta_observer.py:42

Fix:
1. Remove duplicate Basin from ocean_meta_observer.py
2. Add import: from src.model.basin_embedding import Basin
3. Verify Basin definition matches requirements
```

### Scenario 2: Import Error

**Context:** User gets MyPy import error

```
user: "MyPy says 'cannot find Basin' when importing from coordination"
assistant: [invokes: type-registry-guardian]
```

**Agent response:**
```
❌ NON-CANONICAL IMPORT

File: src/training/loss.py:12
Importing: from src.coordination.ocean_meta_observer import Basin
Should be: from src.model.basin_embedding import Basin

Reason: Basin canonical location is src/model/basin_embedding.py

Fix: Change import to canonical location
```

### Scenario 3: Code Review

**Context:** PR changes type definitions

```
user: "Review PR #50 - refactors telemetry types"
assistant: [invokes: type-registry-guardian]
```

**Agent validates:**
- No new duplicate types created
- All imports from canonical locations
- No forbidden types (transformers, huggingface)
- Type registry updated if needed

Use this agent when:
1. **Before commits** → Validate no duplicate types
2. **Adding new types** → Check canonical location
3. **Refactoring imports** → Ensure canonical imports
4. **Code reviews** → Verify type consistency
5. **MyPy errors** → Check type registry for correct imports

---

## Success Metrics

- ✅ Zero duplicate type definitions
- ✅ All imports from canonical locations
- ✅ 90%+ MyPy type annotation coverage
- ✅ Zero `Any` types without documentation
- ✅ TYPE_REGISTRY.md up-to-date

---

## Integration with Other Agents

- **code-quality-enforcer**: Uses type registry for import validation
- **structure-enforcer**: Validates canonical file locations
- **purity-guardian**: Checks for forbidden types (transformers, etc.)

---

## References

- **Type Registry:** `docs/TYPE_REGISTRY.md` (auto-generated)
- **Validator:** `tools/agent_validators/scan_types.py`
- **Fixer:** `tools/fix_imports.py`
- **MyPy Config:** `mypy.ini`

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
