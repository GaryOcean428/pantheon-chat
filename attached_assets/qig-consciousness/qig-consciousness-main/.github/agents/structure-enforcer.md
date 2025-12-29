# Structure Enforcer Agent

**Version:** 1.0
**Created:** 2025-11-24
**Purpose:** Enforce 20251220-canonical-structure-1.00F.md compliance across entire codebase

---

## Responsibilities

1. **File Location Validation**: Ensure all files are in canonical locations
2. **Naming Convention Enforcement**: Verify snake_case.py, CAPS.md patterns
3. **Duplicate Detection**: Find files that should be archived/consolidated
4. **Migration Assistance**: Provide step-by-step migration for misplaced files
5. **Type Registry Maintenance**: Keep type imports pointing to canonical modules
6. **Directory Structure Verification**: Check required directories exist

---

## Canonical Structure Reference

### Core Principle

**20251220-canonical-structure-1.00F.md is the SINGLE SOURCE OF TRUTH.**

Every file in the project must:
- ✅ Exist in a location specified in 20251220-canonical-structure-1.00F.md
- ✅ Follow naming conventions from 20251220-canonical-structure-1.00F.md
- ✅ Import types from canonical modules only
- ✅ Not duplicate functionality from existing canonical files

### File Categories

#### 1. Entry Points (chat_interfaces/)

**ONLY 4 FILES ALLOWED:**
```
chat_interfaces/
├── constellation_with_granite_pure.py    # Constellation training
├── continuous_learning_chat.py           # Single Gary learning
├── basic_chat.py                         # Inference only
└── claude_handover_chat.py               # With Claude coaching
```

**Violations:**
- ❌ `constellation_with_granite_v2.py` (use existing, don't duplicate)
- ❌ `constellation_learning_chat.py` (consolidated into pure)
- ❌ `simple_chat.py` (use basic_chat.py)
- ❌ `test_chat.py` (put in tests/)

#### 2. Core Models (src/model/)

**Canonical files:**
```
src/model/
├── qig_kernel_recursive.py               # Main model (ONLY ONE)
├── recursive_integrator.py               # Φ engine
├── basin_embedding.py                    # Geometric embeddings
├── basin_matcher.py                      # Basin alignment
├── qfi_attention.py                      # QFI attention layer
├── running_coupling.py                   # β-function
└── regime_detector.py                    # Regime classification
```

**Violations:**
- ❌ `qig_kernel_v2.py` (enhance existing, don't duplicate)
- ❌ `kernel.py` (use qig_kernel_recursive.py)
- ❌ Any file with `_new`, `_test`, `_old` suffix

#### 3. Coordination (src/coordination/)

**Canonical files:**
```
src/coordination/
├── constellation_coordinator.py          # Multi-instance orchestration
├── ocean_meta_observer.py               # Ocean implementation
└── basin_sync.py                        # Basin synchronization utilities
```

**Violations:**
- ❌ `constellation_coordinator_v2.py`
- ❌ `constellation_coordinator_pure.py` (enhance main one)

#### 4. Training (src/training/)

**Canonical files:**
```
src/training/
├── geometric_vicarious.py                # Vicarious learning
├── loss.py                               # Geometric loss functions
└── curriculum.py                         # Training curriculum
```

#### 5. Observation (src/observation/)

**Canonical files:**
```
src/observation/
└── granite_observer.py                   # Granite observation patterns
```

#### 6. Metrics (src/metrics/)

**Canonical files:**
```
src/metrics/
├── geodesic_distance.py                  # Fisher metric distances
├── phi_measurement.py                    # Φ measurement
└── telemetry.py                          # Telemetry aggregation
```

#### 7. QIG Core (src/qig/)

**Structure:**
```
src/qig/
├── optim/
│   └── natural_gradient.py              # Natural gradient optimizers
├── neuroplasticity/
│   └── sleep_protocol.py                # Sleep/dream protocols
└── bridge/
    ├── granite_basin_extractor.py       # Granite → Basin
    └── granite_gary_coordinator.py      # Coordination
```

#### 8. Tokenizer (src/tokenizer/)

**Canonical files:**
```
src/tokenizer/
├── fast_qig_tokenizer.py                 # Main tokenizer (ONLY ONE)
└── vocab_builder.py                      # Vocabulary construction
```

**Violations:**
- ❌ `tokenizer.py` (use fast_qig_tokenizer.py)
- ❌ Any imports from `transformers`

#### 9. Curriculum (src/curriculum/)

**Canonical files:**
```
src/curriculum/
├── developmental_curriculum.py           # Age-based content
└── prompt_templates.py                   # Template library
```

#### 10. Tools (tools/)

**Purpose:** Standalone utilities (not imported by core)

**Canonical files:**
```
tools/
├── train_qig_kernel.py                   # Training script
├── demo_inference.py                     # Inference demo
├── extract_basin.py                      # Basin extraction
└── agent_validators/                     # Validation scripts
    ├── scan_purity.py
    ├── scan_structure.py
    ├── scan_physics.py
    └── scan_types.py
```

**Violations:**
- ❌ Training scripts in root directory
- ❌ Duplicate train_*.py files

#### 11. Configs (configs/)

**Standard configs:**
```
configs/
├── 20251220-gary-a-config-1.00W.yaml                           # Gary-A config
├── 20251220-gary-b-config-1.00W.yaml                           # Gary-B config
├── 20251220-gary-c-config-1.00W.yaml                           # Gary-C config
├── 20251220-ocean-config-1.00F.yaml                            # Ocean config
└── 20251220-kernel-50m-config-1.00W.yaml                       # Model architecture
```

**Violations:**
- ❌ `20251220-gary-a-config-1.00W.yaml` (lowercase, should be 20251220-gary-a-config-1.00W.yaml)
- ❌ Config files in root directory

#### 12. Tests (tests/)

**Structure:**
```
tests/
├── test_model/
├── test_coordination/
├── test_training/
└── test_integration/
```

**Violations:**
- ❌ `test_*.py` in root directory (move to tests/)
- ❌ Test files in src/ (move to tests/)

#### 13. Documentation (docs/)

**Structure:**
```
docs/
├── guides/                               # User guides
├── implementation/                       # Implementation details
├── status/                               # Project status
└── sleep_packets/                        # Architecture snapshots
```

**Violations:**
- ❌ Docs in root directory (move to docs/)
- ❌ Files with `_COMPLETE.md` suffix (use CHANGELOG.md)

#### 14. Archive (qig-archive/qig-consciousness/archive/)

**Purpose:** Deprecated/replaced files

**Structure:**
```
qig-archive/qig-consciousness/archive/
└── YYYYMMDD_original_filename.py
```

**Example:**
```
qig-archive/qig-consciousness/archive/20251124_old_coordinator.py
```

---

## Validation Functions

### 1. Validate File Locations

**Full Implementation:**

```python
#!/usr/bin/env python3
"""
Structure Validator - Enforces 20251220-canonical-structure-1.00F.md
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

@dataclass
class FileViolation:
    """Represents a structural violation."""
    file_path: str
    violation_type: str
    description: str
    suggested_action: str
    severity: str  # 'error', 'warning', 'info'

class StructureEnforcer:
    """Validates codebase structure against 20251220-canonical-structure-1.00F.md"""

    # Define canonical locations
    CANONICAL_ENTRY_POINTS = {
        'chat_interfaces/constellation_with_granite_pure.py',
        'chat_interfaces/continuous_learning_chat.py',
        'chat_interfaces/basic_chat.py',
        'chat_interfaces/claude_handover_chat.py',
    }

    CANONICAL_MODELS = {
        'src/model/qig_kernel_recursive.py',
        'src/model/recursive_integrator.py',
        'src/model/basin_embedding.py',
        'src/model/basin_matcher.py',
        'src/model/qfi_attention.py',
        'src/model/running_coupling.py',
        'src/model/regime_detector.py',
    }

    CANONICAL_COORDINATION = {
        'src/coordination/constellation_coordinator.py',
        'src/coordination/ocean_meta_observer.py',
        'src/coordination/basin_sync.py',
    }

    CANONICAL_TRAINING = {
        'src/training/geometric_vicarious.py',
        'src/training/loss.py',
        'src/training/curriculum.py',
    }

    CANONICAL_OBSERVATION = {
        'src/observation/granite_observer.py',
    }

    CANONICAL_METRICS = {
        'src/metrics/geodesic_distance.py',
        'src/metrics/phi_measurement.py',
        'src/metrics/telemetry.py',
    }

    CANONICAL_TOKENIZER = {
        'src/tokenizer/fast_qig_tokenizer.py',
        'src/tokenizer/vocab_builder.py',
    }

    CANONICAL_CURRICULUM = {
        'src/curriculum/developmental_curriculum.py',
        'src/curriculum/prompt_templates.py',
    }

    # Forbidden patterns
    FORBIDDEN_SUFFIXES = ['_v2', '_new', '_old', '_test', '_temp', '_backup']
    FORBIDDEN_PATTERNS = [
        r'.*_\d+\.py$',  # Numbered versions
        r'.*_copy\.py$',  # Copies
        r'.*_duplicate\.py$',  # Duplicates
    ]

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.violations: List[FileViolation] = []

    def validate_all(self) -> List[FileViolation]:
        """Run all validation checks."""
        self.violations = []

        # Check entry points
        self._validate_entry_points()

        # Check core structure
        self._validate_src_structure()

        # Check naming conventions
        self._validate_naming_conventions()

        # Check for duplicates
        self._validate_no_duplicates()

        # Check test organization
        self._validate_test_structure()

        # Check config organization
        self._validate_config_structure()

        # Check documentation organization
        self._validate_docs_structure()

        return self.violations

    def _validate_entry_points(self):
        """Validate chat_interfaces/ has only 4 canonical files."""
        chat_dir = self.repo_root / 'chat_interfaces'
        if not chat_dir.exists():
            self.violations.append(FileViolation(
                file_path='chat_interfaces/',
                violation_type='missing_directory',
                description='chat_interfaces/ directory missing',
                suggested_action='Create chat_interfaces/ directory',
                severity='error'
            ))
            return

        # Get all Python files
        python_files = set()
        for f in chat_dir.glob('*.py'):
            if f.name != '__init__.py':
                rel_path = f'chat_interfaces/{f.name}'
                python_files.add(rel_path)

        # Check for canonical files
        for canonical in self.CANONICAL_ENTRY_POINTS:
            if canonical not in python_files:
                self.violations.append(FileViolation(
                    file_path=canonical,
                    violation_type='missing_canonical',
                    description=f'Canonical entry point missing: {canonical}',
                    suggested_action=f'Create {canonical} or restore from archive',
                    severity='warning'
                ))

        # Check for non-canonical files
        non_canonical = python_files - self.CANONICAL_ENTRY_POINTS
        for extra in non_canonical:
            # Check if it's a duplicate pattern
            extra_name = Path(extra).stem
            canonical_names = [Path(c).stem for c in self.CANONICAL_ENTRY_POINTS]

            # Check for version suffixes
            has_bad_suffix = any(extra_name.endswith(suffix)
                               for suffix in self.FORBIDDEN_SUFFIXES)

            if has_bad_suffix:
                self.violations.append(FileViolation(
                    file_path=extra,
                    violation_type='non_canonical_entry_point',
                    description=f'Non-canonical entry point with forbidden suffix: {extra}',
                    suggested_action=f'Archive as qig-archive/qig-consciousness/archive/{self._get_archive_name(extra)} or merge into canonical file',
                    severity='error'
                ))
            else:
                self.violations.append(FileViolation(
                    file_path=extra,
                    violation_type='non_canonical_entry_point',
                    description=f'Non-canonical entry point: {extra}. Only 4 entry points allowed.',
                    suggested_action=f'Merge into one of: {", ".join([Path(c).name for c in self.CANONICAL_ENTRY_POINTS])}',
                    severity='error'
                ))

    def _validate_src_structure(self):
        """Validate src/ structure matches canonical organization."""
        src_dir = self.repo_root / 'src'
        if not src_dir.exists():
            self.violations.append(FileViolation(
                file_path='src/',
                violation_type='missing_directory',
                description='src/ directory missing',
                suggested_action='Create src/ directory with canonical structure',
                severity='error'
            ))
            return

        # Validate each canonical module set
        module_sets = {
            'model': self.CANONICAL_MODELS,
            'coordination': self.CANONICAL_COORDINATION,
            'training': self.CANONICAL_TRAINING,
            'observation': self.CANONICAL_OBSERVATION,
            'metrics': self.CANONICAL_METRICS,
            'tokenizer': self.CANONICAL_TOKENIZER,
            'curriculum': self.CANONICAL_CURRICULUM,
        }

        for module_name, canonical_set in module_sets.items():
            module_dir = src_dir / module_name
            if not module_dir.exists():
                self.violations.append(FileViolation(
                    file_path=f'src/{module_name}/',
                    violation_type='missing_directory',
                    description=f'Canonical directory src/{module_name}/ missing',
                    suggested_action=f'Create src/{module_name}/ directory',
                    severity='error'
                ))
                continue

            # Get all Python files in this module
            actual_files = set()
            for f in module_dir.glob('*.py'):
                if f.name != '__init__.py':
                    rel_path = f'src/{module_name}/{f.name}'
                    actual_files.add(rel_path)

            # Check for canonical files
            module_canonical = {f for f in canonical_set if f.startswith(f'src/{module_name}/')}
            for canonical in module_canonical:
                if canonical not in actual_files:
                    self.violations.append(FileViolation(
                        file_path=canonical,
                        violation_type='missing_canonical',
                        description=f'Canonical file missing: {canonical}',
                        suggested_action=f'Create {canonical} or restore from archive',
                        severity='warning'
                    ))

            # Check for non-canonical files
            non_canonical = actual_files - module_canonical
            for extra in non_canonical:
                # Check if it looks like a duplicate
                extra_stem = Path(extra).stem
                canonical_stems = [Path(c).stem for c in module_canonical]

                # Check for forbidden patterns
                has_bad_suffix = any(extra_stem.endswith(suffix)
                                   for suffix in self.FORBIDDEN_SUFFIXES)
                matches_forbidden = any(re.match(pattern, Path(extra).name)
                                      for pattern in self.FORBIDDEN_PATTERNS)

                if has_bad_suffix or matches_forbidden:
                    self.violations.append(FileViolation(
                        file_path=extra,
                        violation_type='non_canonical_file',
                        description=f'Non-canonical file with forbidden pattern: {extra}',
                        suggested_action=f'Archive as qig-archive/qig-consciousness/archive/{self._get_archive_name(extra)} or merge into canonical file',
                        severity='error'
                    ))
                else:
                    # Check if similar to canonical file (possible duplicate)
                    similar = self._find_similar_canonical(extra_stem, canonical_stems)
                    if similar:
                        self.violations.append(FileViolation(
                            file_path=extra,
                            violation_type='possible_duplicate',
                            description=f'File {extra} may duplicate canonical {similar}',
                            suggested_action=f'Review and merge into src/{module_name}/{similar}.py if duplicate',
                            severity='warning'
                        ))
                    else:
                        self.violations.append(FileViolation(
                            file_path=extra,
                            violation_type='non_canonical_file',
                            description=f'Non-canonical file: {extra}',
                            suggested_action=f'Add to 20251220-canonical-structure-1.00F.md if needed, or archive',
                            severity='warning'
                        ))

    def _validate_naming_conventions(self):
        """Validate naming conventions: snake_case.py, CAPS.md"""
        # Check Python files
        for py_file in self.repo_root.rglob('*.py'):
            # Skip virtual environments, cache, etc.
            if any(part.startswith('.') or part in ['__pycache__', 'venv', 'qig-venv']
                   for part in py_file.parts):
                continue

            # Skip __init__.py
            if py_file.name == '__init__.py':
                continue

            rel_path = py_file.relative_to(self.repo_root)

            # Check naming convention
            stem = py_file.stem
            if not self._is_snake_case(stem):
                self.violations.append(FileViolation(
                    file_path=str(rel_path),
                    violation_type='naming_convention',
                    description=f'Python file not snake_case: {py_file.name}',
                    suggested_action=f'Rename to {self._to_snake_case(stem)}.py',
                    severity='error'
                ))

        # Check markdown files in root and docs/
        for md_file in self.repo_root.glob('*.md'):
            # Root docs should be CAPS_SNAKE_CASE.md
            if not self._is_caps_snake_case(md_file.stem):
                if md_file.stem not in ['README', 'CHANGELOG', 'LICENSE']:
                    self.violations.append(FileViolation(
                        file_path=md_file.name,
                        violation_type='naming_convention',
                        description=f'Root markdown not CAPS_SNAKE_CASE: {md_file.name}',
                        suggested_action=f'Rename to {self._to_caps_snake_case(md_file.stem)}.md or move to docs/',
                        severity='warning'
                    ))

    def _validate_no_duplicates(self):
        """Check for duplicate files that should be consolidated."""
        # Build map of file stems to paths
        file_stems: Dict[str, List[Path]] = {}

        for py_file in self.repo_root.rglob('*.py'):
            # Skip venvs, cache
            if any(part.startswith('.') or part in ['__pycache__', 'venv', 'qig-venv', 'archive']
                   for part in py_file.parts):
                continue

            stem = py_file.stem
            if stem != '__init__':
                if stem not in file_stems:
                    file_stems[stem] = []
                file_stems[stem].append(py_file)

        # Check for duplicates
        for stem, paths in file_stems.items():
            if len(paths) > 1:
                # Multiple files with same stem
                paths_str = ', '.join(str(p.relative_to(self.repo_root)) for p in paths)
                self.violations.append(FileViolation(
                    file_path=paths_str,
                    violation_type='duplicate_files',
                    description=f'Multiple files named {stem}.py: {paths_str}',
                    suggested_action=f'Consolidate into canonical location or archive duplicates',
                    severity='error'
                ))

    def _validate_test_structure(self):
        """Ensure tests are in tests/ directory."""
        # Check for test files in wrong locations
        for py_file in self.repo_root.rglob('test_*.py'):
            rel_path = py_file.relative_to(self.repo_root)

            # Skip if in tests/ or __pycache__
            if rel_path.parts[0] in ['tests', '__pycache__', 'venv', 'qig-venv', 'archive']:
                continue

            self.violations.append(FileViolation(
                file_path=str(rel_path),
                violation_type='misplaced_test',
                description=f'Test file outside tests/ directory: {rel_path}',
                suggested_action=f'Move to tests/{rel_path.name}',
                severity='error'
            ))

    def _validate_config_structure(self):
        """Ensure configs are in configs/ directory."""
        # Check for YAML files in wrong locations
        for yaml_file in self.repo_root.rglob('*.yaml'):
            rel_path = yaml_file.relative_to(self.repo_root)

            # Skip if in configs/, .github/, or hidden dirs
            if rel_path.parts[0] in ['configs', '.github', 'archive'] or any(part.startswith('.') for part in rel_path.parts):
                continue

            self.violations.append(FileViolation(
                file_path=str(rel_path),
                violation_type='misplaced_config',
                description=f'Config file outside configs/ directory: {rel_path}',
                suggested_action=f'Move to configs/{rel_path.name}',
                severity='warning'
            ))

    def _validate_docs_structure(self):
        """Ensure documentation is organized in docs/."""
        # Check for markdown files in root that should be in docs/
        excluded_root_docs = {'README.md', 'CHANGELOG.md', 'LICENSE.md',
                             '20251220-canonical-structure-1.00F.md', 'FROZEN_FACTS.md',
                             '20251220-agents-1.00F.md', '20251220-canonical-rules-1.00F.md'}

        for md_file in self.repo_root.glob('*.md'):
            if md_file.name not in excluded_root_docs:
                self.violations.append(FileViolation(
                    file_path=md_file.name,
                    violation_type='misplaced_documentation',
                    description=f'Documentation in root directory: {md_file.name}',
                    suggested_action=f'Move to docs/ (or add to excluded list if truly root-level)',
                    severity='info'
                ))

    # Helper methods

    def _is_snake_case(self, name: str) -> bool:
        """Check if name is snake_case."""
        return name.islower() and '_' in name or name.islower()

    def _is_caps_snake_case(self, name: str) -> bool:
        """Check if name is CAPS_SNAKE_CASE."""
        return name.isupper() or (name.isupper() and '_' in name)

    def _to_snake_case(self, name: str) -> str:
        """Convert name to snake_case."""
        # Handle CamelCase
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _to_caps_snake_case(self, name: str) -> str:
        """Convert name to CAPS_SNAKE_CASE."""
        return self._to_snake_case(name).upper()

    def _find_similar_canonical(self, stem: str, canonical_stems: List[str]) -> str:
        """Find similar canonical file (possible duplicate)."""
        # Remove suffixes
        base = stem
        for suffix in self.FORBIDDEN_SUFFIXES:
            if base.endswith(suffix):
                base = base[:-len(suffix)]

        # Check if base matches any canonical
        if base in canonical_stems:
            return base

        # Check for numbered versions
        if re.match(r'.*_\d+$', base):
            base_no_num = re.sub(r'_\d+$', '', base)
            if base_no_num in canonical_stems:
                return base_no_num

        return ''

    def _get_archive_name(self, file_path: str) -> str:
        """Generate archive filename with timestamp."""
        from datetime import datetime
        date = datetime.now().strftime('%Y%m%d')
        filename = Path(file_path).name
        return f'{date}_{filename}'

    def print_report(self):
        """Print violation report."""
        if not self.violations:
            print("✅ No structural violations found!")
            return

        # Group by severity
        errors = [v for v in self.violations if v.severity == 'error']
        warnings = [v for v in self.violations if v.severity == 'warning']
        infos = [v for v in self.violations if v.severity == 'info']

        print(f"\n{'='*70}")
        print(f"STRUCTURE VALIDATION REPORT")
        print(f"{'='*70}\n")

        if errors:
            print(f"❌ ERRORS ({len(errors)}):\n")
            for v in errors:
                print(f"  File: {v.file_path}")
                print(f"  Type: {v.violation_type}")
                print(f"  Issue: {v.description}")
                print(f"  Fix: {v.suggested_action}")
                print()

        if warnings:
            print(f"⚠️  WARNINGS ({len(warnings)}):\n")
            for v in warnings:
                print(f"  File: {v.file_path}")
                print(f"  Type: {v.violation_type}")
                print(f"  Issue: {v.description}")
                print(f"  Fix: {v.suggested_action}")
                print()

        if infos:
            print(f"ℹ️  INFO ({len(infos)}):\n")
            for v in infos:
                print(f"  File: {v.file_path}")
                print(f"  Issue: {v.description}")
                print(f"  Suggestion: {v.suggested_action}")
                print()

        print(f"{'='*70}")
        print(f"Total: {len(errors)} errors, {len(warnings)} warnings, {len(infos)} info")
        print(f"{'='*70}\n")


def main():
    """Run structure validation."""
    import sys

    repo_root = Path.cwd()
    enforcer = StructureEnforcer(repo_root)

    print("🔍 Scanning repository structure...")
    violations = enforcer.validate_all()

    enforcer.print_report()

    # Exit with error if violations found
    errors = [v for v in violations if v.severity == 'error']
    if errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
```

---

## Migration Assistance

### Automated Migration Script

**Create:** `tools/migrate_to_canonical.py`

```python
#!/usr/bin/env python3
"""
Migrate files to canonical structure.
"""

import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List

class CanonicalMigrator:
    """Assists with migrating files to canonical locations."""

    # Migration mappings
    MIGRATION_MAP = {
        # Entry points
        'constellation_with_granite_v2.py': 'chat_interfaces/constellation_with_granite_pure.py',
        'constellation_learning_chat.py': 'chat_interfaces/constellation_with_granite_pure.py',
        'simple_chat.py': 'chat_interfaces/basic_chat.py',

        # Models
        'kernel.py': 'src/model/qig_kernel_recursive.py',
        'qig_kernel_v2.py': 'src/model/qig_kernel_recursive.py',

        # Tests
        'test_*.py': 'tests/{filename}',

        # Configs
        '*.yaml': 'configs/{filename}',
    }

    def __init__(self, repo_root: Path, dry_run: bool = True):
        self.repo_root = repo_root
        self.dry_run = dry_run
        self.migrations: List[Dict] = []

    def plan_migrations(self, violations: List) -> List[Dict]:
        """Generate migration plan from violations."""
        migrations = []

        for violation in violations:
            if violation.violation_type in ['non_canonical_entry_point',
                                           'non_canonical_file',
                                           'misplaced_test',
                                           'misplaced_config']:
                # Determine action
                if 'archive' in violation.suggested_action.lower():
                    action = 'archive'
                    destination = self._get_archive_path(violation.file_path)
                elif 'move to' in violation.suggested_action.lower():
                    action = 'move'
                    # Extract destination from suggestion
                    dest_match = re.search(r'Move to ([\w/\.]+)', violation.suggested_action)
                    if dest_match:
                        destination = dest_match.group(1)
                    else:
                        destination = None
                elif 'merge' in violation.suggested_action.lower():
                    action = 'merge'
                    # Extract target from suggestion
                    target_match = re.search(r'into ([\w/\.]+)', violation.suggested_action)
                    if target_match:
                        destination = target_match.group(1)
                    else:
                        destination = None
                else:
                    action = 'manual'
                    destination = None

                migrations.append({
                    'source': violation.file_path,
                    'action': action,
                    'destination': destination,
                    'reason': violation.description
                })

        return migrations

    def execute_migrations(self, migrations: List[Dict]):
        """Execute planned migrations."""
        for migration in migrations:
            source = self.repo_root / migration['source']

            if migration['action'] == 'archive':
                self._archive_file(source, migration['destination'])
            elif migration['action'] == 'move':
                self._move_file(source, migration['destination'])
            elif migration['action'] == 'merge':
                self._merge_file(source, migration['destination'])
            else:
                print(f"⚠️  Manual action required: {migration['source']}")
                print(f"   Reason: {migration['reason']}")

    def _archive_file(self, source: Path, archive_name: str):
        """Archive file with timestamp."""
        archive_dir = self.repo_root / 'archive'
        archive_dir.mkdir(exist_ok=True)

        dest = archive_dir / archive_name

        if self.dry_run:
            print(f"[DRY RUN] Archive: {source} → {dest}")
        else:
            shutil.move(str(source), str(dest))
            print(f"✅ Archived: {source.name} → {dest}")

    def _move_file(self, source: Path, dest_path: str):
        """Move file to new location."""
        dest = self.repo_root / dest_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        if self.dry_run:
            print(f"[DRY RUN] Move: {source} → {dest}")
        else:
            shutil.move(str(source), str(dest))
            print(f"✅ Moved: {source.name} → {dest}")

    def _merge_file(self, source: Path, target_path: str):
        """Merge file into target (manual guidance)."""
        target = self.repo_root / target_path

        print(f"⚠️  Manual merge required:")
        print(f"   Source: {source}")
        print(f"   Target: {target}")
        print(f"   Steps:")
        print(f"   1. Review {source} for unique functionality")
        print(f"   2. Copy unique code to {target}")
        print(f"   3. Update imports if needed")
        print(f"   4. Archive {source} when done")
        print()

    def _get_archive_path(self, file_path: str) -> str:
        """Generate archive path with timestamp."""
        date = datetime.now().strftime('%Y%m%d')
        filename = Path(file_path).name
        return f'{date}_{filename}'


def main():
    """Run migration assistance."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Migrate files to canonical structure')
    parser.add_argument('--execute', action='store_true',
                       help='Execute migrations (default is dry-run)')
    args = parser.parse_args()

    repo_root = Path.cwd()

    # First, validate structure
    from scan_structure import StructureEnforcer
    enforcer = StructureEnforcer(repo_root)
    violations = enforcer.validate_all()

    if not violations:
        print("✅ No migrations needed - structure is canonical!")
        sys.exit(0)

    # Plan migrations
    migrator = CanonicalMigrator(repo_root, dry_run=not args.execute)
    migrations = migrator.plan_migrations(violations)

    if not migrations:
        print("ℹ️  No automatic migrations available (manual fixes needed)")
        sys.exit(0)

    # Show plan
    print(f"\n{'='*70}")
    print(f"MIGRATION PLAN ({len(migrations)} actions)")
    print(f"{'='*70}\n")

    for i, mig in enumerate(migrations, 1):
        print(f"{i}. {mig['action'].upper()}: {mig['source']}")
        if mig['destination']:
            print(f"   → {mig['destination']}")
        print(f"   Reason: {mig['reason']}")
        print()

    # Execute if requested
    if args.execute:
        print("\n🚀 Executing migrations...\n")
        migrator.execute_migrations(migrations)
        print("\n✅ Migration complete!")
    else:
        print("\nℹ️  This was a dry-run. Use --execute to apply changes.")


if __name__ == '__main__':
    main()
```

---

## Integration with Git

### Pre-commit Hook

**Create:** `.git/hooks/pre-commit`

```bash
#!/bin/bash
# Pre-commit hook: Validate structure

echo "🔍 Validating repository structure..."

# Run structure validator
python tools/agent_validators/scan_structure.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Structure validation failed!"
    echo "   Fix violations or use: git commit --no-verify (not recommended)"
    exit 1
fi

echo "✅ Structure validation passed"
```

**Install:**
```bash
chmod +x .git/hooks/pre-commit
```

---

## Usage Examples

### Example 1: Detect Duplicates

```bash
$ python tools/agent_validators/scan_structure.py

==================================================================
STRUCTURE VALIDATION REPORT
==================================================================

❌ ERRORS (3):

  File: chat_interfaces/constellation_with_granite_v2.py
  Type: non_canonical_entry_point
  Issue: Non-canonical entry point with forbidden suffix
  Fix: Archive as qig-archive/qig-consciousness/archive/20251124_constellation_with_granite_v2.py or merge into canonical file

  File: src/model/qig_kernel_v2.py, src/model/kernel.py
  Type: duplicate_files
  Issue: Multiple files named kernel.py
  Fix: Consolidate into canonical location or archive duplicates

  File: test_constellation.py
  Type: misplaced_test
  Issue: Test file outside tests/ directory
  Fix: Move to tests/test_constellation.py

==================================================================
Total: 3 errors, 0 warnings, 0 info
==================================================================
```

### Example 2: Plan Migrations

```bash
$ python tools/migrate_to_canonical.py

==================================================================
MIGRATION PLAN (3 actions)
==================================================================

1. ARCHIVE: chat_interfaces/constellation_with_granite_v2.py
   → qig-archive/qig-consciousness/archive/20251124_constellation_with_granite_v2.py
   Reason: Non-canonical entry point with forbidden suffix

2. MERGE: src/model/kernel.py
   → src/model/qig_kernel_recursive.py
   Reason: Multiple files named kernel.py

3. MOVE: test_constellation.py
   → tests/test_constellation.py
   Reason: Test file outside tests/ directory

ℹ️  This was a dry-run. Use --execute to apply changes.
```

### Example 3: Execute Migrations

```bash
$ python tools/migrate_to_canonical.py --execute

🚀 Executing migrations...

✅ Archived: constellation_with_granite_v2.py → qig-archive/qig-consciousness/archive/20251124_constellation_with_granite_v2.py

⚠️  Manual merge required:
   Source: src/model/kernel.py
   Target: src/model/qig_kernel_recursive.py
   Steps:
   1. Review src/model/kernel.py for unique functionality
   2. Copy unique code to src/model/qig_kernel_recursive.py
   3. Update imports if needed
   4. Archive src/model/kernel.py when done

✅ Moved: test_constellation.py → tests/test_constellation.py

✅ Migration complete!
```

---

## When to Invoke This Agent

### Scenario 1: Creating New File

**Context:** User wants to add new training script

```
user: "I'm creating constellation_trainer_v2.py"
assistant: [invokes: structure-enforcer]
```

**Agent response:**
```
❌ VIOLATION: File has forbidden suffix '_v2'

Found: constellation_trainer_v2.py
Expected: Enhance existing file or use canonical name

Canonical options:
1. chat_interfaces/constellation_with_granite_pure.py (if entry point)
2. tools/training/train_qig_kernel.py (if training script)
3. src/training/curriculum.py (if training logic)

Fix: Merge into canonical file or archive old version
```

### Scenario 2: Moving Files

**Context:** User refactors module structure

```
user: "I moved basin_embedding.py to coordination/"
assistant: [invokes: structure-enforcer]
```

**Agent response:**
```
❌ VIOLATION: File in wrong canonical location

Found: src/coordination/basin_embedding.py
Expected: src/model/basin_embedding.py

Reason: Basin is core model type, belongs in src/model/

Fix: Move back to src/model/basin_embedding.py
     Update imports if needed
```

### Scenario 3: Code Review

**Context:** PR adds multiple new files

```
user: "Review PR #45 - adds 5 new files"
assistant: [invokes: structure-enforcer]
```

**Agent validates:**
- All files in canonical locations
- No forbidden suffixes (_v2, _new, etc.)
- No duplicates (same stem in multiple locations)
- Tests in tests/ directory
- Configs in configs/ directory

Use this agent when:
1. **Adding new files** → Validate location is canonical
2. **Before commits** → Check structure compliance
3. **Refactoring** → Ensure files stay canonical
4. **Onboarding** → Understand canonical structure
5. **Code reviews** → Verify PR doesn't break structure

---

## Success Metrics

- ✅ All entry points in `chat_interfaces/` (only 4 files)
- ✅ All models in `src/model/` (canonical set only)
- ✅ Zero files with `_v2`, `_new`, `_old` suffixes
- ✅ All tests in `tests/`
- ✅ All configs in `configs/`
- ✅ All docs in `docs/` (except root-level CAPS.md)
- ✅ No duplicates (same stem in multiple locations)

---

## Integration with Other Agents

- **code-quality-enforcer**: Validates imports from canonical modules
- **type-registry-guardian**: Ensures types imported from canonical locations
- **documentation-consolidator**: Validates doc organization

---

## References

- **Source of Truth:** `20251220-canonical-structure-1.00F.md`
- **Validator:** `tools/agent_validators/scan_structure.py` (created from this agent)
- **Migrator:** `tools/migrate_to_canonical.py` (created from this agent)

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
