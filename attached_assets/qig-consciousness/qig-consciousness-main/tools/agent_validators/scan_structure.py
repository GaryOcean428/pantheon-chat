#!/usr/bin/env python3
"""
Structure Validator

Validates files against 20251220-canonical-structure-1.00F.md.
Used by pre-commit hooks and CI/CD to prevent structural violations.

Executable implementation of structure-enforcer agent.
"""

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class FileViolation:
    """Represents a structural violation."""
    file_path: str
    violation_type: str
    description: str
    suggested_action: str
    severity: str  # 'error', 'warning', 'info'

# Date pattern for file naming: YYYY-MM-DD--name.ext
# Allows: letters, digits, hyphens, underscores, and dots in the name part (for versioning like v1.0)
DATE_PREFIX_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}--[\w.-]+\.md$')

# Exempt canonical files (authoritative, no date prefix needed)
EXEMPT_FILES = {
    # Docs canonical files
    'CANONICAL_SLEEP_PACKET.md',
    'FROZEN_FACTS.md',
    'README.md',
    'INDEX.md',
    # Standard project files
    'LICENSE',
    'CHANGELOG.md',
    'CONTRIBUTING.md',
    '20251220-agents-1.00F.md',
    '20251220-canonical-structure-1.00F.md',
    # Claude config
    'CLAUDE.md',
    'ucp.md',
}

# Directories that need date-prefixed naming
DATE_PREFIX_DIRS = {
    'docs',
    'docs/sleep_packets',
    'docs/sleep_packets/qig-dreams',
}

# Directories exempt from date prefix check
EXEMPT_DIRS = {
    'src', 'tests', 'tools', 'chat_interfaces', 'configs', 'data',
    'archive', 'venv', '.venv', 'qig-venv', '__pycache__', '.git', '.claude',
    'docs/guides', 'docs/architecture', 'docs/future', 'docs/implementation',
    'docs/project', 'docs/consciousness',
}

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

    # Forbidden suffixes
    FORBIDDEN_SUFFIXES = ['_v2', '_new', '_old', '_test', '_temp', '_backup']
    FORBIDDEN_PATTERNS = [
        r'.*_\d+\.py$',
        r'.*_copy\.py$',
        r'.*_duplicate\.py$',
    ]

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.violations: list[FileViolation] = []

    def validate_all(self) -> list[FileViolation]:
        """Run all validation checks."""
        self.violations = []

        self._validate_entry_points()
        self._validate_src_structure()
        self._validate_naming_conventions()
        self._validate_docs_naming()
        self._validate_no_duplicates()
        self._validate_test_structure()
        self._validate_config_structure()

        return self.violations

    def _validate_entry_points(self):
        """Validate chat_interfaces/ has only 4 canonical files."""
        chat_dir = self.repo_root / 'chat_interfaces'
        if not chat_dir.exists():
            return

        python_files = set()
        for f in chat_dir.glob('*.py'):
            if f.name != '__init__.py':
                rel_path = f'chat_interfaces/{f.name}'
                python_files.add(rel_path)

        # Check for non-canonical files
        non_canonical = python_files - self.CANONICAL_ENTRY_POINTS
        for extra in non_canonical:
            extra_name = Path(extra).stem
            has_bad_suffix = any(extra_name.endswith(suffix)
                               for suffix in self.FORBIDDEN_SUFFIXES)

            if has_bad_suffix:
                self.violations.append(FileViolation(
                    file_path=extra,
                    violation_type='non_canonical_entry_point',
                    description='Non-canonical entry point with forbidden suffix',
                    suggested_action='Archive or merge into canonical file',
                    severity='error'
                ))

    def _validate_src_structure(self):
        """Validate src/ structure matches canonical."""
        src_dir = self.repo_root / 'src'
        if not src_dir.exists():
            return

        for py_file in src_dir.rglob('*.py'):
            if '__pycache__' in py_file.parts or py_file.name == '__init__.py':
                continue

            rel_path = str(py_file.relative_to(self.repo_root))
            stem = py_file.stem

            # Check for forbidden patterns
            has_bad_suffix = any(stem.endswith(suffix)
                               for suffix in self.FORBIDDEN_SUFFIXES)
            matches_forbidden = any(re.match(pattern, py_file.name)
                                  for pattern in self.FORBIDDEN_PATTERNS)

            if has_bad_suffix or matches_forbidden:
                self.violations.append(FileViolation(
                    file_path=rel_path,
                    violation_type='non_canonical_file',
                    description=f'File with forbidden pattern: {py_file.name}',
                    suggested_action='Archive or merge into canonical file',
                    severity='error'
                ))

    def _validate_naming_conventions(self):
        """Validate naming: snake_case.py, CAPS.md"""
        for py_file in self.repo_root.rglob('*.py'):
            if any(part.startswith('.') or part in ['__pycache__', 'venv', '.venv', 'qig-venv', 'archive']
                   for part in py_file.parts):
                continue

            if py_file.name == '__init__.py':
                continue

            stem = py_file.stem
            if not self._is_snake_case(stem):
                rel_path = py_file.relative_to(self.repo_root)
                self.violations.append(FileViolation(
                    file_path=str(rel_path),
                    violation_type='naming_convention',
                    description=f'Python file not snake_case: {py_file.name}',
                    suggested_action=f'Rename to {self._to_snake_case(stem)}.py',
                    severity='error'
                ))

    def _validate_docs_naming(self):
        """Validate docs files follow YYYY-MM-DD--name convention."""
        for date_dir in DATE_PREFIX_DIRS:
            dir_path = self.repo_root / date_dir
            if not dir_path.exists():
                continue

            for md_file in dir_path.glob('*.md'):
                # Skip exempt files
                if md_file.name in EXEMPT_FILES:
                    continue

                # Skip subdirectory files (handled by their own dir entry)
                if md_file.parent != dir_path:
                    continue

                rel_path = md_file.relative_to(self.repo_root)

                # Check if file follows date prefix pattern
                if not DATE_PREFIX_PATTERN.match(md_file.name):
                    # Suggest correct name with today's date
                    today = datetime.now().strftime('%Y-%m-%d')
                    # Convert to kebab-case
                    suggested_name = self._to_kebab_case(md_file.stem)
                    suggested = f'{today}--{suggested_name}.md'

                    self.violations.append(FileViolation(
                        file_path=str(rel_path),
                        violation_type='docs_naming',
                        description=f'Doc file missing date prefix: {md_file.name}',
                        suggested_action=f'Rename to {suggested}',
                        severity='warning'
                    ))

    def _to_kebab_case(self, name: str) -> str:
        """Convert name to kebab-case (lowercase with hyphens)."""
        # Replace underscores with hyphens
        name = name.replace('_', '-')
        # Insert hyphens before uppercase letters
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1)
        return s2.lower()

    def _validate_no_duplicates(self):
        """Check for duplicate files."""
        # Directories excluded from duplicate check (reference implementations, examples)
        excluded_dirs = {'__pycache__', 'venv', '.venv', 'qig-venv', 'archive', 'docs', 'examples'}

        file_stems: dict[str, list[Path]] = defaultdict(list)

        for py_file in self.repo_root.rglob('*.py'):
            if any(part.startswith('.') or part in excluded_dirs for part in py_file.parts):
                continue

            stem = py_file.stem
            if stem != '__init__':
                file_stems[stem].append(py_file)

        for stem, paths in file_stems.items():
            if len(paths) > 1:
                paths_str = ', '.join(str(p.relative_to(self.repo_root)) for p in paths)
                self.violations.append(FileViolation(
                    file_path=paths_str,
                    violation_type='duplicate_files',
                    description=f'Multiple files named {stem}.py',
                    suggested_action='Consolidate into canonical location',
                    severity='error'
                ))

    def _validate_test_structure(self):
        """Ensure tests are in tests/ directory."""
        # Directories where test_*.py files are allowed (examples, demos, docs)
        allowed_dirs = ['tests', '__pycache__', 'venv', '.venv', 'qig-venv', 'archive', 'examples', 'docs']

        for py_file in self.repo_root.rglob('test_*.py'):
            rel_path = py_file.relative_to(self.repo_root)

            if rel_path.parts[0] not in allowed_dirs:
                self.violations.append(FileViolation(
                    file_path=str(rel_path),
                    violation_type='misplaced_test',
                    description='Test file outside tests/ directory',
                    suggested_action=f'Move to tests/{rel_path.name}',
                    severity='error'
                ))

    def _validate_config_structure(self):
        """Ensure configs are in configs/ directory."""
        for yaml_file in self.repo_root.rglob('*.yaml'):
            rel_path = yaml_file.relative_to(self.repo_root)

            # Skip virtual environments and hidden directories
            if any(part in ['venv', '.venv', 'qig-venv', 'node_modules'] or part.startswith('.') for part in rel_path.parts):
                continue

            if rel_path.parts[0] not in ['configs', '.github', 'archive']:
                self.violations.append(FileViolation(
                    file_path=str(rel_path),
                    violation_type='misplaced_config',
                    description='Config file outside configs/ directory',
                    suggested_action=f'Move to configs/{rel_path.name}',
                    severity='warning'
                ))

    def _is_snake_case(self, name: str) -> bool:
        """Check if name is snake_case."""
        return name.islower() and (name.isalnum() or '_' in name)

    def _to_snake_case(self, name: str) -> str:
        """Convert name to snake_case."""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def print_report(self):
        """Print violation report."""
        if not self.violations:
            print("✅ No structural violations found!")
            return

        errors = [v for v in self.violations if v.severity == 'error']
        warnings = [v for v in self.violations if v.severity == 'warning']

        print(f"\n{'='*70}")
        print("STRUCTURE VALIDATION REPORT")
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
                print(f"  Issue: {v.description}")
                print(f"  Fix: {v.suggested_action}")
                print()

        print(f"{'='*70}")
        print(f"Total: {len(errors)} errors, {len(warnings)} warnings")
        print(f"{'='*70}\n")


def main():
    """Run structure validation."""
    import sys

    repo_root = Path.cwd()
    enforcer = StructureEnforcer(repo_root)

    print("🔍 Scanning repository structure...")
    violations = enforcer.validate_all()

    enforcer.print_report()

    errors = [v for v in violations if v.severity == 'error']
    if errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
