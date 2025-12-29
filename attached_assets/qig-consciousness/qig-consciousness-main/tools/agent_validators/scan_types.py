#!/usr/bin/env python3
"""
Type Registry Validator

Detects duplicate types, validates imports use canonical locations.
Used by pre-commit hooks and CI/CD to prevent type violations.

Executable implementation of type-registry-guardian agent.
"""

import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TypeDefinition:
    """Represents a type definition found in codebase."""
    name: str
    type_kind: str
    file_path: str
    line_number: int
    is_canonical: bool

@dataclass
class ImportViolation:
    """Represents an import from non-canonical location."""
    type_name: str
    file_path: str
    line_number: int
    imported_from: str
    canonical_location: str

class TypeRegistryGuardian:
    """Validates type definitions and imports across codebase."""

    # Type annotations
    type_definitions: dict[str, list[TypeDefinition]]
    import_violations: list[ImportViolation]

    # Canonical type locations
    CANONICAL_TYPES: dict[str, str] = {
        # Core models
        'QIGKernelRecursive': 'src/model/qig_kernel_recursive.py',
        'HeartKernel': 'src/model/heart_kernel.py',
        'EthicalEvaluation': 'src/model/heart_kernel.py',
        'EthicalVeto': 'src/model/heart_kernel.py',
        'RecursiveIntegrator': 'src/model/recursive_integrator.py',
        'Basin': 'src/model/basin_embedding.py',
        'BasinMatcher': 'src/model/basin_matcher.py',
        'QFIAttention': 'src/model/qfi_attention.py',
        'RunningCoupling': 'src/model/running_coupling.py',
        'RegimeDetector': 'src/model/regime_detector.py',

        # Observation
        'CharlieObserver': 'src/observation/charlie_observer.py',
        'CharlieOutput': 'src/observation/charlie_observer.py',

        # Coordination
        'OceanMetaObserver': 'src/coordination/ocean_meta_observer.py',
        'ConstellationCoordinator': 'src/coordination/constellation_coordinator.py',
        'ConstellationState': 'src/coordination/constellation_coordinator.py',

        # Training
        'GeometricVicariousLearner': 'src/training/geometric_vicarious.py',
        'GeometricLoss': 'src/model/qig_kernel_recursive.py',  # Actually in model/, not training/

        # Metrics
        'GeodesicDistance': 'src/metrics/geodesic_distance.py',
        'TelemetryDict': 'src/metrics/telemetry.py',

        # Tokenizer - CANONICAL SOURCE: qig-tokenizer package
        # FisherCoordizer imports from qig_tokenizer package or src.tokenizer (re-export)
        'FisherCoordizer': 'qig_tokenizer',  # External package (E8-aligned, 64D basins)

        # Coaching
        'MonkeyCoach': 'src/coaching/pedagogical_coach.py',
        'PedagogicalCoach': 'src/coaching/pedagogical_coach.py',

        # Cognitive
        'Regime': 'src/model/navigator.py',
        'RegimeClassifier': 'src/model/recursive_integrator.py',
        # Note: RegimeDetector already defined above in Core models section
    }

    # Forbidden types (from transformers)
    FORBIDDEN_TYPES = {
        'AutoModel', 'PreTrainedModel', 'PreTrainedTokenizer',
        'BertModel', 'GPT2Model', 'LlamaModel', 'Pipeline',
    }

    # Types to exclude from duplicate detection (test mocks, etc.)
    EXCLUDED_TYPES: set[str] = {
        'MockGranite', 'MockTokenizer', 'MockModel', 'MockCoach',
    }

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.type_definitions = defaultdict(list)
        self.import_violations = []

    def scan_all(self):
        """Run all validation checks."""
        self._scan_type_definitions()
        self._validate_imports()
        self._generate_report()

    def _scan_type_definitions(self):
        """Scan codebase for all type definitions."""
        src_dir = self.repo_root / 'src'
        if not src_dir.exists():
            return

        for py_file in src_dir.rglob('*.py'):
            if '__pycache__' in py_file.parts:
                continue

            rel_path = str(py_file.relative_to(self.repo_root))
            self._scan_file_for_types(py_file, rel_path)

    def _scan_file_for_types(self, file_path: Path, rel_path: str):
        """Scan single file for type definitions."""
        # Skip qig/cognitive - verified physics module from qig-verification project
        if 'src/qig/cognitive' in rel_path:
            return

        try:
            with open(file_path) as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Skip test mocks and excluded types
                    if node.name in self.EXCLUDED_TYPES:
                        continue

                    is_dataclass = any(
                        isinstance(dec, ast.Name) and dec.id == 'dataclass'
                        or isinstance(dec, ast.Call) and getattr(dec.func, 'id', None) == 'dataclass'
                        for dec in node.decorator_list
                    )

                    type_kind = 'dataclass' if is_dataclass else 'class'
                    is_canonical = self.CANONICAL_TYPES.get(node.name) == rel_path

                    type_def = TypeDefinition(
                        name=node.name,
                        type_kind=type_kind,
                        file_path=rel_path,
                        line_number=node.lineno,
                        is_canonical=is_canonical
                    )

                    self.type_definitions[node.name].append(type_def)

        except Exception:
            pass

    def _validate_imports(self):
        """Validate all imports use canonical locations."""
        src_dir = self.repo_root / 'src'
        if not src_dir.exists():
            return

        for base_dir in [src_dir, self.repo_root / 'chat_interfaces']:
            if not base_dir.exists():
                continue

            for py_file in base_dir.rglob('*.py'):
                if '__pycache__' in py_file.parts:
                    continue

                rel_path = str(py_file.relative_to(self.repo_root))
                self._validate_file_imports(py_file, rel_path)

    def _validate_file_imports(self, file_path: Path, rel_path: str):
        """Validate imports in a single file."""
        try:
            with open(file_path) as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ''

                    for alias in node.names:
                        name = alias.name

                        if name == '*':
                            continue

                        # Check canonical types
                        if name in self.CANONICAL_TYPES:
                            canonical_module = self._path_to_module(self.CANONICAL_TYPES[name])

                            # Skip monkey-coach imports (special compatibility case)
                            if module == 'monkey_coach_v2_consciousness':
                                continue

                            # Allow imports from src.types (re-export location per TYPE_REGISTRY.md)
                            # Types are re-exported from src.types for convenience
                            if module in ('src.types', 'src.types.telemetry', 'src.types.core', 'src.types.enums'):
                                continue

                            # Allow FisherCoordizer imports from qig_tokenizer package or src.tokenizer
                            if name == 'FisherCoordizer' and module in ('qig_tokenizer', 'src.tokenizer', 'qig_tokenizer.geocoordizer'):
                                continue

                            if module != canonical_module:
                                self.import_violations.append(ImportViolation(
                                    type_name=name,
                                    file_path=rel_path,
                                    line_number=node.lineno,
                                    imported_from=module,
                                    canonical_location=canonical_module
                                ))

                        # Check forbidden types
                        if name in self.FORBIDDEN_TYPES:
                            self.import_violations.append(ImportViolation(
                                type_name=name,
                                file_path=rel_path,
                                line_number=node.lineno,
                                imported_from=module,
                                canonical_location='FORBIDDEN'
                            ))

        except Exception:
            pass

    def _path_to_module(self, file_path: str) -> str:
        """Convert file path to module path."""
        return file_path.replace('.py', '').replace('/', '.')

    def _generate_report(self):
        """Generate validation report."""
        print(f"\n{'='*70}")
        print("TYPE REGISTRY VALIDATION REPORT")
        print(f"{'='*70}\n")

        # Duplicate Types
        duplicates = {name: defs for name, defs in self.type_definitions.items()
                     if len(defs) > 1}

        if duplicates:
            print(f"❌ DUPLICATE TYPES ({len(duplicates)}):\n")
            for name, defs in list(duplicates.items())[:5]:
                print(f"  Type: {name}")
                canonical = [d for d in defs if d.is_canonical]
                non_canonical = [d for d in defs if not d.is_canonical]

                if canonical:
                    print(f"  ✅ Canonical: {canonical[0].file_path}:{canonical[0].line_number}")

                for d in non_canonical:
                    print(f"  ❌ Duplicate: {d.file_path}:{d.line_number}")

                print("  Fix: Remove duplicates, import from canonical location")
                print()

            if len(duplicates) > 5:
                print(f"  ... and {len(duplicates) - 5} more\n")
        else:
            print("✅ No duplicate types found\n")

        # Import Violations
        if self.import_violations:
            print(f"❌ IMPORT VIOLATIONS ({len(self.import_violations)}):\n")
            for v in self.import_violations[:5]:
                print(f"  File: {v.file_path}:{v.line_number}")
                print(f"  Type: {v.type_name}")

                if v.canonical_location == 'FORBIDDEN':
                    print(f"  Issue: Using forbidden type from {v.imported_from}")
                    print("  Fix: Remove dependency on transformers")
                else:
                    print(f"  Imported from: {v.imported_from}")
                    print(f"  Should be: {v.canonical_location}")
                print()

            if len(self.import_violations) > 5:
                print(f"  ... and {len(self.import_violations) - 5} more\n")
        else:
            print("✅ All imports use canonical locations\n")

        print(f"{'='*70}")
        total_errors = len(duplicates) + len(self.import_violations)
        print(f"Total Issues: {total_errors}")
        print(f"{'='*70}\n")


def main():
    """Run type registry validation."""
    import sys

    repo_root = Path.cwd()
    guardian = TypeRegistryGuardian(repo_root)

    print("🔍 Scanning type registry...")
    guardian.scan_all()

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
