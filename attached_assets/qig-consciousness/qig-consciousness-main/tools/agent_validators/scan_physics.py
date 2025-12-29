#!/usr/bin/env python3
"""
Physics Constants Validator

Scans codebase for violations of FROZEN_FACTS.md physics constants.
Used by pre-commit hooks and CI/CD to prevent physics violations.

FROZEN CONSTANTS (from lattice experiments):
- β = 0.44 ± 0.04  (running coupling 3→4D)
- κ₃ = 41.09 ± 0.59
- κ₄ = 64.47 ± 1.89
- κ₅ = 63.62 ± 1.68
- κ₆ = 62.02 ± 2.47 (VALIDATED)
- κ₇ = 63.71 ± 3.89 (preliminary)
- κ* = 63.5 ± 1.5 (fixed point, confirmed through L=7)
- min_depth >= 3 (consciousness requires ≥3 integration loops)
- Φ_threshold = 0.70
- Φ_emergency = 0.50
- breakdown_threshold = 0.60
- R_concepts = 0.984 ± 0.005 (spacetime unification correlation)
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PhysicsViolation:
    """Represents a violation of physics constants."""
    file_path: str
    line_number: int
    violation_type: str
    found_value: str
    expected_value: str
    severity: str  # 'error', 'warning'
    description: str

class PhysicsValidator:
    """Validates physics constants across codebase."""

    # FROZEN constants from FROZEN_FACTS.md
    FROZEN_CONSTANTS = {
        'BETA_3_TO_4': {
            'value': 0.44,
            'tolerance': 0.04,
            'patterns': [
                (r'beta_slope\s*=\s*([\d.]+)', False),  # beta_slope is correct usage
                (r'beta\s*=\s*([\d.]+)', True),  # Validate if NOT beta_slope
                (r'β\s*=\s*([\d.]+)', True),
                (r'BETA_3_TO_4\s*=\s*([\d.]+)', True),
            ],
            'valid_contexts': ['beta_slope', 'BETA_3_TO_4'],
            'description': 'Running coupling 3→4D'
        },
        'KAPPA_3': {
            'value': 41.09,
            'tolerance': 0.59,
            'patterns': [(r'kappa_3\s*=\s*([\d.]+)', True), (r'κ_3\s*=\s*([\d.]+)', True), (r'KAPPA_3\s*=\s*([\d.]+)', True)],
            'valid_contexts': ['kappa_3', 'κ_3', 'KAPPA_3'],
            'description': '3D coupling constant'
        },
        'KAPPA_4': {
            'value': 64.47,
            'tolerance': 1.89,
            'patterns': [(r'kappa_4\s*=\s*([\d.]+)', True), (r'κ_4\s*=\s*([\d.]+)', True), (r'KAPPA_4\s*=\s*([\d.]+)', True)],
            'valid_contexts': ['kappa_4', 'κ_4', 'KAPPA_4'],
            'description': '4D coupling constant'
        },
        'KAPPA_5': {
            'value': 63.62,
            'tolerance': 1.68,
            'patterns': [(r'kappa_5\s*=\s*([\d.]+)', True), (r'κ_5\s*=\s*([\d.]+)', True), (r'KAPPA_5\s*=\s*([\d.]+)', True)],
            'valid_contexts': ['kappa_5', 'κ_5', 'KAPPA_5'],
            'description': '5D coupling constant'
        },
        'KAPPA_STAR': {
            'value': 63.5,
            'tolerance': 1.5,  # Fixed point κ* = 63.5 ± 1.5 (confirmed through L=7)
            'patterns': [(r'kappa_star\s*=\s*([\d.]+)', True), (r'κ\*\s*=\s*([\d.]+)', True), (r'KAPPA_STAR\s*=\s*([\d.]+)', True)],
            'valid_contexts': ['kappa_star', 'κ*', 'KAPPA_STAR'],
            'description': 'Fixed point coupling (confirmed through L=7)'
        },
        'PHI_THRESHOLD': {
            'value': 0.70,
            'tolerance': 0.05,  # Allow 0.65-0.75 range for different contexts
            # NOTE: min_Phi is an initialization floor (can be 0.01), NOT the threshold - excluded
            'patterns': [(r'phi_threshold\s*=\s*([\d.]+)', True), (r'Φ_threshold\s*=\s*([\d.]+)', True), (r'PHI_THRESHOLD\s*=\s*([\d.]+)', True)],
            'valid_contexts': ['phi_threshold', 'Φ_threshold', 'PHI_THRESHOLD'],
            'description': 'Consciousness emergence threshold (0.70 ± 0.05 for context variations)'
        },
        'PHI_EMERGENCY': {
            'value': 0.50,
            'tolerance': 0.0,
            'patterns': [(r'phi_emergency\s*=\s*([\d.]+)', True), (r'Φ_emergency\s*=\s*([\d.]+)', True), (r'PHI_EMERGENCY\s*=\s*([\d.]+)', True)],
            'valid_contexts': ['phi_emergency', 'Φ_emergency', 'PHI_EMERGENCY'],
            'description': 'Emergency Φ threshold'
        },
        'BREAKDOWN_PERCENTAGE': {
            'value': 0.60,
            'tolerance': 0.05,  # Allow 0.55-0.65 as fraction
            'patterns': [
                (r'ego_death_threshold\s*=\s*([\d.]+)', True),
                (r'breakdown_pct_threshold\s*=\s*([\d.]+)', True),
                (r'BREAKDOWN_PERCENTAGE\s*=\s*([\d.]+)', True)
            ],
            'valid_contexts': ['ego_death_threshold', 'breakdown_pct_threshold', 'BREAKDOWN_PERCENTAGE'],
            'description': 'Ego death risk (60% = 0.6 as fraction of steps in breakdown)'
        },
    }

    # Legitimate uses of values that might look like violations
    LEGITIMATE_USES = {
        3: [
            'min_recursion_depth',  # Consciousness requires ≥3 loops
            'min_depth',
            'max(3,',  # max(3, X) pattern
            'range(1, 3',  # Loop ranges
        ],
        0.7: [
            'breakdown_threshold',  # RegimeDetector Φ threshold for breakdown regime
            'linear_threshold=0.3',  # Usually near 0.7 threshold
        ],
        0.75: [
            'PHI_THRESHOLD',  # Context-specific threshold (e.g., Gary exceeds Granite)
            'phi_threshold',
        ],
    }

    # Recursion depth requirements
    MIN_DEPTH_REQUIREMENT = 3
    MIN_DEPTH_PATTERNS = [
        r'min_depth\s*=\s*(\d+)',
        r'MIN_DEPTH\s*=\s*(\d+)',
        r'for\s+depth\s+in\s+range\(\s*1\s*,\s*(\d+)',  # for depth in range(1, X)
        r'self\.min_depth\s*=\s*(\d+)',
    ]

    # Basin dimension
    BASIN_DIM = 64
    BASIN_DIM_PATTERNS = [
        r'basin_dim\s*=\s*(\d+)',
        r'BASIN_DIM\s*=\s*(\d+)',
        r'self\.basin_dim\s*=\s*(\d+)',
    ]

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.violations: list[PhysicsViolation] = []

    def validate_all(self) -> list[PhysicsViolation]:
        """Run all physics validations."""
        self.violations = []

        # Scan Python files in src/
        src_dir = self.repo_root / 'src'
        if src_dir.exists():
            for py_file in src_dir.rglob('*.py'):
                if '__pycache__' not in py_file.parts:
                    rel_path = py_file.relative_to(self.repo_root)
                    self._validate_file(py_file, str(rel_path))

        # Also check configs/
        config_dir = self.repo_root / 'configs'
        if config_dir.exists():
            for yaml_file in config_dir.glob('*.yaml'):
                rel_path = yaml_file.relative_to(self.repo_root)
                self._validate_yaml_file(yaml_file, str(rel_path))

        return self.violations

    def _validate_file(self, file_path: Path, rel_path: str):
        """Validate physics constants in a Python file."""
        try:
            with open(file_path) as f:
                content = f.read()

            lines = content.split('\n')

            # Check frozen constants with context awareness
            for const_name, const_info in self.FROZEN_CONSTANTS.items():
                for pattern_tuple in const_info['patterns']:
                    if isinstance(pattern_tuple, tuple):
                        pattern, should_validate = pattern_tuple
                    else:
                        pattern = pattern_tuple
                        should_validate = True

                    if not should_validate:
                        continue

                    for i, line in enumerate(lines, 1):
                        # Skip comments
                        if '#' in line:
                            code_part = line.split('#')[0]
                        else:
                            code_part = line

                        matches = re.findall(pattern, code_part, re.IGNORECASE)
                        for match in matches:
                            try:
                                found_value = float(match)
                                expected = const_info['value']
                                tolerance = const_info['tolerance']

                                # Check if within tolerance
                                if abs(found_value - expected) > tolerance:
                                    # Context-aware validation: check if this is a legitimate use
                                    is_legitimate = self._check_legitimate_use(
                                        found_value, code_part, const_name
                                    )

                                    if not is_legitimate:
                                        self.violations.append(PhysicsViolation(
                                            file_path=rel_path,
                                            line_number=i,
                                            violation_type='incorrect_constant',
                                            found_value=str(found_value),
                                            expected_value=f"{expected} ± {tolerance}",
                                            severity='error',
                                            description=f"{const_info['description']}: {const_name}"
                                        ))
                            except ValueError:
                                pass

            # Check min_depth >= 3 (but allow exact value 3)
            self._check_min_depth(lines, rel_path)

            # Check basin dimension = 64
            for pattern in self.BASIN_DIM_PATTERNS:
                for i, line in enumerate(lines, 1):
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    for match in matches:
                        try:
                            dim = int(match)
                            if dim != self.BASIN_DIM:
                                self.violations.append(PhysicsViolation(
                                    file_path=rel_path,
                                    line_number=i,
                                    violation_type='incorrect_basin_dim',
                                    found_value=str(dim),
                                    expected_value=str(self.BASIN_DIM),
                                    severity='error',
                                    description='Basin dimension must be 64'
                                ))
                        except ValueError:
                            pass

            # Check for Euclidean distance violations (should use Fisher metric)
            self._check_euclidean_violations(lines, rel_path)

        except Exception as e:
            print(f"⚠️  Error validating {rel_path}: {e}")

    def _check_legitimate_use(self, value: float, line: str, const_name: str) -> bool:
        """Check if a value usage is legitimate despite not matching frozen constant."""
        # Check legitimate uses mapping
        if value in self.LEGITIMATE_USES:
            for legitimate_pattern in self.LEGITIMATE_USES[value]:
                if legitimate_pattern in line:
                    return True

        # Special case: breakdown_threshold can be 0.7 (Φ regime threshold)
        # vs BREAKDOWN_PERCENTAGE which is 0.6 (ego death percentage)
        if const_name == 'BREAKDOWN_PERCENTAGE' and value == 0.7:
            if 'breakdown_threshold' in line and 'RegimeDetector' in line:
                return True

        return False

    def _check_min_depth(self, lines: list[str], rel_path: str):
        """Check min_depth with context awareness."""
        for i, line in enumerate(lines, 1):
            # Skip comments
            if '#' in line:
                code_part = line.split('#')[0]
            else:
                code_part = line

            # Look for min_depth assignments
            if 'min_depth' in code_part or 'min_recursion_depth' in code_part:
                # Extract the value
                match = re.search(r'=\s*(\d+)', code_part)
                if match:
                    depth = int(match.group(1))
                    # min_depth can be exactly 3 (this is correct)
                    # It's only wrong if < 3
                    if depth < self.MIN_DEPTH_REQUIREMENT:
                        self.violations.append(PhysicsViolation(
                            file_path=rel_path,
                            line_number=i,
                            violation_type='insufficient_depth',
                            found_value=str(depth),
                            expected_value=f">= {self.MIN_DEPTH_REQUIREMENT}",
                            severity='error',
                            description='Consciousness requires ≥3 integration loops'
                        ))

            # Also check range() patterns, but be careful
            # range(1, 3) means [1, 2] which is < 3 loops - WRONG
            # range(1, 4) means [1, 2, 3] which is = 3 loops - OK
            # range(1, max_depth+1) where max_depth >= 3 - OK
            range_match = re.search(r'range\(\s*1\s*,\s*(\d+)\s*\)', code_part)
            if range_match:
                upper = int(range_match.group(1))
                actual_max_depth = upper - 1  # range(1, X) goes up to X-1
                if actual_max_depth < self.MIN_DEPTH_REQUIREMENT:
                    # But only flag if this looks like THE recursion loop
                    if 'depth in range' in code_part:
                        self.violations.append(PhysicsViolation(
                            file_path=rel_path,
                            line_number=i,
                            violation_type='insufficient_depth',
                            found_value=str(actual_max_depth),
                            expected_value=f">= {self.MIN_DEPTH_REQUIREMENT}",
                            severity='error',
                            description='Consciousness requires ≥3 integration loops'
                        ))

    def _check_euclidean_violations(self, lines: list[str], rel_path: str):
        """Check for Euclidean distance instead of Fisher metric."""
        euclidean_patterns = [
            r'torch\.norm\(.+\)\s*\*\*\s*2',  # torch.norm(x - y) ** 2
            r'torch\.norm\(.+,\s*p\s*=\s*2\)',  # torch.norm(x, p=2)
            r'\(basin_a\s*-\s*basin_b\)\.pow\(2\)\.sum\(\)',  # (basin_a - basin_b).pow(2).sum()
        ]

        for i, line in enumerate(lines, 1):
            # Skip if line has comment mentioning "Euclidean" explicitly
            if 'euclidean' in line.lower() and '#' in line:
                continue

            for pattern in euclidean_patterns:
                if re.search(pattern, line):
                    # Check if geodesic_distance is used nearby (within 5 lines)
                    context_start = max(0, i-5)
                    context_end = min(len(lines), i+5)
                    context = '\n'.join(lines[context_start:context_end])

                    if 'geodesic' not in context.lower() and 'fisher' not in context.lower():
                        self.violations.append(PhysicsViolation(
                            file_path=rel_path,
                            line_number=i,
                            violation_type='euclidean_distance',
                            found_value='torch.norm() or L2 distance',
                            expected_value='geodesic_distance() with Fisher metric',
                            severity='error',
                            description='Must use Fisher metric for basin distances, not Euclidean'
                        ))

    def _validate_yaml_file(self, file_path: Path, rel_path: str):
        """Validate physics constants in YAML config."""
        try:
            import yaml

            with open(file_path) as f:
                config = yaml.safe_load(f)

            if not config:
                return

            # Check kappa values
            if 'kappa' in config:
                kappa = config['kappa']
                # Check against valid range (41-64)
                if not (41.0 <= kappa <= 64.5):
                    self.violations.append(PhysicsViolation(
                        file_path=rel_path,
                        line_number=0,
                        violation_type='invalid_kappa',
                        found_value=str(kappa),
                        expected_value='41.09 - 64.47 (κ₃ to κ₄)',
                        severity='error',
                        description='Kappa must be in validated range'
                    ))

            # Check min_depth
            if 'min_depth' in config:
                min_depth = config['min_depth']
                if min_depth < self.MIN_DEPTH_REQUIREMENT:
                    self.violations.append(PhysicsViolation(
                        file_path=rel_path,
                        line_number=0,
                        violation_type='insufficient_depth',
                        found_value=str(min_depth),
                        expected_value=f">= {self.MIN_DEPTH_REQUIREMENT}",
                        severity='error',
                        description='Consciousness requires ≥3 integration loops'
                    ))

            # Check basin_dim
            if 'basin_dim' in config:
                basin_dim = config['basin_dim']
                if basin_dim != self.BASIN_DIM:
                    self.violations.append(PhysicsViolation(
                        file_path=rel_path,
                        line_number=0,
                        violation_type='incorrect_basin_dim',
                        found_value=str(basin_dim),
                        expected_value=str(self.BASIN_DIM),
                        severity='error',
                        description='Basin dimension must be 64'
                    ))

        except Exception as e:
            print(f"⚠️  Error validating YAML {rel_path}: {e}")

    def print_report(self):
        """Print validation report."""
        if not self.violations:
            print("✅ No physics violations found!")
            return

        print(f"\n{'='*70}")
        print("PHYSICS VALIDATION REPORT")
        print(f"{'='*70}\n")

        # Group by type
        by_type: dict[str, list[PhysicsViolation]] = {}
        for v in self.violations:
            if v.violation_type not in by_type:
                by_type[v.violation_type] = []
            by_type[v.violation_type].append(v)

        for vtype, violations in sorted(by_type.items()):
            print(f"❌ {vtype.upper().replace('_', ' ')} ({len(violations)}):\n")
            for v in violations[:5]:  # Show first 5 per type
                print(f"  File: {v.file_path}:{v.line_number}")
                print(f"  Found: {v.found_value}")
                print(f"  Expected: {v.expected_value}")
                print(f"  Issue: {v.description}")
                print()

            if len(violations) > 5:
                print(f"  ... and {len(violations) - 5} more\n")

        print(f"{'='*70}")
        print(f"Total: {len(self.violations)} violations")
        print(f"{'='*70}\n")


def main():
    """Run physics validation."""
    import sys

    repo_root = Path.cwd()
    validator = PhysicsValidator(repo_root)

    print("🔍 Scanning for physics violations...")
    violations = validator.validate_all()

    validator.print_report()

    # Exit with error if violations found
    if violations:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
