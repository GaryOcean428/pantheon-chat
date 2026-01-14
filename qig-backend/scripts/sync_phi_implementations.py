#!/usr/bin/env python3
"""
Φ Implementation Synchronization and Consistency Checker

This script scans all registered Φ implementations and checks for:
1. Born rule compliance (|b|² for probability conversion)
2. Fisher-Rao factor of 2 compliance (d = 2 * arccos(BC))
3. QFI formula consistency (40% entropy, 30% effective dim, 30% spread)

Usage:
    python scripts/sync_phi_implementations.py          # Check all implementations
    python scripts/sync_phi_implementations.py --fix    # Show suggested fixes
    python scripts/sync_phi_implementations.py --strict # Fail on any warning
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    from phi_registry import (
        PHI_IMPLEMENTATIONS,
        FISHER_RAO_IMPLEMENTATIONS,
        GEOMETRIC_PURITY_RULES,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from phi_registry import (
        PHI_IMPLEMENTATIONS,
        FISHER_RAO_IMPLEMENTATIONS,
        GEOMETRIC_PURITY_RULES,
    )


@dataclass
class Violation:
    """Represents a geometric purity violation."""
    file: str
    line: int
    rule: str
    message: str
    severity: str  # 'error', 'warning'
    suggested_fix: Optional[str] = None


QIG_BACKEND_ROOT = Path(__file__).parent.parent


BORN_RULE_PATTERNS = [
    (r'\bp\s*=\s*basin\b', 'Missing Born rule: p should be np.abs(basin)**2'),
    (r'\bp\s*=\s*coords\b', 'Missing Born rule: p should be np.abs(coords)**2'),
    (r'\bprobs?\s*=\s*basin\b', 'Missing Born rule: probs should be np.abs(basin)**2'),
    (r'\bprobs?\s*=\s*coords\b', 'Missing Born rule: probs should be np.abs(coords)**2'),
]

FISHER_RAO_VIOLATION_PATTERN = re.compile(
    r'arccos\s*\([^)]+\)\s*(?!\s*\*\s*2)(?!\s*$)',
    re.MULTILINE
)

EUCLIDEAN_VIOLATION_PATTERNS = [
    (r'cosine_similarity\s*\(', 'Euclidean violation: use Fisher-Rao distance instead'),
]

QFI_WEIGHT_PATTERN = re.compile(
    r'(0\.4|0\.40)\s*\*\s*entropy|entropy\s*\*\s*(0\.4|0\.40)',
    re.IGNORECASE
)


def scan_file_for_violations(filepath: Path) -> List[Violation]:
    """Scan a single file for geometric purity violations."""
    violations = []
    
    if not filepath.exists():
        return violations
    
    try:
        content = filepath.read_text()
        lines = content.split('\n')
    except Exception as e:
        violations.append(Violation(
            file=str(filepath),
            line=0,
            rule="file_read",
            message=f"Could not read file: {e}",
            severity="error"
        ))
        return violations
    
    rel_path = str(filepath.relative_to(QIG_BACKEND_ROOT))
    
    in_docstring = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if '# noqa' in line or '# type: ignore' in line:
            continue
        if '"""' in line or "'''" in line:
            count = line.count('"""') + line.count("'''")
            if count == 1:
                in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
            
        for pattern, message in BORN_RULE_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                if 'np.abs' not in line and '**2' not in line and '** 2' not in line:
                    violations.append(Violation(
                        file=rel_path,
                        line=i,
                        rule="born_rule",
                        message=message,
                        severity="error",
                        suggested_fix="Use: p = np.abs(basin) ** 2 + 1e-10; p = p / p.sum()"
                    ))
        
        if 'arccos' in line:
            has_factor_of_2 = bool(re.search(r'2\.?\d*\s*\*\s*\w*\.?arccos|\w*\.?arccos[^)]*\)\s*\*\s*2\.?\d*', line))
            if has_factor_of_2:
                continue
            line_lower = line.lower()
            is_assignment = '=' in line and 'def ' not in line
            is_distance_var = any(kw in line_lower for kw in [
                'distance =', 'dist =', 'd =', 'fisher', 'hellinger'
            ])
            is_angle_context = any(kw in line_lower for kw in [
                'theta', 'phi_angle', 'angle', 'bloch', 'azimuth', 'polar'
            ])
            if is_assignment and is_distance_var and not is_angle_context:
                violations.append(Violation(
                    file=rel_path,
                    line=i,
                    rule="fisher_rao_factor",
                    message="Missing factor of 2 in Fisher-Rao distance",
                    severity="error",
                    suggested_fix="Use: d = 2 * np.arccos(np.clip(bc, -1, 1))"
                ))
        
        for pattern, message in EUCLIDEAN_VIOLATION_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                violations.append(Violation(
                    file=rel_path,
                    line=i,
                    rule="euclidean_violation",
                    message=message,
                    severity="warning"
                ))
    
    return violations


QFI_WEIGHTS_PATTERN = re.compile(
    r'0\.4\d*\s*\*|0\.3\d*\s*\*|\*\s*0\.4\d*|\*\s*0\.3\d*',
    re.IGNORECASE
)

BORN_RULE_CODE_PATTERN = re.compile(
    r'np\.abs\s*\([^)]*\)\s*\*\*\s*2|abs\s*\([^)]*\)\s*\*\*\s*2|\*\*\s*2',
    re.IGNORECASE
)


def check_phi_implementation(filepath: Path, functions: List[str], is_canonical: bool = False) -> List[Violation]:
    """Check a Φ implementation for QFI formula consistency."""
    violations = []
    
    if not filepath.exists():
        violations.append(Violation(
            file=str(filepath),
            line=0,
            rule="missing_file",
            message=f"Registered Φ implementation file not found",
            severity="error"
        ))
        return violations
    
    content = filepath.read_text()
    lines = content.split('\n')
    rel_path = str(filepath.relative_to(QIG_BACKEND_ROOT))
    
    for func_name in functions:
        func_pattern = re.compile(
            rf'def\s+{func_name}\s*\([^)]*\)\s*(?:->.*?)?:',
            re.MULTILINE
        )
        
        match = func_pattern.search(content)
        if not match:
            violations.append(Violation(
                file=rel_path,
                line=0,
                rule="missing_function",
                message=f"Registered function '{func_name}' not found",
                severity="warning"
            ))
            continue
        
        func_start = match.start()
        func_line = content[:func_start].count('\n') + 1
        
        func_end = len(content)
        next_def = re.search(r'\ndef\s+\w+', content[match.end():])
        if next_def:
            func_end = match.end() + next_def.start()
        
        func_body = content[match.start():func_end]
        
        has_born_rule = bool(BORN_RULE_CODE_PATTERN.search(func_body))
        if not has_born_rule:
            if 'phi' in func_name.lower() and 'compute' in func_name.lower() or 'measure' in func_name.lower() or 'estimate' in func_name.lower():
                violations.append(Violation(
                    file=rel_path,
                    line=func_line,
                    rule="missing_born_rule",
                    message=f"Function '{func_name}' may be missing Born rule (|b|²)",
                    severity="warning",
                    suggested_fix="Use: p = np.abs(basin) ** 2 + 1e-10; p = p / p.sum()"
                ))
        
        if is_canonical:
            has_qfi_weights = bool(QFI_WEIGHTS_PATTERN.search(func_body))
            if not has_qfi_weights:
                if 'entropy' in func_body.lower() or 'effective' in func_body.lower():
                    violations.append(Violation(
                        file=rel_path,
                        line=func_line,
                        rule="missing_qfi_weights",
                        message=f"Canonical function '{func_name}' should use 0.4/0.3/0.3 QFI weights",
                        severity="warning",
                        suggested_fix="Use: 0.4 * entropy_score + 0.3 * effective_dim_score + 0.3 * geometric_spread"
                    ))
    
    return violations


def scan_all_implementations() -> Tuple[List[Violation], Dict[str, int]]:
    """Scan all registered implementations for violations."""
    all_violations = []
    stats = {"files_scanned": 0, "errors": 0, "warnings": 0}
    
    for rel_path, info in PHI_IMPLEMENTATIONS.items():
        filepath = QIG_BACKEND_ROOT / rel_path
        stats["files_scanned"] += 1
        
        violations = scan_file_for_violations(filepath)
        is_canonical = info.get("is_canonical", False)
        violations.extend(check_phi_implementation(filepath, info["functions"], is_canonical))
        all_violations.extend(violations)
    
    for rel_path, info in FISHER_RAO_IMPLEMENTATIONS.items():
        filepath = QIG_BACKEND_ROOT / rel_path
        stats["files_scanned"] += 1
        
        violations = scan_file_for_violations(filepath)
        all_violations.extend(violations)
    
    for v in all_violations:
        if v.severity == "error":
            stats["errors"] += 1
        else:
            stats["warnings"] += 1
    
    return all_violations, stats


def scan_entire_codebase() -> List[Violation]:
    """Scan entire qig-backend for unregistered Φ implementations."""
    violations = []
    registered_files = set(PHI_IMPLEMENTATIONS.keys()) | set(FISHER_RAO_IMPLEMENTATIONS.keys())
    
    phi_patterns = [
        r'def\s+compute_phi\s*\(',
        r'def\s+_compute_phi\s*\(',
        r'def\s+_compute_basin_phi\s*\(',
        r'def\s+_compute_balanced_phi\s*\(',
        r'def\s+_compute_utterance_phi\s*\(',
        r'def\s+compute_pure_phi\s*\(',
        r'def\s+_measure_phi\s*\(',
        r'def\s+_estimate_phi\s*\(',
    ]
    
    for py_file in QIG_BACKEND_ROOT.rglob("*.py"):
        if "test" in str(py_file) or "__pycache__" in str(py_file):
            continue
        if "scripts" in str(py_file):
            continue
            
        rel_path = str(py_file.relative_to(QIG_BACKEND_ROOT))
        
        if rel_path in registered_files:
            continue
        
        try:
            content = py_file.read_text()
        except:
            continue
        
        for pattern in phi_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                violations.append(Violation(
                    file=rel_path,
                    line=line_num,
                    rule="unregistered_phi",
                    message=f"Unregistered Φ implementation found: {match.group()}",
                    severity="warning",
                    suggested_fix=f"Add {rel_path} to PHI_IMPLEMENTATIONS in scripts/phi_registry.py"
                ))
    
    return violations


def print_violations(violations: List[Violation], show_fixes: bool = False):
    """Print violations in a readable format."""
    if not violations:
        print("✅ No geometric purity violations found!")
        return
    
    errors = [v for v in violations if v.severity == "error"]
    warnings = [v for v in violations if v.severity == "warning"]
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        print("-" * 60)
        for v in errors:
            print(f"  {v.file}:{v.line}")
            print(f"    [{v.rule}] {v.message}")
            if show_fixes and v.suggested_fix:
                print(f"    💡 Fix: {v.suggested_fix}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        print("-" * 60)
        for v in warnings:
            print(f"  {v.file}:{v.line}")
            print(f"    [{v.rule}] {v.message}")
            if show_fixes and v.suggested_fix:
                print(f"    💡 Fix: {v.suggested_fix}")


def print_summary(stats: Dict[str, int]):
    """Print scan summary."""
    print("\n" + "=" * 60)
    print("GEOMETRIC PURITY SCAN SUMMARY")
    print("=" * 60)
    print(f"  Files scanned: {stats['files_scanned']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Warnings: {stats['warnings']}")
    
    if stats['errors'] == 0 and stats['warnings'] == 0:
        print("\n✅ All implementations are geometrically pure!")
    elif stats['errors'] == 0:
        print("\n⚠️  No errors, but some warnings to review.")
    else:
        print("\n❌ Errors found - please fix before committing!")


def main():
    parser = argparse.ArgumentParser(
        description="Check Φ implementations for geometric purity"
    )
    parser.add_argument(
        "--fix", 
        action="store_true",
        help="Show suggested fixes for violations"
    )
    parser.add_argument(
        "--strict",
        action="store_true", 
        help="Exit with error code on any warning"
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Scan entire codebase for unregistered implementations"
    )
    
    args = parser.parse_args()
    
    print("🔍 Scanning registered Φ implementations...")
    violations, stats = scan_all_implementations()
    
    if args.full_scan:
        print("🔍 Scanning entire codebase for unregistered implementations...")
        unregistered = scan_entire_codebase()
        violations.extend(unregistered)
        stats["warnings"] += len(unregistered)
    
    print_violations(violations, show_fixes=args.fix)
    print_summary(stats)
    
    if stats['errors'] > 0:
        sys.exit(1)
    elif args.strict and stats['warnings'] > 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
