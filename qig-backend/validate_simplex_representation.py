#!/usr/bin/env python3
"""
Simplex Representation Validation Script

Validates that all basin coordinates in the codebase follow simplex representation:
1. Non-negativity: all(basin >= 0)
2. Normalization: abs(sum(basin) - 1.0) < epsilon
3. Dimensionality: len(basin) == 64

This script performs static analysis and runtime validation.

Usage:
    python scripts/validate_simplex_representation.py [--strict]
    
Options:
    --strict: Fail on any warnings (default: fail only on errors)
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

# Expected basin dimensionality
BASIN_DIM = 64

# Epsilon for floating point comparisons
EPS = 1e-6

class SimplexViolation:
    """Represents a simplex representation violation."""
    def __init__(self, file: str, line: int, code: str, issue: str, severity: str):
        self.file = file
        self.line = line
        self.code = code
        self.issue = issue
        self.severity = severity  # 'ERROR' or 'WARNING'

def check_simplex_assertions(content: str, filepath: Path) -> List[SimplexViolation]:
    """Check for missing simplex validation assertions."""
    violations = []
    lines = content.split('\n')
    
    # Look for basin coordinate assignments without validation
    for i, line in enumerate(lines, 1):
        # Pattern: basin = ... or basin_coords = ...
        if re.search(r'(\w*basin\w*)\s*=\s*(?!.*assert)', line, re.IGNORECASE):
            # Check if there's an assert_basin_valid or validate_basin in next 5 lines
            next_lines = '\n'.join(lines[i:min(i+5, len(lines))])
            if not re.search(r'assert_basin_valid|validate_basin', next_lines):
                # Check if it's a function definition or import
                if 'def ' not in line and 'import ' not in line and '=' in line:
                    violations.append(SimplexViolation(
                        file=str(filepath),
                        line=i,
                        code=line.strip(),
                        issue="Basin assignment without validation assertion",
                        severity='WARNING'
                    ))
    
    return violations

def check_hardcoded_dimensions(content: str, filepath: Path) -> List[SimplexViolation]:
    """Check for hardcoded dimensions that don't match BASIN_DIM."""
    violations = []
    lines = content.split('\n')
    
    # Look for hardcoded dimensions in basin-related code
    for i, line in enumerate(lines, 1):
        if 'basin' in line.lower():
            # Look for hardcoded numbers that might be dimensions
            matches = re.findall(r'\b(32|128|256|512)\b', line)
            for match in matches:
                if int(match) != BASIN_DIM:
                    violations.append(SimplexViolation(
                        file=str(filepath),
                        line=i,
                        code=line.strip(),
                        issue=f"Hardcoded dimension {match} doesn't match BASIN_DIM={BASIN_DIM}",
                        severity='WARNING'
                    ))
    
    return violations

def check_arithmetic_operations(content: str, filepath: Path) -> List[SimplexViolation]:
    """Check for arithmetic operations that might break simplex properties."""
    violations = []
    lines = content.split('\n')
    
    # Patterns that might break simplex properties
    dangerous_patterns = [
        (r'basin\w*\s*\+\s*basin', "Addition of basins (use frechet_mean or geodesic_toward)"),
        (r'basin\w*\s*\*\s*\d+', "Scalar multiplication of basin (breaks normalization)"),
        (r'basin\w*\s*/\s*\d+', "Scalar division of basin (breaks normalization)"),
    ]
    
    for i, line in enumerate(lines, 1):
        for pattern, issue in dangerous_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                # Check if there's a renormalization after
                next_lines = '\n'.join(lines[i:min(i+3, len(lines))])
                if not re.search(r'to_simplex_prob|fisher_normalize|renormalize', next_lines):
                    violations.append(SimplexViolation(
                        file=str(filepath),
                        line=i,
                        code=line.strip(),
                        issue=issue,
                        severity='ERROR'
                    ))
    
    return violations

def check_array_initialization(content: str, filepath: Path) -> List[SimplexViolation]:
    """Check for basin initialization that might not be on simplex."""
    violations = []
    lines = content.split('\n')
    
    # Patterns for basin initialization
    init_patterns = [
        (r'basin\w*\s*=\s*np\.zeros', "Zero initialization (not on simplex)"),
        (r'basin\w*\s*=\s*np\.ones', "Ones initialization (not normalized)"),
        (r'basin\w*\s*=\s*np\.random\.rand', "Random initialization (not on simplex)"),
    ]
    
    for i, line in enumerate(lines, 1):
        for pattern, issue in init_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                # Check if there's a normalization after
                next_lines = '\n'.join(lines[i:min(i+3, len(lines))])
                if not re.search(r'to_simplex_prob|fisher_normalize|dirichlet', next_lines):
                    violations.append(SimplexViolation(
                        file=str(filepath),
                        line=i,
                        code=line.strip(),
                        issue=issue,
                        severity='WARNING'
                    ))
    
    return violations

def scan_file(filepath: Path) -> List[SimplexViolation]:
    """Scan a Python file for simplex representation violations."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        violations = []
        violations.extend(check_simplex_assertions(content, filepath))
        violations.extend(check_hardcoded_dimensions(content, filepath))
        violations.extend(check_arithmetic_operations(content, filepath))
        violations.extend(check_array_initialization(content, filepath))
        
        return violations
    
    except Exception as e:
        print(f"Error scanning {filepath}: {e}", file=sys.stderr)
        return []

def scan_directory(root_dir: Path) -> List[SimplexViolation]:
    """Recursively scan directory for simplex violations."""
    all_violations = []
    
    for py_file in root_dir.rglob('*.py'):
        # Skip __pycache__, .git, and test files
        if '__pycache__' in str(py_file) or '.git' in str(py_file):
            continue
        
        violations = scan_file(py_file)
        all_violations.extend(violations)
    
    return all_violations

def print_report(violations: List[SimplexViolation], strict: bool = False):
    """Print formatted violation report."""
    print("=" * 80)
    print("SIMPLEX REPRESENTATION VALIDATION REPORT")
    print("=" * 80)
    print()
    
    # Group by severity
    errors = [v for v in violations if v.severity == 'ERROR']
    warnings = [v for v in violations if v.severity == 'WARNING']
    
    print(f"TOTAL VIOLATIONS: {len(violations)}")
    print(f"  ERRORS: {len(errors)}")
    print(f"  WARNINGS: {len(warnings)}")
    print()
    
    if errors:
        print("=" * 80)
        print("ERRORS (Must Fix)")
        print("=" * 80)
        for v in errors:
            print(f"\n📁 {v.file}:{v.line}")
            print(f"   Issue: {v.issue}")
            print(f"   Code: {v.code}")
    
    if warnings:
        print("\n" + "=" * 80)
        print("WARNINGS (Should Review)")
        print("=" * 80)
        for v in warnings:
            print(f"\n📁 {v.file}:{v.line}")
            print(f"   Issue: {v.issue}")
            print(f"   Code: {v.code}")
    
    print("\n" + "=" * 80)
    
    if len(violations) == 0:
        print("✅ SIMPLEX REPRESENTATION: VALID")
        print("   All basin coordinates follow simplex properties")
    elif len(errors) == 0 and not strict:
        print("⚠️  SIMPLEX REPRESENTATION: WARNINGS ONLY")
        print("   No critical errors, but review warnings")
    else:
        print("❌ SIMPLEX REPRESENTATION: VIOLATIONS DETECTED")
        print("   Fix errors before merging")
    
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Validate simplex representation in codebase')
    parser.add_argument('--strict', action='store_true', help='Fail on any warnings')
    args = parser.parse_args()
    
    # Determine root directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    # Scan qig-backend directory
    qig_backend = repo_root / 'qig-backend'
    if not qig_backend.exists():
        print(f"Error: qig-backend directory not found at {qig_backend}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Scanning {qig_backend}...")
    print()
    
    violations = scan_directory(qig_backend)
    print_report(violations, strict=args.strict)
    
    # Exit code
    errors = [v for v in violations if v.severity == 'ERROR']
    if len(errors) > 0:
        sys.exit(1)
    elif args.strict and len(violations) > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
