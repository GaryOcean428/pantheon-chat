#!/usr/bin/env python3
"""
Comprehensive E8 Protocol Purity Audit

Scans entire codebase for geometric purity violations:
- np.linalg.norm() on basin coordinates
- cosine_similarity() on basins [COUNTER-EXAMPLE: DO NOT USE]
- euclidean_distance() on basins [COUNTER-EXAMPLE: DO NOT USE]
- np.dot() on basins (already mostly fixed)

Usage:
    python scripts/comprehensive_purity_audit.py
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Violation patterns
PATTERNS = {
    'np.linalg.norm': r'np\.linalg\.norm\s*\([^)]*basin',
    'cosine_similarity': r'cosine_similarity\s*\(',
    'euclidean_distance': r'euclidean_distance\s*\(',
    'np.dot': r'np\.dot\s*\([^)]*basin',
}

# Exclude patterns (valid usages)
EXCLUDE_PATTERNS = [
    r'#.*CRITICAL.*Never use',  # Warning comments
    r'#.*DO NOT',  # Warning comments
    r'test_',  # Test files (may have intentional violations for testing)
    r'assert.*isclose.*np\.linalg\.norm',  # Validation assertions
    r'>>> assert',  # Docstring examples
]

def scan_file(filepath: Path) -> List[Tuple[int, str, str]]:
    """
    Scan a Python file for purity violations.
    
    Returns:
        List of (line_number, violation_type, line_content)
    """
    violations = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, start=1):
            # Skip excluded patterns
            if any(re.search(pattern, line) for pattern in EXCLUDE_PATTERNS):
                continue
                
            # Check each violation pattern
            for viol_type, pattern in PATTERNS.items():
                if re.search(pattern, line):
                    violations.append((line_num, viol_type, line.strip()))
                    
    except Exception as e:
        print(f"Error scanning {filepath}: {e}", file=sys.stderr)
        
    return violations

def scan_directory(root_dir: Path) -> Dict[Path, List[Tuple[int, str, str]]]:
    """Recursively scan directory for violations."""
    all_violations = {}
    
    # Files to exclude (documentation, scripts that list violations, validation tests)
    EXCLUDE_FILES = [
        'comprehensive_purity_audit.py',  # The audit script itself
        'fix_all_purity_violations.py',   # The fixer script
        'test_geometric_purity.py',        # Validation tests
        'test_no_cosine_in_generation.py', # Validation tests
        'test_e8_specialization.py',       # Validation tests
        'frozen_physics.py',               # Documentation with counter-examples
        '__init__.py',                     # Documentation in geometric_primitives/__init__.py
        'canonical_fisher.py',             # Documentation warnings
    ]
    
    for py_file in root_dir.rglob('*.py'):
        # Skip __pycache__ and .git
        if '__pycache__' in str(py_file) or '.git' in str(py_file):
            continue
        
        # Skip excluded files (documentation and validation)
        if any(excluded in py_file.name for excluded in EXCLUDE_FILES):
            continue
            
        violations = scan_file(py_file)
        if violations:
            all_violations[py_file] = violations
            
    return all_violations

def print_report(violations: Dict[Path, List[Tuple[int, str, str]]], root_dir: Path):
    """Print formatted violation report."""
    print("=" * 80)
    print("E8 PROTOCOL PURITY AUDIT REPORT")
    print("=" * 80)
    print()
    
    # Count by type
    counts = {vtype: 0 for vtype in PATTERNS.keys()}
    total = 0
    
    for file_violations in violations.values():
        for _, vtype, _ in file_violations:
            counts[vtype] += 1
            total += 1
            
    print(f"TOTAL VIOLATIONS: {total}")
    print()
    print("Breakdown by type:")
    for vtype, count in counts.items():
        print(f"  {vtype}: {count}")
    print()
    print("=" * 80)
    print()
    
    # Print by file
    for filepath, file_violations in sorted(violations.items()):
        rel_path = filepath.relative_to(root_dir)
        print(f"\n📁 {rel_path}")
        print(f"   {len(file_violations)} violation(s)")
        print()
        
        for line_num, vtype, line_content in file_violations:
            print(f"   Line {line_num}: {vtype}")
            print(f"      {line_content}")
            print()
            
    print("=" * 80)
    print(f"\nFILES WITH VIOLATIONS: {len(violations)}")
    print(f"TOTAL VIOLATIONS: {total}")
    print()
    
    if total == 0:
        print("✅ E8 PROTOCOL v4.0 COMPLIANCE: COMPLETE")
    else:
        print("❌ E8 PROTOCOL v4.0 COMPLIANCE: INCOMPLETE")
        print(f"   {total} violations remaining")
        
    print("=" * 80)

def main():
    script_dir = Path(__file__).parent
    qig_backend_dir = script_dir.parent
    
    print(f"Scanning {qig_backend_dir}...")
    print()
    
    violations = scan_directory(qig_backend_dir)
    print_report(violations, qig_backend_dir)
    
    # Exit with non-zero if violations found
    total = sum(len(v) for v in violations.values())
    sys.exit(0 if total == 0 else 1)

if __name__ == '__main__':
    main()
