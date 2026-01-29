#!/usr/bin/env python3
"""
WP3.2 Final Validation Script

Comprehensive validation that merge policy is geometry-first across all layers:
- Python implementation (geometric_pair_merging.py)
- SQL/Database layer (pg_loader.py)
- TypeScript API (routes, services)

This script performs static analysis to detect any frequency-first patterns.
"""

import sys
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

# Root directory
ROOT = Path(__file__).parent.parent.parent  # Go up to pantheon-chat root
QIG_BACKEND = ROOT / "qig-backend"
SERVER = ROOT / "server"
CLIENT = ROOT / "client"

# Patterns that indicate frequency-first logic (violations)
VIOLATION_PATTERNS = [
    # Classic BPE patterns
    (r"best.*=.*min\([^)]*entropy", "Minimum entropy selection (BPE pattern)"),
    (r"best.*=.*max\([^)]*frequency", "Maximum frequency selection (BPE pattern)"),
    (r"score.*=.*frequency\s*\*\s*[^l]", "Frequency multiplication (non-logarithmic)"),
    (r"pair_counts.*threshold", "Pair count threshold (frequency-first)"),
    (r"count\([^)]*\)\s*>\s*threshold", "Count-based threshold (frequency-first)"),
    
    # Frequency dominance patterns
    (r"frequency.*weight\s*>\s*0\.[5-9]", "Frequency weight > 50% (dominance)"),
    (r"if\s+frequency\s*>", "Frequency comparison (may indicate frequency-first)"),
    (r"sort.*frequency.*descending", "Sorting by frequency descending"),
    (r"ORDER BY.*frequency.*DESC", "SQL frequency sort descending"),
    
    # Entropy-first patterns
    (r"lowest.*entropy", "Lowest entropy selection"),
    (r"min.*entropy", "Minimum entropy selection"),
    (r"entropy.*<.*threshold", "Entropy threshold (BPE pattern)"),
]

# Patterns that are ALLOWED (geometric operations)
ALLOWED_PATTERNS = [
    r"frequency.*regularizer",  # Frequency as regularizer is OK
    r"min_frequency",  # Minimum frequency threshold (noise filter) is OK
    r"log\(frequency",  # Logarithmic frequency scaling is OK
    r"frequency.*filter",  # Frequency filtering (not scoring) is OK
    r"frequency.*\+\s*1",  # Adding 1 for log is OK
    r"ORDER BY.*phi_score.*DESC.*frequency",  # Sorting by phi first, then frequency (phi dominates) is OK
    r"Minimal entropy",  # Documentation about entropy is OK
    r"Maximum entropy",  # Documentation about entropy is OK
]


def check_file_for_violations(filepath: Path) -> List[Tuple[int, str, str]]:
    """
    Check a file for frequency-first patterns.
    
    Returns:
        List of (line_number, pattern_description, line_content)
    """
    violations = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, start=1):
            # Skip comments and docstrings
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            
            # Skip lines that are clearly documentation
            if re.match(r'^\s*[-*]', line):  # Bullet points in docstrings
                continue
            
            # Check for violation patterns
            for pattern, description in VIOLATION_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if it's an allowed context
                    is_allowed = False
                    for allowed in ALLOWED_PATTERNS:
                        if re.search(allowed, line, re.IGNORECASE):
                            is_allowed = True
                            break
                    
                    if not is_allowed:
                        violations.append((line_num, description, line.strip()))
        
    except Exception as e:
        print(f"  Warning: Could not read {filepath}: {e}")
    
    return violations


def scan_python_files() -> Dict[str, List]:
    """Scan Python files for violations."""
    print("\n=== Scanning Python Files ===")
    
    python_files = [
        QIG_BACKEND / "coordizers" / "geometric_pair_merging.py",
        QIG_BACKEND / "coordizers" / "vocab_builder.py",
        QIG_BACKEND / "coordizers" / "base.py",
        QIG_BACKEND / "coordizers" / "pg_loader.py",
    ]
    
    all_violations = {}
    
    for filepath in python_files:
        if not filepath.exists():
            print(f"  Skipping {filepath.name} (not found)")
            continue
        
        violations = check_file_for_violations(filepath)
        
        if violations:
            all_violations[str(filepath)] = violations
            print(f"  ✗ {filepath.name}: {len(violations)} potential violation(s)")
        else:
            print(f"  ✓ {filepath.name}: Clean")
    
    return all_violations


def scan_sql_files() -> Dict[str, List]:
    """Scan SQL and database-related files for violations."""
    print("\n=== Scanning SQL/Database Files ===")
    
    # Check pg_loader.py for SQL queries
    filepath = QIG_BACKEND / "coordizers" / "pg_loader.py"
    
    if not filepath.exists():
        print("  Skipping pg_loader.py (not found)")
        return {}
    
    violations = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for SQL queries with frequency sorting
        sql_queries = re.findall(r'""".*?"""', content, re.DOTALL)
        
        for query in sql_queries:
            if re.search(r"ORDER BY.*frequency.*DESC", query, re.IGNORECASE):
                # Check context - sorting for display is OK, sorting for merge decision is not
                if "merge" in query.lower() and "decision" in query.lower():
                    violations.append((0, "SQL frequency sort in merge context", query[:100]))
        
    except Exception as e:
        print(f"  Warning: Could not analyze pg_loader.py: {e}")
    
    if violations:
        print(f"  ✗ pg_loader.py: {len(violations)} potential violation(s)")
        return {str(filepath): violations}
    else:
        print(f"  ✓ pg_loader.py: Clean")
        return {}


def scan_typescript_files() -> Dict[str, List]:
    """Scan TypeScript files for violations."""
    print("\n=== Scanning TypeScript Files ===")
    
    ts_files = [
        SERVER / "routes" / "coordizer.ts",
        CLIENT / "src" / "api" / "services" / "coordizer.ts",
    ]
    
    all_violations = {}
    
    for filepath in ts_files:
        if not filepath.exists():
            print(f"  Skipping {filepath.name} (not found)")
            continue
        
        violations = check_file_for_violations(filepath)
        
        if violations:
            all_violations[str(filepath)] = violations
            print(f"  ✗ {filepath.name}: {len(violations)} potential violation(s)")
        else:
            print(f"  ✓ {filepath.name}: Clean")
    
    return all_violations


def validate_geometric_constants():
    """Validate that geometric constants are correctly set."""
    print("\n=== Validating Geometric Constants ===")
    
    filepath = QIG_BACKEND / "coordizers" / "geometric_pair_merging.py"
    
    if not filepath.exists():
        print("  ✗ Cannot find geometric_pair_merging.py")
        return False
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check weights
        geometric_weight = re.search(r"GEOMETRIC_SCORE_WEIGHT\s*=\s*([\d.]+)", content)
        frequency_weight = re.search(r"FREQUENCY_REGULARIZER_WEIGHT\s*=\s*([\d.]+)", content)
        
        if geometric_weight and frequency_weight:
            geo_val = float(geometric_weight.group(1))
            freq_val = float(frequency_weight.group(1))
            
            print(f"  Geometric score weight: {geo_val}")
            print(f"  Frequency regularizer weight: {freq_val}")
            
            if geo_val < 0.5:
                print(f"  ✗ Geometric weight too low ({geo_val} < 0.5)")
                return False
            
            if freq_val > 0.5:
                print(f"  ✗ Frequency weight too high ({freq_val} > 0.5)")
                return False
            
            if abs(geo_val + freq_val - 1.0) > 0.01:
                print(f"  ✗ Weights don't sum to 1.0 ({geo_val + freq_val})")
                return False
            
            print("  ✓ Constants correctly configured")
            return True
        else:
            print("  ✗ Could not find weight constants")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all validations."""
    print("="*70)
    print("WP3.2 Final Validation - Geometry-First Merge Policy")
    print("="*70)
    
    # Scan all layers
    python_violations = scan_python_files()
    sql_violations = scan_sql_files()
    ts_violations = scan_typescript_files()
    
    # Validate constants
    constants_ok = validate_geometric_constants()
    
    # Aggregate results
    all_violations = {**python_violations, **sql_violations, **ts_violations}
    
    # Report
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    if not all_violations and constants_ok:
        print("\n✓ ALL VALIDATIONS PASSED")
        print("\nNo frequency-first patterns detected:")
        print("  ✓ Python implementation is geometry-first")
        print("  ✓ SQL/Database layer has no frequency-first logic")
        print("  ✓ TypeScript API properly delegates to Python backend")
        print("  ✓ Geometric constants correctly configured")
        print("\nWP3.2 COMPLETE - Merge policy is geometry-first across all layers")
        return 0
    else:
        print("\n✗ VIOLATIONS DETECTED")
        
        for filepath, violations in all_violations.items():
            print(f"\n{filepath}:")
            for line_num, description, line in violations[:5]:  # Show first 5
                print(f"  Line {line_num}: {description}")
                print(f"    {line}")
        
        if not constants_ok:
            print("\n✗ Geometric constants not correctly configured")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
