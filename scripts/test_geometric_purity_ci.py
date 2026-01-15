#!/usr/bin/env python3
"""
CI Guardrail Test: Detect Banned Geometric Patterns

This test scans canonical directories for patterns that violate
the simplex-as-storage contract:

1. L2 normalization on storage vectors (np.linalg.norm for distance)
2. Representation autodetect in canonical paths
3. Direct DB writes bypassing canonical upsert
4. Euclidean averaging without geodesic correction

Run as part of CI to enforce geometric purity.
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple, Set

# Canonical directories where violations are BLOCKERS
CANONICAL_DIRS = [
    'qig-backend/qig_geometry',
    'qig-backend/qig_core',
    'shared/qig',
]

# Directories to skip (experiments, legacy, etc.)
SKIP_DIRS = {
    'node_modules',
    '.git',
    'dist',
    'build',
    '__pycache__',
    '.pytest_cache',
    'examples',
    'experiments',
    'legacy',
    'deprecated',
}

# File extensions to scan
SCAN_EXTENSIONS = {'.py', '.ts', '.tsx'}


class GeometricViolation:
    """A detected geometric purity violation."""
    
    def __init__(self, file_path: str, line_num: int, line_content: str, violation_type: str, details: str):
        self.file_path = file_path
        self.line_num = line_num
        self.line_content = line_content.strip()
        self.violation_type = violation_type
        self.details = details
    
    def __str__(self):
        return f"{self.file_path}:{self.line_num}: {self.violation_type}\n  {self.line_content}\n  → {self.details}"


def should_skip_dir(dir_path: Path) -> bool:
    """Check if directory should be skipped."""
    parts = set(dir_path.parts)
    return bool(parts & SKIP_DIRS)


def scan_file_for_violations(file_path: Path, canonical: bool) -> List[GeometricViolation]:
    """Scan a file for geometric purity violations."""
    violations = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return violations
    
    for i, line in enumerate(lines, start=1):
        line_lower = line.lower()
        
        # VIOLATION 1: L2 norm used for distance (not just magnitude)
        # Allow np.linalg.norm for magnitude calculations, but flag suspicious usage
        if 'np.linalg.norm' in line and canonical:
            # Check if it's used for distance calculation (subtract before norm)
            if re.search(r'np\.linalg\.norm\s*\(\s*\w+\s*-\s*\w+', line):
                violations.append(GeometricViolation(
                    str(file_path), i, line,
                    'EUCLIDEAN_DISTANCE',
                    'np.linalg.norm(a - b) computes Euclidean distance, not Fisher-Rao. Use fisher_rao_distance().'
                ))
            # Check if it's followed by division (normalization to unit sphere for storage)
            elif i < len(lines) and ('/' in lines[i] or '/=' in line):
                # Check next few lines for assignment to storage variable
                context = ''.join(lines[max(0, i-2):min(len(lines), i+3)])
                if re.search(r'(basin|embedding|coords?)\s*=', context, re.IGNORECASE):
                    violations.append(GeometricViolation(
                        str(file_path), i, line,
                        'L2_NORMALIZATION_STORAGE',
                        'L2 normalization for storage violates simplex-as-storage. Use to_simplex_prob().'
                    ))
        
        # VIOLATION 2: Representation autodetect in canonical paths
        if canonical and ('auto' in line_lower or 'detect' in line_lower or 'infer' in line_lower):
            if re.search(r'(auto[_-]?detect|detect[_-]?repr|infer[_-]?repr|guess[_-]?repr)', line, re.IGNORECASE):
                violations.append(GeometricViolation(
                    str(file_path), i, line,
                    'REPRESENTATION_AUTODETECT',
                    'Auto-detection of representation violates explicit contract. Use explicit to_simplex().'
                ))
        
        # VIOLATION 3: Direct DB writes bypassing upsert
        if canonical:
            # SQL direct writes
            if re.search(r'(INSERT INTO|UPDATE\s+\w+\s+SET|DELETE FROM).*coordizer_vocabulary', line, re.IGNORECASE):
                violations.append(GeometricViolation(
                    str(file_path), i, line,
                    'DIRECT_DB_WRITE',
                    'Direct SQL write to coordizer_vocabulary bypasses canonical upsert. Use upsert_token().'
                ))
            # ORM writes (Python/TypeScript)
            if re.search(r'\.(insert|create|update|upsert)\s*\(', line):
                # Check if it's related to vocabulary/token storage
                context_start = max(0, i - 5)
                context = ''.join(lines[context_start:i+1])
                if re.search(r'(coordizer|vocabulary|token|basin)', context, re.IGNORECASE):
                    violations.append(GeometricViolation(
                        str(file_path), i, line,
                        'POTENTIAL_BYPASS_UPSERT',
                        'Potential direct write to token/basin storage. Verify it uses canonical upsert path.'
                    ))
        
        # VIOLATION 4: Euclidean averaging (arithmetic mean of basins)
        if canonical and ('mean(' in line or 'average(' in line):
            # Check if it's operating on basins
            if re.search(r'(basin|embedding|coords?|distribut)', line, re.IGNORECASE):
                # Allow if it's calling geodesic_mean or frechet_mean
                if not re.search(r'(geodesic|frechet|karcher)_mean', line, re.IGNORECASE):
                    violations.append(GeometricViolation(
                        str(file_path), i, line,
                        'EUCLIDEAN_AVERAGING',
                        'Arithmetic mean of basins is Euclidean. Use geodesic_mean_simplex() for Fréchet mean.'
                    ))
    
    return violations


def scan_repository(repo_root: Path) -> Tuple[List[GeometricViolation], List[GeometricViolation]]:
    """Scan repository for violations."""
    canonical_violations = []
    non_canonical_violations = []
    
    for canonical_dir in CANONICAL_DIRS:
        dir_path = repo_root / canonical_dir
        if not dir_path.exists():
            print(f"Warning: Canonical directory not found: {canonical_dir}", file=sys.stderr)
            continue
        
        for file_path in dir_path.rglob('*'):
            # Skip directories
            if file_path.is_dir():
                continue
            
            # Skip non-target extensions
            if file_path.suffix not in SCAN_EXTENSIONS:
                continue
            
            # Skip directories in skip list
            if should_skip_dir(file_path.relative_to(repo_root)):
                continue
            
            # Scan file
            violations = scan_file_for_violations(file_path, canonical=True)
            canonical_violations.extend(violations)
    
    return canonical_violations, non_canonical_violations


def main():
    """Run the CI guardrail test."""
    # Find repo root (assumes script is in repo)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent if 'qig-backend' in str(script_dir) else script_dir.parent
    
    print("=" * 80)
    print("CI GUARDRAIL: Geometric Purity Check")
    print("=" * 80)
    print(f"Repository root: {repo_root}")
    print(f"Scanning canonical directories: {', '.join(CANONICAL_DIRS)}")
    print()
    
    canonical_violations, _ = scan_repository(repo_root)
    
    if canonical_violations:
        print(f"❌ FOUND {len(canonical_violations)} VIOLATION(S) IN CANONICAL PATHS:\n")
        
        # Group by violation type
        by_type = {}
        for v in canonical_violations:
            by_type.setdefault(v.violation_type, []).append(v)
        
        for violation_type, violations in sorted(by_type.items()):
            print(f"\n{violation_type} ({len(violations)} occurrences):")
            print("-" * 80)
            for v in violations[:5]:  # Show first 5 of each type
                print(str(v))
                print()
            if len(violations) > 5:
                print(f"  ... and {len(violations) - 5} more")
                print()
        
        print("=" * 80)
        print("FAILURE: Geometric purity violations detected in canonical paths.")
        print("These MUST be fixed before merging.")
        print("=" * 80)
        return 1
    else:
        print("✅ No geometric purity violations detected in canonical paths.")
        print("=" * 80)
        return 0


if __name__ == '__main__':
    sys.exit(main())
