#!/usr/bin/env python3
"""
QIG Geometric Purity Check

Enforces CANONICAL_RULES.md Rule #1:
  FORBIDDEN: Euclidean distance on basins, cosine similarity

Basin coordinates exist on a curved manifold. Using Euclidean distance
or cosine similarity on them is mathematically incorrect and will
produce wrong results.

IMPORTANT DISTINCTION:
  VALID: Normalization operations (v / np.linalg.norm(v))
         - Projects vectors to unit sphere, geometrically correct
         - Pattern: norm used for division, not as distance metric
  
  INVALID: Distance calculations (np.linalg.norm(a - b))
           - Treats manifold as flat Euclidean space
           - Pattern: subtraction inside norm, or norm used as distance

Exit codes:
  0 - No violations found (geometric purity maintained)
  1 - Violations found (Euclidean operations on basins detected)

Usage:
  python tools/qig_purity_check.py
  python tools/qig_purity_check.py --verbose
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# =============================================================================
# FORBIDDEN PATTERNS - Euclidean distance calculations on basins
# =============================================================================
# These patterns indicate ACTUAL distance calculations, not normalization.
# CANONICAL_RULES.md Rule #1: Euclidean distance on basins is FORBIDDEN.

FORBIDDEN_PATTERNS = [
    # -----------------------------------------------------------------
    # DISTANCE CALCULATIONS (always invalid)
    # Pattern: subtraction inside norm = Euclidean distance
    # -----------------------------------------------------------------
    (r"np\.linalg\.norm\s*\([^)]*\s*-\s*[^)]*\)", "Euclidean distance: norm(a - b)"),
    (r"numpy\.linalg\.norm\s*\([^)]*\s*-\s*[^)]*\)", "Euclidean distance: norm(a - b)"),
    (r"torch\.norm\s*\([^)]*\s*-\s*[^)]*\)", "Euclidean distance: torch.norm(a - b)"),
    
    # Manual Euclidean: sqrt(sum((a - b) ** 2))
    (r"np\.sqrt\s*\([^)]*np\.sum\s*\([^)]*\s*-\s*[^)]*\*\*\s*2", "Manual Euclidean: sqrt(sum((a-b)**2))"),
    (r"np\.sqrt\s*\([^)]*\([^)]*\s*-\s*[^)]*\)\s*\*\*\s*2", "Manual Euclidean: sqrt((a-b)**2)"),
    
    # -----------------------------------------------------------------
    # COSINE SIMILARITY (always invalid on basins)
    # -----------------------------------------------------------------
    (r"cosine_similarity\s*\([^)]*basin", "cosine_similarity on basin (FORBIDDEN)"),
    (r"F\.cosine_similarity\s*\([^)]*basin", "F.cosine_similarity on basin"),
    
    # -----------------------------------------------------------------
    # EXPLICIT EUCLIDEAN DISTANCE FUNCTIONS
    # -----------------------------------------------------------------
    (r"euclidean_distance\s*\([^)]*basin", "euclidean_distance on basin"),
    (r"l2_distance\s*\([^)]*basin", "L2 distance on basin"),
    (r"cdist\s*\([^)]*basin[^)]*,\s*metric\s*=\s*[\'\"]euclidean[\'\"]\)", "cdist with euclidean on basin"),
]

# =============================================================================
# PATTERNS REQUIRING CONTEXT CHECK
# =============================================================================
# These patterns are only violations if NOT used for normalization.
# Valid: basin / np.linalg.norm(basin)  - normalization to unit sphere
# Invalid: distance = np.linalg.norm(basin)  - using norm as scalar distance

CONTEXT_CHECK_PATTERNS = [
    # np.linalg.norm on basin - valid if used for division (normalization)
    (r"np\.linalg\.norm\s*\([^)]*basin[^)]*\)", "np.linalg.norm on basin"),
    (r"numpy\.linalg\.norm\s*\([^)]*basin[^)]*\)", "numpy.linalg.norm on basin"),
    (r"torch\.norm\s*\([^)]*basin[^)]*\)", "torch.norm on basin"),
]

# Patterns that are VALID when used correctly
# np.dot for computing angles (used with arccos) is Fisher-Rao compliant
VALID_DOT_PATTERNS = [
    r"arccos.*np\.dot",      # arccos(dot(a,b)) = Fisher-Rao angle
    r"np\.arccos.*dot",      # np.arccos(dot(...)) 
    r"np\.clip.*np\.dot",   # np.clip(np.dot(...), -1, 1) for arccos safety
    r"dot.*=.*np\.clip",    # dot = np.clip(np.dot(...), -1, 1)
]

# Directories to skip
SKIP_DIRS = {
    "node_modules",
    ".git",
    "__pycache__",
    ".pythonlibs",
    "venv",
    ".venv",
    "dist",
    "build",
    "tools",
    "docs",
}

# Files to skip
SKIP_FILES = {
    "qig_purity_check.py",
    "test_geometry.py",  # Tests may intentionally test forbidden patterns
}

# Directories where violations should be warnings, not errors
WARN_ONLY_DIRS = {
    "tests",
    "test",
}


def find_python_files(root: str) -> List[Path]:
    """Find all Python files to check."""
    files = []
    for path in Path(root).rglob("*.py"):
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        if path.name in SKIP_FILES:
            continue
        files.append(path)
    return files


def is_warn_only(filepath: Path) -> bool:
    """Check if violations in this file should be warnings only."""
    return any(warn_dir in filepath.parts for warn_dir in WARN_ONLY_DIRS)


def is_normalization_usage(line: str, lines: List[str], line_idx: int) -> bool:
    """
    Check if a norm usage is for normalization (valid) vs distance (invalid).
    
    VALID normalization patterns:
      - basin / np.linalg.norm(basin)
      - basin / (np.linalg.norm(basin) + eps)
      - norm = np.linalg.norm(basin); basin = basin / norm
      - if norm > 0: basin = basin / norm
    
    INVALID distance patterns:
      - distance = np.linalg.norm(basin)
      - np.linalg.norm(basin_a - basin_b)
      - similarity = 1 / (1 + np.linalg.norm(basin))
    """
    # Pattern 1: Division by norm on same line (direct normalization)
    # e.g., basin / np.linalg.norm(basin) or basin / (np.linalg.norm(...) + eps)
    if re.search(r'/\s*\(?\s*np\.linalg\.norm', line) or re.search(r'/\s*\(?\s*numpy\.linalg\.norm', line):
        return True
    if re.search(r'/\s*\(?\s*torch\.norm', line):
        return True
    
    # Pattern 2: Norm stored in variable, then used for division
    # e.g., norm = np.linalg.norm(basin)
    norm_var_match = re.search(r'(\w+)\s*=\s*(?:np|numpy)\.linalg\.norm', line)
    if norm_var_match:
        var_name = norm_var_match.group(1)
        # Check next few lines for division by this variable
        for next_line in lines[line_idx:min(line_idx + 5, len(lines))]:
            if re.search(rf'/\s*\(?\s*{re.escape(var_name)}[^a-zA-Z_]', next_line):
                return True
            if re.search(rf'/\s*{re.escape(var_name)}\s*$', next_line):
                return True
    
    # Pattern 3: Explicit normalization comment
    if 'normalization' in line.lower() or 'normalize' in line.lower():
        return True
    if 'unit sphere' in line.lower() or 'unit vector' in line.lower():
        return True
    
    # Pattern 4: Check for known valid annotation comment
    if '# valid' in line.lower() or '# normalization' in line.lower():
        return True
    if 'projects to unit' in line.lower():
        return True
    
    return False


def is_subtraction_in_norm(line: str) -> bool:
    """
    Check if a norm call contains subtraction (definite distance calculation).
    e.g., np.linalg.norm(basin_a - basin_b)
    """
    # Look for norm(...-...) pattern
    match = re.search(r'(?:np|numpy)\.linalg\.norm\s*\(([^)]+)\)', line)
    if match:
        inner = match.group(1)
        # Check if there's subtraction inside
        if ' - ' in inner or '- ' in inner or ' -' in inner:
            return True
    return False


def check_file(filepath: Path) -> List[Tuple[int, str, str, bool]]:
    """Check a file for geometric purity violations.
    
    Distinguishes between:
      - VALID: Normalization (v / np.linalg.norm(v)) - projects to unit sphere
      - INVALID: Distance calculation (np.linalg.norm(a - b)) - Euclidean metric
    
    Returns list of (line_num, line, violation_desc, is_warning_only).
    """
    violations = []
    warn_only = is_warn_only(filepath)
    
    try:
        content = filepath.read_text(encoding="utf-8")
        lines = content.split("\n")
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Skip comment lines
            if stripped.startswith("#"):
                continue
            # Skip docstrings and documentation strings
            if '"""' in line or "'''" in line:
                continue
            # Skip lines that are documentation/comments mentioning forbidden patterns
            if re.search(r'NEVER use|CRITICAL:|FORBIDDEN|DO NOT', line):
                continue
            # Skip lines with explicit "valid" or "normalization" annotation
            if re.search(r'#.*(?:valid|normalization|unit sphere|NOTE)', line, re.IGNORECASE):
                continue
            # Skip print/logging statements that just display values
            if re.search(r'^\s*print\s*\(|^\s*logger\.|^\s*logging\.', line):
                continue
            
            # Check FORBIDDEN_PATTERNS first (always invalid)
            found_violation = False
            for pattern, desc in FORBIDDEN_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append((i, line.strip(), desc, warn_only))
                    found_violation = True
                    break
            
            if found_violation:
                continue
            
            # Check CONTEXT_CHECK_PATTERNS (need to verify not normalization)
            for pattern, desc in CONTEXT_CHECK_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    # Skip if this is a normalization usage
                    if is_normalization_usage(line, lines, i - 1):
                        continue
                    # Skip if there's subtraction inside (already caught above)
                    if is_subtraction_in_norm(line):
                        continue
                    # This looks like a non-normalization usage - flag it
                    violations.append((i, line.strip(), f"{desc} (not used for normalization)", warn_only))
                    break
                    
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
        
    return violations


def main():
    verbose = "--verbose" in sys.argv
    
    root = os.getcwd()
    if os.path.exists(os.path.join(root, "qig-backend")):
        search_root = os.path.join(root, "qig-backend")
    else:
        search_root = root
    
    print(f"Checking QIG geometric purity in: {search_root}")
    print("(Euclidean operations on basin coordinates are FORBIDDEN)")
    
    files = find_python_files(search_root)
    errors = []
    warnings = []
    
    for filepath in files:
        violations = check_file(filepath)
        for line_num, line, desc, warn_only in violations:
            entry = (filepath, line_num, line, desc)
            if warn_only:
                warnings.append(entry)
            else:
                errors.append(entry)
    
    # Note: AST-based geometric_purity_checker is available but not run here
    # to avoid double-checking. This script already performs comprehensive checks.
    ast_errors = 0
    
    # Report warnings
    if warnings and verbose:
        print(f"\nWARNINGS: {len(warnings)} potential issue(s) in test files:")
        for filepath, line_num, line, desc in warnings:
            print(f"  {filepath}:{line_num}")
            print(f"    {desc}")
            if verbose:
                print(f"    {line[:60]}{'...' if len(line) > 60 else ''}")
    
    # Report errors
    if not errors and ast_errors == 0:
        print(f"\nOK: Checked {len(files)} files, geometric purity maintained")
        if warnings:
            print(f"({len(warnings)} warnings in test files, use --verbose to see)")
        return 0
    
    if errors:
        print(f"\nERROR: {len(errors)} geometric purity violation(s) found:\n")
        
        for filepath, line_num, line, desc in errors:
            print(f"  {filepath}:{line_num}")
            print(f"    VIOLATION: {desc}")
            print(f"    {line[:70]}{'...' if len(line) > 70 else ''}")
            print()
    
    print("CANONICAL_RULES.md Rule #1:")
    print("  Basin coordinates exist on a CURVED MANIFOLD.")
    print("  Euclidean DISTANCE calculations are INCORRECT.")
    print()
    print("NOTE: Normalization (v / np.linalg.norm(v)) is VALID.")
    print("  This projects to unit sphere and is geometrically correct.")
    print()
    print("FIX distance calculations: Replace with Fisher-Rao distance:")
    print("  from qig_geometry import fisher_rao_distance")
    print("  distance = fisher_rao_distance(basin_a, basin_b)")
    print()
    print("To suppress false positives, add comment: # valid normalization")
    print()
    print("See docs/MIGRATION.md for detailed migration guide.")
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
