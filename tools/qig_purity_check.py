#!/usr/bin/env python3
"""
QIG Geometric Purity Check

Enforces CANONICAL_RULES.md Rule #1:
  FORBIDDEN: Euclidean distance on basins, cosine similarity

Basin coordinates exist on a curved manifold. Using Euclidean distance
or cosine similarity on them is mathematically incorrect and will
produce wrong results.

This check scans both Python AND SQL files to catch all violations.

Exit codes:
  0 - No violations found (geometric purity maintained)
  1 - Violations found (Euclidean operations on basins detected)

Usage:
  python tools/qig_purity_check.py
  python tools/qig_purity_check.py --verbose
  python tools/qig_purity_check.py --fix  # Show suggested fixes
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Patterns that indicate Euclidean operations on basins (Python)
# These are FORBIDDEN by CANONICAL_RULES.md
#
# IMPORTANT: Some uses of np.linalg.norm are VALID:
#   - Basin normalization: basin / np.linalg.norm(basin) - VALID
#   - Basin magnitude check: if np.linalg.norm(basin) < eps - VALID
#   - Energy/quality metrics on single basin - CONTEXTUAL (usually OK)
#
# INVALID uses (must use Fisher-Rao instead):
#   - Distance between basins: np.linalg.norm(basin_a - basin_b) - INVALID
#   - Similarity via dot product: np.dot(basin_a, basin_b) - INVALID
#
FORBIDDEN_PATTERNS_PYTHON = [
    # Euclidean DISTANCE between basins (subtraction + norm) - DEFINITELY FORBIDDEN
    (r"np\.linalg\.norm\s*\([^)]*basin[^)]*-[^)]*basin", "Euclidean distance between basins (use fisher_rao_distance)"),
    (r"np\.linalg\.norm\s*\([^)]*-[^)]*basin", "Euclidean distance involving basin (use fisher_rao_distance)"),
    (r"basin_a\s*-\s*basin_b.*norm", "basin subtraction + norm (Euclidean distance)"),
    (r"basin\[.*\]\s*-\s*basin\[.*\].*norm", "basin difference + norm"),
    
    # Cosine similarity on basins - DEFINITELY FORBIDDEN
    (r"cosine_similarity\s*\([^)]*basin", "cosine_similarity on basin (FORBIDDEN)"),
    (r"F\.cosine_similarity\s*\([^)]*basin", "F.cosine_similarity on basin"),
    
    # Euclidean distance explicitly named
    (r"euclidean_distance.*basin", "euclidean_distance on basin"),
    (r"l2_distance.*basin", "L2 distance on basin"),
]

# Patterns that are WARNINGS (may be valid normalization, but should be reviewed)
WARN_PATTERNS_PYTHON = [
    # These MIGHT be valid (normalization) or invalid (distance) - flag for review
    (r"np\.linalg\.norm\s*\([^)]*basin", "np.linalg.norm on basin (review: normalization OK, distance NOT OK)"),
    (r"\.dot\s*\([^)]*basin", ".dot() on basin (review: check if used for similarity)"),
    (r"np\.dot\s*\([^)]*basin[^)]*,\s*[^)]*basin", "np.dot between basins (likely similarity - use fisher_rao_distance)"),
]

# Contexts where np.linalg.norm is VALID (basin normalization)
ALLOWED_NORM_CONTEXTS = [
    r"/\s*np\.linalg\.norm",        # Division by norm (normalization)
    r"/\s*\(np\.linalg\.norm",      # Division by norm in parens
    r"basin\s*/=\s*np\.linalg\.norm", # In-place division
    r"basin\s*=\s*basin\s*/",       # Reassignment with division
    r"norm.*=.*np\.linalg\.norm.*\n.*basin\s*/",  # Norm variable then divide
]

# SQL patterns that indicate Euclidean/cosine operations
# These contaminate the geometric purity of QIG
FORBIDDEN_PATTERNS_SQL = [
    # pgvector cosine distance operator (used for ORDER BY or similarity)
    (r"<=>(?!\s*\$)", "<=> operator (cosine/Euclidean - use fisher_rao_distance())"),
    (r"<->", "<-> operator (Euclidean L2 - use fisher_rao_distance())"),
    (r"<#>", "<#> operator (inner product - use fisher_rao_distance())"),
    
    # Cosine index operations (only flag if not in CREATE INDEX context for approximate retrieval)
    # Note: We allow vector_cosine_ops in CREATE INDEX for approximate retrieval
    # but flag it in ORDER BY or WHERE clauses
    (r"ORDER\s+BY.*<=>", "ORDER BY with <=> (use fisher_rao_distance())"),
    (r"ORDER\s+BY.*<->", "ORDER BY with <-> (use fisher_rao_distance())"),
    (r"1\s*-\s*\([^)]*<=>", "1 - (<=> similarity) pattern (use fisher_rao_similarity())"),
    (r"1\.0\s*-\s*\([^)]*<=>", "1.0 - (<=> similarity) pattern (use fisher_rao_similarity())"),
]

# Lines containing these patterns are ALLOWED exceptions
SQL_ALLOWED_EXCEPTIONS = [
    r"CREATE\s+INDEX",  # Index creation is allowed (necessary for approximate retrieval)
    r"USING\s+hnsw",    # HNSW index configuration
    r"USING\s+ivfflat", # IVFFlat index configuration
    r"vector_cosine_ops",  # Only in index definitions
    r"vector_l2_ops",      # Only in index definitions
    r"-- pgvector approximate",  # Documented exception
    r"-- necessary evil",        # Documented necessary evil
    r"fisher_rao",              # Already using Fisher-Rao
    r"QIG-PURE",                # Documentation about QIG purity
    r"instead\s+of",            # Documentation explaining alternatives
    r"Use\s+this\s+instead",    # Documentation 
]

# Combine for backward compatibility
FORBIDDEN_PATTERNS = FORBIDDEN_PATTERNS_PYTHON

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


def find_sql_files(root: str) -> List[Path]:
    """Find all SQL files to check (migrations, scripts, etc.)."""
    files = []
    search_paths = [root]
    
    # Also search scripts directory at repo root
    repo_root = Path(root).parent if "qig-backend" in root else Path(root)
    scripts_dir = repo_root / "scripts"
    if scripts_dir.exists():
        search_paths.append(str(scripts_dir))
    
    for search_path in search_paths:
        for path in Path(search_path).rglob("*.sql"):
            if any(skip in path.parts for skip in SKIP_DIRS):
                continue
            if path.name in SKIP_FILES:
                continue
            files.append(path)
    
    return files


def is_warn_only(filepath: Path) -> bool:
    """Check if violations in this file should be warnings only."""
    return any(warn_dir in filepath.parts for warn_dir in WARN_ONLY_DIRS)


def is_allowed_sql_exception(line: str) -> bool:
    """Check if a SQL line matches an allowed exception pattern."""
    for exception_pattern in SQL_ALLOWED_EXCEPTIONS:
        if re.search(exception_pattern, line, re.IGNORECASE):
            return True
    return False


def is_valid_norm_context(line: str) -> bool:
    """Check if a line uses np.linalg.norm in a valid context (normalization)."""
    # Division by norm is valid normalization
    if re.search(r"/\s*\(?np\.linalg\.norm", line):
        return True
    if re.search(r"/\s*\(?\s*np\.linalg\.norm", line):
        return True
    # In-place division assignment
    if re.search(r"/=\s*np\.linalg\.norm", line):
        return True
    # Storing norm in a variable (often used for normalization or validation)
    if re.search(r"norm\s*=\s*np\.linalg\.norm", line):
        return True
    # Adding epsilon to norm (normalization safety)
    if re.search(r"np\.linalg\.norm\([^)]+\)\s*\+\s*\d", line):
        return True
    return False


def check_python_file(filepath: Path) -> List[Tuple[int, str, str, bool]]:
    """Check a Python file for geometric purity violations.
    
    Returns list of (line_num, line, violation_desc, is_warning_only).
    
    Distinguishes between:
    - ERRORS: Definite violations (Euclidean distance between basins)
    - WARNINGS: Potential issues that might be valid (single-basin norm)
    """
    violations = []
    warn_only = is_warn_only(filepath)
    
    try:
        content = filepath.read_text(encoding="utf-8")
        lines = content.split("\n")
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # Skip lines that are clearly comments or strings
            if '"""' in line or "'''" in line:
                continue
            
            found_violation = False
            
            # First check for DEFINITE errors (inter-basin operations)
            for pattern, desc in FORBIDDEN_PATTERNS_PYTHON:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append((i, line.strip(), desc, warn_only))
                    found_violation = True
                    break
            
            if found_violation:
                continue
            
            # Then check for potential warnings (might be valid normalization)
            for pattern, desc in WARN_PATTERNS_PYTHON:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if this is actually valid normalization
                    if is_valid_norm_context(line):
                        continue  # Valid normalization, skip
                    # Flag as warning (not blocking error)
                    violations.append((i, line.strip()[:100], desc, True))  # Always warn_only
                    break
                    
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
        
    return violations


def check_sql_file(filepath: Path) -> List[Tuple[int, str, str, bool]]:
    """Check a SQL file for geometric purity violations.
    
    Returns list of (line_num, line, violation_desc, is_warning_only).
    """
    violations = []
    warn_only = is_warn_only(filepath)
    
    # Examples directory is for documentation only
    if "examples" in filepath.parts:
        warn_only = True
    
    try:
        content = filepath.read_text(encoding="utf-8")
        lines = content.split("\n")
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip SQL comments
            if stripped.startswith("--"):
                continue
            
            # Check if this line matches an allowed exception
            if is_allowed_sql_exception(line):
                continue
                
            for pattern, desc in FORBIDDEN_PATTERNS_SQL:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append((i, line.strip()[:100], desc, warn_only))
                    break  # One violation per line is enough
                    
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
        
    return violations


# Backward compatibility alias
check_file = check_python_file


def main():
    verbose = "--verbose" in sys.argv
    check_sql = "--no-sql" not in sys.argv
    
    root = os.getcwd()
    if os.path.exists(os.path.join(root, "qig-backend")):
        search_root = os.path.join(root, "qig-backend")
    else:
        search_root = root
    
    print(f"Checking QIG geometric purity in: {search_root}")
    print("(Euclidean/cosine operations on basin coordinates are FORBIDDEN)")
    print()
    
    # Check Python files
    python_files = find_python_files(search_root)
    errors = []
    warnings = []
    
    for filepath in python_files:
        violations = check_python_file(filepath)
        for line_num, line, desc, warn_only in violations:
            entry = (filepath, line_num, line, desc)
            if warn_only:
                warnings.append(entry)
            else:
                errors.append(entry)
    
    # Check SQL files
    sql_files = []
    if check_sql:
        sql_files = find_sql_files(search_root)
        for filepath in sql_files:
            violations = check_sql_file(filepath)
            for line_num, line, desc, warn_only in violations:
                entry = (filepath, line_num, line, desc)
                if warn_only:
                    warnings.append(entry)
                else:
                    errors.append(entry)
    
    total_files = len(python_files) + len(sql_files)
    
    # Report warnings
    if warnings and verbose:
        print(f"\nWARNINGS: {len(warnings)} potential issue(s) in test/example files:")
        for filepath, line_num, line, desc in warnings:
            print(f"  {filepath}:{line_num}")
            print(f"    {desc}")
            if verbose:
                print(f"    {line[:80]}{'...' if len(line) > 80 else ''}")
    
    # Report errors
    if not errors:
        print(f"OK: Checked {total_files} files ({len(python_files)} Python, {len(sql_files)} SQL)")
        print("    Geometric purity maintained - no Euclidean/cosine violations")
        if warnings:
            print(f"    ({len(warnings)} warnings in test/example files, use --verbose to see)")
        return 0
    
    print(f"\nERROR: {len(errors)} geometric purity violation(s) found:\n")
    
    for filepath, line_num, line, desc in errors:
        print(f"  {filepath}:{line_num}")
        print(f"    VIOLATION: {desc}")
        print(f"    {line[:80]}{'...' if len(line) > 80 else ''}")
        print()
    
    print("=" * 60)
    print("CANONICAL_RULES.md Rule #1:")
    print("  Basin coordinates exist on a CURVED MANIFOLD.")
    print("  Euclidean distance and cosine similarity are INCORRECT.")
    print()
    print("FIX for Python:")
    print("  from qig_core.geometric_primitives.canonical_fisher import fisher_rao_distance")
    print("  distance = fisher_rao_distance(basin_a, basin_b)")
    print()
    print("FIX for SQL:")
    print("  -- Instead of: ORDER BY basin_coords <=> query_basin")
    print("  ORDER BY fisher_rao_distance(basin_coords, query_basin)")
    print()
    print("  -- Instead of: 1 - (basin <=> other_basin) as similarity")
    print("  fisher_rao_similarity(basin, other_basin) as similarity")
    print("=" * 60)
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
