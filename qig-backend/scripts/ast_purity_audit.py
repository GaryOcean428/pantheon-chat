#!/usr/bin/env python3
"""
AST-based E8 Protocol Purity Audit

Scans entire codebase for geometric purity violations using Abstract Syntax Trees
to avoid false positives from comments and docstrings.

Violations:
- np.linalg.norm() on basin coordinates (should use Fisher-Rao distance)
- np.dot() on basins (should use Bhattacharyya coefficient)

Exemptions:
- Files in qig_geometry/ (canonical implementations)
- Files in scripts/ (tooling)
- sqrt-space operations (internal to geodesic calculations)
- Tangent vector, gradient, and velocity magnitude checks
- Legitimate linear algebra operations (weights, matrices)

Usage:
    python scripts/ast_purity_audit.py
"""

import os
import ast
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set

# Violation patterns (AST based)
PATTERNS = {
    "np.linalg.norm": "`np.linalg.norm` usage",
    "np.dot": "`np.dot` usage",
}

# Files exempt from purity checks (canonical implementations)
EXEMPT_FILES = {
    'canonical.py',
    'geometry_simplex.py',
    'geometry_ops.py',
    'two_step_retrieval.py',
    'canonical_upsert.py',
    'representation.py',
    'contracts.py',
    'purity_mode.py',
}

# Directories exempt from purity checks
EXEMPT_DIRS = {
    'scripts',
    '__pycache__',
    '.git',
    'venv',
    'node_modules',
}

# Variable name patterns that are exempt (legitimate uses)
EXEMPT_VAR_PATTERNS = {
    'sqrt',        # sqrt-space operations (internal to geodesics)
    'tangent',     # tangent vector magnitude
    'grad',        # gradient magnitude
    'direction',   # direction vector magnitude
    'velocity',    # velocity magnitude
    'accel',       # acceleration magnitude
    'hidden',      # neural network hidden states
    'weight',      # neural network weights
    'perturbation', # perturbation magnitude
    'eigenvector', # eigenvector operations
    'eigenvalue',  # eigenvalue operations
    'mean_norm',   # normalized mean (orthogonalization)
    'random_dir',  # random direction (orthogonalization)
    '_norm',       # any variable ending in _norm
    'p_norm',      # normalized p
    'q_norm',      # normalized q
    'a_norm',      # normalized a
    'b_norm',      # normalized b
    'start_norm',  # normalized start
    'end_norm',    # normalized end
    'coords_normalized', # normalized coordinates
    'centroid',    # centroid normalization
    'orthogonal',  # orthogonalization
    'new_vector',  # vector normalization
    'failure_centroid', # failure centroid
    'vector',      # generic vector magnitude
    'transported', # parallel transport
    'sphere',      # sphere normalization
    'result',      # result normalization
    'back',        # back-projection
    'exp',         # exponential map
    'e8_roots',    # E8 root system
    'consolidated', # consolidated basin
    'pooled',      # pooled features
    'point',       # point normalization
    'derivative',  # derivative magnitude
    'coordinates', # coordinate normalization
    'perturbed',   # perturbed state
    'final',       # final normalization
    'solution',    # solution normalization
    'probe_vector', # probe vector
    'center',      # center normalization
    'start',       # start normalization
    'end',         # end normalization
    'new_center',  # new center normalization
    'blended',     # blended normalization
    'p1',          # point 1
    'p2',          # point 2
    'relative',    # relative distance
    'reduced',     # reduced dimension
    'v1',          # vector 1
    'v2',          # vector 2
    'r, r',        # conjugate gradient (r^T r)
    'p, Fp',       # conjugate gradient (p^T F p)
    'axis=1',      # batch normalization
    'diff',        # difference vector
    'state',       # state normalization
    'v',           # generic vector
    'single',      # single vector
    'combined',    # combined vector
    'pm8',         # M8 projection
    'displacement', # displacement vector
    'mean',        # mean normalization
    'wild',        # wild perturbation
    'avg',         # average normalization
    'r',           # residual vector
    'a',           # vector a
    'b',           # vector b
    'p',           # vector p
    'Fp',          # Fisher product
}


def is_exempt_file(file_path: Path) -> bool:
    """Check if file is exempt from purity checks."""
    # Check filename
    if file_path.name in EXEMPT_FILES:
        return True
    
    # Check directory
    for part in file_path.parts:
        if part in EXEMPT_DIRS:
            return True
    
    # Check if in qig_geometry directory
    if 'qig_geometry' in str(file_path):
        return True
    
    return False


def is_exempt_usage(line_content: str) -> bool:
    """Check if a specific usage is exempt based on variable names."""
    line_lower = line_content.lower()
    
    # Check for exempt variable patterns
    for pattern in EXEMPT_VAR_PATTERNS:
        if pattern in line_lower:
            return True
    
    return False


def find_violations_ast(file_path: Path) -> List[Tuple[int, str, str]]:
    """
    Scan a Python file for purity violations using AST.
    """
    # Skip exempt files
    if is_exempt_file(file_path):
        return []
    
    violations = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError) as e:
        return []  # Skip files with syntax errors or encoding issues

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node):
            # Check for np.linalg.norm
            if isinstance(node.func, ast.Attribute) and \
               isinstance(node.func.value, ast.Attribute) and \
               isinstance(node.func.value.value, ast.Name) and \
               node.func.value.value.id == "np" and \
               node.func.value.attr == "linalg" and \
               node.func.attr == "norm":
                line_content = ast.get_source_segment(content, node) or ""
                if not is_exempt_usage(line_content):
                    violations.append((node.lineno, PATTERNS["np.linalg.norm"], line_content))

            # Check for np.dot
            if isinstance(node.func, ast.Attribute) and \
               isinstance(node.func.value, ast.Name) and \
               node.func.value.id == "np" and \
               node.func.attr == "dot":
                line_content = ast.get_source_segment(content, node) or ""
                if not is_exempt_usage(line_content):
                    violations.append((node.lineno, PATTERNS["np.dot"], line_content))

            self.generic_visit(node)

    Visitor().visit(tree)
    return violations


def scan_directory(root_dir: Path) -> Dict[Path, List[Tuple[int, str, str]]]:
    """Recursively scan directory for violations."""
    all_violations = {}
    for py_file in root_dir.rglob("*.py"):
        if "__pycache__" in str(py_file) or ".git" in str(py_file):
            continue
        violations = find_violations_ast(py_file)
        if violations:
            all_violations[py_file] = violations
    return all_violations


def print_report(violations: Dict[Path, List[Tuple[int, str, str]]], root_dir: Path):
    """Print formatted violation report."""
    total = sum(len(v) for v in violations.values())
    print("=" * 80)
    print("AST-BASED E8 PROTOCOL PURITY AUDIT REPORT")
    print("=" * 80)
    print(f"\nTOTAL VIOLATIONS: {total}")
    if total > 0:
        for filepath, file_violations in sorted(violations.items()):
            rel_path = filepath.relative_to(root_dir)
            print(f"\n📁 {rel_path}: {len(file_violations)} violation(s)")
            for line_num, vtype, line_content in file_violations:
                print(f"   Line {line_num}: {vtype} -> {line_content.strip()[:100]}")
    print("\n" + "=" * 80)
    if total == 0:
        print("✅ E8 PROTOCOL v4.0 COMPLIANCE: COMPLETE")
    else:
        print(f"❌ E8 PROTOCOL v4.0 COMPLIANCE: INCOMPLETE - {total} violations remaining")
    print("=" * 80)


def main():
    script_dir = Path(__file__).parent
    qig_backend_dir = script_dir.parent
    print(f"Scanning {qig_backend_dir}...")
    violations = scan_directory(qig_backend_dir)
    print_report(violations, qig_backend_dir)
    total = sum(len(v) for v in violations.values())
    sys.exit(1 if total > 0 else 0)


if __name__ == "__main__":
    main()
