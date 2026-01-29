#!/usr/bin/env python3
"""
AST-based E8 Protocol Purity Audit v2.0

Scans entire codebase for geometric purity violations using Abstract Syntax Trees.
This version includes exemptions for legitimate uses in canonical geometry modules.

Violations:
- np.linalg.norm() on basin coordinates (should use Fisher-Rao distance)
- np.dot() on basins (should use Bhattacharyya coefficient or Fisher distance)

Exemptions:
- Files in qig_geometry/ (canonical implementations)
- Files in scripts/ (tooling)
- sqrt-space operations (internal to geodesic calculations)
- Tangent vector magnitude checks
- Gradient magnitude checks

Usage:
    python scripts/ast_purity_audit_v2.py
"""

import os
import ast
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set

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

# Variable name patterns that are exempt (sqrt-space operations)
EXEMPT_VAR_PATTERNS = {
    'sqrt',
    'tangent',
    'grad',
    'direction',
    'velocity',
    'accel',
    'hidden',
    'weight',
    'perturbation',
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


def is_exempt_usage(node: ast.Call, content: str) -> bool:
    """Check if a specific usage is exempt."""
    line_content = ast.get_source_segment(content, node) or ""
    line_lower = line_content.lower()
    
    # Check for exempt variable patterns
    for pattern in EXEMPT_VAR_PATTERNS:
        if pattern in line_lower:
            return True
    
    return False


def find_violations_ast(file_path: Path) -> List[Tuple[int, str, str]]:
    """Scan a Python file for purity violations using AST."""
    violations = []
    
    # Skip exempt files
    if is_exempt_file(file_path):
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError) as e:
        return []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node):
            # Check for np.linalg.norm
            if isinstance(node.func, ast.Attribute) and \
               isinstance(node.func.value, ast.Attribute) and \
               isinstance(node.func.value.value, ast.Name) and \
               node.func.value.value.id == "np" and \
               node.func.value.attr == "linalg" and \
               node.func.attr == "norm":
                if not is_exempt_usage(node, content):
                    line_content = ast.get_source_segment(content, node) or ""
                    violations.append((node.lineno, "`np.linalg.norm` usage", line_content))

            # Check for np.dot
            if isinstance(node.func, ast.Attribute) and \
               isinstance(node.func.value, ast.Name) and \
               node.func.value.id == "np" and \
               node.func.attr == "dot":
                if not is_exempt_usage(node, content):
                    line_content = ast.get_source_segment(content, node) or ""
                    violations.append((node.lineno, "`np.dot` usage", line_content))

            self.generic_visit(node)

    Visitor().visit(tree)
    return violations


def scan_directory(root_dir: Path) -> Dict[Path, List[Tuple[int, str, str]]]:
    """Recursively scan directory for violations."""
    all_violations = {}
    for py_file in root_dir.rglob("*.py"):
        violations = find_violations_ast(py_file)
        if violations:
            all_violations[py_file] = violations
    return all_violations


def print_report(violations: Dict[Path, List[Tuple[int, str, str]]], root_dir: Path):
    """Print formatted violation report."""
    total = sum(len(v) for v in violations.values())
    print("=" * 80)
    print("AST-BASED E8 PROTOCOL PURITY AUDIT REPORT v2.0")
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
