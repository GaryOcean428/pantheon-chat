#!/usr/bin/env python3
"""
AST-based E8 Protocol Purity Audit

Scans entire codebase for geometric purity violations using Abstract Syntax Trees
to avoid false positives from comments and docstrings.

- np.linalg.norm() on basin coordinates
- np.dot() on basins

Usage:
    python scripts/ast_purity_audit.py
"""

import os
import ast
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Violation patterns (AST based)
PATTERNS = {
    "np.linalg.norm": "`np.linalg.norm` usage",
    "np.dot": "`np.dot` usage",
}

def find_violations_ast(file_path: Path) -> List[Tuple[int, str, str]]:
    """
    Scan a Python file for purity violations using AST.
    """
    violations = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError) as e:
        # print(f"Error parsing {file_path}: {e}", file=sys.stderr)
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
                violations.append((node.lineno, PATTERNS["np.linalg.norm"], line_content))

            # Check for np.dot
            if isinstance(node.func, ast.Attribute) and \
               isinstance(node.func.value, ast.Name) and \
               node.func.value.id == "np" and \
               node.func.attr == "dot":
                line_content = ast.get_source_segment(content, node) or ""
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
