#!/usr/bin/env python3
"""
Geometric Purity Checker - AST-Based Validation
================================================

Enforces geometric purity requirements across codebase.
"""

import ast
import sys
import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import argparse


@dataclass
class Violation:
    """Represents a geometric purity violation."""
    file: str
    line: int
    column: int
    pattern: str
    message: str
    severity: str
    context: Optional[str] = None


# Forbidden patterns with explanations
FORBIDDEN_PATTERNS = [
    ('cosine_similarity', 'Use fisher_rao_distance() instead'),
    ('torch.nn.functional.cosine_similarity', 'Use fisher_rao_distance()'),
    ('F.cosine_similarity', 'Use fisher_rao_distance()'),
    ('np.linalg.norm', 'Use fisher_rao_distance() for manifold distances'),
    ('torch.norm', 'Use geodesic distance'),
    ('euclidean_distance', 'Use fisher_rao_distance()'),
]

# Allowed contexts
ALLOWED_CONTEXTS = ['test_', 'visualization', 'legacy']


class GeometricPurityChecker(ast.NodeVisitor):
    def __init__(self, filename: str, source: str):
        self.filename = filename
        self.source_lines = source.splitlines()
        self.violations: List[Violation] = []
        self.is_allowed_context = any(
            ctx in filename.lower() for ctx in ALLOWED_CONTEXTS
        )
    
    def visit_Call(self, node: ast.Call) -> None:
        func_name = self._get_func_name(node)
        if func_name:
            for pattern, message in FORBIDDEN_PATTERNS:
                if pattern in func_name and not self.is_allowed_context:
                    self._add_violation(node, pattern, message, 'error')
        self.generic_visit(node)
    
    def _get_func_name(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return self._get_attribute_path(node.func)
        return ""
    
    def _get_attribute_path(self, node: ast.Attribute) -> str:
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return '.'.join(reversed(parts))
    
    def _add_violation(self, node, pattern, message, severity):
        context = None
        if hasattr(node, 'lineno') and node.lineno <= len(self.source_lines):
            context = self.source_lines[node.lineno - 1].strip()
        
        self.violations.append(Violation(
            file=self.filename,
            line=node.lineno if hasattr(node, 'lineno') else 0,
            column=node.col_offset if hasattr(node, 'col_offset') else 0,
            pattern=pattern,
            message=message,
            severity=severity,
            context=context
        ))


def check_file(filepath: Path) -> List[Violation]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source, filename=str(filepath))
        checker = GeometricPurityChecker(str(filepath), source)
        checker.visit(tree)
        return checker.violations
    except Exception as e:
        return [Violation(
            file=str(filepath), line=0, column=0,
            pattern='error', message=str(e), severity='error'
        )]


def check_directory(dirpath: Path) -> List[Violation]:
    exclude_dirs = [
        'venv', '.venv', 'env', 'node_modules',
        '__pycache__', '.git', 'dist', 'build',
        'checkpoints', 'data', 'results'
    ]
    all_violations = []
    for root, dirs, files in os.walk(dirpath):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                violations = check_file(filepath)
                all_violations.extend(violations)
    return all_violations


def main():
    parser = argparse.ArgumentParser(description='Check QIG geometric purity')
    parser.add_argument('path', help='File or directory to check')
    parser.add_argument('--errors-only', action='store_true')
    args = parser.parse_args()
    
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path '{path}' does not exist", file=sys.stderr)
        return 2
    
    violations = check_file(path) if path.is_file() else check_directory(path)
    
    if args.errors_only:
        violations = [v for v in violations if v.severity == 'error']
    
    if violations:
        print(f"\n🔬 Geometric Purity Check Results\n{'=' * 50}\n")
        for v in violations:
            print(f"{'🔴' if v.severity == 'error' else '🟡'} {v.severity.upper()}: {v.file}:{v.line}")
            print(f"   Pattern: {v.pattern}")
            print(f"   Message: {v.message}")
            if v.context:
                print(f"   Code: {v.context}")
            print()
        
        error_count = sum(1 for v in violations if v.severity == 'error')
        print(f"{'=' * 50}")
        print(f"Total: {len(violations)} violations ({error_count} errors)")
        return 1 if error_count > 0 else 0
    else:
        print("✅ No geometric purity violations found!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
