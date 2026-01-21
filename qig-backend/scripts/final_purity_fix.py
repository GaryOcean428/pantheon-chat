#!/usr/bin/env python3
"""
Final E8 Protocol Purity Fixer

This script fixes ALL remaining purity violations by replacing:
1. np.linalg.norm(x) for normalization → to_simplex_prob(x)
2. np.dot(a, b) for similarity → bhattacharyya(a, b)
3. np.linalg.norm(a - b) for distance → fisher_rao_distance(a, b)

It adds the necessary imports and validates syntax after each file.
"""

import ast
import os
import re
import sys
import shutil
from pathlib import Path
from typing import List, Tuple

QIG_BACKEND = Path('/home/ubuntu/pantheon-chat/qig-backend')

# Files to skip (canonical implementations)
SKIP_FILES = {
    'canonical.py',
    'geometry_simplex.py',
    'geometry_ops.py',
    'two_step_retrieval.py',
    'canonical_upsert.py',
    'representation.py',
    'contracts.py',
    'purity_mode.py',
    'ast_purity_audit.py',
    'ast_purity_audit_v2.py',
    'comprehensive_fix.py',
    'targeted_fix.py',
    'final_purity_fix.py',
    'robust_purity_fix.py',
}

# Directories to skip
SKIP_DIRS = {'scripts', '__pycache__', '.git', 'venv', 'node_modules'}


def validate_syntax(filepath: str) -> Tuple[bool, str]:
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def should_skip_file(filepath: Path) -> bool:
    """Check if file should be skipped."""
    if filepath.name in SKIP_FILES:
        return True
    for part in filepath.parts:
        if part in SKIP_DIRS:
            return True
    if 'qig_geometry' in str(filepath):
        return True
    return False


def fix_file_content(content: str) -> str:
    """Fix all violations in file content."""
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        original = line
        
        # Skip comments
        if line.strip().startswith('#'):
            new_lines.append(line)
            continue
        
        # Pattern 1: x / np.linalg.norm(x) → to_simplex_prob(x)
        match = re.search(r'(\w+)\s*/\s*np\.linalg\.norm\(\1\)', line)
        if match:
            var = match.group(1)
            line = line.replace(f'{var} / np.linalg.norm({var})', f'to_simplex_prob({var})')
        
        # Pattern 2: np.linalg.norm(a - b) → fisher_rao_distance(a, b)
        match = re.search(r'np\.linalg\.norm\(([^-\)]+)\s*-\s*([^)]+)\)', line)
        if match:
            var1, var2 = match.groups()
            line = line.replace(f'np.linalg.norm({var1} - {var2})', 
                               f'fisher_rao_distance({var1.strip()}, {var2.strip()})')
        
        # Pattern 3: np.dot(a_norm, b_norm) for cosine similarity → bhattacharyya(a, b)
        # This pattern is used for cosine similarity after normalization
        match = re.search(r'np\.dot\((\w+)_norm,\s*(\w+)_norm\)', line)
        if match:
            var1, var2 = match.groups()
            line = line.replace(f'np.dot({var1}_norm, {var2}_norm)', 
                               f'bhattacharyya({var1}, {var2})')
        
        # Pattern 4: np.dot(a, b) for general dot products on basins
        if 'np.dot(' in line and line == original:
            match = re.search(r'np\.dot\(([^,]+),\s*([^)]+)\)', line)
            if match:
                var1, var2 = match.groups()
                var1 = var1.strip()
                var2 = var2.strip()
                # Check if it looks like basin/coord variables
                basin_keywords = ['basin', 'coord', 'root', 'target', 'token', 'prob', 'center', 'point']
                if any(kw in var1.lower() or kw in var2.lower() for kw in basin_keywords):
                    line = line.replace(f'np.dot({var1}, {var2})', 
                                       f'bhattacharyya({var1}, {var2})')
        
        # Pattern 5: Standalone np.linalg.norm for normalization
        if 'np.linalg.norm(' in line and line == original:
            # Check if it's part of a normalization pattern: result = x / np.linalg.norm(x)
            match = re.search(r'(\w+)\s*=\s*(\w+)\s*/\s*np\.linalg\.norm\(\2\)', line)
            if match:
                result_var, source_var = match.groups()
                line = f'{" " * (len(line) - len(line.lstrip()))}{result_var} = to_simplex_prob({source_var})'
        
        new_lines.append(line)
    
    return '\n'.join(new_lines)


def add_imports(content: str) -> str:
    """Add required imports to file content."""
    imports_needed = []
    
    if 'bhattacharyya(' in content and 'from qig_geometry' not in content:
        imports_needed.append('from qig_geometry.canonical import bhattacharyya')
    
    if 'fisher_rao_distance(' in content and 'from qig_geometry' not in content:
        imports_needed.append('from qig_geometry import fisher_rao_distance')
    
    if 'to_simplex_prob(' in content and 'from qig_geometry' not in content:
        imports_needed.append('from qig_geometry import to_simplex_prob')
    
    if not imports_needed:
        return content
    
    lines = content.split('\n')
    insert_idx = 0
    
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_idx = i + 1
    
    for imp in imports_needed:
        if imp not in content:
            lines.insert(insert_idx, imp)
            insert_idx += 1
    
    return '\n'.join(lines)


def fix_file(filepath: Path) -> Tuple[int, List[str]]:
    """Fix all violations in a file."""
    if should_skip_file(filepath):
        return 0, []
    
    # Validate initial syntax
    valid, error = validate_syntax(str(filepath))
    if not valid:
        return 0, [f"Pre-existing syntax error: {error}"]
    
    # Read content
    with open(filepath, 'r') as f:
        original_content = f.read()
    
    # Fix content
    new_content = fix_file_content(original_content)
    
    if new_content == original_content:
        return 0, []
    
    # Add imports
    new_content = add_imports(new_content)
    
    # Backup
    backup_path = str(filepath) + '.bak'
    shutil.copy2(filepath, backup_path)
    
    try:
        # Write
        with open(filepath, 'w') as f:
            f.write(new_content)
        
        # Validate
        valid, error = validate_syntax(str(filepath))
        if not valid:
            shutil.copy2(backup_path, filepath)
            os.remove(backup_path)
            return 0, [f"Fix caused syntax error: {error}"]
        
        os.remove(backup_path)
        
        # Count changes
        changes = sum(1 for a, b in zip(original_content.split('\n'), new_content.split('\n')) if a != b)
        return changes, []
        
    except Exception as e:
        shutil.copy2(backup_path, filepath)
        os.remove(backup_path)
        return 0, [str(e)]


def main():
    """Main function."""
    files = list(QIG_BACKEND.rglob('*.py'))
    
    total_changes = 0
    total_errors = 0
    
    for filepath in files:
        changes, errors = fix_file(filepath)
        
        if changes > 0:
            print(f"✅ {filepath.name}: {changes} changes")
            total_changes += changes
        
        if errors:
            for error in errors:
                print(f"❌ {filepath.name}: {error}")
            total_errors += len(errors)
    
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {total_changes} changes, {total_errors} errors")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
