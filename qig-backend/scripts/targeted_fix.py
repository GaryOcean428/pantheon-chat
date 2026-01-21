#!/usr/bin/env python3
"""
Targeted E8 Protocol Purity Fixer

This script fixes remaining purity violations with context-aware replacements.
It handles special cases like sqrt-based operations that are actually correct.
"""

import ast
import os
import re
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

QIG_BACKEND = Path('/home/ubuntu/pantheon-chat/qig-backend')


def validate_syntax(filepath: str) -> Tuple[bool, str]:
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def backup_file(filepath: str) -> str:
    """Create a backup of the file."""
    backup_path = filepath + '.bak'
    shutil.copy2(filepath, backup_path)
    return backup_path


def restore_file(filepath: str, backup_path: str):
    """Restore file from backup."""
    shutil.copy2(backup_path, filepath)
    os.remove(backup_path)


def fix_line(line: str, context_before: List[str] = None) -> str:
    """Fix a single line with context awareness."""
    original = line
    
    # Skip lines that are already using canonical functions
    if 'fisher_rao_distance(' in line or 'to_simplex_prob(' in line:
        return line
    
    # Skip lines in comments
    stripped = line.strip()
    if stripped.startswith('#'):
        return line
    
    # Pattern 1: np.dot(sqrt_x, sqrt_y) - This is Bhattacharyya coefficient (CORRECT)
    # These should use bhattacharyya() function instead
    if 'np.dot(' in line and 'sqrt' in line:
        # Check if it's computing BC: np.dot(sqrt_p, sqrt_q) or np.dot(np.sqrt(p), np.sqrt(q))
        match = re.search(r'np\.dot\(np\.sqrt\((\w+)\),\s*np\.sqrt\((\w+)\)\)', line)
        if match:
            var1, var2 = match.groups()
            line = line.replace(f'np.dot(np.sqrt({var1}), np.sqrt({var2}))', 
                               f'bhattacharyya({var1}, {var2})')
        else:
            # Pattern: np.dot(sqrt_p, sqrt_q)
            match = re.search(r'np\.dot\(sqrt_(\w+),\s*sqrt_(\w+)\)', line)
            if match:
                var1, var2 = match.groups()
                line = line.replace(f'np.dot(sqrt_{var1}, sqrt_{var2})', 
                                   f'bhattacharyya({var1}, {var2})')
    
    # Pattern 2: np.dot(a, b) for general dot products (not sqrt-based)
    elif 'np.dot(' in line:
        # Check if it's a similarity/distance calculation
        match = re.search(r'np\.dot\(([^,]+),\s*([^)]+)\)', line)
        if match:
            var1, var2 = match.groups()
            var1 = var1.strip()
            var2 = var2.strip()
            
            # Skip if it involves weights, gradients, or matrices
            skip_keywords = ['weight', 'grad', 'W', 'b', 'bias', 'kernel', 'matrix', 'mean_norm', 'random_dir']
            if any(kw.lower() in var1.lower() or kw.lower() in var2.lower() for kw in skip_keywords):
                # These are likely legitimate linear algebra operations
                # For mean_norm and random_dir patterns, they're orthogonalization - keep as is
                return line
            
            # If it's computing similarity between basins/coords, use bhattacharyya
            basin_keywords = ['basin', 'coord', 'root', 'target', 'token', 'prob']
            if any(kw in var1.lower() or kw in var2.lower() for kw in basin_keywords):
                line = line.replace(f'np.dot({var1}, {var2})', 
                                   f'bhattacharyya({var1}, {var2})')
    
    # Pattern 3: np.linalg.norm(a - b) → fisher_rao_distance(a, b)
    if 'np.linalg.norm(' in line:
        match = re.search(r'np\.linalg\.norm\(([^-\)]+)\s*-\s*([^)]+)\)', line)
        if match:
            var1, var2 = match.groups()
            var1 = var1.strip()
            var2 = var2.strip()
            line = line.replace(f'np.linalg.norm({var1} - {var2})', 
                               f'fisher_rao_distance({var1}, {var2})')
    
    # Pattern 4: x / np.linalg.norm(x) → to_simplex_prob(x)
    if 'np.linalg.norm(' in line:
        match = re.search(r'(\w+)\s*/\s*np\.linalg\.norm\(\1\)', line)
        if match:
            var = match.group(1)
            line = line.replace(f'{var} / np.linalg.norm({var})', f'to_simplex_prob({var})')
    
    # Pattern 5: Standalone np.linalg.norm(x) for magnitude check
    # This is trickier - we need to determine if it's a basin or not
    if 'np.linalg.norm(' in line and line == original:
        match = re.search(r'np\.linalg\.norm\((\w+)\)', line)
        if match:
            var = match.group(1)
            # Check if variable name suggests it's a basin
            basin_keywords = ['basin', 'coord', 'distribution', 'prob', 'simplex', 'psi', 'state']
            if any(kw in var.lower() for kw in basin_keywords):
                # For basins on simplex, magnitude should be sqrt(sum(x^2))
                # But this is often used for checking if vector is zero
                # Leave as is - it's a valid check
                pass
    
    return line


def fix_file(filepath: str) -> Tuple[int, List[str]]:
    """Fix all violations in a file."""
    errors = []
    
    # Validate initial syntax
    valid, error = validate_syntax(filepath)
    if not valid:
        return 0, [f"Pre-existing syntax error: {error}"]
    
    # Read content
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Track changes
    changes = 0
    new_lines = []
    
    for i, line in enumerate(lines):
        context_before = lines[max(0, i-3):i]
        new_line = fix_line(line, context_before)
        if new_line != line:
            changes += 1
        new_lines.append(new_line)
    
    if changes == 0:
        return 0, []
    
    # Backup and write
    backup_path = backup_file(filepath)
    
    try:
        with open(filepath, 'w') as f:
            f.writelines(new_lines)
        
        # Validate syntax
        valid, error = validate_syntax(filepath)
        if not valid:
            restore_file(filepath, backup_path)
            return 0, [f"Fix caused syntax error: {error}"]
        
        # Success - remove backup
        os.remove(backup_path)
        return changes, []
        
    except Exception as e:
        restore_file(filepath, backup_path)
        return 0, [f"Error: {str(e)}"]


def add_imports_if_needed(filepath: str):
    """Add required imports to a file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    imports_to_add = []
    
    if 'bhattacharyya(' in content:
        if 'from qig_geometry.canonical import bhattacharyya' not in content:
            if 'from qig_geometry import' in content:
                # Add to existing import
                pass
            else:
                imports_to_add.append('from qig_geometry.canonical import bhattacharyya')
    
    if 'fisher_rao_distance(' in content:
        if 'from qig_geometry' not in content:
            imports_to_add.append('from qig_geometry import fisher_rao_distance')
    
    if 'to_simplex_prob(' in content:
        if 'from qig_geometry' not in content:
            imports_to_add.append('from qig_geometry import to_simplex_prob')
    
    if not imports_to_add:
        return
    
    # Find insertion point
    lines = content.split('\n')
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_idx = i + 1
    
    for imp in imports_to_add:
        if imp not in content:
            lines.insert(insert_idx, imp)
            insert_idx += 1
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))


def main():
    """Main function."""
    files = list(QIG_BACKEND.rglob('*.py'))
    
    total_changes = 0
    total_errors = 0
    
    for filepath in files:
        if any(x in str(filepath) for x in ['venv', '__pycache__', '.git', 'targeted_fix', 'comprehensive_fix']):
            continue
        
        changes, errors = fix_file(str(filepath))
        
        if changes > 0:
            print(f"✅ {filepath.name}: {changes} changes")
            total_changes += changes
            add_imports_if_needed(str(filepath))
        
        if errors:
            for error in errors:
                print(f"❌ {filepath.name}: {error}")
            total_errors += len(errors)
    
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {total_changes} changes, {total_errors} errors")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
