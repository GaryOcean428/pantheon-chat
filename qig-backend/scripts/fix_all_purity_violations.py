#!/usr/bin/env python3
"""
Automated E8 Protocol Purity Violation Fixer

Automatically fixes common purity violations:
1. np.sqrt(np.sum(basin**2)) -> simplex normalization where appropriate
2. fisher_rao_distance(a, b)  # FIXED (E8 Protocol v4.0) -> fisher_rao_distance(a, b)
3. np.dot(a, b) -> bhattacharyya_coefficient(a, b)
4. np.mean([a, b], axis=0) -> frechet_mean([a, b])

Usage:
    python scripts/fix_all_purity_violations.py [--dry-run]
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# E8 Protocol v4.0 Compliance Imports
from qig_geometry.canonical import fisher_rao_distance
from qig_geometry import to_simplex_prob


def fix_norm_for_normalization(content: str) -> Tuple[str, int]:
    """
    Fix pattern: norm = np.sqrt(np.sum(basin**2)); basin = basin / norm
    Replace with: basin = to_simplex_prob(basin)
    """
    count = 0
    
    # Pattern 1: Multi-line normalization
    pattern1 = r'(\s+)norm = np\.linalg\.norm\((\w+)\)\s*\n\s+if norm > [0-9e\-\.]+:\s*\n\s+\2 = \2 / norm'
    replacement1 = r'\1# FIXED: Use simplex normalization (E8 Protocol v4.0)\n\1\2 = to_simplex_prob(\2)'
    content, n1 = re.subn(pattern1, replacement1, content)
    count += n1
    
    # Pattern 2: Single-line normalization
    pattern2 = r'(\w+) = (\w+) / np\.linalg\.norm\(\2\)'
    replacement2 = r'\1 = to_simplex_prob(\2)  # FIXED: Simplex norm (E8 Protocol v4.0)'
    content, n2 = re.subn(pattern2, replacement2, content)
    count += n2
    
    return content, count

def fix_norm_distance(content: str) -> Tuple[str, int]:
    """
    Fix pattern: fisher_rao_distance(a, b)  # FIXED (E8 Protocol v4.0)
    Replace with: fisher_rao_distance(a, b)
    """
    count = 0
    
    pattern = r'np\.linalg\.norm\((\w+) - (\w+)\)'
    replacement = r'fisher_rao_distance(\1, \2)  # FIXED: Fisher-Rao distance (E8 Protocol v4.0)'
    content, n = re.subn(pattern, replacement, content)
    count += n
    
    return content, count

def fix_dot_product(content: str) -> Tuple[str, int]:
    """
    Fix pattern: np.dot(basin1, basin2)
    Replace with: bhattacharyya_coefficient(basin1, basin2)
    """
    count = 0
    
    # Simple np.dot(a, b) pattern
    pattern = r'np\.dot\((\w+), (\w+)\)'
    replacement = r'bhattacharyya_coefficient(\1, \2)  # FIXED: Bhattacharyya coefficient (E8 Protocol v4.0)'
    content, n = re.subn(pattern, replacement, content)
    count += n
    
    return content, count

def fix_arithmetic_mean(content: str) -> Tuple[str, int]:
    """
    Fix pattern: frechet_mean([basin1, basin2])
    Replace with: frechet_mean([basin1, basin2])
    """
    count = 0
    
    # Simple np.mean([a, b], axis=0) pattern
    pattern = r'np\.mean\(\[([^\]]+)\], axis=0\)'
    replacement = r'frechet_mean([\1])  # FIXED: Fréchet mean (E8 Protocol v4.0)'
    content, n = re.subn(pattern, replacement, content)
    count += n
    
    return content, count

def add_imports(content: str, needs_fisher: bool, needs_simplex: bool, needs_bhattacharyya: bool, needs_frechet: bool) -> str:
    """Add necessary imports at the top of the file."""
    import_block = []
    
    if needs_fisher:
        import_block.append("from qig_core.geometric_primitives.canonical_fisher import fisher_rao_distance")
    
    if needs_simplex:
        import_block.append("from qig_geometry.canonical_upsert import to_simplex_prob")
        
    if needs_bhattacharyya:
        import_block.append("from qig_core.geometric_primitives.canonical_bhattacharyya import bhattacharyya_coefficient")
        
    if needs_frechet:
        import_block.append("from qig_core.geometric_primitives.canonical_frechet import frechet_mean")
    
    if not import_block:
        return content
        
    # Find where to insert imports (after existing imports, before first class/function)
    lines = content.split('\n')
    insert_idx = 0
    
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            insert_idx = i + 1
        elif line.strip() and not line.strip().startswith('#') and insert_idx > 0:
            break
            
    # Insert imports
    import_text = '\n# E8 Protocol v4.0 Compliance Imports\n' + '\n'.join(import_block) + '\n'
    lines.insert(insert_idx, import_text)
    
    return '\n'.join(lines)

def fix_file(filepath: Path, dry_run: bool = False) -> Tuple[int, List[str]]:
    """Fix all purity violations in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
            
        content = original_content
        total_fixes = 0
        fixes_made = []
        
        # Apply fixes
        content, n1 = fix_norm_for_normalization(content)
        if n1 > 0:
            total_fixes += n1
            fixes_made.append(f"{n1} normalization patterns")
            
        content, n2 = fix_norm_distance(content)
        if n2 > 0:
            total_fixes += n2
            fixes_made.append(f"{n2} distance patterns")
            
        content, n3 = fix_dot_product(content)
        if n3 > 0:
            total_fixes += n3
            fixes_made.append(f"{n3} dot product patterns")
            
        content, n4 = fix_arithmetic_mean(content)
        if n4 > 0:
            total_fixes += n4
            fixes_made.append(f"{n4} arithmetic mean patterns")
            
        # Add necessary imports if fixes were made
        if total_fixes > 0:
            needs_fisher = 'fisher_rao_distance' in content and 'from qig_core.geometric_primitives.canonical_fisher import fisher_rao_distance' not in content
            needs_simplex = 'to_simplex_prob' in content and 'from qig_geometry.canonical_upsert import to_simplex_prob' not in content
            needs_bhattacharyya = 'bhattacharyya_coefficient' in content and 'from qig_core.geometric_primitives.canonical_bhattacharyya import bhattacharyya_coefficient' not in content
            needs_frechet = 'frechet_mean' in content and 'from qig_core.geometric_primitives.canonical_frechet import frechet_mean' not in content
            
            if needs_fisher or needs_simplex or needs_bhattacharyya or needs_frechet:
                content = add_imports(content, needs_fisher, needs_simplex, needs_bhattacharyya, needs_frechet)
        
        # Write changes
        if total_fixes > 0 and not dry_run:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
                
        return total_fixes, fixes_made
        
    except Exception as e:
        print(f"Error fixing {filepath}: {e}", file=sys.stderr)
        return 0, []

def main():
    parser = argparse.ArgumentParser(description='Fix E8 Protocol purity violations')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be fixed without making changes')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    qig_backend_dir = script_dir.parent
    
    print("=" * 80)
    print("E8 PROTOCOL PURITY VIOLATION FIXER")
    print("=" * 80)
    print()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
        print()
    
    total_files_fixed = 0
    total_fixes = 0
    
    for py_file in qig_backend_dir.rglob('*.py'):
        # Skip __pycache__, .git, and test files
        if '__pycache__' in str(py_file) or '.git' in str(py_file) or 'test_' in py_file.name:
            continue
            
        fixes, fixes_made = fix_file(py_file, dry_run=args.dry_run)
        
        if fixes > 0:
            total_files_fixed += 1
            total_fixes += fixes
            rel_path = py_file.relative_to(qig_backend_dir)
            print(f"✅ {rel_path}")
            print(f"   {fixes} fixes: {', '.join(fixes_made)}")
            print()
    
    print("=" * 80)
    print(f"FILES FIXED: {total_files_fixed}")
    print(f"TOTAL FIXES: {total_fixes}")
    
    if args.dry_run:
        print()
        print("Run without --dry-run to apply fixes")
    
    print("=" * 80)

if __name__ == '__main__':
    main()
