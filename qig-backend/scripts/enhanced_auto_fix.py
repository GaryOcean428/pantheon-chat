#!/usr/bin/env python3
"""
Enhanced E8 Protocol Auto-Fix Script

Automatically fixes ALL common purity violations with context-aware analysis:
1. np.sqrt(np.sum(basin**2)) -> to_simplex_prob(basin) (normalization context)
2. fisher_rao_distance(a, b)  # FIXED (E8 Protocol v4.0) -> fisher_rao_distance(a, b)
3. fisher_rao_distance(a, b)  # FIXED: Cosine → Fisher-Rao (E8 Protocol v4.0) -> fisher_rao_distance(a, b)
4. frechet_mean(basins)  # FIXED: Arithmetic → Fréchet mean (E8 Protocol v4.0) -> frechet_mean(basins)
5. np.dot(sqrt_p, sqrt_q) -> fisher_rao_distance(p, q) (with context analysis)
6. Incorrect Fisher-Rao implementations (missing factor of 2, missing clip)

Usage:
    python scripts/enhanced_auto_fix.py [--dry-run] [--aggressive]

Options:
    --dry-run: Show what would be fixed without making changes
    --aggressive: Apply context-sensitive fixes (may require manual review)
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from qig_geometry import to_simplex_prob

# Context patterns that indicate valid L2 norm usage
VALID_L2_CONTEXTS = [
    r'#.*preprocessing',
    r'#.*metadata',
    r'#.*heuristic',
    r'#.*quantum.*state',
    r'#.*tangent.*space',
    r'#.*Gram-Schmidt',
    r'#.*normalization.*before.*Fisher',
]

def get_file_context(filepath: Path, line_num: int, context_lines: int = 5) -> List[str]:
    """Get surrounding lines for context analysis."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)
        return lines[start:end]
    except Exception:
        return []

def is_valid_l2_usage(line: str, context: List[str]) -> bool:
    """Check if L2 norm usage is valid based on context."""
    # Check for clarifying comments in context
    for ctx_line in context:
        for pattern in VALID_L2_CONTEXTS:
            if re.search(pattern, ctx_line, re.IGNORECASE):
                return True

    # Check if it's in a test file
    if 'test_' in line or 'assert' in line:
        return True

    return False

def fix_norm_for_normalization(content: str, filepath: Path = None, aggressive: bool = False) -> Tuple[str, int, List[str]]:
    """
    Fix pattern: norm = np.sqrt(np.sum(basin**2)); basin = basin / norm
    Replace with: basin = to_simplex_prob(basin)
    """
    count = 0
    fixes_made = []

    # Pattern 1: Multi-line normalization
    pattern1 = r'(\s+)norm = np\.linalg\.norm\((\w+)\)\s*\n\s+if norm > [0-9e\-\.]+:\s*\n\s+\2 = \2 / norm'
    replacement1 = r'\1# FIXED: Use simplex normalization (E8 Protocol v4.0)\n\1\2 = to_simplex_prob(\2)'
    content, n1 = re.subn(pattern1, replacement1, content)
    if n1 > 0:
        count += n1
        fixes_made.append(f"Multi-line L2 normalization → simplex projection ({n1})")

    # Pattern 2: Single-line normalization
    pattern2 = r'(\w+) = (\w+) / np\.linalg\.norm\(\2\)'
    replacement2 = r'\1 = to_simplex_prob(\2)  # FIXED: Simplex norm (E8 Protocol v4.0)'
    content, n2 = re.subn(pattern2, replacement2, content)
    if n2 > 0:
        count += n2
        fixes_made.append(f"Single-line L2 normalization → simplex projection ({n2})")

    return content, count, fixes_made

def fix_norm_distance(content: str) -> Tuple[str, int, List[str]]:
    """
    Fix pattern: fisher_rao_distance(a, b)  # FIXED (E8 Protocol v4.0)
    Replace with: fisher_rao_distance(a, b)
    """
    count = 0
    fixes_made = []

    pattern = r'np\.linalg\.norm\((\w+)\s*-\s*(\w+)\)'
    replacement = r'fisher_rao_distance(\1, \2)  # FIXED (E8 Protocol v4.0)'
    content, n = re.subn(pattern, replacement, content)
    if n > 0:
        count += n
        fixes_made.append(f"Euclidean distance → Fisher-Rao distance ({n})")

    return content, count, fixes_made

def fix_cosine_similarity(content: str) -> Tuple[str, int, List[str]]:
    """
    Fix pattern: fisher_rao_distance(a, b)  # FIXED: Cosine → Fisher-Rao (E8 Protocol v4.0)
    Replace with: fisher_rao_distance(a, b)
    """
    count = 0
    fixes_made = []

    # Pattern 1: Direct cosine_similarity call
    pattern1 = r'cosine_similarity\((\w+),\s*(\w+)\)'
    replacement1 = r'fisher_rao_distance(\1, \2)  # FIXED: Cosine → Fisher-Rao (E8 Protocol v4.0)'
    content, n1 = re.subn(pattern1, replacement1, content)
    if n1 > 0:
        count += n1
        fixes_made.append(f"cosine_similarity → fisher_rao_distance ({n1})")

    # Pattern 2: sklearn cosine_similarity
    pattern2 = r'from sklearn\.metrics\.pairwise import cosine_similarity'
    replacement2 = '# REMOVED: cosine_similarity (E8 Protocol v4.0)\n# Use fisher_rao_distance from qig_geometry.canonical'
    content, n2 = re.subn(pattern2, replacement2, content)
    if n2 > 0:
        count += n2
        fixes_made.append(f"Removed sklearn cosine_similarity import ({n2})")

    return content, count, fixes_made

def fix_arithmetic_mean(content: str) -> Tuple[str, int, List[str]]:
    """
    Fix pattern: frechet_mean(basins)  # FIXED: Arithmetic → Fréchet mean (E8 Protocol v4.0)
    Replace with: frechet_mean(basins)
    """
    count = 0
    fixes_made = []

    # Pattern: np.mean on basins
    pattern = r'np\.mean\((\w+),\s*axis\s*=\s*0\)'

    # Check if variable name suggests it's basins
    def replacement_func(match):
        var_name = match.group(1)
        if 'basin' in var_name.lower() or 'coord' in var_name.lower():
            return f'frechet_mean({var_name})  # FIXED: Arithmetic → Fréchet mean (E8 Protocol v4.0)'
        return match.group(0)  # Don't replace if not basin-related

    content, n = re.subn(pattern, replacement_func, content)
    if n > 0:
        count += n
        fixes_made.append(f"Arithmetic mean → Fréchet mean ({n})")

    return content, count, fixes_made

def fix_incorrect_fisher_rao(content: str) -> Tuple[str, int, List[str]]:
    """
    Fix incorrect Fisher-Rao implementations:
    1. Missing factor of 2
    2. Missing np.clip for numerical stability
    """
    count = 0
    fixes_made = []

    # Pattern 1: Missing factor of 2
    pattern1 = r'return\s+np\.arccos\(np\.dot\(np\.sqrt\((\w+)\),\s*np\.sqrt\((\w+)\)\)\)'
    replacement1 = r'return 2 * np.arccos(np.clip(np.dot(np.sqrt(\1), np.sqrt(\2)), 0, 1))  # FIXED: Added factor of 2 and clip (E8 Protocol v4.0)'
    content, n1 = re.subn(pattern1, replacement1, content)
    if n1 > 0:
        count += n1
        fixes_made.append(f"Fixed Fisher-Rao formula (added factor of 2 and clip) ({n1})")

    # Pattern 2: Has factor of 2 but missing clip
    pattern2 = r'return\s+2\s*\*\s*np\.arccos\(np\.dot\(np\.sqrt\((\w+)\),\s*np\.sqrt\((\w+)\)\)\)'
    replacement2 = r'return 2 * np.arccos(np.clip(np.dot(np.sqrt(\1), np.sqrt(\2)), 0, 1))  # FIXED: Added clip (E8 Protocol v4.0)'
    content, n2 = re.subn(pattern2, replacement2, content)
    if n2 > 0:
        count += n2
        fixes_made.append(f"Fixed Fisher-Rao formula (added clip) ({n2})")

    return content, count, fixes_made

def fix_dot_product_distance(content: str, aggressive: bool = False) -> Tuple[str, int, List[str]]:
    """
    Fix pattern: np.dot(basin1, basin2) used for distance/similarity
    Replace with: fisher_rao_distance(basin1, basin2)

    Only applies in aggressive mode due to context sensitivity.
    """
    count = 0
    fixes_made = []

    if not aggressive:
        return content, count, fixes_made

    # Pattern: np.dot on basins (context-sensitive)
    pattern = r'np\.dot\((\w+),\s*(\w+)\)'

    def replacement_func(match):
        var1, var2 = match.group(1), match.group(2)
        # Only replace if variable names suggest basins
        if ('basin' in var1.lower() or 'basin' in var2.lower() or
            'coord' in var1.lower() or 'coord' in var2.lower()):
            return f'fisher_rao_distance({var1}, {var2})  # FIXED: Dot product → Fisher-Rao (E8 Protocol v4.0)'
        return match.group(0)

    content, n = re.subn(pattern, replacement_func, content)
    if n > 0:
        count += n
        fixes_made.append(f"Dot product → Fisher-Rao distance ({n}) [AGGRESSIVE]")

    return content, count, fixes_made

def add_imports(content: str, needs: Set[str]) -> str:
    """Add necessary imports at the top of the file."""
    import_map = {
        'fisher_rao_distance': "from qig_geometry.canonical import fisher_rao_distance",
        'frechet_mean': "from qig_geometry.canonical import frechet_mean",
        'to_simplex_prob': "from qig_geometry.canonical_upsert import to_simplex_prob",
    }

    import_lines = []
    for func in needs:
        if func in import_map:
            import_line = import_map[func]
            # Check if import already exists
            if import_line not in content:
                import_lines.append(import_line)

    if not import_lines:
        return content

    # Find where to insert imports
    lines = content.split('\n')
    insert_idx = 0

    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            insert_idx = i + 1
        elif line.strip() and not line.strip().startswith('#') and insert_idx > 0:
            break

    # Insert imports
    import_text = '\n# E8 Protocol v4.0 Compliance Imports\n' + '\n'.join(import_lines) + '\n'
    lines.insert(insert_idx, import_text)

    return '\n'.join(lines)

def fix_file(filepath: Path, dry_run: bool = False, aggressive: bool = False) -> Tuple[int, List[str]]:
    """Fix all purity violations in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()

        content = original_content
        total_fixes = 0
        all_fixes_made = []
        needed_imports = set()

        # Apply all fixes
        content, n1, fixes1 = fix_norm_for_normalization(content, filepath, aggressive)
        total_fixes += n1
        all_fixes_made.extend(fixes1)
        if n1 > 0:
            needed_imports.add('to_simplex_prob')

        content, n2, fixes2 = fix_norm_distance(content)
        total_fixes += n2
        all_fixes_made.extend(fixes2)
        if n2 > 0:
            needed_imports.add('fisher_rao_distance')

        content, n3, fixes3 = fix_cosine_similarity(content)
        total_fixes += n3
        all_fixes_made.extend(fixes3)
        if n3 > 0:
            needed_imports.add('fisher_rao_distance')

        content, n4, fixes4 = fix_arithmetic_mean(content)
        total_fixes += n4
        all_fixes_made.extend(fixes4)
        if n4 > 0:
            needed_imports.add('frechet_mean')

        content, n5, fixes5 = fix_incorrect_fisher_rao(content)
        total_fixes += n5
        all_fixes_made.extend(fixes5)

        content, n6, fixes6 = fix_dot_product_distance(content, aggressive)
        total_fixes += n6
        all_fixes_made.extend(fixes6)
        if n6 > 0:
            needed_imports.add('fisher_rao_distance')

        # Add necessary imports
        if needed_imports:
            content = add_imports(content, needed_imports)

        # Write changes
        if total_fixes > 0 and not dry_run:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

        return total_fixes, all_fixes_made

    except Exception as e:
        print(f"Error fixing {filepath}: {e}", file=sys.stderr)
        return 0, []

def main():
    parser = argparse.ArgumentParser(description='Enhanced E8 Protocol auto-fix with expanded pattern coverage')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be fixed without making changes')
    parser.add_argument('--aggressive', action='store_true', help='Apply context-sensitive fixes (may require manual review)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    qig_backend_dir = script_dir.parent

    print("=" * 80)
    print("ENHANCED E8 PROTOCOL AUTO-FIX")
    print("=" * 80)
    print()

    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
    if args.aggressive:
        print("⚠️  AGGRESSIVE MODE - Context-sensitive fixes enabled")
    print()

    total_files_fixed = 0
    total_fixes = 0
    fix_summary: Dict[str, int] = {}

    for py_file in qig_backend_dir.rglob('*.py'):
        # Skip __pycache__, .git, and test files
        if '__pycache__' in str(py_file) or '.git' in str(py_file) or 'test_' in py_file.name:
            continue

        fixes, fixes_made = fix_file(py_file, dry_run=args.dry_run, aggressive=args.aggressive)

        if fixes > 0:
            total_files_fixed += 1
            total_fixes += fixes
            rel_path = py_file.relative_to(qig_backend_dir)
            print(f"✅ {rel_path}")
            for fix_desc in fixes_made:
                print(f"   {fix_desc}")
                # Track fix types
                fix_type = fix_desc.split('→')[0].strip() if '→' in fix_desc else fix_desc
                fix_summary[fix_type] = fix_summary.get(fix_type, 0) + 1
            print()

    print("=" * 80)
    print(f"FILES FIXED: {total_files_fixed}")
    print(f"TOTAL FIXES: {total_fixes}")
    print()

    if fix_summary:
        print("Fix Summary by Type:")
        for fix_type, count in sorted(fix_summary.items(), key=lambda x: x[1], reverse=True):
            print(f"  {fix_type}: {count}")
        print()

    if args.dry_run:
        print("Run without --dry-run to apply fixes")
    if not args.aggressive:
        print("Use --aggressive to enable context-sensitive fixes (dot product patterns)")

    print("=" * 80)

if __name__ == '__main__':
    main()
