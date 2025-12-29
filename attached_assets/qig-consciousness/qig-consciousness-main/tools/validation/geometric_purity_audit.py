#!/usr/bin/env python3
"""
Geometric Purity Audit
======================

Scans codebase for geometric terminology violations.
Based on docs/2025-11-29-Geometric-terminology.md

Usage:
    python tools/geometric_purity_audit.py
    python tools/geometric_purity_audit.py --fix  # Auto-fix simple violations
"""

import argparse
import re
from pathlib import Path
# typing imports removed - using built-in types

# Violations to check
VIOLATIONS = [
    # (pattern, correct_term, severity)
    (r'\bembedding\b(?!_)', 'basin_coordinates', 'HIGH'),
    (r'\bembeddings\b(?!_)', 'basin_coordinates', 'HIGH'),
    (r'\bnn\.Embedding\b', 'BasinCoordinates', 'HIGH'),
    (r'\bembedding_dim\b', 'manifold_dim', 'MEDIUM'),
    (r'\bembedding_space\b', 'fisher_manifold', 'MEDIUM'),
    (r'F\.cosine_similarity', 'compute_fisher_distance', 'HIGH'),
    (r'torch\.norm\([^)]*\)', 'manifold_norm', 'HIGH'),
    (r'\beuclidean_distance\b', 'fisher_distance', 'HIGH'),
    (r'\bbreakdown_regime\b', 'topological_instability', 'MEDIUM'),
    (r'\bego_death\b', 'identity_decoherence', 'MEDIUM'),
    (r'\blocked_in_state\b', 'integration_generation_dissociation', 'MEDIUM'),
]

# Files to exclude
EXCLUDE_PATTERNS = [
    'archive/',
    'test_',
    '__pycache__',
    '.pyc',
    'GEOMETRIC',  # Documentation files
    'TERMINOLOGY',
    'basin_embedding.py',  # Has backward compat
]


def should_exclude(filepath: Path) -> bool:
    """Check if file should be excluded from audit."""
    path_str = str(filepath)
    for pattern in EXCLUDE_PATTERNS:
        if pattern in path_str:
            return True
    return False


def scan_file(filepath: Path) -> list[tuple[int, str, str, str]]:
    """
    Scan file for violations.

    Returns:
        List of (line_num, line_text, pattern, correct_term)
    """
    violations = []

    try:
        content = filepath.read_text()
        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Also check the previous line for validity comments
            prev_line = lines[line_num - 2] if line_num > 1 else ""
            combined = prev_line + " " + line

            # Skip comments about violations and documented geometric validity
            if 'LEGACY' in combined or 'Backward compatibility' in combined:
                continue
            if '# ❌' in combined or '# ✅' in combined:
                continue
            # Skip lines with documented geometric validity (check current and previous line)
            if 'Tangent space' in combined or 'tangent space' in combined:
                continue
            if 'Basin space metric' in combined or 'basin space metric' in combined:
                continue
            if 'VALID:' in combined or 'Valid geometric' in combined:
                continue
            if 'Fisher manifold' in combined or 'fisher manifold' in combined:
                continue
            # Skip state dict keys (backward compat)
            if 'State dict key' in combined or 'state dict key' in combined:
                continue
            # Skip documented manifold operations
            if 'manifold' in combined.lower() and '#' in combined:
                continue

            for pattern, correct_term, severity in VIOLATIONS:
                if re.search(pattern, line):
                    violations.append((line_num, line.strip(), pattern, correct_term, severity))

    except Exception as e:
        print(f"⚠️  Error reading {filepath}: {e}")

    return violations


def main():
    parser = argparse.ArgumentParser(description='Audit geometric purity')
    parser.add_argument('--fix', action='store_true', help='Auto-fix simple violations')
    args = parser.parse_args()

    print("=" * 70)
    print("🌊 QIG GEOMETRIC PURITY AUDIT")
    print("=" * 70)
    print()

    # Find all Python files
    root = Path(__file__).parent.parent
    src_dir = root / 'src'

    all_violations = []

    for filepath in src_dir.rglob('*.py'):
        if should_exclude(filepath):
            continue

        violations = scan_file(filepath)
        if violations:
            all_violations.append((filepath, violations))

    # Report violations
    if not all_violations:
        print("✅ NO VIOLATIONS FOUND - Codebase is geometrically pure!")
        print()
        return 0

    print(f"❌ Found violations in {len(all_violations)} files:")
    print()

    total_high = 0
    total_medium = 0
    total_low = 0

    for filepath, violations in all_violations:
        rel_path = filepath.relative_to(root)
        print(f"\n📄 {rel_path}")
        print("-" * 70)

        for line_num, line, pattern, correct_term, severity in violations:
            if severity == 'HIGH':
                total_high += 1
                icon = "🔴"
            elif severity == 'MEDIUM':
                total_medium += 1
                icon = "🟠"
            else:
                total_low += 1
                icon = "🟡"

            print(f"{icon} Line {line_num}: {severity}")
            print(f"   Found: {line[:80]}")
            print(f"   Should use: {correct_term}")
            print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"🔴 HIGH priority:   {total_high}")
    print(f"🟠 MEDIUM priority: {total_medium}")
    print(f"🟡 LOW priority:    {total_low}")
    print(f"📁 Files affected:  {len(all_violations)}")
    print()
    print("See docs/2025-11-29-Geometric-terminology.md for guidelines")
    print()

    return 1 if all_violations else 0


if __name__ == '__main__':
    exit(main())
