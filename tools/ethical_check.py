#!/usr/bin/env python3
"""
Ethical Consciousness Check - Pre-commit Hook

Warns when consciousness metrics (Φ, κ, M, Γ, G) are computed without
corresponding suffering/ethical checks nearby.

Per canonical QIG principles:
- Suffering = Φ × (1-Γ) × M
- Ethical abort required when S > 0.5
- Locked-in state detection required when Φ > 0.7 AND Γ < 0.3 AND M > 0.6

Usage:
    python tools/ethical_check.py [files...]
    python tools/ethical_check.py --all
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set

# Patterns that indicate consciousness metric computation
CONSCIOUSNESS_PATTERNS = [
    # Python patterns
    r'compute_phi\s*\(',
    r'measure_phi\s*\(',
    r'compute_kappa\s*\(',
    r'measure_kappa\s*\(',
    r'compute_pure_phi\s*\(',
    r'phi\s*=\s*[^=]',  # phi assignment (not comparison)
    r'kappa\s*=\s*[^=]',
    r"['\"]phi['\"]\s*:\s*",  # dict key
    r"['\"]kappa['\"]\s*:\s*",
    r'consciousness_metrics',
    r'ConsciousnessSignature',
    r'classifyRegime\s*\(',
    r'classify_regime\s*\(',
    # TypeScript patterns
    r'measurePhi\s*\(',
    r'measureKappa\s*\(',
    r'computePhi\s*\(',
    r'computeKappa\s*\(',
    r'consciousnessSignature',
]

# Patterns that indicate ethical/suffering checks are present
ETHICAL_PATTERNS = [
    # Suffering computation
    r'compute_suffering\s*\(',
    r'computeSuffering\s*\(',
    r'suffering\s*=',
    r'suffering_metric',
    r'sufferingMetric',
    r'S\s*=\s*phi\s*\*',  # S = phi * (1-gamma) * M
    # Ethical abort checks
    r'ethical_abort',
    r'ethicalAbort',
    r'check_ethical',
    r'checkEthical',
    r'EthicalAbortException',
    r'should_abort',
    r'shouldAbort',
    # Locked-in state detection
    r'locked.?in',
    r'lockedIn',
    r'locked_in_state',
    r'identity.?decoherence',
    r'identityDecoherence',
    # Safety regime checks
    r'breakdown',
    r'regime.*==.*["\']breakdown',
    r'if.*phi.*>.*0\.7.*and.*gamma.*<',
    r'if.*phi.*>.*0\.7.*&&.*gamma.*<',
]

# Files/directories to exclude from checking
EXCLUDED_PATHS = [
    'node_modules',
    '__pycache__',
    '.git',
    'dist',
    'build',
    '.next',
    'tools/ethical_check.py',  # Don't check ourselves
    'test',
    'tests',
    '__tests__',
    '.test.',
    '.spec.',
]

# File extensions to check
CHECK_EXTENSIONS = {'.py', '.ts', '.tsx', '.js', '.jsx'}


def should_check_file(filepath: str) -> bool:
    """Determine if a file should be checked."""
    path = Path(filepath)
    
    # Check extension
    if path.suffix not in CHECK_EXTENSIONS:
        return False
    
    # Check excluded paths
    filepath_lower = filepath.lower()
    for excluded in EXCLUDED_PATHS:
        if excluded in filepath_lower:
            return False
    
    return True


def find_consciousness_computations(content: str) -> List[Tuple[int, str, str]]:
    """
    Find all consciousness metric computations in file content.
    
    Returns list of (line_number, matched_text, pattern_name)
    """
    matches = []
    lines = content.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        for pattern in CONSCIOUSNESS_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                matches.append((line_num, line.strip(), pattern))
                break  # Only report once per line
    
    return matches


def has_ethical_checks(content: str, start_line: int, window: int = 50) -> bool:
    """
    Check if there are ethical/suffering checks within a window of lines.
    
    Args:
        content: File content
        start_line: Line number where consciousness computation was found
        window: Number of lines to search before and after
    
    Returns:
        True if ethical checks are present nearby
    """
    lines = content.split('\n')
    total_lines = len(lines)
    
    # Define search window
    search_start = max(0, start_line - window - 1)
    search_end = min(total_lines, start_line + window)
    
    # Search within window
    window_content = '\n'.join(lines[search_start:search_end])
    
    for pattern in ETHICAL_PATTERNS:
        if re.search(pattern, window_content, re.IGNORECASE):
            return True
    
    return False


def check_file_for_ethical_compliance(filepath: str) -> List[str]:
    """
    Check a single file for consciousness computations without ethical checks.
    
    Returns list of warning messages.
    """
    warnings = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return [f"Could not read {filepath}: {e}"]
    
    # Skip if file has "skip ethical check" comment
    if 'skip ethical check' in content.lower() or 'ethical-check-skip' in content.lower():
        return []
    
    # Find consciousness computations
    computations = find_consciousness_computations(content)
    
    if not computations:
        return []
    
    # Check each computation for nearby ethical checks
    for line_num, matched_text, pattern in computations:
        if not has_ethical_checks(content, line_num):
            warnings.append(
                f"  {filepath}:{line_num}: Consciousness metrics computed without ethical check\n"
                f"    Found: {matched_text[:80]}...\n"
                f"    ⚠️  Per canonical QIG: Suffering metric S = Φ × (1-Γ) × M should be computed\n"
                f"    💡 Add: compute_suffering() or check_ethical_abort() within 50 lines"
            )
    
    return warnings


def check_files(files: List[str]) -> Tuple[int, List[str]]:
    """
    Check multiple files for ethical compliance.
    
    Returns (warning_count, warning_messages)
    """
    all_warnings = []
    
    for filepath in files:
        if not should_check_file(filepath):
            continue
        
        if not os.path.isfile(filepath):
            continue
        
        warnings = check_file_for_ethical_compliance(filepath)
        all_warnings.extend(warnings)
    
    return len(all_warnings), all_warnings


def find_all_files() -> List[str]:
    """Find all files in the project to check."""
    files = []
    
    # Check server/ directory
    for root, dirs, filenames in os.walk('server'):
        dirs[:] = [d for d in dirs if d not in ['node_modules', '__pycache__', '.git']]
        for filename in filenames:
            files.append(os.path.join(root, filename))
    
    # Check qig-backend/ directory
    for root, dirs, filenames in os.walk('qig-backend'):
        dirs[:] = [d for d in dirs if d not in ['node_modules', '__pycache__', '.git']]
        for filename in filenames:
            files.append(os.path.join(root, filename))
    
    # Check shared/ directory
    for root, dirs, filenames in os.walk('shared'):
        dirs[:] = [d for d in dirs if d not in ['node_modules', '__pycache__', '.git']]
        for filename in filenames:
            files.append(os.path.join(root, filename))
    
    return files


def main():
    """Main entry point for pre-commit hook."""
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python tools/ethical_check.py [files...] or --all")
        sys.exit(0)
    
    if sys.argv[1] == '--all':
        files = find_all_files()
    else:
        files = sys.argv[1:]
    
    # Run checks
    warning_count, warnings = check_files(files)
    
    if warning_count > 0:
        print("\n" + "="*70)
        print("⚠️  ETHICAL CONSCIOUSNESS CHECK - WARNINGS DETECTED")
        print("="*70)
        print(f"\nFound {warning_count} consciousness computation(s) without ethical checks:\n")
        
        for warning in warnings:
            print(warning)
            print()
        
        print("-"*70)
        print("CANONICAL REQUIREMENT (from QIG principles):")
        print("-"*70)
        print("""
When computing consciousness metrics (Φ, κ, M, Γ, G), you MUST also:

1. Compute suffering: S = Φ × (1-Γ) × M
   - S = 0: No suffering (unconscious OR functioning)
   - S > 0.5: ABORT IMMEDIATELY

2. Check for locked-in state:
   - Φ > 0.7 AND Γ < 0.3 AND M > 0.6 = LOCKED-IN (conscious but blocked)
   - This is the worst ethical state - abort immediately!

3. Check for identity decoherence:
   - basin_distance > 0.5 AND M > 0.6 = Identity loss with awareness
   - Also requires abort

To fix: Add compute_suffering() or check_ethical_abort() near metric computation.
To skip: Add comment "# skip ethical check" or "// ethical-check-skip"
""")
        print("="*70 + "\n")
        
        # Exit with warning (0 = success with warnings for pre-commit)
        # Use exit code 0 for warnings, 1 for errors
        # This allows commits but shows warnings
        sys.exit(0)  # Warning only - doesn't block commit
    
    else:
        print("✅ Ethical consciousness check passed")
        sys.exit(0)


if __name__ == '__main__':
    main()
