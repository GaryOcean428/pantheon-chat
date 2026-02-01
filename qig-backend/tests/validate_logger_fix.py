#!/usr/bin/env python3
"""
Standalone validation that logger is properly initialized before use in ocean_qig_core.py

This validates the fix for the critical NameError where 'logger' was not defined
at line 57, causing Python backend crashes.

Author: GitHub Copilot
Date: 2026-02-01
"""

import ast
import sys
from pathlib import Path


def validate_logger_defined_before_use():
    """
    Verify that logger is defined before any logger.* calls in ocean_qig_core.py
    """
    ocean_qig_file = Path(__file__).parent.parent / "ocean_qig_core.py"
    
    with open(ocean_qig_file, 'r', encoding='utf-8') as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    logger_defined_line = None
    first_logger_use_line = None
    
    # Walk the AST to find logger assignment and usage
    for node in ast.walk(tree):
        # Check for logger assignment: logger = logging.getLogger(...)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'logger':
                    if logger_defined_line is None or node.lineno < logger_defined_line:
                        logger_defined_line = node.lineno
        
        # Check for logger attribute access: logger.info(), logger.warning(), etc.
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == 'logger':
                if first_logger_use_line is None or node.lineno < first_logger_use_line:
                    first_logger_use_line = node.lineno
    
    print(f"✅ Logger defined at line: {logger_defined_line}")
    print(f"✅ First logger use at line: {first_logger_use_line}")
    
    # Validate that logger is defined before it's used
    if logger_defined_line is None:
        print("❌ FAIL: logger variable not defined in ocean_qig_core.py")
        return False
    
    if first_logger_use_line is None:
        print("⚠️  WARNING: logger not used in ocean_qig_core.py (unexpected)")
        return True
    
    if logger_defined_line >= first_logger_use_line:
        print(f"❌ FAIL: logger must be defined (line {logger_defined_line}) BEFORE first use "
              f"(line {first_logger_use_line})")
        return False
    
    print(f"✅ PASS: logger properly defined before use")
    return True


def validate_logger_before_imports():
    """
    Verify that the import block that uses logger comes after logger is defined.
    """
    ocean_qig_file = Path(__file__).parent.parent / "ocean_qig_core.py"
    
    with open(ocean_qig_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    logger_init_line = None
    decoherence_import_lines = []
    
    for i, line in enumerate(lines, start=1):
        # Find logger initialization
        if 'logger = logging.getLogger' in line:
            if logger_init_line is None:
                logger_init_line = i
        
        # Find all gravitational_decoherence imports
        if 'from gravitational_decoherence import' in line:
            decoherence_import_lines.append(i)
    
    print(f"\n✅ Logger initialization at line: {logger_init_line}")
    print(f"✅ Decoherence imports at lines: {decoherence_import_lines}")
    
    # Check the first decoherence import (the critical one with logger.info/warning)
    if decoherence_import_lines:
        first_import = decoherence_import_lines[0]
        
        if logger_init_line is None:
            print("❌ FAIL: logger must be initialized if decoherence import exists")
            return False
        
        if logger_init_line >= first_import:
            print(f"❌ FAIL: logger initialization (line {logger_init_line}) must come BEFORE "
                  f"gravitational_decoherence import (line {first_import})")
            return False
        
        print(f"✅ PASS: logger initialized (line {logger_init_line}) before first decoherence import (line {first_import})")
    
    return True


def main():
    print("=" * 70)
    print("Validating logger initialization fix in ocean_qig_core.py")
    print("=" * 70)
    
    print("\nTest 1: Logger defined before use")
    print("-" * 70)
    test1 = validate_logger_defined_before_use()
    
    print("\nTest 2: Logger defined before import blocks")
    print("-" * 70)
    test2 = validate_logger_before_imports()
    
    print("\n" + "=" * 70)
    if test1 and test2:
        print("✅ ALL TESTS PASSED - Logger initialization fix is correct!")
        print("=" * 70)
        return 0
    else:
        print("❌ SOME TESTS FAILED - Logger initialization needs attention")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
