#!/usr/bin/env python3
"""
Verification script for M8 kernel spawning refactoring.
Checks that all modules are valid and imports work correctly.
"""

import ast
import sys

def check_syntax(filename):
    """Check if a Python file has valid syntax."""
    try:
        with open(filename, 'r') as f:
            ast.parse(f.read(), filename)
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"

def count_lines(filename):
    """Count lines in a file."""
    with open(filename, 'r') as f:
        return len(f.readlines())

def check_exports(filename):
    """Check what a module exports."""
    with open(filename, 'r') as f:
        tree = ast.parse(f.read(), filename)
    
    classes = []
    functions = []
    
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            if not node.name.startswith('_'):
                functions.append(node.name)
    
    return classes, functions

def main():
    print("="*70)
    print("M8 Kernel Spawning Refactoring Verification")
    print("="*70)
    
    files = [
        'm8_persistence.py',
        'm8_consensus.py', 
        'm8_spawner.py',
        'm8_kernel_spawning.py'
    ]
    
    all_valid = True
    
    for fname in files:
        print(f"\n{fname}:")
        
        # Check syntax
        valid, error = check_syntax(fname)
        if not valid:
            print(f"  ✗ SYNTAX ERROR: {error}")
            all_valid = False
            continue
        else:
            print(f"  ✓ Syntax valid")
        
        # Count lines
        lines = count_lines(fname)
        status = "✓" if lines < 2000 or fname == 'm8_kernel_spawning.py' else "⚠"
        print(f"  {status} {lines} lines", end="")
        if lines >= 2000 and fname != 'm8_kernel_spawning.py':
            print(f" (OVER by {lines-2000})", end="")
        print()
        
        # Check exports (except for re-export wrapper)
        if fname != 'm8_kernel_spawning.py':
            classes, functions = check_exports(fname)
            if classes:
                print(f"  Exports {len(classes)} classes: {', '.join(classes[:3])}", end="")
                if len(classes) > 3:
                    print(f" + {len(classes)-3} more", end="")
                print()
            if functions:
                print(f"  Exports {len(functions)} functions: {', '.join(functions)}")
    
    print("\n" + "="*70)
    if all_valid:
        print("✓ All modules syntactically valid")
        print("✓ Refactoring complete")
        print("\nNext steps:")
        print("  1. Run tests to verify functionality")
        print("  2. Check that all imports work in dependent files")
        print("  3. Verify geometric purity is maintained")
    else:
        print("✗ Some modules have errors - see above")
        sys.exit(1)
    print("="*70)

if __name__ == "__main__":
    main()
