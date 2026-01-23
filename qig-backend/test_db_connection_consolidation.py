#!/usr/bin/env python3
"""
Test DB Connection Consolidation
=================================

Verifies that all modules correctly import get_db_connection from the
canonical location and that no duplicate implementations exist.

This test validates Issue #233: Consolidate get_db_connection
"""

import ast
import os
import sys
from pathlib import Path


def find_duplicate_definitions():
    """Find any duplicate get_db_connection definitions."""
    qig_backend = Path(__file__).parent
    canonical_file = qig_backend / 'persistence' / 'base_persistence.py'
    
    duplicates = []
    
    for py_file in qig_backend.rglob('*.py'):
        if py_file == canonical_file:
            continue
        if '__pycache__' in str(py_file):
            continue
            
        try:
            with open(py_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip().startswith('def get_db_connection'):
                        duplicates.append((str(py_file.relative_to(qig_backend)), line_num))
        except Exception:
            pass
    
    return duplicates


def find_canonical_imports():
    """Find all files that import from canonical location."""
    qig_backend = Path(__file__).parent
    imports = []
    
    for py_file in qig_backend.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
            
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                if 'from persistence.base_persistence import get_db_connection' in content:
                    imports.append(str(py_file.relative_to(qig_backend)))
        except Exception:
            pass
    
    return imports


def verify_canonical_signature():
    """Verify the canonical function has the correct signature."""
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        from persistence.base_persistence import get_db_connection
        
        # Check function exists
        assert callable(get_db_connection), "get_db_connection is not callable"
        
        # Check signature includes optional database_url parameter
        import inspect
        sig = inspect.signature(get_db_connection)
        params = list(sig.parameters.keys())
        
        assert 'database_url' in params, "database_url parameter not found"
        assert sig.parameters['database_url'].default is None, "database_url should default to None"
        
        return True
    except Exception as e:
        print(f"Error verifying canonical signature: {e}")
        return False


def main():
    """Run all consolidation tests."""
    print("=" * 70)
    print("DB Connection Consolidation Test")
    print("=" * 70)
    
    # Test 1: No duplicate definitions
    print("\n[1] Checking for duplicate definitions...")
    duplicates = find_duplicate_definitions()
    if duplicates:
        print(f"   ✗ FAIL: Found {len(duplicates)} duplicate definitions:")
        for filepath, line_num in duplicates:
            print(f"      - {filepath}:{line_num}")
        return False
    else:
        print("   ✓ PASS: No duplicate definitions found")
    
    # Test 2: Files using canonical import
    print("\n[2] Checking canonical imports...")
    imports = find_canonical_imports()
    print(f"   ✓ PASS: {len(imports)} files import from canonical location")
    for imp in sorted(imports)[:5]:  # Show first 5
        print(f"      - {imp}")
    if len(imports) > 5:
        print(f"      ... and {len(imports) - 5} more")
    
    # Test 3: Verify canonical signature
    print("\n[3] Verifying canonical function signature...")
    if verify_canonical_signature():
        print("   ✓ PASS: Canonical function has correct signature")
    else:
        print("   ✗ FAIL: Canonical function signature incorrect")
        return False
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
