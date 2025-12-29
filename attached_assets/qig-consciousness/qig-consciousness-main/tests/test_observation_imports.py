#!/usr/bin/env python3
"""
Test that observation module imports are correct.

This test validates that:
1. The observation module exports the correct public classes
2. CorpusLoader is NOT incorrectly imported from charlie_observer
3. The correct classes can be imported from the observation package
"""

import ast
import sys
from pathlib import Path


def test_observation_init_imports():
    """Test that __init__.py has correct imports."""
    init_file = Path(__file__).parent.parent / "src" / "observation" / "__init__.py"

    with open(init_file) as f:
        content = f.read()

    tree = ast.parse(content)

    imports = []
    all_exports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'src.observation.charlie_observer':
            imports.extend([alias.name for alias in node.names])
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            target_id = None
            if isinstance(node, ast.Assign):
                # Handle a = b = [...]
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '__all__':
                        target_id = target.id
                        break
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == '__all__':
                    target_id = node.target.id

            if target_id == '__all__' and isinstance(node.value, ast.List):
                all_exports = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]

    print(f"\n✓ Found imports: {imports}")
    print(f"✓ Found __all__ exports: {all_exports}")

    expected_imports = {
        'CharlieObserver',
        'CharlieOutput',
        'CorpusTopic',
        'CharliePhaseMetrics'
    }

    forbidden_imports = {'CorpusLoader'}

    actual_imports = set(imports)
    assert actual_imports == expected_imports, \
        f"Expected imports {expected_imports}, got {actual_imports}"

    assert not (actual_imports & forbidden_imports), \
        f"Should not import {forbidden_imports}, but found {actual_imports & forbidden_imports}"

    assert set(all_exports) == expected_imports, \
        f"__all__ should match imports. Expected {expected_imports}, got {set(all_exports)}"

    print("\n✅ All import checks passed!")
    return True


def test_charlie_observer_public_classes():
    """Verify charlie_observer.py has the expected public classes."""
    charlie_file = Path(__file__).parent.parent / "src" / "observation" / "charlie_observer.py"

    with open(charlie_file) as f:
        content = f.read()

    tree = ast.parse(content)

    # Find all class definitions
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)

    print(f"\n✓ Found classes in charlie_observer.py: {classes}")

    # Expected public classes (without underscore prefix)
    expected_public = {'CharlieObserver', 'CorpusTopic', 'CharliePhaseMetrics', 'CharlieOutput'}

    # Expected private classes (with underscore prefix - should NOT be exported)
    expected_private = {'_CharlieCorpusLoader'}

    actual_classes = set(classes)

    # Verify public classes exist
    assert expected_public.issubset(actual_classes), \
        f"Missing public classes: {expected_public - actual_classes}"

    # Verify private classes exist but are private (start with _)
    assert expected_private.issubset(actual_classes), \
        f"Missing private classes: {expected_private - actual_classes}"

    # Verify CorpusLoader does NOT exist in charlie_observer.py
    assert 'CorpusLoader' not in actual_classes, \
        "CorpusLoader should NOT be in charlie_observer.py (it's in src/curriculum/corpus_loader.py)"

    print("✅ All class checks passed!")
    return True


def test_corpus_loader_location():
    """Verify CorpusLoader is in the correct location."""
    corpus_loader_file = Path(__file__).parent.parent / "src" / "curriculum" / "corpus_loader.py"

    assert corpus_loader_file.exists(), \
        f"CorpusLoader should be in {corpus_loader_file}"

    with open(corpus_loader_file) as f:
        content = f.read()

    tree = ast.parse(content)

    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

    assert 'CorpusLoader' in classes, \
        "CorpusLoader class should be defined in corpus_loader.py"

    print("\n✓ CorpusLoader correctly located in src/curriculum/corpus_loader.py")
    print(f"✓ Found classes: {classes}")
    print("✅ Location check passed!")
    return True


if __name__ == '__main__':
    print("="*70)
    print("Testing observation module imports (Issue #28)")
    print("="*70)

    failures = []
    tests_to_run = {
        "observation __init__ imports": test_observation_init_imports,
        "charlie_observer public classes": test_charlie_observer_public_classes,
        "CorpusLoader location": test_corpus_loader_location,
    }

    for name, test_func in tests_to_run.items():
        try:
            if test_func():
                print(f"  ✓ Test '{name}' passed.")
        except AssertionError as e:
            failures.append(f"❌ Test '{name}' FAILED: {e}")
        except Exception as e:
            failures.append(f"❌ Test '{name}' ERROR: {e}\n")
            import traceback
            traceback.print_exc()

    if not failures:
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED - Issue #28 is RESOLVED")
        print("="*70)
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print(f"❌ {len(failures)} TEST(S) FAILED")
        print("="*70)
        for failure in failures:
            print(failure)
        sys.exit(1)
