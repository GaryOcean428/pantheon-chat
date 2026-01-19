#!/usr/bin/env python3
"""
Run All Coherence Tests

Executes all three test configurations and generates comparison report.

Usage:
    python3 tests/coherence/run_all_tests.py

Author: QIG Consciousness Project
Date: January 2026
"""

import sys
import subprocess
from pathlib import Path


def run_test(script_name: str) -> int:
    """
    Run a test script.
    
    Args:
        script_name: Script filename
        
    Returns:
        Exit code
    """
    script_path = Path(__file__).parent / script_name
    
    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=script_path.parent
    )
    
    return result.returncode


def main():
    """Main test runner."""
    print("\n" + "="*80)
    print("QIG COHERENCE TEST SUITE")
    print("="*80)
    print("\nRunning all test configurations...")
    
    # Track failures
    failures = []
    
    # Run tests in order
    tests = [
        'test_skeleton_baseline.py',
        'test_pure_geometric.py',
        'test_plan_realize_repair.py',
    ]
    
    for test in tests:
        exit_code = run_test(test)
        if exit_code != 0:
            failures.append((test, exit_code))
            print(f"\n✗ {test} failed with exit code {exit_code}")
        else:
            print(f"\n✓ {test} passed")
    
    # Run comparison if all tests passed
    if not failures:
        print("\n" + "="*80)
        print("All tests passed! Running comparison...")
        print("="*80)
        
        exit_code = run_test('compare_architectures.py')
        if exit_code != 0:
            failures.append(('compare_architectures.py', exit_code))
    else:
        print("\n⚠️  Skipping comparison due to test failures")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUITE SUMMARY")
    print("="*80)
    
    if not failures:
        print("\n✓ All tests passed!")
        print("\nResults:")
        results_dir = Path(__file__).parent / "results"
        for result_file in sorted(results_dir.glob("*.json")):
            print(f"  - {result_file.name}")
        
        html_report = results_dir / "comparison_report.html"
        if html_report.exists():
            print(f"\n📊 HTML Report: {html_report}")
        
        return 0
    else:
        print(f"\n✗ {len(failures)} test(s) failed:")
        for test, code in failures:
            print(f"  - {test} (exit code: {code})")
        return 1


if __name__ == '__main__':
    sys.exit(main())
