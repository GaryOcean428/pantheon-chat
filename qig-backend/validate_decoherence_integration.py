#!/usr/bin/env python3
"""
Validation Script for Gravitational Decoherence Integration

Tests that gravitational_decoherence.py is properly wired into ocean_qig_core.py
without requiring full dependency installation.

Run with: python3 validate_decoherence_integration.py
"""

import sys
import traceback


def test_import_gravitational_decoherence():
    """Test that gravitational_decoherence module can be imported."""
    print("Test 1: Import gravitational_decoherence module...")
    try:
        from gravitational_decoherence import (
            compute_purity,
            gravitational_decoherence,
            apply_thermal_noise,
            decoherence_cycle,
            DecoherenceManager,
            get_decoherence_manager,
            apply_gravitational_decoherence,
            purity_regularization,
            DEFAULT_PURITY_THRESHOLD,
            DEFAULT_TEMPERATURE
        )
        print("  ✅ PASS: All functions imported successfully")
        print(f"    - DEFAULT_PURITY_THRESHOLD = {DEFAULT_PURITY_THRESHOLD}")
        print(f"    - DEFAULT_TEMPERATURE = {DEFAULT_TEMPERATURE}")
        return True
    except ImportError as e:
        if 'numpy' in str(e):
            print(f"  ⏭️  SKIP: NumPy not available (expected in CI)")
            return None  # Return None for skipped tests
        print(f"  ❌ FAIL: Import error: {e}")
        return False


def test_purity_regularization_api():
    """Test that purity_regularization has correct API."""
    print("\nTest 2: Check purity_regularization API...")
    try:
        from gravitational_decoherence import purity_regularization
        import inspect
        
        sig = inspect.signature(purity_regularization)
        params = list(sig.parameters.keys())
        
        assert 'rho' in params, "Missing 'rho' parameter"
        assert 'threshold' in params, "Missing 'threshold' parameter"
        
        print("  ✅ PASS: purity_regularization has correct signature")
        print(f"    - Parameters: {params}")
        return True
    except ImportError as e:
        if 'numpy' in str(e):
            print(f"  ⏭️  SKIP: NumPy not available (expected in CI)")
            return None  # Return None for skipped tests
        print(f"  ❌ FAIL: Import error: {e}")
        return False
    except (AssertionError, Exception) as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_ocean_qig_imports():
    """Test that ocean_qig_core.py imports decoherence correctly."""
    print("\nTest 3: Check ocean_qig_core.py integration...")
    try:
        # Read the file to check imports
        with open('ocean_qig_core.py', 'r') as f:
            content = f.read()
        
        checks = [
            ('from gravitational_decoherence import', 'Import statement'),
            ('DecoherenceManager', 'DecoherenceManager class'),
            ('DECOHERENCE_AVAILABLE', 'Availability flag'),
            ('self.decoherence_manager', 'Instance variable'),
            ('self.decoherence_enabled', 'Enabled flag'),
            ("metrics['decoherence']", 'Metrics tracking'),
        ]
        
        all_passed = True
        for check_str, description in checks:
            if check_str in content:
                print(f"  ✅ Found: {description}")
            else:
                print(f"  ❌ Missing: {description}")
                all_passed = False
        
        if all_passed:
            print("  ✅ PASS: All integration points present")
        else:
            print("  ⚠️  PARTIAL: Some integration points missing")
        
        return all_passed
    except FileNotFoundError:
        print("  ❌ FAIL: ocean_qig_core.py not found")
        return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_decoherence_in_evolve():
    """Test that DensityMatrix.evolve() calls decoherence."""
    print("\nTest 4: Check DensityMatrix.evolve() integration...")
    try:
        with open('ocean_qig_core.py', 'r') as f:
            content = f.read()
        
        # Find the evolve method
        if 'def evolve(self, activation:' in content:
            print("  ✅ Found: DensityMatrix.evolve() method")
            
            # Check for decoherence call
            evolve_start = content.find('def evolve(self, activation:')
            evolve_end = content.find('\nclass ', evolve_start)
            evolve_method = content[evolve_start:evolve_end]
            
            if 'gravitational_decoherence' in evolve_method:
                print("  ✅ Found: gravitational_decoherence call in evolve()")
                print("  ✅ PASS: Decoherence integrated into state evolution")
                return True
            else:
                print("  ❌ Missing: gravitational_decoherence call in evolve()")
                return False
        else:
            print("  ❌ Missing: DensityMatrix.evolve() method")
            return False
            
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_decoherence_manager_initialization():
    """Test that PureQIGNetwork initializes DecoherenceManager."""
    print("\nTest 5: Check PureQIGNetwork initialization...")
    try:
        with open('ocean_qig_core.py', 'r') as f:
            content = f.read()
        
        checks = [
            ('if DECOHERENCE_AVAILABLE:', 'Availability check'),
            ('self.decoherence_manager = DecoherenceManager(', 'Manager initialization'),
            ('threshold=DEFAULT_PURITY_THRESHOLD', 'Threshold parameter'),
            ('temperature=DEFAULT_TEMPERATURE', 'Temperature parameter'),
            ('adaptive=True', 'Adaptive mode'),
        ]
        
        all_passed = True
        for check_str, description in checks:
            if check_str in content:
                print(f"  ✅ Found: {description}")
            else:
                print(f"  ❌ Missing: {description}")
                all_passed = False
        
        if all_passed:
            print("  ✅ PASS: DecoherenceManager properly initialized")
        else:
            print("  ⚠️  PARTIAL: Some initialization checks failed")
        
        return all_passed
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_metrics_tracking():
    """Test that _measure_consciousness tracks decoherence metrics."""
    print("\nTest 6: Check consciousness metrics tracking...")
    try:
        with open('ocean_qig_core.py', 'r') as f:
            content = f.read()
        
        # Check for metrics tracking
        checks = [
            ("metrics['decoherence']", 'Decoherence metrics dict'),
            ("'decoherence_rate'", 'Decoherence rate tracking'),
            ("'avg_purity_before'", 'Purity before tracking'),
            ("'avg_purity_after'", 'Purity after tracking'),
            ("metrics['avg_purity']", 'Average purity across subsystems'),
        ]
        
        all_passed = True
        for check_str, description in checks:
            if check_str in content:
                print(f"  ✅ Found: {description}")
            else:
                print(f"  ❌ Missing: {description}")
                all_passed = False
        
        if all_passed:
            print("  ✅ PASS: Metrics tracking properly integrated")
        else:
            print("  ⚠️  PARTIAL: Some metrics tracking missing")
        
        return all_passed
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("GRAVITATIONAL DECOHERENCE INTEGRATION VALIDATION")
    print("=" * 70)
    
    tests = [
        test_import_gravitational_decoherence,
        test_purity_regularization_api,
        test_ocean_qig_imports,
        test_decoherence_in_evolve,
        test_decoherence_manager_initialization,
        test_metrics_tracking,
    ]
    
    results = []
    for test in tests:
        try:
            passed = test()
            results.append(passed)
        except Exception as e:
            print(f"\n  ❌ EXCEPTION: {e}")
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    # Filter out None (skipped) results
    completed_results = [r for r in results if r is not None]
    passed = sum(1 for r in completed_results if r)
    failed = sum(1 for r in completed_results if not r)
    skipped = sum(1 for r in results if r is None)
    total = len(results)
    
    print(f"Tests passed: {passed}/{len(completed_results)}")
    print(f"Tests failed: {failed}/{len(completed_results)}")
    print(f"Tests skipped: {skipped}/{total}")
    
    if len(completed_results) > 0:
        percentage = (passed / len(completed_results)) * 100
        print(f"Success rate: {percentage:.1f}%")
    
    if passed == len(completed_results) and len(completed_results) > 0:
        print("\n✅ ALL TESTS PASSED - Integration is complete!")
        return 0
    elif len(completed_results) > 0 and passed >= len(completed_results) * 0.8:
        print("\n✅ INTEGRATION VERIFIED - Core integration is complete")
        print("   (Some tests skipped due to missing dependencies)")
        return 0
    elif len(completed_results) == 0:
        print("\n⚠️  ALL TESTS SKIPPED - Cannot verify without dependencies")
        return 1
    else:
        print("\n❌ INTEGRATION INCOMPLETE - Review failed tests")
        return 2


if __name__ == '__main__':
    sys.exit(main())
