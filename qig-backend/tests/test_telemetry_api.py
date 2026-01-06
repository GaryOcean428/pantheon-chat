"""
Test Capability Telemetry API

Validates that the telemetry API blueprint is properly structured
and can be imported without errors.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_telemetry_api_imports():
    """Test that telemetry API can be imported."""
    try:
        from olympus.telemetry_api import (
            telemetry_bp,
            register_telemetry_routes,
            initialize_god_telemetry,
            get_registry
        )
        assert telemetry_bp is not None
        assert callable(register_telemetry_routes)
        assert callable(initialize_god_telemetry)
        assert callable(get_registry)
        print("✓ Telemetry API imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_telemetry_blueprint_structure():
    """Test that the blueprint has expected routes."""
    try:
        from olympus.telemetry_api import telemetry_bp
        
        # Get all routes registered on the blueprint
        routes = []
        for rule in telemetry_bp.url_map.iter_rules():
            routes.append(rule.rule)
        
        # Expected routes
        expected_routes = [
            '/fleet',
            '/kernel/<kernel_id>/capabilities',
            '/kernel/<kernel_id>/summary',
            '/kernels',
            '/all',
            '/health'
        ]
        
        # Check each expected route exists (will have /api/telemetry prefix)
        for expected in expected_routes:
            found = any(expected in route for route in routes)
            if found:
                print(f"✓ Route found: {expected}")
            else:
                print(f"✗ Route missing: {expected}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Blueprint structure test error: {e}")
        return False


def test_gods_initialization_structure():
    """Test that god initialization data is structured correctly."""
    try:
        from olympus.telemetry_api import initialize_god_telemetry
        
        # The function should be callable
        assert callable(initialize_god_telemetry)
        
        print("✓ God initialization function is callable")
        return True
    except Exception as e:
        print(f"✗ God initialization test error: {e}")
        return False


if __name__ == '__main__':
    print("\n=== Testing Telemetry API Structure ===\n")
    
    tests = [
        ("Import Test", test_telemetry_api_imports),
        ("Blueprint Structure", test_telemetry_blueprint_structure),
        ("God Initialization", test_gods_initialization_structure),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((name, False))
    
    print("\n=== Test Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
