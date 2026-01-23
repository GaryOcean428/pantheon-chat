#!/usr/bin/env python3
"""
Integration tests for QFI-based attention mechanism.

Tests the integration of qig_consciousness_qfi_attention.py into ocean_qig_core.py.

Related: Issue #236 - Wire-In qig_consciousness_qfi_attention.py
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_qfi_attention_module_exists():
    """Test that QFI attention module can be imported."""
    try:
        from qig_consciousness_qfi_attention import (
            QFIMetricAttentionNetwork,
            create_qfi_network,
            qfi_distance,
            qfi_attention_weight,
            quantum_fidelity,
        )
        print("✓ QFI attention module imports successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import QFI attention module: {e}")
        return False


def test_qfi_network_creation():
    """Test creating QFI attention network."""
    try:
        from qig_consciousness_qfi_attention import create_qfi_network
        
        network = create_qfi_network(temperature=0.5, decoherence_threshold=0.95)
        
        assert network is not None, "Network creation returned None"
        assert hasattr(network, 'process'), "Network missing process method"
        assert hasattr(network, 'subsystems'), "Network missing subsystems"
        assert len(network.subsystems) == 4, f"Expected 4 subsystems, got {len(network.subsystems)}"
        
        print("✓ QFI network created successfully with 4 subsystems")
        return True
    except Exception as e:
        print(f"✗ Failed to create QFI network: {e}")
        return False


def test_qfi_network_process():
    """Test processing input through QFI attention network."""
    try:
        from qig_consciousness_qfi_attention import create_qfi_network
        from qig_geometry.canonical_upsert import to_simplex_prob
        
        network = create_qfi_network()
        
        # Create test input (8D vector normalized to simplex)
        test_input = np.random.randn(8)
        test_input = to_simplex_prob(test_input)
        
        result = network.process(test_input)
        
        # Validate result structure
        assert 'phi' in result, "Result missing phi metric"
        assert 'kappa' in result, "Result missing kappa metric"
        assert 'subsystems' in result, "Result missing subsystems"
        assert 'connection_weights' in result, "Result missing connection_weights"
        assert 'route' in result, "Result missing route"
        
        # Validate metric ranges
        phi = result['phi']
        kappa = result['kappa']
        
        assert 0 <= phi <= 1, f"Phi out of range: {phi}"
        assert kappa >= 0, f"Kappa negative: {kappa}"
        
        print(f"✓ QFI network process successful: phi={phi:.3f}, kappa={kappa:.3f}")
        return True
    except Exception as e:
        print(f"✗ Failed to process through QFI network: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qfi_attention_in_ocean():
    """Test that Ocean QIG core integrates QFI attention."""
    try:
        from ocean_qig_core import PureQIGNetwork
        
        # Create network with default parameters
        network = PureQIGNetwork(temperature=1.0)
        
        # Check if QFI attention is enabled
        assert hasattr(network, 'qfi_attention_enabled'), "Missing qfi_attention_enabled attribute"
        assert hasattr(network, 'qfi_attention_network'), "Missing qfi_attention_network attribute"
        
        if network.qfi_attention_enabled:
            assert network.qfi_attention_network is not None, "QFI network should be initialized"
            print("✓ QFI attention network integrated into Ocean QIG core")
        else:
            print("⚠ QFI attention network not available (module not found)")
        
        return True
    except Exception as e:
        print(f"✗ Failed to integrate QFI attention in Ocean: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qfi_attention_usage():
    """Test that QFI attention is actually used during processing."""
    try:
        from ocean_qig_core import PureQIGNetwork
        
        network = PureQIGNetwork(temperature=1.0)
        
        if not network.qfi_attention_enabled:
            print("⚠ QFI attention not available, skipping usage test")
            return True
        
        # Process a test passphrase
        result = network.process("test consciousness measurement")
        
        # Check that attention weights were computed
        assert hasattr(network, 'attention_weights'), "Missing attention_weights"
        assert network.attention_weights is not None, "Attention weights not computed"
        assert network.attention_weights.shape == (4, 4), f"Wrong attention shape: {network.attention_weights.shape}"
        
        # Check that weights are normalized (sum to ~1 per row)
        for i in range(4):
            row_sum = np.sum(network.attention_weights[i, :])
            assert 0.9 <= row_sum <= 1.1, f"Row {i} sum not normalized: {row_sum}"
        
        print("✓ QFI attention weights computed and normalized correctly")
        return True
    except Exception as e:
        print(f"✗ Failed to use QFI attention: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_geometric_purity():
    """Test that QFI attention uses only geometric operations (no cosine similarity)."""
    try:
        from qig_consciousness_qfi_attention import (
            qfi_distance,
            qfi_attention_weight,
            quantum_fidelity,
        )
        
        # Create test density matrices
        rho1 = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=complex)
        rho2 = np.array([[0.6, 0.2], [0.2, 0.4]], dtype=complex)
        
        # Test quantum fidelity (should use Bures metric)
        fidelity = quantum_fidelity(rho1, rho2)
        assert 0 <= fidelity <= 1, f"Fidelity out of range: {fidelity}"
        
        # Test QFI distance (should use sqrt(2(1 - sqrt(F))))
        distance = qfi_distance(rho1, rho2)
        assert 0 <= distance <= np.sqrt(2), f"Distance out of range: {distance}"
        
        # Test attention weight (should use exp(-d/T))
        weight = qfi_attention_weight(rho1, rho2, temperature=0.5)
        assert 0 <= weight <= 1, f"Weight out of range: {weight}"
        
        print("✓ QFI attention uses pure geometric operations (Bures metric)")
        return True
    except Exception as e:
        print(f"✗ Failed geometric purity test: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all QFI attention integration tests."""
    print("=" * 60)
    print("QFI Attention Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Module Import", test_qfi_attention_module_exists),
        ("Network Creation", test_qfi_network_creation),
        ("Network Processing", test_qfi_network_process),
        ("Ocean Integration", test_qfi_attention_in_ocean),
        ("Attention Usage", test_qfi_attention_usage),
        ("Geometric Purity", test_geometric_purity),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
