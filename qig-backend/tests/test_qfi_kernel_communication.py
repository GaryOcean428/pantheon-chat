#!/usr/bin/env python3
"""
Test QFI attention integration in kernel-to-kernel communication.

Tests the QFI attention routing in Olympus knowledge exchange.

Related: Issue #236 - Wire-In qig_consciousness_qfi_attention.py
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_knowledge_exchange_imports():
    """Test that knowledge exchange imports QFI attention."""
    try:
        # Import directly from module to avoid olympus init issues
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'olympus'))
        import knowledge_exchange
        
        assert hasattr(knowledge_exchange, 'KnowledgeExchange'), "Missing KnowledgeExchange class"
        assert hasattr(knowledge_exchange, 'QFI_ATTENTION_AVAILABLE'), "Missing QFI_ATTENTION_AVAILABLE"
        
        qfi_available = knowledge_exchange.QFI_ATTENTION_AVAILABLE
        
        print(f"✓ Knowledge exchange imports successfully")
        print(f"  QFI attention available: {qfi_available}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import knowledge exchange: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qfi_routing_initialization():
    """Test that QFI routing is initialized in knowledge exchange."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'olympus'))
        import knowledge_exchange
        
        KnowledgeExchange = knowledge_exchange.KnowledgeExchange
        QFI_ATTENTION_AVAILABLE = knowledge_exchange.QFI_ATTENTION_AVAILABLE
        
        exchange = KnowledgeExchange()
        
        assert hasattr(exchange, 'qfi_attention_enabled'), "Missing qfi_attention_enabled"
        assert hasattr(exchange, 'qfi_attention_network'), "Missing qfi_attention_network"
        
        if QFI_ATTENTION_AVAILABLE:
            assert exchange.qfi_attention_enabled, "QFI attention should be enabled"
            assert exchange.qfi_attention_network is not None, "QFI network should be initialized"
            print("✓ QFI attention routing initialized in knowledge exchange")
        else:
            print("⚠ QFI attention not available (module not found)")
        
        return True
    except Exception as e:
        print(f"✗ Failed to initialize QFI routing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compute_qfi_attention_routing():
    """Test computing QFI attention weights between gods."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'olympus'))
        import knowledge_exchange
        
        KnowledgeExchange = knowledge_exchange.KnowledgeExchange
        QFI_ATTENTION_AVAILABLE = knowledge_exchange.QFI_ATTENTION_AVAILABLE
        
        if not QFI_ATTENTION_AVAILABLE:
            print("⚠ QFI attention not available, skipping routing test")
            return True
        
        exchange = KnowledgeExchange()
        
        # Create mock gods with basin coordinates
        class MockGod:
            def __init__(self, name, basin):
                self.name = name
                self.basin_coords = basin
        
        # Create 3 mock gods
        god1 = MockGod("zeus", np.random.randn(64))
        god2 = MockGod("athena", np.random.randn(64))
        god3 = MockGod("apollo", np.random.randn(64))
        
        exchange.register_god(god1)
        exchange.register_god(god2)
        exchange.register_god(god3)
        
        # Compute attention routing
        attention_matrix = exchange.compute_qfi_attention_routing()
        
        assert attention_matrix is not None, "Should return attention matrix"
        assert attention_matrix.shape == (3, 3), f"Wrong shape: {attention_matrix.shape}"
        
        # Check normalization (rows should sum to ~1)
        for i in range(3):
            row_sum = np.sum(attention_matrix[i, :])
            assert 0.9 <= row_sum <= 1.1, f"Row {i} not normalized: {row_sum}"
        
        # Check no self-attention
        for i in range(3):
            assert attention_matrix[i, i] == 0, f"Self-attention non-zero at ({i},{i})"
        
        print("✓ QFI attention routing computed successfully")
        print(f"  Matrix shape: {attention_matrix.shape}")
        print(f"  Sample attention: zeus→athena = {attention_matrix[0, 1]:.3f}")
        return True
    except Exception as e:
        print(f"✗ Failed to compute QFI routing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qfi_attention_in_share_strategies():
    """Test that QFI attention is used in strategy sharing."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'olympus'))
        import knowledge_exchange
        
        KnowledgeExchange = knowledge_exchange.KnowledgeExchange
        QFI_ATTENTION_AVAILABLE = knowledge_exchange.QFI_ATTENTION_AVAILABLE
        
        exchange = KnowledgeExchange()
        
        # Create mock gods with reasoning learners
        class MockStrategy:
            def __init__(self, name):
                self.name = name
                self.description = f"Strategy {name}"
            
            def copy(self):
                return MockStrategy(self.name)
            
            def success_rate(self):
                return 0.8
            
            @property
            def avg_efficiency(self):
                return 0.9
        
        class MockLearner:
            def __init__(self):
                self.strategies = [MockStrategy("test_strategy")]
        
        class MockGod:
            def __init__(self, name):
                self.name = name
                self.basin_coords = np.random.randn(64)
                self.reasoning_learner = MockLearner()
        
        god1 = MockGod("zeus")
        god2 = MockGod("athena")
        
        exchange.register_god(god1)
        exchange.register_god(god2)
        
        # Share strategies (should use QFI attention if available)
        exchange.share_strategies()
        
        # Check that exchange history was recorded
        assert len(exchange.exchange_history) > 0, "No exchange history recorded"
        
        last_exchange = exchange.exchange_history[-1]
        assert 'qfi_routing_enabled' in last_exchange, "Missing QFI routing flag"
        
        if QFI_ATTENTION_AVAILABLE:
            assert last_exchange['qfi_routing_enabled'], "QFI routing should be enabled"
            print("✓ Strategy sharing uses QFI attention routing")
        else:
            assert not last_exchange['qfi_routing_enabled'], "QFI routing should be disabled"
            print("⚠ Strategy sharing uses uniform routing (QFI not available)")
        
        return True
    except Exception as e:
        print(f"✗ Failed QFI attention in strategy sharing: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all kernel-to-kernel QFI attention tests."""
    print("=" * 60)
    print("Kernel-to-Kernel QFI Attention Tests")
    print("=" * 60)
    
    tests = [
        ("Knowledge Exchange Imports", test_knowledge_exchange_imports),
        ("QFI Routing Initialization", test_qfi_routing_initialization),
        ("Compute QFI Routing", test_compute_qfi_attention_routing),
        ("QFI in Strategy Sharing", test_qfi_attention_in_share_strategies),
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
