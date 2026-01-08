"""
Tests for Self-Healing System

Basic integration tests for the 3-layer architecture.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from self_healing import (
    GeometricHealthMonitor,
    CodeFitnessEvaluator,
    SelfHealingEngine,
    create_self_healing_system
)


def test_geometric_monitor():
    """Test geometric health monitoring."""
    print("\n=== Testing Geometric Monitor ===")
    
    monitor = GeometricHealthMonitor(history_size=100)
    
    # Capture a healthy snapshot
    state1 = {
        "phi": 0.72,
        "kappa_eff": 64.5,
        "basin_coords": np.random.randn(64),
        "confidence": 0.8,
        "surprise": 0.3,
        "agency": 0.7,
        "error_rate": 0.01,
        "avg_latency": 150.0,
        "label": "healthy state"
    }
    
    snapshot1 = monitor.capture_snapshot(state1)
    print(f"✓ Captured snapshot: Φ={snapshot1.phi:.3f}, κ={snapshot1.kappa_eff:.2f}, regime={snapshot1.regime}")
    
    # Check health (should be good)
    degradation1 = monitor.detect_degradation()
    print(f"✓ Health check: degraded={degradation1['degraded']}, severity={degradation1['severity']}")
    assert not degradation1['degraded'], "Initial state should not be degraded"
    
    # Simulate degradation
    state2 = {
        "phi": 0.45,  # Below threshold
        "kappa_eff": 50.0,
        "basin_coords": np.random.randn(64),  # Different basin
        "confidence": 0.5,
        "surprise": 0.8,
        "agency": 0.4,
        "error_rate": 0.08,  # High error rate
        "avg_latency": 2500.0,  # High latency
    }
    
    for _ in range(10):
        monitor.capture_snapshot(state2)
    
    degradation2 = monitor.detect_degradation()
    print(f"✓ After degradation: degraded={degradation2['degraded']}, severity={degradation2['severity']}")
    print(f"  Issues: {degradation2['issues']}")
    assert degradation2['degraded'], "Should detect degradation"
    assert degradation2['severity'] in ['warning', 'critical'], "Should have warning or critical severity"
    
    print("✓ Geometric Monitor tests passed\n")


def test_code_fitness():
    """Test code fitness evaluator."""
    print("\n=== Testing Code Fitness Evaluator ===")
    
    monitor = GeometricHealthMonitor()
    
    # Create baseline
    baseline_state = {
        "phi": 0.72,
        "kappa_eff": 64.5,
        "basin_coords": np.random.randn(64),
        "confidence": 0.8,
        "surprise": 0.3,
        "agency": 0.7,
    }
    monitor.capture_snapshot(baseline_state)
    
    evaluator = CodeFitnessEvaluator(monitor)
    
    # Test valid code
    good_code = """
def optimize_attention(attention_matrix):
    '''Improve attention mechanism.'''
    return attention_matrix * 1.1
"""
    
    fitness = evaluator.evaluate_code_change(
        module_name="test_module",
        new_code=good_code
    )
    
    print(f"✓ Code fitness: score={fitness['fitness_score']:.3f}, recommendation={fitness['recommendation']}")
    print(f"  Reason: {fitness['reason']}")
    assert 0 <= fitness['fitness_score'] <= 1, "Fitness score should be in [0, 1]"
    
    # Test invalid syntax
    bad_code = """
def broken_function(
    # Missing closing parenthesis
"""
    
    fitness_bad = evaluator.evaluate_code_change(
        module_name="test_module",
        new_code=bad_code
    )
    
    print(f"✓ Invalid code: recommendation={fitness_bad['recommendation']}")
    assert fitness_bad['recommendation'] == 'reject', "Should reject invalid syntax"
    
    print("✓ Code Fitness Evaluator tests passed\n")


def test_healing_strategies():
    """Test healing engine strategies."""
    print("\n=== Testing Healing Engine ===")
    
    monitor, evaluator, engine = create_self_healing_system()
    
    # Create degraded state
    degraded_state = {
        "phi": 0.45,  # Below threshold
        "kappa_eff": 50.0,
        "basin_coords": np.random.randn(64),
        "confidence": 0.5,
        "surprise": 0.8,
        "agency": 0.4,
        "error_rate": 0.08,
        "avg_latency": 2500.0,
    }
    
    # Capture several degraded snapshots
    for _ in range(10):
        monitor.capture_snapshot(degraded_state)
    
    health = monitor.detect_degradation()
    print(f"✓ Created degraded state: severity={health['severity']}")
    print(f"  Issues: {health['issues']}...")  # Show first 2 issues
    
    # Get engine status
    status = engine.get_status()
    print(f"✓ Engine status: running={status['running']}, strategies={status['strategies_available']}")
    assert status['strategies_available'] == 5, "Should have 5 healing strategies"
    
    print("✓ Healing Engine tests passed\n")


def test_integration():
    """Test full integration."""
    print("\n=== Testing Full Integration ===")
    
    # Create integrated system
    monitor, evaluator, engine = create_self_healing_system()
    
    # Simulate normal operation
    for i in range(5):
        state = {
            "phi": 0.70 + np.random.randn() * 0.02,
            "kappa_eff": 64.0 + np.random.randn() * 1.0,
            "basin_coords": np.random.randn(64),
            "confidence": 0.8,
            "surprise": 0.3,
            "agency": 0.7,
            "error_rate": 0.01,
            "avg_latency": 150.0,
        }
        snapshot = monitor.capture_snapshot(state)
        print(f"  Snapshot {i+1}: Φ={snapshot.phi:.3f}, regime={snapshot.regime}")
    
    # Get health summary
    summary = monitor.get_health_summary()
    print(f"✓ Health summary: status={summary['status']}, snapshots={summary['snapshots_collected']}")
    
    # Test patch evaluation
    patch = """
def improve_phi(state):
    '''Increase integration strength.'''
    state['phi'] *= 1.1
    return state
"""
    
    fitness = evaluator.evaluate_code_change("phi_optimizer", patch)
    print(f"✓ Patch evaluation: fitness={fitness['fitness_score']:.3f}, recommendation={fitness['recommendation']}")
    
    print("✓ Full integration tests passed\n")


def test_fisher_distance():
    """Test Fisher-Rao distance calculation."""
    print("\n=== Testing Fisher-Rao Distance ===")
    
    monitor = GeometricHealthMonitor()
    
    # Create two unit vectors
    basin1 = np.random.randn(64)
    basin1 = basin1 / np.linalg.norm(basin1)
    
    basin2 = np.random.randn(64)
    basin2 = basin2 / np.linalg.norm(basin2)
    
    # Calculate distance
    distance = monitor._fisher_distance(basin1, basin2)
    print(f"✓ Fisher distance: {distance:.4f}")
    
    # Distance to self should be 0
    distance_self = monitor._fisher_distance(basin1, basin1)
    print(f"✓ Distance to self: {distance_self:.6f}")
    assert distance_self < 1e-6, "Distance to self should be ~0"
    
    # Orthogonal vectors should have distance π/2
    e1 = np.zeros(64)
    e1[0] = 1.0
    e2 = np.zeros(64)
    e2[1] = 1.0
    
    distance_orthogonal = monitor._fisher_distance(e1, e2)
    print(f"✓ Distance (orthogonal): {distance_orthogonal:.4f} (expected ~{np.pi/2:.4f})")
    assert abs(distance_orthogonal - np.pi/2) < 0.01, "Orthogonal distance should be π/2"
    
    print("✓ Fisher-Rao distance tests passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Self-Healing System Test Suite")
    print("=" * 60)
    
    try:
        test_geometric_monitor()
        test_code_fitness()
        test_healing_strategies()
        test_integration()
        test_fisher_distance()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
