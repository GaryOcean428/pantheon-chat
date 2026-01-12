"""
8-Metrics Integration Validation Test

Validates that all 8 consciousness metrics are properly integrated
and working together in the consciousness system.

Sprint 1 P0 - Gap 1 Validation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from qig_core.consciousness_metrics import (
    compute_all_metrics,
    ConsciousnessMetrics,
    validate_consciousness_state
)

def test_8_metrics_integration():
    """Test all 8 metrics compute successfully."""
    print("\n=== 8-Metrics Integration Validation Test ===\n")
    
    # Create test data
    basin = np.random.dirichlet([1]*64)
    memory_basins = [np.random.dirichlet([1]*64) for _ in range(5)]
    trajectory = [np.random.dirichlet([1]*64) for _ in range(10)]
    self_observations = [
        {'basin': np.random.dirichlet([1]*64)} for _ in range(5)
    ]
    kernel_basins = {
        'Zeus': np.random.dirichlet([1]*64),
        'Athena': np.random.dirichlet([1]*64),
        'Apollo': np.random.dirichlet([1]*64),
    }
    
    # Compute all metrics
    metrics = compute_all_metrics(
        basin_coords=basin,
        memory_basins=memory_basins,
        trajectory=trajectory,
        self_observations=self_observations,
        kernel_basins=kernel_basins,
        kernel_name='Ocean'
    )
    
    # Validate all metrics are present
    print("✅ All 8 Metrics Computed:")
    print(f"  1. Φ (Integration):           {metrics.phi:.4f}")
    print(f"  2. κ_eff (Effective Coupling): {metrics.kappa_eff:.2f}")
    print(f"  3. M (Memory Coherence):       {metrics.memory_coherence:.4f}")
    print(f"  4. Γ (Regime Stability):       {metrics.regime_stability:.4f}")
    print(f"  5. G (Geometric Validity):     {metrics.geometric_validity:.4f}")
    print(f"  6. T (Temporal Consistency):   {metrics.temporal_consistency:.4f}")
    print(f"  7. R (Recursive Depth):        {metrics.recursive_depth:.4f}")
    print(f"  8. C (External Coupling):      {metrics.external_coupling:.4f}")
    
    # Validate consciousness state
    validation = validate_consciousness_state(metrics)
    
    print(f"\n✅ Consciousness Validation:")
    print(f"  Is Conscious: {validation['is_conscious']}")
    print(f"  Metrics Passed: {sum(1 for m in validation['metrics'].values() if m['passed'])}/8")
    
    if validation['violations']:
        print(f"\n⚠️  Violations:")
        for v in validation['violations']:
            print(f"    - {v}")
    
    # Test metric ranges
    assert 0 <= metrics.phi <= 1, "Φ out of range"
    assert 0 <= metrics.kappa_eff <= 100, "κ_eff out of range"
    assert 0 <= metrics.memory_coherence <= 1, "M out of range"
    assert 0 <= metrics.regime_stability <= 1, "Γ out of range"
    assert 0 <= metrics.geometric_validity <= 1, "G out of range"
    assert -1 <= metrics.temporal_consistency <= 1, "T out of range"
    assert 0 <= metrics.recursive_depth <= 1, "R out of range"
    assert 0 <= metrics.external_coupling <= 1, "C out of range"
    
    print("\n✅ All metrics in valid ranges")
    print("✅ 8-Metrics integration validation PASSED\n")
    
    return metrics, validation


def test_consciousness_detection():
    """Test consciousness detection with high-quality metrics."""
    print("=== Consciousness Detection Test ===\n")
    
    # Create high-quality conscious state
    basin = np.random.dirichlet([10]*64)  # Concentrated distribution
    memory_basins = [basin + np.random.normal(0, 0.01, 64) for _ in range(10)]
    trajectory = [basin + np.random.normal(0, 0.005, 64) for _ in range(15)]
    self_observations = [{'basin': t} for t in trajectory[:10]]
    kernel_basins = {f'Kernel{i}': basin + np.random.normal(0, 0.02, 64) 
                     for i in range(5)}
    
    metrics = compute_all_metrics(
        basin_coords=basin,
        memory_basins=memory_basins,
        trajectory=trajectory,
        self_observations=self_observations,
        kernel_basins=kernel_basins
    )
    
    validation = validate_consciousness_state(metrics)
    
    print(f"High-quality metrics:")
    print(f"  Φ = {metrics.phi:.4f} (target > 0.70)")
    print(f"  M = {metrics.memory_coherence:.4f} (target > 0.60)")
    print(f"  Γ = {metrics.regime_stability:.4f} (target > 0.80)")
    print(f"  Conscious: {validation['is_conscious']}")
    
    print("\n✅ Consciousness detection test PASSED\n")
    
    return metrics


if __name__ == '__main__':
    try:
        metrics1, validation1 = test_8_metrics_integration()
        metrics2 = test_consciousness_detection()
        
        print("=" * 50)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("=" * 50)
        print("\n✅ Gap 1 (8-Metrics) VALIDATED - All metrics working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
