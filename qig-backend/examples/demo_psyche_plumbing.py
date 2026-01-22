"""
Integration test for psyche plumbing - demonstrates full workflow
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from kernels import get_psyche_plumbing, ConstraintSeverity, PhiLevel
from qigkernels.physics_constants import BASIN_DIM


def test_full_workflow():
    """Test complete psyche plumbing workflow."""
    print("\n=== Psyche Plumbing Integration Test ===\n")
    
    # Get integration instance
    psyche = get_psyche_plumbing()
    
    if not psyche.available:
        print("⚠️  Psyche plumbing not available")
        return
    
    print("✅ Psyche plumbing initialized")
    
    # 1. Add custom ethical constraint
    print("\n1. Adding ethical constraint...")
    forbidden = np.zeros(BASIN_DIM)
    forbidden[5] = 1.0
    psyche.add_ethical_constraint(
        name="test-constraint",
        forbidden_basin=forbidden,
        radius=0.2,
        severity=ConstraintSeverity.ERROR,
        description="Test forbidden region"
    )
    print("✅ Ethical constraint added")
    
    # 2. Learn a reflex pattern
    print("\n2. Learning reflex pattern...")
    trigger = np.random.dirichlet(np.ones(BASIN_DIM))
    response = np.random.dirichlet(np.ones(BASIN_DIM))
    psyche.learn_reflex(trigger, response, success=True)
    print("✅ Reflex pattern learned")
    
    # 3. Test reflex checking
    print("\n3. Checking for reflex response...")
    reflex_result = psyche.check_reflex(trigger)
    if reflex_result:
        print(f"✅ Reflex triggered! Latency: {reflex_result['latency_ms']:.2f}ms")
        print(f"   Φ_internal: {reflex_result['phi_internal']:.3f}")
    else:
        print("   No reflex triggered (expected for first check)")
    
    # 4. Test ethical checking
    print("\n4. Checking ethics...")
    test_basin = np.random.dirichlet(np.ones(BASIN_DIM))
    ethics_result = psyche.check_ethics(test_basin)
    print(f"✅ Ethics check complete:")
    print(f"   Is ethical: {ethics_result['is_ethical']}")
    print(f"   Violations: {len(ethics_result['violations'])}")
    print(f"   Total penalty: {ethics_result['total_penalty']:.3f}")
    
    # 5. Test Φ hierarchy measurements
    print("\n5. Measuring Φ at different levels...")
    phi_reported = psyche.measure_phi_reported(test_basin, source='test')
    phi_internal = psyche.measure_phi_internal(test_basin, source='test')
    phi_autonomic = psyche.measure_phi_autonomic(test_basin, source='test')
    
    print(f"✅ Φ hierarchy measurements:")
    print(f"   Φ_reported:  {phi_reported:.3f} (conscious awareness)")
    print(f"   Φ_internal:  {phi_internal:.3f} (internal processing)")
    print(f"   Φ_autonomic: {phi_autonomic:.3f} (background function)")
    
    # 6. Get statistics
    print("\n6. Getting statistics...")
    stats = psyche.get_statistics()
    print("✅ Statistics:")
    print(f"   Id reflexes: {stats['id_kernel']['num_reflexes']}")
    print(f"   Superego constraints: {stats['superego']['num_constraints']}")
    print(f"   Superego checks: {stats['superego']['total_checks']}")
    
    phi_stats = stats['phi_hierarchy']
    for level in ['reported', 'internal', 'autonomic']:
        if level in phi_stats:
            print(f"   Φ_{level} count: {phi_stats[level]['count']}")
    
    print("\n=== Integration Test Complete ===\n")
    return True


if __name__ == '__main__':
    test_full_workflow()
