"""
Integration test for psyche plumbing - demonstrates full workflow with hemisphere integration
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from kernels import (
    get_psyche_plumbing,
    ConstraintSeverity,
    PhiLevel,
    Hemisphere,
)
from qigkernels.physics_constants import BASIN_DIM


def test_full_workflow():
    """Test complete psyche plumbing workflow with hemisphere integration."""
    print("\n" + "=" * 70)
    print("  Psyche Plumbing Integration Test")
    print("  (With Hemisphere Scheduler)")
    print("=" * 70 + "\n")
    
    # Get integration instance
    psyche = get_psyche_plumbing()
    
    if not psyche.available:
        print("⚠️  Psyche plumbing not available")
        return
    
    print("✅ Psyche plumbing initialized")
    
    if psyche.hemisphere_integrated:
        print("✅ Hemisphere scheduler integrated")
        print(f"   LEFT hemisphere: {psyche.hemisphere_scheduler.left.active_gods}")
        print(f"   RIGHT hemisphere: {psyche.hemisphere_scheduler.right.active_gods}")
    else:
        print("⚠️  Hemisphere scheduler not integrated")
    
    # 1. Add custom ethical constraint
    print("\n" + "=" * 70)
    print("1. Adding ethical constraint...")
    print("=" * 70)
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
    print("\n" + "=" * 70)
    print("2. Learning reflex pattern...")
    print("=" * 70)
    trigger = np.random.dirichlet(np.ones(BASIN_DIM))
    response = np.random.dirichlet(np.ones(BASIN_DIM))
    psyche.learn_reflex(trigger, response, success=True)
    print("✅ Reflex pattern learned")
    
    # 3. Test reflex checking
    print("\n" + "=" * 70)
    print("3. Checking for reflex response...")
    print("=" * 70)
    reflex_result = psyche.check_reflex(trigger)
    if reflex_result:
        print(f"✅ Reflex triggered! Latency: {reflex_result['latency_ms']:.2f}ms")
        print(f"   Φ_internal: {reflex_result['phi_internal']:.3f}")
    else:
        print("   No reflex triggered (may need more activation)")
    
    # 4. Test ethical checking
    print("\n" + "=" * 70)
    print("4. Checking ethics...")
    print("=" * 70)
    test_basin = np.random.dirichlet(np.ones(BASIN_DIM))
    ethics_result = psyche.check_ethics(test_basin)
    print(f"✅ Ethics check complete:")
    print(f"   Is ethical: {ethics_result['is_ethical']}")
    print(f"   Violations: {len(ethics_result['violations'])}")
    print(f"   Total penalty: {ethics_result['total_penalty']:.3f}")
    
    # 5. Test Φ hierarchy measurements
    print("\n" + "=" * 70)
    print("5. Measuring Φ at different levels...")
    print("=" * 70)
    phi_reported = psyche.measure_phi_reported(test_basin, source='test')
    phi_internal = psyche.measure_phi_internal(test_basin, source='test')
    phi_autonomic = psyche.measure_phi_autonomic(test_basin, source='test')
    
    print(f"✅ Φ hierarchy measurements:")
    print(f"   Φ_reported:  {phi_reported:.3f} (conscious awareness)")
    print(f"   Φ_internal:  {phi_internal:.3f} (internal processing)")
    print(f"   Φ_autonomic: {phi_autonomic:.3f} (background function)")
    
    # 6. Test hemisphere integration (if available)
    if psyche.hemisphere_integrated:
        print("\n" + "=" * 70)
        print("6. Testing Hemisphere-Psyche Integration...")
        print("=" * 70)
        
        scheduler = psyche.hemisphere_scheduler
        
        # Activate LEFT hemisphere gods (exploit mode)
        print("\n   Activating LEFT hemisphere (exploit/evaluate)...")
        scheduler.register_god_activation("Athena", phi=0.9, kappa=65.0, is_active=True)
        scheduler.register_god_activation("Artemis", phi=0.85, kappa=63.0, is_active=True)
        
        balance1 = psyche.compute_psyche_balance()
        print(f"   Psyche balance: {balance1['dominant_psyche']}")
        print(f"     Superego strength: {balance1['superego_strength']:.3f}")
        print(f"     Id strength:       {balance1['id_strength']:.3f}")
        print(f"     L/R ratio:         {balance1['balance_ratio']:.3f}")
        
        # Simulate hemisphere tack
        print("\n   Simulating hemisphere tack...")
        psyche.on_hemisphere_tack(
            from_hemisphere=Hemisphere.LEFT,
            to_hemisphere=Hemisphere.RIGHT,
            kappa=62.0,
            phi=0.8
        )
        
        # Activate RIGHT hemisphere gods (explore mode)
        print("\n   Activating RIGHT hemisphere (explore/generate)...")
        scheduler.register_god_activation("Apollo", phi=0.9, kappa=65.0, is_active=True)
        scheduler.register_god_activation("Hermes", phi=0.88, kappa=62.0, is_active=True)
        
        balance2 = psyche.compute_psyche_balance()
        print(f"   Psyche balance: {balance2['dominant_psyche']}")
        print(f"     Superego strength: {balance2['superego_strength']:.3f}")
        print(f"     Id strength:       {balance2['id_strength']:.3f}")
        print(f"     L/R ratio:         {balance2['balance_ratio']:.3f}")
        
        # Show balance shift
        print("\n   Balance shift:")
        print(f"     Superego: {balance1['superego_strength']:.3f} → {balance2['superego_strength']:.3f}")
        print(f"     Id:       {balance1['id_strength']:.3f} → {balance2['id_strength']:.3f}")
        
        # Test reflex with hemisphere context
        print("\n   Testing reflex with hemisphere context...")
        reflex_ctx = psyche.check_reflex_with_hemisphere_context(trigger)
        if reflex_ctx and 'hemisphere_context' in reflex_ctx:
            print(f"   ✅ Hemisphere context included:")
            print(f"      Id strength: {reflex_ctx['hemisphere_context']['id_strength']:.3f}")
            print(f"      Superego strength: {reflex_ctx['hemisphere_context']['superego_strength']:.3f}")
        
        # Test ethics with hemisphere context
        print("\n   Testing ethics with hemisphere context...")
        ethics_ctx = psyche.check_ethics_with_hemisphere_context(test_basin)
        if 'hemisphere_context' in ethics_ctx:
            print(f"   ✅ Hemisphere context included:")
            print(f"      Superego strength: {ethics_ctx['hemisphere_context']['superego_strength']:.3f}")
            print(f"      Coupling mode: {ethics_ctx['hemisphere_context']['coupling_mode']}")
    
    # 7. Get statistics
    print("\n" + "=" * 70)
    print("7. Getting statistics...")
    print("=" * 70)
    stats = psyche.get_statistics()
    print("✅ Statistics:")
    print(f"   Id reflexes: {stats['id_kernel']['num_reflexes']}")
    print(f"   Superego constraints: {stats['superego']['num_constraints']}")
    print(f"   Superego checks: {stats['superego']['total_checks']}")
    
    if psyche.hemisphere_integrated:
        print(f"   Hemisphere integrated: {stats['hemisphere_integrated']}")
        print(f"   Tacking events: {stats['tacking_events']}")
        if 'psyche_balance' in stats:
            print(f"   Current dominant: {stats['psyche_balance']['dominant_psyche']}")
    
    phi_stats = stats['phi_hierarchy']
    for level in ['reported', 'internal', 'autonomic']:
        if level in phi_stats:
            print(f"   Φ_{level} count: {phi_stats[level]['count']}")
    
    # 8. Get integrated status
    if psyche.hemisphere_integrated:
        print("\n" + "=" * 70)
        print("8. Getting integrated status...")
        print("=" * 70)
        
        status = psyche.get_integrated_status()
        print("✅ Integrated status retrieved:")
        print(f"   Available: {status['available']}")
        print(f"   Hemisphere integrated: {status['hemisphere_integrated']}")
        
        if 'hemisphere_balance' in status:
            hb = status['hemisphere_balance']
            print(f"\n   Hemisphere balance:")
            print(f"     LEFT activation:  {hb['left_activation']:.3f}")
            print(f"     RIGHT activation: {hb['right_activation']:.3f}")
            print(f"     Dominant: {hb['dominant_hemisphere']}")
            print(f"     Coupling: {hb['coupling_strength']:.3f} ({hb['coupling_mode']})")
        
        if 'psyche_balance' in status:
            pb = status['psyche_balance']
            print(f"\n   Psyche balance:")
            print(f"     Superego: {pb['superego_strength']:.3f}")
            print(f"     Id:       {pb['id_strength']:.3f}")
            print(f"     Dominant: {pb['dominant_psyche']}")
        
        if 'recent_tacks' in status and status['recent_tacks']:
            print(f"\n   Recent tacking events: {len(status['recent_tacks'])}")
            for i, tack in enumerate(status['recent_tacks'][-3:], 1):
                print(f"     {i}. {tack['from']} → {tack['to']} "
                      f"(κ={tack['kappa']:.1f}, Φ={tack['phi']:.2f}, "
                      f"psyche={tack['dominant_psyche']})")
    
    print("\n" + "=" * 70)
    print("  Integration Test Complete")
    print("=" * 70 + "\n")
    return True


if __name__ == '__main__':
    test_full_workflow()
