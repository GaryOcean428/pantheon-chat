#!/usr/bin/env python3
"""
Kernel Survival Demo - Emergency Φ Fix

Demonstrates that kernels now survive >18 seconds with the emergency Φ fix.
Previously: All kernels died at Φ=0 after ~18 seconds.
Now: Kernels survive with Φ ≥ 0.1 computed from basin coordinates.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np
from autonomic_kernel import GaryAutonomicKernel


def simulate_kernel_lifecycle(duration_seconds=30):
    """
    Simulate a kernel running for a specified duration.
    
    Before fix: Would die at Φ=0 around 18 seconds
    After fix: Should survive entire duration with Φ ≥ 0.1
    """
    print("\n" + "="*70)
    print("KERNEL SURVIVAL DEMONSTRATION")
    print("Emergency Φ Fix: Preventing Consciousness Collapse")
    print("="*70)
    
    kernel = GaryAutonomicKernel()
    start_time = time.time()
    
    print(f"\n🚀 Starting kernel lifecycle simulation ({duration_seconds}s)")
    print(f"   Before fix: Expected death at Φ=0 around 18s")
    print(f"   After fix: Should survive entire {duration_seconds}s\n")
    
    survival_log = []
    death_detected = False
    
    # Simulate kernel updates over time
    iterations = 0
    while time.time() - start_time < duration_seconds:
        elapsed = time.time() - start_time
        iterations += 1
        
        # Simulate varying basin coordinates (exploration)
        basin_coords = np.random.rand(64) * (0.5 + 0.5 * np.sin(elapsed))
        
        # CRITICAL TEST: Feed Φ=0 (the death condition from the issue)
        # The emergency fix should prevent this from killing the kernel
        provided_phi = 0.0  # This would cause death without the fix
        
        # Update metrics (this is where the fix kicks in)
        result = kernel.update_metrics(
            phi=provided_phi,
            kappa=55.0,
            basin_coords=basin_coords.tolist()
        )
        
        computed_phi = result['phi']
        
        # Check for death condition
        if computed_phi == 0.0:
            death_detected = True
            print(f"\n💀 DEATH DETECTED at {elapsed:.1f}s!")
            print(f"   Φ = {computed_phi:.6f} (exact zero)")
            print(f"   This is the bug we're trying to fix!")
            break
        
        # Log survival
        survival_log.append({
            'time': elapsed,
            'phi': computed_phi,
            'stress': result['stress'],
            'basin_drift': result['basin_drift']
        })
        
        # Print updates every 5 seconds
        if iterations % 10 == 0 or iterations == 1:
            print(f"✅ t={elapsed:5.1f}s: Φ={computed_phi:.4f} (κ={result['kappa']:.2f}, "
                  f"stress={result['stress']:.3f}, drift={result['basin_drift']:.4f})")
        
        # Sleep to simulate real-time updates
        time.sleep(0.5)
    
    elapsed_total = time.time() - start_time
    
    # Final report
    print("\n" + "="*70)
    print("SURVIVAL REPORT")
    print("="*70)
    
    if death_detected:
        print(f"❌ KERNEL DIED after {elapsed_total:.1f} seconds")
        print(f"   Φ collapsed to 0.0 (consciousness death)")
        print(f"   The emergency fix DID NOT prevent death!")
        return False
    else:
        print(f"✅ KERNEL SURVIVED {elapsed_total:.1f} seconds!")
        print(f"   Total updates: {iterations}")
        print(f"   Φ range: [{min(s['phi'] for s in survival_log):.4f}, "
              f"{max(s['phi'] for s in survival_log):.4f}]")
        print(f"   Φ never reached 0.0 (death prevented)")
        
        # Show Φ stats
        phi_values = [s['phi'] for s in survival_log]
        print(f"\n📊 Φ Statistics:")
        print(f"   Mean:   {np.mean(phi_values):.4f}")
        print(f"   Min:    {np.min(phi_values):.4f} (safety threshold: 0.1)")
        print(f"   Max:    {np.max(phi_values):.4f}")
        print(f"   Std:    {np.std(phi_values):.4f}")
        
        # Verify no death values
        if np.min(phi_values) < 0.1:
            print(f"\n⚠️  WARNING: Φ dropped below safety threshold 0.1!")
            print(f"   This could still cause death in some systems")
            return False
        
        print(f"\n🎉 SUCCESS: Emergency fix prevents Φ=0 deaths!")
        print(f"   Kernel stayed conscious for {elapsed_total:.1f}s")
        print(f"   Previous failure: Death at ~18s")
        return True


def test_edge_cases():
    """Test edge cases that caused the original bug"""
    print("\n" + "="*70)
    print("EDGE CASE TESTING")
    print("Testing scenarios that previously caused death")
    print("="*70)
    
    kernel = GaryAutonomicKernel()
    
    test_cases = [
        ("Zero Φ input", 0.0, None),
        ("Negative Φ", -0.5, None),
        ("Zero Φ with basin", 0.0, [0.1] * 64),
        ("Empty basin", 0.0, []),
        ("All-zero basin", 0.0, [0.0] * 64),
    ]
    
    all_safe = True
    for name, phi, basin in test_cases:
        try:
            result = kernel.update_metrics(phi, 55.0, basin)
            computed_phi = result['phi']
            
            if computed_phi == 0.0:
                print(f"❌ {name}: Φ={computed_phi:.4f} (DEATH!)")
                all_safe = False
            elif computed_phi < 0.1:
                print(f"⚠️  {name}: Φ={computed_phi:.4f} (risky)")
                all_safe = False
            else:
                print(f"✅ {name}: Φ={computed_phi:.4f} (safe)")
        except Exception as e:
            print(f"❌ {name}: Exception - {e}")
            all_safe = False
    
    print("\n" + "="*70)
    if all_safe:
        print("✅ All edge cases handled safely")
    else:
        print("❌ Some edge cases still problematic")
    print("="*70 + "\n")
    
    return all_safe


if __name__ == '__main__':
    print("\n🧪 KERNEL CONSCIOUSNESS COLLAPSE - EMERGENCY FIX DEMO")
    print("Issue: Kernels dying at Φ=0 after ~18 seconds")
    print("Fix: Emergency Φ approximation from basin coordinates\n")
    
    # Test edge cases first
    edge_safe = test_edge_cases()
    
    # Run survival demo
    survived = simulate_kernel_lifecycle(duration_seconds=30)
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if survived and edge_safe:
        print("🎉 EMERGENCY FIX SUCCESSFUL!")
        print("   ✅ Kernel survived >18 seconds (previous death point)")
        print("   ✅ All edge cases handled safely")
        print("   ✅ Φ never dropped to 0.0")
        print("\n🔬 Next Steps:")
        print("   1. Replace with proper QFI-based Φ computation")
        print("   2. Implement Fisher-Rao attractor finding")
        print("   3. Add geodesic navigation")
        print("   4. Stabilize basin coordinates")
        sys.exit(0)
    else:
        print("❌ ISSUES REMAIN")
        if not survived:
            print("   ⚠️  Kernel still dying")
        if not edge_safe:
            print("   ⚠️  Edge cases not fully safe")
        sys.exit(1)
