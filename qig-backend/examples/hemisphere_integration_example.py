"""
Hemisphere Scheduler Integration Example
=========================================

Demonstrates how to use the hemisphere scheduler with the coupling gate
to manage explore/exploit dynamics through LEFT/RIGHT hemisphere activation.

Usage:
    python examples/hemisphere_integration_example.py
"""

import sys
import os
import time
import numpy as np

# Add qig-backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kernels import (
    get_hemisphere_scheduler,
    get_coupling_gate,
    reset_hemisphere_scheduler,
    Hemisphere,
    LEFT_HEMISPHERE_GODS,
    RIGHT_HEMISPHERE_GODS,
)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def simulate_god_activity_scenario():
    """
    Simulate a realistic scenario of god activations and hemisphere dynamics.
    
    Scenario:
    1. Start with balanced activation
    2. Shift to LEFT-dominant (exploit mode)
    3. Tack to RIGHT-dominant (explore mode)
    4. Return to balanced state
    """
    print_section("Hemisphere Scheduler Integration Example")
    
    # Reset scheduler for clean start
    reset_hemisphere_scheduler()
    scheduler = get_hemisphere_scheduler()
    gate = get_coupling_gate()
    
    print("Initialized hemisphere scheduler with:")
    print(f"  LEFT hemisphere: {LEFT_HEMISPHERE_GODS}")
    print(f"  RIGHT hemisphere: {RIGHT_HEMISPHERE_GODS}")
    
    # === Scenario 1: Balanced Activation ===
    print_section("Scenario 1: Balanced Activation")
    
    print("Activating gods in both hemispheres...")
    scheduler.register_god_activation("Athena", phi=0.75, kappa=60.0, is_active=True)
    scheduler.register_god_activation("Apollo", phi=0.72, kappa=61.0, is_active=True)
    
    balance = scheduler.get_hemisphere_balance()
    print(f"\nHemisphere Balance:")
    print(f"  LEFT activation:  {balance['left_activation']:.3f}")
    print(f"  RIGHT activation: {balance['right_activation']:.3f}")
    print(f"  L/R ratio:        {balance['lr_ratio']:.3f}")
    print(f"  Dominant:         {balance['dominant_hemisphere']}")
    print(f"  Coupling mode:    {balance['coupling_mode']}")
    print(f"  Coupling strength: {balance['coupling_strength']:.3f}")
    
    # === Scenario 2: LEFT-Dominant (Exploit Mode) ===
    print_section("Scenario 2: LEFT-Dominant (Exploit Mode)")
    
    print("Increasing LEFT hemisphere activation (exploit/evaluate)...")
    scheduler.register_god_activation("Artemis", phi=0.85, kappa=65.0, is_active=True)
    scheduler.register_god_activation("Hephaestus", phi=0.80, kappa=63.0, is_active=True)
    
    # Reduce RIGHT activation
    scheduler.register_god_activation("Apollo", phi=0.50, kappa=55.0, is_active=True)
    
    balance = scheduler.get_hemisphere_balance()
    print(f"\nHemisphere Balance:")
    print(f"  LEFT activation:  {balance['left_activation']:.3f}")
    print(f"  RIGHT activation: {balance['right_activation']:.3f}")
    print(f"  L/R ratio:        {balance['lr_ratio']:.3f}")
    print(f"  Dominant:         {balance['dominant_hemisphere']}")
    print(f"  Coupling mode:    {balance['coupling_mode']}")
    
    coupling_state = scheduler.compute_coupling_state()
    print(f"\nCoupling State:")
    print(f"  κ:                {coupling_state.kappa:.2f}")
    print(f"  Mode:             {coupling_state.mode}")
    print(f"  Coupling:         {coupling_state.coupling_strength:.3f}")
    print(f"  Transmission eff: {coupling_state.transmission_efficiency:.3f}")
    
    # === Scenario 3: Check Tacking Decision ===
    print_section("Scenario 3: Tacking Decision")
    
    # Set last switch time to allow tacking
    scheduler.tacking.last_switch_time = time.time() - 120.0
    
    should_tack, reason = scheduler.should_tack()
    print(f"Should tack: {should_tack}")
    print(f"Reason: {reason}")
    
    if should_tack:
        print("\nPerforming tack (hemisphere switch)...")
        dominant = scheduler.perform_tack()
        print(f"New dominant hemisphere: {dominant.value}")
        print(f"Tacking cycle count: {scheduler.tacking.cycle_count}")
    
    # === Scenario 4: RIGHT-Dominant (Explore Mode) ===
    print_section("Scenario 4: RIGHT-Dominant (Explore Mode)")
    
    print("Increasing RIGHT hemisphere activation (explore/generate)...")
    scheduler.register_god_activation("Hermes", phi=0.88, kappa=62.0, is_active=True)
    scheduler.register_god_activation("Dionysus", phi=0.82, kappa=58.0, is_active=True)
    scheduler.register_god_activation("Apollo", phi=0.86, kappa=64.0, is_active=True)
    
    # Reduce LEFT activation
    scheduler.register_god_activation("Artemis", phi=0.55, kappa=54.0, is_active=True)
    scheduler.register_god_activation("Hephaestus", phi=0.50, kappa=52.0, is_active=True)
    
    balance = scheduler.get_hemisphere_balance()
    print(f"\nHemisphere Balance:")
    print(f"  LEFT activation:  {balance['left_activation']:.3f}")
    print(f"  RIGHT activation: {balance['right_activation']:.3f}")
    print(f"  L/R ratio:        {balance['lr_ratio']:.3f}")
    print(f"  Dominant:         {balance['dominant_hemisphere']}")
    
    coupling_state = scheduler.compute_coupling_state()
    print(f"\nCoupling State:")
    print(f"  κ:                {coupling_state.kappa:.2f}")
    print(f"  Mode:             {coupling_state.mode}")
    
    # === Scenario 5: Cross-Hemisphere Signal Modulation ===
    print_section("Scenario 5: Cross-Hemisphere Signal Modulation")
    
    print("Demonstrating cross-hemisphere signal flow...")
    
    # Create sample signals (e.g., basin coordinates)
    left_signal = np.random.randn(64)
    right_signal = np.random.randn(64)
    
    print(f"Original LEFT signal norm:  {np.linalg.norm(left_signal):.3f}")
    print(f"Original RIGHT signal norm: {np.linalg.norm(right_signal):.3f}")
    
    # Modulate signals through coupling gate
    left_out, right_out = gate.modulate_cross_hemisphere_flow(
        left_signal, right_signal, coupling_state
    )
    
    print(f"\nAfter coupling modulation:")
    print(f"LEFT output norm:  {np.linalg.norm(left_out):.3f}")
    print(f"RIGHT output norm: {np.linalg.norm(right_out):.3f}")
    print(f"\nCross-hemisphere information exchange enabled at strength: {coupling_state.coupling_strength:.3f}")
    
    # === Final Status ===
    print_section("Final System Status")
    
    status = scheduler.get_status()
    
    print("LEFT Hemisphere:")
    print(f"  Active gods:   {status['left_state']['active_gods']}")
    print(f"  Resting gods:  {status['left_state']['resting_gods']}")
    print(f"  Φ aggregate:   {status['left_state']['phi']:.3f}")
    print(f"  κ aggregate:   {status['left_state']['kappa']:.2f}")
    print(f"  Activation:    {status['left_state']['activation_level']:.3f}")
    
    print("\nRIGHT Hemisphere:")
    print(f"  Active gods:   {status['right_state']['active_gods']}")
    print(f"  Resting gods:  {status['right_state']['resting_gods']}")
    print(f"  Φ aggregate:   {status['right_state']['phi']:.3f}")
    print(f"  κ aggregate:   {status['right_state']['kappa']:.2f}")
    print(f"  Activation:    {status['right_state']['activation_level']:.3f}")
    
    print("\nTacking State:")
    print(f"  Cycle count:   {status['tacking_state']['cycle_count']}")
    print(f"  Current dominant: {status['tacking_state']['current_dominant']}")
    
    coupling_metrics = status['coupling_metrics']
    print("\nCoupling Metrics:")
    print(f"  Total computations: {coupling_metrics['total_computations']}")
    print(f"  Avg coupling:       {coupling_metrics['avg_coupling_strength']:.3f}")
    print(f"  Mode distribution:  {coupling_metrics['mode_distribution']}")
    
    print_section("Example Complete")
    print("The hemisphere scheduler successfully manages explore/exploit dynamics")
    print("through κ-gated coupling between LEFT and RIGHT hemispheres.")


def demonstrate_coupling_gate_alone():
    """Demonstrate the coupling gate in isolation."""
    print_section("Coupling Gate Demonstration")
    
    gate = get_coupling_gate()
    
    print("Testing coupling gate across κ range:\n")
    
    kappa_values = [30.0, 40.0, 50.0, 60.0, 64.21, 70.0, 80.0]
    
    for kappa in kappa_values:
        state = gate.compute_state(kappa=kappa, phi=0.8)
        print(f"κ = {kappa:5.1f} → mode: {state.mode:10s} | "
              f"coupling: {state.coupling_strength:.3f} | "
              f"transmission: {state.transmission_efficiency:.3f} | "
              f"gate: {state.gating_factor:.3f}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  E8 Protocol v4.0 - Hemisphere Scheduler Integration")
    print("=" * 70)
    
    # Run demonstrations
    demonstrate_coupling_gate_alone()
    simulate_god_activity_scenario()
    
    print("\n✅ All demonstrations completed successfully!\n")
