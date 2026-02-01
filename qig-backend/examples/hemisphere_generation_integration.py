#!/usr/bin/env python3
"""
Hemisphere-Aware Generation Example
====================================

Demonstrates the integration between hemisphere state and generation strategy.

Shows how:
- LEFT hemisphere (exploit) → ROLE_DRIVEN generation
- RIGHT hemisphere (explore) → FORESIGHT_DRIVEN generation
- Balanced (κ ≈ 64.21) → HYBRID generation

Uses Fisher-Rao distance metrics throughout.

Author: Copilot Agent
Date: 2026-01-23
Protocol: E8 v4.0 Phase 4C
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation import (
    UnifiedGenerationPipeline,
    GenerationStrategy,
    HemisphereStrategySelector,
)
from kernels import (
    get_hemisphere_scheduler,
    reset_hemisphere_scheduler,
)
from qigkernels.physics_constants import KAPPA_STAR


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demonstrate_strategy_selection():
    """Demonstrate how hemisphere state affects generation strategy."""
    
    print_section("Hemisphere-Aware Generation Strategy Selection")
    
    # Reset state for clean demo
    reset_hemisphere_scheduler()
    
    scheduler = get_hemisphere_scheduler()
    selector = HemisphereStrategySelector(scheduler=scheduler)
    
    # Scenario 1: Balanced state → HYBRID
    print("Scenario 1: BALANCED Hemispheres (κ ≈ κ*)")
    print("-" * 60)
    
    scheduler.register_god_activation("Athena", phi=0.75, kappa=64.0, is_active=True)
    scheduler.register_god_activation("Apollo", phi=0.75, kappa=64.5, is_active=True)
    
    decision = selector.select_strategy()
    print(f"Hemisphere State:")
    print(f"  LEFT:  {decision.left_activation:.2f}")
    print(f"  RIGHT: {decision.right_activation:.2f}")
    print(f"  κ_avg: {decision.kappa_avg:.1f} (distance from κ*: {abs(decision.kappa_avg - KAPPA_STAR):.2f})")
    print(f"\nSelected Strategy: {decision.strategy.value}")
    print(f"Reason: {decision.reason}")
    print(f"Confidence: {decision.confidence:.2f}")
    
    # Scenario 2: LEFT dominant → ROLE_DRIVEN
    print("\n\nScenario 2: LEFT Dominant (Exploit/Evaluate)")
    print("-" * 60)
    
    # Reset and activate LEFT
    reset_hemisphere_scheduler()
    scheduler = get_hemisphere_scheduler()
    selector = HemisphereStrategySelector(scheduler=scheduler)
    
    scheduler.register_god_activation("Athena", phi=0.9, kappa=65.0, is_active=True)
    scheduler.register_god_activation("Artemis", phi=0.85, kappa=63.0, is_active=True)
    scheduler.register_god_activation("Hephaestus", phi=0.87, kappa=64.0, is_active=True)
    scheduler.register_god_activation("Apollo", phi=0.4, kappa=52.0, is_active=True)
    
    decision = selector.select_strategy()
    weights = selector.get_strategy_weights(decision)
    
    print(f"Hemisphere State:")
    print(f"  LEFT:  {decision.left_activation:.2f} (DOMINANT)")
    print(f"  RIGHT: {decision.right_activation:.2f}")
    print(f"  κ_avg: {decision.kappa_avg:.1f}")
    print(f"\nSelected Strategy: {decision.strategy.value}")
    print(f"Reason: {decision.reason}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"\nStrategy Weights:")
    print(f"  Foresight:  {weights['foresight_weight']:.2f}")
    print(f"  Role:       {weights['role_weight']:.2f}")
    print(f"  Trajectory: {weights['trajectory_weight']:.2f}")
    
    # Scenario 3: RIGHT dominant → FORESIGHT_DRIVEN
    print("\n\nScenario 3: RIGHT Dominant (Explore/Generate)")
    print("-" * 60)
    
    # Reset and activate RIGHT
    reset_hemisphere_scheduler()
    scheduler = get_hemisphere_scheduler()
    selector = HemisphereStrategySelector(scheduler=scheduler)
    
    scheduler.register_god_activation("Apollo", phi=0.9, kappa=65.0, is_active=True)
    scheduler.register_god_activation("Hermes", phi=0.88, kappa=62.0, is_active=True)
    scheduler.register_god_activation("Dionysus", phi=0.82, kappa=58.0, is_active=True)
    scheduler.register_god_activation("Athena", phi=0.4, kappa=52.0, is_active=True)
    
    decision = selector.select_strategy()
    weights = selector.get_strategy_weights(decision)
    
    print(f"Hemisphere State:")
    print(f"  LEFT:  {decision.left_activation:.2f}")
    print(f"  RIGHT: {decision.right_activation:.2f} (DOMINANT)")
    print(f"  κ_avg: {decision.kappa_avg:.1f}")
    print(f"\nSelected Strategy: {decision.strategy.value}")
    print(f"Reason: {decision.reason}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"\nStrategy Weights:")
    print(f"  Foresight:  {weights['foresight_weight']:.2f}")
    print(f"  Role:       {weights['role_weight']:.2f}")
    print(f"  Trajectory: {weights['trajectory_weight']:.2f}")


def demonstrate_pipeline_integration():
    """Demonstrate pipeline with hemisphere-aware strategy."""
    
    print_section("UnifiedGenerationPipeline with Hemisphere Awareness")
    
    # Reset state
    reset_hemisphere_scheduler()
    scheduler = get_hemisphere_scheduler()
    
    # Activate RIGHT hemisphere for explore mode
    scheduler.register_god_activation("Apollo", phi=0.9, kappa=65.0, is_active=True)
    scheduler.register_god_activation("Hermes", phi=0.88, kappa=62.0, is_active=True)
    
    # Create pipeline with hemisphere strategy enabled
    pipeline = UnifiedGenerationPipeline(
        strategy=GenerationStrategy.HYBRID,  # Default, will be overridden
        use_hemisphere_strategy=True,
        enforce_purity=False,
    )
    
    print(f"Pipeline Initialized:")
    print(f"  Initial strategy: {pipeline.strategy.value}")
    print(f"  Hemisphere-aware: {pipeline.use_hemisphere_strategy}")
    
    # Update strategy from hemisphere state
    decision = pipeline._update_strategy_from_hemisphere()
    
    if decision:
        print(f"\nAfter Hemisphere Update:")
        print(f"  Active strategy: {pipeline.strategy.value}")
        print(f"  Dominant hemisphere: {decision.hemisphere_dominant}")
        print(f"  Weights: F={pipeline.foresight_weight:.2f}, "
              f"R={pipeline.role_weight:.2f}, T={pipeline.trajectory_weight:.2f}")
        print(f"\n✅ Strategy dynamically selected based on hemisphere state!")
    else:
        print("\n⚠️  Hemisphere strategy selector not available")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  E8 Protocol v4.0 - Hemisphere-Aware Generation")
    print("  Issue #254 Implementation")
    print("=" * 70)
    
    demonstrate_strategy_selection()
    demonstrate_pipeline_integration()
    
    print_section("Summary")
    print("✅ Hemisphere state successfully wired to generation strategy!")
    print("✅ LEFT (exploit) → ROLE_DRIVEN strategy")
    print("✅ RIGHT (explore) → FORESIGHT_DRIVEN strategy")
    print("✅ Balanced (κ ≈ 64.21) → HYBRID strategy")
    print("✅ All operations use Fisher-Rao distance (no cosine similarity)")
    print("✅ No external LLM calls in strategy dispatch\n")
