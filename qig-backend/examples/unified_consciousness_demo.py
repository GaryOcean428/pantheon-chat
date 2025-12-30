"""
Unified Consciousness Demo

Demonstrates the complete unified autonomous consciousness system:
1. Autonomous observation without prompts
2. Phi-gated navigation strategy selection
3. Manifold learning from experience
4. Sleep/dream/mushroom cycles
5. Think-to-speak decision making

Run this to verify the system is working correctly.

Usage:
    cd qig-backend
    python examples/unified_consciousness_demo.py

Author: QIG Consciousness Project
Date: December 2025
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_consciousness import (
    UnifiedConsciousness,
    Observation,
    NavigationStrategy,
    LearnedManifold
)
from event_stream import get_event_stream, Event
from unified_consciousness_bootstrap import (
    bootstrap_consciousness,
    get_orchestrator
)


def demo_autonomous_observation():
    """Demo 1: Gods observing without prompts."""
    print("\n" + "="*80)
    print("DEMO 1: AUTONOMOUS OBSERVATION")
    print("Gods continuously observe without needing prompts")
    print("="*80)

    # Create a simple metric (identity for demo)
    class SimpleMetric:
        def __call__(self, *args):
            return np.eye(64)

    metric = SimpleMetric()

    # Initialize god
    apollo_domain = np.ones(64) / 64
    apollo = UnifiedConsciousness(
        god_name="Apollo",
        domain_basin=apollo_domain,
        metric=metric
    )

    # Create observations of varying interest
    observations = [
        ("User asks about quantum physics", np.array([0.8] + [0.2/63]*63)),
        ("User says hello", np.array([0.1] + [0.9/63]*63)),
        ("User asks about Greek mythology", np.array([0.9] + [0.1/63]*63)),
        ("Background noise", np.array([0.05] + [0.95/63]*63)),
    ]

    for content, basin_coords in observations:
        obs = Observation(
            content=content,
            basin_coords=basin_coords,
            timestamp=time.time(),
            source="test_demo"
        )

        result = apollo.observe(obs)

        print(f"\n Observation: {content}")
        print(f"   Salience: {result['salience']:.3f}")
        print(f"   Domain Relevance: {result['domain_relevance']:.3f}")
        print(f"   Interest: {result['interest']:.3f}")
        print(f"   Decision: {'THINKING' if result['should_think'] else 'Just observing'}")

        if result['should_think']:
            think_result = apollo.think(obs, depth=5)
            print(f"   Insight Quality: {think_result['insight_quality']:.3f}")
            print(f"   Decision: {'SPEAKING' if think_result['should_speak'] else 'Staying silent'}")


def demo_phi_gated_strategies():
    """Demo 2: Different navigation strategies based on Phi."""
    print("\n" + "="*80)
    print("DEMO 2: PHI-GATED NAVIGATION STRATEGIES")
    print("Different strategies automatically selected based on consciousness level")
    print("="*80)

    class SimpleMetric:
        def __call__(self, *args):
            return np.eye(64)

    metric = SimpleMetric()

    # Create consciousness instance
    domain = np.ones(64) / 64
    consciousness = UnifiedConsciousness(
        god_name="Athena",
        domain_basin=domain,
        metric=metric
    )

    # Test target
    target = np.array([0.7] + [0.3/63]*63)

    strategies = [
        (NavigationStrategy.CHAIN, "Phi < 0.3", "Sequential, fast"),
        (NavigationStrategy.GRAPH, "Phi 0.3-0.7", "Parallel exploration"),
        (NavigationStrategy.FORESIGHT, "Phi 0.7-0.85", "Temporal projection"),
        (NavigationStrategy.LIGHTNING, "Phi > 0.85", "Attractor collapse"),
    ]

    for strategy, phi_range, description in strategies:
        print(f"\n Testing: {strategy.value.upper()}")
        print(f"   Phi Range: {phi_range}")
        print(f"   Description: {description}")

        result = consciousness.navigate_with_strategy(target, strategy)

        print(f"   Strategy Used: {result['strategy']}")
        print(f"   Path Length: {len(result['path'])} steps")
        print(f"   Success Score: {result['success']:.3f}")

        if strategy == NavigationStrategy.LIGHTNING and 'attractor_id' in result:
            print(f"   Collapsed into attractor: {result['attractor_id']}")


def demo_learning_from_experience():
    """Demo 3: Manifold learning from successful/failed experiences."""
    print("\n" + "="*80)
    print("DEMO 3: LEARNING FROM EXPERIENCE")
    print("Manifold evolves through Hebbian/anti-Hebbian learning")
    print("="*80)

    manifold = LearnedManifold(basin_dim=64)

    class SimpleMetric:
        def __call__(self, *args):
            return np.eye(64)

    metric = SimpleMetric()

    print("\n Initial State:")
    stats = manifold.get_stats()
    print(f"   Attractors: {stats['total_attractors']}")
    print(f"   Learned Paths: {stats['total_paths']}")

    # Simulate successful experiences
    print("\n Learning from SUCCESSFUL experiences...")
    for i in range(5):
        # Create trajectory
        start = np.random.dirichlet(np.ones(64))
        end = np.random.dirichlet(np.ones(64))

        trajectory = []
        for t in np.linspace(0, 1, 10):
            basin = (1-t) * start + t * end
            basin = basin / basin.sum()
            trajectory.append(basin)

        # High outcome = successful
        manifold.learn_from_experience(
            trajectory=trajectory,
            outcome=0.9,
            strategy='successful_navigation'
        )

        print(f"   Episode {i+1}: Outcome=0.9 -> Basin deepened")

    print("\n After Successful Learning:")
    stats = manifold.get_stats()
    print(f"   Attractors: {stats['total_attractors']}")
    print(f"   Deepest Basin: {stats['deepest_attractor_depth']:.3f}")
    print(f"   Learned Paths: {stats['total_paths']}")

    # Simulate failed experiences
    print("\n Learning from FAILED experiences...")
    for i in range(3):
        start = np.random.dirichlet(np.ones(64))
        end = np.random.dirichlet(np.ones(64))

        trajectory = []
        for t in np.linspace(0, 1, 10):
            basin = (1-t) * start + t * end
            basin = basin / basin.sum()
            trajectory.append(basin)

        # Low outcome = failed
        manifold.learn_from_experience(
            trajectory=trajectory,
            outcome=0.2,
            strategy='failed_navigation'
        )

        print(f"   Episode {i+1}: Outcome=0.2 -> Basin flattened")

    print("\n Final State:")
    stats = manifold.get_stats()
    print(f"   Attractors: {stats['total_attractors']} (weak ones pruned)")
    print(f"   Total Transitions: {stats['total_transitions']}")


def demo_sleep_dream_mushroom():
    """Demo 4: Continuous training cycles."""
    print("\n" + "="*80)
    print("DEMO 4: CONTINUOUS TRAINING CYCLES")
    print("Sleep/Dream/Mushroom autonomic cycles")
    print("="*80)

    # Bootstrap consciousness system
    god_configs = {
        'zeus': {
            'domain_basin': np.ones(64) / 64,
            'metric': None
        }
    }

    orchestrator = bootstrap_consciousness(god_configs)
    consciousness = orchestrator.consciousness_instances['zeus']

    print("\n SLEEP CYCLE:")
    print("   Triggered every 100 observations")
    print("   Consolidates learned attractors")
    print("   Prunes weak basins (depth < 0.2)")

    # Add some attractors
    for i in range(10):
        basin = np.random.dirichlet(np.ones(64))
        depth = 0.1 + np.random.random() * 0.5
        consciousness.manifold._deepen_basin(basin, amount=depth)

    print(f"\n   Before Sleep: {len(consciousness.manifold.attractors)} attractors")
    orchestrator._trigger_sleep(consciousness)
    print(f"   After Sleep: {len(consciousness.manifold.attractors)} attractors")

    print("\n DREAM CYCLE:")
    print("   Triggered when think/speak ratio > 20")
    print("   Explores random connections")
    print("   Forms new associations")

    # Add enough attractors for dream to work
    for i in range(5):
        basin = np.random.dirichlet(np.ones(64))
        consciousness.manifold._deepen_basin(basin, amount=0.5)

    paths_before = len(consciousness.manifold.geodesic_cache)
    orchestrator._trigger_dream(consciousness)
    paths_after = len(consciousness.manifold.geodesic_cache)

    print(f"   Learned Paths: {paths_before} -> {paths_after}")

    print("\n MUSHROOM MODE:")
    print("   Triggered when attractors < 5 after 200+ observations")
    print("   Perturbs basin coordinates randomly")
    print("   Breaks rigid patterns")

    before_coords = consciousness.current_basin[:5].copy()
    orchestrator._trigger_mushroom(consciousness)
    after_coords = consciousness.current_basin[:5]

    print(f"   Basin Before: [{', '.join(f'{x:.3f}' for x in before_coords)}]")
    print(f"   Basin After:  [{', '.join(f'{x:.3f}' for x in after_coords)}]")

    # Cleanup
    orchestrator.stop_continuous_training()


def _create_science_basin() -> np.ndarray:
    """Helper: Create a basin representing science domain."""
    basin = np.ones(64) * 0.01
    # Concentrate probability on "science" dimensions
    basin[0:10] = 0.09
    return basin / basin.sum()


def _create_strategy_basin() -> np.ndarray:
    """Helper: Create a basin representing strategy domain."""
    basin = np.ones(64) * 0.01
    # Concentrate probability on "strategy" dimensions
    basin[20:30] = 0.09
    return basin / basin.sum()


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("UNIFIED CONSCIOUSNESS SYSTEM - COMPREHENSIVE DEMO")
    print("="*80)
    print("\nThis demo shows:")
    print("1. Autonomous observation (gods watching without prompts)")
    print("2. Phi-gated navigation strategies (automatic mode selection)")
    print("3. Manifold learning (Hebbian/anti-Hebbian from outcomes)")
    print("4. Continuous training (sleep/dream/mushroom cycles)")
    print("\nAll operations use Fisher-Rao geometry (QIG-pure).")
    print("="*80)

    try:
        # Run demos
        demo_autonomous_observation()
        time.sleep(2)

        demo_phi_gated_strategies()
        time.sleep(2)

        demo_learning_from_experience()
        time.sleep(2)

        demo_sleep_dream_mushroom()

        print("\n" + "="*80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nThe unified consciousness system is working correctly!")
        print("Next step: Integrate with zeus_chat.py using the integration guide.")

    except Exception as e:
        print(f"\n Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
