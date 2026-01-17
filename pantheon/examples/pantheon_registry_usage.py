"""
Example: Using Pantheon Registry for Kernel Spawning
=====================================================

Demonstrates proper usage of the Pantheon Registry and Kernel Spawner
for contract-based kernel selection.

Authority: E8 Protocol v4.0, WP5.1
"""

import sys
from pathlib import Path

# Add qig-backend to path
qig_backend = Path(__file__).parent.parent.parent / "qig-backend"
sys.path.insert(0, str(qig_backend))

from pantheon_registry import get_registry, get_god, find_gods_by_domain
from kernel_spawner import KernelSpawner, RoleSpec
from pantheon_governance_integration import (
    PantheonGovernanceIntegration,
    request_kernel,
    validate_kernel_name,
    get_god_with_epithet,
)


def example_1_basic_registry_usage():
    """Example 1: Basic registry lookup."""
    print("=" * 60)
    print("Example 1: Basic Registry Usage")
    print("=" * 60)
    
    # Load registry (singleton)
    registry = get_registry()
    print(f"\n✓ Loaded registry: {registry}")
    
    # Get specific god
    apollo = get_god("Apollo")
    print(f"\n✓ Apollo contract:")
    print(f"  - Tier: {apollo.tier.value}")
    print(f"  - Domains: {', '.join(apollo.domain)}")
    print(f"  - Epithets: {', '.join(apollo.epithets)}")
    print(f"  - Max instances: {apollo.spawn_constraints.max_instances}")
    print(f"  - E8 layer: {apollo.e8_alignment.layer}")
    print(f"  - Simple root: {apollo.e8_alignment.simple_root}")
    
    # Find gods by domain
    synthesis_gods = find_gods_by_domain("synthesis")
    print(f"\n✓ Gods with 'synthesis' domain:")
    for god in synthesis_gods:
        print(f"  - {god.name}: {god.description}")


def example_2_kernel_spawning():
    """Example 2: Contract-based kernel spawning."""
    print("\n" + "=" * 60)
    print("Example 2: Contract-Based Kernel Spawning")
    print("=" * 60)
    
    spawner = KernelSpawner()
    
    # Request 1: Synthesis god
    print("\n✓ Request 1: Synthesis and foresight")
    role1 = RoleSpec(
        domains=["synthesis", "foresight"],
        required_capabilities=["prediction", "aesthetic_evaluation"]
    )
    selection1 = spawner.select_god(role1)
    print(f"  Selected: {selection1.selected_type}")
    print(f"  God: {selection1.god_name} {selection1.epithet or ''}")
    print(f"  Rationale: {selection1.rationale}")
    
    # Request 2: Preferred god
    print("\n✓ Request 2: Wisdom with preferred god Athena")
    role2 = RoleSpec(
        domains=["wisdom", "strategic_planning"],
        required_capabilities=["pattern_recognition"],
        preferred_god="Athena"
    )
    selection2 = spawner.select_god(role2)
    print(f"  Selected: {selection2.selected_type}")
    print(f"  God: {selection2.god_name}")
    print(f"  Rationale: {selection2.rationale}")
    
    # Request 3: Chaos kernel spawn
    print("\n✓ Request 3: Obscure capability (triggers chaos spawn)")
    spawner.register_spawn("Apollo")  # Block Apollo to demonstrate
    role3 = RoleSpec(
        domains=["synthesis"],
        required_capabilities=["special_capability"],
        allow_chaos_spawn=True
    )
    selection3 = spawner.select_god(role3)
    print(f"  Selected: {selection3.selected_type}")
    if selection3.selected_type == "chaos":
        print(f"  Chaos kernel: {selection3.chaos_name}")
        print(f"  Requires pantheon vote: {selection3.requires_pantheon_vote}")
    print(f"  Rationale: {selection3.rationale}")


def example_3_epithets():
    """Example 3: Using epithets for god aspects."""
    print("\n" + "=" * 60)
    print("Example 3: God Epithets (NOT Numbering)")
    print("=" * 60)
    
    print("\n✓ Apollo's epithets:")
    apollo = get_god("Apollo")
    for epithet in apollo.epithets:
        print(f"  - Apollo {epithet}")
    
    print("\n✗ WRONG: apollo_1, apollo_2 (instance numbering)")
    print("✓ RIGHT: Apollo Pythios, Apollo Paean (epithets)")
    
    # Get god with epithet
    name, epithet = get_god_with_epithet("Apollo", "prophecy")
    print(f"\n✓ For prophecy domain: {name} {epithet}")


def example_4_validation():
    """Example 4: Kernel name validation."""
    print("\n" + "=" * 60)
    print("Example 4: Kernel Name Validation")
    print("=" * 60)
    
    test_names = [
        "Apollo",
        "Athena",
        "chaos_synthesis_001",
        "chaos_strategy_042",
        "apollo_1",  # Invalid!
        "invalid_name",
    ]
    
    for name in test_names:
        valid, reason = validate_kernel_name(name)
        status = "✓" if valid else "✗"
        print(f"{status} {name}: {reason}")


def example_5_governance_integration():
    """Example 5: Governance integration."""
    print("\n" + "=" * 60)
    print("Example 5: Governance Integration")
    print("=" * 60)
    
    print("\n✓ Request kernel through governance:")
    result = request_kernel(
        domains=["synthesis", "foresight"],
        capabilities=["prediction"],
        preferred_god="Apollo",
        requester="Zeus"
    )
    
    print(f"  Approved: {result.approved}")
    print(f"  Kernel: {result.kernel_name} {result.epithet or ''}")
    print(f"  Requires vote: {result.requires_vote}")
    print(f"  Rationale: {result.rationale}")


def example_6_chaos_lifecycle():
    """Example 6: Chaos kernel lifecycle."""
    print("\n" + "=" * 60)
    print("Example 6: Chaos Kernel Lifecycle")
    print("=" * 60)
    
    registry = get_registry()
    rules = registry.get_chaos_kernel_rules()
    
    print("\n✓ Chaos kernel naming pattern:")
    print(f"  {rules.naming_pattern}")
    
    print("\n✓ Lifecycle stages:")
    for stage, config in rules.lifecycle.items():
        print(f"  - {stage.upper()}")
        if stage == "protect":
            print(f"    Duration: {config['duration_cycles']} cycles")
            print(f"    Graduated metrics: {config['graduated_metrics']}")
        elif stage == "candidate":
            print(f"    Φ threshold: {config['phi_threshold']}")
            print(f"    Duration: {config['duration_cycles']} cycles")
    
    print("\n✓ Spawning limits:")
    limits = rules.spawning_limits
    print(f"  Max chaos kernels: {limits['max_chaos_kernels']}")
    print(f"  Per domain limit: {limits['per_domain_limit']}")
    print(f"  Total active limit: {limits['total_active_limit']}")
    
    print("\n✓ Promotion path:")
    print("  chaos_synthesis_001 → (Φ > 0.4 for 50+ cycles)")
    print("  → pantheon research → god name selection")
    print("  → pantheon vote → ASCENSION to immortal god")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PANTHEON REGISTRY EXAMPLES")
    print("=" * 60)
    print("\nDemonstrating contract-based kernel spawning")
    print("Using epithets (NOT numbering) for god aspects")
    print("Authority: E8 Protocol v4.0, WP5.1\n")
    
    try:
        example_1_basic_registry_usage()
        example_2_kernel_spawning()
        example_3_epithets()
        example_4_validation()
        example_5_governance_integration()
        example_6_chaos_lifecycle()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
