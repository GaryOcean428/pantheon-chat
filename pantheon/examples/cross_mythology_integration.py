#!/usr/bin/env python3
"""
Example: Cross-Mythology Integration with Kernel Spawner
=========================================================

Demonstrates how cross-mythology mapping integrates with kernel spawning
and promotion workflows.

Authority: E8 Protocol v4.0, WP5.5
Status: EXAMPLE
Created: 2026-01-19
"""

from pantheon.cross_mythology import (
    resolve_god_name,
    find_similar_gods,
    get_mythology_info,
    get_external_equivalents,
)


def example_user_query_with_external_god():
    """
    Example: User references an external mythology god.
    System translates to Greek archetype.
    """
    print("\n" + "=" * 70)
    print("Example 1: User Query with External God Name")
    print("=" * 70)
    
    user_query = "I want to create a kernel like Thoth, focused on knowledge and writing"
    
    print(f"\nUser query: {user_query}")
    print("\n1. Extract external god name: 'Thoth'")
    
    # Resolve to Greek archetype
    greek_archetype = resolve_god_name("Thoth")
    print(f"2. Resolve to Greek archetype: {greek_archetype}")
    
    # Get mythology info
    info = get_mythology_info("Thoth")
    print(f"3. Domains: {', '.join(info['domain'][:5])}")
    
    # System uses Greek name internally
    print(f"\n✓ System creates chaos kernel with mentor: {greek_archetype}")
    print(f"  Internal naming: chaos_knowledge_001")
    print(f"  Mentor god: {greek_archetype}")
    print(f"  User-friendly reference: 'Thoth-like' or '{greek_archetype}'")


def example_domain_based_god_selection():
    """
    Example: Find appropriate god based on domain requirements.
    """
    print("\n" + "=" * 70)
    print("Example 2: Domain-Based God Selection")
    print("=" * 70)
    
    required_domains = ["wisdom", "strategy", "war"]
    
    print(f"\nRequired domains: {', '.join(required_domains)}")
    print("\n1. Search for matching gods:")
    
    matches = find_similar_gods(required_domains)
    
    for i, (god_name, match_count) in enumerate(matches[:5], 1):
        print(f"   {i}. {god_name} ({match_count} domain matches)")
    
    # Get top match
    best_god, _ = matches[0]
    print(f"\n2. Select best match: {best_god}")
    
    # Show external equivalents for user reference
    equiv = get_external_equivalents(best_god)
    if equiv:
        print(f"\n3. External mythology equivalents:")
        for myth, names in list(equiv.items())[:3]:
            print(f"   - {myth.capitalize()}: {', '.join(names)}")


def example_kernel_promotion_research():
    """
    Example: Chaos kernel reaches promotion candidate status.
    Research god name across mythologies.
    """
    print("\n" + "=" * 70)
    print("Example 3: Kernel Promotion - God Name Research")
    print("=" * 70)
    
    kernel_name = "chaos_strategy_042"
    phi_score = 0.45  # Above promotion threshold
    
    print(f"\nKernel: {kernel_name}")
    print(f"Φ score: {phi_score} (candidate for promotion)")
    print(f"\n1. Research phase: Find appropriate god name")
    
    # Kernel demonstrates strategic capability
    kernel_domains = ["strategy", "planning", "wisdom"]
    
    print(f"   Kernel domains: {', '.join(kernel_domains)}")
    
    # Find matching gods
    matches = find_similar_gods(kernel_domains)
    top_god, match_count = matches[0]
    
    print(f"\n2. Suggested Greek archetype: {top_god} ({match_count} matches)")
    
    # Show cross-mythology equivalents for context
    equiv = get_external_equivalents(top_god)
    print(f"\n3. Cross-mythology context:")
    for myth, names in equiv.items():
        for name in names:
            info = get_mythology_info(name)
            print(f"   - {name} ({myth}): {', '.join(info['domain'][:3])}...")
    
    print(f"\n4. Pantheon vote: Approve promotion to {top_god} aspect")
    print(f"   Epithet: {top_god} Strategos (strategic planner)")
    print(f"   ✓ chaos_strategy_042 → {top_god} Strategos")


def example_mythology_aware_logging():
    """
    Example: Log lifecycle events with mythology context.
    """
    print("\n" + "=" * 70)
    print("Example 4: Mythology-Aware Lifecycle Logging")
    print("=" * 70)
    
    print("\nLifecycle Event Log:")
    print("-" * 70)
    
    # Event 1: User creates kernel with external reference
    external_ref = "Odin"
    greek_archetype = resolve_god_name(external_ref)
    print(f"[SPAWN] User requested '{external_ref}'-like kernel")
    print(f"        Resolved to: {greek_archetype} (Norse: {external_ref})")
    print(f"        Created: chaos_executive_001 with mentor={greek_archetype}")
    
    # Event 2: Kernel learns from mythology
    print(f"\n[LEARN] chaos_executive_001 studying {greek_archetype} mythology")
    equiv = get_external_equivalents(greek_archetype)
    print(f"        Cross-cultural context: {len(equiv)} mythologies")
    
    # Event 3: Promotion with mythology research
    print(f"\n[PROMOTE] chaos_executive_001 → candidate")
    print(f"          Researching god names across mythologies")
    print(f"          Selected: {greek_archetype} Xenios (hospitality)")
    print(f"          External equivalents: {sum(len(v) for v in equiv.values())} gods")
    
    print()


def run_all_examples():
    """Run all integration examples."""
    print("\n" + "=" * 70)
    print("Cross-Mythology Integration Examples")
    print("=" * 70)
    
    example_user_query_with_external_god()
    example_domain_based_god_selection()
    example_kernel_promotion_research()
    example_mythology_aware_logging()
    
    print("\n" + "=" * 70)
    print("Integration Examples Complete")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. External god names map to Greek archetypes (metadata only)")
    print("2. Greek names remain canonical for all internal operations")
    print("3. Cross-mythology mapping enriches user experience and research")
    print("4. No runtime complexity added - simple lookup table")
    print()


if __name__ == "__main__":
    run_all_examples()
