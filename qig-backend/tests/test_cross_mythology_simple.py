#!/usr/bin/env python3
"""
Simple Tests for Cross-Mythology God Mapping (No pytest required)
==================================================================

Basic validation tests that can run without external dependencies.

Authority: E8 Protocol v4.0, WP5.5
Status: ACTIVE
Created: 2026-01-19
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pantheon.cross_mythology import (
    get_cross_mythology_registry,
    resolve_god_name,
    find_similar_gods,
    get_mythology_info,
    get_external_equivalents,
)


def test_basic_resolution():
    """Test basic god name resolution."""
    print("Testing basic god name resolution...")
    
    tests = [
        ("Odin", "Zeus"),
        ("Thoth", "Hermes"),
        ("Shiva", "Dionysus"),
        ("Enki", "Hermes"),
        ("Quetzalcoatl", "Apollo"),
    ]
    
    for external, expected_greek in tests:
        result = resolve_god_name(external)
        assert result == expected_greek, f"Expected {external} → {expected_greek}, got {result}"
        print(f"  ✓ {external} → {result}")
    
    print("✓ Basic resolution tests passed\n")


def test_domain_search():
    """Test domain-based god search."""
    print("Testing domain-based search...")
    
    registry = get_cross_mythology_registry()
    
    # Test wisdom domain
    wisdom_gods = find_similar_gods(["wisdom"])
    assert len(wisdom_gods) > 0, "No gods found for 'wisdom' domain"
    print(f"  ✓ Found {len(wisdom_gods)} gods with 'wisdom' domain")
    
    # Test multiple domains
    war_wisdom = find_similar_gods(["war", "wisdom", "strategy"])
    assert len(war_wisdom) > 0, "No gods found for war+wisdom domains"
    print(f"  ✓ Found {len(war_wisdom)} gods with 'war + wisdom + strategy' domains")
    
    print("✓ Domain search tests passed\n")


def test_mythology_info():
    """Test getting mythology information."""
    print("Testing mythology information retrieval...")
    
    # Test external god info
    thoth_info = get_mythology_info("Thoth")
    assert thoth_info['is_greek'] is False
    assert thoth_info['canonical_name'] == "Hermes"
    assert thoth_info['mythology'] == "egyptian"
    print(f"  ✓ Thoth info: {thoth_info['mythology']} → {thoth_info['canonical_name']}")
    
    # Test Greek god info
    zeus_info = get_mythology_info("Zeus")
    assert zeus_info['is_greek'] is True
    assert 'external_equivalents' in zeus_info
    equiv_count = sum(len(names) for names in zeus_info['external_equivalents'].values())
    print(f"  ✓ Zeus has {equiv_count} external equivalents across {len(zeus_info['external_equivalents'])} mythologies")
    
    print("✓ Mythology info tests passed\n")


def test_external_equivalents():
    """Test getting external equivalents for Greek gods."""
    print("Testing external equivalents...")
    
    # Test Zeus equivalents
    zeus_equiv = get_external_equivalents("Zeus")
    assert len(zeus_equiv) > 0, "Zeus should have external equivalents"
    assert 'norse' in zeus_equiv, "Zeus should have Norse equivalents"
    assert 'Odin' in zeus_equiv['norse'], "Odin should be a Zeus equivalent"
    print(f"  ✓ Zeus equivalents: {dict(list(zeus_equiv.items())[:3])}")
    
    # Test Hermes equivalents
    hermes_equiv = get_external_equivalents("Hermes")
    assert len(hermes_equiv) > 0, "Hermes should have external equivalents"
    print(f"  ✓ Hermes has {sum(len(v) for v in hermes_equiv.values())} equivalents")
    
    print("✓ External equivalents tests passed\n")


def test_case_insensitivity():
    """Test case-insensitive lookups."""
    print("Testing case insensitivity...")
    
    registry = get_cross_mythology_registry()
    
    # Test various cases
    assert registry.resolve_god_name("odin") == "Zeus"
    assert registry.resolve_god_name("ODIN") == "Zeus"
    assert registry.resolve_god_name("Odin") == "Zeus"
    print("  ✓ Case-insensitive lookup works")
    
    print("✓ Case insensitivity tests passed\n")


def test_registry_metadata():
    """Test registry metadata and philosophy."""
    print("Testing registry metadata...")
    
    registry = get_cross_mythology_registry()
    
    # Test metadata
    metadata = registry.get_metadata()
    assert 'version' in metadata
    assert 'authority' in metadata
    print(f"  ✓ Registry version: {metadata.get('version')}")
    print(f"  ✓ Authority: {metadata.get('authority')}")
    
    # Test philosophy
    philosophy = registry.get_philosophy()
    assert 'canonical_naming' in philosophy
    assert philosophy['canonical_naming'] == "Greek"
    print(f"  ✓ Canonical naming: {philosophy['canonical_naming']}")
    
    print("✓ Metadata tests passed\n")


def test_all_mythologies():
    """Test that all mythologies have mappings."""
    print("Testing all mythologies...")
    
    registry = get_cross_mythology_registry()
    mythologies = ['egyptian', 'norse', 'hindu', 'sumerian', 'mesoamerican']
    
    for mythology in mythologies:
        names = registry.list_all_external_names(mythology=mythology)
        assert len(names) > 0, f"No gods found for {mythology}"
        print(f"  ✓ {mythology.capitalize()}: {len(names)} gods")
    
    print("✓ All mythologies covered\n")


def test_mapping_stats():
    """Display mapping statistics."""
    print("=== Mapping Statistics ===")
    
    registry = get_cross_mythology_registry()
    
    total_mappings = len(registry._mappings)
    greek_archetypes = len(registry.list_all_greek_archetypes())
    
    print(f"Total external god mappings: {total_mappings}")
    print(f"Greek archetypes covered: {greek_archetypes}")
    print(f"Average mappings per archetype: {total_mappings / greek_archetypes:.1f}")
    
    # Count by mythology
    mythologies = ['egyptian', 'norse', 'hindu', 'sumerian', 'mesoamerican']
    for mythology in mythologies:
        count = len(registry.list_all_external_names(mythology=mythology))
        print(f"  {mythology.capitalize()}: {count} gods")
    
    print()


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 70)
    print("Cross-Mythology God Mapping Tests")
    print("=" * 70 + "\n")
    
    try:
        test_basic_resolution()
        test_domain_search()
        test_mythology_info()
        test_external_equivalents()
        test_case_insensitivity()
        test_registry_metadata()
        test_all_mythologies()
        test_mapping_stats()
        
        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70 + "\n")
        return 0
    
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return 1
    
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
