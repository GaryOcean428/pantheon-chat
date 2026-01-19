#!/usr/bin/env python3
"""
Tests for Cross-Mythology God Mapping
======================================

Tests the cross-mythology registry loader and lookup functions.

Authority: E8 Protocol v4.0, WP5.5
Status: ACTIVE
Created: 2026-01-19
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pantheon.cross_mythology import (
    CrossMythologyRegistry,
    get_cross_mythology_registry,
    resolve_god_name,
    find_similar_gods,
    get_mythology_info,
    get_external_equivalents,
)


class TestCrossMythologyRegistry:
    """Test the CrossMythologyRegistry class."""
    
    def test_registry_loads(self):
        """Test that registry loads successfully."""
        registry = get_cross_mythology_registry()
        assert registry is not None
        assert registry._raw_data is not None
        assert len(registry._mappings) > 0
    
    def test_resolve_egyptian_gods(self):
        """Test resolving Egyptian god names."""
        registry = get_cross_mythology_registry()
        
        # Test major Egyptian gods
        assert registry.resolve_god_name("Thoth") == "Hermes"
        assert registry.resolve_god_name("Ra") == "Apollo"
        assert registry.resolve_god_name("Anubis") == "Hades"
        assert registry.resolve_god_name("Isis") == "Demeter"
        assert registry.resolve_god_name("Hathor") == "Aphrodite"
    
    def test_resolve_norse_gods(self):
        """Test resolving Norse god names."""
        registry = get_cross_mythology_registry()
        
        # Test major Norse gods
        assert registry.resolve_god_name("Odin") == "Zeus"
        assert registry.resolve_god_name("Thor") == "Zeus"
        assert registry.resolve_god_name("Loki") == "Hermes"
        assert registry.resolve_god_name("Freya") == "Aphrodite"
        assert registry.resolve_god_name("Hel") == "Hades"
    
    def test_resolve_hindu_gods(self):
        """Test resolving Hindu god names."""
        registry = get_cross_mythology_registry()
        
        # Test major Hindu gods
        assert registry.resolve_god_name("Shiva") == "Dionysus"
        assert registry.resolve_god_name("Vishnu") == "Zeus"
        assert registry.resolve_god_name("Brahma") == "Hephaestus"
        assert registry.resolve_god_name("Saraswati") == "Athena"
        assert registry.resolve_god_name("Kali") == "Ares"
    
    def test_resolve_sumerian_gods(self):
        """Test resolving Sumerian god names."""
        registry = get_cross_mythology_registry()
        
        # Test major Sumerian gods
        assert registry.resolve_god_name("Enki") == "Hermes"
        assert registry.resolve_god_name("Enlil") == "Zeus"
        assert registry.resolve_god_name("Inanna") == "Aphrodite"
        assert registry.resolve_god_name("Shamash") == "Apollo"
    
    def test_resolve_mesoamerican_gods(self):
        """Test resolving Mayan/Aztec god names."""
        registry = get_cross_mythology_registry()
        
        # Test major Mesoamerican gods
        assert registry.resolve_god_name("Quetzalcoatl") == "Apollo"
        assert registry.resolve_god_name("Tlaloc") == "Poseidon"
        assert registry.resolve_god_name("Tezcatlipoca") == "Hades"
        assert registry.resolve_god_name("Xochiquetzal") == "Aphrodite"
    
    def test_resolve_case_insensitive(self):
        """Test that god name resolution is case-insensitive."""
        registry = get_cross_mythology_registry()
        
        # Test various cases
        assert registry.resolve_god_name("odin") == "Zeus"
        assert registry.resolve_god_name("ODIN") == "Zeus"
        assert registry.resolve_god_name("Odin") == "Zeus"
        assert registry.resolve_god_name("oDiN") == "Zeus"
    
    def test_resolve_unknown_god_raises_error(self):
        """Test that unknown god names raise KeyError."""
        registry = get_cross_mythology_registry()
        
        with pytest.raises(KeyError):
            registry.resolve_god_name("UnknownGod123")
    
    def test_find_similar_gods_by_domain(self):
        """Test finding gods by domain keywords."""
        registry = get_cross_mythology_registry()
        
        # Test wisdom domain
        wisdom_gods = registry.find_similar_gods(["wisdom"])
        assert len(wisdom_gods) > 0
        
        # Hermes and Athena should appear (common wisdom domain)
        god_names = [name for name, _ in wisdom_gods]
        assert "Hermes" in god_names or "Athena" in god_names
        
        # Test multiple domains
        war_wisdom = registry.find_similar_gods(["war", "wisdom"])
        assert len(war_wisdom) > 0
        
        # Results should be sorted by match count (descending)
        for i in range(len(war_wisdom) - 1):
            assert war_wisdom[i][1] >= war_wisdom[i + 1][1]
    
    def test_find_similar_gods_empty_domain(self):
        """Test that empty domain list returns no matches."""
        registry = get_cross_mythology_registry()
        
        matches = registry.find_similar_gods([])
        assert len(matches) == 0
    
    def test_get_mythology_info_external_god(self):
        """Test getting info for external god name."""
        registry = get_cross_mythology_registry()
        
        info = registry.get_mythology_info("Thoth")
        assert info['is_greek'] is False
        assert info['canonical_name'] == "Hermes"
        assert info['mythology'] == "egyptian"
        assert info['external_name'] == "Thoth"
        assert 'domain' in info
        assert 'notes' in info
        assert len(info['domain']) > 0
    
    def test_get_mythology_info_greek_god(self):
        """Test getting info for Greek god name."""
        registry = get_cross_mythology_registry()
        
        info = registry.get_mythology_info("Zeus")
        assert info['is_greek'] is True
        assert info['canonical_name'] == "Zeus"
        assert 'external_equivalents' in info
        assert len(info['external_equivalents']) > 0
        assert 'mapping_count' in info
    
    def test_get_external_equivalents(self):
        """Test getting external equivalents for Greek gods."""
        registry = get_cross_mythology_registry()
        
        # Zeus should have multiple equivalents
        zeus_equiv = registry.get_external_equivalents("Zeus")
        assert len(zeus_equiv) > 0
        assert 'norse' in zeus_equiv
        assert 'Odin' in zeus_equiv['norse']
        
        # Hermes should have equivalents
        hermes_equiv = registry.get_external_equivalents("Hermes")
        assert len(hermes_equiv) > 0
        assert 'egyptian' in hermes_equiv
        assert 'Thoth' in hermes_equiv['egyptian']
    
    def test_get_external_equivalents_no_mappings(self):
        """Test getting equivalents for Greek god with no mappings."""
        registry = get_cross_mythology_registry()
        
        # Create a mock scenario - some gods may not have external mappings
        # This returns empty dict, not an error
        equiv = registry.get_external_equivalents("NonExistentGod")
        assert equiv == {}
    
    def test_list_all_greek_archetypes(self):
        """Test listing all Greek archetypes."""
        registry = get_cross_mythology_registry()
        
        archetypes = registry.list_all_greek_archetypes()
        assert len(archetypes) > 0
        assert isinstance(archetypes, list)
        
        # Should include major archetypes
        assert "Zeus" in archetypes
        assert "Hermes" in archetypes
        assert "Apollo" in archetypes
        
        # Should be sorted
        assert archetypes == sorted(archetypes)
    
    def test_list_all_external_names(self):
        """Test listing all external god names."""
        registry = get_cross_mythology_registry()
        
        # All external names
        all_names = registry.list_all_external_names()
        assert len(all_names) > 0
        assert "Odin" in all_names
        assert "Thoth" in all_names
        assert "Shiva" in all_names
        
        # Filtered by mythology
        norse_names = registry.list_all_external_names(mythology="norse")
        assert len(norse_names) > 0
        assert "Odin" in norse_names
        assert "Thoth" not in norse_names  # Not Norse
    
    def test_get_number_symbolism(self):
        """Test getting number symbolism info."""
        registry = get_cross_mythology_registry()
        
        symbolism = registry.get_number_symbolism()
        assert isinstance(symbolism, dict)
        
        # Should have entries for key E8 numbers
        assert "4" in symbolism or "8" in symbolism or "64" in symbolism
    
    def test_get_metadata(self):
        """Test getting registry metadata."""
        registry = get_cross_mythology_registry()
        
        metadata = registry.get_metadata()
        assert isinstance(metadata, dict)
        assert 'version' in metadata
        assert 'status' in metadata
        assert 'authority' in metadata
    
    def test_get_philosophy(self):
        """Test getting mapping philosophy."""
        registry = get_cross_mythology_registry()
        
        philosophy = registry.get_philosophy()
        assert isinstance(philosophy, dict)
        assert 'canonical_naming' in philosophy
        assert philosophy['canonical_naming'] == "Greek"


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""
    
    def test_resolve_god_name_function(self):
        """Test resolve_god_name convenience function."""
        assert resolve_god_name("Odin") == "Zeus"
        assert resolve_god_name("Thoth") == "Hermes"
    
    def test_find_similar_gods_function(self):
        """Test find_similar_gods convenience function."""
        matches = find_similar_gods(["wisdom", "knowledge"])
        assert len(matches) > 0
        assert isinstance(matches, list)
        assert isinstance(matches[0], tuple)
        assert len(matches[0]) == 2  # (god_name, count)
    
    def test_get_mythology_info_function(self):
        """Test get_mythology_info convenience function."""
        info = get_mythology_info("Thoth")
        assert info['is_greek'] is False
        assert info['canonical_name'] == "Hermes"
    
    def test_get_external_equivalents_function(self):
        """Test get_external_equivalents convenience function."""
        equiv = get_external_equivalents("Zeus")
        assert len(equiv) > 0
        assert isinstance(equiv, dict)


class TestMappingConsistency:
    """Test consistency of mappings across mythologies."""
    
    def test_all_archetypes_exist_in_pantheon(self):
        """Test that all Greek archetypes referenced exist in pantheon registry."""
        # This is a validation test - in real implementation, we'd load
        # pantheon/registry.yaml and verify all archetypes exist
        registry = get_cross_mythology_registry()
        
        archetypes = registry.list_all_greek_archetypes()
        
        # Core pantheon gods that should be present
        expected_gods = ["Zeus", "Athena", "Apollo", "Hermes", "Ares", 
                        "Artemis", "Hephaestus", "Aphrodite", "Hades", 
                        "Demeter", "Dionysus", "Poseidon", "Hera"]
        
        for god in expected_gods:
            assert god in archetypes or god == "Themis"  # Themis may or may not be in registry
    
    def test_alternative_mappings_are_valid(self):
        """Test that alternative mappings reference valid Greek archetypes."""
        registry = get_cross_mythology_registry()
        
        archetypes = set(registry.list_all_greek_archetypes())
        
        # Check all alternative mappings
        for mapping in registry._mappings.values():
            if mapping.alternative_mapping:
                # Alternative mapping should be a valid Greek archetype
                # or a known Greek god (may not have external mappings)
                assert mapping.alternative_mapping in archetypes or \
                       mapping.alternative_mapping in ["Themis", "Heracles", "Thanatos"]
    
    def test_no_duplicate_external_names(self):
        """Test that external god names are unique (case-insensitive)."""
        registry = get_cross_mythology_registry()
        
        names = [m.external_name.lower() for m in registry._mappings.values()]
        assert len(names) == len(set(names)), "Found duplicate external god names"


class TestDomainCoverage:
    """Test domain coverage across mythologies."""
    
    def test_major_domains_covered(self):
        """Test that major conceptual domains are covered."""
        registry = get_cross_mythology_registry()
        
        # Major domains that should have god mappings
        major_domains = [
            "wisdom", "war", "love", "death", "creation",
            "communication", "justice", "nature", "sun", "underworld"
        ]
        
        for domain in major_domains:
            matches = registry.find_similar_gods([domain])
            assert len(matches) > 0, f"No gods found for domain: {domain}"
    
    def test_each_mythology_has_multiple_mappings(self):
        """Test that each mythology has multiple god mappings."""
        registry = get_cross_mythology_registry()
        
        mythologies = ['egyptian', 'norse', 'hindu', 'sumerian', 'mesoamerican']
        
        for mythology in mythologies:
            names = registry.list_all_external_names(mythology=mythology)
            assert len(names) >= 5, f"Mythology '{mythology}' has too few mappings"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for cross-mythology system."""
    
    def test_research_workflow(self):
        """Test typical research workflow: external name → Greek archetype → info."""
        registry = get_cross_mythology_registry()
        
        # 1. Resolve external name
        external_name = "Odin"
        greek_name = registry.resolve_god_name(external_name)
        assert greek_name == "Zeus"
        
        # 2. Get info about external god
        external_info = registry.get_mythology_info(external_name)
        assert external_info['mythology'] == "norse"
        assert external_info['canonical_name'] == "Zeus"
        
        # 3. Get info about Greek archetype
        greek_info = registry.get_mythology_info(greek_name)
        assert greek_info['is_greek'] is True
        
        # 4. Find external equivalents
        equiv = registry.get_external_equivalents(greek_name)
        assert 'norse' in equiv
        assert 'Odin' in equiv['norse']
    
    def test_domain_search_workflow(self):
        """Test workflow: search by domain → find god → get equivalents."""
        registry = get_cross_mythology_registry()
        
        # 1. Search by domain
        matches = registry.find_similar_gods(["wisdom", "knowledge"])
        assert len(matches) > 0
        
        # 2. Get top match
        top_god, _ = matches[0]
        
        # 3. Get external equivalents
        equiv = registry.get_external_equivalents(top_god)
        assert isinstance(equiv, dict)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
