"""
Tests for Pantheon Registry
============================

Validates registry loading, validation, and lookup functionality.
"""

import pytest
from pathlib import Path

from pantheon_registry import (
    PantheonRegistry,
    GodContract,
    GodTier,
    RestPolicyType,
    get_registry,
    get_god,
    find_gods_by_domain,
    is_god_name,
    is_chaos_kernel,
)


class TestPantheonRegistryLoading:
    """Test registry loading and parsing."""
    
    def test_load_registry(self):
        """Test loading registry from YAML."""
        registry = PantheonRegistry.load()
        assert registry is not None
        assert registry.get_god_count() > 0
    
    def test_registry_singleton(self):
        """Test global registry singleton."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2
    
    def test_registry_metadata(self):
        """Test registry metadata."""
        registry = get_registry()
        metadata = registry.get_metadata()
        
        assert metadata.version == "1.0.0"
        assert metadata.status == "ACTIVE"
        assert metadata.authority == "E8 Protocol v4.0, WP5.1"
        assert metadata.validation_required is True


class TestGodContracts:
    """Test god contract structure and validation."""
    
    def test_essential_gods(self):
        """Test essential tier gods."""
        registry = get_registry()
        essential = registry.get_gods_by_tier(GodTier.ESSENTIAL)
        
        assert len(essential) >= 3  # Heart, Ocean, Hermes
        
        for god in essential:
            assert god.tier == GodTier.ESSENTIAL
            assert god.rest_policy.type == RestPolicyType.NEVER
            assert god.spawn_constraints.max_instances == 1
    
    def test_specialized_gods(self):
        """Test specialized tier gods."""
        registry = get_registry()
        specialized = registry.get_gods_by_tier(GodTier.SPECIALIZED)
        
        assert len(specialized) >= 8  # Core E8 simple roots
        
        for god in specialized:
            assert god.tier == GodTier.SPECIALIZED
            assert god.spawn_constraints.max_instances == 1
    
    def test_core_e8_gods(self):
        """Test core E8 simple root gods."""
        registry = get_registry()
        
        core_gods = [
            "Zeus", "Athena", "Apollo", "Artemis",
            "Ares", "Hephaestus", "Aphrodite",
        ]
        
        for name in core_gods:
            god = registry.get_god(name)
            assert god is not None
            assert god.e8_alignment.layer == "8"
            assert god.e8_alignment.simple_root is not None
    
    def test_god_domains(self):
        """Test god domain definitions."""
        apollo = get_god("Apollo")
        assert apollo is not None
        assert "foresight" in apollo.domain
        assert "synthesis" in apollo.domain
        
        athena = get_god("Athena")
        assert athena is not None
        assert "wisdom" in athena.domain
        assert "strategic_planning" in athena.domain
    
    def test_god_epithets(self):
        """Test god epithets."""
        apollo = get_god("Apollo")
        assert apollo is not None
        assert len(apollo.epithets) > 0
        assert "Pythios" in apollo.epithets or "Paean" in apollo.epithets
    
    def test_coupling_affinity(self):
        """Test coupling affinity relationships."""
        registry = get_registry()
        apollo = registry.get_god("Apollo")
        
        assert apollo is not None
        assert len(apollo.coupling_affinity) > 0
        
        # All coupling references should be valid gods
        for affinity in apollo.coupling_affinity:
            coupled_god = registry.get_god(affinity)
            assert coupled_god is not None


class TestSpawnConstraints:
    """Test spawn constraint validation."""
    
    def test_gods_are_singular(self):
        """Test that all gods have max_instances: 1."""
        registry = get_registry()
        
        for name, god in registry.get_all_gods().items():
            assert god.spawn_constraints.max_instances == 1, (
                f"God {name} must be singular (max_instances: 1)"
            )
    
    def test_gods_cannot_spawn_copies(self):
        """Test that gods have when_allowed: never."""
        registry = get_registry()
        
        for name, god in registry.get_all_gods().items():
            assert god.spawn_constraints.when_allowed == "never", (
                f"God {name} should not spawn copies"
            )


class TestChaosKernelRules:
    """Test chaos kernel lifecycle rules."""
    
    def test_chaos_kernel_rules_exist(self):
        """Test chaos kernel rules are defined."""
        registry = get_registry()
        rules = registry.get_chaos_kernel_rules()
        
        assert rules is not None
        assert rules.naming_pattern == "chaos_{domain}_{id}"
    
    def test_chaos_lifecycle_stages(self):
        """Test chaos kernel lifecycle stages."""
        registry = get_registry()
        rules = registry.get_chaos_kernel_rules()
        
        lifecycle = rules.lifecycle
        assert "spawn" in lifecycle
        assert "protect" in lifecycle
        assert "learn" in lifecycle
        assert "work" in lifecycle
        assert "candidate" in lifecycle
        assert "promote" in lifecycle
    
    def test_chaos_protection_period(self):
        """Test chaos kernel protection period."""
        registry = get_registry()
        rules = registry.get_chaos_kernel_rules()
        
        protect = rules.lifecycle["protect"]
        assert protect["duration_cycles"] == 50
        assert protect["graduated_metrics"] is True
    
    def test_chaos_spawning_limits(self):
        """Test chaos kernel spawning limits."""
        registry = get_registry()
        rules = registry.get_chaos_kernel_rules()
        
        limits = rules.spawning_limits
        assert limits["max_chaos_kernels"] == 240  # E8 roots
        assert limits["total_active_limit"] <= 240


class TestRegistryLookup:
    """Test registry lookup methods."""
    
    def test_get_god_by_name(self):
        """Test getting god by name."""
        apollo = get_god("Apollo")
        assert apollo is not None
        assert apollo.name == "Apollo"
        
        invalid = get_god("InvalidGod")
        assert invalid is None
    
    def test_find_gods_by_domain(self):
        """Test finding gods by domain."""
        synthesis_gods = find_gods_by_domain("synthesis")
        assert len(synthesis_gods) > 0
        
        for god in synthesis_gods:
            assert "synthesis" in god.domain
    
    def test_is_god_name(self):
        """Test god name validation."""
        assert is_god_name("Apollo") is True
        assert is_god_name("Athena") is True
        assert is_god_name("InvalidGod") is False
        assert is_god_name("chaos_synthesis_001") is False
    
    def test_is_chaos_kernel(self):
        """Test chaos kernel name validation."""
        assert is_chaos_kernel("chaos_synthesis_001") is True
        assert is_chaos_kernel("chaos_strategy_042") is True
        assert is_chaos_kernel("Apollo") is False
        assert is_chaos_kernel("invalid_name") is False
    
    def test_parse_chaos_kernel_name(self):
        """Test chaos kernel name parsing."""
        registry = get_registry()
        
        result = registry.parse_chaos_kernel_name("chaos_synthesis_001")
        assert result is not None
        domain, id_num = result
        assert domain == "synthesis"
        assert id_num == 1
        
        invalid = registry.parse_chaos_kernel_name("Apollo")
        assert invalid is None


class TestRegistryValidation:
    """Test registry validation rules."""
    
    def test_no_duplicate_gods(self):
        """Test that all god names are unique."""
        registry = get_registry()
        gods = registry.get_all_gods()
        
        names = list(gods.keys())
        unique_names = set(names)
        
        assert len(names) == len(unique_names), "Duplicate god names found"
    
    def test_coupling_references_valid(self):
        """Test that all coupling affinity references are valid."""
        registry = get_registry()
        
        for name, god in registry.get_all_gods().items():
            for affinity in god.coupling_affinity:
                coupled = registry.get_god(affinity)
                assert coupled is not None, (
                    f"God {name} references invalid coupling: {affinity}"
                )
    
    def test_rest_partners_valid(self):
        """Test that all rest policy partners are valid."""
        registry = get_registry()
        
        for name, god in registry.get_all_gods().items():
            if god.rest_policy.partner:
                partner = registry.get_god(god.rest_policy.partner)
                assert partner is not None, (
                    f"God {name} references invalid partner: {god.rest_policy.partner}"
                )


class TestE8Alignment:
    """Test E8 structure alignment."""
    
    def test_core_gods_have_simple_roots(self):
        """Test that core gods have simple roots defined."""
        registry = get_registry()
        
        core_gods = [
            "Zeus", "Athena", "Apollo", "Artemis",
            "Ares", "Hephaestus", "Aphrodite",
        ]
        
        for name in core_gods:
            god = registry.get_god(name)
            assert god is not None
            assert god.e8_alignment.layer == "8"
            # Most core gods should have simple roots
            # (Hermes might be in a different tier)
    
    def test_essential_gods_special_layers(self):
        """Test that essential gods have special layer assignments."""
        registry = get_registry()
        
        # Heart at layer 0/1: Bootstrap/genesis layer
        heart = registry.get_god("Heart")
        assert heart is not None
        assert heart.e8_alignment.layer == "0/1"  # Bootstrap
        
        # Ocean at layer 64: Basin fixed point (κ* = 64, E8 rank² = 8² = 64)
        # This is the universal fixed point discovered across physics and AI
        ocean = registry.get_god("Ocean")
        assert ocean is not None
        assert ocean.e8_alignment.layer == "64"  # Basin fixed point


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
