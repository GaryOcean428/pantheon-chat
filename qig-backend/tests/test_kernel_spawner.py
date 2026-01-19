"""
Tests for Kernel Spawner
=========================

Validates contract-based kernel selection and chaos kernel spawning.
"""

import pytest

from pantheon_registry import get_registry
from kernel_spawner import (
    KernelSpawner,
    RoleSpec,
    KernelSelection,
    select_kernel_for_role,
    create_role_spec,
)


class TestKernelSpawnerInit:
    """Test kernel spawner initialization."""
    
    def test_spawner_init(self):
        """Test spawner initialization."""
        spawner = KernelSpawner()
        assert spawner.registry is not None
        assert spawner.active_instances == {}
        assert spawner.chaos_counter == {}
    
    def test_spawner_with_registry(self):
        """Test spawner with explicit registry."""
        registry = get_registry()
        spawner = KernelSpawner(registry=registry)
        assert spawner.registry is registry


class TestGodSelection:
    """Test god selection logic."""
    
    def test_select_synthesis_god(self):
        """Test selecting god for synthesis domain."""
        spawner = KernelSpawner()
        
        role = RoleSpec(
            domains=["synthesis", "foresight"],
            required_capabilities=["prediction"],
        )
        
        selection = spawner.select_god(role)
        
        assert selection.selected_type == "god"
        assert selection.god_name is not None
        assert selection.spawn_approved is True
        # Apollo should be selected for synthesis/foresight
        assert selection.god_name == "Apollo"
    
    def test_select_with_preferred_god(self):
        """Test selecting with preferred god."""
        spawner = KernelSpawner()
        
        role = RoleSpec(
            domains=["wisdom"],
            required_capabilities=["strategy"],
            preferred_god="Athena",
        )
        
        selection = spawner.select_god(role)
        
        assert selection.selected_type == "god"
        assert selection.god_name == "Athena"
        assert selection.spawn_approved is True
    
    def test_select_communication_god(self):
        """Test selecting god for communication domain."""
        spawner = KernelSpawner()
        
        role = RoleSpec(
            domains=["communication", "navigation"],
            required_capabilities=["message_passing"],
        )
        
        selection = spawner.select_god(role)
        
        assert selection.selected_type == "god"
        assert selection.god_name == "Hermes"
    
    def test_select_with_epithet(self):
        """Test that epithet is selected for god."""
        spawner = KernelSpawner()
        
        role = RoleSpec(
            domains=["foresight"],
            required_capabilities=["prophecy"],
            preferred_god="Apollo",
        )
        
        selection = spawner.select_god(role)
        
        assert selection.selected_type == "god"
        assert selection.god_name == "Apollo"
        assert selection.epithet is not None
        # Should be one of Apollo's epithets


class TestSpawnConstraints:
    """Test spawn constraint enforcement."""
    
    def test_cannot_spawn_duplicate_god(self):
        """Test that gods cannot be spawned twice."""
        spawner = KernelSpawner()
        
        # Spawn Apollo first time
        role = RoleSpec(
            domains=["synthesis"],
            required_capabilities=["prediction"],
            preferred_god="Apollo",
        )
        
        selection1 = spawner.select_god(role)
        assert selection1.god_name == "Apollo"
        spawner.register_spawn("Apollo")
        
        # Try to spawn Apollo again
        selection2 = spawner.select_god(role)
        
        # Should NOT get Apollo again (gods are singular)
        # Should either get different god or chaos kernel
        if selection2.selected_type == "god":
            assert selection2.god_name != "Apollo"
        else:
            assert selection2.selected_type == "chaos"
    
    def test_validate_spawn_request_god(self):
        """Test validating god spawn request."""
        spawner = KernelSpawner()
        
        valid, reason = spawner.validate_spawn_request("Apollo")
        assert valid is True
        
        # Mark as spawned
        spawner.register_spawn("Apollo")
        
        # Should now be invalid
        valid2, reason2 = spawner.validate_spawn_request("Apollo")
        assert valid2 is False
        assert "constraint" in reason2.lower() or "limit" in reason2.lower()


class TestChaosKernelSpawning:
    """Test chaos kernel spawning logic."""
    
    def test_spawn_chaos_for_gap(self):
        """Test spawning chaos kernel when no god available."""
        spawner = KernelSpawner()
        
        # Create role with obscure domain that no god handles
        role = RoleSpec(
            domains=["obscure_capability"],
            required_capabilities=["special_task"],
            allow_chaos_spawn=True,
        )
        
        selection = spawner.select_god(role)
        
        # Should spawn chaos kernel
        assert selection.selected_type == "chaos"
        assert selection.chaos_name is not None
        assert selection.requires_pantheon_vote is True
        assert selection.spawn_approved is False
    
    def test_chaos_kernel_naming(self):
        """Test chaos kernel naming pattern."""
        spawner = KernelSpawner()
        
        role = RoleSpec(
            domains=["synthesis"],
            required_capabilities=["test"],
            allow_chaos_spawn=True,
        )
        
        # Block Apollo to force chaos spawn
        spawner.register_spawn("Apollo")
        
        selection = spawner.select_god(role)
        
        if selection.selected_type == "chaos":
            assert selection.chaos_name.startswith("chaos_")
            assert "_" in selection.chaos_name
            # Should be chaos_synthesis_001 or similar
    
    def test_chaos_kernel_counter(self):
        """Test chaos kernel counter increments."""
        spawner = KernelSpawner()
        
        role = RoleSpec(
            domains=["test_domain"],
            required_capabilities=["test"],
            allow_chaos_spawn=True,
        )
        
        # Generate multiple chaos kernels for same domain
        names = []
        for _ in range(3):
            chaos_name = spawner._generate_chaos_kernel_name(role)
            names.append(chaos_name)
        
        # Should have sequential IDs
        assert "chaos_test_domain_001" in names
        assert "chaos_test_domain_002" in names
        assert "chaos_test_domain_003" in names
    
    def test_chaos_spawn_not_allowed(self):
        """Test when chaos spawn not allowed."""
        spawner = KernelSpawner()
        
        role = RoleSpec(
            domains=["obscure_capability"],
            required_capabilities=["special_task"],
            allow_chaos_spawn=False,
        )
        
        selection = spawner.select_god(role)
        
        # Should return no selection
        assert selection.selected_type == "none"
        assert selection.spawn_approved is False


class TestInstanceTracking:
    """Test instance tracking functionality."""
    
    def test_register_spawn(self):
        """Test registering spawn increments counter."""
        spawner = KernelSpawner()
        
        assert spawner.get_active_count("Apollo") == 0
        
        spawner.register_spawn("Apollo")
        assert spawner.get_active_count("Apollo") == 1
    
    def test_register_death(self):
        """Test registering death decrements counter."""
        spawner = KernelSpawner()
        
        spawner.register_spawn("Apollo")
        assert spawner.get_active_count("Apollo") == 1
        
        spawner.register_death("Apollo")
        assert spawner.get_active_count("Apollo") == 0
    
    def test_death_floor_zero(self):
        """Test death counter doesn't go negative."""
        spawner = KernelSpawner()
        
        spawner.register_death("Apollo")
        assert spawner.get_active_count("Apollo") == 0
    
    def test_get_total_chaos_count(self):
        """Test getting total chaos kernel count."""
        spawner = KernelSpawner()
        
        assert spawner.get_total_chaos_count() == 0
        
        role = RoleSpec(domains=["test"], required_capabilities=[])
        spawner._generate_chaos_kernel_name(role)
        spawner._generate_chaos_kernel_name(role)
        
        assert spawner.get_total_chaos_count() == 2


class TestValidation:
    """Test validation methods."""
    
    def test_validate_valid_god(self):
        """Test validating valid god name."""
        spawner = KernelSpawner()
        
        valid, reason = spawner.validate_spawn_request("Apollo")
        assert valid is True
        assert "approved" in reason.lower()
    
    def test_validate_invalid_name(self):
        """Test validating invalid name."""
        spawner = KernelSpawner()
        
        valid, reason = spawner.validate_spawn_request("invalid_name_123")
        assert valid is False
        assert "invalid" in reason.lower()
    
    def test_validate_chaos_kernel(self):
        """Test validating chaos kernel name."""
        spawner = KernelSpawner()
        
        valid, reason = spawner.validate_spawn_request("chaos_synthesis_001")
        assert valid is True


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_select_kernel_for_role(self):
        """Test convenience function for kernel selection."""
        role = RoleSpec(
            domains=["synthesis"],
            required_capabilities=["prediction"],
        )
        
        selection = select_kernel_for_role(role)
        assert selection is not None
        assert selection.selected_type in ["god", "chaos"]
    
    def test_create_role_spec(self):
        """Test convenience function for creating role spec."""
        role = create_role_spec(
            domains=["wisdom"],
            capabilities=["strategy"],
            preferred_god="Athena",
        )
        
        assert role.domains == ["wisdom"]
        assert role.required_capabilities == ["strategy"]
        assert role.preferred_god == "Athena"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
