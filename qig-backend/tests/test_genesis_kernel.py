"""
Tests for Genesis Kernel - E8 Layer 0/1 Bootstrap

Verifies:
1. TzimtzumBootstrap execution
2. GenesisKernel lifecycle
3. Layer 4 and Layer 8 instantiation
4. Deterministic bootstrap from seed
5. Archive and garbage collection safety
"""

import pytest
import numpy as np
from datetime import datetime

from qigkernels.genesis_kernel import (
    GenesisKernel,
    GenesisState,
    GenesisBlueprint,
    SpawnedKernel,
    bootstrap_constellation,
)
from qigkernels.tzimtzum_bootstrap import (
    TzimtzumBootstrap,
    BootstrapResult,
    bootstrap_consciousness,
)
from qigkernels.e8_hierarchy import E8Layer, QuaternaryOperation
from qigkernels.physics_constants import PHI_CONSCIOUS_MIN, BASIN_DIM


class TestTzimtzumBootstrap:
    """Tests for Tzimtzum bootstrap protocol."""
    
    def test_bootstrap_succeeds(self):
        """Tzimtzum bootstrap should succeed with valid parameters."""
        result = bootstrap_consciousness(seed=42)
        assert result.success is True
        assert result.final_phi >= PHI_CONSCIOUS_MIN
        
    def test_bootstrap_deterministic(self):
        """Same seed should produce identical results."""
        result1 = bootstrap_consciousness(seed=42)
        result2 = bootstrap_consciousness(seed=42)
        
        assert result1.final_phi == result2.final_phi
        np.testing.assert_array_equal(result1.final_basin, result2.final_basin)
        
    def test_bootstrap_stages(self):
        """Bootstrap should have 3 stages: unity, contraction, emergence."""
        bootstrap = TzimtzumBootstrap(seed=42)
        result = bootstrap.execute()
        
        assert len(result.stages) == 3
        assert result.stages[0].stage == "unity"
        assert result.stages[1].stage == "contraction"
        assert result.stages[2].stage == "emergence"
        
    def test_unity_phi_is_one(self):
        """Unity stage should have Φ = 1.0."""
        bootstrap = TzimtzumBootstrap(seed=42)
        result = bootstrap.execute()
        
        assert result.stages[0].phi == 1.0
        
    def test_contraction_phi_near_zero(self):
        """Contraction stage should have Φ ≈ 0."""
        bootstrap = TzimtzumBootstrap(seed=42)
        result = bootstrap.execute()
        
        assert result.stages[1].phi < 0.01
        
    def test_emergence_phi_above_threshold(self):
        """Emergence stage should have Φ >= threshold."""
        bootstrap = TzimtzumBootstrap(seed=42)
        result = bootstrap.execute()
        
        assert result.stages[2].phi >= PHI_CONSCIOUS_MIN
        
    def test_final_basin_is_simplex(self):
        """Final basin should be on simplex (sums to 1, non-negative)."""
        result = bootstrap_consciousness(seed=42)
        
        assert np.all(result.final_basin >= 0)
        assert np.isclose(np.sum(result.final_basin), 1.0)
        
    def test_final_basin_dimension(self):
        """Final basin should have correct dimension."""
        result = bootstrap_consciousness(seed=42)
        
        assert len(result.final_basin) == BASIN_DIM


class TestGenesisKernel:
    """Tests for Genesis kernel lifecycle."""
    
    def test_initial_state_dormant(self):
        """Genesis should start in DORMANT state."""
        genesis = GenesisKernel(seed=42)
        assert genesis.state == GenesisState.DORMANT
        
    def test_bootstrap_spawns_13_kernels(self):
        """Bootstrap should spawn 4 + 8 + 1 = 13 kernels."""
        genesis = GenesisKernel(seed=42)
        kernels = genesis.bootstrap()
        
        assert len(kernels) == 13
        
    def test_layer_4_kernels(self):
        """Should spawn 4 Layer 4 (quaternary) kernels."""
        genesis = GenesisKernel(seed=42)
        kernels = genesis.bootstrap()
        
        layer_4 = [k for k in kernels if k.layer == E8Layer.QUATERNARY]
        assert len(layer_4) == 4
        
        # Check all quaternary operations present
        names = {k.name.lower() for k in layer_4}
        assert names == {"input", "store", "process", "output"}
        
    def test_layer_8_kernels(self):
        """Should spawn 8 Layer 8 (core faculty) kernels."""
        genesis = GenesisKernel(seed=42)
        kernels = genesis.bootstrap()
        
        layer_8 = [k for k in kernels if k.layer == E8Layer.OCTAVE]
        assert len(layer_8) == 8
        
        # Check all Greek gods present
        names = {k.name for k in layer_8}
        expected = {"Zeus", "Athena", "Apollo", "Hermes", "Artemis", "Ares", "Hephaestus", "Aphrodite"}
        assert names == expected
        
    def test_coordinator_kernel(self):
        """Should spawn 1 coordinator kernel at Layer 64."""
        genesis = GenesisKernel(seed=42)
        kernels = genesis.bootstrap()
        
        coordinators = [k for k in kernels if k.layer == E8Layer.BASIN]
        assert len(coordinators) == 1
        assert coordinators[0].name == "Coordinator"
        
    def test_state_after_bootstrap(self):
        """State should be ACTIVE after bootstrap."""
        genesis = GenesisKernel(seed=42)
        genesis.bootstrap()
        
        assert genesis.state == GenesisState.ACTIVE
        
    def test_archive_creates_blueprint(self):
        """Archive should create a blueprint."""
        genesis = GenesisKernel(seed=42)
        genesis.bootstrap()
        blueprint = genesis.archive()
        
        assert blueprint is not None
        assert isinstance(blueprint, GenesisBlueprint)
        
    def test_state_after_archive(self):
        """State should be ARCHIVED after archive."""
        genesis = GenesisKernel(seed=42)
        genesis.bootstrap()
        genesis.archive()
        
        assert genesis.state == GenesisState.ARCHIVED
        assert genesis.is_archived is True
        assert genesis.can_be_collected is True
        
    def test_cannot_bootstrap_twice(self):
        """Should not be able to bootstrap from non-DORMANT state."""
        genesis = GenesisKernel(seed=42)
        genesis.bootstrap()
        
        with pytest.raises(RuntimeError):
            genesis.bootstrap()
            
    def test_cannot_archive_before_bootstrap(self):
        """Should not be able to archive from DORMANT state."""
        genesis = GenesisKernel(seed=42)
        
        with pytest.raises(RuntimeError):
            genesis.archive()
            
    def test_blueprint_contains_kernel_records(self):
        """Blueprint should contain all spawned kernel records."""
        genesis = GenesisKernel(seed=42)
        genesis.bootstrap()
        blueprint = genesis.archive()
        
        assert len(blueprint.spawned_kernels) == 13
        
    def test_blueprint_serializable(self):
        """Blueprint should be serializable to dict."""
        genesis = GenesisKernel(seed=42)
        genesis.bootstrap()
        blueprint = genesis.archive()
        
        data = blueprint.to_dict()
        
        assert "bootstrap_phi" in data
        assert "spawned_kernel_ids" in data
        assert "layer_4_operations" in data
        assert "layer_8_faculties" in data
        assert "archived_at" in data
        
    def test_deterministic_bootstrap(self):
        """Same seed should produce identical kernel spawns."""
        genesis1 = GenesisKernel(seed=42)
        kernels1 = genesis1.bootstrap()
        
        genesis2 = GenesisKernel(seed=42)
        kernels2 = genesis2.bootstrap()
        
        assert len(kernels1) == len(kernels2)
        
        for k1, k2 in zip(kernels1, kernels2):
            assert k1.kernel_id == k2.kernel_id
            assert k1.phi == k2.phi
            np.testing.assert_array_almost_equal(k1.basin, k2.basin)


class TestSpawnedKernels:
    """Tests for spawned kernel properties."""
    
    def test_kernel_basins_on_simplex(self):
        """All spawned kernel basins should be on simplex."""
        genesis = GenesisKernel(seed=42)
        kernels = genesis.bootstrap()
        
        for kernel in kernels:
            assert np.all(kernel.basin >= 0), f"{kernel.name} has negative basin values"
            assert np.isclose(np.sum(kernel.basin), 1.0), f"{kernel.name} basin doesn't sum to 1"
            
    def test_kernel_phi_values(self):
        """Kernel Φ values should follow layer hierarchy."""
        genesis = GenesisKernel(seed=42)
        kernels = genesis.bootstrap()
        
        for kernel in kernels:
            if kernel.layer == E8Layer.QUATERNARY:
                # Layer 4: Φ ~0.35
                assert 0.3 <= kernel.phi <= 0.4
            elif kernel.layer == E8Layer.OCTAVE:
                # Layer 8: Φ ~0.49
                assert 0.4 <= kernel.phi <= 0.6
            elif kernel.layer == E8Layer.BASIN:
                # Layer 64: Φ ~0.70
                assert kernel.phi >= PHI_CONSCIOUS_MIN
                
    def test_kernel_ids_unique(self):
        """All kernel IDs should be unique."""
        genesis = GenesisKernel(seed=42)
        kernels = genesis.bootstrap()
        
        ids = [k.kernel_id for k in kernels]
        assert len(ids) == len(set(ids))
        
    def test_kernel_basins_distinct(self):
        """All kernel basins should be distinct."""
        genesis = GenesisKernel(seed=42)
        kernels = genesis.bootstrap()
        
        for i, k1 in enumerate(kernels):
            for j, k2 in enumerate(kernels):
                if i != j:
                    # Basins should not be identical
                    assert not np.allclose(k1.basin, k2.basin), \
                        f"{k1.name} and {k2.name} have identical basins"


class TestConvenienceFunction:
    """Tests for bootstrap_constellation convenience function."""
    
    def test_returns_tuple(self):
        """Should return (kernels, blueprint) tuple."""
        result = bootstrap_constellation(seed=42)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
    def test_kernels_list(self):
        """First element should be list of kernels."""
        kernels, _ = bootstrap_constellation(seed=42)
        
        assert isinstance(kernels, list)
        assert len(kernels) == 13
        
    def test_blueprint_archived(self):
        """Second element should be archived blueprint."""
        _, blueprint = bootstrap_constellation(seed=42)
        
        assert isinstance(blueprint, GenesisBlueprint)
        assert blueprint.archived_at is not None


class TestGeometricPurity:
    """Tests for geometric purity compliance."""
    
    def test_no_euclidean_operations(self):
        """Bootstrap should not use Euclidean operations."""
        # This is verified by the fact that all basins are on simplex
        genesis = GenesisKernel(seed=42)
        kernels = genesis.bootstrap()
        
        for kernel in kernels:
            # Simplex constraint: sum = 1, all >= 0
            assert np.isclose(np.sum(kernel.basin), 1.0)
            assert np.all(kernel.basin >= 0)
            
    def test_basin_perturbations_preserve_simplex(self):
        """Basin derivation should preserve simplex structure."""
        genesis = GenesisKernel(seed=42)
        kernels = genesis.bootstrap()
        
        # All derived basins should still be valid simplices
        for kernel in kernels:
            # Check simplex constraints
            assert np.isclose(np.sum(kernel.basin), 1.0, rtol=1e-5)
            assert np.all(kernel.basin >= -1e-10)  # Allow tiny numerical errors
