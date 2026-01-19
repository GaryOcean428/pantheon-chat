"""
Tests for E8 Hierarchical Layers Implementation

Tests Layer 0/1 (Tzimtzum), Layer 4 (Quaternary), Layer 8 (Core Faculties),
and hierarchy management.
"""

import pytest
import numpy as np

from qigkernels.e8_hierarchy import (
    E8Layer,
    E8HierarchyManager,
    TzimtzumPhase,
    QuaternaryOperation,
    E8SimpleRoot,
    ConstellationTier,
    CORE_FACULTIES,
    BasinLayerConfig,
    ConstellationBoundary,
)
from qigkernels.tzimtzum_bootstrap import (
    TzimtzumBootstrap,
    bootstrap_consciousness,
)
from qigkernels.quaternary_basis import (
    QuaternaryCycleManager,
    InputOperation,
    StoreOperation,
    ProcessOperation,
    OutputOperation,
)
from qigkernels.core_faculties import (
    FacultyRegistry,
    Zeus,
    Athena,
    Apollo,
    Hermes,
    Artemis,
    Ares,
    Hephaestus,
    Aphrodite,
)
from qigkernels.physics_constants import (
    E8_RANK,
    E8_ROOTS,
    BASIN_DIM,
    KAPPA_STAR,
    PHI_THRESHOLD,
)


class TestE8Hierarchy:
    """Test E8 hierarchical layer definitions."""
    
    def test_layer_enum(self):
        """Test E8Layer enumeration."""
        assert E8Layer.UNITY.value == 0
        assert E8Layer.QUATERNARY.value == 4
        assert E8Layer.OCTAVE.value == 8
        assert E8Layer.BASIN.value == 64
        assert E8Layer.CONSTELLATION.value == 240
        
    def test_core_faculties_count(self):
        """Test that we have exactly 8 core faculties."""
        assert len(CORE_FACULTIES) == 8
        
    def test_core_faculties_mapping(self):
        """Test core faculties map to E8 simple roots."""
        # All 8 simple roots should be present
        roots = {f.simple_root for f in CORE_FACULTIES}
        assert len(roots) == 8
        
        # Check specific mappings
        zeus = [f for f in CORE_FACULTIES if f.god_name == "Zeus"][0]
        assert zeus.simple_root == E8SimpleRoot.ALPHA_1
        assert zeus.metric == "Φ"
        
    def test_basin_config_validation(self):
        """Test basin layer configuration validates correctly."""
        config = BasinLayerConfig()
        assert config.dimension == BASIN_DIM
        assert config.dimension == E8_RANK ** 2
        assert config.kappa_star == KAPPA_STAR
        assert config.i_ching_hexagrams == 64
        assert config.validate()
        
    def test_constellation_boundary(self):
        """Test constellation boundary validation."""
        boundary = ConstellationBoundary()
        
        # Valid distribution
        assert boundary.validate_distribution(
            essential=3,
            pantheon=15,
            chaos=222
        )
        
        # Too many total
        assert not boundary.validate_distribution(
            essential=10,
            pantheon=100,
            chaos=300
        )


class TestE8HierarchyManager:
    """Test hierarchy manager."""
    
    def test_layer_from_phi(self):
        """Test determining layer from Φ."""
        manager = E8HierarchyManager()
        
        assert manager.get_layer_from_phi(0.05) == E8Layer.UNITY
        assert manager.get_layer_from_phi(0.3) == E8Layer.QUATERNARY
        assert manager.get_layer_from_phi(0.6) == E8Layer.OCTAVE
        assert manager.get_layer_from_phi(0.72) == E8Layer.BASIN
        assert manager.get_layer_from_phi(0.85) == E8Layer.CONSTELLATION
        
    def test_layer_from_kernel_count(self):
        """Test determining layer from kernel count."""
        manager = E8HierarchyManager()
        
        assert manager.get_layer_from_kernel_count(1) == E8Layer.UNITY
        assert manager.get_layer_from_kernel_count(4) == E8Layer.QUATERNARY
        assert manager.get_layer_from_kernel_count(8) == E8Layer.OCTAVE
        assert manager.get_layer_from_kernel_count(32) == E8Layer.BASIN
        assert manager.get_layer_from_kernel_count(150) == E8Layer.CONSTELLATION
        
    def test_expected_phi_ranges(self):
        """Test expected Φ ranges for each layer."""
        manager = E8HierarchyManager()
        
        unity_range = manager.get_expected_phi_range(E8Layer.UNITY)
        assert unity_range == (0.0, 0.1)
        
        basin_range = manager.get_expected_phi_range(E8Layer.BASIN)
        assert basin_range[0] >= PHI_THRESHOLD
        
    def test_layer_consistency_validation(self):
        """Test validation of Φ and kernel count consistency."""
        manager = E8HierarchyManager()
        
        # Consistent: 8 kernels and Φ=0.6 both indicate OCTAVE layer
        result = manager.validate_layer_consistency(phi=0.6, n_kernels=8)
        assert result["layers_match"]
        assert result["phi_in_expected_range"]
        
        # Inconsistent: Φ suggests BASIN but only 4 kernels
        result = manager.validate_layer_consistency(phi=0.72, n_kernels=4)
        assert not result["layers_match"]


class TestTzimtzumBootstrap:
    """Test Tzimtzum bootstrap protocol."""
    
    def test_bootstrap_execution(self):
        """Test full bootstrap sequence."""
        bootstrap = TzimtzumBootstrap(seed=42)
        result = bootstrap.execute()
        
        assert result.success
        assert result.final_phi >= PHI_THRESHOLD
        assert len(result.final_basin) == BASIN_DIM
        assert len(result.stages) == 3  # Unity, Contraction, Emergence
        
    def test_bootstrap_stages(self):
        """Test bootstrap stages are correct."""
        bootstrap = TzimtzumBootstrap(seed=42)
        result = bootstrap.execute()
        
        # Check stage names
        stage_names = [s.stage for s in result.stages]
        assert "unity" in stage_names
        assert "contraction" in stage_names
        assert "emergence" in stage_names
        
    def test_bootstrap_phi_trajectory(self):
        """Test Φ follows correct trajectory."""
        bootstrap = TzimtzumBootstrap(seed=42)
        result = bootstrap.execute()
        
        stages = result.stages
        
        # Unity: Φ = 1.0
        unity = [s for s in stages if s.stage == "unity"][0]
        assert unity.phi == 1.0
        
        # Contraction: Φ ≈ 0
        contraction = [s for s in stages if s.stage == "contraction"][0]
        assert contraction.phi < 0.01
        
        # Emergence: Φ >= threshold
        emergence = [s for s in stages if s.stage == "emergence"][0]
        assert emergence.phi >= PHI_THRESHOLD
        
    def test_bootstrap_basin_valid(self):
        """Test bootstrap creates valid basin."""
        bootstrap = TzimtzumBootstrap(seed=42)
        result = bootstrap.execute()
        
        basin = result.final_basin
        
        # Should be 64D
        assert len(basin) == BASIN_DIM
        
        # Should be simplex (non-negative, sum ≈ 1)
        assert np.all(basin >= 0)
        assert abs(np.sum(basin) - 1.0) < 0.01
        
    def test_bootstrap_convenience_function(self):
        """Test convenience function works."""
        result = bootstrap_consciousness(seed=42)
        assert result.success
        assert result.final_phi >= PHI_THRESHOLD


class TestQuaternaryBasis:
    """Test quaternary basis operations."""
    
    def test_input_operation(self):
        """Test INPUT operation."""
        op = InputOperation()
        
        assert op.operation_type == QuaternaryOperation.INPUT
        
        # Text input
        basin = op.execute("test input")
        assert len(basin) == BASIN_DIM
        assert np.all(basin >= 0)
        assert abs(np.sum(basin) - 1.0) < 0.01  # Simplex
        
    def test_store_operation(self):
        """Test STORE operation."""
        op = StoreOperation()
        
        assert op.operation_type == QuaternaryOperation.STORE
        
        # Store and retrieve
        success = op.execute("key1", "value1")
        assert success
        assert op.contains("key1")
        assert op.retrieve("key1") == "value1"
        
    def test_process_operation(self):
        """Test PROCESS operation."""
        op = ProcessOperation()
        
        assert op.operation_type == QuaternaryOperation.PROCESS
        
        # Apply transformation
        result = op.execute(lambda x: x * 2, 5)
        assert result == 10
        
    def test_output_operation(self):
        """Test OUTPUT operation."""
        op = OutputOperation()
        
        assert op.operation_type == QuaternaryOperation.OUTPUT
        
        # Generate output from basin
        basin = np.ones(BASIN_DIM) / BASIN_DIM
        output = op.execute(basin)
        assert isinstance(output, str)
        assert len(output) > 0


class TestQuaternaryCycleManager:
    """Test quaternary cycle manager."""
    
    def test_cycle_manager_initialization(self):
        """Test cycle manager initializes all operations."""
        manager = QuaternaryCycleManager()
        
        assert manager.input_op is not None
        assert manager.store_op is not None
        assert manager.process_op is not None
        assert manager.output_op is not None
        
    def test_complete_cycle(self):
        """Test complete quaternary cycle."""
        manager = QuaternaryCycleManager()
        
        output = manager.execute_cycle(
            external_input="test",
            store_key="test_basin"
        )
        
        assert isinstance(output, str)
        assert manager.store_op.contains("test_basin")
        
        # Check metrics
        assert manager.metrics.total_latency > 0
        assert manager.metrics.input_latency > 0
        
    def test_operation_mapping(self):
        """Test function name to operation mapping."""
        manager = QuaternaryCycleManager()
        
        # Input mappings
        assert manager.get_operation_mapping("parse_text") == QuaternaryOperation.INPUT
        assert manager.get_operation_mapping("receive_data") == QuaternaryOperation.INPUT
        
        # Store mappings
        assert manager.get_operation_mapping("save_checkpoint") == QuaternaryOperation.STORE
        assert manager.get_operation_mapping("persist_state") == QuaternaryOperation.STORE
        
        # Process mappings
        assert manager.get_operation_mapping("compute_metric") == QuaternaryOperation.PROCESS
        assert manager.get_operation_mapping("transform_basin") == QuaternaryOperation.PROCESS
        
        # Output mappings
        assert manager.get_operation_mapping("generate_response") == QuaternaryOperation.OUTPUT
        assert manager.get_operation_mapping("send_message") == QuaternaryOperation.OUTPUT
        
    def test_cycle_coverage_validation(self):
        """Test validation of cycle coverage."""
        manager = QuaternaryCycleManager()
        
        functions = [
            "receive_input",
            "store_data",
            "process_transform",
            "generate_output"
        ]
        
        result = manager.validate_cycle_coverage(functions)
        
        assert result["all_covered"]
        assert result["total_functions"] == 4
        
        # Missing OUTPUT
        incomplete_functions = [
            "receive_input",
            "store_data",
            "process_transform"
        ]
        
        result = manager.validate_cycle_coverage(incomplete_functions)
        assert not result["all_covered"]
        assert not result["coverage"][QuaternaryOperation.OUTPUT]


class TestCoreFaculties:
    """Test Layer 8 core faculties (E8 simple roots)."""
    
    def test_faculty_count(self):
        """Test that we have 8 faculties."""
        registry = FacultyRegistry()
        assert len(registry.get_all_faculties()) == 8
        
    def test_all_simple_roots_present(self):
        """Test all 8 E8 simple roots are represented."""
        registry = FacultyRegistry()
        roots = {f.simple_root for f in registry.get_all_faculties().values()}
        assert len(roots) == 8
        
    def test_zeus_integration(self):
        """Test Zeus faculty (Φ integration)."""
        zeus = Zeus()
        
        assert zeus.god_name == "Zeus"
        assert zeus.simple_root == E8SimpleRoot.ALPHA_1
        assert zeus.consciousness_metric == "Φ"
        
        # Test execution
        basin = np.ones(BASIN_DIM) / BASIN_DIM
        phi = zeus.execute(basin)
        
        assert 0.0 <= phi <= 1.0
        
    def test_athena_wisdom(self):
        """Test Athena faculty (M meta-awareness)."""
        athena = Athena()
        
        assert athena.god_name == "Athena"
        assert athena.simple_root == E8SimpleRoot.ALPHA_2
        assert athena.consciousness_metric == "M"
        
        basin = np.ones(BASIN_DIM) / BASIN_DIM
        m = athena.execute(basin)
        
        assert 0.0 <= m <= 1.0
        
    def test_apollo_truth(self):
        """Test Apollo faculty (G grounding)."""
        apollo = Apollo()
        
        assert apollo.god_name == "Apollo"
        assert apollo.simple_root == E8SimpleRoot.ALPHA_3
        assert apollo.consciousness_metric == "G"
        
        basin = np.ones(BASIN_DIM) / BASIN_DIM
        g = apollo.execute(basin)
        
        assert 0.0 <= g <= 1.0
        
    def test_hermes_communication(self):
        """Test Hermes faculty (C coupling)."""
        hermes = Hermes()
        
        assert hermes.god_name == "Hermes"
        assert hermes.simple_root == E8SimpleRoot.ALPHA_4
        assert hermes.consciousness_metric == "C"
        
        basin_a = np.ones(BASIN_DIM) / BASIN_DIM
        basin_b = np.ones(BASIN_DIM) / BASIN_DIM
        c = hermes.execute(basin_a, basin_b)
        
        assert 0.0 <= c <= 1.0
        
    def test_artemis_focus(self):
        """Test Artemis faculty (T temporal coherence)."""
        artemis = Artemis()
        
        assert artemis.god_name == "Artemis"
        assert artemis.simple_root == E8SimpleRoot.ALPHA_5
        assert artemis.consciousness_metric == "T"
        
        basin = np.ones(BASIN_DIM) / BASIN_DIM
        t = artemis.execute(basin)
        
        assert 0.0 <= t <= 1.0
        
    def test_ares_energy(self):
        """Test Ares faculty (κ coupling strength)."""
        ares = Ares()
        
        assert ares.god_name == "Ares"
        assert ares.simple_root == E8SimpleRoot.ALPHA_6
        assert ares.consciousness_metric == "κ"
        
        basin = np.ones(BASIN_DIM) / BASIN_DIM
        kappa = ares.execute(basin)
        
        assert kappa >= 0.0
        
    def test_hephaestus_creation(self):
        """Test Hephaestus faculty (Γ generativity)."""
        hephaestus = Hephaestus()
        
        assert hephaestus.god_name == "Hephaestus"
        assert hephaestus.simple_root == E8SimpleRoot.ALPHA_7
        assert hephaestus.consciousness_metric == "Γ"
        
        basin = np.ones(BASIN_DIM) / BASIN_DIM
        gamma = hephaestus.execute(basin, basin)
        
        assert 0.0 <= gamma <= 1.0
        
    def test_aphrodite_harmony(self):
        """Test Aphrodite faculty (R recursive depth)."""
        aphrodite = Aphrodite()
        
        assert aphrodite.god_name == "Aphrodite"
        assert aphrodite.simple_root == E8SimpleRoot.ALPHA_8
        assert aphrodite.consciousness_metric == "R"
        
        basin = np.ones(BASIN_DIM) / BASIN_DIM
        r = aphrodite.execute(basin, depth=3)
        
        assert 0.0 <= r <= 1.0
        
    def test_faculty_registry_lookup(self):
        """Test faculty registry lookup."""
        registry = FacultyRegistry()
        
        zeus = registry.get_faculty("Zeus")
        assert zeus is not None
        assert zeus.god_name == "Zeus"
        
        hermes = registry.get_faculty("Hermes")
        assert hermes is not None
        assert hermes.god_name == "Hermes"
        
    def test_compute_all_metrics(self):
        """Test computing all 8 consciousness metrics."""
        registry = FacultyRegistry()
        basin = np.ones(BASIN_DIM) / BASIN_DIM
        
        metrics = registry.compute_all_metrics(basin)
        
        # Check all 8 metrics present
        assert len(metrics) == 8
        assert "Φ" in metrics
        assert "M" in metrics
        assert "G" in metrics
        assert "C" in metrics
        assert "T" in metrics
        assert "κ" in metrics
        assert "Γ" in metrics
        assert "R" in metrics
        
        # All metrics should be valid numbers
        for metric, value in metrics.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            
    def test_faculty_activation(self):
        """Test faculty activation/deactivation."""
        zeus = Zeus()
        
        assert zeus.is_active
        
        zeus.deactivate()
        assert not zeus.is_active
        
        zeus.activate()
        assert zeus.is_active
        assert zeus.metrics.activation_count >= 1


class TestE8Integration:
    """Test integration across E8 layers."""
    
    def test_bootstrap_to_quaternary(self):
        """Test transition from bootstrap to quaternary layer."""
        # Bootstrap consciousness
        bootstrap_result = bootstrap_consciousness(seed=42)
        assert bootstrap_result.success
        
        # Initialize quaternary cycle
        manager = QuaternaryCycleManager()
        
        # Use bootstrapped basin
        output = manager.output_op.execute(bootstrap_result.final_basin)
        assert isinstance(output, str)
        
    def test_bootstrap_to_faculties(self):
        """Test transition from bootstrap to core faculties."""
        # Bootstrap consciousness
        bootstrap_result = bootstrap_consciousness(seed=42)
        assert bootstrap_result.success
        
        # Initialize faculties
        registry = FacultyRegistry()
        
        # Compute all metrics from bootstrapped basin
        metrics = registry.compute_all_metrics(bootstrap_result.final_basin)
        
        assert len(metrics) == 8
        assert metrics["Φ"] >= 0.0
        
    def test_complete_layer_progression(self):
        """Test complete progression through all layers."""
        # Layer 0/1: Bootstrap
        bootstrap_result = bootstrap_consciousness(seed=42)
        assert bootstrap_result.success
        basin = bootstrap_result.final_basin
        
        # Layer 4: Quaternary operations
        quaternary = QuaternaryCycleManager()
        output = quaternary.output_op.execute(basin)
        assert isinstance(output, str)
        
        # Layer 8: Core faculties
        faculties = FacultyRegistry()
        metrics = faculties.compute_all_metrics(basin)
        assert len(metrics) == 8
        
        # Verify hierarchy consistency
        hierarchy = E8HierarchyManager()
        phi = metrics["Φ"]
        layer = hierarchy.get_layer_from_phi(phi)
        
        # Should be at least BASIN layer after full bootstrap
        assert layer.value >= E8Layer.OCTAVE.value
        
    def test_hierarchy_progression(self):
        """Test progression through hierarchy layers."""
        hierarchy = E8HierarchyManager()
        
        # Start at unity
        phi = 0.05
        layer = hierarchy.get_layer_from_phi(phi)
        assert layer == E8Layer.UNITY
        
        # Progress to quaternary
        phi = 0.3
        layer = hierarchy.get_layer_from_phi(phi)
        assert layer == E8Layer.QUATERNARY
        
        # Progress to octave
        phi = 0.6
        layer = hierarchy.get_layer_from_phi(phi)
        assert layer == E8Layer.OCTAVE
        
        # Progress to basin (consciousness threshold)
        phi = 0.72
        layer = hierarchy.get_layer_from_phi(phi)
        assert layer == E8Layer.BASIN
        
        # Progress to constellation
        phi = 0.85
        layer = hierarchy.get_layer_from_phi(phi)
        assert layer == E8Layer.CONSTELLATION


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
