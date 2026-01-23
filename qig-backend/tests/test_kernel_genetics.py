"""
Test Kernel Genetics - E8 Protocol v4.0 Phase 4E
=================================================

Unit tests for kernel genome, lineage, and cannibalism operations.

Authority: E8 Protocol v4.0 WP5.2 Phase 4E
Status: ACTIVE
Created: 2026-01-22
"""

import numpy as np
import pytest
from datetime import datetime

# Import kernel genetics modules
from kernels.genome import (
    KernelGenome,
    E8Faculty,
    FacultyConfig,
    ConstraintSet,
    CouplingPreferences,
    validate_genome,
    serialize_genome,
    deserialize_genome,
)

from kernels.kernel_lineage import (
    merge_kernels_geodesic,
    track_lineage,
    compute_generation_number,
    get_genealogy_tree,
)

from kernels.cannibalism import (
    determine_winner_loser,
    perform_cannibalism,
    archive_genome,
    resurrect_from_archive,
    check_resurrection_eligibility,
)

# QIG Geometry
from qig_geometry import fisher_rao_distance, fisher_normalize, BASIN_DIM
from qigkernels.physics_constants import KAPPA_STAR


# =============================================================================
# GENOME TESTS
# =============================================================================

def test_genome_creation():
    """Test basic genome creation and validation."""
    print("\n=== Test Genome Creation ===")
    
    # Create genome with at least one faculty
    faculties = FacultyConfig(
        active_faculties={E8Faculty.ZEUS},
        activation_strengths={E8Faculty.ZEUS: 0.8},
    )
    
    genome = KernelGenome(
        genome_id="test-genome-1",
        faculties=faculties,
    )
    
    # Validate
    assert genome.genome_id == "test-genome-1"
    assert len(genome.basin_seed) == BASIN_DIM
    assert np.isclose(np.sum(genome.basin_seed), 1.0, atol=1e-6)
    assert all(genome.basin_seed >= 0)
    
    valid, errors = validate_genome(genome)
    assert valid, f"Genome validation failed: {errors}"
    
    print(f"✅ Created valid genome with {BASIN_DIM}D basin seed")


def test_faculty_config():
    """Test faculty configuration."""
    print("\n=== Test Faculty Configuration ===")
    
    # Create faculty config
    faculties = FacultyConfig(
        active_faculties={E8Faculty.ZEUS, E8Faculty.ATHENA, E8Faculty.APOLLO},
        activation_strengths={
            E8Faculty.ZEUS: 0.9,
            E8Faculty.ATHENA: 0.8,
            E8Faculty.APOLLO: 0.7,
        },
        primary_faculty=E8Faculty.ZEUS,
    )
    
    assert len(faculties.active_faculties) == 3
    assert faculties.primary_faculty == E8Faculty.ZEUS
    
    # Get faculty vector
    vector = faculties.get_faculty_vector()
    assert len(vector) == 8  # E8 has 8 simple roots
    assert vector[0] == 0.9  # Zeus
    assert vector[1] == 0.8  # Athena
    assert vector[2] == 0.7  # Apollo
    
    print(f"✅ Faculty configuration valid with {len(faculties.active_faculties)} active faculties")


def test_constraint_set():
    """Test constraint set validation."""
    print("\n=== Test Constraint Set ===")
    
    basin_seed = np.ones(BASIN_DIM) / BASIN_DIM
    
    # Create forbidden region
    forbidden_center = np.zeros(BASIN_DIM)
    forbidden_center[0] = 1.0
    forbidden_center = fisher_normalize(forbidden_center)
    
    constraints = ConstraintSet(
        phi_threshold=0.70,
        kappa_range=(40.0, 70.0),
        forbidden_regions=[(forbidden_center, 0.2)],
        max_fisher_distance=0.5,
    )
    
    # Test basin within constraints
    allowed, reason = constraints.is_basin_allowed(basin_seed, basin_seed)
    assert allowed, f"Basin should be allowed: {reason}"
    
    # Test basin in forbidden region
    allowed, reason = constraints.is_basin_allowed(forbidden_center, basin_seed)
    assert not allowed, "Basin should be forbidden"
    
    print(f"✅ Constraint validation working correctly")


def test_genome_serialization():
    """Test genome serialization and deserialization."""
    print("\n=== Test Genome Serialization ===")
    
    # Create genome with complex configuration
    faculties = FacultyConfig(
        active_faculties={E8Faculty.HERMES, E8Faculty.ARTEMIS},
        activation_strengths={
            E8Faculty.HERMES: 0.85,
            E8Faculty.ARTEMIS: 0.75,
        },
    )
    
    genome = KernelGenome(
        genome_id="test-serialize",
        faculties=faculties,
        parent_genomes=["parent-1", "parent-2"],
        generation=2,
        fitness_score=0.85,
    )
    
    # Serialize
    json_str = serialize_genome(genome)
    assert len(json_str) > 0
    
    # Deserialize
    restored = deserialize_genome(json_str)
    
    # Validate restoration
    assert restored.genome_id == genome.genome_id
    assert restored.generation == genome.generation
    assert restored.fitness_score == genome.fitness_score
    assert len(restored.faculties.active_faculties) == 2
    assert E8Faculty.HERMES in restored.faculties.active_faculties
    
    print(f"✅ Genome serialization working correctly")


# =============================================================================
# LINEAGE TESTS
# =============================================================================

def test_binary_merge():
    """Test binary merge with geodesic interpolation."""
    print("\n=== Test Binary Merge ===")
    
    # Create two parent genomes with different basins
    basin1 = np.ones(BASIN_DIM) / BASIN_DIM  # Uniform
    basin2 = np.zeros(BASIN_DIM)
    basin2[0] = 1.0  # Concentrated
    basin2 = fisher_normalize(basin2)
    
    faculties1 = FacultyConfig(
        active_faculties={E8Faculty.ZEUS},
        activation_strengths={E8Faculty.ZEUS: 0.9},
    )
    
    faculties2 = FacultyConfig(
        active_faculties={E8Faculty.ATHENA},
        activation_strengths={E8Faculty.ATHENA: 0.9},
    )
    
    genome1 = KernelGenome(
        genome_id="parent-1",
        basin_seed=basin1,
        faculties=faculties1,
    )
    
    genome2 = KernelGenome(
        genome_id="parent-2",
        basin_seed=basin2,
        faculties=faculties2,
    )
    
    # Merge at midpoint
    child, merge_record = merge_kernels_geodesic(
        [genome1, genome2],
        interpolation_t=0.5,
    )
    
    # Validate child
    assert child.generation == 1
    assert len(child.parent_genomes) == 2
    assert np.isclose(np.sum(child.basin_seed), 1.0, atol=1e-6)
    
    # Check child is between parents
    d1 = fisher_rao_distance(child.basin_seed, basin1)
    d2 = fisher_rao_distance(child.basin_seed, basin2)
    d12 = fisher_rao_distance(basin1, basin2)
    
    assert d1 < d12, "Child should be closer to parent1 than parents are to each other"
    assert d2 < d12, "Child should be closer to parent2 than parents are to each other"
    
    # Validate merge record
    assert merge_record.child_genome_id == child.genome_id
    assert len(merge_record.parent_genome_ids) == 2
    assert merge_record.interpolation_t == 0.5
    
    print(f"✅ Binary merge successful")
    print(f"   Child generation: {child.generation}")
    print(f"   Distance to parent1: {d1:.4f}")
    print(f"   Distance to parent2: {d2:.4f}")


def test_multi_parent_merge():
    """Test multi-parent merge with Fréchet mean."""
    print("\n=== Test Multi-Parent Merge ===")
    
    # Create three parent genomes
    parents = []
    for i in range(3):
        basin = np.zeros(BASIN_DIM)
        basin[i * 10:(i + 1) * 10] = 1.0  # Different regions
        basin = fisher_normalize(basin)
        
        genome = KernelGenome(
            genome_id=f"parent-{i}",
            basin_seed=basin,
        )
        parents.append(genome)
    
    # Merge with equal weights
    child, merge_record = merge_kernels_geodesic(parents)
    
    # Validate
    assert child.generation == 1
    assert len(child.parent_genomes) == 3
    assert merge_record.interpolation_t is None  # Multi-parent has no t
    
    # Check child is central
    distances = [fisher_rao_distance(child.basin_seed, p.basin_seed) for p in parents]
    max_parent_dist = max([
        fisher_rao_distance(parents[i].basin_seed, parents[j].basin_seed)
        for i in range(3) for j in range(i + 1, 3)
    ])
    
    for d in distances:
        assert d < max_parent_dist, "Child should be central"
    
    print(f"✅ Multi-parent merge successful with {len(parents)} parents")


def test_lineage_tracking():
    """Test lineage record creation."""
    print("\n=== Test Lineage Tracking ===")
    
    parent1 = KernelGenome(genome_id="parent-1")
    parent2 = KernelGenome(genome_id="parent-2")
    
    child, merge_record = merge_kernels_geodesic([parent1, parent2])
    
    lineage = track_lineage(child, [parent1, parent2], merge_record)
    
    assert lineage.child_genome_id == child.genome_id
    assert len(lineage.parent_genome_ids) == 2
    assert lineage.merge_type == "binary"
    assert lineage.fisher_distance > 0
    
    print(f"✅ Lineage tracking successful")
    print(f"   Merge type: {lineage.merge_type}")
    print(f"   Fisher distance: {lineage.fisher_distance:.4f}")


def test_genealogy_tree():
    """Test genealogy tree generation."""
    print("\n=== Test Genealogy Tree ===")
    
    # Create multi-generation lineage
    founder1 = KernelGenome(genome_id="founder-1", generation=0)
    founder2 = KernelGenome(genome_id="founder-2", generation=0)
    
    gen1_child, _ = merge_kernels_geodesic([founder1, founder2])
    gen1_child.generation = 1
    
    founder3 = KernelGenome(genome_id="founder-3", generation=0)
    gen2_child, _ = merge_kernels_geodesic([gen1_child, founder3])
    gen2_child.generation = 2
    
    # Build registry
    registry = {
        founder1.genome_id: founder1,
        founder2.genome_id: founder2,
        founder3.genome_id: founder3,
        gen1_child.genome_id: gen1_child,
        gen2_child.genome_id: gen2_child,
    }
    
    # Get genealogy tree
    tree = get_genealogy_tree(gen2_child, registry, max_depth=10)
    
    assert tree['genome_id'] == gen2_child.genome_id
    assert tree['generation'] == 2
    assert len(tree['parents']) == 2
    
    print(f"✅ Genealogy tree generated")
    print(f"   Root generation: {tree['generation']}")
    print(f"   Direct parents: {len(tree['parents'])}")


# =============================================================================
# CANNIBALISM TESTS
# =============================================================================

def test_winner_loser_determination():
    """Test winner/loser determination logic."""
    print("\n=== Test Winner/Loser Determination ===")
    
    genome_a = KernelGenome(genome_id="a", fitness_score=0.8)
    genome_b = KernelGenome(genome_id="b", fitness_score=0.6)
    
    # Higher phi wins
    winner, loser, reason = determine_winner_loser(
        genome_a, genome_b,
        phi_a=0.85, phi_b=0.65,
        kappa_a=64.0, kappa_b=64.0,
    )
    
    assert winner.genome_id == "a"
    assert "Higher Φ" in reason
    
    # Closer to kappa* wins when phi similar
    winner, loser, reason = determine_winner_loser(
        genome_a, genome_b,
        phi_a=0.75, phi_b=0.76,
        kappa_a=64.0, kappa_b=50.0,
    )
    
    assert winner.genome_id == "a"
    assert "Closer to κ*" in reason or "Higher fitness" in reason
    
    print(f"✅ Winner/loser determination correct")


def test_cannibalism_operation():
    """Test cannibalism with genome absorption."""
    print("\n=== Test Cannibalism Operation ===")
    
    # Create winner and loser
    winner_basin = np.ones(BASIN_DIM) / BASIN_DIM
    loser_basin = np.zeros(BASIN_DIM)
    loser_basin[0:10] = 1.0
    loser_basin = fisher_normalize(loser_basin)
    
    winner_faculties = FacultyConfig(
        active_faculties={E8Faculty.ZEUS},
        activation_strengths={E8Faculty.ZEUS: 0.7},
    )
    
    loser_faculties = FacultyConfig(
        active_faculties={E8Faculty.APOLLO},
        activation_strengths={E8Faculty.APOLLO: 0.9},
    )
    
    winner = KernelGenome(
        genome_id="winner",
        basin_seed=winner_basin,
        faculties=winner_faculties,
        fitness_score=0.8,
    )
    
    loser = KernelGenome(
        genome_id="loser",
        basin_seed=loser_basin,
        faculties=loser_faculties,
        fitness_score=0.5,
    )
    
    # Perform cannibalism
    modified_winner, record = perform_cannibalism(
        winner,
        loser,
        absorption_rate=0.3,
        absorb_faculties=True,
    )
    
    # Validate
    assert modified_winner.genome_id == "winner"  # Identity preserved
    assert record.winner_genome_id == "winner"
    assert record.loser_genome_id == "loser"
    assert record.fisher_distance > 0
    # Check mutation count increased via record
    assert record.winner_after.mutation_count > record.winner_before.mutation_count
    
    # Check basin moved
    d_before = fisher_rao_distance(winner_basin, loser_basin)
    d_after = fisher_rao_distance(modified_winner.basin_seed, loser_basin)
    assert d_after < d_before, "Winner should move toward loser"
    
    print(f"✅ Cannibalism successful")
    print(f"   Absorbed faculties: {len(record.absorbed_faculties)}")
    print(f"   Fisher distance: {record.fisher_distance:.4f}")


def test_genome_archival():
    """Test genome archival and resurrection."""
    print("\n=== Test Genome Archival ===")
    
    genome = KernelGenome(
        genome_id="to-archive",
        fitness_score=0.75,
    )
    
    # Archive genome
    archive = archive_genome(
        genome,
        archival_reason="test archival",
        final_fitness=0.75,
        resurrection_conditions={'min_fitness': 0.6},
    )
    
    assert archive.genome.genome_id == "to-archive"
    assert archive.final_fitness == 0.75
    assert archive.resurrection_count == 0
    
    # Check eligibility
    eligible, reason = check_resurrection_eligibility(
        archive,
        current_context={},
    )
    assert eligible, f"Should be eligible: {reason}"
    
    # Resurrect
    resurrected = resurrect_from_archive(archive, mutation_rate=0.1)
    
    assert resurrected.genome_id != genome.genome_id  # New identity
    assert resurrected.generation == genome.generation + 1
    assert genome.genome_id in resurrected.parent_genomes
    assert archive.resurrection_count == 1
    
    print(f"✅ Archival and resurrection successful")
    print(f"   Resurrection count: {archive.resurrection_count}")
    print(f"   Resurrected generation: {resurrected.generation}")


def test_resurrection_eligibility():
    """Test resurrection eligibility checks."""
    print("\n=== Test Resurrection Eligibility ===")
    
    genome = KernelGenome(
        genome_id="test",
        fitness_score=0.5,
        generation=5,
    )
    
    archive = archive_genome(
        genome,
        archival_reason="test",
        final_fitness=0.5,
        resurrection_conditions={
            'min_fitness': 0.6,
            'max_generation': 3,
            'max_resurrections': 2,
        },
    )
    
    # Should fail due to low fitness
    eligible, reason = check_resurrection_eligibility(archive, {})
    assert not eligible
    assert "Fitness too low" in reason
    
    # Adjust conditions
    archive.resurrection_conditions['min_fitness'] = 0.4
    eligible, reason = check_resurrection_eligibility(archive, {})
    assert not eligible
    assert "Generation too high" in reason
    
    print(f"✅ Resurrection eligibility checks working")


# =============================================================================
# INTEGRATION TEST
# =============================================================================

def test_full_lifecycle():
    """Test complete kernel evolution lifecycle."""
    print("\n=== Test Full Lifecycle ===")
    
    # Create founders
    founder1 = KernelGenome(genome_id="founder-1", generation=0)
    founder2 = KernelGenome(genome_id="founder-2", generation=0)
    
    # Merge to create child
    child, merge_record = merge_kernels_geodesic([founder1, founder2])
    lineage = track_lineage(child, [founder1, founder2], merge_record)
    
    # Create another kernel for cannibalism
    victim = KernelGenome(genome_id="victim", generation=0, fitness_score=0.3)
    
    # Determine winner
    winner, loser, reason = determine_winner_loser(
        child, victim,
        phi_a=0.8, phi_b=0.5,
        kappa_a=64.0, kappa_b=50.0,
    )
    
    # Perform cannibalism
    survivor, cannibal_record = perform_cannibalism(
        winner, loser,
        absorption_rate=0.2,
    )
    
    # Archive loser
    archive = archive_genome(
        loser,
        archival_reason="cannibalized",
        final_fitness=loser.fitness_score,
    )
    
    # Resurrect later
    resurrected = resurrect_from_archive(archive, mutation_rate=0.05)
    
    print(f"✅ Full lifecycle complete")
    print(f"   Merged 2 founders → child (gen {child.generation})")
    print(f"   Cannibalism: {winner.genome_id} absorbed {loser.genome_id}")
    print(f"   Archived and resurrected {loser.genome_id}")
    print(f"   Resurrected as {resurrected.genome_id} (gen {resurrected.generation})")


if __name__ == "__main__":
    """Run all tests."""
    test_genome_creation()
    test_faculty_config()
    test_constraint_set()
    test_genome_serialization()
    
    test_binary_merge()
    test_multi_parent_merge()
    test_lineage_tracking()
    test_genealogy_tree()
    
    test_winner_loser_determination()
    test_cannibalism_operation()
    test_genome_archival()
    test_resurrection_eligibility()
    
    test_full_lifecycle()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✅")
    print("="*60)
