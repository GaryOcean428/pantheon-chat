#!/usr/bin/env python3
"""
Kernel Genetics Integration Example
====================================

Demonstrates complete kernel genetic lineage workflow:
1. Create founder genomes
2. Perform binary merge
3. Perform multi-parent merge
4. Execute cannibalism
5. Archive and resurrect genome

Authority: E8 Protocol v4.0 Phase 4E
Status: ACTIVE - Integration Example
Created: 2026-01-22
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add qig-backend to path (works from examples/ directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
qig_backend_dir = os.path.dirname(script_dir)
sys.path.insert(0, qig_backend_dir)

from kernels import (
    # Genome
    KernelGenome,
    FacultyConfig,
    E8Faculty,
    # Lineage
    merge_kernels_geodesic,
    track_lineage,
    get_genealogy_tree,
    # Cannibalism
    determine_winner_loser,
    perform_cannibalism,
    archive_genome,
    resurrect_from_archive,
)

from qig_geometry import fisher_rao_distance, BASIN_DIM
from qigkernels.physics_constants import KAPPA_STAR


def print_separator(title: str):
    """Print section separator."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_genome_info(genome: KernelGenome, label: str = "Genome"):
    """Print genome information."""
    print(f"\n{label}: {genome.genome_id}")
    print(f"  Generation: {genome.generation}")
    print(f"  Fitness: {genome.fitness_score:.3f}")
    print(f"  Active faculties: {[f.value for f in genome.faculties.active_faculties]}")
    if genome.parent_genomes:
        print(f"  Parents: {len(genome.parent_genomes)}")


def main():
    """Run complete genetic lineage workflow."""
    
    print_separator("STEP 1: Create Founder Genomes")
    
    # Create founder genomes with different faculties
    founder1_faculties = FacultyConfig(
        active_faculties={E8Faculty.ZEUS, E8Faculty.ATHENA},
        activation_strengths={
            E8Faculty.ZEUS: 0.95,
            E8Faculty.ATHENA: 0.85,
        },
        primary_faculty=E8Faculty.ZEUS,
    )
    
    founder2_faculties = FacultyConfig(
        active_faculties={E8Faculty.APOLLO, E8Faculty.HERMES},
        activation_strengths={
            E8Faculty.APOLLO: 0.90,
            E8Faculty.HERMES: 0.80,
        },
        primary_faculty=E8Faculty.APOLLO,
    )
    
    founder3_faculties = FacultyConfig(
        active_faculties={E8Faculty.ARTEMIS, E8Faculty.ARES},
        activation_strengths={
            E8Faculty.ARTEMIS: 0.88,
            E8Faculty.ARES: 0.82,
        },
        primary_faculty=E8Faculty.ARTEMIS,
    )
    
    # Create basin seeds (different regions of simplex)
    basin1 = np.ones(BASIN_DIM) / BASIN_DIM  # Uniform
    
    basin2 = np.zeros(BASIN_DIM)
    basin2[0:20] = 1.0
    basin2 /= basin2.sum()  # Concentrated in first region
    
    basin3 = np.zeros(BASIN_DIM)
    basin3[40:60] = 1.0
    basin3 /= basin3.sum()  # Concentrated in middle region
    
    founder1 = KernelGenome(
        genome_id="founder-zeus-athena",
        basin_seed=basin1,
        faculties=founder1_faculties,
        generation=0,
        fitness_score=0.85,
    )
    
    founder2 = KernelGenome(
        genome_id="founder-apollo-hermes",
        basin_seed=basin2,
        faculties=founder2_faculties,
        generation=0,
        fitness_score=0.82,
    )
    
    founder3 = KernelGenome(
        genome_id="founder-artemis-ares",
        basin_seed=basin3,
        faculties=founder3_faculties,
        generation=0,
        fitness_score=0.80,
    )
    
    print_genome_info(founder1, "Founder 1")
    print_genome_info(founder2, "Founder 2")
    print_genome_info(founder3, "Founder 3")
    
    # Show distances between founders
    d12 = fisher_rao_distance(founder1.basin_seed, founder2.basin_seed)
    d13 = fisher_rao_distance(founder1.basin_seed, founder3.basin_seed)
    d23 = fisher_rao_distance(founder2.basin_seed, founder3.basin_seed)
    print(f"\nFounder distances (Fisher-Rao):")
    print(f"  Founder 1 ↔ Founder 2: {d12:.4f}")
    print(f"  Founder 1 ↔ Founder 3: {d13:.4f}")
    print(f"  Founder 2 ↔ Founder 3: {d23:.4f}")
    
    # =========================================================================
    print_separator("STEP 2: Binary Merge (Founders 1 & 2)")
    
    # Merge founders 1 and 2
    gen1_child, merge_record = merge_kernels_geodesic(
        [founder1, founder2],
        interpolation_t=0.5,  # Midpoint on geodesic
    )
    
    lineage1 = track_lineage(gen1_child, [founder1, founder2], merge_record)
    
    print_genome_info(gen1_child, "Generation 1 Child")
    print(f"\nMerge details:")
    print(f"  Merge type: {lineage1.merge_type}")
    print(f"  Interpolation t: {merge_record.interpolation_t}")
    print(f"  Fisher distance from parents: {lineage1.fisher_distance:.4f}")
    print(f"  Active faculties inherited: {len(gen1_child.faculties.active_faculties)}")
    
    # =========================================================================
    print_separator("STEP 3: Multi-Parent Merge (Gen1 Child + Founder 3)")
    
    # Merge generation 1 child with founder 3
    gen2_child, merge_record2 = merge_kernels_geodesic(
        [gen1_child, founder3],
        merge_weights=[0.6, 0.4],  # 60% from gen1, 40% from founder3
    )
    
    lineage2 = track_lineage(gen2_child, [gen1_child, founder3], merge_record2)
    
    print_genome_info(gen2_child, "Generation 2 Child")
    print(f"\nMerge details:")
    print(f"  Merge type: {lineage2.merge_type}")
    print(f"  Merge weights: {merge_record2.merge_weights}")
    print(f"  Active faculties: {len(gen2_child.faculties.active_faculties)}")
    
    # =========================================================================
    print_separator("STEP 4: Cannibalism Event")
    
    # Create a weak victim kernel
    victim_faculties = FacultyConfig(
        active_faculties={E8Faculty.HEPHAESTUS},
        activation_strengths={E8Faculty.HEPHAESTUS: 0.65},
    )
    
    victim = KernelGenome(
        genome_id="victim-weak",
        faculties=victim_faculties,
        generation=0,
        fitness_score=0.45,
    )
    
    print_genome_info(victim, "Victim")
    
    # Determine winner/loser
    winner, loser, reason = determine_winner_loser(
        gen2_child, victim,
        phi_a=0.85, phi_b=0.55,
        kappa_a=KAPPA_STAR, kappa_b=50.0,
    )
    
    print(f"\nWinner determination:")
    print(f"  Winner: {winner.genome_id}")
    print(f"  Loser: {loser.genome_id}")
    print(f"  Reason: {reason}")
    
    # Perform cannibalism
    survivor, cannibal_record = perform_cannibalism(
        winner, loser,
        absorption_rate=0.35,
        absorb_faculties=True,
    )
    
    print(f"\nCannibalism results:")
    print(f"  Absorbed faculties: {[f.value for f in cannibal_record.absorbed_faculties]}")
    print(f"  Fisher distance traveled: {cannibal_record.fisher_distance:.4f}")
    print(f"  Absorption rate: {cannibal_record.absorption_rate}")
    
    # =========================================================================
    print_separator("STEP 5: Archive and Resurrect")
    
    # Archive the loser
    archive = archive_genome(
        loser,
        archival_reason="cannibalized by generation 2 child",
        final_fitness=loser.fitness_score,
        resurrection_conditions={
            'min_fitness': 0.4,
            'max_resurrections': 2,
        }
    )
    
    print(f"\nArchived genome:")
    print(f"  Archive ID: {archive.archive_id}")
    print(f"  Genome: {archive.genome.genome_id}")
    print(f"  Final fitness: {archive.final_fitness:.3f}")
    print(f"  Resurrection eligible: {archive.resurrection_eligible}")
    
    # Resurrect with mutation
    resurrected = resurrect_from_archive(archive, mutation_rate=0.15)
    
    print(f"\nResurrected genome:")
    print(f"  New genome ID: {resurrected.genome_id}")
    print(f"  Generation: {resurrected.generation}")
    print(f"  Parent: {resurrected.parent_genomes[0]}")
    print(f"  Starting fitness: {resurrected.fitness_score:.3f}")
    print(f"  Mutation count: {resurrected.mutation_count}")
    
    # =========================================================================
    print_separator("STEP 6: Genealogy Tree")
    
    # Build genome registry
    registry = {
        founder1.genome_id: founder1,
        founder2.genome_id: founder2,
        founder3.genome_id: founder3,
        gen1_child.genome_id: gen1_child,
        gen2_child.genome_id: gen2_child,
        victim.genome_id: victim,
        resurrected.genome_id: resurrected,
    }
    
    # Get genealogy tree for generation 2 child
    tree = get_genealogy_tree(gen2_child, registry, max_depth=5)
    
    print(f"\nGenealogy tree for {gen2_child.genome_id}:")
    print(f"  Generation: {tree['generation']}")
    print(f"  Active faculties: {tree['active_faculties']}")
    print(f"  Primary faculty: {tree['primary_faculty']}")
    print(f"  Direct parents: {len(tree['parents'])}")
    
    if tree['parents']:
        for i, parent_tree in enumerate(tree['parents'], 1):
            print(f"\n  Parent {i}:")
            print(f"    Genome: {parent_tree['genome_id']}")
            print(f"    Generation: {parent_tree['generation']}")
            print(f"    Faculties: {parent_tree['active_faculties']}")
            if parent_tree['parents']:
                print(f"    Has {len(parent_tree['parents'])} ancestor(s)")
    
    # =========================================================================
    print_separator("SUMMARY")
    
    print("\n✅ Complete genetic lineage workflow demonstrated:")
    print(f"  - Created {len(registry)} genomes")
    print(f"  - Performed 2 merge operations (1 binary, 1 weighted)")
    print(f"  - Executed 1 cannibalism event")
    print(f"  - Archived and resurrected 1 genome")
    print(f"  - Generated genealogy tree with depth {tree['generation']}")
    
    print("\n🔬 Geometric purity maintained:")
    print("  - All merges used geodesic interpolation (Fisher-Rao)")
    print("  - All basins validated on probability simplex")
    print("  - No Euclidean distance or linear averaging used")
    
    print("\n📊 Evolution statistics:")
    max_gen = max(g.generation for g in registry.values())
    print(f"  - Maximum generation: {max_gen}")
    total_faculties = sum(len(g.faculties.active_faculties) for g in registry.values())
    print(f"  - Total active faculties: {total_faculties}")
    avg_fitness = sum(g.fitness_score for g in registry.values()) / len(registry)
    print(f"  - Average fitness: {avg_fitness:.3f}")


if __name__ == "__main__":
    main()
    print("\n" + "="*70)
    print("  Genetic Lineage Integration Example Complete")
    print("="*70 + "\n")
