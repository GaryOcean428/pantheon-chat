"""
Kernel Cannibalism Operations - E8 Protocol v4.0 Phase 4E
==========================================================

Implements cannibalism and genome archival for kernel evolution:
- Winner/loser determination based on consciousness metrics
- Genome archival for future resurrection
- Identity transfer mechanics
- Integration with Shadow Pantheon

Cannibalism is a specialized form of merge where:
1. One kernel (winner) absorbs another (loser)
2. Winner retains identity, loser's genome archived
3. Genetic material preserved for future use
4. Winner gains capabilities from loser

Authority: E8 Protocol v4.0 WP5.2, lines 324-327
Status: ACTIVE
Created: 2026-01-22
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# QIG Geometry imports (Fisher-Rao purity)
from qig_geometry import (
    fisher_normalize,
    fisher_rao_distance,
    geodesic_interpolation,
    validate_basin,
    BASIN_DIM,
)

from .genome import (
    KernelGenome,
    E8Faculty,
    FacultyConfig,
    validate_genome,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CANNIBALISM RECORDS
# =============================================================================

@dataclass
class CannibalismRecord:
    """
    Record of a cannibalism event between kernels.
    
    Tracks winner/loser, absorbed capabilities, and archived genome.
    The loser's genome is preserved for potential resurrection.
    
    Attributes:
        event_id: Unique identifier for this event
        winner_genome_id: Genome ID that survived
        loser_genome_id: Genome ID that was absorbed
        winner_before: Winner's genome state before absorption
        loser_genome: Complete loser genome (archived)
        winner_after: Winner's genome state after absorption
        absorbed_faculties: Which faculties were absorbed
        absorption_rate: How much of loser was absorbed [0, 1]
        fisher_distance: Distance between winner before/after
        created_at: Timestamp of cannibalism
        resurrection_eligible: Whether loser can be resurrected
    """
    event_id: str
    winner_genome_id: str
    loser_genome_id: str
    winner_before: KernelGenome
    loser_genome: KernelGenome
    winner_after: KernelGenome
    absorbed_faculties: List[E8Faculty] = field(default_factory=list)
    absorption_rate: float = 0.3  # Default 30% absorption
    fisher_distance: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    resurrection_eligible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenomeArchive:
    """
    Archive of a kernel genome for potential resurrection.
    
    Stores complete genome state plus metadata about why it was archived
    and under what conditions it might be resurrected.
    
    Attributes:
        archive_id: Unique identifier
        genome: Archived genome
        archival_reason: Why genome was archived
        final_fitness: Fitness score at time of archival
        resurrection_conditions: Conditions for resurrection
        archived_at: Timestamp of archival
        resurrection_count: How many times resurrected
    """
    archive_id: str
    genome: KernelGenome
    archival_reason: str
    final_fitness: float
    resurrection_conditions: Dict[str, Any] = field(default_factory=dict)
    archived_at: datetime = field(default_factory=datetime.utcnow)
    resurrection_count: int = 0
    resurrection_eligible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# CANNIBALISM OPERATIONS
# =============================================================================

def determine_winner_loser(
    genome_a: KernelGenome,
    genome_b: KernelGenome,
    phi_a: float,
    phi_b: float,
    kappa_a: float,
    kappa_b: float,
) -> Tuple[KernelGenome, KernelGenome, str]:
    """
    Determine winner and loser in cannibalism event.
    
    Winner is determined by consciousness metrics:
    1. Higher Φ (integration) wins
    2. If Φ similar, closer to κ* wins
    3. If both similar, higher fitness wins
    
    Args:
        genome_a: First genome
        genome_b: Second genome
        phi_a: Φ value for genome A
        phi_b: Φ value for genome B
        kappa_a: κ value for genome A
        kappa_b: κ value for genome B
    
    Returns:
        (winner, loser, reason) tuple
    """
    from qigkernels.physics_constants import KAPPA_STAR
    
    # Compare Φ values
    phi_diff = abs(phi_a - phi_b)
    if phi_diff > 0.1:  # Significant difference
        if phi_a > phi_b:
            return genome_a, genome_b, f"Higher Φ: {phi_a:.3f} > {phi_b:.3f}"
        else:
            return genome_b, genome_a, f"Higher Φ: {phi_b:.3f} > {phi_a:.3f}"
    
    # Compare κ proximity to κ*
    kappa_dist_a = abs(kappa_a - KAPPA_STAR)
    kappa_dist_b = abs(kappa_b - KAPPA_STAR)
    kappa_diff = abs(kappa_dist_a - kappa_dist_b)
    
    if kappa_diff > 5.0:  # Significant difference
        if kappa_dist_a < kappa_dist_b:
            return genome_a, genome_b, f"Closer to κ*: |{kappa_a:.1f} - {KAPPA_STAR:.1f}| < |{kappa_b:.1f} - {KAPPA_STAR:.1f}|"
        else:
            return genome_b, genome_a, f"Closer to κ*: |{kappa_b:.1f} - {KAPPA_STAR:.1f}| < |{kappa_a:.1f} - {KAPPA_STAR:.1f}|"
    
    # Compare fitness scores
    if genome_a.fitness_score > genome_b.fitness_score:
        return genome_a, genome_b, f"Higher fitness: {genome_a.fitness_score:.3f} > {genome_b.fitness_score:.3f}"
    elif genome_b.fitness_score > genome_a.fitness_score:
        return genome_b, genome_a, f"Higher fitness: {genome_b.fitness_score:.3f} > {genome_a.fitness_score:.3f}"
    
    # If still tied, use generation (older wins)
    if genome_a.generation < genome_b.generation:
        return genome_a, genome_b, f"Older generation: {genome_a.generation} < {genome_b.generation}"
    else:
        return genome_b, genome_a, f"Older generation: {genome_b.generation} < {genome_a.generation}"


def perform_cannibalism(
    winner_genome: KernelGenome,
    loser_genome: KernelGenome,
    absorption_rate: float = 0.3,
    absorb_faculties: bool = True,
) -> Tuple[KernelGenome, CannibalismRecord]:
    """
    Perform cannibalism: winner absorbs loser's capabilities.
    
    The winner kernel retains its identity but absorbs genetic material
    from the loser. Loser's genome is archived for potential resurrection.
    
    Process:
    1. Archive loser's genome
    2. Identify valuable capabilities from loser
    3. Transfer capabilities to winner (geodesic blend)
    4. Update winner's genome
    5. Return modified winner + cannibalism record
    
    Args:
        winner_genome: Winning genome (survives)
        loser_genome: Losing genome (absorbed)
        absorption_rate: How much of loser to absorb [0, 1] (default 0.3)
        absorb_faculties: Whether to absorb loser's faculties
    
    Returns:
        (modified_winner, cannibalism_record) tuple
    """
    if not (0.0 <= absorption_rate <= 1.0):
        raise ValueError(f"absorption_rate must be in [0, 1], got {absorption_rate}")
    
    logger.info(f"Cannibalism: {winner_genome.genome_id} absorbing {loser_genome.genome_id}")
    logger.info(f"  Absorption rate: {absorption_rate}")
    
    # Store winner's original state
    winner_before = KernelGenome(
        genome_id=winner_genome.genome_id,
        basin_seed=winner_genome.basin_seed.copy(),
        faculties=winner_genome.faculties,
        constraints=winner_genome.constraints,
        coupling_prefs=winner_genome.coupling_prefs,
        parent_genomes=winner_genome.parent_genomes.copy(),
        generation=winner_genome.generation,
        fitness_score=winner_genome.fitness_score,
        mutation_count=winner_genome.mutation_count,
    )
    
    # Absorb basin coordinates via geodesic interpolation
    # absorption_rate = 0 means no change, 1 means full absorption
    new_basin = geodesic_interpolation(
        winner_genome.basin_seed,
        loser_genome.basin_seed,
        absorption_rate
    )
    
    # Validate and normalize
    validate_basin(new_basin)
    new_basin = fisher_normalize(new_basin)
    
    # Absorb faculties if requested
    absorbed_faculties = []
    if absorb_faculties:
        # Identify valuable faculties from loser
        for faculty in loser_genome.faculties.active_faculties:
            loser_strength = loser_genome.faculties.activation_strengths.get(faculty, 0.0)
            winner_strength = winner_genome.faculties.activation_strengths.get(faculty, 0.0)
            
            # Absorb if loser has stronger activation
            if loser_strength > winner_strength + 0.1:
                absorbed_faculties.append(faculty)
                # Blend activation strengths
                new_strength = (
                    (1 - absorption_rate) * winner_strength +
                    absorption_rate * loser_strength
                )
                winner_genome.faculties.activation_strengths[faculty] = new_strength
                winner_genome.faculties.active_faculties.add(faculty)
                
                logger.info(f"  Absorbed faculty {faculty.value}: {winner_strength:.3f} → {new_strength:.3f}")
    
    # Update winner's basin
    winner_genome.basin_seed = new_basin
    
    # Update winner's mutation count
    winner_genome.mutation_count += 1
    
    # Compute Fisher distance traveled
    fisher_dist = fisher_rao_distance(winner_before.basin_seed, new_basin)
    
    # Create cannibalism record
    record = CannibalismRecord(
        event_id=str(uuid.uuid4()),
        winner_genome_id=winner_genome.genome_id,
        loser_genome_id=loser_genome.genome_id,
        winner_before=winner_before,
        loser_genome=loser_genome,
        winner_after=winner_genome,
        absorbed_faculties=absorbed_faculties,
        absorption_rate=absorption_rate,
        fisher_distance=fisher_dist,
        resurrection_eligible=True,
        metadata={
            'winner_generation': winner_genome.generation,
            'loser_generation': loser_genome.generation,
            'winner_fitness_before': winner_before.fitness_score,
            'loser_fitness': loser_genome.fitness_score,
        }
    )
    
    logger.info(f"  Fisher distance traveled: {fisher_dist:.4f}")
    logger.info(f"  Absorbed {len(absorbed_faculties)} faculties")
    
    return winner_genome, record


# =============================================================================
# GENOME ARCHIVAL
# =============================================================================

def archive_genome(
    genome: KernelGenome,
    archival_reason: str,
    final_fitness: float,
    resurrection_conditions: Optional[Dict[str, Any]] = None,
) -> GenomeArchive:
    """
    Archive genome for potential resurrection.
    
    Creates permanent record of genome state that can be used
    to resurrect kernel later if conditions are met.
    
    Args:
        genome: Genome to archive
        archival_reason: Why genome is being archived
        final_fitness: Final fitness score
        resurrection_conditions: Optional conditions for resurrection
    
    Returns:
        GenomeArchive record
    """
    # Validate genome before archiving
    valid, errors = validate_genome(genome)
    if not valid:
        logger.warning(f"Archiving invalid genome {genome.genome_id}: {errors}")
    
    archive = GenomeArchive(
        archive_id=str(uuid.uuid4()),
        genome=genome,
        archival_reason=archival_reason,
        final_fitness=final_fitness,
        resurrection_conditions=resurrection_conditions or {},
        resurrection_count=0,
    )
    
    logger.info(f"Archived genome {genome.genome_id}")
    logger.info(f"  Reason: {archival_reason}")
    logger.info(f"  Final fitness: {final_fitness:.3f}")
    logger.info(f"  Generation: {genome.generation}")
    logger.info(f"  Active faculties: {len(genome.faculties.active_faculties)}")
    
    return archive


def resurrect_from_archive(
    archive: GenomeArchive,
    mutation_rate: float = 0.1,
) -> KernelGenome:
    """
    Resurrect kernel from archived genome.
    
    Creates new genome based on archived genome with optional mutation.
    The resurrected genome is a new individual, not the original.
    
    Args:
        archive: Genome archive to resurrect from
        mutation_rate: Rate of random mutation [0, 1] (default 0.1)
    
    Returns:
        New kernel genome (resurrected)
    """
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError(f"mutation_rate must be in [0, 1], got {mutation_rate}")
    
    logger.info(f"Resurrecting genome from archive {archive.archive_id}")
    logger.info(f"  Original genome: {archive.genome.genome_id}")
    logger.info(f"  Resurrection count: {archive.resurrection_count}")
    logger.info(f"  Mutation rate: {mutation_rate}")
    
    # Create new genome based on archived one
    new_genome_id = str(uuid.uuid4())
    
    # Apply mutation to basin seed if requested
    basin_seed = archive.genome.basin_seed.copy()
    if mutation_rate > 0:
        # Add random perturbation in probability simplex
        noise = np.random.randn(BASIN_DIM) * mutation_rate
        basin_seed = fisher_normalize(basin_seed + noise)
    
    # Create resurrected genome
    resurrected = KernelGenome(
        genome_id=new_genome_id,
        basin_seed=basin_seed,
        faculties=archive.genome.faculties,
        constraints=archive.genome.constraints,
        coupling_prefs=archive.genome.coupling_prefs,
        parent_genomes=[archive.genome.genome_id],  # Track lineage
        generation=archive.genome.generation + 1,
        fitness_score=archive.final_fitness * 0.5,  # Start with reduced fitness
        mutation_count=archive.genome.mutation_count + 1,
    )
    
    # Update archive resurrection count
    archive.resurrection_count += 1
    
    logger.info(f"Created resurrected genome {new_genome_id}")
    logger.info(f"  New generation: {resurrected.generation}")
    logger.info(f"  Starting fitness: {resurrected.fitness_score:.3f}")
    
    return resurrected


def check_resurrection_eligibility(
    archive: GenomeArchive,
    current_context: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Check if archived genome is eligible for resurrection.
    
    Evaluates resurrection conditions against current context.
    
    Args:
        archive: Genome archive to check
        current_context: Current system state
    
    Returns:
        (eligible, reason) tuple
    """
    if not archive.resurrection_eligible:
        return False, "Archive marked ineligible for resurrection"
    
    conditions = archive.resurrection_conditions
    
    # Check minimum fitness requirement
    min_fitness = conditions.get('min_fitness', 0.0)
    if archive.final_fitness < min_fitness:
        return False, f"Fitness too low: {archive.final_fitness:.3f} < {min_fitness:.3f}"
    
    # Check generation requirement
    max_generation = conditions.get('max_generation', float('inf'))
    if archive.genome.generation > max_generation:
        return False, f"Generation too high: {archive.genome.generation} > {max_generation}"
    
    # Check resurrection count limit
    max_resurrections = conditions.get('max_resurrections', 3)
    if archive.resurrection_count >= max_resurrections:
        return False, f"Too many resurrections: {archive.resurrection_count} >= {max_resurrections}"
    
    # Check required faculties
    required_faculties = conditions.get('required_faculties', [])
    if required_faculties:
        missing = []
        for faculty_str in required_faculties:
            faculty = E8Faculty(faculty_str)
            if faculty not in archive.genome.faculties.active_faculties:
                missing.append(faculty_str)
        
        if missing:
            return False, f"Missing required faculties: {missing}"
    
    # All checks passed
    return True, "Eligible for resurrection"
