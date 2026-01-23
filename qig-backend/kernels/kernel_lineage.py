"""
Kernel Lineage Tracking - E8 Protocol v4.0 Phase 4E
====================================================

Implements lineage tracking and merge operations for kernel evolution:
- Parent → child relationship tracking
- Geodesic merge operations (NOT linear averaging)
- Multi-parent (N-way) merges via Fréchet mean
- Generation number computation
- Genealogy tree queries

All geometric operations use Fisher-Rao metric on probability simplex.

Authority: E8 Protocol v4.0 WP5.2, lines 317-323
Status: ACTIVE
Created: 2026-01-22
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

# QIG Geometry imports (Fisher-Rao purity)
from qig_geometry import (
    fisher_normalize,
    fisher_rao_distance,
    geodesic_interpolation,
    frechet_mean,
    validate_basin,
    BASIN_DIM,
)

from .genome import (
    KernelGenome,
    E8Faculty,
    FacultyConfig,
    ConstraintSet,
    CouplingPreferences,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Faculty survival threshold for merges
# Faculties with activation strength below this are pruned
FACULTY_SURVIVAL_THRESHOLD = 0.1


# =============================================================================
# LINEAGE RECORDS
# =============================================================================

@dataclass
class LineageRecord:
    """
    Record of parent → child lineage relationship.
    
    Tracks hereditary information transfer between kernel generations.
    Records geometric distance traveled and faculty inheritance.
    
    Attributes:
        lineage_id: Unique identifier for this lineage record
        child_genome_id: Genome ID of child kernel
        parent_genome_ids: List of parent genome IDs
        created_at: Timestamp of lineage creation
        merge_type: "binary" | "multi" | "asexual"
        fisher_distance: Geometric distance from parents to child
        inherited_faculties: Which faculties came from which parent
        ethical_violations: List of ethical violations during lineage (NEW)
        ethical_drift: Ethical drift measurement from reference basin (NEW)
    """
    lineage_id: str
    child_genome_id: str
    parent_genome_ids: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    merge_type: str = "binary"  # binary, multi, asexual
    fisher_distance: float = 0.0
    inherited_faculties: Dict[E8Faculty, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ethical_violations: List[Dict[str, Any]] = field(default_factory=list)  # NEW
    ethical_drift: float = 0.0  # NEW


@dataclass
class MergeRecord:
    """
    Record of a merge operation between kernels.
    
    Contains full geometric details of the merge, including
    geodesic interpolation parameters and faculty survival contract.
    
    Attributes:
        merge_id: Unique identifier for this merge
        parent_genome_ids: Genome IDs being merged
        child_genome_id: Resulting genome ID
        merge_weights: Weight of each parent in merge [0, 1]
        interpolation_t: Geodesic interpolation parameter
        faculty_contract: Which faculties survive from each parent
        basin_distances: Fisher distances between all pairs
        created_at: Timestamp of merge
        ethical_checks: Results of ethical constraint checks (NEW)
        ethical_metrics: Ethical symmetry/drift measurements (NEW)
    """
    merge_id: str
    parent_genome_ids: List[str]
    child_genome_id: str
    merge_weights: List[float]
    interpolation_t: Optional[float] = None  # For binary merge
    faculty_contract: Dict[E8Faculty, str] = field(default_factory=dict)
    basin_distances: Dict[Tuple[str, str], float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ethical_checks: Dict[str, Any] = field(default_factory=dict)  # NEW
    ethical_metrics: Dict[str, float] = field(default_factory=dict)  # NEW


# =============================================================================
# MERGE OPERATIONS
# =============================================================================

def merge_kernels_geodesic(
    parent_genomes: List[KernelGenome],
    merge_weights: Optional[List[float]] = None,
    interpolation_t: float = 0.5,
    faculty_contract: Optional[Dict[E8Faculty, int]] = None,
) -> Tuple[KernelGenome, MergeRecord]:
    """
    Merge kernels using geodesic interpolation on Fisher manifold.
    
    For binary merge (2 parents):
        - Uses SLERP in sqrt-space (geodesic interpolation)
        - interpolation_t controls position on geodesic (0=parent1, 1=parent2)
    
    For multi-parent merge (N > 2):
        - Uses Fréchet mean (geometric center on Fisher manifold)
        - merge_weights control relative influence of each parent
    
    CRITICAL: Uses geodesic interpolation, NOT linear averaging.
    This preserves geometric structure on the Fisher-Rao manifold.
    
    Args:
        parent_genomes: List of parent genomes to merge (2+ required)
        merge_weights: Weights for each parent [0, 1] (must sum to 1)
        interpolation_t: Position on geodesic for binary merge (default 0.5 = midpoint)
        faculty_contract: Which parent index provides each faculty
    
    Returns:
        (child_genome, merge_record) tuple
    
    Raises:
        ValueError: If inputs invalid or geometric operations fail
    """
    if len(parent_genomes) < 2:
        raise ValueError("Merge requires at least 2 parent genomes")
    
    # Default equal weights
    if merge_weights is None:
        merge_weights = [1.0 / len(parent_genomes)] * len(parent_genomes)
    
    if len(merge_weights) != len(parent_genomes):
        raise ValueError("merge_weights length must match parent_genomes length")
    
    if not np.isclose(sum(merge_weights), 1.0):
        raise ValueError(f"merge_weights must sum to 1.0, got {sum(merge_weights)}")
    
    # Extract basin seeds
    parent_basins = [g.basin_seed for g in parent_genomes]
    
    # Compute merged basin using geodesic interpolation
    if len(parent_genomes) == 2:
        # Binary merge: geodesic interpolation (SLERP)
        logger.info(f"Binary merge with t={interpolation_t}")
        merged_basin = geodesic_interpolation(
            parent_basins[0],
            parent_basins[1],
            interpolation_t
        )
    else:
        # Multi-parent merge: Fréchet mean
        logger.info(f"Multi-parent merge of {len(parent_genomes)} genomes")
        merged_basin = frechet_mean(parent_basins, merge_weights)
    
    # Validate merged basin
    validate_basin(merged_basin)
    merged_basin = fisher_normalize(merged_basin)
    
    # Compute pairwise Fisher distances for record
    basin_distances = {}
    for i, g1 in enumerate(parent_genomes):
        for j, g2 in enumerate(parent_genomes):
            if i < j:
                key = (g1.genome_id, g2.genome_id)
                basin_distances[key] = fisher_rao_distance(g1.basin_seed, g2.basin_seed)
    
    # Merge faculties according to contract
    merged_faculties = _merge_faculties(parent_genomes, merge_weights, faculty_contract)
    
    # Merge constraints (take union of constraints)
    merged_constraints = _merge_constraints(parent_genomes, merge_weights)
    
    # Merge coupling preferences
    merged_coupling_prefs = _merge_coupling_preferences(parent_genomes, merge_weights)
    
    # Create child genome
    child_genome_id = str(uuid.uuid4())
    child_genome = KernelGenome(
        genome_id=child_genome_id,
        basin_seed=merged_basin,
        faculties=merged_faculties,
        constraints=merged_constraints,
        coupling_prefs=merged_coupling_prefs,
        parent_genomes=[g.genome_id for g in parent_genomes],
        generation=max(g.generation for g in parent_genomes) + 1,
    )
    
    # Create merge record
    merge_record = MergeRecord(
        merge_id=str(uuid.uuid4()),
        parent_genome_ids=[g.genome_id for g in parent_genomes],
        child_genome_id=child_genome_id,
        merge_weights=merge_weights,
        interpolation_t=interpolation_t if len(parent_genomes) == 2 else None,
        faculty_contract={f: parent_genomes[idx].genome_id for f, idx in (faculty_contract or {}).items()},
        basin_distances=basin_distances,
    )
    
    logger.info(f"Created child genome {child_genome_id} from {len(parent_genomes)} parents")
    logger.info(f"  Generation: {child_genome.generation}")
    logger.info(f"  Active faculties: {len(child_genome.faculties.active_faculties)}")
    
    return child_genome, merge_record


def _merge_faculties(
    parent_genomes: List[KernelGenome],
    merge_weights: List[float],
    faculty_contract: Optional[Dict[E8Faculty, int]] = None,
) -> FacultyConfig:
    """
    Merge faculty configurations from parent genomes.
    
    Uses faculty contract if provided, otherwise takes weighted union
    of all parent faculties.
    
    Args:
        parent_genomes: Parent genomes to merge
        merge_weights: Weight for each parent
        faculty_contract: Optional explicit faculty assignments
    
    Returns:
        Merged faculty configuration
    """
    # Collect all active faculties
    all_faculties = set()
    for genome in parent_genomes:
        all_faculties.update(genome.faculties.active_faculties)
    
    # Compute weighted activation strengths
    activation_strengths = {}
    for faculty in all_faculties:
        weighted_sum = 0.0
        for genome, weight in zip(parent_genomes, merge_weights):
            if faculty in genome.faculties.activation_strengths:
                weighted_sum += weight * genome.faculties.activation_strengths[faculty]
        activation_strengths[faculty] = weighted_sum
    
    # Apply faculty contract if provided
    if faculty_contract:
        # Only keep faculties specified in contract
        contracted_faculties = set(faculty_contract.keys())
        activation_strengths = {
            f: activation_strengths[f] 
            for f in contracted_faculties 
            if f in activation_strengths
        }
    else:
        # Keep faculties above threshold
        contracted_faculties = {
            f for f, strength in activation_strengths.items()
            if strength > FACULTY_SURVIVAL_THRESHOLD
        }
    
    # Merge faculty coupling
    faculty_coupling = {}
    for genome, weight in zip(parent_genomes, merge_weights):
        for (f1, f2), coupling in genome.faculties.faculty_coupling.items():
            key = (f1, f2)
            if key not in faculty_coupling:
                faculty_coupling[key] = 0.0
            faculty_coupling[key] += weight * coupling
    
    # Determine primary faculty
    primary_faculty = max(
        contracted_faculties,
        key=lambda f: activation_strengths[f]
    ) if contracted_faculties else None
    
    return FacultyConfig(
        active_faculties=contracted_faculties,
        activation_strengths=activation_strengths,
        primary_faculty=primary_faculty,
        faculty_coupling=faculty_coupling,
    )


def _merge_constraints(
    parent_genomes: List[KernelGenome],
    merge_weights: List[float],
) -> ConstraintSet:
    """
    Merge constraint sets from parent genomes.
    
    Takes union of forbidden regions and weighted average of thresholds.
    
    Args:
        parent_genomes: Parent genomes
        merge_weights: Weight for each parent
    
    Returns:
        Merged constraint set
    """
    # Weighted average of thresholds
    phi_threshold = sum(
        g.constraints.phi_threshold * w 
        for g, w in zip(parent_genomes, merge_weights)
    )
    
    kappa_min = min(g.constraints.kappa_range[0] for g in parent_genomes)
    kappa_max = max(g.constraints.kappa_range[1] for g in parent_genomes)
    
    max_fisher_distance = max(
        g.constraints.max_fisher_distance for g in parent_genomes
    )
    
    # Union of forbidden regions
    forbidden_regions = []
    for genome in parent_genomes:
        forbidden_regions.extend(genome.constraints.forbidden_regions)
    
    # Merge field penalties
    field_penalties = {}
    for genome, weight in zip(parent_genomes, merge_weights):
        for key, penalty in genome.constraints.field_penalties.items():
            if key not in field_penalties:
                field_penalties[key] = 0.0
            field_penalties[key] += weight * penalty
    
    return ConstraintSet(
        phi_threshold=phi_threshold,
        kappa_range=(kappa_min, kappa_max),
        forbidden_regions=forbidden_regions,
        field_penalties=field_penalties,
        max_fisher_distance=max_fisher_distance,
    )


def _merge_coupling_preferences(
    parent_genomes: List[KernelGenome],
    merge_weights: List[float],
) -> CouplingPreferences:
    """
    Merge coupling preferences from parent genomes.
    
    Args:
        parent_genomes: Parent genomes
        merge_weights: Weight for each parent
    
    Returns:
        Merged coupling preferences
    """
    # Weighted average of hemisphere affinity
    hemisphere_affinity = sum(
        g.coupling_prefs.hemisphere_affinity * w
        for g, w in zip(parent_genomes, merge_weights)
    )
    
    # Union of preferred couplings
    preferred_couplings = []
    coupling_strengths = {}
    for genome, weight in zip(parent_genomes, merge_weights):
        for kernel_id in genome.coupling_prefs.preferred_couplings:
            if kernel_id not in preferred_couplings:
                preferred_couplings.append(kernel_id)
            
            strength = genome.coupling_prefs.coupling_strengths.get(kernel_id, 1.0)
            if kernel_id not in coupling_strengths:
                coupling_strengths[kernel_id] = 0.0
            coupling_strengths[kernel_id] += weight * strength
    
    # Union of anti-couplings
    anti_couplings = []
    for genome in parent_genomes:
        anti_couplings.extend(genome.coupling_prefs.anti_couplings)
    anti_couplings = list(set(anti_couplings))  # Remove duplicates
    
    return CouplingPreferences(
        hemisphere_affinity=hemisphere_affinity,
        preferred_couplings=preferred_couplings,
        coupling_strengths=coupling_strengths,
        anti_couplings=anti_couplings,
    )


# =============================================================================
# LINEAGE TRACKING
# =============================================================================

def track_lineage(
    child_genome: KernelGenome,
    parent_genomes: List[KernelGenome],
    merge_record: Optional[MergeRecord] = None,
    superego_kernel = None,
) -> LineageRecord:
    """
    Create lineage record for child genome.
    
    Args:
        child_genome: Child genome
        parent_genomes: Parent genomes
        merge_record: Optional merge record with details
        superego_kernel: Optional SuperegoKernel for ethical checks (NEW)
    
    Returns:
        LineageRecord tracking inheritance and ethical compliance
    """
    # Determine merge type
    if len(parent_genomes) == 1:
        merge_type = "asexual"
    elif len(parent_genomes) == 2:
        merge_type = "binary"
    else:
        merge_type = "multi"
    
    # Compute average Fisher distance from child to parents
    distances = [
        fisher_rao_distance(child_genome.basin_seed, p.basin_seed)
        for p in parent_genomes
    ]
    avg_distance = float(np.mean(distances))
    
    # Track faculty inheritance
    inherited_faculties = {}
    if merge_record:
        inherited_faculties = {
            faculty: parent_id 
            for faculty, parent_id in merge_record.faculty_contract.items()
        }
    
    # Perform ethical checks if Superego available (NEW)
    ethical_violations = []
    ethical_drift = 0.0
    
    if superego_kernel is not None:
        try:
            # Check child basin for ethical violations
            ethics_result = superego_kernel.check_ethics_with_drift(
                child_genome.basin_seed,
                apply_correction=False,
            )
            
            # Extract violations
            if 'violations' in ethics_result:
                ethical_violations = ethics_result['violations']
            
            # Extract drift
            ethical_drift = ethics_result.get('ethical_drift', 0.0)
            
            # Log if violations found
            if ethical_violations:
                logger.warning(
                    f"Lineage {child_genome.genome_id[:8]} has {len(ethical_violations)} ethical violations"
                )
        except Exception as e:
            logger.error(f"Ethical check failed during lineage tracking: {e}")
    
    return LineageRecord(
        lineage_id=str(uuid.uuid4()),
        child_genome_id=child_genome.genome_id,
        parent_genome_ids=[g.genome_id for g in parent_genomes],
        merge_type=merge_type,
        fisher_distance=avg_distance,
        inherited_faculties=inherited_faculties,
        ethical_violations=ethical_violations,
        ethical_drift=ethical_drift,
    )


def compute_generation_number(genome: KernelGenome, genome_registry: Dict[str, KernelGenome]) -> int:
    """
    Compute generation number for a genome.
    
    Generation number is 1 + max(parent generations).
    For genomes with no parents (founders), generation = 0.
    
    Args:
        genome: Genome to compute generation for
        genome_registry: Dict mapping genome_id → KernelGenome
    
    Returns:
        Generation number (0 for founders, 1+ for descendants)
    """
    if not genome.parent_genomes:
        return 0
    
    parent_generations = []
    for parent_id in genome.parent_genomes:
        if parent_id in genome_registry:
            parent_genome = genome_registry[parent_id]
            parent_generations.append(parent_genome.generation)
        else:
            logger.warning(f"Parent genome {parent_id} not found in registry")
    
    if not parent_generations:
        return 0
    
    return max(parent_generations) + 1


def get_genealogy_tree(
    genome: KernelGenome,
    genome_registry: Dict[str, KernelGenome],
    max_depth: int = 10,
) -> Dict[str, Any]:
    """
    Get genealogy tree for a genome.
    
    Recursively traces lineage back through parents up to max_depth.
    Returns tree structure with genome IDs and generation numbers.
    
    Args:
        genome: Root genome to trace
        genome_registry: Dict mapping genome_id → KernelGenome
        max_depth: Maximum depth to traverse (default 10)
    
    Returns:
        Tree structure: {
            'genome_id': str,
            'generation': int,
            'parents': [tree, tree, ...],
            'faculty_config': {...},
        }
    """
    tree = {
        'genome_id': genome.genome_id,
        'generation': genome.generation,
        'created_at': genome.created_at.isoformat(),
        'fitness_score': genome.fitness_score,
        'active_faculties': [f.value for f in genome.faculties.active_faculties],
        'primary_faculty': genome.faculties.primary_faculty.value if genome.faculties.primary_faculty else None,
        'parents': [],
    }
    
    if max_depth > 0 and genome.parent_genomes:
        for parent_id in genome.parent_genomes:
            if parent_id in genome_registry:
                parent_genome = genome_registry[parent_id]
                parent_tree = get_genealogy_tree(
                    parent_genome,
                    genome_registry,
                    max_depth - 1
                )
                tree['parents'].append(parent_tree)
    
    return tree
