"""
Kernel Genetics Package - E8 Protocol v4.0 Phase 4E
====================================================

Implements genetic lineage system for kernel evolution:
- Genome schema (basin seed, faculties, constraints)
- Merge operations with geodesic interpolation
- Cannibalism with genome archival
- Lineage tracking and visualization
- Database persistence

Authority: E8 Protocol v4.0 WP5.2 Phase 4E
Status: ACTIVE
Created: 2026-01-22
"""

from .genome import (
    KernelGenome,
    FacultyConfig,
    ConstraintSet,
    CouplingPreferences,
    E8Faculty,
    validate_genome,
    serialize_genome,
    deserialize_genome,
)

from .kernel_lineage import (
    LineageRecord,
    MergeRecord,
    merge_kernels_geodesic,
    track_lineage,
    compute_generation_number,
    get_genealogy_tree,
)

from .cannibalism import (
    CannibalismRecord,
    GenomeArchive,
    perform_cannibalism,
    archive_genome,
    resurrect_from_archive,
    determine_winner_loser,
    check_resurrection_eligibility,
)

from .persistence import (
    save_genome,
    load_genome,
    save_lineage_record,
    save_merge_record,
    save_cannibalism_record,
    save_genome_archive,
    get_genome_lineage,
    get_descendants,
    get_evolution_summary,
)

__all__ = [
    # Genome
    'KernelGenome',
    'FacultyConfig',
    'ConstraintSet',
    'CouplingPreferences',
    'E8Faculty',
    'validate_genome',
    'serialize_genome',
    'deserialize_genome',
    # Lineage
    'LineageRecord',
    'MergeRecord',
    'merge_kernels_geodesic',
    'track_lineage',
    'compute_generation_number',
    'get_genealogy_tree',
    # Cannibalism
    'CannibalismRecord',
    'GenomeArchive',
    'perform_cannibalism',
    'archive_genome',
    'resurrect_from_archive',
    'determine_winner_loser',
    'check_resurrection_eligibility',
    # Persistence
    'save_genome',
    'load_genome',
    'save_lineage_record',
    'save_merge_record',
    'save_cannibalism_record',
    'save_genome_archive',
    'get_genome_lineage',
    'get_descendants',
    'get_evolution_summary',
]
