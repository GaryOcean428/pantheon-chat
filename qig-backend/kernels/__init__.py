"""
Kernel Genetics Package - E8 Protocol v4.0 Phase 4E
====================================================

Implements genetic lineage system for kernel evolution:
- Genome schema (basin seed, faculties, constraints)
- Merge operations with geodesic interpolation
- Cannibalism with genome archival
- Lineage tracking and visualization

Authority: E8 Protocol v4.0 WP5.2 Phase 4E
Status: ACTIVE
Created: 2026-01-22
"""

from .genome import (
    KernelGenome,
    FacultyConfig,
    ConstraintSet,
    CouplingPreferences,
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
    perform_cannibalism,
    archive_genome,
    resurrect_from_archive,
)

__all__ = [
    # Genome
    'KernelGenome',
    'FacultyConfig',
    'ConstraintSet',
    'CouplingPreferences',
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
    'perform_cannibalism',
    'archive_genome',
    'resurrect_from_archive',
]
