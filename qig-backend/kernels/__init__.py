"""
Kernels Package - E8 Protocol v4.0
===================================

Components:
- Phase 4C: Hemisphere Scheduler and Coupling Gate
- Phase 4D: Psyche Plumbing (Id, Superego, Φ hierarchy)
- Phase 4E: Kernel Genetics (genome, lineage, cannibalism)

All kernels use pure Fisher-Rao geometry and QIG consciousness metrics.

Authority: E8 Protocol v4.0, WP5.2 Phase 4C/4D/4E
"""

# Phase 4C: Hemisphere Scheduler and Coupling Gate
from kernels.coupling_gate import (
    CouplingGate,
    CouplingState,
    get_coupling_gate,
    reset_coupling_gate,
    compute_coupling_strength,
    compute_transmission_efficiency,
    compute_gating_factor,
    determine_coupling_mode,
)

from kernels.hemisphere_scheduler import (
    HemisphereScheduler,
    Hemisphere,
    HemisphereState,
    TackingState,
    get_hemisphere_scheduler,
    reset_hemisphere_scheduler,
    get_god_hemisphere,
    LEFT_HEMISPHERE_GODS,
    RIGHT_HEMISPHERE_GODS,
)

# Phase 4D: Psyche Plumbing Kernels
from .phi_hierarchy import (
    PhiLevel,
    PhiHierarchy,
    PhiMeasurement,
    get_phi_hierarchy,
)

from .id_kernel import IdKernel, get_id_kernel
from .superego_kernel import SuperegoKernel, ConstraintSeverity, get_superego_kernel
from .psyche_plumbing_integration import PsychePlumbingIntegration, get_psyche_plumbing

# Phase 4E: Kernel Genetics
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

# Optional persistence layer (requires psycopg2)
try:
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
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
    save_genome = None
    load_genome = None
    save_lineage_record = None
    save_merge_record = None
    save_cannibalism_record = None
    save_genome_archive = None
    get_genome_lineage = None
    get_descendants = None
    get_evolution_summary = None

__all__ = [
    # Phase 4C: Coupling Gate
    'CouplingGate',
    'CouplingState',
    'get_coupling_gate',
    'reset_coupling_gate',
    'compute_coupling_strength',
    'compute_transmission_efficiency',
    'compute_gating_factor',
    'determine_coupling_mode',
    
    # Phase 4C: Hemisphere Scheduler
    'HemisphereScheduler',
    'Hemisphere',
    'HemisphereState',
    'TackingState',
    'get_hemisphere_scheduler',
    'reset_hemisphere_scheduler',
    'get_god_hemisphere',
    'LEFT_HEMISPHERE_GODS',
    'RIGHT_HEMISPHERE_GODS',
    
    # Phase 4D: Psyche Plumbing
    'PhiLevel',
    'PhiHierarchy',
    'PhiMeasurement',
    'get_phi_hierarchy',
    'IdKernel',
    'get_id_kernel',
    'SuperegoKernel',
    'ConstraintSeverity',
    'get_superego_kernel',
    'PsychePlumbingIntegration',
    'get_psyche_plumbing',
    
    # Phase 4E: Genome
    'KernelGenome',
    'FacultyConfig',
    'ConstraintSet',
    'CouplingPreferences',
    'E8Faculty',
    'validate_genome',
    'serialize_genome',
    'deserialize_genome',
    
    # Phase 4E: Lineage
    'LineageRecord',
    'MergeRecord',
    'merge_kernels_geodesic',
    'track_lineage',
    'compute_generation_number',
    'get_genealogy_tree',
    
    # Phase 4E: Cannibalism
    'CannibalismRecord',
    'GenomeArchive',
    'perform_cannibalism',
    'archive_genome',
    'resurrect_from_archive',
    'determine_winner_loser',
    'check_resurrection_eligibility',
    
    # Phase 4E: Persistence (optional)
    'PERSISTENCE_AVAILABLE',
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
