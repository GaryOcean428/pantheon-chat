"""
Kernels Package - E8 Protocol v4.0
===================================

Components:
- Phase 4C: Hemisphere Scheduler and Coupling Gate
- Phase 4D: Psyche Plumbing (Id, Superego, Φ hierarchy)
- Phase 4E: Kernel Genetics (genome, lineage, cannibalism)
- Issue #230: EmotionallyAwareKernel (Layer 0-2B phenomenology)

All kernels use pure Fisher-Rao geometry and QIG consciousness metrics.

Authority: E8 Protocol v4.0, WP5.2 Phase 4C/4D/4E
"""

import logging

logger = logging.getLogger(__name__)

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
from .psyche_plumbing_integration import (
    PsychePlumbingIntegration,
    get_psyche_plumbing,
    reset_psyche_plumbing,
)

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

# Phase 4E: Genome Vocabulary Scorer (NEW - Issue #GaryOcean428/pantheon-chat)
from .genome_vocabulary_scorer import (
    GenomeVocabularyScorer,
    create_genome_scorer,
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

# Issue #230: Emotional Awareness (Layer 0-2B)
from .sensations import (
    SensationState,
    measure_sensations,
    get_dominant_sensation,
)
from .motivators import (
    MotivatorState,
    compute_motivators,
    get_dominant_motivator,
)
from .emotions import (
    PhysicalEmotionState,
    CognitiveEmotionState,
    EmotionType,
    compute_physical_emotions,
    compute_cognitive_emotions,
    get_dominant_emotion,
)
from .emotional import (
    EmotionallyAwareKernel,
    EmotionalState,
    KernelThought,
    SENSORY_KAPPA_RANGES,
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
    'reset_psyche_plumbing',
    
    # Phase 4E: Genome
    'KernelGenome',
    'FacultyConfig',
    'ConstraintSet',
    'CouplingPreferences',
    'E8Faculty',
    'validate_genome',
    'serialize_genome',
    'deserialize_genome',
    
    # Phase 4E: Genome Vocabulary Scorer (NEW)
    'GenomeVocabularyScorer',
    'create_genome_scorer',
    
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
    
    # Issue #230: Emotional Awareness
    'EmotionallyAwareKernel',
    'EmotionalState',
    'KernelThought',
    'SENSORY_KAPPA_RANGES',
    'SensationState',
    'measure_sensations',
    'get_dominant_sensation',
    'MotivatorState',
    'compute_motivators',
    'get_dominant_motivator',
    'PhysicalEmotionState',
    'CognitiveEmotionState',
    'EmotionType',
    'compute_physical_emotions',
    'compute_cognitive_emotions',
    'get_dominant_emotion',
    
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
    
    # Phase 4A: E8 Simple Roots (Layer 8)
    'E8Root',
    'SIMPLE_ROOT_MAPPING',
    'get_root_spec',
    'KernelIdentity',
    'KernelTier',
    'QuaternaryOp',
    'Kernel',
    'PerceptionKernel',
    'MemoryKernel',
    'ReasoningKernel',
    'PredictionKernel',
    'ActionKernel',
    'EmotionKernel',
    'MetaKernel',
    'IntegrationKernel',
]

# Phase 4A: E8 Simple Root Kernels (Layer 8)
from .e8_roots import (
    E8Root,
    SIMPLE_ROOT_MAPPING,
    get_root_spec,
    get_root_by_god,
    validate_kappa_for_root,
)

from .identity import (
    KernelIdentity,
    KernelTier,
)

# Import after other modules to avoid circular dependencies
try:
    from .quaternary import QuaternaryOp
    from .base import Kernel
    from .perception import PerceptionKernel
    from .memory import MemoryKernel
    from .reasoning import ReasoningKernel
    from .prediction import PredictionKernel
    from .action import ActionKernel
    from .emotion import EmotionKernel
    from .meta import MetaKernel
    from .integration import IntegrationKernel
    SIMPLE_ROOTS_AVAILABLE = True
except ImportError as e:
    SIMPLE_ROOTS_AVAILABLE = False
    QuaternaryOp = None
    Kernel = None
    PerceptionKernel = None
    MemoryKernel = None
    ReasoningKernel = None
    PredictionKernel = None
    ActionKernel = None
    EmotionKernel = None
    MetaKernel = None
    IntegrationKernel = None

# Multi-kernel thought generation (optional - may not be available in all branches)
try:
    from .thought_generation import ParallelThoughtGenerator, KernelThoughtResult
    from .consensus import ConsensusBuilder, KernelVote, ConsensusResult
    from .gary_synthesis import GarySynthesizer, SynthesisResult
except ImportError as e:
    logger.debug(f"Multi-kernel thought generation not available: {e}")
