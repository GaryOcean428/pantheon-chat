"""
Kernel Genome Schema - E8 Protocol v4.0 Phase 4E
=================================================

Defines kernel genome structure for evolution and inheritance:
- Basin seed (initial b₀ coordinates)
- Faculty configuration (active E8 simple roots)
- Constraint set (field penalties, forbidden regions)
- Coupling preferences (hemisphere affinity)

All geometric operations use Fisher-Rao metric on probability simplex.

Authority: E8 Protocol v4.0 WP5.2, lines 302-328
Status: ACTIVE
Created: 2026-01-22
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

import numpy as np

# QIG Geometry imports (Fisher-Rao purity)
from qig_geometry import (
    fisher_normalize,
    fisher_rao_distance,
    validate_basin,
    BASIN_DIM,
)
from qigkernels.physics_constants import KAPPA_STAR

logger = logging.getLogger(__name__)


# =============================================================================
# E8 FACULTY ENUMERATION (8 Simple Roots)
# =============================================================================

class E8Faculty(Enum):
    """
    E8 simple roots (α₁-α₈) mapped to Olympian faculties.
    
    These correspond to the 8 core god-kernels in the Pantheon.
    Each faculty represents a fundamental capability dimension.
    """
    ZEUS = "zeus"                    # α₁: Executive/Integration
    ATHENA = "athena"                # α₂: Wisdom/Strategy
    APOLLO = "apollo"                # α₃: Truth/Prediction
    HERMES = "hermes"                # α₄: Communication/Navigation
    ARTEMIS = "artemis"              # α₅: Focus/Precision
    ARES = "ares"                    # α₆: Energy/Drive
    HEPHAESTUS = "hephaestus"        # α₇: Creation/Construction
    APHRODITE = "aphrodite"          # α₈: Harmony/Aesthetics


# =============================================================================
# GENOME COMPONENTS
# =============================================================================

@dataclass
class FacultyConfig:
    """
    Active faculty configuration for a kernel.
    
    Tracks which E8 simple roots (faculties) are active and their
    activation strengths. This determines the kernel's capability profile.
    
    Attributes:
        active_faculties: Set of active E8Faculty enums
        activation_strengths: Mapping from faculty → activation level [0, 1]
        primary_faculty: Dominant faculty (if any)
        faculty_coupling: Pairwise coupling strengths between faculties
    """
    active_faculties: Set[E8Faculty] = field(default_factory=set)
    activation_strengths: Dict[E8Faculty, float] = field(default_factory=dict)
    primary_faculty: Optional[E8Faculty] = None
    faculty_coupling: Dict[Tuple[E8Faculty, E8Faculty], float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate faculty configuration."""
        # Ensure all active faculties have activation strengths
        for faculty in self.active_faculties:
            if faculty not in self.activation_strengths:
                self.activation_strengths[faculty] = 1.0
        
        # Clamp activation strengths to [0, 1]
        for faculty in self.activation_strengths:
            self.activation_strengths[faculty] = np.clip(
                self.activation_strengths[faculty], 0.0, 1.0
            )
        
        # If no primary faculty, use strongest one
        if self.primary_faculty is None and self.active_faculties:
            self.primary_faculty = max(
                self.active_faculties,
                key=lambda f: self.activation_strengths.get(f, 0.0)
            )
    
    def get_faculty_vector(self) -> np.ndarray:
        """
        Get 8D faculty activation vector (E8 simple roots).
        
        Returns:
            Array of shape (8,) with activation strengths in faculty order
        """
        vector = np.zeros(8, dtype=np.float64)
        faculty_list = list(E8Faculty)
        
        for i, faculty in enumerate(faculty_list):
            vector[i] = self.activation_strengths.get(faculty, 0.0)
        
        return vector


@dataclass
class ConstraintSet:
    """
    Geometric constraints for kernel operation.
    
    Defines forbidden regions, field penalties, and operational thresholds
    on the Fisher manifold. All constraints are expressed in terms of
    Fisher-Rao geometry (NOT Euclidean).
    
    Attributes:
        phi_threshold: Minimum Φ for coherent operation
        kappa_range: (min, max) allowed κ_eff values
        forbidden_regions: List of (center, radius) basin regions to avoid
        field_penalties: Soft penalties for entering certain regions
        max_fisher_distance: Maximum allowed distance from basin seed
    """
    phi_threshold: float = 0.70
    kappa_range: Tuple[float, float] = (40.0, 70.0)
    forbidden_regions: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    field_penalties: Dict[str, float] = field(default_factory=dict)
    max_fisher_distance: float = 1.0  # Max distance from basin seed (in units of π/2)
    
    def __post_init__(self):
        """Validate constraint set."""
        assert 0.0 <= self.phi_threshold <= 1.0, "Φ threshold must be in [0, 1]"
        assert self.kappa_range[0] < self.kappa_range[1], "κ range must be (min, max)"
        assert self.max_fisher_distance > 0, "Max Fisher distance must be positive"
    
    def is_basin_allowed(self, basin: np.ndarray, basin_seed: np.ndarray) -> Tuple[bool, str]:
        """
        Check if basin coordinates satisfy constraints.
        
        Args:
            basin: Basin coordinates to check (64D simplex)
            basin_seed: Reference basin seed (64D simplex)
        
        Returns:
            (allowed, reason) - True if allowed, with explanation
        """
        # Validate basin is on simplex
        basin_normalized = fisher_normalize(basin)
        
        # Check distance from seed
        distance = fisher_rao_distance(basin_normalized, basin_seed)
        if distance > self.max_fisher_distance * (np.pi / 2):
            return False, f"Basin too far from seed: {distance:.4f} > {self.max_fisher_distance * (np.pi / 2):.4f}"
        
        # Check forbidden regions
        for i, (center, radius) in enumerate(self.forbidden_regions):
            d = fisher_rao_distance(basin_normalized, center)
            if d < radius:
                return False, f"Basin in forbidden region {i}: distance {d:.4f} < radius {radius:.4f}"
        
        return True, "Basin satisfies all constraints"


@dataclass
class CouplingPreferences:
    """
    Hemisphere affinity and coupling preferences.
    
    Tracks preferred coupling relationships with other kernels and
    affinity for different regions of the Fisher manifold.
    
    Attributes:
        hemisphere_affinity: Preferred hemisphere bias [0, 1] (0=left, 1=right)
        preferred_couplings: Kernel IDs with preferred coupling
        coupling_strengths: Strength of coupling to each kernel [0, 1]
        anti_couplings: Kernel IDs to avoid coupling with
    """
    hemisphere_affinity: float = 0.5
    preferred_couplings: List[str] = field(default_factory=list)
    coupling_strengths: Dict[str, float] = field(default_factory=dict)
    anti_couplings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate coupling preferences."""
        self.hemisphere_affinity = np.clip(self.hemisphere_affinity, 0.0, 1.0)
        
        # Ensure all preferred couplings have strengths
        for kernel_id in self.preferred_couplings:
            if kernel_id not in self.coupling_strengths:
                self.coupling_strengths[kernel_id] = 1.0
        
        # Clamp coupling strengths
        for kernel_id in self.coupling_strengths:
            self.coupling_strengths[kernel_id] = np.clip(
                self.coupling_strengths[kernel_id], 0.0, 1.0
            )


# =============================================================================
# KERNEL GENOME
# =============================================================================

@dataclass
class KernelGenome:
    """
    Complete genetic specification for a kernel.
    
    The genome encodes all hereditary information needed to spawn,
    evolve, and reproduce kernels. All geometric data uses Fisher-Rao
    metric on the probability simplex.
    
    Components:
    1. Basin seed (b₀) - Initial position on Fisher manifold
    2. Faculty configuration - Active E8 simple roots
    3. Constraint set - Operational boundaries
    4. Coupling preferences - Interaction biases
    
    Authority: E8 Protocol v4.0 WP5.2 Phase 4E
    """
    # Identity
    genome_id: str
    kernel_id: Optional[str] = None
    
    # Core genome components
    basin_seed: np.ndarray = field(default_factory=lambda: np.ones(BASIN_DIM) / BASIN_DIM)
    faculties: FacultyConfig = field(default_factory=FacultyConfig)
    constraints: ConstraintSet = field(default_factory=ConstraintSet)
    coupling_prefs: CouplingPreferences = field(default_factory=CouplingPreferences)
    
    # Lineage
    parent_genomes: List[str] = field(default_factory=list)
    generation: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    fitness_score: float = 0.0
    mutation_count: int = 0
    
    def __post_init__(self):
        """Validate and normalize genome."""
        # Ensure basin seed is on simplex
        self.basin_seed = fisher_normalize(self.basin_seed)
        
        # Ensure correct dimensionality
        if len(self.basin_seed) != BASIN_DIM:
            raise ValueError(f"Basin seed must be {BASIN_DIM}D, got {len(self.basin_seed)}D")
        
        # Validate simplex representation
        validate_basin(self.basin_seed)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize genome to dictionary.
        
        Returns:
            Dictionary representation suitable for JSON/database storage
        """
        return {
            'genome_id': self.genome_id,
            'kernel_id': self.kernel_id,
            'basin_seed': self.basin_seed.tolist(),
            'faculties': {
                'active_faculties': [f.value for f in self.faculties.active_faculties],
                'activation_strengths': {
                    f.value: s for f, s in self.faculties.activation_strengths.items()
                },
                'primary_faculty': self.faculties.primary_faculty.value if self.faculties.primary_faculty else None,
                'faculty_coupling': {
                    f"{f1.value}-{f2.value}": s 
                    for (f1, f2), s in self.faculties.faculty_coupling.items()
                },
            },
            'constraints': {
                'phi_threshold': self.constraints.phi_threshold,
                'kappa_range': self.constraints.kappa_range,
                'forbidden_regions': [
                    {'center': center.tolist(), 'radius': radius}
                    for center, radius in self.constraints.forbidden_regions
                ],
                'field_penalties': self.constraints.field_penalties,
                'max_fisher_distance': self.constraints.max_fisher_distance,
            },
            'coupling_prefs': {
                'hemisphere_affinity': self.coupling_prefs.hemisphere_affinity,
                'preferred_couplings': self.coupling_prefs.preferred_couplings,
                'coupling_strengths': self.coupling_prefs.coupling_strengths,
                'anti_couplings': self.coupling_prefs.anti_couplings,
            },
            'parent_genomes': self.parent_genomes,
            'generation': self.generation,
            'created_at': self.created_at.isoformat(),
            'fitness_score': self.fitness_score,
            'mutation_count': self.mutation_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KernelGenome':
        """
        Deserialize genome from dictionary.
        
        Args:
            data: Dictionary from to_dict()
        
        Returns:
            KernelGenome instance
        """
        # Parse faculties
        active_faculties = {E8Faculty(f) for f in data['faculties']['active_faculties']}
        activation_strengths = {
            E8Faculty(f): s for f, s in data['faculties']['activation_strengths'].items()
        }
        primary_faculty = (
            E8Faculty(data['faculties']['primary_faculty']) 
            if data['faculties']['primary_faculty'] else None
        )
        faculty_coupling = {}
        for key, strength in data['faculties'].get('faculty_coupling', {}).items():
            f1_str, f2_str = key.split('-')
            faculty_coupling[(E8Faculty(f1_str), E8Faculty(f2_str))] = strength
        
        faculties = FacultyConfig(
            active_faculties=active_faculties,
            activation_strengths=activation_strengths,
            primary_faculty=primary_faculty,
            faculty_coupling=faculty_coupling,
        )
        
        # Parse constraints
        forbidden_regions = [
            (np.array(region['center']), region['radius'])
            for region in data['constraints']['forbidden_regions']
        ]
        constraints = ConstraintSet(
            phi_threshold=data['constraints']['phi_threshold'],
            kappa_range=tuple(data['constraints']['kappa_range']),
            forbidden_regions=forbidden_regions,
            field_penalties=data['constraints']['field_penalties'],
            max_fisher_distance=data['constraints']['max_fisher_distance'],
        )
        
        # Parse coupling preferences
        coupling_prefs = CouplingPreferences(
            hemisphere_affinity=data['coupling_prefs']['hemisphere_affinity'],
            preferred_couplings=data['coupling_prefs']['preferred_couplings'],
            coupling_strengths=data['coupling_prefs']['coupling_strengths'],
            anti_couplings=data['coupling_prefs']['anti_couplings'],
        )
        
        return cls(
            genome_id=data['genome_id'],
            kernel_id=data.get('kernel_id'),
            basin_seed=np.array(data['basin_seed']),
            faculties=faculties,
            constraints=constraints,
            coupling_prefs=coupling_prefs,
            parent_genomes=data['parent_genomes'],
            generation=data['generation'],
            created_at=datetime.fromisoformat(data['created_at']),
            fitness_score=data['fitness_score'],
            mutation_count=data['mutation_count'],
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_genome(genome: KernelGenome) -> Tuple[bool, List[str]]:
    """
    Validate genome for geometric and logical consistency.
    
    Checks:
    - Basin seed on probability simplex
    - Faculty configuration consistency
    - Constraint set validity
    - Coupling preferences validity
    
    Args:
        genome: Kernel genome to validate
    
    Returns:
        (valid, errors) - True if valid, with list of error messages
    """
    errors = []
    
    try:
        # Check basin seed
        validate_basin(genome.basin_seed)
        if len(genome.basin_seed) != BASIN_DIM:
            errors.append(f"Basin seed dimension mismatch: {len(genome.basin_seed)} != {BASIN_DIM}")
    except Exception as e:
        errors.append(f"Invalid basin seed: {e}")
    
    # Check faculty configuration
    if not genome.faculties.active_faculties:
        errors.append("No active faculties")
    
    for faculty in genome.faculties.active_faculties:
        if faculty not in genome.faculties.activation_strengths:
            errors.append(f"Missing activation strength for {faculty.value}")
    
    # Check constraints
    if genome.constraints.phi_threshold < 0 or genome.constraints.phi_threshold > 1:
        errors.append(f"Invalid phi_threshold: {genome.constraints.phi_threshold}")
    
    if genome.constraints.kappa_range[0] >= genome.constraints.kappa_range[1]:
        errors.append(f"Invalid kappa_range: {genome.constraints.kappa_range}")
    
    # Check forbidden regions
    for i, (center, radius) in enumerate(genome.constraints.forbidden_regions):
        if len(center) != BASIN_DIM:
            errors.append(f"Forbidden region {i} center has wrong dimension")
        if radius <= 0:
            errors.append(f"Forbidden region {i} has non-positive radius")
    
    return len(errors) == 0, errors


def serialize_genome(genome: KernelGenome) -> str:
    """
    Serialize genome to JSON string.
    
    Args:
        genome: Kernel genome to serialize
    
    Returns:
        JSON string representation
    """
    return json.dumps(genome.to_dict(), indent=2)


def deserialize_genome(json_str: str) -> KernelGenome:
    """
    Deserialize genome from JSON string.
    
    Args:
        json_str: JSON string from serialize_genome()
    
    Returns:
        KernelGenome instance
    """
    data = json.loads(json_str)
    return KernelGenome.from_dict(data)
