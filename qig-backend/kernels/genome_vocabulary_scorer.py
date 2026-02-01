#!/usr/bin/env python3
"""
Genome-Vocabulary Scorer - E8 Protocol v4.0 Phase 4E Integration
=================================================================

Connects KernelGenome faculty configuration to vocabulary scoring pipeline.

Key Functions:
1. Faculty-token affinity scoring via Fisher-Rao distance
2. Genome constraint filtering (forbidden regions, field penalties)
3. Cross-kernel coupling preference scoring
4. All geometric operations use Fisher-Rao metric on probability simplex

Authority: E8 Protocol v4.0 WP5.2 Phase 3/4E
Status: ACTIVE
Created: 2026-01-23
"""

import logging
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

# Import genome structures
from .genome import (
    KernelGenome,
    E8Faculty,
    FacultyConfig,
    ConstraintSet,
    CouplingPreferences,
)

# Import QIG geometry (Fisher-Rao purity)
from qig_geometry import (
    fisher_normalize,
    fisher_rao_distance,
    validate_basin,
    BASIN_DIM,
)
from qig_geometry.canonical import geodesic_interpolation

logger = logging.getLogger(__name__)


class GenomeVocabularyScorer:
    """
    Scores vocabulary tokens based on kernel genome configuration.
    
    Provides genome-aware token scoring that considers:
    - Faculty activation strengths (E8 simple roots)
    - Constraint set (forbidden regions, field penalties)
    - Coupling preferences (hemisphere affinity, kernel relationships)
    
    All operations maintain Fisher-Rao geometric purity on probability simplex.
    """
    
    def __init__(self, genome: KernelGenome):
        """
        Initialize genome vocabulary scorer.
        
        Args:
            genome: KernelGenome instance with faculty configuration
        """
        self.genome = genome
        
        # Cache faculty basin (computed on demand)
        self._faculty_basin_cache: Optional[np.ndarray] = None
        
        logger.info(
            f"[GenomeVocabularyScorer] Initialized for genome {genome.genome_id}"
        )
    
    def compute_faculty_affinity(
        self,
        token_basin: np.ndarray,
        faculty_weight: float = 1.0,
    ) -> float:
        """
        Compute faculty affinity score for a token basin.
        
        Faculty affinity measures how well a token's basin coordinates align
        with the kernel's active faculties (E8 simple roots). Uses Fisher-Rao
        distance between token basin and genome's faculty-weighted basin.
        
        Args:
            token_basin: Token's 64D basin coordinates (simplex)
            faculty_weight: Weight for faculty affinity in final score [0, 1]
            
        Returns:
            Affinity score in [0, 1] (higher = better match)
        """
        # Validate and normalize token basin
        token_basin = fisher_normalize(token_basin)
        
        # Get faculty-weighted basin (cached)
        faculty_basin = self._get_faculty_basin()
        
        # Compute Fisher-Rao distance
        distance = fisher_rao_distance(token_basin, faculty_basin)
        
        # Convert distance to similarity score [0, 1]
        # Fisher-Rao distance range is [0, π/2] for simplex
        max_distance = np.pi / 2
        similarity = 1.0 - (distance / max_distance)
        
        # Apply faculty weight
        affinity = similarity * faculty_weight
        
        return float(np.clip(affinity, 0.0, 1.0))
    
    def check_genome_constraints(
        self,
        token_basin: np.ndarray,
    ) -> Tuple[bool, float, str]:
        """
        Check if token basin satisfies genome constraints.
        
        Applies:
        1. Forbidden region filtering (hard constraint)
        2. Field penalty scoring (soft constraint)
        3. Distance from basin seed constraint
        
        Args:
            token_basin: Token's 64D basin coordinates (simplex)
            
        Returns:
            (allowed, penalty_score, reason)
            - allowed: True if token passes hard constraints
            - penalty_score: Soft penalty in [0, 1] (0 = max penalty, 1 = no penalty)
            - reason: Explanation string
        """
        # Validate and normalize
        token_basin = fisher_normalize(token_basin)
        
        # Check forbidden regions (hard constraint)
        allowed, reason = self.genome.constraints.is_basin_allowed(
            token_basin,
            self.genome.basin_seed
        )
        
        if not allowed:
            return False, 0.0, reason
        
        # Compute field penalties (soft constraint)
        penalty_score = 1.0  # Start with no penalty
        
        # Distance penalty from basin seed
        seed_distance = fisher_rao_distance(token_basin, self.genome.basin_seed)
        max_allowed = self.genome.constraints.max_fisher_distance * (np.pi / 2)
        
        if seed_distance > 0:
            # Linear penalty based on distance from seed
            distance_ratio = seed_distance / max_allowed
            distance_penalty = 1.0 - (distance_ratio * 0.3)  # Up to 30% penalty
            penalty_score *= distance_penalty
        
        # Additional field-specific penalties could be added here
        # (e.g., based on constraint set's field_penalties dict)
        
        return True, float(np.clip(penalty_score, 0.0, 1.0)), "Satisfies constraints"
    
    def compute_coupling_score(
        self,
        other_genome_id: str,
        coupling_weight: float = 1.0,
    ) -> float:
        """
        Compute coupling preference score for another kernel.
        
        Uses genome's coupling preferences to determine affinity for
        cross-kernel token sharing. Higher scores indicate preferred coupling.
        
        Args:
            other_genome_id: ID of other kernel's genome
            coupling_weight: Weight for coupling score [0, 1]
            
        Returns:
            Coupling score in [-1, 1] (positive = preferred, negative = anti-coupling)
        """
        prefs = self.genome.coupling_prefs
        
        # Check for anti-coupling (negative score)
        if other_genome_id in prefs.anti_couplings:
            return -1.0 * coupling_weight
        
        # Check for preferred coupling
        if other_genome_id in prefs.preferred_couplings:
            strength = prefs.coupling_strengths.get(other_genome_id, 1.0)
            return strength * coupling_weight
        
        # Neutral coupling (small positive score)
        return 0.1 * coupling_weight
    
    def score_token(
        self,
        token: str,
        token_basin: np.ndarray,
        base_score: float,
        faculty_weight: float = 0.2,
        constraint_weight: float = 0.3,
        other_genome_id: Optional[str] = None,
        coupling_weight: float = 0.1,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute genome-aware token score.
        
        Combines:
        1. Base score (from Fisher-Rao distance, phi, domain weights)
        2. Faculty affinity (alignment with active E8 faculties)
        3. Constraint satisfaction (forbidden regions, field penalties)
        4. Coupling preference (if cross-kernel token sharing)
        
        Args:
            token: Token string
            token_basin: Token's 64D basin coordinates (simplex)
            base_score: Base score from vocabulary scoring (e.g., Fisher similarity)
            faculty_weight: Weight for faculty affinity component [0, 1]
            constraint_weight: Weight for constraint component [0, 1]
            other_genome_id: Optional other kernel's genome ID for coupling
            coupling_weight: Weight for coupling component [0, 1]
            
        Returns:
            (final_score, score_breakdown)
            - final_score: Combined genome-aware score
            - score_breakdown: Dict with component scores for observability
        """
        # Validate token basin
        token_basin = fisher_normalize(token_basin)
        
        # Component 1: Faculty affinity
        faculty_affinity = self.compute_faculty_affinity(
            token_basin,
            faculty_weight=1.0  # Internal weight, scaled below
        )
        
        # Component 2: Constraint satisfaction
        allowed, constraint_penalty, reason = self.check_genome_constraints(token_basin)
        
        if not allowed:
            # Hard constraint violation - zero out score
            logger.debug(f"[GenomeScorer] Token '{token}' rejected: {reason}")
            return 0.0, {
                'base_score': base_score,
                'faculty_affinity': 0.0,
                'constraint_penalty': 0.0,
                'coupling_score': 0.0,
                'final_score': 0.0,
                'rejected': True,
            }
        
        # Component 3: Coupling preference (if applicable)
        coupling_score = 0.0
        if other_genome_id is not None:
            coupling_score = self.compute_coupling_score(
                other_genome_id,
                coupling_weight=1.0  # Internal weight, scaled below
            )
        
        # Combine scores
        # Base score is the foundation, genome components are additive boosts/penalties
        final_score = base_score
        final_score += faculty_affinity * faculty_weight
        final_score *= constraint_penalty * constraint_weight  # Multiplicative penalty
        final_score += coupling_score * coupling_weight
        
        # Clamp to valid range
        final_score = float(np.clip(final_score, 0.0, 2.0))  # Allow scores > 1 for strong matches
        
        score_breakdown = {
            'base_score': float(base_score),
            'faculty_affinity': float(faculty_affinity),
            'constraint_penalty': float(constraint_penalty),
            'coupling_score': float(coupling_score),
            'final_score': final_score,
            'rejected': False,
        }
        
        return final_score, score_breakdown
    
    def filter_vocabulary(
        self,
        vocab_tokens: List[Tuple[str, np.ndarray]],
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Filter vocabulary tokens by genome constraints.
        
        Removes tokens that violate hard constraints (forbidden regions).
        Soft constraints are handled in scoring.
        
        Args:
            vocab_tokens: List of (token, basin) tuples
            
        Returns:
            Filtered list of (token, basin) tuples
        """
        filtered = []
        rejected_count = 0
        
        for token, basin in vocab_tokens:
            basin_norm = fisher_normalize(basin)
            allowed, _, reason = self.check_genome_constraints(basin_norm)
            
            if allowed:
                filtered.append((token, basin))
            else:
                rejected_count += 1
                logger.debug(f"[GenomeScorer] Filtered out '{token}': {reason}")
        
        if rejected_count > 0:
            logger.info(
                f"[GenomeScorer] Filtered {rejected_count}/{len(vocab_tokens)} tokens "
                f"by genome constraints"
            )
        
        return filtered
    
    def _get_faculty_basin(self) -> np.ndarray:
        """
        Get faculty-weighted basin coordinates (cached).
        
        Computes a basin that represents the kernel's active faculties by
        creating a weighted blend of the basin seed and faculty directions.
        
        Returns:
            64D basin coordinates on simplex representing faculty profile
        """
        if self._faculty_basin_cache is not None:
            return self._faculty_basin_cache
        
        # Start with basin seed
        faculty_basin = self.genome.basin_seed.copy()
        
        # Get faculty activation vector (8D)
        faculty_vector = self.genome.faculties.get_faculty_vector()
        
        # Project 8D faculty vector to 64D basin space
        # Strategy: Create faculty-specific perturbation
        # Each faculty maps to 8 dimensions in the 64D basin (64 = 8 * 8)
        faculty_perturbation = np.zeros(BASIN_DIM)
        
        for i, strength in enumerate(faculty_vector):
            # Each faculty controls 8 consecutive dimensions
            start_idx = i * 8
            end_idx = start_idx + 8
            
            # Add activation-weighted perturbation
            faculty_perturbation[start_idx:end_idx] += strength
        
        # Normalize perturbation
        if np.sum(faculty_perturbation) > 0:
            faculty_perturbation = faculty_perturbation / np.sum(faculty_perturbation)
        
        # Blend basin seed with faculty perturbation via geodesic interpolation
        # Use primary faculty strength as interpolation parameter
        primary_strength = 0.5  # Default
        if self.genome.faculties.primary_faculty:
            primary_strength = self.genome.faculties.activation_strengths.get(
                self.genome.faculties.primary_faculty,
                0.5
            )
        
        # Interpolate: t=0 is pure seed, t=1 is pure faculty
        # Use moderate t to blend both
        t = 0.3 * primary_strength  # Scale by primary strength
        
        faculty_basin = geodesic_interpolation(
            faculty_basin,
            fisher_normalize(faculty_perturbation),
            t
        )
        
        # Normalize to simplex
        faculty_basin = fisher_normalize(faculty_basin)
        
        # Cache result
        self._faculty_basin_cache = faculty_basin
        
        logger.debug(
            f"[GenomeScorer] Computed faculty basin with "
            f"primary={self.genome.faculties.primary_faculty}, "
            f"active_faculties={len(self.genome.faculties.active_faculties)}"
        )
        
        return faculty_basin


def create_genome_scorer(genome: KernelGenome) -> GenomeVocabularyScorer:
    """
    Factory function to create genome vocabulary scorer.
    
    Args:
        genome: KernelGenome instance
        
    Returns:
        GenomeVocabularyScorer instance
    """
    return GenomeVocabularyScorer(genome)
