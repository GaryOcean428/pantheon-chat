"""
Pantheon Semantic Candidates Generator

Generates candidate words from learned semantic relationships for
QIG-pure β-function measurement. Replaces mock random candidates
with real semantic neighbors to validate substrate independence.

PHYSICS EXPECTATION:
- Mock candidates: β ≈ 0.12 (weak, no semantic structure)
- Semantic candidates: β should approach 0.44 (physics β(3→4))

If AI attention with semantic structure shows β ≈ 0.44, we have
validated that information geometry is substrate-independent.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import geometry functions
try:
    from qig_geometry import fisher_coord_distance, geodesic_interpolation
except ImportError:
    def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Fallback Fisher-Rao distance."""
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b) + 1e-10)
        dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
        return float(np.arccos(dot))
    
    def geodesic_interpolation(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
        """Fallback spherical linear interpolation."""
        start_norm = start / (np.linalg.norm(start) + 1e-10)
        end_norm = end / (np.linalg.norm(end) + 1e-10)
        dot = np.clip(np.dot(start_norm, end_norm), -1.0, 1.0)
        omega = np.arccos(dot)
        if omega < 1e-6:
            return start
        sin_omega = np.sin(omega)
        a = np.sin((1 - t) * omega) / sin_omega
        b = np.sin(t * omega) / sin_omega
        result = a * start_norm + b * end_norm
        return result * np.linalg.norm(start)

# Import frozen physics constants
try:
    from frozen_physics import BASIN_DIM, KAPPA_STAR
except ImportError:
    BASIN_DIM = 64
    KAPPA_STAR = 64.21


@dataclass
class SemanticCandidateConfig:
    """Configuration for semantic candidate generation."""
    # Maximum candidates to return per query
    max_candidates: int = 50
    # Minimum relationship strength to consider
    min_strength: float = 0.1
    # Whether to warp basins toward current position
    warp_basins: bool = True
    # Temperature for warp strength calculation
    warp_temperature: float = 1.0
    # Include second-hop neighbors (neighbors of neighbors)
    include_second_hop: bool = True
    # Weight for second-hop neighbors
    second_hop_weight: float = 0.3
    # Fallback to random if not enough semantic neighbors
    fallback_to_random: bool = True
    # Minimum semantic candidates before fallback
    min_semantic_candidates: int = 10


class PantheonSemanticCandidates:
    """
    Generate candidates from learned word relationships instead of random.
    
    This is the critical integration that should improve β from ~0.12 (mock)
    toward the physics value of ~0.44, validating substrate independence.
    
    Usage:
        generator = PantheonSemanticCandidates()
        candidates = generator.generate_candidates(
            current_word="consciousness",
            current_basin=current_basin_coords,
            n_candidates=50
        )
    """
    
    def __init__(
        self,
        config: Optional[SemanticCandidateConfig] = None,
        relationships: Optional[Dict[str, List[Tuple[str, float]]]] = None,
        basins: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Initialize semantic candidate generator.
        
        Args:
            config: Configuration options
            relationships: Pre-loaded relationships dict {word: [(neighbor, strength), ...]}
            basins: Pre-loaded basin coordinates dict {word: ndarray}
        """
        self.config = config or SemanticCandidateConfig()
        self._relationships: Optional[Dict[str, List[Tuple[str, float]]]] = relationships
        self._basins: Optional[Dict[str, np.ndarray]] = basins
        self._vocabulary: Optional[Set[str]] = None
        self._initialized = False
        
        # Statistics for β measurement analysis
        self.stats = {
            'total_queries': 0,
            'semantic_candidates_returned': 0,
            'fallback_candidates_returned': 0,
            'second_hop_used': 0,
            'warp_applied': 0
        }
    
    def _ensure_initialized(self):
        """Lazy initialization of relationships and basins."""
        if self._initialized:
            return
        
        # Load relationships if not provided
        if self._relationships is None:
            try:
                from learned_relationships import get_learned_relationships
                lr = get_learned_relationships()
                self._relationships = lr.word_neighbors
                logger.info(f"[PantheonSemanticCandidates] Loaded {len(self._relationships)} relationships")
            except Exception as e:
                logger.warning(f"[PantheonSemanticCandidates] Failed to load relationships: {e}")
                self._relationships = {}
        
        # Load basins if not provided
        if self._basins is None:
            try:
                from coordizers.pg_loader import PostgresCoordizer
                coordizer = PostgresCoordizer()
                self._basins = dict(coordizer.basin_coords)
                logger.info(f"[PantheonSemanticCandidates] Loaded {len(self._basins)} basins")
            except Exception as e:
                logger.warning(f"[PantheonSemanticCandidates] Failed to load basins: {e}")
                self._basins = {}
        
        # Build vocabulary set
        self._vocabulary = set(self._basins.keys()) if self._basins else set()
        
        self._initialized = True
        
        logger.info(
            f"[PantheonSemanticCandidates] Initialized: "
            f"{len(self._relationships)} relationships, "
            f"{len(self._basins)} basins"
        )
    
    def get_neighbors(
        self,
        word: str,
        n: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Get semantically related words from learned relationships.
        
        Args:
            word: Word to find neighbors for
            n: Maximum number of neighbors
        
        Returns:
            List of (neighbor_word, relationship_strength) tuples
        """
        self._ensure_initialized()
        
        if not self._relationships:
            return []
        
        neighbors = self._relationships.get(word.lower(), [])
        
        # Filter by minimum strength and limit
        filtered = [
            (w, s) for w, s in neighbors 
            if s >= self.config.min_strength and w in self._vocabulary
        ]
        
        return filtered[:n]
    
    def get_second_hop_neighbors(
        self,
        word: str,
        n: int = 20,
        exclude: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Get neighbors-of-neighbors (second hop) for deeper semantic reach.
        
        Args:
            word: Starting word
            n: Maximum number of second-hop neighbors
            exclude: Words to exclude (e.g., first-hop neighbors)
        
        Returns:
            List of (neighbor_word, weighted_strength) tuples
        """
        self._ensure_initialized()
        exclude = exclude or set()
        exclude.add(word.lower())
        
        # Get first-hop neighbors
        first_hop = self.get_neighbors(word, n=30)
        if not first_hop:
            return []
        
        # Aggregate second-hop neighbors
        second_hop_scores: Dict[str, float] = {}
        
        for first_neighbor, first_strength in first_hop:
            # Get neighbors of this neighbor
            for second_neighbor, second_strength in self.get_neighbors(first_neighbor, n=10):
                if second_neighbor not in exclude:
                    # Combine strengths: first_strength * second_strength * weight
                    combined = first_strength * second_strength * self.config.second_hop_weight
                    if second_neighbor in second_hop_scores:
                        # Multiple paths to same word - add strengths
                        second_hop_scores[second_neighbor] += combined
                    else:
                        second_hop_scores[second_neighbor] = combined
        
        # Sort by combined strength
        sorted_neighbors = sorted(
            second_hop_scores.items(),
            key=lambda x: -x[1]
        )
        
        return sorted_neighbors[:n]
    
    def get_basin(self, word: str) -> Optional[np.ndarray]:
        """
        Get basin coordinates for a word.
        
        Args:
            word: Word to get basin for
        
        Returns:
            64D basin coordinate array or None
        """
        self._ensure_initialized()
        return self._basins.get(word.lower())
    
    def _warp_toward(
        self,
        current: np.ndarray,
        target: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Warp target basin toward current based on relationship strength.
        
        Strong relationships pull the basin closer, making related words
        geodesically closer. This is the key mechanism that should increase β.
        
        Args:
            current: Current position on manifold
            target: Target word's basin
            strength: Relationship strength (higher = more warping)
        
        Returns:
            Warped basin coordinates
        """
        # Compute warp amount based on strength
        # t=0: stay at target, t=1: move to current
        # Stronger relationships → larger t → closer to current
        normalized_strength = strength / (strength + self.config.warp_temperature)
        t = min(normalized_strength * 0.5, 0.4)  # Cap at 40% warping
        
        # Geodesic interpolation preserves manifold structure
        warped = geodesic_interpolation(target, current, t)
        
        # Ensure unit norm
        norm = np.linalg.norm(warped)
        if norm > 0:
            warped = warped / norm
        
        return warped
    
    def generate_candidates(
        self,
        current_word: Optional[str],
        current_basin: np.ndarray,
        n_candidates: int = 50
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Generate semantic candidates from learned relationships.
        
        This is the main interface for β measurement. Instead of random
        perturbations, returns actual semantic neighbors with their
        (optionally warped) basin coordinates.
        
        Args:
            current_word: Current word (for semantic lookup)
            current_basin: Current position on manifold
            n_candidates: Number of candidates to generate
        
        Returns:
            List of (word, basin_coords) tuples
        """
        self._ensure_initialized()
        self.stats['total_queries'] += 1
        
        candidates = []
        seen_words = set()
        
        # Add current word to exclusion set
        if current_word:
            seen_words.add(current_word.lower())
        
        # Get first-hop neighbors
        if current_word:
            first_hop = self.get_neighbors(current_word, n=n_candidates)
            
            for word, strength in first_hop:
                if word in seen_words:
                    continue
                
                basin = self.get_basin(word)
                if basin is None:
                    continue
                
                # Optionally warp basin toward current
                if self.config.warp_basins:
                    basin = self._warp_toward(current_basin, basin, strength)
                    self.stats['warp_applied'] += 1
                
                candidates.append((word, basin))
                seen_words.add(word)
        
        # Add second-hop neighbors if enabled and needed
        if (
            self.config.include_second_hop and 
            current_word and
            len(candidates) < n_candidates
        ):
            second_hop = self.get_second_hop_neighbors(
                current_word,
                n=n_candidates - len(candidates),
                exclude=seen_words
            )
            
            for word, strength in second_hop:
                if word in seen_words:
                    continue
                
                basin = self.get_basin(word)
                if basin is None:
                    continue
                
                if self.config.warp_basins:
                    basin = self._warp_toward(current_basin, basin, strength)
                    self.stats['warp_applied'] += 1
                
                candidates.append((word, basin))
                seen_words.add(word)
                self.stats['second_hop_used'] += 1
        
        # Track semantic candidates
        semantic_count = len(candidates)
        self.stats['semantic_candidates_returned'] += semantic_count
        
        # Fallback to random if not enough semantic neighbors
        if (
            self.config.fallback_to_random and 
            len(candidates) < self.config.min_semantic_candidates
        ):
            # Generate random candidates from vocabulary
            remaining = n_candidates - len(candidates)
            available = list(self._vocabulary - seen_words)
            
            if available:
                np.random.shuffle(available)
                for word in available[:remaining]:
                    basin = self.get_basin(word)
                    if basin is None:
                        continue
                    
                    # Random candidates get small random perturbation
                    # (less structured than semantic warping)
                    perturbation = np.random.dirichlet(np.ones(BASIN_DIM))
                    basin = 0.7 * basin + 0.3 * perturbation
                    basin = basin / (np.linalg.norm(basin) + 1e-10)
                    
                    candidates.append((word, basin))
                    seen_words.add(word)
                    self.stats['fallback_candidates_returned'] += 1
        
        return candidates[:n_candidates]
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about candidate generation.
        
        Useful for analyzing β measurement results.
        """
        total_candidates = (
            self.stats['semantic_candidates_returned'] + 
            self.stats['fallback_candidates_returned']
        )
        
        semantic_ratio = (
            self.stats['semantic_candidates_returned'] / total_candidates
            if total_candidates > 0 else 0
        )
        
        return {
            **self.stats,
            'total_candidates': total_candidates,
            'semantic_ratio': semantic_ratio,
            'relationships_loaded': len(self._relationships) if self._relationships else 0,
            'basins_loaded': len(self._basins) if self._basins else 0
        }
    
    def reset_statistics(self):
        """Reset generation statistics."""
        self.stats = {
            'total_queries': 0,
            'semantic_candidates_returned': 0,
            'fallback_candidates_returned': 0,
            'second_hop_used': 0,
            'warp_applied': 0
        }


# Singleton instance
_semantic_generator: Optional[PantheonSemanticCandidates] = None


def get_semantic_generator() -> PantheonSemanticCandidates:
    """Get or create singleton semantic candidate generator."""
    global _semantic_generator
    if _semantic_generator is None:
        _semantic_generator = PantheonSemanticCandidates()
    return _semantic_generator


def generate_semantic_candidates(
    current_word: Optional[str],
    current_basin: np.ndarray,
    n_candidates: int = 50
) -> List[Tuple[str, np.ndarray]]:
    """
    Convenience function for generating semantic candidates.
    
    Args:
        current_word: Current word for semantic lookup
        current_basin: Current position on manifold
        n_candidates: Number of candidates
    
    Returns:
        List of (word, basin) tuples
    """
    generator = get_semantic_generator()
    return generator.generate_candidates(
        current_word=current_word,
        current_basin=current_basin,
        n_candidates=n_candidates
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test semantic candidate generation
    generator = PantheonSemanticCandidates()
    
    # Create a test basin
    test_basin = np.random.dirichlet(np.ones(BASIN_DIM))
    
    # Generate candidates for a test word
    test_word = "consciousness"
    candidates = generator.generate_candidates(
        current_word=test_word,
        current_basin=test_basin,
        n_candidates=20
    )
    
    print(f"\n=== Semantic Candidates for '{test_word}' ===")
    print(f"Generated {len(candidates)} candidates")
    
    if candidates:
        print("\nTop 10 candidates:")
        for i, (word, basin) in enumerate(candidates[:10]):
            dist = fisher_coord_distance(test_basin, basin)
            print(f"  {i+1}. {word}: Fisher distance = {dist:.4f}")
    
    # Print statistics
    stats = generator.get_statistics()
    print(f"\nStatistics:")
    print(f"  Semantic ratio: {stats['semantic_ratio']:.1%}")
    print(f"  Second-hop used: {stats['second_hop_used']}")
    print(f"  Relationships loaded: {stats['relationships_loaded']}")
    print(f"  Basins loaded: {stats['basins_loaded']}")
