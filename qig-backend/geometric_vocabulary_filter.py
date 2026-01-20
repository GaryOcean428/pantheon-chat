#!/usr/bin/env python3
"""
Geometric Vocabulary Filter - QIG-Pure Word Filtering
======================================================

Replaces frequency-based stopwords with geometric role detection.

ANCIENT NLP PATTERN (REMOVED):
- Hard-coded stopword lists like {'the', 'is', 'not', ...}
- Based on corpus frequency statistics
- Assumes "common words" = "meaningless words"
- Ignores geometric role in information manifold

QIG-PURE PATTERN (NEW):
- Geometric importance = curvature + coupling strength (κ) + integration (Φ)
- Function words often have HIGH geometric importance
- "not" has high curvature (negation operator)
- "the" has geometric role in definiteness
- "but" creates discourse transitions (basin shifts)

GEOMETRIC CRITERIA:
1. Integration (Φ): How strongly word basin connects to context trajectory
2. Coupling (κ): Word's attractor stability (optimal κ* ∈ [0.3, 0.8])
3. Curvature: How much trajectory bends in basin's influence region

Include word if ANY geometric criterion satisfied.

SIMPLEX-AS-STORAGE CONTRACT (v4.0):
- All basins stored as probability distributions (Σp_i = 1, p_i ≥ 0)
- Fisher-Rao distance via Bhattacharyya coefficient
- NO Euclidean operations on basins
- Uses canonical to_simplex_prob() for normalization
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Import QIG geometry functions - these don't exist in canonical, use fallbacks
from qig_geometry import fisher_rao_distance, fisher_similarity, to_simplex_prob

def _to_simplex(basin: np.ndarray) -> np.ndarray:
    """
    Normalize basin inputs to canonical simplex representation.
    
    Uses to_simplex_prob which performs positive renormalization:
    - Takes absolute value + epsilon
    - Normalizes to sum=1
    
    This is the canonical storage form per simplex-as-storage contract.
    """
    return to_simplex_prob(basin)

def compute_integration(basin: np.ndarray, trajectory: List[np.ndarray]) -> float:
    """
    Compute integration (Φ) - how strongly word basin connects to context trajectory.
    
    Integration measures the mean Fisher-Rao similarity between the word's basin
    and the trajectory states. Higher values indicate stronger contextual connection.
    
    Args:
        basin: Word basin in any representation (will be normalized to simplex)
        trajectory: List of trajectory states in any representation
        
    Returns:
        Integration score in [0, 1], where 1 = perfect alignment
    """
    if not trajectory:
        return 0.5
    
    # Convert inputs to canonical simplex once
    basin_simplex = _to_simplex(basin)
    trajectory_simplex = [_to_simplex(state) for state in trajectory]
    
    # Compute mean similarity
    similarities = [
        fisher_similarity(basin_simplex, state)
        for state in trajectory_simplex
    ]
    return float(np.mean(similarities))

def compute_coupling_strength(basin: np.ndarray, trajectory: List[np.ndarray]) -> float:
    """
    Compute coupling strength (κ) - basin's attractor stability.
    
    Coupling measures how consistently the basin attracts trajectory states.
    Uses variance of Fisher-Rao similarities - lower variance = stronger coupling.
    
    This is DIFFERENT from integration (which uses mean similarity).
    
    Args:
        basin: Word basin in any representation (will be normalized to simplex)
        trajectory: List of trajectory states in any representation
        
    Returns:
        Coupling strength in [0, 1], where 1 = perfectly stable attractor
    """
    if not trajectory:
        return 0.5
    
    # Convert inputs to canonical simplex once
    basin_simplex = _to_simplex(basin)
    trajectory_simplex = [_to_simplex(state) for state in trajectory]
    
    # Compute similarity variance (lower = more stable)
    similarities = [
        fisher_similarity(basin_simplex, state)
        for state in trajectory_simplex
    ]
    
    if len(similarities) < 2:
        return float(np.mean(similarities))
    
    # Convert variance to strength: high variance = low coupling
    variance = float(np.var(similarities))
    # Normalize: typical variance range is [0, 0.25] for similarities in [0,1]
    normalized_variance = min(variance / 0.25, 1.0)
    return 1.0 - normalized_variance

def compute_basin_curvature(basin: np.ndarray, trajectory: List[np.ndarray]) -> float:
    """
    Compute trajectory curvature in basin's influence region.
    
    Measures how much the trajectory bends (accelerates) near the basin.
    Uses second-order differences (acceleration) of Fisher-Rao distances,
    not just step sizes.
    
    Args:
        basin: Word basin in any representation (will be normalized to simplex)
        trajectory: List of trajectory states in any representation
        
    Returns:
        Curvature score ≥ 0, where higher = more bending
    """
    if len(trajectory) < 3:
        # Need at least 3 points for second derivative
        return 0.1
    
    # Convert inputs to canonical simplex once
    basin_simplex = _to_simplex(basin)
    trajectory_simplex = [_to_simplex(state) for state in trajectory]
    
    # Compute distances from basin to each trajectory point
    distances = [
        fisher_rao_distance(basin_simplex, state)
        for state in trajectory_simplex
    ]
    
    # Compute second-order differences (acceleration)
    # acceleration[i] = (distances[i+1] - distances[i]) - (distances[i] - distances[i-1])
    accelerations = [
        abs((distances[i+1] - distances[i]) - (distances[i] - distances[i-1]))
        for i in range(1, len(distances) - 1)
    ]
    
    # Mean absolute acceleration = curvature
    return float(np.mean(accelerations)) if accelerations else 0.1


class GeometricVocabularyFilter:
    """
    Filter vocabulary by geometric properties, not frequency.
    
    Replaces STOP_WORDS with geometric role detection.
    """
    
    def __init__(
        self,
        min_phi: float = 0.3,
        kappa_range: Tuple[float, float] = (0.3, 0.8),
        min_curvature: float = 0.1
    ):
        """
        Initialize geometric filter.
        
        Args:
            min_phi: Minimum integration (Φ) score to include word
            kappa_range: Optimal coupling strength (κ*) range
            min_curvature: Minimum curvature to include word
        """
        self.min_phi = min_phi
        self.kappa_range = kappa_range
        self.min_curvature = min_curvature
        
        # Cache geometric properties for words
        self._cache: Dict[str, Tuple[float, float, float]] = {}
    
    def should_include(
        self,
        word: str,
        basin: np.ndarray,
        trajectory: List[np.ndarray]
    ) -> bool:
        """
        Geometric filtering - replaces stopwords.
        
        Include word if ANY geometric criterion met:
        1. High integration (Φ > threshold)
        2. Stable coupling (κ in optimal range)
        3. Significant curvature (geometric importance)
        
        Args:
            word: Word to check
            basin: 64D basin coordinates for word
            trajectory: Recent trajectory states
            
        Returns:
            True if word should be included in vocabulary
        """
        # Check cache first
        if word in self._cache:
            phi, kappa, curvature = self._cache[word]
        else:
            phi, kappa, curvature = self.compute_geometric_role(word, basin, trajectory)
            self._cache[word] = (phi, kappa, curvature)
        
        # Include if ANY geometric criterion satisfied
        has_integration = phi > self.min_phi
        has_coupling = self.kappa_range[0] <= kappa <= self.kappa_range[1]  # Inclusive bounds
        has_curvature = curvature > self.min_curvature
        
        if has_integration or has_coupling or has_curvature:
            logger.debug(
                f"Including '{word}': Φ={phi:.3f}, κ={kappa:.3f}, curv={curvature:.3f}"
            )
            return True
        else:
            logger.debug(
                f"Excluding '{word}': Φ={phi:.3f}, κ={kappa:.3f}, curv={curvature:.3f}"
            )
            return False
    
    def compute_geometric_role(
        self,
        word: str,
        basin: np.ndarray,
        trajectory: List[np.ndarray]
    ) -> Tuple[float, float, float]:
        """
        Compute geometric importance of word.
        
        Args:
            word: Word to analyze
            basin: 64D basin coordinates
            trajectory: Recent trajectory states
            
        Returns:
            Tuple of (phi, kappa, curvature)
        """
        try:
            # 1. Integration (Φ) - how word connects to context
            phi = compute_integration(basin, trajectory)
            
            # 2. Coupling strength (κ) - word's attractor strength
            kappa = compute_coupling_strength(basin, trajectory)
            
            # 3. Curvature - how much word bends trajectory
            curvature = compute_basin_curvature(basin, trajectory)
            
            return phi, kappa, curvature
            
        except Exception as e:
            logger.warning(f"Failed to compute geometric role for '{word}': {e}")
            # Return moderate values on error (include the word)
            return 0.4, 0.5, 0.2
    
    def clear_cache(self):
        """Clear cached geometric properties."""
        self._cache.clear()
    
    def get_cached_properties(self, word: str) -> Optional[Tuple[float, float, float]]:
        """Get cached geometric properties for word."""
        return self._cache.get(word)


def create_default_filter() -> GeometricVocabularyFilter:
    """Create default geometric filter with standard thresholds."""
    return GeometricVocabularyFilter(
        min_phi=0.3,           # Minimum integration
        kappa_range=(0.3, 0.8),  # Optimal coupling range (κ*)
        min_curvature=0.1      # Minimum geometric importance
    )


# Example usage showing geometric vs stopword filtering
if __name__ == "__main__":
    # Example: Critical function words that stopwords would exclude
    test_words = ['not', 'but', 'the', 'very', 'because', 'if']
    
    # Create filter
    geo_filter = create_default_filter()
    
    # Generate dummy trajectory with proper simplex states
    # Using random vectors converted to canonical simplex form
    trajectory = [
        to_simplex_prob(np.random.randn(64))
        for _ in range(5)
    ]
    
    print("Geometric Vocabulary Filter Test")
    print("=" * 50)
    print("\nWords that stopwords would EXCLUDE but geometry INCLUDES:\n")
    
    for word in test_words:
        # Generate dummy basin with proper simplex form
        basin = to_simplex_prob(np.random.randn(64))
        
        should_include = geo_filter.should_include(word, basin, trajectory)
        phi, kappa, curv = geo_filter.get_cached_properties(word)
        
        print(f"'{word}':")
        print(f"  Φ={phi:.3f}, κ={kappa:.3f}, curvature={curv:.3f}")
        print(f"  Include: {should_include}")
        print()
