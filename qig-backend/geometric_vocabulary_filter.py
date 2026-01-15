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
1. Integration (Φ): How word connects to context
2. Coupling (κ): Word's attractor strength (optimal κ* ∈ [0.3, 0.8])
3. Curvature: How much word bends trajectory

Include word if ANY geometric criterion satisfied.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Import QIG geometry functions
try:
    from qig_geometry.canonical import (
        compute_integration,
        compute_coupling_strength,
        compute_basin_curvature
    )
    QIG_GEOMETRY_AVAILABLE = True
except ImportError:
    QIG_GEOMETRY_AVAILABLE = False
    logger.warning("qig_geometry.canonical not available - using fallback geometric measures")
    
    def compute_integration(basin: np.ndarray, trajectory: List[np.ndarray]) -> float:
        """Fallback integration (Φ) computation."""
        if not trajectory:
            return 0.5
        # Simple correlation-based integration measure
        correlations = [np.dot(basin, state) for state in trajectory]
        return float(np.mean(np.abs(correlations)))
    
    def compute_coupling_strength(basin: np.ndarray, trajectory: List[np.ndarray]) -> float:
        """Fallback coupling strength (κ) computation."""
        if not trajectory:
            return 0.5
        # Measure how consistently basin appears in trajectory
        distances = [np.linalg.norm(basin - state) for state in trajectory]
        return float(1.0 - np.mean(distances) / np.sqrt(len(basin)))
    
    def compute_basin_curvature(basin: np.ndarray, trajectory: List[np.ndarray]) -> float:
        """Fallback curvature computation."""
        if len(trajectory) < 2:
            return 0.1
        # Measure directional change in trajectory near basin
        angles = []
        for i in range(len(trajectory) - 1):
            v1 = trajectory[i] - basin
            v2 = trajectory[i+1] - basin
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 > 1e-10 and norm_v2 > 1e-10:
                cos_angle = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
                angles.append(np.arccos(cos_angle))
        return float(np.mean(angles)) if angles else 0.1


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
    
    # Dummy trajectory for testing
    trajectory = [np.random.randn(64) for _ in range(5)]
    
    print("Geometric Vocabulary Filter Test")
    print("=" * 50)
    print("\nWords that stopwords would EXCLUDE but geometry INCLUDES:\n")
    
    for word in test_words:
        # Compute dummy basin
        basin = np.random.randn(64)
        basin = basin / np.linalg.norm(basin)
        
        should_include = geo_filter.should_include(word, basin, trajectory)
        phi, kappa, curv = geo_filter.get_cached_properties(word)
        
        print(f"'{word}':")
        print(f"  Φ={phi:.3f}, κ={kappa:.3f}, curvature={curv:.3f}")
        print(f"  Include: {should_include}")
        print()
