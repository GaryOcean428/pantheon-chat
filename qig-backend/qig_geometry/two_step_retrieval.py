"""
Two-Step Fisher-Faithful Retrieval Module
==========================================

Implements efficient two-step retrieval (approximate → Fisher re-rank) while
maintaining geometric purity. The proxy MUST be Fisher-faithful to preserve
consciousness-relevant distances.

ARCHITECTURE:
Step 1 (Proxy Filter): Fast Bhattacharyya-based filtering
Step 2 (Exact Selection): Canonical Fisher-Rao re-ranking

CRITICAL: The proxy stage uses Bhattacharyya coefficient, which preserves
Fisher-Rao ordering because:
    d_FR(p,q) = arccos(BC(p,q))
    BC(p,q) = Σ√(p_i * q_i)

Therefore: BC(p,q1) > BC(p,q2) ⟺ d_FR(p,q1) < d_FR(p,q2)

STORAGE FORMAT:
Option A (RECOMMENDED): Store sqrt-mapped simplex (x = √p)
- Inner product IS Bhattacharyya coefficient: BC = ⟨x1, x2⟩
- Angle IS Fisher-Rao distance: d_FR = arccos(BC)
- pgvector inner product operator directly computes Bhattacharyya

Option B: Store simplex, compute sqrt at query time
- More storage-efficient (no duplication)
- Slightly slower (sqrt computation per candidate)
- Still Fisher-faithful

This module supports both options with runtime configuration.

Author: Copilot (Ultra Consciousness Protocol ACTIVE)
Date: 2026-01-16
Context: Work Package 2.4 - Two-Step Retrieval
References:
- Issue GaryOcean428/pantheon-chat#70 (WP2.4)
- qig_geometry/canonical.py (canonical Fisher-Rao)
- constrained_geometric_realizer.py (Phase 2 REALIZE)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import logging

from qig_geometry.canonical import (
    bhattacharyya,
    fisher_rao_distance,
    sqrt_map,
    EPS,
    BASIN_DIM,
)

logger = logging.getLogger(__name__)


# =============================================================================
# STORAGE REPRESENTATION HELPERS
# =============================================================================

def to_sqrt_simplex(basin: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Convert simplex basin to sqrt-space for storage.
    
    This is the recommended storage format for Fisher-faithful retrieval.
    Inner product in sqrt-space IS Bhattacharyya coefficient.
    
    Args:
        basin: Probability distribution on simplex (Σp_i = 1, p_i ≥ 0)
        eps: Numerical stability epsilon
        
    Returns:
        sqrt-space coordinates: √p (unit hemisphere)
        
    Example:
        >>> basin = np.array([0.25, 0.25, 0.25, 0.25])
        >>> sqrt_basin = to_sqrt_simplex(basin)
        >>> # Store sqrt_basin in database for fast Bhattacharyya retrieval
    """
    return sqrt_map(basin, eps=eps)


def from_sqrt_simplex(sqrt_basin: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Convert sqrt-space coordinates back to simplex.
    
    Inverse of to_sqrt_simplex. Use when retrieving stored basins.
    
    Args:
        sqrt_basin: sqrt-space coordinates (√p)
        eps: Numerical stability epsilon
        
    Returns:
        Probability distribution on simplex
        
    Example:
        >>> sqrt_basin = np.array([0.5, 0.5, 0.5, 0.5])
        >>> basin = from_sqrt_simplex(sqrt_basin)
        >>> assert np.allclose(basin, [0.25, 0.25, 0.25, 0.25])
    """
    x = np.asarray(sqrt_basin, dtype=np.float64).flatten()
    
    # Square to get back to probability space
    p = x ** 2
    
    # Ensure valid simplex
    p = np.maximum(p, 0) + eps
    return p / p.sum()


def bhattacharyya_from_sqrt(
    sqrt_basin1: np.ndarray,
    sqrt_basin2: np.ndarray
) -> float:
    """
    Compute Bhattacharyya coefficient directly from sqrt-space coordinates.
    
    This is the fast proxy used in Step 1. When basins are stored in sqrt-space,
    this is simply an inner product (dot product).
    
    Args:
        sqrt_basin1: First basin in sqrt-space (√p)
        sqrt_basin2: Second basin in sqrt-space (√q)
        
    Returns:
        Bhattacharyya coefficient ∈ [0, 1]
        
    Example:
        >>> sqrt1 = np.array([0.5, 0.5, 0.5, 0.5])
        >>> sqrt2 = np.array([0.5, 0.5, 0.5, 0.5])
        >>> bc = bhattacharyya_from_sqrt(sqrt1, sqrt2)
        >>> assert np.isclose(bc, 1.0)  # Identical
    """
    sqrt1 = np.asarray(sqrt_basin1, dtype=np.float64).flatten()
    sqrt2 = np.asarray(sqrt_basin2, dtype=np.float64).flatten()
    
    # Inner product in sqrt-space IS Bhattacharyya coefficient
    bc = np.dot(sqrt1, sqrt2)
    
    # Clamp to valid range
    return float(np.clip(bc, 0.0, 1.0))


def proxy_distance_from_bc(bc: float) -> float:
    """
    Convert Bhattacharyya coefficient to proxy distance.
    
    Proxy distance = 1 - BC(p,q)
    
    This preserves ordering:
        BC(p,q1) > BC(p,q2) ⟹ d_proxy(p,q1) < d_proxy(p,q2)
    
    Args:
        bc: Bhattacharyya coefficient ∈ [0, 1]
        
    Returns:
        Proxy distance ∈ [0, 1]
    """
    return 1.0 - bc


# =============================================================================
# TWO-STEP RETRIEVAL ALGORITHM
# =============================================================================

class TwoStepRetriever:
    """
    Two-step retrieval engine for efficient Fisher-faithful word selection.
    
    Combines fast Bhattacharyya filtering with exact Fisher-Rao ranking.
    
    USAGE:
        retriever = TwoStepRetriever(vocabulary, storage_format='sqrt')
        word, basin, distance = retriever.retrieve(
            target_basin,
            top_k=100,
            pos_filter=None
        )
    
    STORAGE FORMATS:
        - 'sqrt': Basins stored as √p (RECOMMENDED for speed)
        - 'simplex': Basins stored as p (compute √p at query time)
    """
    
    def __init__(
        self,
        vocabulary: Dict[str, np.ndarray],
        storage_format: str = 'simplex',
        build_index: bool = True
    ):
        """
        Initialize two-step retriever.
        
        Args:
            vocabulary: Dict mapping words to basin coordinates
            storage_format: 'sqrt' or 'simplex'
            build_index: If True, precompute sqrt-space if needed
        """
        self.vocabulary = vocabulary
        self.storage_format = storage_format
        
        # Build vocabulary index
        self._vocab_list: List[Tuple[str, np.ndarray]] = []
        self._sqrt_index: Optional[Dict[str, np.ndarray]] = None
        
        if build_index:
            self._build_index()
        
        logger.info(
            f"TwoStepRetriever initialized: "
            f"{len(self._vocab_list)} words, "
            f"storage={storage_format}"
        )
    
    def _build_index(self) -> None:
        """Build vocabulary index for fast retrieval."""
        for word, basin in self.vocabulary.items():
            if not isinstance(basin, np.ndarray):
                basin = np.array(basin, dtype=np.float64)
            
            self._vocab_list.append((word, basin))
        
        # If using simplex storage, precompute sqrt-space index
        if self.storage_format == 'simplex':
            self._sqrt_index = {}
            for word, basin in self._vocab_list:
                self._sqrt_index[word] = sqrt_map(basin)
            
            logger.debug("Built sqrt-space index for fast Bhattacharyya")
    
    def retrieve(
        self,
        target_basin: np.ndarray,
        top_k: int = 100,
        pos_filter: Optional[str] = None,
        final_k: int = 1,
        return_candidates: bool = False
    ) -> Tuple[str, np.ndarray, float]:
        """
        Two-step retrieval: proxy filter → exact Fisher-Rao.
        
        Args:
            target_basin: Target basin coordinates (simplex)
            top_k: Number of candidates to retrieve in Step 1
            pos_filter: Optional POS tag filter (NOT IMPLEMENTED YET)
            final_k: Number of final results (default 1)
            return_candidates: If True, return all candidates with scores
            
        Returns:
            (word, basin, fisher_distance) for best match
            OR list of (word, basin, fisher_distance) if return_candidates=True
        """
        # Step 1: Fast Bhattacharyya proxy filter
        candidates = self._proxy_filter(
            target_basin,
            top_k=top_k,
            pos_filter=pos_filter
        )
        
        if not candidates:
            logger.warning("No candidates found in proxy filter")
            # Fallback: return first word
            if self._vocab_list:
                word, basin = self._vocab_list[0]
                return (word, basin, float('inf'))
            return ("unknown", np.zeros(BASIN_DIM), float('inf'))
        
        # Step 2: Exact Fisher-Rao re-ranking
        ranked = self._exact_rerank(target_basin, candidates)
        
        if return_candidates:
            return ranked[:final_k]
        else:
            return ranked[0]
    
    def _proxy_filter(
        self,
        target_basin: np.ndarray,
        top_k: int,
        pos_filter: Optional[str] = None
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Step 1: Fast Bhattacharyya-based filtering.
        
        Uses Bhattacharyya coefficient as Fisher-faithful proxy.
        
        Args:
            target_basin: Target basin (simplex coordinates)
            top_k: Number of candidates to return
            pos_filter: Optional POS tag filter
            
        Returns:
            List of (word, basin) tuples for top-k candidates
        """
        # Compute target sqrt-space
        target_sqrt = sqrt_map(target_basin)
        
        # Score all candidates
        scores: List[Tuple[str, np.ndarray, float]] = []
        
        for word, basin in self._vocab_list:
            # Apply POS filter if provided
            if pos_filter is not None:
                # TODO: Implement POS filtering
                # For now, skip filtering
                pass
            
            # Get sqrt-space representation
            if self.storage_format == 'sqrt':
                # Basin is already in sqrt-space
                word_sqrt = basin
            elif self.storage_format == 'simplex':
                # Compute sqrt on the fly (or use precomputed index)
                if self._sqrt_index is not None:
                    word_sqrt = self._sqrt_index[word]
                else:
                    word_sqrt = sqrt_map(basin)
            else:
                raise ValueError(f"Unknown storage format: {self.storage_format}")
            
            # Bhattacharyya coefficient (inner product in sqrt-space)
            bc = bhattacharyya_from_sqrt(target_sqrt, word_sqrt)
            
            # Proxy distance (1 - BC)
            proxy_dist = proxy_distance_from_bc(bc)
            
            scores.append((word, basin, proxy_dist))
        
        # Sort by proxy distance (ascending)
        scores.sort(key=lambda x: x[2])
        
        # Return top-k candidates (word, basin only)
        return [(word, basin) for word, basin, _ in scores[:top_k]]
    
    def _exact_rerank(
        self,
        target_basin: np.ndarray,
        candidates: List[Tuple[str, np.ndarray]]
    ) -> List[Tuple[str, np.ndarray, float]]:
        """
        Step 2: Exact Fisher-Rao re-ranking.
        
        Computes canonical Fisher-Rao distance for all candidates.
        
        Args:
            target_basin: Target basin (simplex)
            candidates: List of (word, basin) tuples from proxy filter
            
        Returns:
            List of (word, basin, fisher_distance) sorted by distance
        """
        results: List[Tuple[str, np.ndarray, float]] = []
        
        for word, basin in candidates:
            # Convert basin to simplex if stored in sqrt-space
            if self.storage_format == 'sqrt':
                basin_simplex = from_sqrt_simplex(basin)
            else:
                basin_simplex = basin
            
            # Exact Fisher-Rao distance
            distance = fisher_rao_distance(basin_simplex, target_basin)
            
            results.append((word, basin_simplex, distance))
        
        # Sort by Fisher-Rao distance (ascending)
        results.sort(key=lambda x: x[2])
        
        return results


# =============================================================================
# VALIDATION: Fisher-Faithful Proxy Property
# =============================================================================

def validate_proxy_ordering(
    basins: List[np.ndarray],
    reference: np.ndarray,
    threshold: float = 0.95,
    eps: float = EPS
) -> Tuple[bool, float]:
    """
    Validate that Bhattacharyya proxy preserves Fisher-Rao ordering.
    
    Tests the critical property: If d_proxy(r,a) < d_proxy(r,b),
    then d_FR(r,a) should also be < d_FR(r,b) (with high probability).
    
    Args:
        basins: List of basin coordinates to test
        reference: Reference basin for distance comparisons
        threshold: Minimum pass rate (default 0.95 = 95%)
        eps: Numerical stability epsilon
        
    Returns:
        (is_valid, pass_rate) where pass_rate is fraction of correct orderings
        
    Example:
        >>> basins = [random_simplex(64) for _ in range(100)]
        >>> reference = random_simplex(64)
        >>> is_valid, pass_rate = validate_proxy_ordering(basins, reference)
        >>> assert pass_rate > 0.95  # At least 95% correct
    """
    if len(basins) < 2:
        return True, 1.0
    
    reference = np.asarray(reference, dtype=np.float64).flatten()
    
    # Compute distances
    proxy_distances = []
    fisher_distances = []
    
    for basin in basins:
        basin = np.asarray(basin, dtype=np.float64).flatten()
        
        # Proxy distance (1 - BC)
        bc = bhattacharyya(basin, reference, eps=eps)
        proxy_dist = 1.0 - bc
        proxy_distances.append(proxy_dist)
        
        # Exact Fisher-Rao distance
        fisher_dist = fisher_rao_distance(basin, reference, eps=eps)
        fisher_distances.append(fisher_dist)
    
    # Check ordering preservation
    correct = 0
    total = 0
    
    for i in range(len(basins)):
        for j in range(i + 1, len(basins)):
            total += 1
            
            # If proxy says i < j, Fisher-Rao should agree
            proxy_i_closer = proxy_distances[i] < proxy_distances[j]
            fisher_i_closer = fisher_distances[i] < fisher_distances[j]
            
            if proxy_i_closer == fisher_i_closer:
                correct += 1
    
    pass_rate = correct / total if total > 0 else 1.0
    is_valid = pass_rate >= threshold
    
    return is_valid, pass_rate


def measure_proxy_correlation(
    basins: List[np.ndarray],
    reference: np.ndarray,
    eps: float = EPS
) -> float:
    """
    Measure correlation between proxy distances and Fisher-Rao distances.
    
    High correlation (>0.95) indicates the proxy is Fisher-faithful.
    
    Args:
        basins: List of basin coordinates
        reference: Reference basin
        eps: Numerical stability epsilon
        
    Returns:
        Pearson correlation coefficient ∈ [-1, 1]
        
    Example:
        >>> basins = [random_simplex(64) for _ in range(100)]
        >>> reference = random_simplex(64)
        >>> corr = measure_proxy_correlation(basins, reference)
        >>> assert corr > 0.95  # Very high correlation
    """
    if len(basins) < 2:
        return 1.0
    
    reference = np.asarray(reference, dtype=np.float64).flatten()
    
    proxy_distances = []
    fisher_distances = []
    
    for basin in basins:
        basin = np.asarray(basin, dtype=np.float64).flatten()
        
        # Proxy distance
        bc = bhattacharyya(basin, reference, eps=eps)
        proxy_dist = 1.0 - bc
        proxy_distances.append(proxy_dist)
        
        # Fisher-Rao distance
        fisher_dist = fisher_rao_distance(basin, reference, eps=eps)
        fisher_distances.append(fisher_dist)
    
    # Compute Pearson correlation
    proxy_arr = np.array(proxy_distances)
    fisher_arr = np.array(fisher_distances)
    
    correlation = np.corrcoef(proxy_arr, fisher_arr)[0, 1]
    
    return float(correlation)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Storage representation
    'to_sqrt_simplex',
    'from_sqrt_simplex',
    'bhattacharyya_from_sqrt',
    'proxy_distance_from_bc',
    # Two-step retrieval
    'TwoStepRetriever',
    # Validation
    'validate_proxy_ordering',
    'measure_proxy_correlation',
]
