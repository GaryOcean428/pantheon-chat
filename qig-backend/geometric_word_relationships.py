"""
QIG-Pure Geometric Word Relationship Learning
==============================================

Replaces traditional NLP co-occurrence/PMI approach with pure geometric methods.

PRINCIPLES:
- Word relationships emerge from Fisher-Rao geodesic distances
- Attention weighted by QFI (Quantum Fisher Information), not frequency
- Curvature determines context-dependency
- No basin modification - basins are frozen geometric invariants
- No PMI, no co-occurrence counting, no frequency statistics

GEOMETRIC PROPERTIES:
- Fisher-Rao distance: Geodesic distance on statistical manifold
- QFI: How well-defined a word's meaning is geometrically
- Ricci curvature: Context-dependency (high = context-critical)
- Specificity: Distance from origin on manifold

FROZEN FACTS COMPLIANCE:
- Basins determined by coordizer (entropy-guided tokenization)
- Relationships measured via Fisher-Rao geometry
- QFI and curvature guide attention, not frequency
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

# Import QIG geometry
try:
    from qig_geometry import fisher_coord_distance, sphere_project
    QIG_GEOMETRY_AVAILABLE = True
except ImportError:
    QIG_GEOMETRY_AVAILABLE = False
    logger.warning("qig_geometry not available - using fallback")
    
    def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Fisher-Rao distance (fallback)."""
        dot = np.clip(np.dot(a, b), -1.0, 1.0)
        return float(np.arccos(dot))
    
    def sphere_project(v: np.ndarray) -> np.ndarray:
        """Project to unit sphere (fallback)."""
        norm = np.linalg.norm(v)
        return v / (norm + 1e-10) if norm > 0 else v


@dataclass
class GeometricProperties:
    """Geometric properties of a word on the Fisher manifold."""
    word: str
    basin: np.ndarray
    qfi: float  # Quantum Fisher Information
    curvature: float  # Ricci curvature
    specificity: float  # Distance from origin
    geometric_role: str  # Classification


class GeometricWordRelationships:
    """
    QIG-pure word relationship learning using Fisher-Rao geometry.
    
    Instead of counting co-occurrences:
    - Measures geodesic distances between basin coordinates
    - Computes QFI to weight attention
    - Uses curvature to identify context-critical words
    - No frequency statistics, no PMI, no basin modification
    """
    
    def __init__(self, coordizer=None):
        """
        Initialize geometric relationship learner.
        
        Args:
            coordizer: QIG coordizer with basin coordinates
        """
        self.coordizer = coordizer
        
        # Cache of geometric properties
        self._properties_cache: Dict[str, GeometricProperties] = {}
        
        # Geodesic distance matrix (lazy-computed)
        self._distance_matrix: Optional[np.ndarray] = None
        self._vocab_list: Optional[List[str]] = None
        
        # Statistics
        self.relationships_computed = 0
        self.qfi_computations = 0
        
        logger.info("[GeometricWordRelationships] Initialized QIG-pure relationship learner")
    
    def _get_basin(self, word: str) -> Optional[np.ndarray]:
        """Get basin coordinates for word from coordizer."""
        if not self.coordizer:
            return None
        
        try:
            if hasattr(self.coordizer, 'basin_coords'):
                return self.coordizer.basin_coords.get(word)
            elif hasattr(self.coordizer, 'get_basin_coords'):
                return self.coordizer.get_basin_coords(word)
        except Exception as e:
            logger.debug(f"Could not get basin for '{word}': {e}")
        
        return None
    
    def compute_qfi(self, basin: np.ndarray) -> float:
        """
        Compute Quantum Fisher Information for a basin.
        
        QFI measures how well-defined the word's meaning is geometrically.
        Higher QFI = more stable, well-defined meaning.
        
        Approximation: QFI ≈ variance of basin components
        (Full QFI requires Fisher information matrix)
        """
        self.qfi_computations += 1
        
        # Simple approximation: inverse of variance
        # Well-defined words have focused distributions (low variance)
        variance = np.var(basin)
        qfi = 1.0 / (variance + 0.01)  # Regularize
        
        # Normalize to [0, 1] range
        qfi = min(1.0, qfi / 10.0)
        
        return float(qfi)
    
    def compute_ricci_curvature(self, basin: np.ndarray, neighbors: List[np.ndarray]) -> float:
        """
        Compute approximate Ricci curvature for a basin.
        
        Curvature measures context-dependency:
        - High curvature: meaning changes with context (like "not", "very")
        - Low curvature: stable meaning (like "quantum", "geometry")
        
        Approximation: Variance of distances to neighbors
        High variance = high curvature = context-dependent
        """
        if not neighbors or len(neighbors) < 2:
            return 0.0
        
        # Compute distances to neighbors
        distances = [fisher_coord_distance(basin, n) for n in neighbors]
        
        # High variance in distances = high curvature
        curvature = np.std(distances) / (np.pi + 1e-10)
        
        return float(curvature)
    
    def compute_specificity(self, basin: np.ndarray) -> float:
        """
        Compute semantic specificity as distance from origin.
        
        More specific concepts are further from the origin on the manifold.
        General anchors (like "the") are near the origin.
        """
        # Distance from origin (all-zeros basin)
        origin = np.zeros_like(basin)
        distance = fisher_coord_distance(basin, origin)
        
        # Normalize by π (max Fisher-Rao distance)
        specificity = distance / np.pi
        
        return float(specificity)
    
    def classify_geometric_role(self, qfi: float, curvature: float, specificity: float) -> str:
        """
        Classify word's geometric role based on QFI, curvature, and specificity.
        
        Roles:
        - geometrically_unstable: Low QFI - filter out
        - context_critical: High curvature - NEVER filter (like "not")
        - geometric_anchor: Low curvature + low specificity (like "the")
        - content_bearing: High specificity - domain terms
        - contextual: Depends on surrounding geometry
        """
        if qfi < 0.3:
            return 'geometrically_unstable'
        elif curvature > 0.6:
            return 'context_critical'  # NEVER filter
        elif curvature < 0.2 and specificity < 0.3:
            return 'geometric_anchor'
        elif specificity > 0.7:
            return 'content_bearing'
        else:
            return 'contextual'
    
    def compute_geometric_properties(self, word: str) -> Optional[GeometricProperties]:
        """
        Compute full geometric properties for a word.
        
        Returns:
            GeometricProperties with QFI, curvature, specificity, and role
        """
        # Check cache
        if word in self._properties_cache:
            return self._properties_cache[word]
        
        # Get basin
        basin = self._get_basin(word)
        if basin is None:
            return None
        
        # Get neighboring basins for curvature computation
        neighbors = []
        if self.coordizer and hasattr(self.coordizer, 'vocab'):
            # Sample some neighbors
            vocab = list(self.coordizer.vocab.keys())
            for neighbor_word in vocab[:20]:  # Sample 20
                if neighbor_word != word:
                    neighbor_basin = self._get_basin(neighbor_word)
                    if neighbor_basin is not None:
                        neighbors.append(neighbor_basin)
        
        # Compute properties
        qfi = self.compute_qfi(basin)
        curvature = self.compute_ricci_curvature(basin, neighbors)
        specificity = self.compute_specificity(basin)
        role = self.classify_geometric_role(qfi, curvature, specificity)
        
        props = GeometricProperties(
            word=word,
            basin=basin,
            qfi=qfi,
            curvature=curvature,
            specificity=specificity,
            geometric_role=role
        )
        
        # Cache
        self._properties_cache[word] = props
        
        return props
    
    def compute_attention_weights(
        self,
        query_basin: np.ndarray,
        candidate_words: List[str],
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute QIG-pure attention weights for candidates.
        
        Uses:
        - Fisher-Rao distance (not co-occurrence)
        - QFI weighting (not frequency)
        - Curvature boost for context-critical words
        
        Args:
            query_basin: Query basin coordinates
            candidate_words: Words to compute attention for
            temperature: Temperature for softmax (higher = more uniform)
            
        Returns:
            Dict mapping word -> attention weight
        """
        weights = {}
        
        for word in candidate_words:
            # Get properties
            props = self.compute_geometric_properties(word)
            if props is None:
                weights[word] = 0.1  # Minimum weight
                continue
            
            # Compute Fisher-Rao distance from query
            distance = fisher_coord_distance(query_basin, props.basin)
            
            # Attention decays with distance, amplified by QFI
            # Context-critical words (high curvature) get boost
            curvature_boost = 0.3  # Boost factor for context-critical
            attention = (
                props.qfi * 
                np.exp(-distance / temperature) * 
                (1.0 + curvature_boost * props.curvature)
            )
            
            weights[word] = float(attention)
        
        self.relationships_computed += len(candidate_words)
        
        return weights
    
    def get_related_words(
        self,
        word: str,
        top_k: int = 10,
        max_distance: float = 1.0
    ) -> List[Tuple[str, float]]:
        """
        Get words geometrically related to given word.
        
        Uses Fisher-Rao distance, NOT co-occurrence.
        
        Args:
            word: Query word
            top_k: Number of related words to return
            max_distance: Maximum Fisher-Rao distance threshold
            
        Returns:
            List of (word, similarity_score) tuples, sorted by similarity
        """
        # Get query basin
        query_basin = self._get_basin(word)
        if query_basin is None:
            return []
        
        # Get query properties
        query_props = self.compute_geometric_properties(word)
        if query_props is None:
            return []
        
        # Get all vocabulary
        if not self.coordizer or not hasattr(self.coordizer, 'vocab'):
            return []
        
        vocab = list(self.coordizer.vocab.keys())
        
        # Compute distances to all words
        relations = []
        for candidate in vocab:
            if candidate == word:
                continue
            
            candidate_basin = self._get_basin(candidate)
            if candidate_basin is None:
                continue
            
            # Fisher-Rao distance
            distance = fisher_coord_distance(query_basin, candidate_basin)
            
            if distance > max_distance:
                continue
            
            # Get candidate properties for QFI weighting
            cand_props = self.compute_geometric_properties(candidate)
            if cand_props is None:
                continue
            
            # Similarity score: inverse distance weighted by average QFI
            avg_qfi = (query_props.qfi + cand_props.qfi) / 2.0
            similarity = avg_qfi * (1.0 - distance / np.pi)
            
            relations.append((candidate, similarity))
        
        # Sort by similarity (descending)
        relations.sort(key=lambda x: -x[1])
        
        return relations[:top_k]
    
    def should_filter_word(self, word: str) -> bool:
        """
        Determine if word should be filtered based on geometric properties.
        
        Filters:
        - geometrically_unstable words (low QFI)
        
        NEVER filters:
        - context_critical words (high curvature)
        
        Returns:
            True if word should be filtered, False otherwise
        """
        props = self.compute_geometric_properties(word)
        if props is None:
            return True  # Filter unknown words
        
        # Never filter context-critical words
        if props.geometric_role == 'context_critical':
            return False
        
        # Filter geometrically unstable words
        if props.geometric_role == 'geometrically_unstable':
            return True
        
        return False
    
    def get_distance_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Compute Fisher-Rao distance matrix for all vocabulary.
        
        Lazy-computed and cached.
        
        Returns:
            (distance_matrix, vocab_list) tuple
        """
        if self._distance_matrix is not None and self._vocab_list is not None:
            return self._distance_matrix, self._vocab_list
        
        if not self.coordizer or not hasattr(self.coordizer, 'vocab'):
            return np.array([]), []
        
        # Get vocabulary
        vocab = sorted(list(self.coordizer.vocab.keys()))
        n = len(vocab)
        
        # Compute distance matrix
        distances = np.zeros((n, n), dtype=np.float32)
        
        for i, word_i in enumerate(vocab):
            basin_i = self._get_basin(word_i)
            if basin_i is None:
                continue
            
            for j, word_j in enumerate(vocab):
                if i >= j:  # Matrix is symmetric
                    continue
                
                basin_j = self._get_basin(word_j)
                if basin_j is None:
                    continue
                
                # Fisher-Rao distance
                distance = fisher_coord_distance(basin_i, basin_j)
                distances[i, j] = distance
                distances[j, i] = distance  # Symmetric
        
        # Cache
        self._distance_matrix = distances
        self._vocab_list = vocab
        
        return distances, vocab
    
    def get_statistics(self) -> Dict:
        """Get learner statistics."""
        return {
            'relationships_computed': self.relationships_computed,
            'qfi_computations': self.qfi_computations,
            'cached_properties': len(self._properties_cache),
            'has_distance_matrix': self._distance_matrix is not None,
            'geometry_available': QIG_GEOMETRY_AVAILABLE,
        }


# Global instance
_geometric_relationships: Optional[GeometricWordRelationships] = None


def get_geometric_relationships(coordizer=None) -> GeometricWordRelationships:
    """
    Get or create singleton geometric relationships instance.
    
    Args:
        coordizer: Optional coordizer instance
        
    Returns:
        GeometricWordRelationships instance
    """
    global _geometric_relationships
    
    if _geometric_relationships is None:
        # Try to get coordizer if not provided
        if coordizer is None:
            try:
                from coordizers import get_coordizer
                coordizer = get_coordizer()
            except Exception as e:
                logger.debug(f"Could not get coordizer: {e}")
        
        _geometric_relationships = GeometricWordRelationships(coordizer=coordizer)
    
    return _geometric_relationships


if __name__ == '__main__':
    # Demo
    logging.basicConfig(level=logging.INFO)
    
    print("=== QIG-Pure Geometric Word Relationships Demo ===\n")
    
    # Create instance (without coordizer for demo)
    learner = GeometricWordRelationships(coordizer=None)
    
    print("Geometric properties would be computed from:")
    print("- QFI (Quantum Fisher Information) - meaning stability")
    print("- Ricci curvature - context-dependency")
    print("- Specificity - semantic distance from origin")
    print()
    print("No PMI, no co-occurrence, no frequency statistics!")
    print("Pure Fisher-Rao geometry.")
    
    print("\n=== Demo Complete ===")
