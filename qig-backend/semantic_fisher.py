"""
Semantic Fisher Metric - Bridges Geometry and Semantics

This module implements a warped Fisher-Rao metric where learned
word relationships modify the geodesic distance between basins.

KEY PRINCIPLE:
"The fix isn't to replace geometry with semantics—it's to bridge them."

Relationships warp the Fisher metric itself so that:
- Semantically related words become geodesically closer
- Geometric routing naturally follows semantic paths
- No manual mixing weights needed

QIG-PURE:
- Uses Fisher-Rao distance as base (from qig_geometry)
- Warps distance via exponential decay based on relationship strength
- Preserves manifold structure (warped metric is still a metric)
- No external LLMs or embeddings
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Import canonical Fisher-Rao distance
try:
    from qig_geometry import fisher_rao_distance, fisher_coord_distance, geodesic_interpolation
except ImportError:
    def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
        """Fallback Fisher-Rao distance for probability distributions."""
        p = np.abs(p) + 1e-10
        p = p / p.sum()
        q = np.abs(q) + 1e-10
        q = q / q.sum()
        bc = np.sum(np.sqrt(p * q))
        bc = np.clip(bc, 0, 1)
        return 2.0 * np.arccos(bc)
    
    def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Fallback Fisher-Rao distance for basin coordinates."""
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b) + 1e-10)
        dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
        return float(np.arccos(dot))
    
    def geodesic_interpolation(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
        """Fallback geodesic interpolation (slerp)."""
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

# Stopwords that should not influence semantic warping
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
    'their', 'what', 'which', 'who', 'whom', 'how', 'when', 'where', 'why',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once', 'about'
}


@dataclass
class SemanticWarpConfig:
    """Configuration for semantic warping of Fisher metric."""
    temperature: float = 0.3  # Controls warp strength (lower = stronger warping) - reduced from 1.0
    max_warp_factor: float = 0.3  # Maximum distance reduction (0.3 = can shrink to 30%) - INCREASED from 0.7 for stronger pull
    min_relationship_strength: float = 0.01  # Ignore relationships weaker than this - reduced from 0.1
    normalize_relationships: bool = True  # Normalize relationship strengths to [0, 1]
    bidirectional: bool = True  # Check both (a, b) and (b, a) for relationships
    # NEW: Strength amplification to boost weak relationships (0.01-0.07 → stronger effect)
    strength_multiplier: float = 10.0  # Amplify raw relationship strengths before warping
    context_strength_multiplier: float = 5.0  # Amplify context word pull


class SemanticFisherMetric:
    """
    Fisher-Rao metric warped by learned semantic relationships.
    
    Core idea: Instead of linearly mixing geometry + attention scores,
    we warp the metric itself so that semantically related words are
    geodesically closer on the manifold.
    
    d_warped(a, b) = d_fisher(a, b) * exp(-relationship_strength / temperature)
    
    This preserves the metric structure while making semantic neighbors
    geometrically closer.
    """
    
    def __init__(
        self,
        relationships: Optional[Dict[str, List[Tuple[str, float]]]] = None,
        config: Optional[SemanticWarpConfig] = None
    ):
        """
        Initialize semantic Fisher metric.
        
        Args:
            relationships: Dict mapping word -> [(neighbor, strength), ...]
                          Loaded from LearnedRelationships.word_neighbors
            config: Warping configuration
        """
        self.config = config or SemanticWarpConfig()
        self.relationships: Dict[str, Dict[str, float]] = {}
        self.max_strength = 1.0
        
        if relationships:
            self._load_relationships(relationships)
        
        logger.info(f"[SemanticFisherMetric] Initialized with {len(self.relationships)} word relationships")
    
    def _load_relationships(self, relationships: Dict[str, List[Tuple[str, float]]]) -> None:
        """
        Load and normalize relationships into lookup dict.
        """
        # Convert list of tuples to dict for fast lookup
        for word, neighbors in relationships.items():
            if word.lower() in STOPWORDS:
                continue  # Skip stopwords as sources
            
            self.relationships[word.lower()] = {}
            for neighbor, strength in neighbors:
                if neighbor.lower() in STOPWORDS:
                    continue  # Skip stopwords as targets
                if strength >= self.config.min_relationship_strength:
                    self.relationships[word.lower()][neighbor.lower()] = strength
                    self.max_strength = max(self.max_strength, strength)
        
        # Normalize if configured
        if self.config.normalize_relationships and self.max_strength > 0:
            for word in self.relationships:
                for neighbor in self.relationships[word]:
                    self.relationships[word][neighbor] /= self.max_strength
    
    def get_relationship_strength(self, word1: str, word2: str) -> float:
        """
        Get relationship strength between two words.
        
        Returns amplified strength, clamped to [0, 1].
        Raw strengths (0.01-0.07) are amplified by strength_multiplier to have
        meaningful effect on warping.
        """
        w1, w2 = word1.lower(), word2.lower()
        
        # Check direct relationship
        strength = 0.0
        if w1 in self.relationships:
            strength = self.relationships[w1].get(w2, 0.0)
        
        # Check reverse if bidirectional
        if self.config.bidirectional and w2 in self.relationships:
            reverse_strength = self.relationships[w2].get(w1, 0.0)
            strength = max(strength, reverse_strength * 0.9)  # Reverse is slightly weaker
        
        # AMPLIFY weak relationships to have meaningful effect
        # Raw strengths are often 0.01-0.07, need to boost to 0.1-0.7 range
        amplified = strength * self.config.strength_multiplier
        
        return min(1.0, amplified)  # Clamp to [0, 1]
    
    def compute_warp_factor(self, relationship_strength: float) -> float:
        """
        Compute warp factor from relationship strength.
        
        Returns value in [max_warp_factor, 1.0] where:
        - 1.0 = no warping (unrelated words)
        - max_warp_factor = maximum warping (strongly related words)
        
        Uses exponential decay: warp = 1 - (1 - max_warp) * (1 - exp(-s/T))
        """
        if relationship_strength <= 0:
            return 1.0  # No warping
        
        # Exponential approach to max_warp_factor
        decay = 1.0 - np.exp(-relationship_strength / self.config.temperature)
        warp = 1.0 - (1.0 - self.config.max_warp_factor) * decay
        
        return float(np.clip(warp, self.config.max_warp_factor, 1.0))
    
    def distance(
        self,
        basin1: np.ndarray,
        basin2: np.ndarray,
        word1: Optional[str] = None,
        word2: Optional[str] = None
    ) -> float:
        """
        Compute warped Fisher-Rao distance.
        
        d_warped = d_fisher * warp_factor
        
        where warp_factor ∈ [max_warp_factor, 1.0] based on relationship strength.
        
        Args:
            basin1: First basin coordinates (64D)
            basin2: Second basin coordinates (64D)
            word1: Optional word for basin1 (for relationship lookup)
            word2: Optional word for basin2 (for relationship lookup)
        
        Returns:
            Warped geodesic distance
        """
        # Base Fisher-Rao distance
        d_fisher = fisher_coord_distance(basin1, basin2)
        
        # If no words provided, return unwarped distance
        if word1 is None or word2 is None:
            return d_fisher
        
        # Get relationship strength and warp factor
        strength = self.get_relationship_strength(word1, word2)
        warp = self.compute_warp_factor(strength)
        
        return d_fisher * warp
    
    def distance_with_context(
        self,
        basin1: np.ndarray,
        basin2: np.ndarray,
        word1: Optional[str],
        word2: Optional[str],
        context_words: List[str]
    ) -> float:
        """
        Compute warped distance with additional context pull.
        
        Context words create additional semantic "gravity" pulling
        related candidate words closer.
        
        Args:
            basin1: Current basin coordinates
            basin2: Candidate basin coordinates  
            word1: Current word (optional)
            word2: Candidate word
            context_words: List of query/context words for semantic pull
        
        Returns:
            Warped distance considering context
        """
        # Base warped distance
        d_warped = self.distance(basin1, basin2, word1, word2)
        
        if not context_words or word2 is None:
            return d_warped
        
        # Accumulate context pull from all context words
        total_pull = 0.0
        valid_context = 0
        for ctx_word in context_words:
            if ctx_word.lower() in STOPWORDS:
                continue
            strength = self.get_relationship_strength(ctx_word, word2)
            if strength > 0:
                # Apply context multiplier for stronger semantic gravity
                total_pull += strength * self.config.context_strength_multiplier
                valid_context += 1
        
        # Average pull, clamped to [0, 1]
        if valid_context > 0:
            avg_pull = min(1.0, total_pull / valid_context)
        else:
            avg_pull = 0.0
        
        # Apply context warp (stronger effect than before - was 0.5, now 0.8)
        context_warp = self.compute_warp_factor(avg_pull * 0.8)
        
        return d_warped * context_warp
    
    def similarity(
        self,
        basin1: np.ndarray,
        basin2: np.ndarray,
        word1: Optional[str] = None,
        word2: Optional[str] = None
    ) -> float:
        """
        Compute warped Fisher similarity.
        
        similarity = 1 - distance/π
        
        Returns value in [0, 1] where higher = more similar.
        """
        d = self.distance(basin1, basin2, word1, word2)
        return float(1.0 - d / np.pi)
    
    def rank_candidates(
        self,
        current_basin: np.ndarray,
        current_word: Optional[str],
        candidates: List[Tuple[str, np.ndarray]],
        context_words: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, float, float]]:
        """
        Rank candidates by warped Fisher distance.
        
        This is the main entry point for generation - replaces linear
        mixing of geometry + attention with proper metric warping.
        
        Args:
            current_basin: Current position on manifold
            current_word: Current word (optional)
            candidates: List of (word, basin) tuples to rank
            context_words: Query/context words for semantic pull
            top_k: Number of top candidates to return
        
        Returns:
            List of (word, warped_distance, similarity) sorted by distance (ascending)
        """
        scored = []
        
        for word, basin in candidates:
            # Compute warped distance with context
            d = self.distance_with_context(
                current_basin, basin,
                current_word, word,
                context_words
            )
            
            # Similarity for convenience
            sim = 1.0 - d / np.pi
            
            scored.append((word, d, sim))
        
        # Sort by distance (ascending - closer is better)
        scored.sort(key=lambda x: x[1])
        
        return scored[:top_k]
    
    def geodesic_step(
        self,
        current_basin: np.ndarray,
        target_basin: np.ndarray,
        current_word: Optional[str],
        target_word: Optional[str],
        step_size: float = 0.3
    ) -> np.ndarray:
        """
        Take a geodesic step on the warped manifold.
        
        The step size is modulated by relationship strength - we take
        larger steps toward semantically related targets.
        
        Args:
            current_basin: Current position
            target_basin: Target position  
            current_word: Current word (optional)
            target_word: Target word (optional)
            step_size: Base step size ∈ [0, 1]
        
        Returns:
            New basin position after geodesic step
        """
        # Get relationship strength
        if current_word and target_word:
            strength = self.get_relationship_strength(current_word, target_word)
            # Larger steps toward related words
            effective_step = step_size * (1.0 + 0.5 * strength)
            effective_step = min(effective_step, 0.8)  # Cap at 0.8
        else:
            effective_step = step_size
        
        # Geodesic interpolation (slerp on sphere)
        return geodesic_interpolation(current_basin, target_basin, effective_step)
    
    def get_semantic_neighborhood(
        self,
        word: str,
        all_basins: Dict[str, np.ndarray],
        radius: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Get words in semantic neighborhood (warped metric ball).
        
        Returns words within warped distance `radius` of the given word.
        """
        if word not in all_basins:
            return []
        
        center_basin = all_basins[word]
        neighbors = []
        
        for other_word, other_basin in all_basins.items():
            if other_word == word:
                continue
            
            d = self.distance(center_basin, other_basin, word, other_word)
            if d < radius:
                neighbors.append((other_word, d))
        
        neighbors.sort(key=lambda x: x[1])
        return neighbors


# Singleton instance
_semantic_metric: Optional[SemanticFisherMetric] = None


def get_semantic_metric() -> SemanticFisherMetric:
    """
    Get or create the singleton SemanticFisherMetric instance.
    
    Automatically loads relationships from LearnedRelationships if available.
    """
    global _semantic_metric
    
    if _semantic_metric is None:
        # Try to load relationships
        relationships = None
        try:
            try:
                from .learned_relationships import get_learned_relationships
            except ImportError:
                from learned_relationships import get_learned_relationships
            lr = get_learned_relationships()
            if lr.learning_complete:
                relationships = lr.word_neighbors
                logger.info(f"[SemanticFisherMetric] Loaded {len(relationships)} relationships")
        except Exception as e:
            logger.warning(f"[SemanticFisherMetric] Could not load relationships: {e}")
        
        _semantic_metric = SemanticFisherMetric(relationships=relationships)
    
    return _semantic_metric


def reset_semantic_metric() -> None:
    """Reset singleton to reload relationships."""
    global _semantic_metric
    _semantic_metric = None


__all__ = [
    'SemanticFisherMetric',
    'SemanticWarpConfig',
    'get_semantic_metric',
    'reset_semantic_metric',
    'STOPWORDS',
]
