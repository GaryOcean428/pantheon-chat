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
    # N-gram context settings for coherence
    ngram_window: int = 3  # How many recent tokens to consider for n-gram matching
    bigram_weight: float = 2.0  # Weight for bigram (word pair) coherence
    trigram_weight: float = 1.5  # Weight for trigram (3-word) coherence
    ngram_boost_factor: float = 3.0  # How much to boost candidates with good n-gram scores


class SemanticFisherMetric:
    """
    Fisher-Rao metric warped by learned semantic relationships.
    
    Core idea: Instead of linearly mixing geometry + attention scores,
    we warp the metric itself so that semantically related words are
    geodesically closer on the manifold.
    
    d_warped(a, b) = d_fisher(a, b) * exp(-relationship_strength / temperature)
    
    This preserves the metric structure while making semantic neighbors
    geometrically closer.
    
    NEW: N-gram context awareness for improved coherence.
    Tracks bigram/trigram patterns to boost candidates that continue
    natural word sequences.
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
        
        # N-gram tracking for coherence
        self.bigram_counts: Dict[Tuple[str, str], int] = {}
        self.trigram_counts: Dict[Tuple[str, str, str], int] = {}
        
        if relationships:
            self._load_relationships(relationships)
            self._learn_ngrams_from_relationships()
        
        logger.info(
            f"[SemanticFisherMetric] Initialized with {len(self.relationships)} word relationships, "
            f"{len(self.bigram_counts)} bigrams, {len(self.trigram_counts)} trigrams"
        )
    
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
    
    def _learn_ngrams_from_relationships(self) -> None:
        """
        Learn n-gram patterns from relationship graph.
        
        Creates bigram/trigram counts based on relationship chains.
        Words that are related likely form natural sequences.
        """
        # Build bigrams from direct relationships
        for word1, neighbors in self.relationships.items():
            for word2, strength in neighbors.items():
                if strength > self.config.min_relationship_strength:
                    # Strong relationship = likely co-occurrence
                    bigram = (word1, word2)
                    self.bigram_counts[bigram] = self.bigram_counts.get(bigram, 0) + 1
                    # Also reverse for flexibility
                    bigram_rev = (word2, word1)
                    self.bigram_counts[bigram_rev] = self.bigram_counts.get(bigram_rev, 0) + 1
        
        # Build trigrams from relationship chains (A->B->C)
        for word1, neighbors1 in self.relationships.items():
            for word2, strength1 in neighbors1.items():
                if strength1 > self.config.min_relationship_strength and word2 in self.relationships:
                    for word3, strength2 in self.relationships[word2].items():
                        if strength2 > self.config.min_relationship_strength and word3 != word1:
                            trigram = (word1, word2, word3)
                            # Weight by combined strength
                            weight = int((strength1 + strength2) * 5)
                            self.trigram_counts[trigram] = self.trigram_counts.get(trigram, 0) + max(1, weight)
        
        logger.info(
            f"[SemanticFisherMetric] Learned {len(self.bigram_counts)} bigrams, "
            f"{len(self.trigram_counts)} trigrams from relationships"
        )
    
    def add_observed_ngrams(self, tokens: List[str]) -> None:
        """
        Add observed n-grams from generated text to improve future coherence.
        
        Call this with each generated sentence to learn patterns.
        """
        tokens = [t.lower() for t in tokens if t.lower() not in STOPWORDS and len(t) > 1]
        
        # Add bigrams
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i+1])
            self.bigram_counts[bigram] = self.bigram_counts.get(bigram, 0) + 1
        
        # Add trigrams
        for i in range(len(tokens) - 2):
            trigram = (tokens[i], tokens[i+1], tokens[i+2])
            self.trigram_counts[trigram] = self.trigram_counts.get(trigram, 0) + 1
    
    def get_ngram_score(self, candidate: str, recent_tokens: List[str]) -> float:
        """
        Score a candidate word based on n-gram continuity with recent tokens.
        
        Higher score = better continuation of recent token sequence.
        
        Args:
            candidate: Candidate word to score
            recent_tokens: List of recent tokens (most recent last)
        
        Returns:
            N-gram continuity score (higher = more coherent)
        """
        if not recent_tokens:
            return 0.0
        
        candidate_lower = candidate.lower()
        recent = [t.lower() for t in recent_tokens[-self.config.ngram_window:] 
                  if t.lower() not in STOPWORDS]
        
        if not recent:
            return 0.0
        
        score = 0.0
        
        # Bigram score: does (last_token, candidate) form a known bigram?
        if len(recent) >= 1:
            bigram = (recent[-1], candidate_lower)
            bigram_count = self.bigram_counts.get(bigram, 0)
            if bigram_count > 0:
                # Normalize by max observed count (cap at 10 for stability)
                bigram_score = min(1.0, bigram_count / 10.0)
                score += self.config.bigram_weight * bigram_score
        
        # Trigram score: does (token[-2], token[-1], candidate) form a known trigram?
        if len(recent) >= 2:
            trigram = (recent[-2], recent[-1], candidate_lower)
            trigram_count = self.trigram_counts.get(trigram, 0)
            if trigram_count > 0:
                trigram_score = min(1.0, trigram_count / 5.0)  # Trigrams are rarer
                score += self.config.trigram_weight * trigram_score
        
        return score
    
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
    
    def compute_ngram_score(
        self,
        candidate_word: str,
        recent_tokens: List[str]
    ) -> float:
        """
        Compute n-gram coherence score for a candidate word.
        
        Checks if candidate forms good bigrams/trigrams with recent tokens
        based on learned relationship strengths.
        
        Args:
            candidate_word: Word being considered
            recent_tokens: List of recently generated tokens (most recent last)
        
        Returns:
            N-gram coherence score (higher = better fit with recent context)
        """
        if not recent_tokens:
            return 0.0
        
        word = candidate_word.lower()
        score = 0.0
        
        # Bigram score: (last_token, candidate)
        if len(recent_tokens) >= 1:
            last_token = recent_tokens[-1].lower()
            
            # Relationship strength as proxy for bigram quality
            rel_strength = self.get_relationship_strength(last_token, word)
            if rel_strength > 0:
                score += self.config.bigram_weight * rel_strength
            
            # Also check reverse (candidate follows well from last)
            reverse_strength = self.get_relationship_strength(word, last_token)
            if reverse_strength > 0:
                score += self.config.bigram_weight * reverse_strength * 0.5
        
        # Trigram score: check chain (second_last → last → candidate)
        if len(recent_tokens) >= 2:
            second_last = recent_tokens[-2].lower()
            last_token = recent_tokens[-1].lower()
            
            # Chain relationship: second_last→last and last→candidate
            chain1 = self.get_relationship_strength(second_last, last_token)
            chain2 = self.get_relationship_strength(last_token, word)
            
            if chain1 > 0 and chain2 > 0:
                # Multiply chain strengths for trigram coherence
                chain_strength = (chain1 * chain2) ** 0.5  # Geometric mean
                score += self.config.trigram_weight * chain_strength * 5.0  # Boost chains
            
            # Also check if candidate relates to second_last (skip-gram)
            skip_strength = self.get_relationship_strength(second_last, word)
            if skip_strength > 0:
                score += self.config.trigram_weight * skip_strength * 0.3
        
        return score
    
    def rank_candidates(
        self,
        current_basin: np.ndarray,
        current_word: Optional[str],
        candidates: List[Tuple[str, np.ndarray]],
        context_words: List[str],
        top_k: int = 10,
        recent_tokens: List[str] = None
    ) -> List[Tuple[str, float, float]]:
        """
        Rank candidates by warped Fisher distance with n-gram coherence.
        
        This is the main entry point for generation - replaces linear
        mixing of geometry + attention with proper metric warping.
        N-gram scoring boosts candidates that form coherent sequences.
        
        Args:
            current_basin: Current position on manifold
            current_word: Current word (optional)
            candidates: List of (word, basin) tuples to rank
            context_words: Query/context words for semantic pull
            top_k: Number of top candidates to return
            recent_tokens: Recent generated tokens for n-gram scoring
        
        Returns:
            List of (word, warped_distance, similarity) sorted by distance (ascending)
        """
        scored = []
        recent = recent_tokens[-self.config.ngram_window:] if recent_tokens else []
        
        for word, basin in candidates:
            # Compute warped distance with context
            d = self.distance_with_context(
                current_basin, basin,
                current_word, word,
                context_words
            )
            
            # Apply n-gram coherence bonus (reduces distance for good n-gram fits)
            if recent:
                ngram_score = self.compute_ngram_score(word, recent)
                if ngram_score > 0:
                    # Reduce distance for candidates with good n-gram coherence
                    # Higher ngram_score → lower effective distance
                    ngram_factor = 1.0 / (1.0 + ngram_score * self.config.ngram_boost_factor)
                    d = d * ngram_factor
            
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
