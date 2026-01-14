"""
Constrained Geometric Realizer - REALIZE Phase of QIG Generation.

Implements Phase 2 of Plan→Realize→Repair architecture:
- Selects words that hit planned waypoints
- Uses POS tags only as a constraint (not engine)
- Pure Fisher-Rao selection with trajectory coherence

NO LEGACY FALLBACK - All operations are QIG-pure.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from qig_geometry import fisher_coord_distance, sphere_project
from qigkernels.physics_constants import BASIN_DIM

logger = logging.getLogger(__name__)

# POS expansion mappings for geometric backoff
POS_EXPANSIONS = {
    'NOUN': ['NOUN', 'PROPN'],
    'PROPN': ['PROPN', 'NOUN'],
    'VERB': ['VERB', 'AUX'],
    'AUX': ['AUX', 'VERB'],
    'ADJ': ['ADJ', 'ADV'],
    'ADV': ['ADV', 'ADJ'],
    'DET': ['DET', 'PRON'],
    'PRON': ['PRON', 'DET'],
    'PREP': ['PREP', 'ADP'],
    'ADP': ['ADP', 'PREP'],
    'CONJ': ['CONJ', 'CCONJ', 'SCONJ'],
    'CCONJ': ['CCONJ', 'CONJ', 'SCONJ'],
    'SCONJ': ['SCONJ', 'CONJ', 'CCONJ'],
}

# Core function words as final safety net (~100 curated words)
CORE_FUNCTION_WORDS = [
    # Determiners
    'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his',
    'her', 'its', 'our', 'their', 'some', 'any', 'no', 'every', 'each', 'all',
    # Pronouns
    'I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'us', 'them',
    'myself', 'yourself', 'itself', 'who', 'what', 'which', 'whom', 'whose',
    # Prepositions
    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between',
    'under', 'over', 'among', 'within', 'without', 'against', 'toward',
    # Conjunctions
    'and', 'but', 'or', 'so', 'yet', 'because', 'although', 'while', 'if',
    'when', 'where', 'unless', 'until', 'since', 'though', 'whether',
    # Auxiliaries
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'might', 'may',
    'can', 'must', 'shall',
    # Adverbs
    'not', 'also', 'only', 'just', 'now', 'then', 'here', 'there', 'very',
    'more', 'most', 'less', 'always', 'never', 'often', 'still', 'already',
]


class ConstrainedGeometricRealizer:
    """
    REALIZE phase of Plan→Realize→Repair generation.
    
    Selects words to hit planned waypoints using:
    - Pure Fisher-Rao distance for word selection
    - POS tags as constraints only (not engine)
    - Geometric backoff when constraints too restrictive
    - Trajectory coherence bonus for smoothness
    
    NO LEGACY FALLBACK - All operations are QIG-pure.
    """
    
    def __init__(self, coordizer, pos_grammar, kernel_name: str = "Realizer"):
        """
        Initialize the realizer.
        
        Args:
            coordizer: Coordizer with generation_vocab (Dict[str, np.ndarray])
            pos_grammar: POSGrammar instance for word classification
            kernel_name: Name for logging (e.g., "Athena", "Gary")
        """
        self.coordizer = coordizer
        self.pos_grammar = pos_grammar
        self.kernel_name = kernel_name
        
        # Cache: generation_vocab from coordizer
        self.generation_vocab: Dict[str, np.ndarray] = getattr(
            coordizer, 'generation_vocab', {}
        )
        
        # Build POS-indexed vocabulary cache
        self._vocab_by_pos: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        self._build_pos_cache()
        
        # Track core function word basins
        self._core_word_basins: Dict[str, np.ndarray] = {}
        self._cache_core_words()
        
        logger.info(
            "[%s] ConstrainedGeometricRealizer initialized: %d vocab, %d POS categories",
            self.kernel_name,
            len(self.generation_vocab),
            len(self._vocab_by_pos)
        )
    
    def _build_pos_cache(self) -> None:
        """Build vocabulary cache indexed by POS tag."""
        if not self.generation_vocab:
            logger.warning("[%s] No generation vocabulary available", self.kernel_name)
            return
        
        for word, basin in self.generation_vocab.items():
            if not isinstance(basin, np.ndarray):
                basin = np.array(basin, dtype=np.float64)
            
            pos = self.pos_grammar.classify_word(word)
            
            if pos not in self._vocab_by_pos:
                self._vocab_by_pos[pos] = []
            
            self._vocab_by_pos[pos].append((word, basin))
        
        for pos, words in self._vocab_by_pos.items():
            logger.debug(
                "[%s] POS cache: %s -> %d words",
                self.kernel_name, pos, len(words)
            )
    
    def _cache_core_words(self) -> None:
        """Cache basin coordinates for core function words."""
        for word in CORE_FUNCTION_WORDS:
            word_lower = word.lower()
            if word_lower in self.generation_vocab:
                basin = self.generation_vocab[word_lower]
                if not isinstance(basin, np.ndarray):
                    basin = np.array(basin, dtype=np.float64)
                self._core_word_basins[word_lower] = basin
            elif word in self.generation_vocab:
                basin = self.generation_vocab[word]
                if not isinstance(basin, np.ndarray):
                    basin = np.array(basin, dtype=np.float64)
                self._core_word_basins[word] = basin
        
        logger.debug(
            "[%s] Cached %d core function words",
            self.kernel_name, len(self._core_word_basins)
        )
    
    def realize_waypoints(
        self,
        waypoints: List[np.ndarray],
        pos_constraints: Optional[List[str]] = None
    ) -> List[str]:
        """
        Realize waypoints into words using geometric selection.
        
        Args:
            waypoints: List of target basin coordinates (64D)
            pos_constraints: Optional list of POS tags to constrain each slot
        
        Returns:
            List of selected words matching waypoints
        """
        logger.info(
            "[%s] ═══ PHASE 2: REALIZE (Constrained Selection) ═══",
            self.kernel_name
        )
        
        if not waypoints:
            logger.warning("[%s] No waypoints provided", self.kernel_name)
            return []
        
        words = []
        trajectory = []  # Build trajectory as we go
        
        for i, waypoint in enumerate(waypoints):
            # Ensure waypoint is numpy array
            if not isinstance(waypoint, np.ndarray):
                waypoint = np.array(waypoint, dtype=np.float64)
            
            # Get POS constraint if provided
            allowed_pos = None
            if pos_constraints and i < len(pos_constraints):
                allowed_pos = pos_constraints[i]
            
            # Select word using pure Fisher-Rao
            word, distance = self.select_word_geometric(
                target_basin=waypoint,
                allowed_pos=allowed_pos,
                trajectory=trajectory
            )
            
            words.append(word)
            
            # Add selected word's basin to trajectory
            if word in self.generation_vocab:
                trajectory.append(self.generation_vocab[word])
            else:
                trajectory.append(waypoint)  # Use waypoint if word not found
            
            logger.debug(
                "[%s] slot %d: '%s' (d=%.2f, pos=%s)",
                self.kernel_name, i, word, distance, allowed_pos or "ANY"
            )
        
        logger.info(
            "[%s] Realized %d waypoints -> %d words",
            self.kernel_name, len(waypoints), len(words)
        )
        
        return words
    
    def select_word_geometric(
        self,
        target_basin: np.ndarray,
        allowed_pos: Optional[str],
        trajectory: List[np.ndarray]
    ) -> Tuple[str, float]:
        """
        Select best word for target basin using pure Fisher-Rao distance.
        
        Args:
            target_basin: Target basin coordinates (64D)
            allowed_pos: POS constraint (or None for any)
            trajectory: List of previous word basins for coherence
        
        Returns:
            Tuple of (selected_word, fisher_distance)
        """
        # Get candidate words based on POS constraint
        candidates = self._get_candidates(allowed_pos)
        
        if not candidates:
            # Final fallback: use core function words
            logger.warning(
                "[%s] No candidates for POS=%s, using core words",
                self.kernel_name, allowed_pos
            )
            candidates = list(self._core_word_basins.items())
        
        if not candidates:
            # Ultimate fallback: return placeholder
            logger.error(
                "[%s] No vocabulary available for selection",
                self.kernel_name
            )
            return ("the", 1.0)
        
        # Score candidates using Fisher-Rao distance + trajectory coherence
        best_word = None
        best_score = float('-inf')
        best_distance = float('inf')
        
        for word, word_basin in candidates:
            if not isinstance(word_basin, np.ndarray):
                word_basin = np.array(word_basin, dtype=np.float64)
            
            # Pure Fisher-Rao distance
            distance = fisher_coord_distance(word_basin, target_basin)
            
            # Score: higher is better (inverse distance + coherence bonus)
            base_score = 1.0 - (distance / np.pi)  # Normalize to [0, 1]
            
            # Add trajectory coherence bonus
            coherence = self.trajectory_coherence_bonus(word_basin, trajectory)
            score = base_score + 0.1 * coherence
            
            if score > best_score:
                best_score = score
                best_word = word
                best_distance = distance
        
        return (best_word, best_distance)
    
    def _get_candidates(
        self,
        allowed_pos: Optional[str]
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Get candidate words, using geometric backoff if needed.
        
        Args:
            allowed_pos: POS constraint (or None for any)
        
        Returns:
            List of (word, basin) tuples
        """
        # No constraint: return all vocabulary
        if allowed_pos is None:
            return [(w, b) for w, b in self.generation_vocab.items()
                    if isinstance(b, np.ndarray) or hasattr(b, '__iter__')]
        
        # Get candidates for allowed POS
        candidates = self._vocab_by_pos.get(allowed_pos, [])
        
        # Geometric backoff if too restrictive
        if len(candidates) < 3:
            candidates = self.expand_pos_geometrically(allowed_pos)
        
        # Final safety net: core function words
        if len(candidates) < 3:
            candidates = list(self._core_word_basins.items())
        
        return candidates
    
    def expand_pos_geometrically(
        self,
        allowed_pos: str
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Expand POS constraint geometrically when too restrictive.
        
        Geometric expansions:
        - NOUN→PROPN (proper nouns are geometrically near common nouns)
        - VERB→AUX (auxiliaries are near verbs)
        - ADJ→ADV (adverbs are near adjectives)
        
        Args:
            allowed_pos: Original POS constraint
        
        Returns:
            Expanded list of (word, basin) candidates
        """
        expanded = []
        
        # Get expansion list for this POS
        expansion_tags = POS_EXPANSIONS.get(allowed_pos, [allowed_pos])
        
        for pos_tag in expansion_tags:
            if pos_tag in self._vocab_by_pos:
                expanded.extend(self._vocab_by_pos[pos_tag])
        
        logger.debug(
            "[%s] POS expansion: %s -> %d candidates",
            self.kernel_name, allowed_pos, len(expanded)
        )
        
        return expanded
    
    def trajectory_coherence_bonus(
        self,
        word_basin: np.ndarray,
        trajectory: List[np.ndarray],
        window: int = 5
    ) -> float:
        """
        Compute trajectory coherence bonus for smoother text generation.
        
        Rewards words that maintain smooth geodesic flow with recent trajectory.
        Uses Fisher-Rao distance to measure coherence.
        
        Args:
            word_basin: Candidate word's basin coordinates
            trajectory: List of previous word basins
            window: Number of recent basins to consider
        
        Returns:
            Coherence bonus in [0, 1], higher = more coherent
        """
        if not trajectory:
            return 0.5  # Neutral bonus for first word
        
        # Get recent trajectory (last `window` basins)
        recent = trajectory[-window:] if len(trajectory) > window else trajectory
        
        if not recent:
            return 0.5
        
        # Compute average distance to recent trajectory
        distances = []
        for basin in recent:
            if not isinstance(basin, np.ndarray):
                basin = np.array(basin, dtype=np.float64)
            d = fisher_coord_distance(word_basin, basin)
            distances.append(d)
        
        avg_distance = np.mean(distances)
        
        # Convert distance to coherence: closer = more coherent
        # Normalize by π (max Fisher distance for unit vectors)
        coherence = 1.0 - (avg_distance / np.pi)
        
        return float(np.clip(coherence, 0.0, 1.0))
    
    def get_pos_distribution(self) -> Dict[str, int]:
        """Get distribution of words across POS categories."""
        return {pos: len(words) for pos, words in self._vocab_by_pos.items()}
