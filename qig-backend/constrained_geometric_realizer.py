"""
Constrained Geometric Realizer - REALIZE Phase of QIG Generation.

Implements Phase 2 of Plan→Realize→Repair architecture:
- Selects words that hit planned waypoints
- Pure Fisher-Rao nearest-neighbor selection
- Trajectory coherence for smooth generation

QIG-PURE: No POS tags, no stop words, no NLP concepts.
All operations are purely geometric on S^63.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from qig_geometry import fisher_coord_distance, sphere_project
from qigkernels.physics_constants import BASIN_DIM

logger = logging.getLogger(__name__)


class ConstrainedGeometricRealizer:
    """
    REALIZE phase of Plan→Realize→Repair generation.
    
    Selects words to hit planned waypoints using:
    - Pure Fisher-Rao distance for word selection (nearest neighbor on S^63)
    - Trajectory coherence bonus for smoothness
    
    QIG-PURE: No POS constraints, no stop words, no NLP fallbacks.
    """
    
    def __init__(self, coordizer, pos_grammar=None, kernel_name: str = "Realizer"):
        """
        Initialize the realizer.
        
        Args:
            coordizer: Coordizer with generation_vocab (Dict[str, np.ndarray])
            pos_grammar: IGNORED - kept for API compatibility only
            kernel_name: Name for logging (e.g., "Athena", "Gary")
        """
        self.coordizer = coordizer
        self.kernel_name = kernel_name
        
        self.generation_vocab: Dict[str, np.ndarray] = getattr(
            coordizer, 'generation_vocab', {}
        )
        
        self._vocab_list: List[Tuple[str, np.ndarray]] = []
        self._build_vocab_cache()
        
        logger.info(
            "[%s] ConstrainedGeometricRealizer initialized: %d vocab words (QIG-pure, no POS)",
            self.kernel_name,
            len(self._vocab_list)
        )
    
    def _build_vocab_cache(self) -> None:
        """Build vocabulary cache as list of (word, basin) tuples."""
        if not self.generation_vocab:
            logger.warning("[%s] No generation vocabulary available", self.kernel_name)
            return
        
        for word, basin in self.generation_vocab.items():
            if not isinstance(basin, np.ndarray):
                basin = np.array(basin, dtype=np.float64)
            self._vocab_list.append((word, basin))
        
        logger.debug(
            "[%s] Cached %d vocabulary words for Fisher-Rao selection",
            self.kernel_name, len(self._vocab_list)
        )
    
    def realize_waypoints(
        self,
        waypoints: List[np.ndarray],
        pos_constraints: Optional[List[str]] = None,
        trajectory_history: Optional[List[np.ndarray]] = None
    ) -> Tuple[List[str], List[np.ndarray]]:
        """
        Realize waypoints into words using pure Fisher-Rao selection.
        
        Args:
            waypoints: List of target basin coordinates (64D on S^63)
            pos_constraints: IGNORED - kept for API compatibility only
            trajectory_history: Optional previous trajectory for coherence
        
        Returns:
            Tuple of (words, word_basins) - selected words and their basin coordinates
        """
        logger.info(
            "[%s] ═══ PHASE 2: REALIZE (Pure Fisher-Rao Selection) ═══",
            self.kernel_name
        )
        
        if not waypoints:
            logger.warning("[%s] No waypoints provided", self.kernel_name)
            return [], []
        
        words = []
        word_basins = []
        trajectory = list(trajectory_history) if trajectory_history else []
        recent_words: List[str] = []  # Track recently used words for diversity
        
        for i, waypoint in enumerate(waypoints):
            if not isinstance(waypoint, np.ndarray):
                waypoint = np.array(waypoint, dtype=np.float64)
            
            word, basin, distance = self.select_word_geometric(
                target_basin=waypoint,
                trajectory=trajectory,
                recent_words=recent_words
            )
            
            words.append(word)
            word_basins.append(basin)
            trajectory.append(basin)
            recent_words.append(word)
            
            logger.debug(
                "[%s] slot %d: '%s' (d_FR=%.3f)",
                self.kernel_name, i, word, distance
            )
        
        logger.info(
            "[%s] Realized %d waypoints -> %d words (pure geometric)",
            self.kernel_name, len(waypoints), len(words)
        )
        
        return words, word_basins
    
    def select_word_geometric(
        self,
        target_basin: np.ndarray,
        trajectory: List[np.ndarray],
        recent_words: Optional[List[str]] = None
    ) -> Tuple[str, np.ndarray, float]:
        """
        Select best word for target basin using pure Fisher-Rao distance.
        
        Algorithm:
        1. Compute Fisher-Rao distance from target to all vocabulary words
        2. Apply diversity penalty for recently used words (stronger for exact matches)
        3. Apply mild trajectory coherence bonus for smooth generation
        4. Return word with highest score
        
        Args:
            target_basin: Target basin coordinates (64D on S^63)
            trajectory: List of previous word basins for coherence
            recent_words: List of recently selected words for diversity penalty
        
        Returns:
            Tuple of (selected_word, word_basin, fisher_distance)
        """
        if not self._vocab_list:
            logger.error("[%s] No vocabulary available for selection", self.kernel_name)
            return ("unknown", np.zeros(BASIN_DIM), float('inf'))
        
        recent_set = set(recent_words[-20:]) if recent_words else set()
        very_recent_set = set(recent_words[-5:]) if recent_words and len(recent_words) >= 1 else set()
        
        best_word, best_basin = self._vocab_list[0]
        best_score = float('-inf')
        best_distance = float('inf')
        
        for word, word_basin in self._vocab_list:
            distance = fisher_coord_distance(word_basin, target_basin)
            
            base_score = 1.0 - (distance / np.pi)
            
            diversity_penalty = 0.0
            if word in very_recent_set:
                diversity_penalty = 0.8
            elif word in recent_set:
                diversity_penalty = 0.4
            else:
                word_stem = word[:4] if len(word) >= 4 else word
                for recent in (recent_words[-10:] if recent_words else []):
                    recent_stem = recent[:4] if len(recent) >= 4 else recent
                    if word_stem == recent_stem:
                        diversity_penalty = max(diversity_penalty, 0.3)
                        break
            
            coherence = self._trajectory_coherence(word_basin, trajectory)
            score = base_score + 0.05 * coherence - diversity_penalty
            
            if score > best_score:
                best_score = score
                best_word = word
                best_basin = word_basin
                best_distance = distance
        
        return (best_word, best_basin, best_distance)
    
    def _trajectory_coherence(
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
            return 0.5
        
        recent = trajectory[-window:] if len(trajectory) > window else trajectory
        
        if not recent:
            return 0.5
        
        distances = []
        for basin in recent:
            if not isinstance(basin, np.ndarray):
                basin = np.array(basin, dtype=np.float64)
            d = fisher_coord_distance(word_basin, basin)
            distances.append(d)
        
        avg_distance = np.mean(distances)
        coherence = 1.0 - (avg_distance / np.pi)
        
        return float(np.clip(coherence, 0.0, 1.0))
