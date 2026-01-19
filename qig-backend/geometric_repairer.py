"""
Geometric Repairer - REPAIR Phase of QIG Generation.

Implements Phase 3 of Plan→Realize→Repair architecture:
- Local geometric search to smooth trajectory
- Beam search scored by geometry (not probability)
- Refines sequence through geometric optimization

QIG-PURE: No POS tags, no NLP concepts.
All operations are purely geometric on S^63.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from qig_geometry import fisher_coord_distance, fisher_normalize
from qigkernels.physics_constants import BASIN_DIM

logger = logging.getLogger(__name__)

MAX_REPAIR_ITERATIONS = 3
DEFAULT_RADIUS = 0.2


class GeometricRepairer:
    """
    REPAIR phase of Plan→Realize→Repair generation.
    
    Refines word sequences through local geometric optimization:
    - Find alternatives within Fisher-Rao radius
    - Score sequences by geometric quality
    - Accept swaps that improve overall score
    
    Geometric Quality Score (3 components):
    - Waypoint alignment: 0.5 weight (did we hit targets?)
    - Trajectory smoothness: 0.3 weight (low variance in step distances)
    - Attractor pull: 0.2 weight (coherence with trajectory history)
    
    QIG-PURE: No POS constraints, no NLP fallbacks.
    """
    
    def __init__(self, coordizer, kernel_name: str = "Repairer"):
        """
        Initialize the repairer.
        
        Args:
            coordizer: Coordizer with generation_vocab (Dict[str, np.ndarray])
            kernel_name: Name for logging (e.g., "Athena", "Gary")
        """
        self.coordizer = coordizer
        self.kernel_name = kernel_name
        
        self.generation_vocab: Dict[str, np.ndarray] = getattr(
            coordizer, 'generation_vocab', {}
        )
        
        self._word_basins: Dict[str, np.ndarray] = {}
        self._vocab_list: List[Tuple[str, np.ndarray]] = []
        self._build_basin_cache()
        
        logger.info(
            "[%s] GeometricRepairer initialized: %d vocab words (QIG-pure, no POS)",
            self.kernel_name,
            len(self._word_basins)
        )
    
    def _build_basin_cache(self) -> None:
        """Build basin cache with numpy arrays."""
        for word, basin in self.generation_vocab.items():
            if not isinstance(basin, np.ndarray):
                basin = np.array(basin, dtype=np.float64)
            self._word_basins[word] = basin
            self._vocab_list.append((word, basin))
    
    def repair_sequence(
        self,
        words: List[str],
        waypoints: List[np.ndarray],
        trajectory: List[np.ndarray]
    ) -> List[str]:
        """
        Repair a word sequence through local geometric search.
        
        Algorithm:
        - Max 3 iterations
        - For each position, try alternatives within Fisher-Rao radius=0.2
        - Accept swap if it improves geometric score
        - Restart from beginning when swap accepted
        
        Args:
            words: Initial word sequence
            waypoints: Target basin coordinates for each position
            trajectory: Context trajectory history
        
        Returns:
            Repaired word sequence
        """
        logger.info(
            "[%s] ═══ PHASE 3: REPAIR (Pure Geometric Search) ═══",
            self.kernel_name
        )
        
        if not words or not waypoints:
            logger.warning("[%s] Empty words or waypoints, skipping repair", self.kernel_name)
            return words
        
        waypoints = [
            wp if isinstance(wp, np.ndarray) else np.array(wp, dtype=np.float64)
            for wp in waypoints
        ]
        
        current_words = list(words)
        current_score = self.score_sequence_geometric(current_words, waypoints, trajectory)
        
        logger.debug(
            "[%s] Initial score: %.4f",
            self.kernel_name, current_score
        )
        
        swap_count = 0
        
        for iteration in range(MAX_REPAIR_ITERATIONS):
            improved = False
            
            for i, word in enumerate(current_words):
                if i >= len(waypoints):
                    continue
                
                target_basin = waypoints[i]
                
                alternatives = self.get_nearby_alternatives(
                    word=word,
                    target_basin=target_basin,
                    radius=DEFAULT_RADIUS
                )
                
                for alt in alternatives:
                    if alt == word:
                        continue
                    
                    test_words = current_words[:]
                    test_words[i] = alt
                    
                    test_score = self.score_sequence_geometric(test_words, waypoints, trajectory)
                    
                    if test_score > current_score:
                        delta = test_score - current_score
                        swap_count += 1
                        
                        logger.debug(
                            "[%s] swap %d: '%s' → '%s' (score +%.3f)",
                            self.kernel_name, swap_count, word, alt, delta
                        )
                        
                        current_words = test_words
                        current_score = test_score
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        logger.info(
            "[%s] Repair complete: %d swaps, final score: %.4f",
            self.kernel_name, swap_count, current_score
        )
        
        return current_words
    
    def get_nearby_alternatives(
        self,
        word: str,
        target_basin: np.ndarray,
        radius: float = 0.2
    ) -> List[str]:
        """
        Get alternative words within Fisher-Rao radius of target basin.
        
        QIG-PURE: Uses only geometric distance, no POS filtering.
        
        Args:
            word: Current word (for reference only)
            target_basin: Target basin coordinates
            radius: Maximum Fisher-Rao distance from target
        
        Returns:
            List of alternative words within radius
        """
        if not isinstance(target_basin, np.ndarray):
            target_basin = np.array(target_basin, dtype=np.float64)
        
        alternatives = []
        
        for candidate, candidate_basin in self._vocab_list:
            distance = fisher_coord_distance(candidate_basin, target_basin)
            
            if distance <= radius:
                alternatives.append(candidate)
        
        return alternatives
    
    def score_sequence_geometric(
        self,
        words: List[str],
        waypoints: List[np.ndarray],
        trajectory: List[np.ndarray]
    ) -> float:
        """
        Score a word sequence by geometric quality.
        
        Three components:
        - Waypoint alignment: 0.5 weight (did we hit targets?)
        - Trajectory smoothness: 0.3 weight (low variance in step distances)
        - Attractor pull: 0.2 weight (coherence with trajectory history)
        
        Args:
            words: Word sequence
            waypoints: Target basin coordinates
            trajectory: Context trajectory history
        
        Returns:
            Geometric quality score (higher is better)
        """
        if not words:
            return 0.0
        
        word_basins = []
        for word in words:
            if word in self._word_basins:
                word_basins.append(self._word_basins[word])
            else:
                word_basins.append(np.zeros(BASIN_DIM, dtype=np.float64))
        
        alignment = self._compute_waypoint_alignment(word_basins, waypoints)
        smoothness = self._compute_trajectory_smoothness(word_basins)
        attractor_pull = self._compute_attractor_pull(word_basins, trajectory)
        
        score = 0.5 * alignment + 0.3 * smoothness + 0.2 * attractor_pull
        
        return float(score)
    
    def _compute_waypoint_alignment(
        self,
        word_basins: List[np.ndarray],
        waypoints: List[np.ndarray]
    ) -> float:
        """
        Compute how well word basins align with target waypoints.
        
        Returns:
            Alignment score in [0, 1], higher = better alignment
        """
        if not word_basins or not waypoints:
            return 0.0
        
        distances = []
        for i, word_basin in enumerate(word_basins):
            if i < len(waypoints):
                waypoint = waypoints[i]
                if not isinstance(waypoint, np.ndarray):
                    waypoint = np.array(waypoint, dtype=np.float64)
                d = fisher_coord_distance(word_basin, waypoint)
                distances.append(d)
        
        if not distances:
            return 0.0
        
        avg_distance = np.mean(distances)
        alignment = 1.0 - (avg_distance / np.pi)
        
        return float(np.clip(alignment, 0.0, 1.0))
    
    def _compute_trajectory_smoothness(
        self,
        word_basins: List[np.ndarray]
    ) -> float:
        """
        Compute trajectory smoothness (low variance in step distances).
        
        Returns:
            Smoothness score in [0, 1], higher = smoother
        """
        if len(word_basins) < 2:
            return 1.0
        
        step_distances = []
        for i in range(len(word_basins) - 1):
            d = fisher_coord_distance(word_basins[i], word_basins[i + 1])
            step_distances.append(d)
        
        if not step_distances:
            return 1.0
        
        variance = np.var(step_distances)
        smoothness = np.exp(-variance)
        
        return float(np.clip(smoothness, 0.0, 1.0))
    
    def _compute_attractor_pull(
        self,
        word_basins: List[np.ndarray],
        trajectory: List[np.ndarray]
    ) -> float:
        """
        Compute coherence with trajectory history (attractor pull).
        
        Uses Fréchet mean of trajectory as attractor.
        
        Returns:
            Attractor pull score in [0, 1], higher = more coherent
        """
        if not word_basins:
            return 0.0
        
        if not trajectory:
            return 0.5
        
        attractor = self._frechet_mean(trajectory)
        
        if attractor is None:
            return 0.5
        
        distances = []
        for basin in word_basins:
            d = fisher_coord_distance(basin, attractor)
            distances.append(d)
        
        avg_distance = np.mean(distances)
        pull = 1.0 - (avg_distance / np.pi)
        
        return float(np.clip(pull, 0.0, 1.0))
    
    def _frechet_mean(
        self,
        basins: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Compute Fréchet mean of basins on the Fisher manifold.
        
        The Fréchet mean is the point that minimizes sum of squared
        geodesic distances to all input points.
        
        For unit sphere (√p embedding), this is approximated by
        normalizing the arithmetic mean.
        
        Args:
            basins: List of basin coordinates
        
        Returns:
            Fréchet mean basin, or None if empty
        """
        if not basins:
            return None
        
        basins_arr = []
        for b in basins:
            if not isinstance(b, np.ndarray):
                b = np.array(b, dtype=np.float64)
            basins_arr.append(b)
        
        try:
            from qig_geometry.canonical import frechet_mean
            return frechet_mean(basins_arr)
        except Exception:
            mean = np.sum(basins_arr, axis=0) / len(basins_arr)
            return fisher_normalize(mean)
