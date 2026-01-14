"""
Geometric Repairer - REPAIR Phase of QIG Generation.

Implements Phase 3 of Plan→Realize→Repair architecture:
- Local geometric search to smooth trajectory
- Beam search scored by geometry (not probability)
- Refines sequence through geometric optimization

NO LEGACY FALLBACK - All operations are QIG-pure.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from qig_geometry import fisher_coord_distance, sphere_project
from qigkernels.physics_constants import BASIN_DIM

logger = logging.getLogger(__name__)

# Maximum repair iterations
MAX_REPAIR_ITERATIONS = 3

# Default Fisher-Rao radius for alternatives
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
    
    NO LEGACY FALLBACK - All operations are QIG-pure.
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
        
        # Cache: generation_vocab from coordizer
        self.generation_vocab: Dict[str, np.ndarray] = getattr(
            coordizer, 'generation_vocab', {}
        )
        
        # Build word->basin lookup with numpy arrays
        self._word_basins: Dict[str, np.ndarray] = {}
        self._build_basin_cache()
        
        # Build POS-indexed vocabulary cache for same_pos filtering
        self._vocab_by_pos: Dict[str, List[str]] = {}
        self._word_pos: Dict[str, str] = {}
        self._build_pos_cache()
        
        logger.info(
            "[%s] GeometricRepairer initialized: %d vocab words",
            self.kernel_name,
            len(self._word_basins)
        )
    
    def _build_basin_cache(self) -> None:
        """Build basin cache with numpy arrays."""
        for word, basin in self.generation_vocab.items():
            if not isinstance(basin, np.ndarray):
                basin = np.array(basin, dtype=np.float64)
            self._word_basins[word] = basin
    
    def _build_pos_cache(self) -> None:
        """Build vocabulary cache indexed by POS tag (simple suffix heuristics)."""
        verb_suffixes = ('ize', 'ify', 'ate', 'en', 'ing', 'ed')
        noun_suffixes = ('tion', 'sion', 'ment', 'ness', 'ity', 'ance', 'ence', 'er', 'or', 'ist')
        adj_suffixes = ('able', 'ible', 'al', 'ful', 'less', 'ous', 'ive', 'ic', 'ish')
        
        for word in self._word_basins:
            word_lower = word.lower()
            
            if word_lower.endswith('ly') and len(word_lower) > 4:
                pos = 'ADV'
            elif any(word_lower.endswith(s) for s in verb_suffixes):
                pos = 'VERB'
            elif any(word_lower.endswith(s) for s in adj_suffixes):
                pos = 'ADJ'
            elif any(word_lower.endswith(s) for s in noun_suffixes):
                pos = 'NOUN'
            else:
                pos = 'NOUN'  # Default
            
            self._word_pos[word] = pos
            if pos not in self._vocab_by_pos:
                self._vocab_by_pos[pos] = []
            self._vocab_by_pos[pos].append(word)
    
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
            "[%s] ═══ PHASE 3: REPAIR (Local Search) ═══",
            self.kernel_name
        )
        
        if not words or not waypoints:
            logger.warning("[%s] Empty words or waypoints, skipping repair", self.kernel_name)
            return words
        
        # Ensure waypoints are numpy arrays
        waypoints = [
            wp if isinstance(wp, np.ndarray) else np.array(wp, dtype=np.float64)
            for wp in waypoints
        ]
        
        # Work with a copy
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
                
                # Get nearby alternatives (same POS, within Fisher radius)
                alternatives = self.get_nearby_alternatives(
                    word=word,
                    target_basin=target_basin,
                    same_pos=True,
                    radius=DEFAULT_RADIUS
                )
                
                for alt in alternatives:
                    if alt == word:
                        continue
                    
                    # Test swap
                    test_words = current_words[:]
                    test_words[i] = alt
                    
                    test_score = self.score_sequence_geometric(test_words, waypoints, trajectory)
                    
                    if test_score > current_score:
                        delta = test_score - current_score
                        swap_count += 1
                        
                        logger.debug(
                            "[%s] swap %d: '%s' → '%s' (score +%.2f)",
                            self.kernel_name, swap_count, word, alt, delta
                        )
                        
                        current_words = test_words
                        current_score = test_score
                        improved = True
                        break  # Restart from beginning
                
                if improved:
                    break
            
            if not improved:
                # No improvements found in this iteration
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
        same_pos: bool = True,
        radius: float = 0.2
    ) -> List[str]:
        """
        Get alternative words within Fisher-Rao radius of target basin.
        
        Args:
            word: Current word
            target_basin: Target basin coordinates
            same_pos: If True, only return words with same POS as current
            radius: Maximum Fisher-Rao distance from target
        
        Returns:
            List of alternative words within radius
        """
        if not isinstance(target_basin, np.ndarray):
            target_basin = np.array(target_basin, dtype=np.float64)
        
        alternatives = []
        
        # Get candidate pool
        if same_pos and word in self._word_pos:
            word_pos = self._word_pos[word]
            candidate_words = self._vocab_by_pos.get(word_pos, [])
        else:
            candidate_words = list(self._word_basins.keys())
        
        # Find words within radius
        for candidate in candidate_words:
            if candidate not in self._word_basins:
                continue
            
            candidate_basin = self._word_basins[candidate]
            distance = self.fisher_rao_distance(candidate_basin, target_basin)
            
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
        
        # Get basins for words
        word_basins = []
        for word in words:
            if word in self._word_basins:
                word_basins.append(self._word_basins[word])
            else:
                # Use zero vector if word not found (will penalize score)
                word_basins.append(np.zeros(BASIN_DIM, dtype=np.float64))
        
        # 1. Waypoint alignment (0.5 weight)
        alignment = self._compute_waypoint_alignment(word_basins, waypoints)
        
        # 2. Trajectory smoothness (0.3 weight)
        smoothness = self._compute_trajectory_smoothness(word_basins)
        
        # 3. Attractor pull (0.2 weight)
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
        
        # Compare each word basin to corresponding waypoint
        distances = []
        for i, word_basin in enumerate(word_basins):
            if i < len(waypoints):
                waypoint = waypoints[i]
                if not isinstance(waypoint, np.ndarray):
                    waypoint = np.array(waypoint, dtype=np.float64)
                d = self.fisher_rao_distance(word_basin, waypoint)
                distances.append(d)
        
        if not distances:
            return 0.0
        
        # Convert average distance to alignment score
        avg_distance = np.mean(distances)
        alignment = 1.0 - (avg_distance / np.pi)  # Normalize by max distance
        
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
            return 1.0  # Single word is maximally smooth
        
        # Compute step distances
        step_distances = []
        for i in range(len(word_basins) - 1):
            d = self.fisher_rao_distance(word_basins[i], word_basins[i + 1])
            step_distances.append(d)
        
        if not step_distances:
            return 1.0
        
        # Smoothness = inverse of variance
        variance = np.var(step_distances)
        
        # Normalize: low variance -> high smoothness
        # Using exponential decay: smoothness = exp(-variance)
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
            return 0.5  # Neutral score without trajectory context
        
        # Compute Fréchet mean of trajectory
        attractor = self.frechet_mean(trajectory)
        
        if attractor is None:
            return 0.5
        
        # Compute average distance from word basins to attractor
        distances = []
        for basin in word_basins:
            d = self.fisher_rao_distance(basin, attractor)
            distances.append(d)
        
        avg_distance = np.mean(distances)
        pull = 1.0 - (avg_distance / np.pi)  # Normalize by max distance
        
        return float(np.clip(pull, 0.0, 1.0))
    
    def frechet_mean(
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
        
        # Ensure numpy arrays
        basins_arr = []
        for b in basins:
            if not isinstance(b, np.ndarray):
                b = np.array(b, dtype=np.float64)
            basins_arr.append(b)
        
        # Arithmetic mean
        mean = np.mean(basins_arr, axis=0)
        
        # Project to unit sphere
        return sphere_project(mean)
    
    def fisher_rao_distance(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> float:
        """
        Compute Fisher-Rao distance between two basin coordinates.
        
        For unit vectors: d = arccos(a · b)
        
        Args:
            a: First basin coordinate vector
            b: Second basin coordinate vector
        
        Returns:
            Fisher-Rao distance (0 to π)
        """
        return fisher_coord_distance(a, b)
