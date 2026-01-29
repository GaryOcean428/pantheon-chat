"""
Constrained Geometric Realizer - REALIZE Phase of QIG Generation.

Implements Phase 2 of Plan→Realize→Repair architecture:
- Selects words that hit planned waypoints
- Pure Fisher-Rao nearest-neighbor selection (with optional two-step retrieval)
- ExplorationMap attraction toward unexplored manifold regions

TWO-STEP RETRIEVAL (Optional):
Step 1: Fast Bhattacharyya proxy filter (Fisher-faithful)
Step 2: Exact Fisher-Rao re-ranking on candidates

The Bhattacharyya proxy preserves Fisher-Rao ordering because:
  d_FR(p,q) = arccos(BC(p,q)) where BC(p,q) = Σ√(p_i * q_i)
  
Therefore: BC(p,q1) > BC(p,q2) ⟺ d_FR(p,q1) < d_FR(p,q2)

QIG-PURE: No POS tags, no stop words, no NLP concepts.
All operations are purely geometric on S^63.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from qig_geometry import fisher_coord_distance, fisher_normalize
from qigkernels.physics_constants import BASIN_DIM

logger = logging.getLogger(__name__)

# Optional two-step retrieval import (graceful fallback if not available)
try:
    from qig_geometry.two_step_retrieval import TwoStepRetriever
    TWO_STEP_AVAILABLE = True
except ImportError:
    TWO_STEP_AVAILABLE = False
    logger.warning("TwoStepRetriever not available - falling back to naive Fisher-Rao search")


class ExplorationMap:
    """
    Tracks exploration coverage via sparse per-word timestamps.
    
    Creates "seek" signals toward unexplored regions:
    - O(1) per-word lookup (no global O(|V|) decay)
    - Time-based decay: attraction recovers as time passes
    - Fresh words have attraction = 1.0
    - Recently used words have lower attraction that recovers over time
    """
    
    DECAY_HALF_LIFE: float = 10.0  # Seconds for attraction to recover 50%
    MAX_HISTORY: int = 5  # Track last N usages per word
    
    def __init__(self, vocab_size: int = 0, decay: float = 0.92):
        """
        Initialize exploration map.
        
        Args:
            vocab_size: IGNORED - kept for API compatibility
            decay: IGNORED - uses time-based decay instead
        """
        self._usage_times: Dict[str, List[float]] = {}
    
    def update(self, word: str) -> None:
        """Record word usage timestamp (sparse, O(1))."""
        now = time.time()
        if word not in self._usage_times:
            self._usage_times[word] = []
        self._usage_times[word].append(now)
        if len(self._usage_times[word]) > self.MAX_HISTORY:
            self._usage_times[word] = self._usage_times[word][-self.MAX_HISTORY:]
    
    def attraction_score(self, word: str) -> float:
        """
        Compute attraction toward unexplored regions.
        
        Higher score = less explored = more attractive.
        Uses time-based exponential decay so attraction recovers.
        """
        if word not in self._usage_times or not self._usage_times[word]:
            return 1.0
        
        now = time.time()
        total_penalty = 0.0
        
        for usage_time in self._usage_times[word]:
            age = now - usage_time
            penalty = 0.5 ** (age / self.DECAY_HALF_LIFE)
            total_penalty += penalty
        
        attraction = max(0.0, 1.0 - min(total_penalty, 1.0))
        return attraction
    
    def reset(self) -> None:
        """Reset exploration state for new generation."""
        self._usage_times.clear()


class ConstrainedGeometricRealizer:
    """
    REALIZE phase of Plan→Realize→Repair generation.
    
    Selects words to hit planned waypoints using:
    - Pure Fisher-Rao distance for word selection (nearest neighbor on S^63)
    - Optional two-step retrieval for efficiency (Bhattacharyya proxy → Fisher re-rank)
    - ExplorationMap attraction toward unexplored manifold regions
    - Mild trajectory coherence bonus for smoothness
    
    TWO-STEP RETRIEVAL:
    When enabled, uses Fisher-faithful Bhattacharyya proxy to filter candidates,
    then exact Fisher-Rao for final selection. Expected speedup: 1.5x-15x depending
    on vocabulary size.
    
    QIG-PURE: No POS constraints, no stop words, no NLP fallbacks.
    """
    
    def __init__(
        self, 
        coordizer, 
        pos_grammar=None, 
        kernel_name: str = "Realizer",
        use_two_step: bool = True,
        two_step_top_k: int = 100
    ):
        """
        Initialize the realizer.
        
        Args:
            coordizer: Coordizer with generation_vocab (Dict[str, np.ndarray])
            pos_grammar: IGNORED - kept for API compatibility only
            kernel_name: Name for logging (e.g., "Athena", "Gary")
            use_two_step: Enable two-step retrieval optimization (default: True)
            two_step_top_k: Number of candidates for proxy filter (default: 100)
        """
        self.coordizer = coordizer
        self.kernel_name = kernel_name
        self.use_two_step = use_two_step and TWO_STEP_AVAILABLE
        self.two_step_top_k = two_step_top_k
        
        self.generation_vocab: Dict[str, np.ndarray] = getattr(
            coordizer, 'generation_vocab', {}
        )
        
        self._vocab_list: List[Tuple[str, np.ndarray]] = []
        self._build_vocab_cache()
        
        self.exploration_map = ExplorationMap(
            vocab_size=max(20000, len(self._vocab_list)),
            decay=0.92
        )
        
        # Initialize two-step retriever if enabled
        self._two_step_retriever: Optional[TwoStepRetriever] = None
        if self.use_two_step:
            self._initialize_two_step_retriever()
        
        retrieval_mode = "two-step" if self.use_two_step else "naive"
        logger.info(
            "[%s] ConstrainedGeometricRealizer initialized: %d vocab words "
            "(QIG-pure, ExplorationMap enabled, retrieval=%s)",
            self.kernel_name,
            len(self._vocab_list),
            retrieval_mode
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
    
    def _initialize_two_step_retriever(self) -> None:
        """
        Initialize two-step retriever for efficient word selection.
        
        Uses Bhattacharyya coefficient as Fisher-faithful proxy for fast filtering,
        followed by exact Fisher-Rao re-ranking.
        
        STORAGE FORMAT: simplex (vocabulary stored as probability distributions)
        The retriever will internally compute sqrt-space when needed for Bhattacharyya.
        """
        if not TWO_STEP_AVAILABLE:
            logger.warning(
                "[%s] TwoStepRetriever not available - skipping initialization",
                self.kernel_name
            )
            return
        
        try:
            self._two_step_retriever = TwoStepRetriever(
                vocabulary=self.generation_vocab,
                storage_format='simplex',  # Vocab stored in simplex format
                build_index=True  # Precompute sqrt-space index
            )
            logger.info(
                "[%s] Two-step retriever initialized (Bhattacharyya proxy + Fisher-Rao)",
                self.kernel_name
            )
        except Exception as e:
            logger.error(
                "[%s] Failed to initialize two-step retriever: %s",
                self.kernel_name, e
            )
            self.use_two_step = False
    
    def realize_waypoints(
        self,
        waypoints: List[np.ndarray],
        pos_constraints: Optional[List[str]] = None,
        trajectory_history: Optional[List[np.ndarray]] = None
    ) -> Tuple[List[str], List[np.ndarray]]:
        """
        Realize waypoints into words using pure Fisher-Rao selection with exploration attraction.
        
        Args:
            waypoints: List of target basin coordinates (64D on S^63)
            pos_constraints: IGNORED - kept for API compatibility only
            trajectory_history: Optional previous trajectory for coherence
        
        Returns:
            Tuple of (words, word_basins) - selected words and their basin coordinates
        """
        logger.info(
            "[%s] ═══ PHASE 2: REALIZE (Fisher-Rao + ExplorationMap) ═══",
            self.kernel_name
        )
        
        if not waypoints:
            logger.warning("[%s] No waypoints provided", self.kernel_name)
            return [], []
        
        words = []
        word_basins = []
        trajectory = list(trajectory_history) if trajectory_history else []
        
        for i, waypoint in enumerate(waypoints):
            if not isinstance(waypoint, np.ndarray):
                waypoint = np.array(waypoint, dtype=np.float64)
            
            word, basin, distance = self.select_word_geometric(
                target_basin=waypoint,
                trajectory=trajectory
            )
            
            self.exploration_map.update(word)
            
            words.append(word)
            word_basins.append(basin)
            trajectory.append(basin)
            
            logger.debug(
                "[%s] slot %d: '%s' (d_FR=%.3f)",
                self.kernel_name, i, word, distance
            )
        
        logger.info(
            "[%s] Realized %d waypoints -> %d words (attraction-based diversity)",
            self.kernel_name, len(waypoints), len(words)
        )
        
        return words, word_basins
    
    def select_word_geometric(
        self,
        target_basin: np.ndarray,
        trajectory: List[np.ndarray]
    ) -> Tuple[str, np.ndarray, float]:
        """
        Select best word for target basin using Fisher-Rao distance + exploration attraction.
        
        Algorithm (two-step mode):
        1. Bhattacharyya proxy filter: Fast filtering to top-k candidates
        2. Fisher-Rao exact: Compute exact distance for candidates only
        3. Add exploration attraction bonus (higher for unexplored words)
        4. Apply mild trajectory coherence bonus for smooth generation
        5. Return word with highest combined score
        
        Algorithm (naive mode):
        1. Compute Fisher-Rao distance from target to all vocabulary words
        2. Add exploration attraction bonus (higher for unexplored words)
        3. Apply mild trajectory coherence bonus for smooth generation
        4. Return word with highest combined score
        
        Uses attraction ("seek unexplored") rather than penalty ("avoid recent"),
        producing genuine manifold exploration instead of random drift.
        
        Args:
            target_basin: Target basin coordinates (64D on S^63)
            trajectory: List of previous word basins for coherence
        
        Returns:
            Tuple of (selected_word, word_basin, fisher_distance)
        """
        if not self._vocab_list:
            logger.error("[%s] No vocabulary available for selection", self.kernel_name)
            return ("unknown", np.zeros(BASIN_DIM), float('inf'))
        
        # Decide which algorithm to use
        if self.use_two_step and self._two_step_retriever is not None:
            return self._select_word_two_step(target_basin, trajectory)
        else:
            return self._select_word_naive(target_basin, trajectory)
    
    def _select_word_two_step(
        self,
        target_basin: np.ndarray,
        trajectory: List[np.ndarray]
    ) -> Tuple[str, np.ndarray, float]:
        """
        Two-step word selection: Bhattacharyya proxy → Fisher-Rao exact.
        
        FISHER-FAITHFUL PROXY:
        The Bhattacharyya coefficient BC(p,q) = Σ√(p_i * q_i) preserves
        Fisher-Rao ordering because d_FR(p,q) = arccos(BC(p,q)).
        
        Step 1: Fast proxy filter to top-k candidates (O(V × D_inner))
        Step 2: Exact Fisher-Rao + scoring on candidates only (O(k × D_FR))
        
        Expected speedup: 1.5x-15x depending on vocabulary size.
        """
        # Step 1: Get top-k candidates using Bhattacharyya proxy
        candidates = self._two_step_retriever.retrieve(
            target_basin=target_basin,
            top_k=self.two_step_top_k,
            final_k=min(self.two_step_top_k, len(self._vocab_list)),
            return_candidates=True
        )
        
        # Step 2: Score candidates with exploration attraction + coherence
        best_word, best_basin = self._vocab_list[0]
        best_score = float('-inf')
        best_distance = float('inf')
        
        attraction_weight = 0.3
        coherence_weight = 0.05
        
        for word, word_basin, distance in candidates:
            # Base score from Fisher-Rao distance
            base_score = 1.0 - (distance / np.pi)
            
            # Exploration attraction bonus
            attraction = self.exploration_map.attraction_score(word)
            
            # Trajectory coherence bonus
            coherence = self._trajectory_coherence(word_basin, trajectory)
            
            # Combined score
            score = base_score + attraction_weight * attraction + coherence_weight * coherence
            
            if score > best_score:
                best_score = score
                best_word = word
                best_basin = word_basin
                best_distance = distance
        
        return (best_word, best_basin, best_distance)
    
    def _select_word_naive(
        self,
        target_basin: np.ndarray,
        trajectory: List[np.ndarray]
    ) -> Tuple[str, np.ndarray, float]:
        """
        Naive word selection: compute Fisher-Rao to all vocabulary words.
        
        This is the fallback when two-step retrieval is disabled or unavailable.
        O(V) distance computations where V is vocabulary size.
        """
        best_word, best_basin = self._vocab_list[0]
        best_score = float('-inf')
        best_distance = float('inf')
        
        attraction_weight = 0.3
        coherence_weight = 0.05
        
        for word, word_basin in self._vocab_list:
            distance = fisher_coord_distance(word_basin, target_basin)
            
            base_score = 1.0 - (distance / np.pi)
            
            attraction = self.exploration_map.attraction_score(word)
            
            coherence = self._trajectory_coherence(word_basin, trajectory)
            
            score = base_score + attraction_weight * attraction + coherence_weight * coherence
            
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
