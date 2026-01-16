"""
Two-Step Retrieval Integration Example
=======================================

This example shows how to integrate Fisher-faithful two-step retrieval
with the ConstrainedGeometricRealizer for efficient word selection in
the REALIZE phase.

Author: Copilot (Ultra Consciousness Protocol ACTIVE)
Date: 2026-01-16
Context: Work Package 2.4 - Integration Example
"""

import numpy as np
from typing import List, Tuple, Optional

from qig_geometry.two_step_retrieval import TwoStepRetriever
from qig_geometry.canonical import BASIN_DIM


class OptimizedGeometricRealizer:
    """
    REALIZE phase with Fisher-faithful two-step retrieval.
    
    Replaces naive O(V) Fisher-Rao search with two-step:
    1. Bhattacharyya proxy filter: O(V × D_inner)
    2. Fisher-Rao exact ranking: O(k × D_FR)
    
    Expected speedup: 1.5x-15x depending on vocabulary size and filtering.
    """
    
    def __init__(self, coordizer, pos_grammar=None, kernel_name: str = "Realizer"):
        """
        Initialize realizer with two-step retrieval.
        
        Args:
            coordizer: Coordizer with generation_vocab
            pos_grammar: Optional POS grammar for filtering
            kernel_name: Name for logging
        """
        self.coordizer = coordizer
        self.pos_grammar = pos_grammar
        self.kernel_name = kernel_name
        
        # Build two-step retriever
        self.retriever = TwoStepRetriever(
            vocabulary=coordizer.generation_vocab,
            storage_format='simplex',  # or 'sqrt' if pre-converted
            build_index=True
        )
        
        print(f"[{kernel_name}] Initialized with two-step retrieval")
    
    def realize_waypoints(
        self,
        waypoints: List[np.ndarray],
        pos_constraints: Optional[List[str]] = None,
        trajectory_history: Optional[List[np.ndarray]] = None
    ) -> Tuple[List[str], List[np.ndarray]]:
        """
        Realize waypoints into words using two-step retrieval.
        
        Args:
            waypoints: Target basin coordinates from PLAN phase
            pos_constraints: Optional POS tags for filtering
            trajectory_history: Optional previous trajectory
            
        Returns:
            (words, word_basins) - selected words and their basins
        """
        print(f"[{self.kernel_name}] ═══ PHASE 2: REALIZE (Two-Step Retrieval) ═══")
        
        if not waypoints:
            return [], []
        
        words = []
        word_basins = []
        trajectory = list(trajectory_history) if trajectory_history else []
        
        for i, waypoint in enumerate(waypoints):
            # Two-step retrieval for this waypoint
            word, basin, distance = self.retriever.retrieve(
                target_basin=waypoint,
                top_k=100,      # Step 1: Bhattacharyya filter to 100 candidates
                final_k=1       # Step 2: Fisher-Rao selection
            )
            
            words.append(word)
            word_basins.append(basin)
            trajectory.append(basin)
            
            if i % 10 == 0:
                print(f"[{self.kernel_name}] Realized {i}/{len(waypoints)} waypoints")
        
        print(f"[{self.kernel_name}] ✓ Realized {len(waypoints)} waypoints")
        
        return words, word_basins
    
    def realize_single_waypoint(
        self,
        waypoint: np.ndarray,
        pos_constraint: Optional[str] = None,
        top_k: int = 100,
        return_candidates: bool = False
    ):
        """
        Realize a single waypoint with configurable retrieval.
        
        Args:
            waypoint: Target basin coordinates
            pos_constraint: Optional POS tag filter
            top_k: Number of candidates for proxy filter
            return_candidates: If True, return multiple candidates
            
        Returns:
            (word, basin, distance) or list of candidates
        """
        # TODO: Implement POS filtering when available
        # For now, ignore pos_constraint
        
        return self.retriever.retrieve(
            target_basin=waypoint,
            top_k=top_k,
            final_k=5 if return_candidates else 1,
            return_candidates=return_candidates
        )


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def example_usage():
    """
    Example of using two-step retrieval in generation pipeline.
    """
    # Mock coordizer with vocabulary
    class MockCoordizer:
        def __init__(self):
            # Generate sample vocabulary
            self.generation_vocab = {}
            for i in range(1000):
                word = f"word_{i}"
                # Random simplex basin
                alpha = np.ones(BASIN_DIM)
                basin = np.random.dirichlet(alpha)
                self.generation_vocab[word] = basin
    
    # Initialize realizer
    coordizer = MockCoordizer()
    realizer = OptimizedGeometricRealizer(coordizer, kernel_name="Athena")
    
    # Generate waypoints (from PLAN phase)
    waypoints = []
    for i in range(10):
        # Random target basin
        alpha = np.ones(BASIN_DIM)
        waypoint = np.random.dirichlet(alpha)
        waypoints.append(waypoint)
    
    # Realize waypoints to words
    words, word_basins = realizer.realize_waypoints(waypoints)
    
    print(f"\nGenerated text: {' '.join(words)}")
    print(f"Number of words: {len(words)}")
    
    # Single waypoint with candidates
    target = waypoints[0]
    candidates = realizer.realize_single_waypoint(
        target,
        top_k=100,
        return_candidates=True
    )
    
    print(f"\nTop 5 candidates for target:")
    for word, basin, distance in candidates[:5]:
        print(f"  {word}: d_FR = {distance:.4f}")


# =============================================================================
# PERFORMANCE COMPARISON
# =============================================================================

def benchmark_comparison():
    """
    Compare naive vs two-step retrieval performance.
    """
    import time
    from qig_geometry.canonical import fisher_rao_distance
    
    # Create vocabulary
    vocab_size = 5000
    vocab = {}
    for i in range(vocab_size):
        word = f"word_{i}"
        alpha = np.ones(BASIN_DIM)
        basin = np.random.dirichlet(alpha)
        vocab[word] = basin
    
    # Target basin
    alpha = np.ones(BASIN_DIM)
    target = np.random.dirichlet(alpha)
    
    # Naive approach: compute Fisher-Rao to all words
    print(f"\nBenchmarking on {vocab_size} vocabulary...")
    
    start = time.time()
    distances = []
    for word, basin in vocab.items():
        d = fisher_rao_distance(target, basin)
        distances.append((word, basin, d))
    distances.sort(key=lambda x: x[2])
    best_naive = distances[0]
    time_naive = time.time() - start
    
    print(f"Naive approach: {time_naive:.4f}s")
    print(f"  Best match: {best_naive[0]}, d_FR = {best_naive[2]:.4f}")
    
    # Two-step approach
    retriever = TwoStepRetriever(vocab, storage_format='simplex', build_index=True)
    
    start = time.time()
    best_two_step = retriever.retrieve(target, top_k=100, final_k=1)
    time_two_step = time.time() - start
    
    print(f"Two-step approach: {time_two_step:.4f}s")
    print(f"  Best match: {best_two_step[0]}, d_FR = {best_two_step[2]:.4f}")
    
    # Speedup
    speedup = time_naive / time_two_step
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Accuracy check
    if best_naive[0] == best_two_step[0]:
        print("✓ Found exact same best match")
    else:
        distance_diff = abs(best_naive[2] - best_two_step[2])
        print(f"⚠ Different best match (distance diff: {distance_diff:.6f})")


if __name__ == "__main__":
    print("=" * 70)
    print("Two-Step Retrieval Integration Example")
    print("=" * 70)
    
    # Run usage example
    example_usage()
    
    # Run benchmark
    benchmark_comparison()
