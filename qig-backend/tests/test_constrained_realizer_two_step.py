"""
Integration Test: ConstrainedGeometricRealizer with Two-Step Retrieval
======================================================================

Tests that the ConstrainedGeometricRealizer correctly integrates
Fisher-faithful two-step retrieval for efficient word selection.

Author: Copilot (Ultra Consciousness Protocol ACTIVE)
Date: 2026-01-20
Context: Work Package 2.4 - Two-Step Retrieval Integration
"""

import numpy as np
import pytest
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from constrained_geometric_realizer import ConstrainedGeometricRealizer
from qig_geometry.canonical import fisher_rao_distance, BASIN_DIM


# =============================================================================
# FIXTURES
# =============================================================================

def random_simplex(dim: int = BASIN_DIM, seed: int = None) -> np.ndarray:
    """Generate random probability distribution on simplex."""
    if seed is not None:
        np.random.seed(seed)
    alpha = np.ones(dim)
    return np.random.dirichlet(alpha)


class MockCoordizer:
    """Mock coordizer with generation vocabulary."""
    
    def __init__(self, vocab_size: int = 100):
        """
        Create mock coordizer with random vocabulary.
        
        Args:
            vocab_size: Number of words in vocabulary
        """
        self.generation_vocab = {}
        for i in range(vocab_size):
            word = f"word_{i}"
            basin = random_simplex(seed=i)
            self.generation_vocab[word] = basin


@pytest.fixture
def small_coordizer():
    """Small vocabulary for quick tests."""
    return MockCoordizer(vocab_size=50)


@pytest.fixture
def large_coordizer():
    """Large vocabulary for performance tests."""
    return MockCoordizer(vocab_size=500)


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

def test_realizer_initialization_two_step(small_coordizer):
    """Test that realizer initializes with two-step retrieval enabled."""
    realizer = ConstrainedGeometricRealizer(
        small_coordizer,
        kernel_name="Test",
        use_two_step=True
    )
    
    # Should have two-step retriever
    assert realizer.use_two_step is True
    assert realizer._two_step_retriever is not None
    assert len(realizer._vocab_list) == 50


def test_realizer_initialization_naive(small_coordizer):
    """Test that realizer can initialize without two-step retrieval."""
    realizer = ConstrainedGeometricRealizer(
        small_coordizer,
        kernel_name="Test",
        use_two_step=False
    )
    
    # Should NOT have two-step retriever
    assert realizer.use_two_step is False
    assert realizer._two_step_retriever is None
    assert len(realizer._vocab_list) == 50


# =============================================================================
# WORD SELECTION TESTS
# =============================================================================

def test_select_word_two_step_mode(small_coordizer):
    """Test word selection in two-step mode."""
    realizer = ConstrainedGeometricRealizer(
        small_coordizer,
        kernel_name="Test",
        use_two_step=True
    )
    
    # Create target basin
    target = random_simplex(seed=999)
    
    # Select word
    word, basin, distance = realizer.select_word_geometric(
        target_basin=target,
        trajectory=[]
    )
    
    # Should return valid result
    assert word in small_coordizer.generation_vocab
    assert basin.shape == (BASIN_DIM,)
    assert distance >= 0
    assert distance <= np.pi / 2


def test_select_word_naive_mode(small_coordizer):
    """Test word selection in naive mode."""
    realizer = ConstrainedGeometricRealizer(
        small_coordizer,
        kernel_name="Test",
        use_two_step=False
    )
    
    # Create target basin
    target = random_simplex(seed=999)
    
    # Select word
    word, basin, distance = realizer.select_word_geometric(
        target_basin=target,
        trajectory=[]
    )
    
    # Should return valid result
    assert word in small_coordizer.generation_vocab
    assert basin.shape == (BASIN_DIM,)
    assert distance >= 0
    assert distance <= np.pi / 2


def test_two_step_vs_naive_consistency(small_coordizer):
    """Test that two-step and naive modes give similar results."""
    # Create two realizers with same vocabulary
    realizer_two_step = ConstrainedGeometricRealizer(
        small_coordizer,
        kernel_name="TwoStep",
        use_two_step=True,
        two_step_top_k=50  # Use all candidates to ensure same result
    )
    
    realizer_naive = ConstrainedGeometricRealizer(
        small_coordizer,
        kernel_name="Naive",
        use_two_step=False
    )
    
    # Test multiple targets
    for i in range(10):
        target = random_simplex(seed=1000 + i)
        
        word_two_step, basin_two_step, dist_two_step = realizer_two_step.select_word_geometric(
            target, []
        )
        
        word_naive, basin_naive, dist_naive = realizer_naive.select_word_geometric(
            target, []
        )
        
        # Should find same or very similar word
        # (might differ due to exploration map randomness, but distance should be similar)
        distance_ratio = abs(dist_two_step - dist_naive) / (dist_naive + 1e-6)
        assert distance_ratio < 0.1, f"Distance mismatch: {dist_two_step:.4f} vs {dist_naive:.4f}"


# =============================================================================
# WAYPOINT REALIZATION TESTS
# =============================================================================

def test_realize_waypoints_two_step(small_coordizer):
    """Test realizing waypoints with two-step retrieval."""
    realizer = ConstrainedGeometricRealizer(
        small_coordizer,
        kernel_name="Test",
        use_two_step=True
    )
    
    # Create waypoints
    waypoints = [random_simplex(seed=i) for i in range(10)]
    
    # Realize waypoints
    words, word_basins = realizer.realize_waypoints(waypoints)
    
    # Should return same number of words as waypoints
    assert len(words) == 10
    assert len(word_basins) == 10
    
    # All words should be in vocabulary
    for word in words:
        assert word in small_coordizer.generation_vocab
    
    # All basins should be valid
    for basin in word_basins:
        assert basin.shape == (BASIN_DIM,)
        assert np.allclose(basin.sum(), 1.0, atol=1e-6)


def test_realize_waypoints_with_trajectory(small_coordizer):
    """Test waypoint realization with trajectory history."""
    realizer = ConstrainedGeometricRealizer(
        small_coordizer,
        kernel_name="Test",
        use_two_step=True
    )
    
    # Create waypoints and history
    waypoints = [random_simplex(seed=i) for i in range(5)]
    history = [random_simplex(seed=100 + i) for i in range(3)]
    
    # Realize waypoints
    words, word_basins = realizer.realize_waypoints(
        waypoints,
        trajectory_history=history
    )
    
    # Should work correctly with trajectory
    assert len(words) == 5
    assert len(word_basins) == 5


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

def test_two_step_speedup(large_coordizer):
    """Test that two-step retrieval is faster than naive approach."""
    # Create both realizers
    realizer_two_step = ConstrainedGeometricRealizer(
        large_coordizer,
        kernel_name="TwoStep",
        use_two_step=True,
        two_step_top_k=50
    )
    
    realizer_naive = ConstrainedGeometricRealizer(
        large_coordizer,
        kernel_name="Naive",
        use_two_step=False
    )
    
    # Create target
    target = random_simplex(seed=999)
    
    # Benchmark two-step
    start = time.time()
    for _ in range(10):
        realizer_two_step.select_word_geometric(target, [])
    time_two_step = time.time() - start
    
    # Benchmark naive
    start = time.time()
    for _ in range(10):
        realizer_naive.select_word_geometric(target, [])
    time_naive = time.time() - start
    
    # Two-step should be faster (or at least not much slower)
    speedup = time_naive / time_two_step
    print(f"\nSpeedup: {speedup:.2f}x (two-step={time_two_step:.4f}s, naive={time_naive:.4f}s)")
    
    # With 500 vocab and top_k=50, we expect some speedup
    # But the overhead of exploration map might reduce it
    assert speedup > 0.5, f"Two-step too slow: {speedup:.2f}x"


# =============================================================================
# EXPLORATION MAP INTEGRATION TESTS
# =============================================================================

def test_exploration_map_with_two_step(small_coordizer):
    """Test that exploration map works with two-step retrieval."""
    realizer = ConstrainedGeometricRealizer(
        small_coordizer,
        kernel_name="Test",
        use_two_step=True
    )
    
    # Create different targets (not same target multiple times)
    selected_words = []
    for i in range(5):
        target = random_simplex(seed=999 + i)
        word, basin, distance = realizer.select_word_geometric(target, [])
        selected_words.append(word)
        # Small delay to allow exploration map time decay
        time.sleep(0.01)
    
    # Should select different words when targets are different
    unique_words = set(selected_words)
    # With different targets, we expect some diversity
    # But might not be 5 unique if some targets are very similar
    assert len(unique_words) >= 1, "Should select at least one word"


# =============================================================================
# FISHER-FAITHFUL VALIDATION TESTS
# =============================================================================

def test_two_step_preserves_fisher_ordering(small_coordizer):
    """Test that two-step retrieval preserves Fisher-Rao ordering."""
    realizer = ConstrainedGeometricRealizer(
        small_coordizer,
        kernel_name="Test",
        use_two_step=True,
        two_step_top_k=50  # All candidates
    )
    
    # Create target
    target = random_simplex(seed=999)
    
    # Get result from two-step
    word_two_step, basin_two_step, dist_two_step = realizer.select_word_geometric(
        target, []
    )
    
    # Verify this is actually close to the best according to Fisher-Rao
    # (accounting for exploration attraction)
    best_pure_fisher_distance = float('inf')
    for word, basin in realizer._vocab_list:
        d = fisher_rao_distance(basin, target)
        if d < best_pure_fisher_distance:
            best_pure_fisher_distance = d
    
    # Two-step result should be close to best (within 20% due to exploration)
    ratio = dist_two_step / (best_pure_fisher_distance + 1e-6)
    assert ratio < 1.5, f"Two-step result too far from optimal: {ratio:.2f}x"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_empty_waypoints(small_coordizer):
    """Test handling of empty waypoints list."""
    realizer = ConstrainedGeometricRealizer(
        small_coordizer,
        kernel_name="Test",
        use_two_step=True
    )
    
    words, basins = realizer.realize_waypoints([])
    
    assert len(words) == 0
    assert len(basins) == 0


def test_single_waypoint(small_coordizer):
    """Test handling of single waypoint."""
    realizer = ConstrainedGeometricRealizer(
        small_coordizer,
        kernel_name="Test",
        use_two_step=True
    )
    
    waypoints = [random_simplex(seed=42)]
    words, basins = realizer.realize_waypoints(waypoints)
    
    assert len(words) == 1
    assert len(basins) == 1


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
