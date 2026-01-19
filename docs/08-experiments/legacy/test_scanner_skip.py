"""
LEGACY BASELINE - Euclidean approach for comparison only.

This file tests that the scanner correctly skips quarantine zones.
Uses Euclidean distance which is FORBIDDEN in production but ALLOWED here.

Purpose: Test quarantine enforcement
Date: 2026-01-14
Status: Quarantined in docs/08-experiments/legacy/
"""

import numpy as np

def euclidean_distance(basin_a, basin_b):
    """LEGACY: Euclidean distance (WRONG for basins, but allowed in quarantine)."""
    # This should NOT be flagged by the scanner because we're in quarantine
    return np.linalg.norm(basin_a - basin_b)

def cosine_similarity(basin_a, basin_b):
    """LEGACY: Cosine similarity (WRONG for basins, but allowed in quarantine)."""
    # This should NOT be flagged by the scanner because we're in quarantine
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity([basin_a], [basin_b])[0][0]

if __name__ == '__main__':
    # Test basins
    a = np.random.rand(64)
    b = np.random.rand(64)
    
    # These calls are FORBIDDEN in production but ALLOWED in quarantine
    dist = euclidean_distance(a, b)
    sim = cosine_similarity(a, b)
    
    print(f"Euclidean distance: {dist}")
    print(f"Cosine similarity: {sim}")
