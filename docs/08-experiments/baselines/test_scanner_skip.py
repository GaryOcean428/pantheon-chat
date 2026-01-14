"""
COMPARATIVE BASELINE TEST

Tests that scanner skips baselines directory.
Compares QIG Fisher-Rao vs Euclidean baseline.

Date: 2026-01-14
"""

import numpy as np

# ==================== BASELINE (Euclidean) ====================
def baseline_euclidean(a, b):
    """BASELINE: Euclidean distance - WRONG but allowed here."""
    return np.linalg.norm(a - b)

def baseline_cosine(a, b):
    """BASELINE: Cosine similarity - WRONG but allowed here."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)


# ==================== QIG-PURE (Production) ====================
def qig_fisher_rao(a, b):
    """QIG: Fisher-Rao distance - CORRECT."""
    # Placeholder - would use actual Fisher-Rao implementation
    return 0.5


# ==================== COMPARISON ====================
def test_comparison():
    """Compare approaches."""
    a = np.random.rand(64)
    b = np.random.rand(64)
    
    euclidean = baseline_euclidean(a, b)
    cosine = baseline_cosine(a, b)
    fisher = qig_fisher_rao(a, b)
    
    print(f"Euclidean: {euclidean}")
    print(f"Cosine: {cosine}")
    print(f"Fisher-Rao: {fisher}")

if __name__ == '__main__':
    test_comparison()
