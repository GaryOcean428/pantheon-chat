
import sys
from pathlib import Path
import os
import logging
import numpy as np

# Setup paths
backend_path = Path(__file__).parent.parent / "pantheon-replit" / "qig-backend"
sys.path.insert(0, str(backend_path))

# Mock environment
os.environ['DATABASE_URL'] = 'postgresql://dummy:dummy@localhost:5432/dummy'

from word_relationship_learner import WordRelationshipLearner
from learned_relationships import LearnedRelationships

def test_purity_verification():
    print("Verifying Geometric Purity Fixes...")
    
    # 1. Test tokenize_text (Pure filtering)
    vocab = {"quantum", "geometry"}
    learner = WordRelationshipLearner(vocab, window_size=5)
    
    text = "A quantum-geometry test for E8 symmetry."
    tokens = learner.tokenize_text(text)
    print(f"Tokens from '{text}': {tokens}")
    # Expected: ['quantum', 'geometry', 'test', 'symmetry'] or similar, NOT ['a']
    assert "a" not in tokens, "Stopword 'a' should be filtered"
    
    # 2. Test update_from_learner (Neighbor population)
    learner.learn_from_text("quantum geometry quantum geometry")
    
    fresh_lr = LearnedRelationships.__new__(LearnedRelationships)
    fresh_lr.word_neighbors = {}
    fresh_lr.adjusted_basins = {}
    fresh_lr.word_frequency = {}
    
    print("Updating Fresh LR from Learner...")
    fresh_lr.update_from_learner(learner, {})
    
    print(f"Word Neighbors: {fresh_lr.word_neighbors}")
    assert "quantum" in fresh_lr.word_neighbors, "'quantum' should have neighbors"
    assert "geometry" in fresh_lr.word_neighbors, "'geometry' should have neighbors"
    
    # 3. Test adjust_basin_coordinates (Spherical stability)
    basins = {
        "quantum": np.random.randn(64),
        "geometry": np.random.randn(64)
    }
    # Normalize
    for k in basins: basins[k] /= np.linalg.norm(basins[k])
    
    print("Adjusting Basin Coordinates...")
    adjusted = learner.adjust_basin_coordinates(basins, iterations=1)
    
    for word, basin in adjusted.items():
        norm = np.linalg.norm(basin)
        print(f"Basin '{word}' norm: {norm}")
        assert abs(norm - 1.0) < 1e-5, f"Basin '{word}' should be unit norm (Fisher manifold)"

    print("Geometric Purity Verification PASSED ✅")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_purity_verification()
