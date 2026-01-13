
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

from learned_relationships import LearnedRelationships
from geometric_word_relationships import GeometricWordRelationships

def test_purity_verification():
    print("Verifying Geometric Purity Fixes...")
    
    class _DummyCoordizer:
        def __init__(self):
            self.vocab = {"quantum": 1, "geometry": 1, "symmetry": 1}
            self.basin_coords = {
                "quantum": np.ones(64, dtype=np.float64) / 64.0,
                "geometry": np.linspace(1.0, 2.0, 64, dtype=np.float64),
                "symmetry": np.linspace(2.0, 1.0, 64, dtype=np.float64),
            }

    learner = GeometricWordRelationships(coordizer=_DummyCoordizer())
    
    # 1. Test update_from_learner (Neighbor population)
    fresh_lr = LearnedRelationships.__new__(LearnedRelationships)
    fresh_lr.word_neighbors = {}
    fresh_lr.adjusted_basins = {}
    fresh_lr.word_frequency = {}
    fresh_lr.learning_complete = False
    fresh_lr._relationship_phi = {}
    
    print("Updating Fresh LR from Learner...")
    fresh_lr.update_from_learner(learner, {})
    
    print(f"Word Neighbors: {fresh_lr.word_neighbors}")
    assert "quantum" in fresh_lr.word_neighbors, "'quantum' should have neighbors"
    assert "geometry" in fresh_lr.word_neighbors, "'geometry' should have neighbors"

    print("Geometric Purity Verification PASSED ✅")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_purity_verification()
