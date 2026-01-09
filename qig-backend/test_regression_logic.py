
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

def test_regression_trigger():
    print("Testing regression trigger logic...")
    
    # 1. Setup mock learner with 0 pairs
    vocab = {"quantum", "geometry", "consciousness"}
    learner = WordRelationshipLearner(vocab, window_size=5)
    
    # 2. Setup mock fresh_lr
    fresh_lr = LearnedRelationships.__new__(LearnedRelationships)
    fresh_lr.word_neighbors = {}
    fresh_lr.adjusted_basins = {}
    fresh_lr.word_frequency = {}
    fresh_lr.learning_complete = False
    
    # 3. Call update_from_learner
    print("Calling fresh_lr.update_from_learner...")
    fresh_lr.update_from_learner(learner, {})
    
    # 4. Check word_neighbors
    neighbor_count = len(fresh_lr.word_neighbors)
    print(f"Learned relationships count: {neighbor_count}")
    
    baseline_count = 100 # Dummy baseline
    regression_threshold = baseline_count * 0.95
    
    if neighbor_count < regression_threshold:
        print(f"REGRESSION TRIGGERED: {neighbor_count} < {regression_threshold}")
    else:
        print("No regression triggered.")
        
    return neighbor_count

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_regression_trigger()
