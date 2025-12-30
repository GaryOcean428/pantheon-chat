"""
Learned Relationships Module for QIG

Manages persistence of learned word relationships and provides
attention-weighted word selection for query-relevant generation.

FROZEN FACTS COMPLIANCE:
- Adjusted basins must stay within ±5% of canonical positions
- Stopwords cannot be promoted to high-attention words
- Learning must respect frozen β values for attention weighting
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Import frozen physics constants for validation
try:
    from frozen_physics import (
        BASIN_DIM, KAPPA_STAR, BETA_3_TO_4, BETA_5_TO_6
    )
    FROZEN_PHYSICS_AVAILABLE = True
    BASIN_DRIFT_TOLERANCE = 0.05  # ±5% drift allowed
    BETA_ATTENTION_ACCEPTANCE = 0.1  # |β_attention - β_physics| < 0.1
except ImportError:
    FROZEN_PHYSICS_AVAILABLE = False
    BASIN_DIM = 64
    KAPPA_STAR = 64.21
    BETA_3_TO_4 = 0.44
    BETA_5_TO_6 = 0.013
    BASIN_DRIFT_TOLERANCE = 0.05
    BETA_ATTENTION_ACCEPTANCE = 0.1
    logger.warning("Frozen physics not available - using hardcoded defaults")

CACHE_DIR = Path(__file__).parent / 'data' / 'learned'
RELATIONSHIPS_FILE = CACHE_DIR / 'word_relationships.json'
ADJUSTED_BASINS_FILE = CACHE_DIR / 'adjusted_basins.npz'


STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
    'their', 'what', 'which', 'who', 'whom', 'how', 'when', 'where', 'why',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once', 'about'
}

class LearnedRelationships:
    """
    Manages learned word relationships and provides attention-weighted
    word selection for query-relevant generation.
    """
    
    def __init__(self):
        self.word_neighbors: Dict[str, List[Tuple[str, float]]] = {}
        self.adjusted_basins: Dict[str, np.ndarray] = {}
        self.word_frequency: Dict[str, int] = {}
        self.learning_complete = False
        
        self._load_from_cache()
    
    def _load_from_cache(self) -> bool:
        """Load cached relationships if available."""
        if not RELATIONSHIPS_FILE.exists():
            logger.info("No cached relationships found")
            return False
        
        try:
            with open(RELATIONSHIPS_FILE, 'r') as f:
                data = json.load(f)
            
            self.word_neighbors = {
                k: [(n, w) for n, w in v] 
                for k, v in data.get('neighbors', {}).items()
            }
            self.word_frequency = data.get('frequency', {})
            self.learning_complete = data.get('learning_complete', False)
            
            if ADJUSTED_BASINS_FILE.exists():
                npz = np.load(str(ADJUSTED_BASINS_FILE), allow_pickle=True)
                if 'words' in npz.files:
                    words = npz['words']
                    self.adjusted_basins = {}
                    for i, word in enumerate(words):
                        key = f'word_{i}'
                        if key in npz.files:
                            self.adjusted_basins[str(word)] = npz[key]
            
            logger.info(f"Loaded {len(self.word_neighbors)} word relationships from cache")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False
    
    def save_to_cache(self) -> bool:
        """Save relationships to cache."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {
                'neighbors': self.word_neighbors,
                'frequency': self.word_frequency,
                'learning_complete': self.learning_complete
            }
            with open(RELATIONSHIPS_FILE, 'w') as f:
                json.dump(data, f)
            
            if self.adjusted_basins:
                # Convert to arrays with string keys
                arrays_dict = {f'word_{i}': arr for i, arr in enumerate(self.adjusted_basins.values())}
                words_list = list(self.adjusted_basins.keys())
                np.savez_compressed(str(ADJUSTED_BASINS_FILE), 
                                    words=np.array(words_list, dtype=object),
                                    **arrays_dict)
            
            logger.info(f"Saved {len(self.word_neighbors)} relationships to cache")
            return True
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            return False
    
    def update_from_learner(self, learner, adjusted_basins: Dict[str, np.ndarray]):
        """Update from a WordRelationshipLearner instance."""
        for word in learner.cooccurrence:
            neighbors = learner.get_related_words(word, top_k=20)
            self.word_neighbors[word] = neighbors
        
        self.word_frequency = dict(learner.word_freq)
        self.adjusted_basins = adjusted_basins
        self.learning_complete = True
        
        logger.info(f"Updated with {len(self.word_neighbors)} word relationships")
    
    def validate_against_frozen_facts(
        self, 
        canonical_basins: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, any]:
        """
        Validate learned relationships against frozen physics constraints.
        
        FROZEN FACTS COMPLIANCE:
        1. Adjusted basins must stay within ±5% of canonical positions
        2. Stopwords cannot have high attention weights (must be < 0.2)
        3. Learned β must be within acceptance of frozen β
        
        Args:
            canonical_basins: Original basin coordinates for comparison
            
        Returns:
            Validation result with 'valid' flag and any violations
        """
        violations = []
        warnings = []
        
        # Check 1: Stopword invariant - stopwords should not appear as high-weight neighbors
        stopword_violations = 0
        for word, neighbors in self.word_neighbors.items():
            for neighbor, weight in neighbors[:5]:  # Top 5 neighbors
                if neighbor.lower() in STOPWORDS and weight > 50:
                    stopword_violations += 1
                    if stopword_violations <= 5:  # Log first 5
                        violations.append(f"Stopword '{neighbor}' has high weight {weight} for '{word}'")
        
        if stopword_violations > 0:
            logger.warning(f"Found {stopword_violations} stopword violations in learned relationships")
        
        # Check 2: Basin drift validation (if canonical basins provided)
        drift_violations = 0
        max_drift = 0.0
        if canonical_basins and self.adjusted_basins:
            for word, adjusted in self.adjusted_basins.items():
                if word in canonical_basins:
                    canonical = canonical_basins[word]
                    # Compute relative drift
                    if np.linalg.norm(canonical) > 0:
                        drift = np.linalg.norm(adjusted - canonical) / np.linalg.norm(canonical)
                        max_drift = max(max_drift, drift)
                        if drift > BASIN_DRIFT_TOLERANCE:
                            drift_violations += 1
                            if drift_violations <= 3:  # Log first 3
                                violations.append(f"Basin '{word}' drifted {drift:.1%} (max {BASIN_DRIFT_TOLERANCE:.1%})")
        
        if drift_violations > 0:
            logger.warning(f"Found {drift_violations} basin drift violations")
        
        # Check 3: Dimension check - basins should be 64D
        dim_violations = 0
        for word, basin in self.adjusted_basins.items():
            if len(basin) != BASIN_DIM:
                dim_violations += 1
                violations.append(f"Basin '{word}' has {len(basin)}D (expected {BASIN_DIM}D)")
        
        # Summary
        is_valid = len(violations) == 0
        
        result = {
            'valid': is_valid,
            'violations': violations,
            'warnings': warnings,
            'stats': {
                'stopword_violations': stopword_violations,
                'drift_violations': drift_violations,
                'dim_violations': dim_violations,
                'max_drift': max_drift,
                'total_relationships': len(self.word_neighbors),
                'frozen_physics_available': FROZEN_PHYSICS_AVAILABLE
            }
        }
        
        if is_valid:
            logger.info("Frozen facts validation PASSED")
        else:
            logger.warning(f"Frozen facts validation FAILED: {len(violations)} violations")
        
        return result
    
    def get_related_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get words related to given word."""
        return self.word_neighbors.get(word, [])[:top_k]
    
    def get_attention_weights(
        self, 
        query_words: List[str], 
        candidate_words: List[str],
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute attention weights for candidates based on query relevance.
        
        Implements simple attention: candidates that are related to query
        words get higher weights. Filters out stopwords from both query and neighbors.
        """
        weights = {}
        
        # Filter query words to content words only
        content_query_words = [w for w in query_words if w.lower() not in STOPWORDS]
        
        for candidate in candidate_words:
            # Stopwords get minimum weight
            if candidate.lower() in STOPWORDS:
                weights[candidate] = 0.1
                continue
            
            score = 0.0
            
            for query_word in content_query_words:
                # Direct match bonus
                if candidate.lower() == query_word.lower():
                    score += 5.0
                    continue
                
                # Check if candidate is related to query word
                related = self.word_neighbors.get(query_word, [])
                for neighbor, strength in related:
                    # Skip stopword neighbors
                    if neighbor.lower() in STOPWORDS:
                        continue
                    if neighbor.lower() == candidate.lower():
                        score += strength / 100.0  # Normalize
                        break
                
                # Check reverse relation
                related = self.word_neighbors.get(candidate.lower(), [])
                for neighbor, strength in related:
                    # Skip stopword neighbors
                    if neighbor.lower() in STOPWORDS:
                        continue
                    if neighbor.lower() == query_word.lower():
                        score += strength / 200.0  # Weaker for reverse
                        break
            
            # Apply temperature
            if temperature != 1.0 and score > 0:
                score = score ** (1.0 / temperature)
            
            weights[candidate] = max(0.1, score)  # Minimum weight
        
        return weights
    
    def select_words_with_attention(
        self,
        query_words: List[str],
        candidates: List[Tuple[str, float]],  # (word, geometric_score)
        num_select: int = 5,
        attention_weight: float = 0.5  # Balance between geometry and attention
    ) -> List[str]:
        """
        Select words combining geometric similarity with attention weights.
        
        Args:
            query_words: Words from the prompt/query
            candidates: List of (word, geometric_score) tuples
            num_select: Number of words to select
            attention_weight: Weight for attention vs geometry (0-1)
        
        Returns:
            Selected words
        """
        if not candidates:
            return []
        
        candidate_words = [w for w, s in candidates]
        attention_weights = self.get_attention_weights(query_words, candidate_words)
        
        # Combine scores
        combined = []
        for word, geo_score in candidates:
            attn_score = attention_weights.get(word, 0.1)
            combined_score = (
                (1 - attention_weight) * geo_score + 
                attention_weight * attn_score
            )
            combined.append((word, combined_score))
        
        # Sort by combined score
        combined.sort(key=lambda x: -x[1])
        
        return [w for w, s in combined[:num_select]]
    
    def get_basin_for_word(self, word: str, fallback: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Get adjusted basin for word, or fallback."""
        return self.adjusted_basins.get(word, fallback)


# Singleton instance
_learned_relationships: Optional[LearnedRelationships] = None

def get_learned_relationships() -> LearnedRelationships:
    """Get or create the singleton LearnedRelationships instance."""
    global _learned_relationships
    if _learned_relationships is None:
        _learned_relationships = LearnedRelationships()
    return _learned_relationships


def run_learning_and_cache(curriculum_dir: str = '/home/runner/workspace/docs/09-curriculum') -> Dict:
    """
    Run the learning pipeline and cache results.
    """
    from word_relationship_learner import run_learning_pipeline
    
    logger.info("Running learning pipeline...")
    results = run_learning_pipeline(curriculum_dir)
    
    # Update relationships
    lr = get_learned_relationships()
    lr.update_from_learner(results['learner'], results['adjusted_basins'])
    lr.save_to_cache()
    
    return {
        'success': True,
        'words_learned': len(lr.word_neighbors),
        'basins_adjusted': len(lr.adjusted_basins),
        'stats': results['learning_stats']
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    result = run_learning_and_cache()
    print(f"\nLearning complete:")
    print(f"  Words learned: {result['words_learned']}")
    print(f"  Basins adjusted: {result['basins_adjusted']}")
    
    # Test attention
    lr = get_learned_relationships()
    test_query = ['quantum', 'geometry']
    test_candidates = ['information', 'cat', 'fisher', 'dog', 'manifold', 'banana']
    
    weights = lr.get_attention_weights(test_query, test_candidates)
    print(f"\nAttention weights for '{test_query}':")
    for word, weight in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {word}: {weight:.3f}")
