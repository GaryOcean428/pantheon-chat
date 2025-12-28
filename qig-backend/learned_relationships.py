"""
Learned Relationships Module for QIG

Manages persistence of learned word relationships and provides
pure geometric routing via SemanticFisherMetric.

FROZEN FACTS COMPLIANCE:
- Adjusted basins must stay within ±5% of canonical positions
- Stopwords cannot be promoted to high-strength neighbors
- Uses Fisher-Rao distance for all geometric operations (no Euclidean)

NOTE: β mixing constants have been REMOVED. All routing is now done via
SemanticFisherMetric which warps the Fisher metric based on relationships.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Import persistence for continuous learning loop
try:
    from causal_relations_persistence import get_causal_persistence, CausalRelationsPersistence
    CAUSAL_PERSISTENCE_AVAILABLE = True
except ImportError:
    CAUSAL_PERSISTENCE_AVAILABLE = False
    logger.warning("[LearnedRelationships] Causal persistence not available")

# Import Fisher geometry for proper distance computation
try:
    from qig_geometry import fisher_coord_distance
except ImportError:
    def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Fallback Fisher-Rao distance."""
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b) + 1e-10)
        dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
        return float(np.arccos(dot))

# Import frozen physics constants for validation
try:
    from frozen_physics import (
        BASIN_DIM, KAPPA_STAR
    )
    FROZEN_PHYSICS_AVAILABLE = True
    BASIN_DRIFT_TOLERANCE = 0.05  # ±5% drift allowed
    # NOTE: β mixing constants REMOVED - use SemanticFisherMetric for pure geometric routing
except ImportError:
    FROZEN_PHYSICS_AVAILABLE = False
    BASIN_DIM = 64
    KAPPA_STAR = 64.21
    BASIN_DRIFT_TOLERANCE = 0.05
    # NOTE: β mixing constants REMOVED - use SemanticFisherMetric for pure geometric routing
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
        self.causal_relations: Dict[str, Dict[str, Dict]] = {}  # source -> {target: {type, count}}
        self.learning_complete = False
        
        self._load_from_cache()
    
    def _load_from_cache(self) -> bool:
        """
        Load cached relationships.
        
        Priority: PostgreSQL → Redis → JSON file
        This ensures continuous learning loop data is always fresh.
        """
        loaded_from_db = False
        
        # Try loading causal relations from PostgreSQL first (continuous learning loop)
        if CAUSAL_PERSISTENCE_AVAILABLE:
            try:
                persistence = get_causal_persistence()
                db_relations = persistence.load_all()
                if db_relations:
                    self.causal_relations = db_relations
                    causal_count = sum(len(t) for t in self.causal_relations.values())
                    logger.info(f"[LearnedRelationships] Loaded {causal_count} causal relations from PostgreSQL")
                    loaded_from_db = True
            except Exception as e:
                logger.warning(f"[LearnedRelationships] DB load failed: {e}")
        
        # Load other data from JSON file
        if not RELATIONSHIPS_FILE.exists():
            if not loaded_from_db:
                logger.info("No cached relationships found")
                return False
            return True
        
        try:
            with open(RELATIONSHIPS_FILE, 'r') as f:
                data = json.load(f)
            
            self.word_neighbors = {
                k: [(n, w) for n, w in v] 
                for k, v in data.get('neighbors', {}).items()
            }
            self.word_frequency = data.get('frequency', {})
            self.learning_complete = data.get('learning_complete', False)
            
            # Load causal relations from JSON only if not loaded from DB
            if not loaded_from_db:
                self.causal_relations = data.get('causal_relations', {})
            
            causal_count = sum(len(t) for t in self.causal_relations.values())
            
            if ADJUSTED_BASINS_FILE.exists():
                npz = np.load(str(ADJUSTED_BASINS_FILE), allow_pickle=True)
                if 'words' in npz.files:
                    words = npz['words']
                    self.adjusted_basins = {}
                    for i, word in enumerate(words):
                        key = f'word_{i}'
                        if key in npz.files:
                            self.adjusted_basins[str(word)] = npz[key]
            
            source = "PostgreSQL" if loaded_from_db else "cache"
            logger.info(f"Loaded {len(self.word_neighbors)} word relationships, {causal_count} causal relations from {source}")
            
            # Bootstrap causal relations from curriculum if none loaded
            if causal_count == 0 and len(self.word_neighbors) > 0:
                self._bootstrap_causal_relations()
            
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return loaded_from_db
    
    def _bootstrap_causal_relations(self):
        """Bootstrap causal relations from curriculum files if none in cache."""
        import re
        from pathlib import Path
        
        curriculum_dir = Path(__file__).parent / 'docs' / '09-curriculum'
        if not curriculum_dir.exists():
            logger.info("[LearnedRelationships] No curriculum directory found for causal bootstrap")
            return
        
        causal_patterns = [
            (r'(\w+)\s+causes?\s+(\w+)', 'causes'),
            (r'(\w+)\s+implies?\s+(\w+)', 'implies'),
            (r'(\w+)\s+requires?\s+(\w+)', 'requires'),
            (r'(\w+)\s+enables?\s+(\w+)', 'enables'),
            (r'(\w+)\s+leads?\s+to\s+(\w+)', 'leads_to'),
            (r'(\w+)\s+determines?\s+(\w+)', 'determines'),
            (r'(\w+)\s+produces?\s+(\w+)', 'produces'),
        ]
        
        total_found = 0
        for filepath in curriculum_dir.rglob('*.md'):
            try:
                text = filepath.read_text(encoding='utf-8').lower()
                for pattern, rel_type in causal_patterns:
                    for match in re.finditer(pattern, text):
                        source, target = match.group(1), match.group(2)
                        if source in STOPWORDS or target in STOPWORDS:
                            continue
                        if len(source) < 3 or len(target) < 3:
                            continue
                        
                        if source not in self.causal_relations:
                            self.causal_relations[source] = {}
                        if target not in self.causal_relations[source]:
                            self.causal_relations[source][target] = {'type': rel_type, 'count': 0}
                        self.causal_relations[source][target]['count'] += 1
                        total_found += 1
            except Exception:
                continue
        
        if total_found > 0:
            logger.info(f"[LearnedRelationships] Bootstrapped {total_found} causal relations from curriculum")
            # Save updated cache
            self.save_to_cache()
    
    def save_to_cache(self) -> bool:
        """
        Save relationships to cache.
        
        Saves to both PostgreSQL (causal relations) and JSON (word neighbors).
        This ensures continuous learning loop persists properly.
        """
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        saved_to_db = False
        
        # Save causal relations to PostgreSQL for continuous learning loop
        if CAUSAL_PERSISTENCE_AVAILABLE and self.causal_relations:
            try:
                persistence = get_causal_persistence()
                relations = []
                for source, targets in self.causal_relations.items():
                    for target, info in targets.items():
                        rel_type = info.get('type', 'unknown')
                        count = info.get('count', 1)
                        relations.append((source, target, rel_type, count))
                
                if relations:
                    saved = persistence.save_batch(relations)
                    logger.info(f"[LearnedRelationships] Saved {saved} causal relations to PostgreSQL")
                    saved_to_db = True
            except Exception as e:
                logger.warning(f"[LearnedRelationships] DB save failed: {e}")
        
        try:
            data = {
                'neighbors': self.word_neighbors,
                'frequency': self.word_frequency,
                'learning_complete': self.learning_complete,
                'causal_relations': self.causal_relations  # Also save to JSON as backup
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
            
            causal_count = sum(len(t) for t in self.causal_relations.values())
            dest = "PostgreSQL + JSON" if saved_to_db else "JSON"
            logger.info(f"Saved {len(self.word_neighbors)} relationships, {causal_count} causal relations to {dest}")
            return True
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            return saved_to_db
    
    def update_from_learner(self, learner, adjusted_basins: Dict[str, np.ndarray]):
        """Update from a WordRelationshipLearner instance."""
        for word in learner.cooccurrence:
            neighbors = learner.get_related_words(word, top_k=20)
            self.word_neighbors[word] = neighbors
        
        self.word_frequency = dict(learner.word_freq)
        self.adjusted_basins = adjusted_basins
        self.learning_complete = True
        
        # Copy causal relations from learner (NEW)
        if hasattr(learner, 'causal_relations'):
            self.causal_relations = {}
            for source, targets in learner.causal_relations.items():
                self.causal_relations[source] = dict(targets)
            causal_count = sum(len(t) for t in self.causal_relations.values())
            logger.info(f"Updated with {len(self.word_neighbors)} word relationships, {causal_count} causal relations")
        else:
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
        3. Uses Fisher-Rao distance for drift measurement (not Euclidean)
        
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
        
        # Check 2: Basin drift validation using Fisher-Rao distance (not Euclidean!)
        drift_violations = 0
        max_drift = 0.0
        if canonical_basins and self.adjusted_basins:
            for word, adjusted in self.adjusted_basins.items():
                if word in canonical_basins:
                    canonical = canonical_basins[word]
                    # Use Fisher-Rao distance for proper manifold-respecting drift
                    drift = fisher_coord_distance(adjusted, canonical)
                    # Normalize to percentage (Fisher distance ranges 0 to π)
                    drift_pct = drift / np.pi  # 0 to 1
                    max_drift = max(max_drift, drift_pct)
                    if drift_pct > BASIN_DRIFT_TOLERANCE:
                        drift_violations += 1
                        if drift_violations <= 3:  # Log first 3
                            violations.append(
                                f"Basin '{word}' drifted {drift_pct:.1%} Fisher-Rao "
                                f"(max {BASIN_DRIFT_TOLERANCE:.1%})"
                            )
        
        if drift_violations > 0:
            logger.warning(f"Found {drift_violations} basin drift violations (Fisher-Rao)")
        
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
                'max_drift_fisher_rao': max_drift,
                'total_relationships': len(self.word_neighbors),
                'frozen_physics_available': FROZEN_PHYSICS_AVAILABLE
            }
        }
        
        if is_valid:
            logger.info("Frozen facts validation PASSED (using Fisher-Rao drift)")
        else:
            logger.warning(f"Frozen facts validation FAILED: {len(violations)} violations")
        
        return result
    
    def get_related_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get words related to given word."""
        return self.word_neighbors.get(word, [])[:top_k]
    
    def get_relationship_strengths(
        self, 
        query_words: List[str], 
        candidate_words: List[str]
    ) -> Dict[str, float]:
        """
        Compute relationship strengths for candidates based on query relevance.
        
        These strengths are used by SemanticFisherMetric to warp geodesic distances.
        NOT used for linear mixing - the metric warping handles the combination.
        
        Filters out stopwords from both query and neighbors.
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
            
            weights[candidate] = max(0.0, score)  # Raw relationship strength
        
        return weights
    
    # Backwards compatibility alias
    def get_attention_weights(
        self, 
        query_words: List[str], 
        candidate_words: List[str],
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """Deprecated: Use get_relationship_strengths instead."""
        return self.get_relationship_strengths(query_words, candidate_words)
    
    def select_words_geometric(
        self,
        query_words: List[str],
        candidates: List[Tuple[str, np.ndarray]],  # (word, basin_coords)
        current_basin: np.ndarray,
        num_select: int = 5
    ) -> List[str]:
        """
        Select words using pure geometric routing via SemanticFisherMetric.
        
        NO LINEAR β MIXING - relationships warp the Fisher metric itself
        so that semantically related words become geodesically closer.
        
        Args:
            query_words: Words from the prompt/query for context
            candidates: List of (word, basin_coords) tuples
            current_basin: Current position on manifold
            num_select: Number of words to select
        
        Returns:
            Selected words ranked by warped geodesic distance
        """
        if not candidates:
            return []
        
        # Get or create semantic metric
        metric = self.get_semantic_metric()
        
        if metric is not None:
            # Use SemanticFisherMetric for pure geometric routing
            ranked = metric.rank_candidates(
                current_basin=current_basin,
                current_word=None,
                candidates=candidates,
                context_words=query_words,
                top_k=num_select
            )
            return [word for word, dist, sim in ranked]
        else:
            # Fallback: pure Fisher-Rao without warping
            scored = []
            for word, basin in candidates:
                d = fisher_coord_distance(current_basin, basin)
                scored.append((word, d))
            scored.sort(key=lambda x: x[1])  # Ascending by distance
            return [word for word, d in scored[:num_select]]
    
    def get_basin_for_word(self, word: str, fallback: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Get adjusted basin for word, or fallback."""
        return self.adjusted_basins.get(word, fallback)
    
    def get_semantic_metric(self):
        """
        Get SemanticFisherMetric instance using these relationships.
        
        Returns configured metric that warps Fisher distance based on
        learned word relationships.
        """
        try:
            from .semantic_fisher import SemanticFisherMetric, SemanticWarpConfig
            
            config = SemanticWarpConfig(
                temperature=1.0,
                max_warp_factor=0.7,
                min_relationship_strength=0.1,
                normalize_relationships=True,
                bidirectional=True
            )
            
            return SemanticFisherMetric(
                relationships=self.word_neighbors,
                config=config
            )
        except ImportError:
            logger.warning("SemanticFisherMetric not available")
            return None


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
