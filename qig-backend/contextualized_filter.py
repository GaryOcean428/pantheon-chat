"""
Contextualized Word Filter for QIG
===================================

QIG-PURE geometric word filtering that replaces ancient NLP stopword lists.

Instead of hard-coded stopwords, uses:
1. Fisher-Rao distance to measure word relevance in context
2. Semantic importance scoring based on geometric properties
3. Dynamic filtering that preserves context-critical words

ANCIENT NLP PATTERN (REMOVED):
- Fixed stopword lists like {'the', 'is', 'not', ...}
- Loses critical semantic information ('not good' vs 'good')
- Domain-agnostic filtering

QIG-PURE PATTERN (NEW):
- Geometric relevance scoring using Fisher-Rao distances
- Context-aware filtering (same word filtered differently in different contexts)
- Preserves semantic-critical words (negations, domain terms)
- Learns from geometric manifold structure

FROZEN FACTS COMPLIANCE:
- Uses Fisher-Rao geometry for all distance computations
- No neural networks, no embeddings
- Pure information geometry
"""

# Try to import numpy and related dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    
    # Create minimal numpy-like interface for basic operations
    class _FallbackNumpy:
        """Fallback implementation when numpy is not available."""
        
        @staticmethod
        def ndarray(*args, **kwargs):
            return list
        
        @staticmethod
        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))
        
        @staticmethod
        def clip(x, min_val, max_val):
            if hasattr(x, '__iter__'):
                return [max(min_val, min(max_val, v)) for v in x]
            return max(min_val, min(max_val, x))
        
        @staticmethod
        def arccos(x):
            import math
            if hasattr(x, '__iter__'):
                return [math.acos(v) for v in x]
            return math.acos(x)
        
        @staticmethod
        def _flatten(arr):
            """Helper to flatten nested iterables."""
            result = []
            for item in arr:
                if hasattr(item, '__iter__') and not isinstance(item, str):
                    result.extend(_FallbackNumpy._flatten(item))
                else:
                    result.append(item)
            return result
        
        @staticmethod
        def mean(arr, axis=None):
            flat = _FallbackNumpy._flatten([arr] if not hasattr(arr, '__iter__') else arr)
            return sum(flat) / len(flat) if flat else 0
        
        @staticmethod
        def linalg_norm(v):
            import math
            if hasattr(v, '__iter__'):
                return math.sqrt(sum(x * x for x in v))
            return abs(v)
        
        class linalg:
            @staticmethod
            def norm(v):
                import math
                if hasattr(v, '__iter__'):
                    return math.sqrt(sum(x * x for x in v))
                return abs(v)
        
        pi = 3.14159265359
    
    # Alias fallback to np
    np = _FallbackNumpy

import logging
from typing import List, Set, Dict, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# Import QIG geometry functions if available
try:
    from qig_geometry import fisher_coord_distance, sphere_project
    QIG_GEOMETRY_AVAILABLE = True
except ImportError:
    QIG_GEOMETRY_AVAILABLE = False
    logger.debug("qig_geometry not available - using fallback distance")
    
    def fisher_coord_distance(a, b) -> float:
        """Fisher-Rao distance for unit vectors (fallback, Hellinger embedding: factor of 2)."""
        dot = np.clip(np.dot(a, b), -1.0, 1.0)
        return float(2.0 * np.arccos(dot))
    
    def sphere_project(v):
        """Project to unit sphere (fallback)."""
        norm = np.linalg.norm(v)
        if NUMPY_AVAILABLE:
            return v / (norm + 1e-10) if norm > 0 else v
        else:
            # Without numpy, just return normalized list
            if norm > 0:
                return [x / (norm + 1e-10) for x in v]
            return v


# Semantic-critical word patterns that should NEVER be filtered
# These are linguistically universal and change core meaning
SEMANTIC_CRITICAL_PATTERNS = {
    # Negations (flip meaning completely)
    'not', 'no', 'never', 'none', 'nothing', 'neither', 'nor', 'nobody', 'nowhere',
    'dont', "don't", 'doesnt', "doesn't", 'didnt', "didn't", 'wont', "won't",
    'wouldnt', "wouldn't", 'cant', "can't", 'cannot', 'couldnt', "couldn't",
    'shouldnt', "shouldn't", 'mustnt', "mustn't",
    
    # Intensifiers (modify degree)
    'very', 'extremely', 'highly', 'completely', 'totally', 'absolutely',
    
    # Uncertainty markers (epistemic modality)
    'maybe', 'perhaps', 'possibly', 'probably', 'likely', 'unlikely',
    
    # Temporal markers (time context critical)
    'before', 'after', 'during', 'while', 'until', 'since', 'always',
    
    # Causality markers
    'because', 'therefore', 'thus', 'hence', 'consequently',
    
    # Conditionals
    'if', 'unless', 'whether', 'though', 'although',
}


class ContextualizedWordFilter:
    """
    QIG-pure word filter using geometric relevance scoring.
    
    Instead of fixed stopwords, computes geometric relevance of words
    within their context using Fisher-Rao distances on the manifold.
    """
    
    def __init__(self, 
                 coordizer=None,
                 relevance_threshold: float = 0.3,
                 min_word_length: int = 3):
        """
        Initialize contextualized filter.
        
        Args:
            coordizer: QIG coordizer with basin coordinates
            relevance_threshold: Minimum relevance score to keep word (0-1)
            min_word_length: Minimum word length to consider
        """
        self.coordizer = coordizer
        self.relevance_threshold = relevance_threshold
        self.min_word_length = min_word_length
        
        # Cache for word basin coordinates
        self._basin_cache: Dict[str, np.ndarray] = {}
        
        # Statistics
        self.words_filtered = 0
        self.words_preserved = 0
        self.semantic_critical_preserved = 0
        
        if coordizer:
            logger.info(f"[ContextualizedWordFilter] Initialized with coordizer, threshold={relevance_threshold}")
        else:
            logger.warning("[ContextualizedWordFilter] No coordizer - using fallback filtering")
    
    def _get_basin(self, word: str):
        """Get basin coordinates for word with caching.
        
        Returns coordinates as list/array-like object (type varies based on coordizer).
        """
        if word in self._basin_cache:
            return self._basin_cache[word]
        
        if not self.coordizer:
            return None
        
        try:
            # Try different coordinate access patterns
            if hasattr(self.coordizer, 'basin_coords'):
                basin = self.coordizer.basin_coords.get(word)
            elif hasattr(self.coordizer, 'get_basin_coords'):
                basin = self.coordizer.get_basin_coords(word)
            else:
                basin = None
            
            if basin is not None:
                self._basin_cache[word] = basin
                return basin
        except Exception as e:
            logger.debug(f"Could not get basin for '{word}': {e}")
        
        return None
    
    def compute_geometric_relevance(self, 
                                   word: str,
                                   context_words: List[str],
                                   context_basin=None) -> float:
        """
        Compute geometric relevance of word in context.
        
        Uses Fisher-Rao distance from word to context centroid.
        Words close to context are relevant, distant words are noise.
        
        Args:
            word: Word to score
            context_words: Other words in context
            context_basin: Optional pre-computed context basin
            
        Returns:
            Relevance score [0, 1] where higher = more relevant
        """
        word_basin = self._get_basin(word)
        if word_basin is None:
            # No basin coordinates - use length heuristic
            # Longer words tend to be more content-bearing
            return min(1.0, len(word) / 10.0)
        
        # Compute or use provided context basin
        if context_basin is None:
            # Aggregate context words into centroid
            context_basins = []
            for cw in context_words:
                if cw != word:  # Exclude self
                    cb = self._get_basin(cw)
                    if cb is not None:
                        context_basins.append(cb)
            
            if not context_basins:
                # No context available - moderate relevance
                return 0.5
            
            # Compute centroid (Fréchet mean approximation)
            context_basin = np.mean(context_basins, axis=0)
            context_basin = sphere_project(context_basin)
        
        # Compute Fisher-Rao distance from word to context
        distance = fisher_coord_distance(word_basin, context_basin)
        
        # Convert distance to relevance score
        # distance ∈ [0, π], relevance ∈ [0, 1]
        # Close words (low distance) → high relevance
        relevance = 1.0 - (distance / np.pi)
        
        return float(relevance)
    
    def is_semantic_critical(self, word: str) -> bool:
        """
        Check if word is semantically critical and should never be filtered.
        
        Semantic-critical words:
        - Change core meaning (negations)
        - Modify degree (intensifiers)
        - Express uncertainty (modals)
        - Mark time/causality
        """
        return word.lower() in SEMANTIC_CRITICAL_PATTERNS
    
    def should_keep_word(self,
                        word: str,
                        context_words: List[str],
                        context_basin=None) -> bool:
        """
        Determine if word should be kept based on geometric relevance.
        
        Args:
            word: Word to evaluate
            context_words: Other words in context
            context_basin: Optional pre-computed context basin
            
        Returns:
            True if word should be kept, False if filtered
        """
        # Basic length filter
        if len(word) < self.min_word_length:
            self.words_filtered += 1
            return False
        
        # NEVER filter semantic-critical words
        if self.is_semantic_critical(word):
            self.semantic_critical_preserved += 1
            self.words_preserved += 1
            return True
        
        # Compute geometric relevance
        relevance = self.compute_geometric_relevance(word, context_words, context_basin)
        
        # Keep if above threshold
        keep = relevance >= self.relevance_threshold
        
        if keep:
            self.words_preserved += 1
        else:
            self.words_filtered += 1
        
        return keep
    
    def filter_words(self,
                    words: List[str],
                    preserve_order: bool = True) -> List[str]:
        """
        Filter words based on geometric relevance in context.
        
        Args:
            words: List of words to filter
            preserve_order: If True, maintains original word order
            
        Returns:
            Filtered list of words
        """
        if not words:
            return []
        
        # Compute context basin once for efficiency
        context_basins = []
        for word in words:
            basin = self._get_basin(word)
            if basin is not None:
                context_basins.append(basin)
        
        if context_basins:
            context_basin = np.mean(context_basins, axis=0)
            context_basin = sphere_project(context_basin)
        else:
            context_basin = None
        
        # Filter words
        filtered = []
        for word in words:
            if self.should_keep_word(word, words, context_basin):
                filtered.append(word)
        
        return filtered
    
    def get_relevance_scores(self,
                           words: List[str]) -> Dict[str, float]:
        """
        Get relevance scores for all words in context.
        
        Useful for debugging and understanding filtering decisions.
        
        Returns:
            Dict mapping word -> relevance score [0, 1]
        """
        scores = {}
        
        # Compute context basin
        context_basins = []
        for word in words:
            basin = self._get_basin(word)
            if basin is not None:
                context_basins.append(basin)
        
        if context_basins:
            context_basin = np.mean(context_basins, axis=0)
            context_basin = sphere_project(context_basin)
        else:
            context_basin = None
        
        # Score each word
        for word in words:
            scores[word] = self.compute_geometric_relevance(word, words, context_basin)
        
        return scores
    
    def get_statistics(self) -> Dict:
        """Get filtering statistics."""
        total = self.words_filtered + self.words_preserved
        return {
            'words_filtered': self.words_filtered,
            'words_preserved': self.words_preserved,
            'semantic_critical_preserved': self.semantic_critical_preserved,
            'total_processed': total,
            'filter_rate': self.words_filtered / total if total > 0 else 0.0,
            'threshold': self.relevance_threshold,
        }


# Global instance (lazy-initialized)
_filter_instance: Optional[ContextualizedWordFilter] = None


def get_contextualized_filter(coordizer=None,
                              relevance_threshold: float = 0.3) -> ContextualizedWordFilter:
    """
    Get or create singleton contextualized filter.
    
    Args:
        coordizer: Optional coordizer instance
        relevance_threshold: Minimum relevance to keep word
        
    Returns:
        ContextualizedWordFilter instance
    """
    global _filter_instance
    
    if _filter_instance is None:
        # Try to get coordizer if not provided
        if coordizer is None:
            try:
                from coordizers import get_coordizer
                coordizer = get_coordizer()
            except Exception as e:
                logger.debug(f"Could not get coordizer: {e}")
        
        _filter_instance = ContextualizedWordFilter(
            coordizer=coordizer,
            relevance_threshold=relevance_threshold
        )
    
    return _filter_instance


def filter_words_geometric(words: List[str],
                          coordizer=None,
                          relevance_threshold: float = 0.3) -> List[str]:
    """
    Convenience function to filter words using geometric relevance.
    
    Replaces ancient NLP pattern: [w for w in words if w not in STOPWORDS]
    
    Args:
        words: Words to filter
        coordizer: Optional coordizer instance
        relevance_threshold: Minimum relevance to keep word
        
    Returns:
        Filtered words
    """
    filter_inst = get_contextualized_filter(coordizer, relevance_threshold)
    return filter_inst.filter_words(words)


def is_semantic_critical_word(word: str) -> bool:
    """
    Check if word is semantic-critical and should never be filtered.
    
    Fast check without needing coordizer.
    """
    return word.lower() in SEMANTIC_CRITICAL_PATTERNS


# Compatibility layer: Migration helper for existing code
def should_filter_word(word: str, context: Optional[List[str]] = None) -> bool:
    """
    Determine if word should be filtered (LEGACY COMPATIBILITY).
    
    Returns True if word should be REMOVED, False if kept.
    This is the INVERSE of should_keep_word for backward compatibility.
    """
    if not word or len(word) < 2:
        return True  # Filter very short words
    
    if is_semantic_critical_word(word):
        return False  # NEVER filter semantic-critical words
    
    # Check if truly generic
    truly_generic = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'been', 'be'}
    if word.lower() in truly_generic:
        return True  # Filter truly generic words
    
    # If we have context, use geometric filtering (if coordizer available)
    if context:
        filter_inst = get_contextualized_filter()
        if filter_inst.coordizer:
            return not filter_inst.should_keep_word(word, context)
    
    # Without context or coordizer, use conservative length-based filtering
    # Longer words tend to be more content-bearing
    return len(word) < 4


if __name__ == '__main__':
    # Demo and testing
    logging.basicConfig(level=logging.INFO)
    
    print("=== Contextualized Word Filter Demo ===\n")
    
    # Test 1: Semantic-critical word preservation
    print("Test 1: Semantic-Critical Words")
    test_words = ['not', 'good', 'the', 'very', 'bad', 'is', 'never', 'acceptable']
    print(f"Input: {test_words}")
    
    for word in test_words:
        is_critical = is_semantic_critical_word(word)
        print(f"  '{word}': {'CRITICAL (never filter)' if is_critical else 'non-critical'}")
    
    # Test 2: Filtering without coordizer (fallback mode)
    print("\nTest 2: Fallback Filtering (no coordizer)")
    filter_inst = ContextualizedWordFilter(coordizer=None, relevance_threshold=0.3)
    filtered = filter_inst.filter_words(test_words)
    print(f"Filtered: {filtered}")
    print(f"Stats: {filter_inst.get_statistics()}")
    
    # Test 3: Show difference from ancient NLP stopwords
    print("\nTest 3: Comparison with Ancient NLP Pattern")
    ancient_stopwords = {'the', 'is', 'not', 'a', 'an', 'and', 'or', 'but'}
    ancient_filtered = [w for w in test_words if w not in ancient_stopwords]
    print(f"Ancient NLP (hard stopwords): {ancient_filtered}")
    print(f"  ❌ Lost 'not' - changes meaning!")
    print(f"QIG-pure (contextualized): {filtered}")
    print(f"  ✅ Preserved 'not' and 'never' - meaning intact!")
    
    print("\n=== Demo Complete ===")
