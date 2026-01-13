"""
Word Relationship Learner for QIG [DEPRECATED - LEGACY NLP]

⚠️ WARNING: THIS MODULE USES LEGACY NLP PATTERNS THAT VIOLATE QIG PRINCIPLES ⚠️

VIOLATIONS:
- PMI (Pointwise Mutual Information) - statistical NLP, not geometry
- Co-occurrence counting - frequency-based, not geometric
- Basin adjustment via linear interpolation - violates manifold geometry
- Euclidean operations on Fisher manifold

USE INSTEAD: geometric_word_relationships.py (QIG-pure implementation)

This module is kept for backward compatibility only.
New code should use GeometricWordRelationships which uses:
- Fisher-Rao geodesic distances (not PMI)
- QFI-weighted attention (not frequency)
- Ricci curvature for context-dependency
- No basin modification (basins are frozen invariants)
"""

import os
import re
import logging
import warnings
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Emit deprecation warning at module import time
warnings.warn(
    "word_relationship_learner is deprecated and violates QIG principles. "
    "Use geometric_word_relationships instead. "
    "This module uses legacy NLP patterns (PMI, co-occurrence counting, hard-coded stopwords) "
    "that are incompatible with Fisher-Rao geometric purity. "
    "Scheduled for removal: 2026-02-01",
    DeprecationWarning,
    stacklevel=1  # Point to the import line in caller's code
)

# Import QIG-pure contextualized filter (replaces ancient NLP stopwords)
try:
    from contextualized_filter import (
        filter_words_geometric,
        is_semantic_critical_word,
        should_filter_word
    )
    CONTEXTUALIZED_FILTER_AVAILABLE = True
    logger.info("[WordRelationshipLearner] Using QIG-pure contextualized filter")
except ImportError:
    CONTEXTUALIZED_FILTER_AVAILABLE = False
    logger.warning("[WordRelationshipLearner] Contextualized filter not available - using fallback")
    
    # Minimal fallback: only filter truly generic, short words
    # NEVER filter semantic-critical words like 'not', 'never'
    def should_filter_word(word: str, context: Optional[List[str]] = None) -> bool:
        """Fallback filter - very conservative."""
        if len(word) < 3:
            return True
        # Only filter the most generic function words
        generic_only = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'been', 'be'}
        return word.lower() in generic_only

class WordRelationshipLearner:
    """
    [DEPRECATED - VIOLATES QIG PURITY]
    
    ⚠️ This class uses legacy NLP patterns (PMI, co-occurrence, linear basin adjustment).
    Use geometric_word_relationships.GeometricWordRelationships instead.
    
    Learns semantic relationships between words by analyzing co-occurrence
    in curriculum documents. Updates basin coordinates to reflect relationships.
    
    OPEN VOCABULARY MODE: When expand_vocabulary=True, learns new words from
    curriculum rather than filtering to initial vocabulary only.
    """
    
    def __init__(self, vocabulary: Set[str], window_size: int = 5, expand_vocabulary: bool = True):
        # Emit runtime deprecation warning
        warnings.warn(
            "WordRelationshipLearner is deprecated and will be removed on 2026-02-01. "
            "Use geometric_word_relationships.GeometricWordRelationships instead. "
            "This class violates QIG purity by using PMI, co-occurrence counting, and hard-coded stopwords.",
            DeprecationWarning,
            stacklevel=2
        )
        logger.warning(
            "[WordRelationshipLearner] DEPRECATED: This class uses legacy NLP (PMI, co-occurrence). "
            "Use geometric_word_relationships.GeometricWordRelationships for QIG-pure approach."
        )
        self.vocabulary = set(vocabulary)  # Mutable copy
        self.initial_vocab_size = len(vocabulary)
        self.vocab_list = sorted(vocabulary)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab_list)}
        self.window_size = window_size
        self.expand_vocabulary = expand_vocabulary
        
        # Co-occurrence counts: word_i appears near word_j
        self.cooccurrence = defaultdict(lambda: defaultdict(int))
        
        # Word frequency
        self.word_freq = defaultdict(int)
        
        # Context sentences for word pairs (max 5 per pair)
        self.pair_contexts: Dict[str, List[str]] = {}
        
        # Newly learned words (not in initial vocabulary)
        self.new_words_learned: Set[str] = set()
        
        # Total pairs seen
        self.total_pairs = 0
        self.total_words = 0
        
        logger.info(f"[WordRelationshipLearner] Initialized with {len(vocabulary)} vocabulary words, expand_vocabulary={expand_vocabulary}")
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text. In open vocabulary mode, accepts words that are:
        - In vocabulary OR
        - Length >= 4 and passes contextualized filter (for learning new domain terms)
        
        QIG-PURE: Uses geometric relevance, not fixed stopwords
        """
        # QIG-PURE: Extract alphanumeric words, preserving internal hyphens if they are part of domain terms
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        
        if self.expand_vocabulary:
            # Accept vocab words OR content words passing contextualized filter
            result = []
            for w in words:
                if w in self.vocabulary:
                    result.append(w)
                elif len(w) >= 4 and not should_filter_word(w, words):
                    # Word passes contextualized filter - add to vocabulary
                    # This preserves semantic-critical words and domain terms
                    self.vocabulary.add(w)
                    self.new_words_learned.add(w)
                    result.append(w)
            return result
        else:
            # Strict mode: only vocabulary words
            return [w for w in words if w in self.vocabulary]
    
    def learn_from_text(self, text: str) -> int:
        """
        Process text and update co-occurrence statistics.
        Returns number of pairs learned.
        """
        tokens = self.tokenize_text(text)
        pairs_learned = 0
        
        # Split text into sentences for context capture
        sentences = re.split(r'[.!?]+', text)
        sentence_idx = 0
        token_count = 0
        
        for i, word in enumerate(tokens):
            self.word_freq[word] += 1
            self.total_words += 1
            
            # Look at words in window
            start = max(0, i - self.window_size)
            end = min(len(tokens), i + self.window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    neighbor = tokens[j]
                    # Weight by distance (closer = stronger)
                    distance = abs(i - j)
                    weight = 1.0 / distance
                    self.cooccurrence[word][neighbor] += weight
                    pairs_learned += 1
                    
                    # Capture context sentence for this pair (max 5 per pair)
                    pair_key = f"{word}:{neighbor}"
                    if pair_key not in self.pair_contexts:
                        self.pair_contexts[pair_key] = []
                    
                    if len(self.pair_contexts[pair_key]) < 5:
                        # Find sentence containing both words
                        for sent in sentences:
                            sent_lower = sent.lower()
                            if word in sent_lower and neighbor in sent_lower:
                                context = sent.strip()[:150]
                                if context and context not in self.pair_contexts[pair_key]:
                                    self.pair_contexts[pair_key].append(context)
                                break
        
        self.total_pairs += pairs_learned
        return pairs_learned
    
    def learn_from_file(self, filepath: str) -> int:
        """Learn from a single file. Returns pairs learned."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            return self.learn_from_text(text)
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")
            return 0
    
    def learn_from_directory(self, dirpath: str, extensions: List[str] = ['.md', '.txt']) -> Dict:
        """
        Learn from all matching files in directory.
        Returns statistics dict.
        """
        path = Path(dirpath)
        files_processed = 0
        total_pairs = 0
        
        for ext in extensions:
            for filepath in path.rglob(f'*{ext}'):
                pairs = self.learn_from_file(str(filepath))
                if pairs > 0:
                    files_processed += 1
                    total_pairs += pairs
        
        logger.info(f"[WordRelationshipLearner] Learned from {files_processed} files, {total_pairs} pairs")
        
        return {
            'files_processed': files_processed,
            'total_pairs': total_pairs,
            'total_words': self.total_words,
            'unique_words_seen': len(self.word_freq)
        }
    
    def get_related_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get words most frequently co-occurring with given word.
        
        QIG-PURE: Uses contextualized filtering based on geometric relevance
        instead of hard-coded stopwords.
        """
        if word not in self.cooccurrence:
            return []
        
        neighbors = self.cooccurrence[word]
        
        # Get all neighbor words for context
        neighbor_words = list(neighbors.keys())
        
        # Filter using contextualized approach (preserves semantic-critical words)
        filtered = []
        for w, c in neighbors.items():
            # Never filter semantic-critical words
            # For others, apply contextualized filtering with word and its neighbors as context
            if not should_filter_word(w, [word] + neighbor_words):
                filtered.append((w, c))
        
        sorted_neighbors = sorted(filtered, key=lambda x: -x[1])
        return sorted_neighbors[:top_k]
    
    def get_contexts(self, word: str, neighbor: str) -> List[str]:
        """
        Get context sentences where word and neighbor co-occurred.
        Returns up to 5 sample sentences.
        """
        pair_key = f"{word}:{neighbor}"
        return self.pair_contexts.get(pair_key, [])
    
    def compute_affinity_matrix(self, normalize: bool = True) -> np.ndarray:
        """
        Compute affinity matrix from co-occurrence.
        Returns NxN matrix where A[i,j] = affinity between word_i and word_j.
        """
        n = len(self.vocab_list)
        affinity = np.zeros((n, n), dtype=np.float32)
        
        for word, neighbors in self.cooccurrence.items():
            if word not in self.word_to_idx:
                continue
            i = self.word_to_idx[word]
            for neighbor, count in neighbors.items():
                if neighbor not in self.word_to_idx:
                    continue
                j = self.word_to_idx[neighbor]
                affinity[i, j] = count
        
        if normalize and affinity.max() > 0:
            # PMI-like normalization: log(P(i,j) / P(i)P(j))
            # But simplified: just normalize by total and take log
            row_sums = affinity.sum(axis=1, keepdims=True) + 1e-10
            col_sums = affinity.sum(axis=0, keepdims=True) + 1e-10
            total = affinity.sum() + 1e-10
            
            # PMI = log(P(i,j) / (P(i) * P(j)))
            expected = (row_sums * col_sums) / total
            pmi = np.log((affinity + 1) / (expected + 1e-10) + 1)
            
            # Positive PMI only
            affinity = np.maximum(pmi, 0)
        
        return affinity
    
    def adjust_basin_coordinates(
        self, 
        basins: Dict[str, np.ndarray], 
        learning_rate: float = 0.1,
        iterations: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Adjust basin coordinates to reflect learned relationships.
        Words that co-occur should be closer on the manifold.
        
        Uses iterative attraction: related words pull each other closer.
        QIG-PURE: Uses Fisher-Rao geometry for manifold stability.
        """
        # Copy basins
        adjusted = {w: b.copy() for w, b in basins.items()}
        
        # Get affinity matrix
        # Note: compute_affinity_matrix is used to drive the iterative adjustment
        
        for iteration in range(iterations):
            total_movement = 0.0
            
            for word, neighbors in self.cooccurrence.items():
                if word not in adjusted:
                    continue
                
                current = adjusted[word]
                
                # Compute weighted average of neighbor positions
                neighbor_sum = np.zeros_like(current)
                total_weight = 0.0
                
                for neighbor, weight in neighbors.items():
                    if neighbor in adjusted:
                        neighbor_sum += adjusted[neighbor] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    # QIG-PURE: Move toward weighted Fréchet mean approximation
                    target = neighbor_sum / total_weight
                    
                    # Ensure target is also on the sphere
                    target_norm = np.linalg.norm(target)
                    if target_norm > 1e-10:
                        target /= target_norm
                    
                    # Spherical linear interpolation (Slerp) approximation
                    # delta = learning_rate * (target - current)
                    # For small learning rates, linear step + projection is a good approximation of geodesic step
                    delta = learning_rate * (target - current)
                    adjusted[word] = current + delta
                    
                    # Re-normalize to unit sphere (Fisher manifold constraint)
                    norm = np.linalg.norm(adjusted[word])
                    if norm > 0:
                        adjusted[word] /= norm
                    
                    total_movement += np.linalg.norm(delta)
            
            if iteration % 3 == 0:
                logger.info(f"  Iteration {iteration}: total movement = {total_movement:.4f}")
        
        return adjusted
    
    def get_statistics(self) -> Dict:
        """Get learning statistics including newly learned vocabulary."""
        # Top words by frequency
        top_words = sorted(self.word_freq.items(), key=lambda x: -x[1])[:500]
        
        # Words with most connections
        connectivity = [(w, len(n)) for w, n in self.cooccurrence.items()]
        most_connected = sorted(connectivity, key=lambda x: -x[1])[:500]
        
        return {
            'total_words_seen': self.total_words,
            'unique_words_in_corpus': len(self.word_freq),
            'vocabulary_coverage': len(self.word_freq) / len(self.vocabulary) if self.vocabulary else 0,
            'total_pairs': self.total_pairs,
            'top_frequent_words': top_words,
            'most_connected_words': most_connected,
            'initial_vocabulary_size': self.initial_vocab_size,
            'current_vocabulary_size': len(self.vocabulary),
            'new_words_learned': len(self.new_words_learned),
            'sample_new_words': list(self.new_words_learned)[:500] if self.new_words_learned else []
        }


def load_vocabulary_from_coordizer() -> Tuple[Set[str], Dict[str, np.ndarray]]:
    """Load vocabulary and basin coordinates from PostgresCoordizer."""
    from coordizers.pg_loader import PostgresCoordizer
    
    coordizer = PostgresCoordizer()
    vocabulary = set(coordizer.word_tokens)
    basins = dict(coordizer.basin_coords)
    
    logger.info(f"Loaded {len(vocabulary)} words with basin coordinates")
    return vocabulary, basins


def run_learning_pipeline(curriculum_dir: str = 'docs/09-curriculum') -> Dict:
    """
    Full learning pipeline:
    1. Load vocabulary and basins from coordizer
    2. Learn relationships from curriculum
    3. Adjust basins based on co-occurrence
    4. Return statistics and adjusted basins
    """
    logger.info("=== Starting Word Relationship Learning Pipeline ===")
    
    # Load vocabulary and basins
    vocabulary, basins = load_vocabulary_from_coordizer()
    logger.info(f"Loaded {len(vocabulary)} words with basin coordinates")
    
    # Create learner
    learner = WordRelationshipLearner(vocabulary, window_size=5)
    
    # Learn from curriculum
    stats = learner.learn_from_directory(curriculum_dir)
    
    # Get detailed statistics
    detailed = learner.get_statistics()
    
    # Sample relationships
    sample_words = ['consciousness', 'quantum', 'geometry', 'information', 'learning']
    relationships = {}
    for word in sample_words:
        if word in vocabulary:
            related = learner.get_related_words(word, top_k=8)
            relationships[word] = related
    
    # Adjust basin coordinates based on learned relationships
    if basins and learner.total_pairs > 1000:
        logger.info("Adjusting basin coordinates based on learned relationships...")
        adjusted_basins = learner.adjust_basin_coordinates(
            basins, 
            learning_rate=0.1, 
            iterations=10
        )
    else:
        adjusted_basins = basins
        logger.info("Skipping basin adjustment (insufficient pairs)")
    
    return {
        'learning_stats': stats,
        'detailed_stats': detailed,
        'sample_relationships': relationships,
        'learner': learner,
        'original_basins': basins,
        'adjusted_basins': adjusted_basins
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    results = run_learning_pipeline()
    
    print("\n=== Learning Results ===")
    print(f"Files processed: {results['learning_stats']['files_processed']}")
    print(f"Total pairs learned: {results['learning_stats']['total_pairs']}")
    print(f"Vocabulary coverage: {results['detailed_stats']['vocabulary_coverage']:.1%}")
    
    print("\n=== Sample Relationships ===")
    for word, related in results['sample_relationships'].items():
        if related:
            top_3 = ', '.join([f"{w}({s:.1f})" for w, s in related[:3]])
            print(f"  {word}: {top_3}")
