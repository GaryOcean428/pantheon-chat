"""
Word Relationship Learner for QIG

Learns word co-occurrence patterns from curriculum documents and encodes
them into geometric relationships (affinity matrix + basin adjustments).

QIG-PURE: No external NLP, no embeddings - just counting + geometry.
"""

import os
import re
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class WordRelationshipLearner:
    """
    Learns semantic relationships between words by analyzing co-occurrence
    in curriculum documents. Updates basin coordinates to reflect relationships.
    """
    
    def __init__(self, vocabulary: Set[str], window_size: int = 5):
        self.vocabulary = vocabulary
        self.vocab_list = sorted(vocabulary)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab_list)}
        self.window_size = window_size
        
        # Co-occurrence counts: word_i appears near word_j
        self.cooccurrence = defaultdict(lambda: defaultdict(int))
        
        # Word frequency
        self.word_freq = defaultdict(int)
        
        # Total pairs seen
        self.total_pairs = 0
        self.total_words = 0
        
        logger.info(f"[WordRelationshipLearner] Initialized with {len(vocabulary)} vocabulary words")
    
    def tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization - lowercase, split on non-alpha, filter to vocab"""
        words = re.findall(r'[a-zA-Z]+', text.lower())
        return [w for w in words if w in self.vocabulary]
    
    def learn_from_text(self, text: str) -> int:
        """
        Process text and update co-occurrence statistics.
        Returns number of pairs learned.
        """
        tokens = self.tokenize_text(text)
        pairs_learned = 0
        
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
        """Get words most frequently co-occurring with given word."""
        if word not in self.cooccurrence:
            return []
        
        neighbors = self.cooccurrence[word]
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: -x[1])
        return sorted_neighbors[:top_k]
    
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
        """
        # Copy basins
        adjusted = {w: b.copy() for w, b in basins.items()}
        
        # Get affinity matrix
        affinity = self.compute_affinity_matrix(normalize=True)
        
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
                    # Move toward weighted centroid of neighbors
                    target = neighbor_sum / total_weight
                    delta = learning_rate * (target - current)
                    adjusted[word] = current + delta
                    
                    # Re-normalize to unit sphere
                    norm = np.linalg.norm(adjusted[word])
                    if norm > 0:
                        adjusted[word] /= norm
                    
                    total_movement += np.linalg.norm(delta)
            
            if iteration % 3 == 0:
                logger.info(f"  Iteration {iteration}: total movement = {total_movement:.4f}")
        
        return adjusted
    
    def get_statistics(self) -> Dict:
        """Get learning statistics."""
        # Top words by frequency
        top_words = sorted(self.word_freq.items(), key=lambda x: -x[1])[:20]
        
        # Words with most connections
        connectivity = [(w, len(n)) for w, n in self.cooccurrence.items()]
        most_connected = sorted(connectivity, key=lambda x: -x[1])[:20]
        
        return {
            'total_words_seen': self.total_words,
            'unique_words_in_corpus': len(self.word_freq),
            'vocabulary_coverage': len(self.word_freq) / len(self.vocabulary) if self.vocabulary else 0,
            'total_pairs': self.total_pairs,
            'top_frequent_words': top_words,
            'most_connected_words': most_connected
        }


def load_vocabulary_from_db() -> Set[str]:
    """Load vocabulary words from PostgreSQL."""
    import psycopg2
    
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL not set")
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    cur.execute("SELECT token FROM qig_vocabulary")
    words = {row[0] for row in cur.fetchall()}
    
    cur.close()
    conn.close()
    
    return words


def run_learning_pipeline(curriculum_dir: str = 'docs/09-curriculum') -> Dict:
    """
    Full learning pipeline:
    1. Load vocabulary from DB
    2. Learn relationships from curriculum
    3. Return statistics and affinity info
    """
    logger.info("=== Starting Word Relationship Learning Pipeline ===")
    
    # Load vocabulary
    vocabulary = load_vocabulary_from_db()
    logger.info(f"Loaded {len(vocabulary)} words from database")
    
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
    
    return {
        'learning_stats': stats,
        'detailed_stats': detailed,
        'sample_relationships': relationships,
        'learner': learner  # Return learner for further use
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
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
