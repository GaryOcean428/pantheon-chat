"""
Word Relationship Learner for QIG

Learns word co-occurrence patterns from curriculum documents and encodes
them into geometric relationships (affinity matrix + basin adjustments).

QIG-PURE: No external NLP, no embeddings - just counting + geometry.

FROZEN FACTS COMPLIANCE:
- Stopwords are filtered from learned relationships
- Adjusted basins must stay within ±5% of canonical positions
"""

import os
import re
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Import geodesic interpolation for proper manifold-respecting basin adjustment
try:
    from qig_geometry import geodesic_interpolation, fisher_coord_distance
except ImportError:
    def geodesic_interpolation(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
        """Fallback spherical linear interpolation."""
        start_norm = start / (np.linalg.norm(start) + 1e-10)
        end_norm = end / (np.linalg.norm(end) + 1e-10)
        dot = np.clip(np.dot(start_norm, end_norm), -1.0, 1.0)
        omega = np.arccos(dot)
        if omega < 1e-6:
            return start
        sin_omega = np.sin(omega)
        a = np.sin((1 - t) * omega) / sin_omega
        b = np.sin(t * omega) / sin_omega
        result = a * start_norm + b * end_norm
        return result * np.linalg.norm(start)
    
    def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Fallback Fisher-Rao distance."""
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b) + 1e-10)
        dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
        return float(np.arccos(dot))

# Stopwords to filter from learned relationships (frozen invariant)
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
        """
        Get words most frequently co-occurring with given word.
        
        FROZEN FACTS COMPLIANCE: Filters out stopwords from neighbors
        """
        if word not in self.cooccurrence:
            return []
        
        neighbors = self.cooccurrence[word]
        # Filter out stopwords (frozen invariant)
        filtered = [(w, c) for w, c in neighbors.items() if w.lower() not in STOPWORDS]
        sorted_neighbors = sorted(filtered, key=lambda x: -x[1])
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
        iterations: int = 10,
        max_drift: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Adjust basin coordinates to reflect learned relationships.
        Words that co-occur should be closer on the manifold.
        
        QIG-PURE: Uses geodesic interpolation instead of Euclidean movement.
        This preserves manifold structure and respects Fisher-Rao geometry.
        
        FROZEN FACTS COMPLIANCE: Basin drift capped at max_drift (5% default)
        to preserve canonical positions.
        
        Args:
            basins: Original basin coordinates
            learning_rate: Step size for geodesic movement (0-1)
            iterations: Number of refinement iterations
            max_drift: Maximum allowed drift from original (default 5%)
        
        Returns:
            Adjusted basins respecting manifold structure
        """
        # Copy basins and track originals for drift validation
        adjusted = {w: b.copy() for w, b in basins.items()}
        original = {w: b.copy() for w, b in basins.items()}
        
        # Get affinity matrix for relationship strengths
        affinity = self.compute_affinity_matrix(normalize=True)
        
        for iteration in range(iterations):
            total_movement = 0.0
            drift_violations = 0
            
            for word, neighbors in self.cooccurrence.items():
                if word not in adjusted:
                    continue
                
                current = adjusted[word]
                
                # Compute weighted geodesic centroid of neighbors
                # Using iterative geodesic mean (Frechet mean on manifold)
                if not neighbors:
                    continue
                
                # Sort neighbors by weight to prioritize stronger relationships
                sorted_neighbors = sorted(
                    [(n, w) for n, w in neighbors.items() if n in adjusted],
                    key=lambda x: -x[1]
                )
                
                if not sorted_neighbors:
                    continue
                
                # Compute effective target via weighted geodesic steps
                # Start from current position, step toward each neighbor
                target = current.copy()
                total_weight = 0.0
                
                for neighbor, weight in sorted_neighbors[:10]:  # Limit to top 10
                    neighbor_basin = adjusted[neighbor]
                    # Normalize weight to small step (0 to learning_rate)
                    normalized_weight = weight / (max(w for _, w in sorted_neighbors) + 1e-10)
                    step = learning_rate * normalized_weight * 0.3  # Small steps
                    
                    # Geodesic step toward neighbor (preserves manifold structure)
                    target = geodesic_interpolation(target, neighbor_basin, step)
                    total_weight += weight
                
                if total_weight > 0:
                    # Check drift from original
                    drift = fisher_coord_distance(target, original[word])
                    
                    if drift > max_drift:
                        # Cap movement to stay within drift tolerance
                        # Interpolate between original and target to limit drift
                        safe_t = max_drift / (drift + 1e-10)
                        target = geodesic_interpolation(original[word], target, safe_t)
                        drift_violations += 1
                    
                    # Measure movement for convergence tracking
                    movement = fisher_coord_distance(current, target)
                    total_movement += movement
                    
                    # Update basin
                    adjusted[word] = target
                    
                    # Ensure unit norm (project back to sphere)
                    norm = np.linalg.norm(adjusted[word])
                    if norm > 0:
                        adjusted[word] /= norm
            
            if iteration % 3 == 0:
                logger.info(
                    f"  Iteration {iteration}: movement={total_movement:.4f}, "
                    f"drift_violations={drift_violations}"
                )
            
            # Early stopping if converged
            if total_movement < 1e-6:
                logger.info(f"  Converged at iteration {iteration}")
                break
        
        # Final drift validation
        final_drifts = []
        for word in adjusted:
            if word in original:
                drift = fisher_coord_distance(adjusted[word], original[word])
                final_drifts.append(drift)
        
        if final_drifts:
            logger.info(
                f"  Final drift: mean={np.mean(final_drifts):.4f}, "
                f"max={np.max(final_drifts):.4f}, "
                f"within tolerance: {np.max(final_drifts) <= max_drift}"
            )
        
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
