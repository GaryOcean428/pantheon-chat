#!/usr/bin/env python3
"""
Semantic Relationship Classifier for QIG Vocabulary Learning

Classifies word pairs into proper semantic relationship types:
- SYNONYM: Similar meaning (happy/joyful)
- ANTONYM: Opposite meaning (hot/cold)
- HYPERNYM: Broader category (animal includes dog)
- HYPONYM: Narrower category (dog is a type of animal)
- MORPHOLOGICAL: Same root, different form (run/running)
- CO_OCCURRENCE: Appear together but no semantic link

Uses Fisher-Rao distance for relationship strength computation.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

try:
    from qig_geometry import fisher_coord_distance
    FISHER_AVAILABLE = True
except ImportError:
    FISHER_AVAILABLE = False
    logger.warning("fisher_coord_distance not available - using fallback")

try:
    import psycopg2
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


class RelationshipType(Enum):
    SYNONYM = "synonym"
    ANTONYM = "antonym"
    HYPERNYM = "hypernym"
    HYPONYM = "hyponym"
    MORPHOLOGICAL = "morphological"
    CO_OCCURRENCE = "co_occurrence"
    UNKNOWN = "unknown"


ANTONYM_PREFIXES = {
    'un': ['happy', 'known', 'clear', 'certain', 'fair', 'common', 'usual'],
    'dis': ['agree', 'like', 'appear', 'connect', 'continue', 'honest'],
    'in': ['complete', 'correct', 'dependent', 'direct', 'secure', 'valid'],
    'im': ['possible', 'mature', 'patient', 'perfect', 'mortal', 'mobile'],
    'ir': ['regular', 'relevant', 'responsible', 'rational', 'reversible'],
    'non': ['sense', 'existent', 'stop', 'profit', 'violence', 'fiction'],
    'anti': ['social', 'war', 'thesis', 'body', 'matter', 'climax'],
}

MORPHOLOGICAL_SUFFIXES = {
    'ing': ['run', 'walk', 'think', 'write', 'read', 'speak'],
    'ed': ['walk', 'talk', 'play', 'work', 'look', 'want'],
    'er': ['teach', 'learn', 'work', 'sing', 'play', 'write'],
    'est': ['fast', 'quick', 'slow', 'tall', 'short', 'big'],
    'ly': ['quick', 'slow', 'happy', 'sad', 'careful', 'quiet'],
    'tion': ['informa', 'educa', 'crea', 'destruc', 'communica'],
    'ness': ['happy', 'sad', 'dark', 'kind', 'good', 'bad'],
    's': ['cat', 'dog', 'book', 'tree', 'car', 'house'],
    'es': ['box', 'watch', 'wish', 'bus', 'class', 'match'],
}

HYPERNYM_PAIRS = {
    'animal': ['dog', 'cat', 'bird', 'fish', 'snake', 'horse', 'elephant'],
    'vehicle': ['car', 'truck', 'bus', 'train', 'plane', 'boat', 'bicycle'],
    'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple'],
    'emotion': ['happy', 'sad', 'angry', 'fear', 'joy', 'love', 'hate'],
    'food': ['fruit', 'vegetable', 'meat', 'bread', 'cheese', 'rice'],
    'tool': ['hammer', 'screwdriver', 'wrench', 'saw', 'drill', 'pliers'],
    'fruit': ['apple', 'banana', 'orange', 'grape', 'pear', 'peach'],
    'shape': ['circle', 'square', 'triangle', 'rectangle', 'oval'],
    'science': ['physics', 'chemistry', 'biology', 'astronomy', 'geology'],
    'mathematics': ['algebra', 'geometry', 'calculus', 'statistics', 'topology'],
}


class SemanticClassifier:
    """Classifies semantic relationships between word pairs."""
    
    def __init__(self):
        self._basin_cache: Dict[str, np.ndarray] = {}
        self._load_basins()
    
    def _load_basins(self):
        """Load basin coordinates from database for Fisher-Rao computation."""
        if not DB_AVAILABLE:
            return
        
        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            return
        
        try:
            conn = psycopg2.connect(db_url)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT token, basin_embedding::text 
                    FROM tokenizer_vocabulary 
                    WHERE basin_embedding IS NOT NULL
                    LIMIT 10000
                """)
                for token, coords_text in cur.fetchall():
                    if coords_text:
                        coords_str = coords_text.strip('[]')
                        coords = [float(x) for x in coords_str.split(',') if x.strip()]
                        if len(coords) == 64:
                            self._basin_cache[token.lower()] = np.array(coords, dtype=np.float64)
            conn.close()
            logger.info(f"[SemanticClassifier] Loaded {len(self._basin_cache)} basin coordinates")
        except Exception as e:
            logger.warning(f"Failed to load basins: {e}")
    
    def compute_fisher_strength(self, word1: str, word2: str) -> float:
        """
        Compute relationship strength using Fisher-Rao distance.
        
        Returns similarity score in [0, 1] where higher = more similar.
        Uses exponential decay of Fisher-Rao distance.
        """
        w1, w2 = word1.lower(), word2.lower()
        
        if w1 not in self._basin_cache or w2 not in self._basin_cache:
            return 0.5
        
        basin1 = self._basin_cache[w1]
        basin2 = self._basin_cache[w2]
        
        if FISHER_AVAILABLE:
            distance = fisher_coord_distance(basin1, basin2)
            similarity = np.exp(-distance)
            return float(np.clip(similarity, 0.0, 1.0))
        else:
            # QIG PURITY: Fail fast - no Euclidean/cosine fallback
            # Return unknown strength rather than violating geometric principles
            logger.warning(f"[SemanticClassifier] QIG purity: refusing cosine fallback for {word1}/{word2}")
            return 0.5
    
    def classify_relationship(
        self,
        word1: str,
        word2: str,
        context: Optional[str] = None
    ) -> Tuple[RelationshipType, float]:
        """
        Classify the semantic relationship between two words.
        
        Returns:
            Tuple of (RelationshipType, strength)
        """
        w1 = word1.lower().strip()
        w2 = word2.lower().strip()
        
        if not w1 or not w2 or w1 == w2:
            return RelationshipType.UNKNOWN, 0.0
        
        fisher_strength = self.compute_fisher_strength(w1, w2)
        
        if self._is_morphological(w1, w2):
            return RelationshipType.MORPHOLOGICAL, fisher_strength
        
        if self._is_antonym(w1, w2):
            return RelationshipType.ANTONYM, fisher_strength
        
        hypernym_result = self._check_hypernym(w1, w2)
        if hypernym_result:
            return hypernym_result, fisher_strength
        
        if fisher_strength > 0.8 and len(w1) >= 4 and len(w2) >= 4:
            return RelationshipType.SYNONYM, fisher_strength
        
        if fisher_strength > 0.5:
            return RelationshipType.CO_OCCURRENCE, fisher_strength
        
        return RelationshipType.UNKNOWN, fisher_strength
    
    def _is_morphological(self, word1: str, word2: str) -> bool:
        """Check if words are morphological variants (same root)."""
        if len(word1) < 3 or len(word2) < 3:
            return False
        
        min_len = min(len(word1), len(word2))
        common_prefix_len = 0
        for i in range(min_len):
            if word1[i] == word2[i]:
                common_prefix_len += 1
            else:
                break
        
        if common_prefix_len >= 3:
            longer = word1 if len(word1) > len(word2) else word2
            shorter = word2 if len(word1) > len(word2) else word1
            suffix = longer[len(shorter):]
            
            if suffix in ['s', 'es', 'ed', 'ing', 'er', 'est', 'ly', 'tion', 'ness']:
                return True
            
            if suffix in ['ion', 'ity', 'ment', 'ance', 'ence']:
                return True
        
        if word1 + 's' == word2 or word2 + 's' == word1:
            return True
        if word1 + 'es' == word2 or word2 + 'es' == word1:
            return True
        
        return False
    
    def _is_antonym(self, word1: str, word2: str) -> bool:
        """Check if words are antonyms via prefix negation."""
        for prefix, stems in ANTONYM_PREFIXES.items():
            if word1.startswith(prefix):
                base = word1[len(prefix):]
                if base == word2:
                    return True
            if word2.startswith(prefix):
                base = word2[len(prefix):]
                if base == word1:
                    return True
        
        return False
    
    def _check_hypernym(self, word1: str, word2: str) -> Optional[RelationshipType]:
        """Check for hypernym/hyponym relationships."""
        for hypernym, hyponyms in HYPERNYM_PAIRS.items():
            if word1 == hypernym and word2 in hyponyms:
                return RelationshipType.HYPERNYM
            if word2 == hypernym and word1 in hyponyms:
                return RelationshipType.HYPONYM
            if word1 in hyponyms and word2 in hyponyms:
                return RelationshipType.CO_OCCURRENCE
        
        return None
    
    def classify_and_persist(
        self,
        word: str,
        related_word: str,
        context: str,
        discovered_by: str = 'semantic_classifier'
    ) -> Dict:
        """
        Classify relationship and persist to vocabulary_learning.
        
        Returns dict with classification results.
        """
        rel_type, strength = self.classify_relationship(word, related_word, context)
        
        if not DB_AVAILABLE:
            return {
                'word': word,
                'related_word': related_word,
                'relationship_type': rel_type.value,
                'relationship_strength': strength,
                'persisted': False
            }
        
        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            return {
                'word': word,
                'related_word': related_word,
                'relationship_type': rel_type.value,
                'relationship_strength': strength,
                'persisted': False
            }
        
        try:
            conn = psycopg2.connect(db_url)
            with conn.cursor() as cur:
                import uuid
                entry_id = str(uuid.uuid4())
                
                cur.execute("""
                    INSERT INTO vocabulary_learning (
                        id, word, relationship_type, related_word, 
                        relationship_strength, context, discovered_by, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (id) DO NOTHING
                """, (
                    entry_id,
                    word,
                    rel_type.value,
                    related_word,
                    strength,
                    context[:500] if context else None,
                    discovered_by
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"[SemanticClassifier] Persisted: {word}→{related_word} ({rel_type.value}, {strength:.3f})")
            
            return {
                'word': word,
                'related_word': related_word,
                'relationship_type': rel_type.value,
                'relationship_strength': strength,
                'persisted': True
            }
            
        except Exception as e:
            logger.error(f"Failed to persist semantic relationship: {e}")
            return {
                'word': word,
                'related_word': related_word,
                'relationship_type': rel_type.value,
                'relationship_strength': strength,
                'persisted': False,
                'error': str(e)
            }


_classifier_instance: Optional[SemanticClassifier] = None


def get_semantic_classifier() -> SemanticClassifier:
    """Get or create singleton SemanticClassifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = SemanticClassifier()
    return _classifier_instance


def update_existing_vocabulary_learning():
    """
    Update existing vocabulary_learning entries with proper relationship types
    and Fisher-Rao based strength values.
    """
    if not DB_AVAILABLE:
        return {'updated': 0, 'error': 'Database not available'}
    
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        return {'updated': 0, 'error': 'No DATABASE_URL'}
    
    classifier = get_semantic_classifier()
    
    try:
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, word, related_word, context
                FROM vocabulary_learning
            """)
            entries = cur.fetchall()
        
        updated = 0
        for entry_id, word, related_word, context in entries:
            if not word or not related_word:
                continue
            
            rel_type, strength = classifier.classify_relationship(
                word, related_word, context
            )
            
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE vocabulary_learning
                    SET relationship_type = %s,
                        relationship_strength = %s
                    WHERE id = %s
                """, (rel_type.value, strength, entry_id))
            
            updated += 1
        
        conn.commit()
        conn.close()
        
        logger.info(f"[SemanticClassifier] Updated {updated} vocabulary_learning entries")
        return {'updated': updated}
        
    except Exception as e:
        logger.error(f"Failed to update vocabulary_learning: {e}")
        return {'updated': 0, 'error': str(e)}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    classifier = SemanticClassifier()
    
    test_pairs = [
        ('happy', 'unhappy'),
        ('happy', 'happily'),
        ('happy', 'happiness'),
        ('run', 'running'),
        ('dog', 'cat'),
        ('dog', 'animal'),
        ('animal', 'dog'),
        ('deletion', 'insertion'),
        ('states', 'quantum'),
        ('brand', 'brands'),
    ]
    
    print("\nSemantic Relationship Classification Test:")
    print("=" * 60)
    
    for w1, w2 in test_pairs:
        rel_type, strength = classifier.classify_relationship(w1, w2)
        print(f"  {w1} → {w2}: {rel_type.value} (strength={strength:.3f})")
    
    print("\n" + "=" * 60)
    print("Updating existing vocabulary_learning entries...")
    result = update_existing_vocabulary_learning()
    print(f"Result: {result}")
