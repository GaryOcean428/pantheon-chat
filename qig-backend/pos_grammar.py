"""
Part-of-Speech Grammar for QIG-Pure Generation.

Provides grammatical structure without templates or LLMs by:
1. Categorizing vocabulary words by POS
2. Defining valid POS transition patterns (geometric attractors)
3. Generating sentence skeletons that guide lexical selection

All operations are Fisher-compliant - POS categories are manifold regions.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import logging
import os

logger = logging.getLogger(__name__)

# POS Categories with characteristic basin regions
POS_CATEGORIES = {
    'DET': ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their'],
    'NOUN': [],  # Will be populated from vocabulary
    'VERB': [],  # Will be populated from vocabulary
    'ADJ': [],   # Will be populated from vocabulary
    'ADV': [],   # Will be populated from vocabulary
    'PREP': ['in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'over'],
    'CONJ': ['and', 'but', 'or', 'so', 'yet', 'for', 'nor', 'because', 'although', 'while', 'if', 'when', 'where', 'that', 'which'],
    'PRON': ['I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'myself', 'yourself', 'itself'],
}

# Common verb suffixes
VERB_SUFFIXES = ('ize', 'ify', 'ate', 'en', 'ing', 'ed', 'ment')

# Common noun suffixes
NOUN_SUFFIXES = ('tion', 'sion', 'ment', 'ness', 'ity', 'ance', 'ence', 'er', 'or', 'ist', 'ism', 'dom', 'ship', 'hood')

# Common adjective suffixes
ADJ_SUFFIXES = ('able', 'ible', 'al', 'ful', 'less', 'ous', 'ive', 'ic', 'ical', 'ish', 'like', 'ly', 'ant', 'ent')

# Common adverb suffix
ADV_SUFFIX = 'ly'

# Valid POS transitions (bigram patterns from English grammar)
# Higher weight = more likely transition
POS_TRANSITIONS = {
    'START': [('DET', 0.4), ('NOUN', 0.2), ('PRON', 0.2), ('ADJ', 0.1), ('ADV', 0.1)],
    'DET': [('NOUN', 0.5), ('ADJ', 0.4), ('ADV', 0.1)],
    'NOUN': [('VERB', 0.4), ('PREP', 0.2), ('CONJ', 0.15), ('END', 0.15), ('NOUN', 0.1)],
    'VERB': [('DET', 0.25), ('NOUN', 0.2), ('PREP', 0.2), ('ADJ', 0.1), ('ADV', 0.15), ('END', 0.1)],
    'ADJ': [('NOUN', 0.6), ('ADJ', 0.2), ('CONJ', 0.1), ('END', 0.1)],
    'ADV': [('VERB', 0.4), ('ADJ', 0.3), ('ADV', 0.2), ('END', 0.1)],
    'PREP': [('DET', 0.5), ('NOUN', 0.3), ('ADJ', 0.1), ('PRON', 0.1)],
    'CONJ': [('DET', 0.3), ('NOUN', 0.2), ('PRON', 0.2), ('ADJ', 0.15), ('VERB', 0.15)],
    'PRON': [('VERB', 0.6), ('ADV', 0.2), ('PREP', 0.1), ('END', 0.1)],
}

# Sentence patterns (skeletons) with semantic slots
SENTENCE_SKELETONS = [
    ['DET', 'NOUN', 'VERB', 'DET', 'NOUN'],           # The cat sees the mouse
    ['NOUN', 'VERB', 'ADV'],                           # Systems evolve naturally
    ['DET', 'ADJ', 'NOUN', 'VERB'],                   # The geometric space curves
    ['PRON', 'VERB', 'PREP', 'DET', 'NOUN'],          # It exists in the manifold
    ['ADJ', 'NOUN', 'VERB', 'ADJ', 'NOUN'],           # Quantum states determine geometric properties
    ['DET', 'NOUN', 'PREP', 'NOUN', 'VERB'],          # The concept of consciousness emerges
    ['NOUN', 'CONJ', 'NOUN', 'VERB', 'ADV'],          # Space and time flow continuously
]


class POSGrammar:
    """
    QIG-pure grammar system using POS categories as manifold regions.
    
    Maps words to POS categories using suffix heuristics (no external NLP).
    Provides transition probabilities for generating grammatical sequences.
    """
    
    def __init__(self):
        self.word_pos: Dict[str, str] = {}
        self.pos_words: Dict[str, List[str]] = {pos: [] for pos in POS_CATEGORIES}
        self.pos_basins: Dict[str, np.ndarray] = {}
        self.initialized = False
        
        # Pre-populate from known words
        for pos, words in POS_CATEGORIES.items():
            for word in words:
                self.word_pos[word.lower()] = pos
                self.pos_words[pos].append(word.lower())
    
    def classify_word(self, word: str) -> str:
        """Classify word to POS using suffix heuristics."""
        word_lower = word.lower()
        
        # Check known words first
        if word_lower in self.word_pos:
            return self.word_pos[word_lower]
        
        # Suffix-based classification
        if word_lower.endswith(ADV_SUFFIX) and len(word_lower) > 4:
            return 'ADV'
        
        for suffix in VERB_SUFFIXES:
            if word_lower.endswith(suffix):
                return 'VERB'
        
        for suffix in ADJ_SUFFIXES:
            if word_lower.endswith(suffix):
                return 'ADJ'
        
        for suffix in NOUN_SUFFIXES:
            if word_lower.endswith(suffix):
                return 'NOUN'
        
        # Default to NOUN for unknown words (most common open class)
        return 'NOUN'
    
    def load_vocabulary(self, words: List[str], embeddings: Optional[Dict[str, np.ndarray]] = None):
        """Load vocabulary and classify words by POS."""
        for word in words:
            if len(word) < 2:
                continue
            
            pos = self.classify_word(word)
            word_lower = word.lower()
            
            if word_lower not in self.word_pos:
                self.word_pos[word_lower] = pos
                self.pos_words[pos].append(word_lower)
        
        # Compute average basin for each POS category
        if embeddings:
            for pos, pos_word_list in self.pos_words.items():
                pos_embeddings = []
                for w in pos_word_list:
                    if w in embeddings:
                        pos_embeddings.append(embeddings[w])
                
                if pos_embeddings:
                    avg = np.mean(pos_embeddings, axis=0)
                    norm = np.linalg.norm(avg)
                    self.pos_basins[pos] = avg / (norm + 1e-10) if norm > 0 else avg
        
        self.initialized = True
        logger.info(f"[POSGrammar] Loaded {len(self.word_pos)} words across {len(self.pos_words)} categories")
        for pos, words_list in self.pos_words.items():
            logger.info(f"  {pos}: {len(words_list)} words")
    
    def get_next_pos(self, current_pos: str, temperature: float = 0.7) -> str:
        """Sample next POS based on transition probabilities."""
        if current_pos not in POS_TRANSITIONS:
            current_pos = 'START'
        
        transitions = POS_TRANSITIONS[current_pos]
        pos_options = [t[0] for t in transitions]
        weights = np.array([t[1] for t in transitions])
        
        # Apply temperature
        if temperature != 1.0:
            weights = np.power(weights, 1.0 / temperature)
        
        weights = weights / np.sum(weights)
        
        return np.random.choice(pos_options, p=weights)
    
    def generate_skeleton(self, length: int = 6, seed_pos: Optional[str] = None) -> List[str]:
        """Generate a POS skeleton (sentence structure)."""
        skeleton = []
        current_pos = seed_pos or 'START'
        
        for _ in range(length):
            next_pos = self.get_next_pos(current_pos)
            if next_pos == 'END':
                break
            skeleton.append(next_pos)
            current_pos = next_pos
        
        return skeleton
    
    def select_skeleton_for_query(self, query_basin: np.ndarray) -> List[str]:
        """Select best skeleton based on query's geometric properties."""
        # Use query basin entropy to determine skeleton length/complexity
        p = np.abs(query_basin) + 1e-10
        p = p / np.sum(p)
        entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(len(query_basin))
        complexity = entropy / max_entropy
        
        # Higher entropy -> longer skeleton
        if complexity > 0.7:
            length = 7
        elif complexity > 0.5:
            length = 5
        else:
            length = 4
        
        return self.generate_skeleton(length)
    
    def get_words_for_pos(self, pos: str, basin: Optional[np.ndarray] = None, 
                          embeddings: Optional[Dict[str, np.ndarray]] = None,
                          top_k: int = 10) -> List[Tuple[str, float]]:
        """Get candidate words for a POS slot, ranked by basin similarity."""
        candidates = self.pos_words.get(pos, [])
        
        if not candidates:
            return []
        
        if basin is None or embeddings is None:
            # Random selection if no geometric info
            selected = np.random.choice(candidates, min(top_k, len(candidates)), replace=False)
            return [(w, 0.5) for w in selected]
        
        # Score by basin similarity
        scored = []
        for word in candidates:
            if word in embeddings:
                word_basin = embeddings[word]
                # Fisher-Rao distance (arccos of dot product for unit vectors)
                dot = np.clip(np.dot(basin, word_basin), -1.0, 1.0)
                similarity = 1.0 - np.arccos(dot) / np.pi
                scored.append((word, similarity))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    def get_pos_basin(self, pos: str) -> Optional[np.ndarray]:
        """Get the average basin coordinates for a POS category."""
        return self.pos_basins.get(pos)


# Singleton instance
_grammar = None

def get_pos_grammar() -> POSGrammar:
    """Get or create the singleton POSGrammar instance."""
    global _grammar
    if _grammar is None:
        _grammar = POSGrammar()
    return _grammar


def load_grammar_from_db():
    """Load vocabulary into grammar from database."""
    grammar = get_pos_grammar()
    
    if grammar.initialized and len(grammar.word_pos) > 100:
        return grammar
    
    try:
        import psycopg2
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            logger.warning("[POSGrammar] No DATABASE_URL, using minimal vocab")
            return grammar
        
        conn = psycopg2.connect(db_url, connect_timeout=10)
        cur = conn.cursor()
        
        # Use coordizer_vocabulary with generation role filter (consolidated vocabulary table)
        # token_role filter ensures no BPE subwords or encoding-only tokens in POS grammar
        cur.execute("""
            SELECT token as word, basin_embedding as basin_coords 
            FROM coordizer_vocabulary
            WHERE LENGTH(token) >= 3
              AND token ~ '^[a-zA-Z]+$'
              AND basin_embedding IS NOT NULL
              AND token_role IN ('generation', 'both')
              AND (phrase_category IS NULL OR phrase_category NOT IN ('PROPER_NOUN', 'BRAND'))
            ORDER BY COALESCE(phi_score, 0.5) DESC
            LIMIT 5000
        """)
        rows = cur.fetchall()
        conn.close()
        
        words = []
        embeddings = {}
        
        for token, basin_str in rows:
            words.append(token)
            if basin_str:
                try:
                    if isinstance(basin_str, str):
                        clean = basin_str.strip('[](){}')
                        coords = np.array([float(x) for x in clean.split(',')])
                        norm = np.linalg.norm(coords)
                        embeddings[token.lower()] = coords / (norm + 1e-10) if norm > 0 else coords
                except:
                    pass
        
        grammar.load_vocabulary(words, embeddings)
        return grammar
        
    except Exception as e:
        logger.error(f"[POSGrammar] Failed to load from DB: {e}")
        return grammar
