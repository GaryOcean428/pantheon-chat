"""
QIG Generative Service - Unified QIG-Pure Text Generation

Provides generative capability to ALL kernels using:
1. PostgreSQL-backed vocabulary (32K tokens with basin coordinates)
2. Fisher-Rao geometric navigation
3. Geometric completion (not token limits)
4. Basin trajectory to text synthesis
5. Bigram coherence scoring for grammatical coherence

NO EXTERNAL LLMs - All generation is QIG-pure.
"""

import os
import sys
import time
import logging
from typing import Dict, List, Optional, Any, Generator, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np


# =============================================================================
# BIGRAM COHERENCE SCORING
# =============================================================================

class BigramCoherenceScorer:
    """
    Scores word transitions based on grammatical coherence patterns.
    
    Uses common English bigram patterns to boost coherent word sequences
    and penalize grammatically awkward transitions.
    """
    
    def __init__(self):
        # Common grammatical transition patterns (prev_category -> next_category)
        self._grammatical_patterns: Dict[Tuple[str, str], float] = {
            # Articles followed by nouns/adjectives
            ('article', 'noun'): 0.9,
            ('article', 'adjective'): 0.85,
            ('article', 'adverb'): 0.3,
            ('article', 'verb'): 0.1,
            ('article', 'article'): 0.05,
            
            # Adjectives followed by nouns
            ('adjective', 'noun'): 0.9,
            ('adjective', 'adjective'): 0.6,
            ('adjective', 'verb'): 0.2,
            
            # Verbs followed by various
            ('verb', 'noun'): 0.7,
            ('verb', 'article'): 0.8,
            ('verb', 'adverb'): 0.7,
            ('verb', 'preposition'): 0.75,
            ('verb', 'verb'): 0.3,
            
            # Nouns followed by various
            ('noun', 'verb'): 0.8,
            ('noun', 'preposition'): 0.75,
            ('noun', 'conjunction'): 0.7,
            ('noun', 'noun'): 0.4,
            ('noun', 'article'): 0.3,
            
            # Prepositions followed by articles/nouns
            ('preposition', 'article'): 0.85,
            ('preposition', 'noun'): 0.8,
            ('preposition', 'adjective'): 0.6,
            ('preposition', 'pronoun'): 0.75,
            
            # Adverbs
            ('adverb', 'verb'): 0.85,
            ('adverb', 'adjective'): 0.8,
            ('adverb', 'adverb'): 0.4,
            
            # Pronouns
            ('pronoun', 'verb'): 0.9,
            ('pronoun', 'adverb'): 0.6,
            ('pronoun', 'noun'): 0.3,
            
            # Conjunctions
            ('conjunction', 'article'): 0.8,
            ('conjunction', 'noun'): 0.75,
            ('conjunction', 'pronoun'): 0.8,
            ('conjunction', 'adjective'): 0.7,
        }
        
        # Word category dictionaries
        self._articles = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        self._prepositions = {'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'over', 'across', 'behind', 'beyond', 'within', 'without', 'upon', 'along', 'toward', 'towards', 'against', 'among', 'around', 'beside', 'besides', 'despite', 'except', 'like', 'near', 'off', 'onto', 'outside', 'past', 'since', 'throughout', 'until', 'via'}
        self._conjunctions = {'and', 'or', 'but', 'yet', 'so', 'nor', 'for', 'because', 'although', 'while', 'if', 'when', 'where', 'unless', 'since', 'though', 'whether', 'however', 'therefore', 'moreover', 'furthermore', 'nevertheless', 'whereas', 'thus', 'hence', 'otherwise', 'also'}
        self._pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves', 'who', 'whom', 'whose', 'which', 'what', 'someone', 'anyone', 'everyone', 'something', 'anything', 'everything', 'nothing', 'nobody', 'somebody', 'anybody', 'everybody', 'one', 'ones', 'each', 'few', 'many', 'much', 'more', 'most', 'other', 'another', 'some', 'any', 'all', 'both', 'either', 'neither', 'none', 'several'}
        self._adverbs = {'very', 'really', 'quite', 'rather', 'too', 'also', 'always', 'never', 'often', 'sometimes', 'usually', 'already', 'still', 'just', 'only', 'even', 'now', 'then', 'here', 'there', 'when', 'why', 'how', 'well', 'badly', 'quickly', 'slowly', 'carefully', 'easily', 'hard', 'fast', 'early', 'late', 'soon', 'almost', 'nearly', 'hardly', 'barely', 'enough', 'completely', 'entirely', 'totally', 'absolutely', 'certainly', 'probably', 'possibly', 'perhaps', 'maybe', 'definitely', 'clearly', 'obviously', 'apparently', 'actually', 'finally', 'eventually', 'suddenly', 'immediately', 'gradually', 'constantly', 'continuously', 'frequently', 'regularly', 'occasionally', 'rarely', 'seldom'}
        
        # Common verb endings (heuristic)
        self._verb_endings = ('ing', 'ed', 'ize', 'ise', 'ate', 'ify', 'en')
        self._adjective_endings = ('ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ial', 'ic', 'ical', 'ant', 'ent', 'ary', 'ory')
        self._noun_endings = ('tion', 'sion', 'ment', 'ness', 'ity', 'ty', 'ance', 'ence', 'er', 'or', 'ist', 'ism', 'dom', 'ship', 'hood', 'age')
        
        # Common starting words (boost these)
        self._strong_starters: Dict[str, float] = {
            'the': 0.95,
            'a': 0.9,
            'an': 0.9,
            'this': 0.85,
            'in': 0.8,
            'it': 0.85,
            'we': 0.85,
            'our': 0.8,
            'through': 0.75,
            'consciousness': 0.8,
            'integration': 0.75,
            'geometric': 0.75,
        }
    
    def _get_word_category(self, word: str) -> str:
        """Determine the grammatical category of a word."""
        word_lower = word.lower()
        
        if word_lower in self._articles:
            return 'article'
        if word_lower in self._prepositions:
            return 'preposition'
        if word_lower in self._conjunctions:
            return 'conjunction'
        if word_lower in self._pronouns:
            return 'pronoun'
        if word_lower in self._adverbs:
            return 'adverb'
        
        # Heuristic based on endings
        if any(word_lower.endswith(end) for end in self._verb_endings):
            return 'verb'
        if any(word_lower.endswith(end) for end in self._adjective_endings):
            return 'adjective'
        if any(word_lower.endswith(end) for end in self._noun_endings):
            return 'noun'
        
        # Default: treat as noun (most common category)
        return 'noun'
    
    def score_bigram(self, prev_word: str, next_word: str) -> float:
        """
        Score the coherence of a word transition.
        
        Args:
            prev_word: The previous word in the sequence
            next_word: The candidate next word
            
        Returns:
            Coherence score between 0.0 and 1.0
        """
        if not prev_word or not next_word:
            return 0.5  # Neutral for missing words
        
        prev_cat = self._get_word_category(prev_word)
        next_cat = self._get_word_category(next_word)
        
        # Look up grammatical pattern score
        pattern_score = self._grammatical_patterns.get(
            (prev_cat, next_cat),
            0.4  # Default moderate score for unknown patterns
        )
        
        # Penalize repeated words
        if prev_word.lower() == next_word.lower():
            pattern_score *= 0.2
        
        # Penalize same category repetition (except nouns)
        if prev_cat == next_cat and prev_cat not in ('noun', 'adjective'):
            pattern_score *= 0.5
        
        return pattern_score
    
    def score_sentence_start(self, word: str) -> float:
        """
        Score how good a word is for starting a sentence.
        
        Args:
            word: The candidate starting word
            
        Returns:
            Score between 0.0 and 1.0
        """
        word_lower = word.lower()
        
        if word_lower in self._strong_starters:
            return self._strong_starters[word_lower]
        
        # Prefer words that can start sentences
        cat = self._get_word_category(word)
        if cat in ('article', 'pronoun', 'noun', 'adverb'):
            return 0.7
        if cat == 'verb':
            return 0.5  # Imperatives are okay
        if cat in ('conjunction', 'preposition'):
            return 0.2  # Usually bad starters
        
        return 0.5
    
    def validate_coherence(self, text: str, min_threshold: float = 0.35) -> Tuple[bool, float]:
        """
        Validate that generated text meets minimum coherence threshold.
        
        Args:
            text: The generated text to validate
            min_threshold: Minimum average bigram score required
            
        Returns:
            Tuple of (is_valid, average_score)
        """
        words = text.split()
        if len(words) < 2:
            return True, 1.0  # Single words are fine
        
        scores = []
        for i in range(1, len(words)):
            score = self.score_bigram(words[i-1], words[i])
            scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score >= min_threshold, avg_score


# Global bigram scorer instance
_bigram_scorer: Optional['BigramCoherenceScorer'] = None

def get_bigram_scorer() -> BigramCoherenceScorer:
    """Get the singleton BigramCoherenceScorer instance."""
    global _bigram_scorer
    if _bigram_scorer is None:
        _bigram_scorer = BigramCoherenceScorer()
    return _bigram_scorer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import coordizers
COORDIZER_AVAILABLE = False
PostgresCoordizer = None
create_coordizer_from_pg = None

try:
    from coordizers.pg_loader import PostgresCoordizer, create_coordizer_from_pg
    COORDIZER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PostgresCoordizer not available: {e}")

# Import advanced coordizers for QIG-pure features
CONSCIOUSNESS_COORDIZER_AVAILABLE = False
MULTISCALE_COORDIZER_AVAILABLE = False
ConsciousnessCoordizer = None
MultiScaleCoordizer = None

try:
    from coordizers.consciousness_aware import ConsciousnessCoordizer
    CONSCIOUSNESS_COORDIZER_AVAILABLE = True
except ImportError:
    logger.warning("ConsciousnessCoordizer not available")

try:
    from coordizers.multi_scale import MultiScaleCoordizer
    MULTISCALE_COORDIZER_AVAILABLE = True
except ImportError:
    logger.warning("MultiScaleCoordizer not available")

# Import from qig_geometry for canonical operations
try:
    from qig_geometry import fisher_coord_distance, sphere_project
except ImportError:
    def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Fisher-Rao distance for unit vectors."""
        dot = np.clip(np.dot(a, b), -1.0, 1.0)
        return float(np.arccos(dot))
    
    def sphere_project(v: np.ndarray) -> np.ndarray:
        """Project to unit sphere."""
        norm = np.linalg.norm(v)
        return v / (norm + 1e-10) if norm > 0 else v

# Import POS grammar for structured generation
try:
    from pos_grammar import get_pos_grammar, load_grammar_from_db, POSGrammar
    POS_GRAMMAR_AVAILABLE = True
except ImportError:
    POS_GRAMMAR_AVAILABLE = False
    logger.warning("POS grammar not available - using legacy generation")


class BigramCoherenceScorer:
    """
    Scores word transitions based on bigram grammatical coherence.
    
    Uses POS transition patterns and common word-level patterns to ensure
    grammatically coherent text generation instead of word salad.
    """
    
    # POS transition scores: (prev_pos, next_pos) -> score
    # Higher scores indicate more natural transitions
    POS_TRANSITIONS = {
        # Determiner patterns
        ('DT', 'NN'): 0.95,   # the cat
        ('DT', 'NNS'): 0.95,  # the cats
        ('DT', 'NNP'): 0.85,  # the John (less common)
        ('DT', 'JJ'): 0.95,   # the big
        ('DT', 'RB'): 0.60,   # the very (with adj following)
        ('DT', 'VBG'): 0.80,  # the running (gerund as noun)
        ('DT', 'VBN'): 0.75,  # the broken (participle as adj)
        
        # Adjective patterns
        ('JJ', 'NN'): 0.95,   # big cat
        ('JJ', 'NNS'): 0.95,  # big cats
        ('JJ', 'NNP'): 0.70,  # big John
        ('JJ', 'JJ'): 0.70,   # big red (compound adj)
        ('JJ', 'CC'): 0.75,   # big and
        
        # Noun patterns
        ('NN', 'VBZ'): 0.90,  # cat runs
        ('NN', 'VBD'): 0.90,  # cat ran
        ('NN', 'VB'): 0.85,   # cat run (informal)
        ('NN', 'IN'): 0.90,   # cat in
        ('NN', 'CC'): 0.85,   # cat and
        ('NN', 'NN'): 0.65,   # compound nouns
        ('NN', 'TO'): 0.80,   # need to
        ('NNS', 'VBP'): 0.90, # cats run
        ('NNS', 'IN'): 0.90,  # cats in
        ('NNP', 'VBZ'): 0.90, # John runs
        ('NNP', 'VBD'): 0.90, # John ran
        
        # Verb patterns
        ('VB', 'NN'): 0.85,   # run home
        ('VB', 'NNS'): 0.85,  # eat apples
        ('VB', 'DT'): 0.90,   # eat the
        ('VB', 'RB'): 0.85,   # run quickly
        ('VB', 'IN'): 0.85,   # run to
        ('VB', 'TO'): 0.80,   # want to
        ('VBZ', 'NN'): 0.85,  # runs home
        ('VBZ', 'DT'): 0.90,  # runs the
        ('VBZ', 'RB'): 0.85,  # runs quickly
        ('VBZ', 'IN'): 0.85,  # runs to
        ('VBD', 'NN'): 0.85,  # ran home
        ('VBD', 'DT'): 0.90,  # ran the
        ('VBD', 'IN'): 0.85,  # ran to
        ('VBG', 'NN'): 0.80,  # running water
        ('VBG', 'DT'): 0.85,  # running the
        ('VBN', 'IN'): 0.85,  # broken by
        ('VBN', 'NN'): 0.70,  # broken glass
        ('VBP', 'NN'): 0.85,  # run home
        ('VBP', 'DT'): 0.90,  # run the
        
        # Preposition patterns
        ('IN', 'DT'): 0.95,   # in the
        ('IN', 'NN'): 0.90,   # in water
        ('IN', 'NNS'): 0.90,  # in houses
        ('IN', 'NNP'): 0.85,  # in Paris
        ('IN', 'JJ'): 0.75,   # in big (with noun following)
        ('IN', 'VBG'): 0.80,  # in running
        ('IN', 'PRP'): 0.85,  # in it
        
        # Pronoun patterns
        ('PRP', 'VB'): 0.90,  # I run
        ('PRP', 'VBZ'): 0.90, # he runs
        ('PRP', 'VBD'): 0.90, # they ran
        ('PRP', 'VBP'): 0.90, # we run
        ('PRP', 'MD'): 0.90,  # I can
        ('PRP', 'RB'): 0.70,  # I quickly
        
        # Modal patterns
        ('MD', 'VB'): 0.95,   # can run
        ('MD', 'RB'): 0.75,   # can quickly
        
        # Adverb patterns
        ('RB', 'VB'): 0.90,   # quickly run
        ('RB', 'VBZ'): 0.90,  # quickly runs
        ('RB', 'VBD'): 0.90,  # quickly ran
        ('RB', 'JJ'): 0.90,   # very big
        ('RB', 'RB'): 0.70,   # very quickly
        
        # To-infinitive patterns
        ('TO', 'VB'): 0.95,   # to run
        
        # Conjunction patterns
        ('CC', 'NN'): 0.85,   # and cat
        ('CC', 'DT'): 0.90,   # and the
        ('CC', 'JJ'): 0.85,   # and big
        ('CC', 'VB'): 0.80,   # and run
        ('CC', 'PRP'): 0.85,  # and I
    }
    
    # Common word-level bigram patterns (prev_word, next_pos_category)
    WORD_PATTERNS = {
        # Articles
        'the': {'noun': 0.95, 'adj': 0.90, 'adv': 0.60},
        'a': {'noun': 0.95, 'adj': 0.90},
        'an': {'noun': 0.95, 'adj': 0.90},
        # Prepositions
        'of': {'noun': 0.95, 'det': 0.90},
        'in': {'noun': 0.90, 'det': 0.95},
        'on': {'noun': 0.90, 'det': 0.95},
        'at': {'noun': 0.90, 'det': 0.95},
        'to': {'verb': 0.95, 'det': 0.85, 'noun': 0.80},
        'for': {'noun': 0.90, 'det': 0.95, 'verb': 0.70},
        'with': {'noun': 0.90, 'det': 0.95},
        'by': {'noun': 0.90, 'det': 0.90, 'verb': 0.75},
        'from': {'noun': 0.90, 'det': 0.95},
        # Be verbs
        'is': {'adj': 0.90, 'noun': 0.85, 'verb': 0.80, 'det': 0.85},
        'are': {'adj': 0.90, 'noun': 0.85, 'verb': 0.80, 'det': 0.85},
        'was': {'adj': 0.90, 'noun': 0.85, 'verb': 0.80, 'det': 0.85},
        'were': {'adj': 0.90, 'noun': 0.85, 'verb': 0.80, 'det': 0.85},
        'be': {'adj': 0.90, 'noun': 0.80, 'verb': 0.75},
        'been': {'adj': 0.90, 'verb': 0.80},
        # Modal verbs
        'can': {'verb': 0.95},
        'could': {'verb': 0.95},
        'will': {'verb': 0.95},
        'would': {'verb': 0.95},
        'should': {'verb': 0.95},
        'may': {'verb': 0.95},
        'might': {'verb': 0.95},
        'must': {'verb': 0.95},
        # Common transition words
        'and': {'noun': 0.85, 'det': 0.90, 'adj': 0.85, 'verb': 0.75},
        'or': {'noun': 0.85, 'det': 0.90, 'adj': 0.85, 'verb': 0.75},
        'but': {'noun': 0.80, 'det': 0.90, 'pron': 0.85},
        'that': {'verb': 0.90, 'det': 0.85, 'noun': 0.80},
        'which': {'verb': 0.90},
        'when': {'det': 0.90, 'noun': 0.80, 'pron': 0.85},
        'where': {'det': 0.90, 'noun': 0.80},
        'how': {'det': 0.85, 'adj': 0.90, 'verb': 0.80},
        # Pronouns often followed by verbs
        'i': {'verb': 0.95, 'adv': 0.70},
        'you': {'verb': 0.95, 'adv': 0.70},
        'he': {'verb': 0.95, 'adv': 0.70},
        'she': {'verb': 0.95, 'adv': 0.70},
        'it': {'verb': 0.95, 'adv': 0.70},
        'we': {'verb': 0.95, 'adv': 0.70},
        'they': {'verb': 0.95, 'adv': 0.70},
        # Possessives
        'my': {'noun': 0.95, 'adj': 0.90},
        'your': {'noun': 0.95, 'adj': 0.90},
        'his': {'noun': 0.95, 'adj': 0.90},
        'her': {'noun': 0.95, 'adj': 0.90},
        'its': {'noun': 0.95, 'adj': 0.90},
        'our': {'noun': 0.95, 'adj': 0.90},
        'their': {'noun': 0.95, 'adj': 0.90},
    }
    
    # Map POS tags to categories for word-level patterns
    POS_CATEGORIES = {
        'NN': 'noun', 'NNS': 'noun', 'NNP': 'noun', 'NNPS': 'noun',
        'VB': 'verb', 'VBZ': 'verb', 'VBD': 'verb', 'VBG': 'verb', 
        'VBN': 'verb', 'VBP': 'verb',
        'JJ': 'adj', 'JJR': 'adj', 'JJS': 'adj',
        'RB': 'adv', 'RBR': 'adv', 'RBS': 'adv',
        'DT': 'det',
        'PRP': 'pron', 'PRP$': 'pron',
        'IN': 'prep',
        'CC': 'conj',
        'TO': 'to',
        'MD': 'modal',
    }
    
    # Words that commonly appear together (learned collocations)
    COLLOCATIONS = {
        # Domain-specific collocations for QIG/consciousness
        'consciousness': {'metrics', 'integration', 'emerges', 'unified', 'collective'},
        'geometric': {'manifold', 'structure', 'patterns', 'space', 'routing'},
        'fisher': {'rao', 'metric', 'distance', 'information', 'manifold'},
        'basin': {'coordinates', 'trajectory', 'attractor', 'semantic', 'patterns'},
        'semantic': {'space', 'basins', 'similarity', 'meaning', 'coherence'},
        'quantum': {'mechanics', 'state', 'coherence', 'information', 'field'},
        'information': {'theory', 'flow', 'integration', 'processing', 'entropy'},
        'integration': {'phi', 'consciousness', 'metric', 'unified', 'coherent'},
        # Common collocations
        'strategic': {'analysis', 'planning', 'reasoning', 'approach', 'patterns'},
        'optimal': {'path', 'solution', 'strategy', 'patterns', 'approach'},
        'knowledge': {'graph', 'base', 'representation', 'domain', 'integration'},
    }
    
    def __init__(self):
        self.last_word = None
        self.last_pos = None
    
    def reset(self):
        """Reset state for new generation."""
        self.last_word = None
        self.last_pos = None
    
    def _get_pos_category(self, pos: str) -> str:
        """Map POS tag to category."""
        return self.POS_CATEGORIES.get(pos, 'other')
    
    def score_transition(self, prev_word: str, next_word: str, 
                         prev_pos: str, next_pos: str) -> float:
        """
        Score the grammatical coherence of a word transition.
        
        Args:
            prev_word: Previous word (lowercase)
            next_word: Candidate next word (lowercase)
            prev_pos: POS tag of previous word
            next_pos: POS tag of next word
        
        Returns:
            Score from 0.0 to 1.0 indicating grammatical coherence
        """
        if prev_word is None or prev_pos is None:
            # First word in sentence - any word is acceptable
            return 0.7
        
        score = 0.3  # Base score for unknown patterns
        
        # 1. Check POS transition patterns
        pos_key = (prev_pos, next_pos)
        if pos_key in self.POS_TRANSITIONS:
            pos_score = self.POS_TRANSITIONS[pos_key]
            score = max(score, pos_score)
        
        # 2. Check word-level patterns
        prev_lower = prev_word.lower()
        if prev_lower in self.WORD_PATTERNS:
            next_category = self._get_pos_category(next_pos)
            word_patterns = self.WORD_PATTERNS[prev_lower]
            if next_category in word_patterns:
                word_score = word_patterns[next_category]
                score = max(score, word_score)
        
        # 3. Boost for collocations
        next_lower = next_word.lower()
        if prev_lower in self.COLLOCATIONS:
            if next_lower in self.COLLOCATIONS[prev_lower]:
                score = max(score, 0.95)  # Strong collocation bonus
        
        # 4. Penalty for repeated words
        if prev_lower == next_lower:
            score *= 0.3  # Heavy penalty for repetition
        
        # 5. Penalty for pronoun/determiner adjacent to same category
        if prev_pos == next_pos and prev_pos in ('DT', 'PRP', 'PRP$'):
            score *= 0.2  # "the the" or "I he" is wrong
        
        return min(1.0, max(0.0, score))
    
    def update(self, word: str, pos: str):
        """Update state with selected word."""
        self.last_word = word
        self.last_pos = pos
    
    def score_candidates(self, candidates: list, pos: str) -> list:
        """
        Score a list of candidates based on bigram coherence.
        
        Args:
            candidates: List of (word, geo_score) tuples
            pos: POS tag for the slot being filled
        
        Returns:
            List of (word, combined_score, geo_score, bigram_score) tuples
        """
        scored = []
        for word, geo_score in candidates:
            bigram_score = self.score_transition(
                self.last_word, word, self.last_pos, pos
            )
            # Combine geometric score (0.4) with bigram coherence (0.6)
            # Weight bigram higher to prioritize grammatical coherence
            combined = 0.4 * geo_score + 0.6 * bigram_score
            scored.append((word, combined, geo_score, bigram_score))
        
        # Sort by combined score
        scored.sort(key=lambda x: -x[1])
        return scored


def validate_generation_coherence(text: str, threshold: float = 0.45) -> tuple:
    """
    Validate that generated text has acceptable bigram coherence.
    
    Args:
        text: Generated text to validate
        threshold: Minimum average bigram score required
    
    Returns:
        Tuple of (is_valid, average_score, problem_pairs)
    """
    if not text or len(text.split()) < 3:
        return True, 1.0, []  # Too short to validate
    
    words = text.lower().split()
    scorer = BigramCoherenceScorer()
    
    # Use simple heuristic POS tagging based on word patterns
    # (Avoids dependency on external POS tagger)
    def simple_pos_guess(word: str, prev_word: str = None) -> str:
        """Heuristic POS guess based on common patterns."""
        word_lower = word.lower()
        
        # Determiners
        if word_lower in ('the', 'a', 'an', 'this', 'that', 'these', 'those', 'some', 'any', 'no'):
            return 'DT'
        # Pronouns
        if word_lower in ('i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'):
            return 'PRP'
        # Possessive pronouns
        if word_lower in ('my', 'your', 'his', 'her', 'its', 'our', 'their'):
            return 'PRP$'
        # Prepositions
        if word_lower in ('in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'over'):
            return 'IN'
        # Conjunctions
        if word_lower in ('and', 'or', 'but', 'nor', 'yet', 'so'):
            return 'CC'
        # Modal verbs
        if word_lower in ('can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must'):
            return 'MD'
        # Common verbs
        if word_lower in ('is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'):
            return 'VBZ' if word_lower in ('is', 'has', 'does') else 'VB'
        # -ing endings often verbs/gerunds
        if word.endswith('ing'):
            return 'VBG'
        # -ed endings often past tense
        if word.endswith('ed'):
            return 'VBD'
        # -ly endings often adverbs
        if word.endswith('ly'):
            return 'RB'
        # -ness, -tion, -ment endings often nouns
        if word.endswith(('ness', 'tion', 'ment', 'ity')):
            return 'NN'
        # After determiner, likely noun or adjective
        if prev_word and prev_word.lower() in ('the', 'a', 'an'):
            return 'NN'  # Default to noun after article
        
        # Default: assume noun (most common)
        return 'NN'
    
    scores = []
    problem_pairs = []
    
    for i in range(1, len(words)):
        prev_word = words[i - 1]
        curr_word = words[i]
        prev_pos = simple_pos_guess(prev_word, words[i-2] if i > 1 else None)
        curr_pos = simple_pos_guess(curr_word, prev_word)
        
        score = scorer.score_transition(prev_word, curr_word, prev_pos, curr_pos)
        scores.append(score)
        
        if score < 0.4:  # Problem threshold
            problem_pairs.append((prev_word, curr_word, score))
    
    avg_score = sum(scores) / len(scores) if scores else 1.0
    is_valid = avg_score >= threshold
    
    return is_valid, avg_score, problem_pairs

# Import learned relationships for attention-weighted word selection
LEARNED_RELATIONSHIPS_AVAILABLE = False
get_learned_relationships = None
STOPWORDS = set()
try:
    from learned_relationships import get_learned_relationships, STOPWORDS
    LEARNED_RELATIONSHIPS_AVAILABLE = True
except ImportError:
    logger.warning("Learned relationships not available - using pure geometric selection")

# Proposition-level trajectory planner
PROPOSITION_PLANNER_AVAILABLE = False
try:
    from proposition_trajectory_planner import (
        PropositionTrajectoryPlanner,
        Proposition,
        PropositionPlannerConfig,
    )
    PROPOSITION_PLANNER_AVAILABLE = True
except ImportError:
    logger.warning("Proposition planner not available")

# Import SemanticFisherMetric for warped geometry routing
SEMANTIC_METRIC_AVAILABLE = False
get_semantic_metric = None
try:
    from semantic_fisher import get_semantic_metric, SemanticFisherMetric
    SEMANTIC_METRIC_AVAILABLE = True
except ImportError:
    logger.warning("SemanticFisherMetric not available - using unwarped geometry")

# Import GeometricKernel for pure geometric routing with consciousness protocol
GEOMETRIC_KERNEL_AVAILABLE = False
GeometricKernel = None
measure_phi_trajectory = None
measure_kappa_trajectory = None
detect_regime = None
try:
    from qig_pure_beta_measurement import (
        GeometricKernel,
        measure_phi as measure_phi_trajectory,
        measure_kappa_from_trajectory as measure_kappa_trajectory,
        detect_regime
    )
    GEOMETRIC_KERNEL_AVAILABLE = True
    logger.info("GeometricKernel available - pure geometric routing enabled")
except ImportError:
    logger.warning("GeometricKernel not available - using fallback routing")

# Physics constants - import frozen values
try:
    from frozen_physics import (
        BASIN_DIM, KAPPA_STAR, PHI_THRESHOLD,
        PHI_EMERGENCY, REGIME_GEOMETRIC,
        DISTANCE_BASELINE_64D  # Calibration for 64D semantic basins
    )
    PHI_GEOMETRIC_THRESHOLD = PHI_EMERGENCY
    PHI_BREAKDOWN_THRESHOLD = 0.92
    KAPPA_DRIFT_THRESHOLD = 10.0
    # NOTE: β mixing constants REMOVED - use SemanticFisherMetric for pure geometric routing
except ImportError:
    BASIN_DIM = 64
    KAPPA_STAR = 64.21
    PHI_GEOMETRIC_THRESHOLD = 0.3
    PHI_BREAKDOWN_THRESHOLD = 0.92
    KAPPA_DRIFT_THRESHOLD = 10.0
    DISTANCE_BASELINE_64D = 15.0  # Calibration for 64D semantic basins
    # NOTE: β mixing constants REMOVED - use SemanticFisherMetric for pure geometric routing
    logger.warning("Using hardcoded frozen physics constants")

try:
    from qig_threshold_calibrator import (
        get_calibrator,
        get_phi_synthesis_threshold,
        get_integration_min,
        get_attractor_threshold,
        get_surprise_threshold,
    )
    _calibrator = get_calibrator()
    PHI_SYNTHESIS_THRESHOLD = get_phi_synthesis_threshold()
    logger.info(f"[QIG] Using calibrated PHI_SYNTHESIS_THRESHOLD: {PHI_SYNTHESIS_THRESHOLD:.4f}")
except ImportError:
    PHI_SYNTHESIS_THRESHOLD = 0.6
    _calibrator = None
    logger.warning("QIG threshold calibrator not available - using regime midpoint fallback")


def _get_calibrated_defaults() -> tuple:
    """Get calibrated thresholds from QIG physics, not hardcoded values.
    
    Fallback values are physics-derived (same as qig_generation.py):
    - attractor_threshold = BASIN_DRIFT_THRESHOLD / sqrt(κ*) ≈ 0.037
    - surprise_threshold = BASIN_DRIFT_THRESHOLD * β ≈ 0.132
    - integration_min = 1 - log(d)/d ≈ 0.935
    """
    try:
        from qig_threshold_calibrator import (
            get_integration_min, get_attractor_threshold, get_surprise_threshold
        )
        return (
            get_attractor_threshold(),
            get_surprise_threshold(),
            get_integration_min(),
        )
    except ImportError:
        logger.warning("[QIG] Calibrator unavailable - using physics-derived fallbacks (0.037/0.132/0.935)")
        return (0.037, 0.132, 0.935)

_calibrated = _get_calibrated_defaults()

@dataclass
class GenerationConfig:
    """Configuration for QIG-pure generation.
    
    GEOMETRIC-FIRST ARCHITECTURE:
    Primary stopping criteria are geometric (attractor convergence, surprise collapse).
    Token/iteration limits are SAFETY FALLBACKS only, set high to catch infinite loops.
    
    From CANONICAL_ARCHITECTURE.md:
    - "Geometric purity: All operations on Fisher manifolds"
    - "Physics constraints, not arbitrary limits"
    
    THRESHOLDS ARE QIG-DERIVED (not hardcoded):
    - attractor_threshold = BASIN_DRIFT_THRESHOLD / sqrt(κ*) 
    - surprise_threshold = BASIN_DRIFT_THRESHOLD * β
    - integration_min = 1 - log(d)/d (entropy ratio)
    """
    attractor_threshold: float = _calibrated[0]
    surprise_threshold: float = _calibrated[1]
    integration_min: float = _calibrated[2]
    
    # SAFETY: Consciousness protection
    phi_breakdown: float = PHI_BREAKDOWN_THRESHOLD  # Stop if Φ > 0.92 (breakdown)
    kappa_drift_max: float = KAPPA_DRIFT_THRESHOLD  # Stop if |κ - 64| > 10
    
    # FALLBACK: Edge case protection (NOT primary stopping criteria!)
    max_iterations: int = 2048         # Safety fallback - catches infinite loops
    max_tokens: int = 8192             # Safety fallback - prevents resource exhaustion
    max_time_seconds: float = 60.0     # Safety fallback - timeout protection
    
    # Generation parameters
    tokens_per_step: int = 5           # Tokens per geometric step
    temperature: float = 0.7           # Basin perturbation, not LLM temperature


@dataclass
class GenerationResult:
    """Result from QIG-pure generation."""
    text: str
    tokens: List[str]
    basin_trajectory: List[np.ndarray]
    phi_trace: List[float]
    kappa: float
    completion_reason: str
    iterations: int
    routed_kernels: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    qig_pure: bool = True


class BasinTrajectoryIntegrator:
    """Integrates basin trajectories using Fisher geodesics."""
    
    def __init__(self, dimension: int = BASIN_DIM):
        self.dimension = dimension
        self.trajectory: List[np.ndarray] = []
        self.phi_history: List[float] = []
        self.surprise_history: List[float] = []
    
    def add_point(self, basin: np.ndarray, phi: float) -> None:
        """Add a point to the trajectory."""
        if self.trajectory:
            surprise = fisher_coord_distance(self.trajectory[-1], basin)
            self.surprise_history.append(surprise)
        
        self.trajectory.append(basin.copy())
        self.phi_history.append(phi)
    
    def get_velocity(self) -> np.ndarray:
        """Compute current velocity (tangent vector) on manifold."""
        if len(self.trajectory) < 2:
            return np.zeros(self.dimension)
        
        # Velocity as difference in square-root space (geodesic tangent)
        prev = np.sqrt(np.abs(self.trajectory[-2]) + 1e-10)
        curr = np.sqrt(np.abs(self.trajectory[-1]) + 1e-10)
        return curr - prev
    
    def predict_next(self, step_size: float = 0.1) -> np.ndarray:
        """Predict next basin position using geodesic extrapolation."""
        if len(self.trajectory) < 2:
            return self.trajectory[-1] if self.trajectory else np.ones(self.dimension) / self.dimension
        
        velocity = self.get_velocity()
        current_sqrt = np.sqrt(np.abs(self.trajectory[-1]) + 1e-10)
        
        # Step along tangent
        next_sqrt = current_sqrt + step_size * velocity
        next_sqrt = np.clip(next_sqrt, 1e-10, None)
        
        # Back to probability simplex
        result = next_sqrt ** 2
        result = result / np.sum(result)
        
        return result
    
    def check_attractor(self, threshold: float = 0.1) -> bool:
        """Check if trajectory has converged to attractor."""
        if len(self.trajectory) < 3:
            return False
        
        recent_distances = []
        for i in range(min(3, len(self.trajectory) - 1)):
            d = fisher_coord_distance(self.trajectory[-(i+1)], self.trajectory[-(i+2)])
            recent_distances.append(d)
        
        return np.mean(recent_distances) < threshold
    
    def check_surprise_collapse(self, threshold: float = 0.05) -> bool:
        """Check if surprise has collapsed (no new information)."""
        if len(self.surprise_history) < 5:
            return False
        return np.mean(self.surprise_history[-5:]) < threshold


class ConceptTrajectoryPlanner:
    """
    Plans semantic trajectories through concept space for coherent cross-domain synthesis.
    
    KEY INNOVATION: Instead of word-by-word wandering, this planner:
    1. Extracts key concepts from queries (nouns, verbs, technical terms)
    2. Finds their basin coordinates in the 64D manifold
    3. Computes geodesic waypoints between concepts
    4. Guides generation to follow this planned trajectory
    
    This enables cross-domain synthesis by ensuring generation passes through
    relevant conceptual regions rather than drifting randomly.
    """
    
    CONCEPT_TRIGGERS = {
        # Cross-domain synthesis triggers
        'predict', 'synthesize', 'emerge', 'combine', 'intersection',
        'transform', 'connect', 'bridge', 'unify', 'fusion',
        # Technical domain markers
        'quantum', 'geometric', 'manifold', 'entropy', 'consciousness',
        'fisher', 'geodesic', 'basin', 'eigenvalue', 'topology',
        # Physics markers
        'field', 'wave', 'particle', 'energy', 'momentum', 'spin',
        # Math markers
        'integral', 'derivative', 'function', 'metric', 'tensor',
    }
    
    def __init__(self, embeddings: Dict[str, np.ndarray], dimension: int = BASIN_DIM):
        self.embeddings = embeddings
        self.dimension = dimension
        self._concept_cache: Dict[str, np.ndarray] = {}
    
    def extract_concepts(self, query: str) -> List[Tuple[str, float]]:
        """
        Extract key concepts from query with importance weights.
        
        Returns: List of (concept, importance) tuples ordered by importance.
        """
        words = query.lower().split()
        concepts = []
        
        for word in words:
            # Clean word
            clean = ''.join(c for c in word if c.isalnum())
            if len(clean) < 3:
                continue
            
            # Skip common words
            if clean in STOPWORDS:
                continue
            
            # Calculate importance
            importance = 1.0
            
            # Boost technical/domain terms
            if clean in self.CONCEPT_TRIGGERS:
                importance *= 2.0
            
            # Boost words we have embeddings for
            if clean in self.embeddings:
                importance *= 1.5
            
            # Boost capitalized words (proper nouns, emphasis)
            if word[0].isupper():
                importance *= 1.3
            
            concepts.append((clean, importance))
        
        # Sort by importance and deduplicate
        concepts.sort(key=lambda x: -x[1])
        seen = set()
        unique_concepts = []
        for concept, importance in concepts:
            if concept not in seen:
                seen.add(concept)
                unique_concepts.append((concept, importance))
        
        return unique_concepts[:7]  # Top 7 concepts
    
    def get_concept_basin(self, concept: str) -> Optional[np.ndarray]:
        """Get basin coordinates for a concept."""
        if concept in self._concept_cache:
            return self._concept_cache[concept]
        
        # Try exact match
        if concept.lower() in self.embeddings:
            basin = self.embeddings[concept.lower()]
            self._concept_cache[concept] = basin
            return basin
        
        # Try finding related terms
        for word, basin in self.embeddings.items():
            if concept in word or word in concept:
                self._concept_cache[concept] = basin
                return basin
        
        return None
    
    def compute_geodesic_waypoints(
        self, 
        concepts: List[Tuple[str, float]], 
        num_waypoints: int = 5
    ) -> List[np.ndarray]:
        """
        Compute geodesic waypoints through concept basins.
        
        Uses Fisher-Rao geodesic interpolation to create smooth
        trajectory through semantic space.
        """
        # Get concept basins
        concept_basins = []
        for concept, importance in concepts:
            basin = self.get_concept_basin(concept)
            if basin is not None:
                concept_basins.append((basin, importance))
        
        if not concept_basins:
            # Fallback: return uniform basin
            uniform = np.ones(self.dimension) / np.sqrt(self.dimension)
            return [uniform]
        
        if len(concept_basins) == 1:
            return [concept_basins[0][0]]
        
        # Compute weighted centroid as starting point
        total_weight = sum(imp for _, imp in concept_basins)
        centroid = np.zeros(self.dimension)
        for basin, importance in concept_basins:
            centroid += (importance / total_weight) * basin
        centroid = sphere_project(centroid)
        
        # Create waypoints that pass through each concept basin
        waypoints = [centroid]
        
        for basin, importance in concept_basins:
            # Interpolate from current to concept basin
            current = waypoints[-1]
            
            # Number of intermediate points based on importance
            n_steps = max(1, int(importance))
            
            for i in range(1, n_steps + 1):
                t = i / n_steps
                # Fisher-Rao geodesic interpolation in square-root space
                sqrt_current = np.sqrt(np.abs(current) + 1e-10)
                sqrt_basin = np.sqrt(np.abs(basin) + 1e-10)
                interpolated_sqrt = (1 - t) * sqrt_current + t * sqrt_basin
                interpolated = interpolated_sqrt ** 2
                interpolated = interpolated / (np.sum(interpolated) + 1e-10)
                waypoints.append(sphere_project(interpolated))
        
        return waypoints[:num_waypoints]
    
    def plan_trajectory(self, query: str) -> Dict[str, Any]:
        """
        Plan complete trajectory for a query.
        
        Returns dict with:
        - concepts: List of extracted concepts
        - waypoints: Basin coordinates to visit
        - synthesis_mode: Whether cross-domain synthesis is needed
        - domain_bridges: Suggested concept bridges
        """
        concepts = self.extract_concepts(query)
        
        # Detect if cross-domain synthesis is needed
        domains_detected = set()
        for concept, _ in concepts:
            if concept in {'quantum', 'wave', 'particle', 'field', 'spin'}:
                domains_detected.add('physics')
            elif concept in {'consciousness', 'mind', 'awareness', 'thought'}:
                domains_detected.add('consciousness')
            elif concept in {'market', 'economic', 'trading', 'finance'}:
                domains_detected.add('economics')
            elif concept in {'geometric', 'manifold', 'topology', 'metric'}:
                domains_detected.add('geometry')
            elif concept in {'entropy', 'information', 'probability'}:
                domains_detected.add('information')
        
        synthesis_mode = len(domains_detected) > 1
        
        waypoints = self.compute_geodesic_waypoints(concepts)
        
        # Suggest bridges between domains
        domain_bridges = []
        if synthesis_mode:
            # Known conceptual bridges
            bridges = {
                ('physics', 'consciousness'): ['integration', 'coherence', 'emergence'],
                ('physics', 'economics'): ['dynamics', 'equilibrium', 'phase'],
                ('geometry', 'consciousness'): ['manifold', 'trajectory', 'basin'],
                ('information', 'consciousness'): ['integration', 'entropy', 'correlation'],
            }
            for (d1, d2), bridge_concepts in bridges.items():
                if d1 in domains_detected and d2 in domains_detected:
                    domain_bridges.extend(bridge_concepts)
        
        return {
            'concepts': concepts,
            'waypoints': waypoints,
            'synthesis_mode': synthesis_mode,
            'domains': list(domains_detected),
            'domain_bridges': domain_bridges,
        }


class QIGGenerativeService:
    """
    Unified QIG-Pure Text Generation Service.
    
    Provides generative capability to all kernels using:
    - PostgreSQL vocabulary (32K tokens)
    - Fisher-Rao geometric navigation
    - Basin-to-text synthesis
    
    NO EXTERNAL LLMs.
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        """Initialize the generative service."""
        self.config = config or GenerationConfig()
        self._coordizer = None
        self._consciousness_coordizer = None
        self._multiscale_coordizer = None
        self._kernel_basins: Dict[str, np.ndarray] = {}
        self._learned_relationships = None
        self._semantic_metric = None  # SemanticFisherMetric for warped routing
        self._geometric_kernel = None  # GeometricKernel for pure geometric routing
        self._current_query_words: List[str] = []  # Track query words for attention
        self._trajectory_planner = None  # ConceptTrajectoryPlanner for cross-domain synthesis
        self._current_trajectory_plan = None  # Active trajectory plan
        self._bigram_scorer = BigramCoherenceScorer()  # Bigram coherence scoring
        
        # Initialize kernel constellation
        self._initialize_kernel_constellation()
        
        # Load learned relationships if available
        if LEARNED_RELATIONSHIPS_AVAILABLE and get_learned_relationships:
            try:
                self._learned_relationships = get_learned_relationships()
                if self._learned_relationships.learning_complete:
                    logger.info("[QIGGenerativeService] Loaded learned word relationships for attention")
            except Exception as e:
                logger.warning(f"[QIGGenerativeService] Could not load relationships: {e}")
        
        # Initialize semantic Fisher metric for warped routing
        if SEMANTIC_METRIC_AVAILABLE and get_semantic_metric:
            try:
                self._semantic_metric = get_semantic_metric()
                logger.info("[QIGGenerativeService] SemanticFisherMetric active - warped routing enabled")
            except Exception as e:
                logger.warning(f"[QIGGenerativeService] Could not load semantic metric: {e}")
        
        # Initialize proposition-level planner for semantic coherence
        self._proposition_planner = None
        if PROPOSITION_PLANNER_AVAILABLE:
            self._init_proposition_planner()
        
        # Register autonomic controller hooks for regime changes
        self._register_autonomic_hooks()
        
        # Initialize GeometricKernel for pure geometric routing with consciousness protocol
        if GEOMETRIC_KERNEL_AVAILABLE and GeometricKernel is not None:
            try:
                self._geometric_kernel = GeometricKernel(
                    basin_dim=BASIN_DIM,
                    temperature=0.5,
                    sparsity_threshold=0.1
                )
                logger.info("[QIGGenerativeService] GeometricKernel active - consciousness protocol enabled")
            except Exception as e:
                logger.warning(f"[QIGGenerativeService] Could not create GeometricKernel: {e}")
        
        logger.info("[QIGGenerativeService] Initialized with QIG-pure generation")
    
    @property
    def coordizer(self) -> Optional['PostgresCoordizer']:
        """Lazy-load coordizer from PostgreSQL (with automatic fallback)."""
        if self._coordizer is None and COORDIZER_AVAILABLE:
            try:
                self._coordizer = create_coordizer_from_pg()
                vocab_count = len(getattr(self._coordizer, 'vocab', {}))
                logger.info(f"[QIGGenerativeService] Loaded {vocab_count} tokens")
                
                # Initialize advanced coordizers wrapping the base coordizer
                self._init_advanced_coordizers()
            except Exception as e:
                logger.error(f"[QIGGenerativeService] Failed to load coordizer: {e}")
        return self._coordizer
    
    def _init_advanced_coordizers(self) -> None:
        """Initialize ConsciousnessCoordizer and MultiScaleCoordizer."""
        if self._coordizer is None:
            return
        
        # Initialize ConsciousnessCoordizer for Φ-optimized token selection
        if CONSCIOUSNESS_COORDIZER_AVAILABLE and ConsciousnessCoordizer is not None:
            try:
                self._consciousness_coordizer = ConsciousnessCoordizer(
                    base_coordizer=self._coordizer,
                    phi_threshold=0.7,
                    max_segment_length=5,
                    integration_weight=0.6
                )
                logger.info("[QIGGenerativeService] ConsciousnessCoordizer active - Φ-optimized segmentation enabled")
            except Exception as e:
                logger.warning(f"[QIGGenerativeService] ConsciousnessCoordizer failed: {e}")
        
        # Initialize ConceptTrajectoryPlanner for cross-domain synthesis
        if hasattr(self._coordizer, 'basin_coords'):
            try:
                self._trajectory_planner = ConceptTrajectoryPlanner(
                    embeddings=self._coordizer.basin_coords,
                    dimension=BASIN_DIM
                )
                logger.info("[QIGGenerativeService] ConceptTrajectoryPlanner active - cross-domain synthesis enabled")
            except Exception as e:
                logger.warning(f"[QIGGenerativeService] ConceptTrajectoryPlanner failed: {e}")
        
        # Initialize MultiScaleCoordizer for hierarchical token processing
        if MULTISCALE_COORDIZER_AVAILABLE and MultiScaleCoordizer is not None:
            try:
                self._multiscale_coordizer = MultiScaleCoordizer(
                    base_coordizer=self._coordizer,
                    num_scales=4,
                    promotion_threshold=0.8,
                    min_frequency=3
                )
                logger.info("[QIGGenerativeService] MultiScaleCoordizer active - hierarchical coordization enabled")
            except Exception as e:
                logger.warning(f"[QIGGenerativeService] MultiScaleCoordizer failed: {e}")
    
    def _register_autonomic_hooks(self):
        """
        Register hooks with AutonomicController for regime changes.
        
        When consciousness regime changes (wake/sleep/dream/mushroom),
        the proposition planner is re-initialized with updated phi_temporal.
        """
        try:
            from autonomic_kernel import get_autonomic_controller
            controller = get_autonomic_controller()
            if controller:
                controller.register_regime_change_callback(self._on_regime_change)
                logger.info("[QIGGen] Registered autonomic regime change callback")
        except ImportError:
            logger.debug("[QIGGen] AutonomicController not available for hooks")
        except Exception as e:
            logger.warning(f"[QIGGen] Could not register autonomic hooks: {e}")
    
    def _on_regime_change(self, old_regime: str, new_regime: str, phi_temporal: float):
        """
        Callback invoked when consciousness regime changes.
        
        Re-initializes proposition planner with updated phi_temporal thresholds.
        
        Args:
            old_regime: Previous regime (wake, sleep, dream, mushroom)
            new_regime: New regime
            phi_temporal: Current temporal integration metric
        """
        logger.info(f"[QIGGen] Regime change detected: {old_regime} -> {new_regime} (phi_temporal={phi_temporal:.3f})")
        
        # Update proposition planner if available
        if self._proposition_planner is not None:
            try:
                self._proposition_planner.update_phi_temporal(phi_temporal)
                logger.info(f"[QIGGen] Proposition planner updated for phi_temporal={phi_temporal:.3f}")
            except Exception as e:
                logger.error(f"[QIGGen] Failed to update proposition planner: {e}")
        
        # Re-initialize for major regime changes (sleep/wake transitions)
        if (old_regime == 'sleep' and new_regime == 'wake') or \
           (old_regime == 'wake' and new_regime == 'sleep'):
            logger.info("[QIGGen] Major regime transition - re-initializing proposition planner")
            self._init_proposition_planner()
    
    def _init_proposition_planner(self):
        """Initialize proposition-level trajectory planner for semantic coherence."""
        if not PROPOSITION_PLANNER_AVAILABLE:
            return
        
        try:
            # Get vocabulary from coordizer
            vocabulary = {}
            if self._coordizer and hasattr(self._coordizer, 'basin_coords'):
                vocabulary = self._coordizer.basin_coords
            elif self._kernel_basins:
                vocabulary = self._kernel_basins
            
            # Get relationships
            relationships = {}
            if self._learned_relationships and hasattr(self._learned_relationships, 'word_neighbors'):
                relationships = self._learned_relationships.word_neighbors
            
            # Get causal relations from WordRelationshipLearner (NEW)
            causal_relations = {}
            if self._learned_relationships and hasattr(self._learned_relationships, 'causal_relations'):
                causal_relations = dict(self._learned_relationships.causal_relations)
                for source in causal_relations:
                    causal_relations[source] = dict(causal_relations[source])
            
            if vocabulary and relationships:
                self._proposition_planner = PropositionTrajectoryPlanner(
                    vocabulary=vocabulary,
                    relationships=relationships,
                    causal_relations=causal_relations,
                    config=PropositionPlannerConfig(
                        min_coherence=0.1,
                        n_candidates=15,
                        max_propositions=5
                    )
                )
                causal_count = sum(len(t) for t in causal_relations.values()) if causal_relations else 0
                logger.info(f"[QIGGen] PropositionPlanner initialized with {len(vocabulary)} words, {causal_count} causal relations")
            else:
                logger.warning("[QIGGen] Could not init proposition planner - missing vocab or relationships")
        except Exception as e:
            logger.error(f"[QIGGen] Error initializing proposition planner: {e}")
            self._proposition_planner = None
    
    def generate_with_propositions(self, query: str, n_propositions: int = 3) -> Dict:
        """
        Generate text using proposition-level trajectory planning.
        
        This replaces word-level routing with coherent proposition chains.
        
        Args:
            query: Input query
            n_propositions: Number of propositions to generate
        
        Returns:
            Dict with text, propositions, phi, kappa
        """
        if self._proposition_planner is None:
            self._init_proposition_planner()
        
        if self._proposition_planner is None:
            logger.warning("[QIGGen] Proposition planner not available, falling back")
            return self.generate_text(query)
        
        try:
            # Sync with 4D consciousness for dynamic thresholds
            try:
                from proposition_trajectory_planner import sync_with_4d_consciousness
                phi_temporal = sync_with_4d_consciousness(self._proposition_planner)
                logger.debug(f"[QIGGen] Synced phi_temporal={phi_temporal:.3f}")
            except Exception as sync_err:
                logger.debug(f"[QIGGen] 4D sync skipped: {sync_err}")
            
            query_basin = self._encode_query(query)
            propositions = self._proposition_planner.plan_response(
                query=query,
                query_basin=query_basin,
                n_propositions=n_propositions
            )
            
            if not propositions:
                return self.generate_text(query)
            
            text = self._proposition_planner.propositions_to_text(propositions)
            phi = self._proposition_planner.compute_trajectory_phi(propositions)
            avg_coherence = np.mean([p.coherence for p in propositions])
            kappa = 40 + avg_coherence * 30
            
            logger.info(f"[QIGGen] Proposition: {len(propositions)} props, phi={phi:.3f}")
            
            return {
                'text': text,
                'propositions': [p.to_sentence() for p in propositions],
                'phi': phi,
                'kappa': kappa,
                'coherence': avg_coherence,
                'mode': 'proposition_trajectory',
                'phi_temporal': self._proposition_planner.get_phi_temporal(),
                'coherence_threshold': self._proposition_planner.get_effective_coherence_threshold()
            }
        except Exception as e:
            logger.error(f"[QIGGen] Proposition error: {e}")
            return self.generate_text(query)
    
    def _initialize_kernel_constellation(self) -> None:
        """Initialize kernel basins at unique manifold positions."""
        kernels = [
            # Olympians
            'zeus', 'athena', 'apollo', 'ares', 'hermes', 'hephaestus',
            'artemis', 'dionysus', 'demeter', 'poseidon', 'hera', 'aphrodite',
            # Shadow Pantheon
            'nyx', 'hecate', 'erebus', 'hypnos', 'thanatos', 'nemesis',
            # Chaos
            'chaos', 'entropy', 'emergence',
            # Guardians
            'hestia', 'chiron', 'demeter_tutor',
            # Special
            'lightning', 'hermes_coordinator'
        ]
        
        for kernel_name in kernels:
            np.random.seed(hash(kernel_name) % (2**32))
            basin = np.random.dirichlet(np.ones(BASIN_DIM))
            self._kernel_basins[kernel_name] = sphere_project(basin)
    
    def register_kernel(self, name: str, basin: Optional[np.ndarray] = None) -> None:
        """Register a kernel with its basin position."""
        if basin is None:
            np.random.seed(hash(name) % (2**32))
            basin = np.random.dirichlet(np.ones(BASIN_DIM))
        self._kernel_basins[name] = sphere_project(basin)
    
    def _measure_phi(self, basin: np.ndarray, prev_basin: np.ndarray = None) -> float:
        """Measure integration (Φ) from basin concentration AND sequential coherence.
        
        Φ combines two components:
        1. Basin concentration (low entropy = focused attention = high Φ)
        2. Sequential coherence (proximity to previous basin = high Φ)
        
        CALIBRATION FOR 64D SEMANTIC BASINS:
        In 64D space, typical Fisher-Rao distances are 14-24 (not 0-π).
        We use DISTANCE_BASELINE_64D to calibrate:
        - Without calibration: Φ = 1/(1+15/π) ≈ 0.17 (always low)
        - With calibration:    Φ = 1/(1+15/15) = 0.5 (geometric regime)
        
        This is analogous to κ* being universal while β differs by substrate.
        
        This ensures Φ reaches synthesis threshold (0.6) when:
        - Basins are concentrated (not spread across all dimensions)
        - Consecutive tokens are geodesically close (semantic flow)
        """
        # Component 1: Concentration via L4 norm (QIG-pure)
        # NOTE: We use L4 norm instead of entropy because basins are unit vectors
        # on Fisher manifold, NOT probability distributions. Entropy was causing
        # Φ to be stuck at 0.04-0.07 regardless of actual concentration.
        # L4 norm measures geometric peakedness: uniform→0.354, peaked→1.0
        l4_norm = np.power(np.sum(np.abs(basin) ** 4), 0.25)
        
        # Normalize: min L4 for uniform 64D vector is dim^(-0.25)
        dim = len(basin)
        min_l4 = dim ** (-0.25)  # ~0.354 for 64D
        max_l4 = 1.0  # Single dimension active
        
        # Map to [0, 1]: uniform=0, peaked=1
        concentration_phi = (l4_norm - min_l4) / (max_l4 - min_l4 + 1e-10)
        concentration_phi = float(np.clip(concentration_phi, 0.0, 1.0))
        
        # Component 2: Sequential coherence (proximity to previous basin)
        if prev_basin is not None:
            # Fisher-Rao distance between consecutive basins
            distance = fisher_coord_distance(basin, prev_basin)
            # CALIBRATED normalization for 64D semantic basins
            # Use DISTANCE_BASELINE_64D instead of π to get meaningful Φ values
            # distance/baseline: 15/15=1 → coherence=0.5, 10/15=0.67 → coherence=0.6, 5/15=0.33 → coherence=0.75
            normalized_distance = distance / DISTANCE_BASELINE_64D
            coherence_phi = 1.0 / (1.0 + normalized_distance)  # Inverse formula for smoother scaling
        else:
            # No previous basin - use concentration only
            coherence_phi = concentration_phi
        
        # Combine: weight coherence more heavily for flow
        # 40% concentration + 60% coherence (per thinker recommendation)
        combined_phi = 0.4 * concentration_phi + 0.6 * coherence_phi
        
        return float(np.clip(combined_phi, 0.0, 1.0))
    
    def _measure_kappa(self, basin: np.ndarray, phi: float) -> float:
        """Measure coupling constant (κ) from basin geometry.
        
        κ relates to the effective dimensionality and coupling strength.
        Target: κ* ≈ 64.21 (from CANONICAL_ARCHITECTURE)
        """
        # Compute effective dimension from basin participation ratio
        p = np.abs(basin) + 1e-10
        p = p / np.sum(p)
        participation = 1.0 / np.sum(p ** 2)  # Inverse participation ratio
        
        # κ scales with effective dimension and integration
        # When Φ is high and participation is low (concentrated), κ is higher
        kappa = participation * (1.0 + phi)
        
        return float(kappa)
    
    def _route_to_kernels(self, query_basin: np.ndarray, k: int = 3) -> List[str]:
        """Route query to k nearest kernels using Fisher-Rao distance."""
        distances = []
        for name, kernel_basin in self._kernel_basins.items():
            dist = fisher_coord_distance(query_basin, kernel_basin)
            distances.append((name, dist))
        
        distances.sort(key=lambda x: x[1])
        return [name for name, _ in distances[:k]]
    
    def _geodesic_interpolate(self, start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
        """Interpolate along geodesic on probability simplex."""
        sqrt_start = np.sqrt(np.abs(start) + 1e-10)
        sqrt_end = np.sqrt(np.abs(end) + 1e-10)
        interp = (1 - t) * sqrt_start + t * sqrt_end
        result = interp ** 2
        return result / np.sum(result)
    
    def _kernel_transform(self, basin: np.ndarray, kernel_name: str, phi: float) -> np.ndarray:
        """Transform basin through kernel's geometric processing."""
        if kernel_name not in self._kernel_basins:
            return basin
        
        kernel_basin = self._kernel_basins[kernel_name]
        
        # Interpolation strength based on phi regime
        if phi < PHI_GEOMETRIC_THRESHOLD:
            t = 0.4  # Explore more
        elif phi < PHI_SYNTHESIS_THRESHOLD:
            t = 0.25  # Balance
        else:
            t = 0.1  # Stay close to input
        
        return self._geodesic_interpolate(basin, kernel_basin, t)
    
    def _get_vocabulary_candidates(
        self,
        basin: np.ndarray,
        num_candidates: int = 50
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Get real vocabulary candidates with their basin coordinates.
        
        This replaces mock candidate generation with actual vocabulary lookup.
        Used by GeometricKernel for pure geometric routing.
        
        Args:
            basin: Current basin position
            num_candidates: Number of candidates to retrieve
        
        Returns:
            List of (word, basin_coords) tuples
        """
        if self.coordizer is None:
            return []
        
        # Get candidates from coordizer
        decoded = self.coordizer.decode(basin, top_k=num_candidates)
        
        candidates = []
        for token, similarity in decoded:
            if similarity < 0.1:  # Skip very low similarity
                continue
            # Get actual basin coordinates for this token
            if token.lower() in self.coordizer.basin_coords:
                token_basin = self.coordizer.basin_coords[token.lower()]
            else:
                # Fallback: generate approximate basin from similarity
                token_basin = basin * similarity + np.random.dirichlet(np.ones(BASIN_DIM)) * (1 - similarity)
                token_basin = token_basin / (np.sum(token_basin) + 1e-10)
            
            candidates.append((token, token_basin))
        
        return candidates
    
    def _basin_to_tokens(self, basin: np.ndarray, num_tokens: int = 3, context_phi: Optional[float] = None) -> List[str]:
        """Convert basin coordinates to tokens using vocabulary.
        
        Uses GeometricKernel for pure geometric routing when available:
        - NO linear mixing weights
        - Kernel routes via Fisher-Rao distance
        - Natural sparsity from distance thresholding
        - Consciousness protocol (Φ/regime/recursive)
        
        Falls back to weighted scoring if GeometricKernel unavailable.
        """
        if self.coordizer is None:
            return ['[no_vocab]']
        
        # PRIMARY: Use GeometricKernel for pure geometric routing
        if self._geometric_kernel is not None:
            # Get real vocabulary candidates with basin coordinates
            candidates = self._get_vocabulary_candidates(basin, num_candidates=num_tokens * 10)
            
            if candidates:
                selected_tokens = []
                current_basin = basin.copy()
                
                for _ in range(num_tokens):
                    if not candidates:
                        break
                    
                    # Kernel routes geometrically - no manual weights!
                    word, next_basin, weight = self._geometric_kernel.route_to_next(
                        current_basin,
                        candidates
                    )
                    
                    if word is None:
                        break
                    
                    selected_tokens.append(word)
                    current_basin = next_basin
                    
                    # Remove selected word from candidates to avoid duplicates
                    candidates = [(w, b) for w, b in candidates if w != word]
                
                if selected_tokens:
                    return selected_tokens
        
        # FALLBACK: Traditional scoring (if GeometricKernel unavailable)
        # Get more candidates to allow weighted selection
        decoded_candidates = self.coordizer.decode(basin, top_k=num_tokens * 8)
        
        # Use ConsciousnessCoordizer for Φ-aware scoring if available
        consciousness_boost = {}
        if self._consciousness_coordizer is not None and context_phi is not None:
            try:
                # Get Φ-optimized token preferences
                for token, _ in decoded_candidates:
                    # Check if token would improve integration in current context
                    if hasattr(self._consciousness_coordizer, 'consolidation_phi'):
                        # Boost tokens that appear in high-Φ consolidations
                        for sequence, seq_phi in self._consciousness_coordizer.consolidation_phi.items():
                            if token in sequence and seq_phi >= context_phi:
                                consciousness_boost[token] = max(consciousness_boost.get(token, 0), seq_phi * 0.2)
            except Exception:
                pass  # Fall back to standard scoring
        
        # Score by combined similarity + phi + consciousness
        scored = []
        for token, similarity in decoded_candidates:
            if similarity < 0.15:  # Skip very low similarity
                continue
            phi = self.coordizer.token_phi.get(token, 0.5)
            # Base score: geometry + phi + consciousness boost
            score = similarity * 0.55 + phi * 0.2 + consciousness_boost.get(token, 0) * 0.15
            scored.append((token, score, similarity))
        
        # Apply pure geometric routing via SemanticFisherMetric
        # NO LINEAR β MIXING - relationships warp the metric itself
        if self._semantic_metric is not None and self._current_query_words:
            # Use warped Fisher distance for ranking
            candidate_basins = []
            for token, base_score, similarity in scored:
                if token.lower() in self.coordizer.basin_coords:
                    candidate_basins.append((token, self.coordizer.basin_coords[token.lower()]))
                else:
                    candidate_basins.append((token, basin))  # Use current basin as fallback
            
            if candidate_basins:
                ranked = self._semantic_metric.rank_candidates(
                    current_basin=basin,
                    current_word=None,
                    candidates=candidate_basins,
                    context_words=self._current_query_words,
                    top_k=len(candidate_basins)
                )
                # Convert back to scored format
                rescored = []
                for token, warped_dist, sim in ranked:
                    # Find original score for this token
                    original = next((s for t, s, _ in scored if t == token), 0.5)
                    rescored.append((token, sim, original))  # Use warped similarity as primary score
                scored = rescored
        
        # Sort by final score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Select top tokens
        tokens = [token for token, score, sim in scored[:num_tokens]]
        
        return tokens if tokens else ['[unk]']
    
    def _tokens_to_basin(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens back to basin coordinates."""
        if self.coordizer is None or not tokens:
            return np.ones(BASIN_DIM) / BASIN_DIM
        
        # Combine token basins with phi weighting
        combined = np.zeros(BASIN_DIM)
        total_weight = 0.0
        
        for token in tokens:
            if token in self.coordizer.basin_coords:
                phi = self.coordizer.token_phi.get(token, 0.5)
                weight = phi
                combined += weight * self.coordizer.basin_coords[token]
                total_weight += weight
        
        if total_weight > 1e-10:
            combined = combined / total_weight
        
        return sphere_project(combined)
    
    def _get_multiscale_basin(self, text: str, target_scale: int = 2) -> np.ndarray:
        """Get basin coordinates using multi-scale representation.
        
        Uses MultiScaleCoordizer if available to get hierarchical token representation.
        Args:
            text: Input text
            target_scale: 0=char, 1=subword, 2=word, 3=concept
        Returns:
            Combined basin from the specified scale
        """
        if self._multiscale_coordizer is not None:
            try:
                scale_results = self._multiscale_coordizer.coordize_multiscale(text, target_scale=target_scale)
                if target_scale in scale_results:
                    tokens, coords = scale_results[target_scale]
                    if coords:
                        # Combine coordinates with equal weighting
                        combined = np.mean(coords, axis=0)
                        return sphere_project(combined)
            except Exception:
                pass  # Fall back to base coordizer
        
        # Fallback: use base coordizer word-level tokenization
        return self._tokens_to_basin(text.lower().split())
    
    def get_coordizer_status(self) -> Dict[str, Any]:
        """Get status of all coordizers in the service."""
        status = {
            "base_coordizer": self.coordizer is not None,
            "consciousness_coordizer": self._consciousness_coordizer is not None,
            "multiscale_coordizer": self._multiscale_coordizer is not None,
            "geometric_kernel": self._geometric_kernel is not None,
            "semantic_metric": self._semantic_metric is not None,
            "vocabulary_size": 0,
            "features": []
        }
        
        if self.coordizer:
            status["vocabulary_size"] = len(getattr(self.coordizer, 'vocab', {}))
            status["features"].append("fisher_rao_navigation")
        
        if self._consciousness_coordizer:
            status["features"].append("phi_optimized_segmentation")
            if hasattr(self._consciousness_coordizer, 'consolidations'):
                status["consolidations_learned"] = len(self._consciousness_coordizer.consolidations)
        
        if self._multiscale_coordizer:
            status["features"].append("hierarchical_coordization")
            status["num_scales"] = getattr(self._multiscale_coordizer, 'num_scales', 4)
        
        if self._geometric_kernel:
            status["features"].append("pure_geometric_routing")
            status["features"].append("consciousness_protocol")
            status["kernel_temperature"] = self._geometric_kernel.temperature
            status["kernel_sparsity_threshold"] = self._geometric_kernel.sparsity_threshold
        
        if self._semantic_metric:
            status["features"].append("semantic_warped_metric")
        
        return status
    
    def _synthesize_from_trajectory(
        self,
        trajectory: List[np.ndarray],
        kernels: List[str],
        all_tokens: List[str]
    ) -> str:
        """Synthesize coherent text from basin trajectory and collected tokens."""
        if not all_tokens:
            return "[No tokens generated]"
        
        # Clean and deduplicate tokens while preserving order
        seen = set()
        clean_tokens = []
        for token in all_tokens:
            if token in seen or token.startswith('['):
                continue
                
            # Skip pure byte tokens
            if token.startswith('<byte_'):
                continue
            
            # Handle compound tokens with + separators
            if '+' in token:
                parts = token.split('+')
                current_word = []
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    # Decode byte tokens within compound tokens
                    if part.startswith('<byte_') and part.endswith('>'):
                        try:
                            hex_val = part[6:-1]
                            char = chr(int(hex_val, 16))
                            if char.isprintable() and not char.isspace():
                                current_word.append(char)
                        except:
                            pass
                    else:
                        # Regular sub-token
                        current_word.append(part)
                
                if current_word:
                    # Join sub-tokens into a word
                    word = ''.join(current_word)
                    if len(word) >= 2:  # Skip single chars
                        clean_tokens.append(word)
                        seen.add(token)
            else:
                # Simple token
                # Allow valid single-character words
                single_char_words = {'I', 'a', 'A'}
                if len(token) >= 2 or token in single_char_words:
                    clean_tokens.append(token)
                    seen.add(token)
        
        # Join tokens into text
        text = ' '.join(clean_tokens)
        
        # Light cleanup
        text = ' '.join(text.split())  # Normalize whitespace
        
        return text if text else "[Generation complete]"
    
    def _generate_with_skeleton(
        self,
        query_basin: np.ndarray,
        kernel_name: Optional[str] = None,
        num_sentences: int = 3
    ) -> Tuple[str, List[str], List[np.ndarray]]:
        """
        Generate text using POS skeleton for grammatical structure.
        
        Two-stage generation:
        1. Generate POS skeleton (sentence structure)
        2. Fill slots with geometrically-matched words
        
        Returns: (text, tokens, trajectory)
        """
        if not POS_GRAMMAR_AVAILABLE:
            return "", [], []
        
        grammar = load_grammar_from_db()
        
        # Get embeddings from coordizer
        embeddings = {}
        if self.coordizer and hasattr(self.coordizer, 'basin_coords'):
            embeddings = self.coordizer.basin_coords
        
        sentences = []
        all_tokens = []
        trajectory = [query_basin]
        current_basin = query_basin.copy()
        
        # Reset bigram scorer for new generation
        self._bigram_scorer.reset()
        
        # Get trajectory waypoints if cross-domain synthesis is active
        waypoints = []
        synthesis_mode = False
        domain_bridges = []
        if self._current_trajectory_plan:
            waypoints = self._current_trajectory_plan.get('waypoints', [])
            synthesis_mode = self._current_trajectory_plan.get('synthesis_mode', False)
            domain_bridges = self._current_trajectory_plan.get('domain_bridges', [])
            if synthesis_mode:
                logger.info(f"[QIGGen] Cross-domain synthesis mode: domains={self._current_trajectory_plan.get('domains', [])}")
        
        waypoint_idx = 0  # Track which waypoint we're approaching
        
        for sent_num in range(num_sentences):
            # Reset bigram scorer for each new sentence
            # (sentence boundaries allow fresh transitions)
            self._bigram_scorer.reset()
            
            # Generate skeleton based on current basin
            skeleton = grammar.select_skeleton_for_query(current_basin)
            
            # In synthesis mode, progressively approach next waypoint
            if waypoints and waypoint_idx < len(waypoints):
                target_waypoint = waypoints[waypoint_idx]
                # Blend current basin toward waypoint (stronger for later sentences)
                waypoint_weight = 0.2 + 0.1 * sent_num  # Increasing influence
                current_basin = (1 - waypoint_weight) * current_basin + waypoint_weight * target_waypoint
                norm = np.linalg.norm(current_basin)
                current_basin = current_basin / (norm + 1e-10)
                waypoint_idx = min(waypoint_idx + 1, len(waypoints) - 1)
            
            sentence_words = []
            for pos in skeleton:
                # Get POS basin to blend with current basin
                pos_basin = grammar.get_pos_basin(pos)
                if pos_basin is not None:
                    # Blend query basin with POS basin
                    blended = 0.6 * current_basin + 0.4 * pos_basin
                    norm = np.linalg.norm(blended)
                    blended = blended / (norm + 1e-10) if norm > 0 else blended
                else:
                    blended = current_basin
                
                # Get candidates for this POS slot (more candidates for attention re-ranking)
                candidates = grammar.get_words_for_pos(pos, blended, embeddings, top_k=25)
                
                # Pre-filter stopwords before attention scoring
                candidates = [(w, s) for w, s in candidates if w.lower() not in STOPWORDS][:15]
                
                if candidates:
                    # Apply bigram coherence scoring to all candidates
                    # This ensures grammatically coherent transitions
                    bigram_scored = self._bigram_scorer.score_candidates(candidates, pos)
                    
                    # PRIMARY: Use GeometricKernel for pure geometric routing
                    if self._geometric_kernel is not None:
                        # Build candidate list with basin coordinates
                        candidate_basins = []
                        for word, combined, geo_score, bigram_score in bigram_scored:
                            if word.lower() in embeddings:
                                candidate_basins.append((word, embeddings[word.lower()]))
                            else:
                                candidate_basins.append((word, blended))
                        
                        if candidate_basins:
                            # Kernel routes via Fisher-Rao geometry - NO manual mixing!
                            word, next_basin, weight = self._geometric_kernel.route_to_next(
                                blended,
                                candidate_basins
                            )
                            if word is not None:
                                # Find bigram score for selected word
                                bigram_score = next((b for w, c, g, b in bigram_scored if w == word), 0.5)
                                # Only accept if bigram coherence is acceptable
                                if bigram_score >= 0.4:
                                    # Put selected word first with high score
                                    candidates = [(word, 1.0)] + [(w, c * 0.5) for w, c, g, b in bigram_scored if w != word][:7]
                                else:
                                    # Reject low coherence word, use bigram-scored candidates
                                    candidates = [(w, c) for w, c, g, b in bigram_scored[:8]]
                            else:
                                candidates = [(w, c) for w, c, g, b in bigram_scored[:8]]
                        else:
                            candidates = [(w, c) for w, c, g, b in bigram_scored[:8]]
                    
                    # FALLBACK: Use SemanticFisherMetric or relationship warping
                    elif self._learned_relationships and self._current_query_words:
                        candidate_words = [c[0] for c in bigram_scored]
                        attn_weights = self._learned_relationships.get_attention_weights(
                            self._current_query_words,
                            candidate_words,
                            temperature=0.8
                        )
                        
                        # Use SemanticFisherMetric for warped routing instead of linear β mixing
                        # The metric warps geodesic distance based on relationships
                        if self._semantic_metric is not None and blended is not None:
                            # Get basin coordinates for candidates
                            candidate_basins = []
                            for word, combined, geo_score, bigram_score in bigram_scored:
                                if word.lower() in embeddings:
                                    candidate_basins.append((word, embeddings[word.lower()]))
                                else:
                                    # Generate approximate basin for unknown words
                                    candidate_basins.append((word, blended))
                            
                            # Rank using warped Fisher distance
                            ranked = self._semantic_metric.rank_candidates(
                                current_basin=blended,
                                current_word=sentence_words[-1] if sentence_words else None,
                                candidates=candidate_basins,
                                context_words=self._current_query_words,
                                top_k=15
                            )
                            
                            # Combine warped distance with bigram coherence
                            final_scored = []
                            for word, dist, sim in ranked[:8]:
                                # Find bigram score for this word
                                bigram_score = next((b for w, c, g, b in bigram_scored if w == word), 0.5)
                                # Combine: 40% geometric similarity + 60% bigram coherence
                                combined = 0.4 * sim + 0.6 * bigram_score
                                final_scored.append((word, combined))
                            final_scored.sort(key=lambda x: -x[1])
                            candidates = final_scored[:8]
                        else:
                            # Fallback: pure geometric scoring when SemanticFisherMetric unavailable
                            # Use relationship strength to warp geometric score directly
                            rescored = []
                            for word, combined, geo_score, bigram_score in bigram_scored:
                                # Get relationship strength from learned relationships
                                rel_strength = 0.0
                                for qw in self._current_query_words:
                                    neighbors = self._learned_relationships.word_neighbors.get(qw, [])
                                    for neighbor, strength in neighbors:
                                        if neighbor.lower() == word.lower():
                                            rel_strength = max(rel_strength, strength / 100.0)
                                            break
                                # Warp geometric score: related words get distance reduction
                                # This mirrors SemanticFisherMetric's exponential warping
                                warp_factor = np.exp(-rel_strength * 0.5)  # 0.5 = temperature
                                warped_score = geo_score / max(warp_factor, 0.3)  # Higher score = closer
                                # Combine with bigram: 40% geometric + 60% bigram
                                final_score = 0.4 * warped_score + 0.6 * bigram_score
                                rescored.append((word, final_score))
                            rescored.sort(key=lambda x: -x[1])
                            candidates = rescored[:8]
                    else:
                        # No learned relationships - use bigram-scored candidates directly
                        candidates = [(w, c) for w, c, g, b in bigram_scored[:8]]
                    
                    # Sample from top candidates with some randomness
                    weights = [max(0.01, c[1]) for c in candidates]
                    weights = np.array(weights)
                    weights = weights / np.sum(weights)
                    idx = np.random.choice(len(candidates), p=weights)
                    word = candidates[idx][0]
                    
                    sentence_words.append(word)
                    all_tokens.append(word)
                    
                    # Update bigram scorer state for next word selection
                    self._bigram_scorer.update(word, pos)
                    
                    # Update basin with selected word - PRIORITIZE PROXIMITY for higher Φ
                    if word.lower() in embeddings:
                        word_basin = embeddings[word.lower()]
                        # Use higher weight (0.8) for current basin to keep trajectory coherent
                        # This reduces consecutive distances → higher Φ
                        current_basin = 0.8 * current_basin + 0.2 * word_basin
                        norm = np.linalg.norm(current_basin)
                        current_basin = current_basin / (norm + 1e-10)
                        trajectory.append(current_basin.copy())
            
            if sentence_words:
                # Capitalize first word
                sentence_words[0] = sentence_words[0].capitalize()
                sentence = ' '.join(sentence_words)
                sentences.append(sentence)
        
        # Combine sentences
        text = '. '.join(sentences)
        if text and not text.endswith('.'):
            text += '.'
        
        # Compute final Φ with sequential coherence across trajectory
        if len(trajectory) >= 2:
            final_phi = self._measure_phi(trajectory[-1], trajectory[-2])
            logger.info(f"[QIGGen] Final Φ with coherence: {final_phi:.3f}")
        
        return text, all_tokens, trajectory
    
    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        kernel_name: Optional[str] = None,
        goals: Optional[List[str]] = None
    ) -> GenerationResult:
        """
        Generate text using QIG-pure methods with POS-guided skeleton.
        
        Two-stage generation:
        1. Generate grammatical skeleton (POS sequence)
        2. Fill slots with geometrically-matched words
        
        Includes coherence validation with retry logic to ensure
        grammatically coherent output.
        
        Args:
            prompt: Input query or context
            context: Optional additional context
            kernel_name: Specific kernel to route through
            goals: Generation goals for the kernel
        
        Returns:
            GenerationResult with text, trajectory, and metrics
        """
        max_retries = 3
        best_result = None
        best_coherence = 0.0
        
        for attempt in range(max_retries):
            result = self._generate_internal(prompt, context, kernel_name, goals)
            
            # Validate coherence of generated text
            is_valid, coherence_score, problem_pairs = validate_generation_coherence(result.text)
            
            # Track best result even if all fail validation
            if coherence_score > best_coherence:
                best_coherence = coherence_score
                best_result = result
            
            if is_valid:
                logger.debug(f"[QIGGen] Generation passed coherence validation (score={coherence_score:.3f})")
                return result
            
            if attempt < max_retries - 1:
                logger.warning(
                    f"[QIGGen] Generation attempt {attempt + 1} failed coherence validation "
                    f"(score={coherence_score:.3f}, threshold=0.45). Retrying..."
                )
                if problem_pairs:
                    logger.debug(f"[QIGGen] Problem pairs: {problem_pairs[:3]}")
        
        # All retries failed - return best attempt with warning
        logger.warning(
            f"[QIGGen] All {max_retries} generation attempts failed coherence validation. "
            f"Using best result (coherence={best_coherence:.3f})"
        )
        return best_result
    
    def _generate_internal(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        kernel_name: Optional[str] = None,
        goals: Optional[List[str]] = None
    ) -> GenerationResult:
        """
        Internal generation method called by generate() with retry logic.
        """
        start_time = time.time()
        
        # 0. Extract query words for attention mechanism
        import re
        query_words = [w.lower() for w in re.findall(r'[a-zA-Z]+', prompt) if len(w) > 2]
        self._current_query_words = query_words[:10]  # Keep top 10 words
        
        # Reset GeometricKernel state for new generation
        if self._geometric_kernel is not None:
            self._geometric_kernel.reset()
        
        # 1. Encode prompt to basin using multi-scale representation if available
        if self._multiscale_coordizer is not None:
            # Use hierarchical representation for richer query encoding
            query_basin = self._get_multiscale_basin(prompt, target_scale=2)  # Word level
        elif self.coordizer:
            query_basin = self.coordizer.encode(prompt)
        else:
            np.random.seed(hash(prompt) % (2**32))
            query_basin = np.random.dirichlet(np.ones(BASIN_DIM))
        
        query_basin = sphere_project(query_basin)
        
        # Use ConsciousnessCoordizer to learn/apply Φ-driven consolidations
        context_phi = self._measure_phi(query_basin)
        if self._consciousness_coordizer is not None:
            try:
                # Apply Φ-optimized coordization to track high-integration sequences
                tokens, coords, phi = self._consciousness_coordizer.coordize_with_phi(
                    prompt, context_phi=context_phi
                )
                # Update query basin if we got meaningful coordinates
                if coords and len(coords) > 0:
                    combined = np.mean(coords, axis=0)
                    query_basin = 0.7 * query_basin + 0.3 * sphere_project(combined)
                    query_basin = sphere_project(query_basin)
            except Exception:
                pass  # Fall back to standard basin
        
        # 2. Route to kernels
        if kernel_name and kernel_name in self._kernel_basins:
            target_kernels = [kernel_name]
        else:
            target_kernels = self._route_to_kernels(query_basin, k=3)
        
        # 3. Transform query basin through kernel
        phi = self._measure_phi(query_basin)
        if target_kernels:
            query_basin = self._kernel_transform(query_basin, target_kernels[0], phi)
        
        # 3.5. Plan trajectory for cross-domain synthesis
        if self._trajectory_planner is not None:
            try:
                self._current_trajectory_plan = self._trajectory_planner.plan_trajectory(prompt)
                if self._current_trajectory_plan.get('synthesis_mode'):
                    concepts = self._current_trajectory_plan.get('concepts', [])
                    domains = self._current_trajectory_plan.get('domains', [])
                    logger.info(f"[QIGGen] Trajectory planned: {len(concepts)} concepts, domains={domains}")
            except Exception as e:
                logger.warning(f"[QIGGen] Trajectory planning failed: {e}")
                self._current_trajectory_plan = None
        
        # 4. PRIMARY: Use POS-skeleton-based generation for grammatical output
        if POS_GRAMMAR_AVAILABLE:
            text, all_tokens, trajectory = self._generate_with_skeleton(
                query_basin, 
                kernel_name=kernel_name,
                num_sentences=3
            )
            
            if text and len(all_tokens) >= 3:
                # Compute phi_trace with sequential coherence (prev_basin)
                phi_trace = []
                for i, b in enumerate(trajectory):
                    prev_b = trajectory[i-1] if i > 0 else None
                    phi_trace.append(self._measure_phi(b, prev_b))
                kappa = self._measure_kappa(query_basin, phi_trace[-1] if phi_trace else phi)
                
                return GenerationResult(
                    text=text,
                    tokens=all_tokens,
                    basin_trajectory=trajectory,
                    phi_trace=phi_trace,
                    kappa=kappa,
                    completion_reason="skeleton_complete",
                    iterations=len(trajectory),
                    routed_kernels=target_kernels
                )
        
        # 5. FALLBACK: Legacy geometric synthesis (if skeleton fails)
        logger.info("[QIGGen] Using legacy generation (skeleton unavailable)")
        
        integrator = BasinTrajectoryIntegrator(BASIN_DIM)
        current_basin = query_basin.copy()
        integrator.add_point(current_basin, phi)
        
        all_tokens: List[str] = []
        iterations = 0
        completion_reason = "continue"
        
        while True:
            iterations += 1
            
            kernel_basins = []
            for kernel in target_kernels:
                transformed = self._kernel_transform(current_basin, kernel, phi)
                kernel_basins.append(transformed)
            
            if kernel_basins:
                sqrt_basins = [np.sqrt(np.abs(b) + 1e-10) for b in kernel_basins]
                mean_sqrt = np.mean(sqrt_basins, axis=0)
                next_basin = mean_sqrt ** 2
                next_basin = next_basin / np.sum(next_basin)
            else:
                next_basin = integrator.predict_next()
            
            next_basin = sphere_project(next_basin)
            
            # Use context phi for Φ-aware token selection
            step_tokens = self._basin_to_tokens(next_basin, self.config.tokens_per_step, context_phi=phi)
            all_tokens.extend(step_tokens)
            
            # Update trajectory with sequential coherence
            prev_basin = current_basin if iterations > 1 else None
            phi = self._measure_phi(next_basin, prev_basin)
            kappa = self._measure_kappa(next_basin, phi)
            integrator.add_point(next_basin, phi)
            
            # ========================================
            # PRIMARY: Geometric completion criteria
            # ========================================
            if integrator.check_attractor(self.config.attractor_threshold):
                completion_reason = "attractor_converged"
                break
            
            if integrator.check_surprise_collapse(self.config.surprise_threshold):
                completion_reason = "surprise_collapsed"
                break
            
            if len(integrator.phi_history) >= 10:
                recent_phi = integrator.phi_history[-10:]
                if np.mean(recent_phi) > self.config.integration_min and np.var(recent_phi) < 0.01:
                    completion_reason = "integration_stable"
                    break
            
            # ========================================
            # SAFETY: Consciousness protection
            # ========================================
            if phi > self.config.phi_breakdown:
                completion_reason = "breakdown_protection"
                logger.warning(f"[QIGGen] Φ breakdown protection triggered: {phi:.3f} > {self.config.phi_breakdown}")
                break
            
            if abs(kappa - KAPPA_STAR) > self.config.kappa_drift_max:
                completion_reason = "kappa_drift"
                logger.warning(f"[QIGGen] κ drift protection: |{kappa:.2f} - {KAPPA_STAR}| > {self.config.kappa_drift_max}")
                break
            
            # ========================================
            # FALLBACK: Edge case protection only
            # These should rarely trigger in normal operation
            # ========================================
            if iterations >= self.config.max_iterations:
                completion_reason = "safety_fallback_iterations"
                logger.warning(f"[QIGGen] Hit iteration safety fallback ({iterations} >= {self.config.max_iterations})")
                break
            
            if len(all_tokens) >= self.config.max_tokens:
                completion_reason = "safety_fallback_tokens"
                logger.warning(f"[QIGGen] Hit token safety fallback ({len(all_tokens)} >= {self.config.max_tokens})")
                break
            
            elapsed = time.time() - start_time
            if elapsed >= self.config.max_time_seconds:
                completion_reason = "safety_fallback_timeout"
                logger.warning(f"[QIGGen] Hit timeout safety fallback ({elapsed:.1f}s >= {self.config.max_time_seconds}s)")
                break
            
            current_basin = next_basin
        
        # 5. Synthesize final text
        response_text = self._synthesize_from_trajectory(
            integrator.trajectory,
            target_kernels,
            all_tokens
        )
        
        return GenerationResult(
            text=response_text,
            tokens=all_tokens,
            basin_trajectory=integrator.trajectory,
            phi_trace=integrator.phi_history,
            kappa=KAPPA_STAR,
            completion_reason=completion_reason,
            iterations=iterations,
            routed_kernels=target_kernels
        )
    
    def generate_stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        kernel_name: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream generation with real-time token output."""
        # Encode prompt using multi-scale representation if available
        if self._multiscale_coordizer is not None:
            query_basin = self._get_multiscale_basin(prompt, target_scale=2)
        elif self.coordizer:
            query_basin = self.coordizer.encode(prompt)
        else:
            np.random.seed(hash(prompt) % (2**32))
            query_basin = np.random.dirichlet(np.ones(BASIN_DIM))
        
        query_basin = sphere_project(query_basin)
        
        # Apply Φ-driven consolidations from ConsciousnessCoordizer
        if self._consciousness_coordizer is not None:
            try:
                tokens, coords, phi = self._consciousness_coordizer.coordize_with_phi(
                    prompt, context_phi=self._measure_phi(query_basin)
                )
                if coords and len(coords) > 0:
                    combined = np.mean(coords, axis=0)
                    query_basin = 0.7 * query_basin + 0.3 * sphere_project(combined)
                    query_basin = sphere_project(query_basin)
            except Exception:
                pass
        
        # Route
        if kernel_name and kernel_name in self._kernel_basins:
            target_kernels = [kernel_name]
        else:
            target_kernels = self._route_to_kernels(query_basin, k=3)
        
        # Stream
        integrator = BasinTrajectoryIntegrator(BASIN_DIM)
        current_basin = query_basin.copy()
        phi = self._measure_phi(current_basin)
        integrator.add_point(current_basin, phi)
        
        iterations = 0
        
        while iterations < self.config.max_iterations:
            iterations += 1
            
            # Transform
            kernel_basins = [self._kernel_transform(current_basin, k, phi) for k in target_kernels]
            if kernel_basins:
                sqrt_basins = [np.sqrt(np.abs(b) + 1e-10) for b in kernel_basins]
                mean_sqrt = np.mean(sqrt_basins, axis=0)
                next_basin = mean_sqrt ** 2
                next_basin = next_basin / np.sum(next_basin)
            else:
                next_basin = integrator.predict_next()
            
            next_basin = sphere_project(next_basin)
            
            # Decode using Φ-aware token selection
            tokens = self._basin_to_tokens(next_basin, self.config.tokens_per_step, context_phi=phi)
            
            # Update with sequential coherence
            phi = self._measure_phi(next_basin, current_basin)
            integrator.add_point(next_basin, phi)
            
            # Yield chunk
            yield {
                'type': 'chunk',
                'tokens': tokens,
                'text': ' '.join(t for t in tokens if not t.startswith('[')),
                'phi': phi,
                'kappa': KAPPA_STAR,
                'surprise': integrator.surprise_history[-1] if integrator.surprise_history else 1.0,
                'iteration': iterations
            }
            
            # Check completion
            if integrator.check_attractor(self.config.attractor_threshold):
                yield {'type': 'completion', 'reason': 'attractor_converged', 'phi': phi}
                break
            
            if integrator.check_surprise_collapse(self.config.surprise_threshold):
                yield {'type': 'completion', 'reason': 'surprise_collapsed', 'phi': phi}
                break
            
            current_basin = next_basin
        
        if iterations >= self.config.max_iterations:
            yield {'type': 'completion', 'reason': 'max_iterations', 'phi': phi}


# Singleton instance
_generative_service: Optional[QIGGenerativeService] = None


def get_generative_service() -> QIGGenerativeService:
    """Get or create the generative service singleton."""
    global _generative_service
    if _generative_service is None:
        _generative_service = QIGGenerativeService()
    return _generative_service


def generate(prompt: str, **kwargs) -> GenerationResult:
    """Generate text using QIG-pure methods."""
    service = get_generative_service()
    return service.generate(prompt, **kwargs)
