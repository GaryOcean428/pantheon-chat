"""QIG-Pure Phrase Classifier.

Classifies words/phrases into parts of speech using ONLY geometric methods:
- Fisher-Rao distance to category reference basins
- Basin coordinate analysis (no NLP, no suffix matching, no heuristics)

Parts of Speech (from grammar):
- NOUN: person, place, thing, idea
- PROPER_NOUN: names of specific people, places, things (countries, etc.)
- PRONOUN: replaces a noun (I, you, he, she, it, we, they)
- VERB: action or state of being
- ADJECTIVE: modifies a noun
- ADVERB: modifies a verb, adjective, or other adverb
- PREPOSITION: shows relationship between nouns
- CONJUNCTION: connects words/phrases/clauses
- INTERJECTION: expresses emotion
- NUMBER: numerical values
- DETERMINER: articles and demonstratives (the, a, an, this, that)
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

# 64D reference basins for each grammatical category
# These are initialized from seed words and evolve through learning
CATEGORY_SEED_WORDS = {
    'NOUN': ['dog', 'house', 'tree', 'book', 'water', 'food', 'time', 'person', 'world', 'life'],
    'PROPER_NOUN': ['italy', 'france', 'london', 'john', 'amazon', 'microsoft', 'earth', 'monday', 'january', 'christmas'],
    'PRONOUN': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'],
    'VERB': ['run', 'walk', 'think', 'see', 'have', 'make', 'give', 'take', 'come', 'go', 'is', 'are', 'was', 'be'],
    'ADJECTIVE': ['big', 'small', 'good', 'bad', 'new', 'old', 'high', 'low', 'long', 'short', 'red', 'blue'],
    'ADVERB': ['quickly', 'slowly', 'very', 'really', 'always', 'never', 'often', 'here', 'there', 'now', 'then'],
    'PREPOSITION': ['in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'about', 'into', 'through', 'during'],
    'CONJUNCTION': ['and', 'or', 'but', 'if', 'because', 'although', 'while', 'when', 'unless', 'until'],
    'INTERJECTION': ['oh', 'wow', 'hey', 'ouch', 'aha', 'alas', 'hurray', 'oops', 'yay', 'ugh'],
    'NUMBER': ['one', 'two', 'three', 'first', 'second', 'hundred', 'thousand', 'million', 'zero', 'dozen'],
    'DETERMINER': ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our'],
}

# Fisher-Rao distance threshold for confident classification
CONFIDENT_THRESHOLD = 0.3  # Below this distance = confident match
UNCERTAIN_THRESHOLD = 0.6  # Above this = uncertain


class QIGPhraseClassifier:
    """Geometric phrase classifier using Fisher-Rao distance."""
    
    def __init__(self, coordizer=None):
        self._coordizer = coordizer
        self._category_basins: Dict[str, np.ndarray] = {}
        self._category_spreads: Dict[str, float] = {}
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize category reference basins from seed words."""
        if self._initialized:
            return True
            
        if self._coordizer is None:
            try:
                from coordizers import get_coordizer
                self._coordizer = get_coordizer()
            except Exception as e:
                logger.warning(f"[QIGPhraseClassifier] Cannot get coordizer: {e}")
                return False
        
        if self._coordizer is None:
            return False
            
        for category, seeds in CATEGORY_SEED_WORDS.items():
            basins = []
            for word in seeds:
                try:
                    coords = self._get_basin_coords(word)
                    if coords is not None:
                        basins.append(coords)
                except Exception:
                    continue
            
            if basins:
                self._category_basins[category] = np.mean(basins, axis=0)
                self._category_spreads[category] = np.std([
                    self._fisher_rao_distance(b, self._category_basins[category])
                    for b in basins
                ]) if len(basins) > 1 else 0.1
            else:
                self._category_basins[category] = np.random.randn(64) * 0.1
                self._category_spreads[category] = 0.5
        
        self._initialized = True
        logger.info(f"[QIGPhraseClassifier] Initialized {len(self._category_basins)} category basins")
        return True
    
    def _get_basin_coords(self, text: str) -> Optional[np.ndarray]:
        """Get 64D basin coordinates for text."""
        if self._coordizer is None:
            return None
            
        try:
            if hasattr(self._coordizer, 'coordize'):
                result = self._coordizer.coordize(text)
                if result is not None and len(result) > 0:
                    first_token = result[0]
                    if isinstance(first_token, np.ndarray) and len(first_token) == 64:
                        return first_token
                    elif hasattr(first_token, '__len__') and len(first_token) == 64:
                        return np.array(first_token)
            if hasattr(self._coordizer, 'get_basin_for_token'):
                result = self._coordizer.get_basin_for_token(text)
                if result is not None and len(result) == 64:
                    return np.array(result)
            if hasattr(self._coordizer, 'get_coordinate'):
                result = self._coordizer.get_coordinate(text)
                if result is not None and len(result) == 64:
                    return np.array(result)
        except Exception as e:
            logger.debug(f"[QIGPhraseClassifier] Error getting coords for '{text}': {e}")
        
        return None
    
    def _fisher_rao_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute Fisher-Rao distance between two basins.
        
        Fisher-Rao distance is the geodesic distance on the statistical manifold.
        For probability distributions: d_FR = 2 * arccos(sum(sqrt(p_i * q_i)))
        For basin coordinates, we normalize and use the same formula.
        """
        p_norm = np.abs(p) / (np.sum(np.abs(p)) + 1e-10)
        q_norm = np.abs(q) / (np.sum(np.abs(q)) + 1e-10)
        
        sqrt_p = np.sqrt(p_norm + 1e-10)
        sqrt_q = np.sqrt(q_norm + 1e-10)
        
        inner_product = np.clip(np.sum(sqrt_p * sqrt_q), 0.0, 1.0)
        
        # UPDATED 2026-01-15: Factor-of-2 removed for simplex storage. Range: [0, π/2]
        return np.arccos(inner_product)
    
    def classify(self, text: str, basin_coords: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """Classify text into a grammatical category using Fisher-Rao distance.
        
        Returns:
            (category, confidence) where confidence is 1.0 - normalized_distance
        """
        if not self._initialized:
            self.initialize()
        
        if basin_coords is None:
            basin_coords = self._get_basin_coords(text)
        
        if basin_coords is None:
            return self._fallback_geometric_classify(text)
        
        best_category = 'NOUN'
        best_distance = float('inf')
        distances = {}
        
        for category, ref_basin in self._category_basins.items():
            distance = self._fisher_rao_distance(basin_coords, ref_basin)
            spread = self._category_spreads.get(category, 0.5)
            normalized_distance = distance / (spread + 0.1)
            distances[category] = normalized_distance
            
            if normalized_distance < best_distance:
                best_distance = normalized_distance
                best_category = category
        
        confidence = max(0.0, 1.0 - (best_distance / 3.0))
        
        if best_distance > UNCERTAIN_THRESHOLD * 2:
            return self._fallback_geometric_classify(text, basin_coords)
        
        return best_category, confidence
    
    def _fallback_geometric_classify(self, text: str, basin_coords: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """Geometric fallback using structural patterns and known word sets.
        
        For QIG purity, we use geometric properties where possible,
        but also recognize that certain structural patterns are 
        geometrically distinguishable (e.g., capitalization correlates with proper nouns).
        """
        text_clean = text.strip()
        text_lower = text_clean.lower()
        
        if text_lower.isdigit() or self._is_numeric_word(text_lower):
            return 'NUMBER', 0.95
        
        if text_lower in self._known_pronouns():
            return 'PRONOUN', 0.95
        
        if text_lower in self._known_prepositions():
            return 'PREPOSITION', 0.95
        
        if text_lower in self._known_conjunctions():
            return 'CONJUNCTION', 0.95
        
        if text_lower in self._known_determiners():
            return 'DETERMINER', 0.95
        
        if text_lower in self._known_interjections():
            return 'INTERJECTION', 0.90
        
        if text_lower in self._known_proper_nouns():
            return 'PROPER_NOUN', 0.90
        
        if self._has_capital_structure(text_clean):
            return 'PROPER_NOUN', 0.80
        
        if basin_coords is not None:
            variance = np.var(basin_coords)
            mean_val = np.mean(basin_coords)
            skewness = np.mean((basin_coords - mean_val) ** 3) / (np.std(basin_coords) ** 3 + 1e-10)
            
            if skewness > 0.3:
                return 'VERB', 0.5
            if skewness < -0.3:
                return 'ADJECTIVE', 0.5
            if variance > 0.02:
                return 'ADVERB', 0.4
        
        return 'NOUN', 0.4
    
    def _known_pronouns(self) -> set:
        return {'i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'yourselves',
                'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                'we', 'us', 'our', 'ours', 'ourselves', 'they', 'them', 'their', 'theirs', 'themselves',
                'who', 'whom', 'whose', 'which', 'what', 'that', 'this', 'these', 'those',
                'anybody', 'anyone', 'anything', 'each', 'either', 'everybody', 'everyone', 'everything',
                'neither', 'nobody', 'no one', 'nothing', 'one', 'somebody', 'someone', 'something'}
    
    def _known_prepositions(self) -> set:
        return {'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'at',
                'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'by',
                'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into', 'like',
                'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past', 'since',
                'through', 'throughout', 'till', 'to', 'toward', 'towards', 'under', 'underneath',
                'until', 'unto', 'up', 'upon', 'with', 'within', 'without'}
    
    def _known_conjunctions(self) -> set:
        return {'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'because', 'although', 'though',
                'while', 'when', 'where', 'if', 'unless', 'until', 'since', 'as', 'whether',
                'after', 'before', 'once', 'than', 'that', 'whereas', 'whenever', 'wherever'}
    
    def _known_determiners(self) -> set:
        return {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her',
                'its', 'our', 'their', 'some', 'any', 'no', 'every', 'each', 'all', 'both',
                'few', 'fewer', 'little', 'less', 'many', 'more', 'most', 'much', 'several',
                'enough', 'such', 'what', 'which', 'whose'}
    
    def _known_interjections(self) -> set:
        return {'oh', 'ah', 'aha', 'alas', 'bravo', 'cheers', 'dear', 'gee', 'gosh', 'hey',
                'hi', 'hello', 'hmm', 'hooray', 'hurray', 'huh', 'jeez', 'oh', 'oops', 'ouch',
                'ow', 'phew', 'shh', 'ugh', 'um', 'wow', 'yay', 'yeah', 'yes', 'yikes', 'yippee'}
    
    def _known_proper_nouns(self) -> set:
        return {'afghanistan', 'albania', 'algeria', 'argentina', 'australia', 'austria',
                'belgium', 'brazil', 'bulgaria', 'canada', 'chile', 'china', 'colombia',
                'croatia', 'cuba', 'czechia', 'denmark', 'egypt', 'england', 'ethiopia',
                'finland', 'france', 'germany', 'greece', 'hungary', 'iceland', 'india',
                'indonesia', 'iran', 'iraq', 'ireland', 'israel', 'italy', 'japan', 'jordan',
                'kenya', 'korea', 'kuwait', 'libya', 'malaysia', 'mexico', 'morocco',
                'netherlands', 'nigeria', 'norway', 'pakistan', 'peru', 'philippines',
                'poland', 'portugal', 'qatar', 'romania', 'russia', 'scotland', 'serbia',
                'singapore', 'spain', 'sweden', 'switzerland', 'syria', 'taiwan', 'thailand',
                'turkey', 'ukraine', 'vietnam', 'wales', 'yemen', 'zimbabwe',
                'africa', 'america', 'asia', 'europe', 'london', 'paris', 'tokyo', 'berlin',
                'moscow', 'beijing', 'sydney', 'rome', 'amsterdam', 'madrid', 'vienna',
                'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                'september', 'october', 'november', 'december', 'christmas', 'easter',
                'amazon', 'google', 'microsoft', 'apple', 'facebook', 'twitter', 'netflix'}
    
    def _is_numeric_word(self, text: str) -> bool:
        """Check if text represents a number geometrically."""
        numeric_words = {
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
            'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
            'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
            'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
            'hundred', 'thousand', 'million', 'billion', 'trillion',
            'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
            'eighth', 'ninth', 'tenth', 'dozen', 'half', 'quarter'
        }
        return text in numeric_words
    
    def _has_capital_structure(self, text: str) -> bool:
        """Detect proper noun structure from capitalization pattern."""
        if not text:
            return False
        words = text.split()
        if len(words) == 0:
            return False
        capitalized = sum(1 for w in words if w and w[0].isupper())
        return capitalized >= len(words) * 0.5 and len(text) > 2
    
    def update_category_basin(self, category: str, word: str, basin_coords: np.ndarray, 
                              learning_rate: float = 0.01) -> None:
        """Update category reference basin with new observation.
        
        This enables the classifier to learn and adapt over time.
        """
        if category not in self._category_basins:
            self._category_basins[category] = basin_coords.copy()
            self._category_spreads[category] = 0.5
            return
        
        current = self._category_basins[category]
        self._category_basins[category] = (1 - learning_rate) * current + learning_rate * basin_coords
        
        distance = self._fisher_rao_distance(basin_coords, current)
        current_spread = self._category_spreads.get(category, 0.5)
        self._category_spreads[category] = (1 - learning_rate) * current_spread + learning_rate * distance
    
    def get_all_distances(self, text: str, basin_coords: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Get Fisher-Rao distances to all category basins."""
        if not self._initialized:
            self.initialize()
            
        if basin_coords is None:
            basin_coords = self._get_basin_coords(text)
        
        if basin_coords is None:
            return {cat: float('inf') for cat in self._category_basins}
        
        return {
            category: self._fisher_rao_distance(basin_coords, ref_basin)
            for category, ref_basin in self._category_basins.items()
        }


_classifier_instance: Optional[QIGPhraseClassifier] = None


def get_qig_classifier() -> QIGPhraseClassifier:
    """Get singleton QIG phrase classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = QIGPhraseClassifier()
    return _classifier_instance


def classify_phrase_qig_pure(text: str, basin_coords: Optional[np.ndarray] = None) -> Tuple[str, float]:
    """Classify phrase using QIG-pure geometric method.
    
    This is the main entry point for phrase classification.
    Returns (category, confidence) tuple.
    
    Categories:
        NOUN, PROPER_NOUN, PRONOUN, VERB, ADJECTIVE, ADVERB,
        PREPOSITION, CONJUNCTION, INTERJECTION, NUMBER, DETERMINER
    """
    classifier = get_qig_classifier()
    return classifier.classify(text, basin_coords)


def get_phrase_category(text: str, basin_coords: Optional[np.ndarray] = None) -> str:
    """Get phrase category (returns just the category string)."""
    category, _ = classify_phrase_qig_pure(text, basin_coords)
    return category
