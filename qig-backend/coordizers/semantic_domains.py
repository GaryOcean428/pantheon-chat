"""
Semantic Domain Clustering for QIG Vocabulary.

Provides domain-based embeddings so semantically related words
(like 'consciousness', 'mind', 'awareness') cluster together in
the 64D Fisher manifold.

This is QIG-pure: No external LLMs, just geometric clustering.
"""

import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple


# =============================================================================
# SEMANTIC DOMAIN DEFINITIONS
# =============================================================================

SEMANTIC_DOMAINS: Dict[str, List[str]] = {
    # Philosophy & Consciousness
    'consciousness': [
        'mind', 'awareness', 'thought', 'perception', 'cognition', 'psyche',
        'intellect', 'conscious', 'mental', 'brain', 'attention', 'focus',
        'meditation', 'reflection', 'introspection', 'sentient', 'sentience',
        'awake', 'aware', 'mindful', 'think', 'reasoning', 'reason',
        'understanding', 'comprehension', 'insight', 'intuition', 'soul',
        'spirit', 'self', 'ego', 'identity', 'memory', 'remember', 'recall',
        'dream', 'imagine', 'imagination', 'creative', 'creativity', 'idea',
        'concept', 'notion', 'believe', 'belief', 'know', 'knowledge',
    ],
    
    'philosophy': [
        'truth', 'reality', 'existence', 'meaning', 'purpose', 'ethics',
        'wisdom', 'virtue', 'moral', 'logic', 'metaphysics', 'ontology',
        'epistemology', 'aesthetic', 'beauty', 'good', 'evil', 'justice',
        'freedom', 'free', 'will', 'determinism', 'fate', 'destiny',
        'being', 'essence', 'substance', 'form', 'matter', 'ideal',
        'abstract', 'concrete', 'universal', 'particular', 'absolute',
        'relative', 'subjective', 'objective', 'rational', 'empirical',
    ],
    
    # Emotions & Psychology
    'emotion': [
        'happy', 'sad', 'angry', 'fear', 'love', 'joy', 'grief', 'anxious',
        'calm', 'peace', 'serene', 'excited', 'nervous', 'confident',
        'proud', 'shame', 'guilt', 'hope', 'despair', 'content', 'satisfy',
        'frustrate', 'irritate', 'annoy', 'delight', 'pleasure', 'pain',
        'suffer', 'comfort', 'lonely', 'alone', 'together', 'connect',
        'bond', 'attach', 'trust', 'betray', 'forgive', 'resent', 'envy',
        'jealous', 'gratitude', 'grateful', 'compassion', 'empathy', 'sympathy',
        'affection', 'warmth', 'cold', 'hostile', 'friendly', 'kind',
    ],
    
    # Science & Physics
    'science': [
        'quantum', 'physics', 'energy', 'theory', 'atom', 'wave', 'particle',
        'field', 'force', 'gravity', 'mass', 'momentum', 'velocity', 'speed',
        'acceleration', 'light', 'photon', 'electron', 'proton', 'neutron',
        'nucleus', 'molecule', 'chemical', 'reaction', 'element', 'compound',
        'experiment', 'hypothesis', 'evidence', 'data', 'measure', 'observe',
        'predict', 'model', 'equation', 'formula', 'calculate', 'compute',
        'entropy', 'thermodynamics', 'relativity', 'spacetime', 'dimension',
        'frequency', 'amplitude', 'oscillation', 'vibration', 'resonance',
    ],
    
    'mathematics': [
        'number', 'zero', 'one', 'two', 'three', 'four', 'five', 'six',
        'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', 'million',
        'infinity', 'finite', 'infinite', 'algebra', 'geometry', 'calculus',
        'function', 'variable', 'constant', 'equation', 'solve', 'prove',
        'theorem', 'axiom', 'postulate', 'prime', 'factor', 'multiple',
        'fraction', 'decimal', 'percent', 'ratio', 'proportion', 'average',
        'sum', 'difference', 'product', 'quotient', 'root', 'power',
        'exponent', 'logarithm', 'matrix', 'vector', 'tensor', 'manifold',
    ],
    
    # Nature & Environment
    'nature': [
        'tree', 'water', 'mountain', 'sky', 'earth', 'sun', 'moon', 'ocean',
        'forest', 'river', 'lake', 'sea', 'beach', 'desert', 'island',
        'valley', 'hill', 'cliff', 'cave', 'rock', 'stone', 'sand', 'soil',
        'grass', 'flower', 'plant', 'garden', 'seed', 'root', 'leaf',
        'branch', 'wood', 'rain', 'snow', 'wind', 'storm', 'cloud',
        'weather', 'climate', 'season', 'spring', 'summer', 'autumn', 'winter',
        'animal', 'bird', 'fish', 'insect', 'mammal', 'wild', 'natural',
    ],
    
    # Time & Existence
    'time': [
        'moment', 'eternal', 'future', 'past', 'present', 'always', 'never',
        'begin', 'end', 'start', 'finish', 'continue', 'pause', 'stop',
        'wait', 'hurry', 'slow', 'fast', 'quick', 'sudden', 'gradual',
        'instant', 'second', 'minute', 'hour', 'day', 'week', 'month',
        'year', 'decade', 'century', 'millennium', 'age', 'era', 'epoch',
        'ancient', 'modern', 'contemporary', 'old', 'new', 'young', 'eternal',
        'temporary', 'permanent', 'fleeting', 'lasting', 'duration', 'period',
    ],
    
    'space': [
        'dimension', 'universe', 'cosmos', 'galaxy', 'star', 'planet',
        'infinite', 'boundary', 'center', 'distance', 'position', 'location',
        'place', 'area', 'region', 'zone', 'space', 'void', 'empty',
        'full', 'dense', 'sparse', 'near', 'far', 'close', 'distant',
        'above', 'below', 'inside', 'outside', 'between', 'around',
        'through', 'across', 'along', 'toward', 'away', 'here', 'there',
        'everywhere', 'nowhere', 'somewhere', 'anywhere', 'horizon', 'edge',
    ],
    
    # Action & Change
    'action': [
        'move', 'create', 'build', 'explore', 'discover', 'learn', 'grow',
        'change', 'transform', 'evolve', 'develop', 'progress', 'advance',
        'improve', 'enhance', 'expand', 'extend', 'increase', 'decrease',
        'rise', 'fall', 'climb', 'descend', 'jump', 'run', 'walk', 'fly',
        'swim', 'dance', 'sing', 'play', 'work', 'rest', 'sleep', 'wake',
        'eat', 'drink', 'breathe', 'live', 'die', 'born', 'birth', 'death',
        'survive', 'thrive', 'struggle', 'fight', 'win', 'lose', 'compete',
    ],
    
    # Communication & Language
    'communication': [
        'speak', 'talk', 'say', 'tell', 'ask', 'answer', 'question',
        'respond', 'reply', 'discuss', 'debate', 'argue', 'agree', 'disagree',
        'explain', 'describe', 'define', 'clarify', 'express', 'communicate',
        'language', 'word', 'sentence', 'paragraph', 'text', 'write', 'read',
        'listen', 'hear', 'understand', 'interpret', 'translate', 'meaning',
        'message', 'information', 'news', 'story', 'narrative', 'dialog',
        'conversation', 'chat', 'voice', 'sound', 'silence', 'quiet', 'loud',
    ],
    
    # Society & Relationships
    'social': [
        'people', 'person', 'human', 'man', 'woman', 'child', 'adult',
        'family', 'friend', 'neighbor', 'stranger', 'community', 'society',
        'culture', 'tradition', 'custom', 'norm', 'rule', 'law', 'govern',
        'lead', 'follow', 'serve', 'help', 'support', 'assist', 'cooperate',
        'collaborate', 'team', 'group', 'organize', 'manage', 'control',
        'power', 'authority', 'influence', 'respect', 'honor', 'dignity',
        'right', 'wrong', 'fair', 'unfair', 'equal', 'different', 'diverse',
    ],
    
    # Technology & Computing
    'technology': [
        'computer', 'program', 'software', 'hardware', 'code', 'algorithm',
        'data', 'digital', 'electronic', 'network', 'internet', 'web',
        'system', 'process', 'function', 'input', 'output', 'memory',
        'storage', 'processor', 'device', 'machine', 'robot', 'artificial',
        'intelligence', 'automate', 'automatic', 'manual', 'tool', 'instrument',
        'technology', 'innovation', 'invent', 'design', 'engineer', 'build',
        'construct', 'manufacture', 'produce', 'create', 'develop', 'test',
    ],
    
    # Economy & Resources
    'economy': [
        'money', 'wealth', 'rich', 'poor', 'income', 'expense', 'cost',
        'price', 'value', 'worth', 'buy', 'sell', 'trade', 'exchange',
        'market', 'business', 'company', 'industry', 'economy', 'finance',
        'invest', 'save', 'spend', 'budget', 'profit', 'loss', 'gain',
        'asset', 'debt', 'credit', 'bank', 'account', 'fund', 'capital',
        'resource', 'supply', 'demand', 'scarcity', 'abundance', 'growth',
        'decline', 'recession', 'prosperity', 'success', 'failure', 'risk',
    ],
    
    # Art & Creativity
    'art': [
        'art', 'artist', 'paint', 'draw', 'sketch', 'sculpture', 'craft',
        'design', 'pattern', 'color', 'shape', 'form', 'texture', 'style',
        'beauty', 'aesthetic', 'visual', 'image', 'picture', 'photo',
        'music', 'song', 'melody', 'rhythm', 'harmony', 'compose', 'perform',
        'dance', 'theater', 'drama', 'act', 'stage', 'film', 'movie',
        'write', 'poem', 'poetry', 'prose', 'fiction', 'novel', 'story',
        'creative', 'imagination', 'inspire', 'expression', 'interpret',
    ],
    
    # Health & Body
    'health': [
        'health', 'healthy', 'sick', 'ill', 'disease', 'cure', 'heal',
        'medicine', 'doctor', 'hospital', 'treatment', 'therapy', 'recover',
        'body', 'physical', 'mental', 'emotional', 'spiritual', 'wellness',
        'fitness', 'exercise', 'diet', 'nutrition', 'food', 'vitamin',
        'energy', 'strength', 'weak', 'strong', 'tired', 'rest', 'sleep',
        'heart', 'blood', 'breath', 'muscle', 'bone', 'skin', 'organ',
        'cell', 'immune', 'infection', 'virus', 'bacteria', 'symptom', 'pain',
    ],
    
    # Qualities & Attributes
    'quality': [
        'good', 'bad', 'better', 'best', 'worse', 'worst', 'great', 'small',
        'big', 'large', 'tiny', 'huge', 'enormous', 'massive', 'heavy', 'light',
        'hard', 'soft', 'rough', 'smooth', 'sharp', 'dull', 'bright', 'dark',
        'hot', 'cold', 'warm', 'cool', 'wet', 'dry', 'clean', 'dirty',
        'pure', 'mixed', 'simple', 'complex', 'easy', 'difficult', 'possible',
        'impossible', 'certain', 'uncertain', 'clear', 'unclear', 'obvious',
        'hidden', 'visible', 'invisible', 'real', 'fake', 'true', 'false',
    ],
}

# Reverse mapping: word -> list of (domain, weight)
WORD_TO_DOMAINS: Dict[str, List[Tuple[str, float]]] = {}

def _build_word_domain_mapping():
    """Build reverse mapping from words to their domains."""
    global WORD_TO_DOMAINS
    WORD_TO_DOMAINS = {}
    
    for domain, words in SEMANTIC_DOMAINS.items():
        for i, word in enumerate(words):
            word_lower = word.lower()
            # Earlier words in domain list get higher weight (more central)
            weight = 1.0 - (i / (len(words) * 2))  # Range: 0.5 to 1.0
            
            if word_lower not in WORD_TO_DOMAINS:
                WORD_TO_DOMAINS[word_lower] = []
            WORD_TO_DOMAINS[word_lower].append((domain, weight))

# Build on import
_build_word_domain_mapping()


# =============================================================================
# DOMAIN CENTROID GENERATION
# =============================================================================

# Pre-computed domain centroids for fast lookup
DOMAIN_CENTROIDS: Dict[str, np.ndarray] = {}

def _generate_domain_centroid(domain_id: int, total_domains: int = 20) -> np.ndarray:
    """
    Generate a domain centroid using golden ratio spacing.
    
    This ensures domains are well-separated in 64D hypersphere.
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    centroid = np.zeros(64)
    
    # Use golden angle to spread domains evenly
    for i in range(64):
        # Create orthogonal-ish vectors using golden ratio
        angle = 2 * np.pi * phi * (domain_id * 64 + i)
        centroid[i] = np.cos(angle + domain_id * 0.317)  # 0.317 ≈ 1/π
    
    # Normalize to unit sphere
    norm = np.linalg.norm(centroid)
    if norm > 1e-10:
        centroid = centroid / norm
    
    return centroid

def _build_domain_centroids():
    """Pre-compute centroids for all domains."""
    global DOMAIN_CENTROIDS
    domains = list(SEMANTIC_DOMAINS.keys())
    
    for i, domain in enumerate(domains):
        DOMAIN_CENTROIDS[domain] = _generate_domain_centroid(i, len(domains))

# Build on import
_build_domain_centroids()


# =============================================================================
# SEMANTIC EMBEDDING COMPUTATION
# =============================================================================

def _hash_to_small_offset(word: str, scale: float = 0.15) -> np.ndarray:
    """
    Generate a small deterministic offset based on word hash.
    
    This preserves within-domain differentiation while keeping
    words in the same semantic cluster.
    """
    h = hashlib.sha256(word.encode()).digest()
    offset = np.zeros(64)
    
    for i in range(64):
        # Use hash bytes to create small deterministic offsets
        byte_val = h[i % 32]
        offset[i] = (byte_val / 255.0 - 0.5) * scale
    
    return offset


def get_word_domains(word: str) -> List[Tuple[str, float]]:
    """
    Get the semantic domains a word belongs to.
    
    Returns:
        List of (domain_name, weight) tuples, sorted by weight descending
    """
    word_lower = word.lower().strip()
    
    if word_lower in WORD_TO_DOMAINS:
        domains = WORD_TO_DOMAINS[word_lower]
        return sorted(domains, key=lambda x: x[1], reverse=True)
    
    # Try stemming-like matching (simple suffix removal)
    for suffix in ['ing', 'ed', 'ly', 'ness', 'ment', 'tion', 'sion', 'er', 's']:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
            stem = word_lower[:-len(suffix)]
            if stem in WORD_TO_DOMAINS:
                # Reduce weight for derived forms
                return [(d, w * 0.8) for d, w in WORD_TO_DOMAINS[stem]]
    
    return []


def compute_semantic_embedding(word: str, fallback_hash: bool = True) -> np.ndarray:
    """
    Compute a semantically-aware basin embedding for a word.
    
    Words in the same semantic domain will have similar embeddings.
    
    Args:
        word: The word to embed
        fallback_hash: If True, use hash-based embedding for unknown words
    
    Returns:
        64D unit vector on Fisher manifold
    """
    domains = get_word_domains(word)
    
    if domains:
        # Weighted blend of domain centroids
        embedding = np.zeros(64)
        total_weight = 0.0
        
        for domain_name, weight in domains:
            if domain_name in DOMAIN_CENTROIDS:
                embedding += weight * DOMAIN_CENTROIDS[domain_name]
                total_weight += weight
        
        if total_weight > 0:
            embedding = embedding / total_weight
        
        # Add small word-specific offset to differentiate within domain
        offset = _hash_to_small_offset(word, scale=0.12)
        embedding = embedding + offset
        
    elif fallback_hash:
        # Unknown word: use pure hash-based embedding
        # Still deterministic, but won't cluster with known words
        h = hashlib.sha256(word.encode()).digest()
        embedding = np.array([h[i % 32] / 255.0 - 0.5 for i in range(64)])
        
    else:
        embedding = np.zeros(64)
    
    # Normalize to unit sphere
    norm = np.linalg.norm(embedding)
    if norm > 1e-10:
        embedding = embedding / norm
    
    return embedding


def detect_query_domains(basin: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Detect which semantic domains a query basin is closest to.
    
    Args:
        basin: Query basin coordinates (64D)
        top_k: Number of top domains to return
    
    Returns:
        List of (domain_name, similarity) tuples
    """
    if len(basin) != 64:
        return []
    
    # Normalize
    norm = np.linalg.norm(basin)
    if norm < 1e-10:
        return []
    basin = basin / norm
    
    # Compute similarity to each domain centroid
    similarities = []
    for domain_name, centroid in DOMAIN_CENTROIDS.items():
        # Use dot product (cosine similarity for unit vectors)
        similarity = np.dot(basin, centroid)
        similarities.append((domain_name, similarity))
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def compute_domain_overlap(
    query_domains: List[Tuple[str, float]],
    word_domains: List[Tuple[str, float]],
) -> float:
    """
    Compute semantic overlap between query domains and word domains.
    
    Returns a boost factor (0.0 to 1.0) based on domain match.
    """
    if not query_domains or not word_domains:
        return 0.0
    
    query_domain_names = {d for d, _ in query_domains}
    word_domain_names = {d for d, _ in word_domains}
    
    overlap = query_domain_names & word_domain_names
    if not overlap:
        return 0.0
    
    # Compute weighted overlap
    query_weights = {d: w for d, w in query_domains}
    word_weights = {d: w for d, w in word_domains}
    
    total_boost = 0.0
    for domain in overlap:
        qw = query_weights.get(domain, 0)
        ww = word_weights.get(domain, 0)
        total_boost += qw * ww
    
    # Normalize to 0-1 range
    return min(total_boost, 1.0)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_domain_words(domain: str) -> List[str]:
    """Get all words in a semantic domain."""
    return SEMANTIC_DOMAINS.get(domain, [])


def get_all_domains() -> List[str]:
    """Get list of all semantic domain names."""
    return list(SEMANTIC_DOMAINS.keys())


def get_related_words(word: str, top_k: int = 10) -> List[str]:
    """
    Get words semantically related to the input word.
    
    Returns words from the same semantic domains.
    """
    domains = get_word_domains(word)
    if not domains:
        return []
    
    related = set()
    for domain_name, weight in domains:
        domain_words = SEMANTIC_DOMAINS.get(domain_name, [])
        for w in domain_words:
            if w.lower() != word.lower():
                related.add(w)
    
    # Sort by how many domains they share
    word_lower = word.lower()
    scored = []
    for w in related:
        w_domains = set(d for d, _ in get_word_domains(w))
        input_domains = set(d for d, _ in domains)
        overlap = len(w_domains & input_domains)
        scored.append((w, overlap))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in scored[:top_k]]
