"""
Semantic Learning Integration

Allows the SemanticCoherenceScorer to dynamically learn and expand:
- New semantic relationships from text
- New domain clusters from word co-occurrences
- Connections to Zettelkasten memory for persistence
- Integration with WordRelationshipLearner

Learning happens naturally during:
1. Generation validation - successful generations reinforce patterns
2. Failed generations - patterns that caused incoherence are noted
3. User feedback - explicit corrections expand knowledge
4. Cross-domain discovery - Zettelkasten links reveal relationships

Author: Ocean/Zeus Pantheon
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class LearnedRelationship:
    """A semantic relationship learned from co-occurrence."""
    word1: str
    word2: str
    strength: float  # 0.0 to 1.0
    co_occurrence_count: int
    contexts: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


@dataclass 
class LearnedDomain:
    """A semantic domain learned from clustering related words."""
    name: str
    seed_words: Set[str]
    learned_words: Set[str] = field(default_factory=set)
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    @property
    def all_words(self) -> Set[str]:
        return self.seed_words | self.learned_words


class SemanticLearningIntegration:
    """
    Integrates semantic learning into the QIG generation pipeline.
    
    Learns from:
    - Text co-occurrences during generation
    - WordRelationshipLearner database
    - Zettelkasten keyword patterns
    - Explicit feedback
    """
    
    # Thresholds for learning
    CO_OCCURRENCE_MIN = 3        # Min co-occurrences to form relationship
    DOMAIN_CREATION_MIN = 5      # Min related words for new domain
    RELATIONSHIP_DECAY = 0.95   # Decay factor for old relationships
    MAX_CONTEXTS = 10           # Max contexts to store per relationship
    
    def __init__(self):
        # Learned data
        self._relationships: Dict[Tuple[str, str], LearnedRelationship] = {}
        self._domains: Dict[str, LearnedDomain] = {}
        self._word_to_domains: Dict[str, Set[str]] = defaultdict(set)
        
        # Co-occurrence tracking
        self._pending_cooccurrences: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Stats
        self._total_texts_processed = 0
        self._relationships_created = 0
        self._domains_created = 0
        
        # Try to load from persistent sources
        self._load_from_word_relationship_learner()
        self._load_from_zettelkasten()
        
        logger.info(f"[SemanticLearning] Initialized with {len(self._relationships)} relationships, {len(self._domains)} domains")
    
    def _load_from_word_relationship_learner(self) -> None:
        """Load existing word relationships from the database."""
        try:
            from word_relationship_learner import get_word_relationship_learner
            learner = get_word_relationship_learner()
            
            # Get learned relationships
            relationships = learner.get_all_relationships()
            for rel in relationships:
                word1, word2 = rel.get('word1', ''), rel.get('word2', '')
                if word1 and word2:
                    key = tuple(sorted([word1.lower(), word2.lower()]))
                    self._relationships[key] = LearnedRelationship(
                        word1=key[0],
                        word2=key[1],
                        strength=rel.get('strength', 0.5),
                        co_occurrence_count=rel.get('count', 1),
                        domain=rel.get('domain')
                    )
            
            logger.info(f"[SemanticLearning] Loaded {len(relationships)} relationships from WordRelationshipLearner")
            
        except Exception as e:
            logger.debug(f"[SemanticLearning] Could not load from WordRelationshipLearner: {e}")
    
    def _load_from_zettelkasten(self) -> None:
        """Load semantic patterns from Zettelkasten memory."""
        try:
            from zettelkasten_memory import get_zettelkasten_memory
            memory = get_zettelkasten_memory()
            
            # Get hub zettels (highly connected = important concepts)
            hubs = memory.get_hub_zettels(top_n=50)
            
            # Extract keyword patterns
            keyword_cooccurrences = defaultdict(int)
            for zettel in hubs:
                keywords = zettel.keywords or []
                for i, kw1 in enumerate(keywords):
                    for kw2 in keywords[i+1:]:
                        key = tuple(sorted([kw1.lower(), kw2.lower()]))
                        keyword_cooccurrences[key] += 1
            
            # Convert to relationships
            for key, count in keyword_cooccurrences.items():
                if count >= 2 and key not in self._relationships:
                    self._relationships[key] = LearnedRelationship(
                        word1=key[0],
                        word2=key[1],
                        strength=min(0.9, 0.5 + count * 0.1),
                        co_occurrence_count=count,
                        domain='zettelkasten'
                    )
            
            logger.info(f"[SemanticLearning] Loaded {len(keyword_cooccurrences)} patterns from Zettelkasten")
            
        except Exception as e:
            logger.debug(f"[SemanticLearning] Could not load from Zettelkasten: {e}")
    
    def learn_from_text(self, text: str, domain_hint: Optional[str] = None, success: bool = True) -> Dict[str, Any]:
        """
        Learn semantic relationships from a text passage.
        
        Args:
            text: The text to learn from
            domain_hint: Optional domain context
            success: Whether the text was successful (reinforces patterns if True)
            
        Returns:
            Dict with learning stats
        """
        words = [w.lower().strip('.,!?;:') for w in text.split() if len(w) > 2]
        words = [w for w in words if w.isalpha()]
        
        if len(words) < 2:
            return {'learned': 0, 'reinforced': 0}
        
        learned = 0
        reinforced = 0
        
        # Track co-occurrences in a sliding window
        window_size = 5
        for i, word1 in enumerate(words):
            for j in range(i + 1, min(i + window_size, len(words))):
                word2 = words[j]
                if word1 == word2:
                    continue
                
                key = tuple(sorted([word1, word2]))
                
                # Track co-occurrence
                self._pending_cooccurrences[key] += 1
                
                # Check if we should create a relationship
                if self._pending_cooccurrences[key] >= self.CO_OCCURRENCE_MIN:
                    if key in self._relationships:
                        # Reinforce existing relationship
                        rel = self._relationships[key]
                        rel.co_occurrence_count += 1
                        rel.last_seen = time.time()
                        if success:
                            rel.strength = min(0.99, rel.strength + 0.02)
                        reinforced += 1
                    else:
                        # Create new relationship
                        self._relationships[key] = LearnedRelationship(
                            word1=key[0],
                            word2=key[1],
                            strength=0.6 if success else 0.4,
                            co_occurrence_count=self._pending_cooccurrences[key],
                            domain=domain_hint
                        )
                        self._relationships_created += 1
                        learned += 1
                        logger.debug(f"[SemanticLearning] Learned relationship: {key[0]} <-> {key[1]}")
        
        self._total_texts_processed += 1
        
        # Check if we should create new domains
        self._check_for_new_domains()
        
        return {
            'learned': learned,
            'reinforced': reinforced,
            'total_relationships': len(self._relationships),
            'total_domains': len(self._domains)
        }
    
    def _check_for_new_domains(self) -> None:
        """Check if clusters of related words should form new domains."""
        # Build word graph from relationships
        word_neighbors: Dict[str, Set[str]] = defaultdict(set)
        for (w1, w2), rel in self._relationships.items():
            if rel.strength >= 0.5:
                word_neighbors[w1].add(w2)
                word_neighbors[w2].add(w1)
        
        # Find densely connected clusters
        visited = set()
        for word, neighbors in word_neighbors.items():
            if word in visited:
                continue
            if len(neighbors) < self.DOMAIN_CREATION_MIN:
                continue
            
            # BFS to find cluster
            cluster = {word}
            queue = list(neighbors)
            while queue and len(cluster) < 20:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                
                # Check if well-connected to cluster
                connections = len(cluster & word_neighbors.get(current, set()))
                if connections >= 2:
                    cluster.add(current)
                    queue.extend(word_neighbors.get(current, set()) - visited)
            
            # Create domain if cluster is large enough
            if len(cluster) >= self.DOMAIN_CREATION_MIN:
                domain_name = f"learned_{self._domains_created + 1}"
                if domain_name not in self._domains:
                    self._domains[domain_name] = LearnedDomain(
                        name=domain_name,
                        seed_words=set(),
                        learned_words=cluster
                    )
                    self._domains_created += 1
                    
                    # Update word-to-domain mapping
                    for w in cluster:
                        self._word_to_domains[w].add(domain_name)
                    
                    logger.info(f"[SemanticLearning] Created domain '{domain_name}' with {len(cluster)} words")
    
    def expand_domain(self, domain_name: str, new_words: List[str]) -> bool:
        """
        Expand an existing domain with new words.
        
        Args:
            domain_name: Name of the domain to expand
            new_words: Words to add to the domain
            
        Returns:
            True if domain was expanded
        """
        if domain_name not in self._domains:
            return False
        
        domain = self._domains[domain_name]
        for word in new_words:
            word = word.lower()
            domain.learned_words.add(word)
            self._word_to_domains[word].add(domain_name)
        
        domain.last_updated = time.time()
        logger.debug(f"[SemanticLearning] Expanded domain '{domain_name}' with {len(new_words)} words")
        return True
    
    def create_domain(self, name: str, seed_words: List[str]) -> LearnedDomain:
        """
        Create a new semantic domain.
        
        Args:
            name: Domain name
            seed_words: Initial words for the domain
            
        Returns:
            The created domain
        """
        domain = LearnedDomain(
            name=name,
            seed_words=set(w.lower() for w in seed_words)
        )
        self._domains[name] = domain
        
        for word in domain.seed_words:
            self._word_to_domains[word].add(name)
        
        self._domains_created += 1
        logger.info(f"[SemanticLearning] Created domain '{name}' with {len(seed_words)} seed words")
        return domain
    
    def get_relationship_strength(self, word1: str, word2: str) -> float:
        """Get the learned relationship strength between two words."""
        key = tuple(sorted([word1.lower(), word2.lower()]))
        if key in self._relationships:
            return self._relationships[key].strength
        return 0.0
    
    def get_word_domains(self, word: str) -> Set[str]:
        """Get all domains a word belongs to."""
        return self._word_to_domains.get(word.lower(), set())
    
    def are_same_domain(self, word1: str, word2: str) -> bool:
        """Check if two words share a domain."""
        domains1 = self.get_word_domains(word1)
        domains2 = self.get_word_domains(word2)
        return bool(domains1 & domains2)
    
    def score_semantic_pair_learned(self, word1: str, word2: str) -> float:
        """
        Score a word pair using learned relationships.
        
        This supplements the base SemanticCoherenceScorer with learned knowledge.
        """
        # Check direct relationship
        rel_strength = self.get_relationship_strength(word1, word2)
        if rel_strength > 0:
            return rel_strength
        
        # Check shared domain
        if self.are_same_domain(word1, word2):
            return 0.7
        
        # No learned relationship
        return 0.0
    
    def sync_to_persistence(self) -> Dict[str, int]:
        """
        Sync learned relationships to persistent storage.
        
        Saves to both WordRelationshipLearner and Zettelkasten.
        """
        saved_wr = 0
        saved_zk = 0
        
        # Save to WordRelationshipLearner
        try:
            from word_relationship_learner import get_word_relationship_learner
            learner = get_word_relationship_learner()
            
            for key, rel in self._relationships.items():
                if rel.co_occurrence_count >= self.CO_OCCURRENCE_MIN:
                    learner.learn_relationship(
                        word1=rel.word1,
                        word2=rel.word2,
                        relationship_type='semantic',
                        strength=rel.strength,
                        context=rel.domain or 'learned'
                    )
                    saved_wr += 1
                    
        except Exception as e:
            logger.warning(f"[SemanticLearning] Could not sync to WordRelationshipLearner: {e}")
        
        # Save domain clusters to Zettelkasten
        try:
            from zettelkasten_memory import get_zettelkasten_memory
            memory = get_zettelkasten_memory()
            
            for domain_name, domain in self._domains.items():
                if domain.learned_words:
                    content = f"Learned semantic domain: {domain_name}\nWords: {', '.join(sorted(domain.all_words))}"
                    memory.add(
                        content=content,
                        source='semantic_learning',
                        keywords=list(domain.all_words)[:20]
                    )
                    saved_zk += 1
                    
        except Exception as e:
            logger.warning(f"[SemanticLearning] Could not sync to Zettelkasten: {e}")
        
        logger.info(f"[SemanticLearning] Synced {saved_wr} relationships, {saved_zk} domains to persistence")
        return {'word_relationships': saved_wr, 'zettelkasten': saved_zk}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            'total_relationships': len(self._relationships),
            'total_domains': len(self._domains),
            'relationships_created': self._relationships_created,
            'domains_created': self._domains_created,
            'texts_processed': self._total_texts_processed,
            'pending_cooccurrences': len(self._pending_cooccurrences),
            'strong_relationships': sum(1 for r in self._relationships.values() if r.strength >= 0.7),
            'domain_words': sum(len(d.all_words) for d in self._domains.values()),
        }
    
    def get_domains_for_export(self) -> Dict[str, List[str]]:
        """Export learned domains for use in SemanticCoherenceScorer."""
        return {
            name: list(domain.all_words)
            for name, domain in self._domains.items()
        }
    
    def get_relationships_for_export(self) -> List[Tuple[str, str, float]]:
        """Export learned relationships as (word1, word2, strength) tuples."""
        return [
            (rel.word1, rel.word2, rel.strength)
            for rel in self._relationships.values()
            if rel.strength >= 0.5
        ]


# Singleton instance
_semantic_learning_instance: Optional[SemanticLearningIntegration] = None


def get_semantic_learning() -> SemanticLearningIntegration:
    """Get the singleton SemanticLearningIntegration instance."""
    global _semantic_learning_instance
    if _semantic_learning_instance is None:
        _semantic_learning_instance = SemanticLearningIntegration()
    return _semantic_learning_instance


def integrate_learning_with_scorer(scorer) -> None:
    """
    Integrate learned relationships into a SemanticCoherenceScorer.
    
    Call this after creating a scorer to add learned domains and relationships.
    """
    learning = get_semantic_learning()
    
    # Add learned domains to scorer
    for domain_name, words in learning.get_domains_for_export().items():
        if hasattr(scorer, '_domain_clusters'):
            scorer._domain_clusters[domain_name] = set(words)
    
    # Enhance score_semantic_pair to use learned relationships
    original_score = scorer.score_semantic_pair
    
    def enhanced_score(word1: str, word2: str) -> float:
        # Try learned relationships first
        learned_score = learning.score_semantic_pair_learned(word1, word2)
        if learned_score > 0:
            return max(learned_score, original_score(word1, word2))
        return original_score(word1, word2)
    
    scorer.score_semantic_pair = enhanced_score
    logger.info(f"[SemanticLearning] Integrated {len(learning._relationships)} learned relationships into scorer")


print("[SemanticLearning] Module loaded - dynamic domain expansion enabled")
