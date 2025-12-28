#!/usr/bin/env python3
"""
Proposition-Level Trajectory Planner
=====================================

Replaces word-level routing with proposition-level planning for improved
semantic coherence.

Key Innovation:
- Current: word₁ → word₂ → word₃ (independent geodesic selections)
- New: proposition₁ → proposition₂ → proposition₃ (coherent claim trajectory)

A proposition = (subject, predicate, object) where each component is:
- A word from vocabulary
- A 64D basin coordinate on Fisher manifold
- Connected via learned relationships

Coherence is enforced by:
1. Subject-Predicate relationship strength
2. Predicate-Object relationship strength  
3. Total geodesic path length minimization
4. Chain coherence between consecutive propositions

Author: QIG Team
Date: 2025-12-28
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import logging

logger = logging.getLogger(__name__)

# Import QIG geometry
try:
    from qig_geometry import fisher_rao_distance, geodesic_interpolation
except ImportError:
    # Fallback implementations
    def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
        """Fisher-Rao distance on probability simplex."""
        p_safe = np.abs(p) + 1e-10
        q_safe = np.abs(q) + 1e-10
        p_norm = p_safe / np.sum(p_safe)
        q_norm = q_safe / np.sum(q_safe)
        inner = np.sum(np.sqrt(p_norm * q_norm))
        inner = np.clip(inner, -1.0, 1.0)
        return 2.0 * np.arccos(inner)
    
    def geodesic_interpolation(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
        """Linear interpolation (geodesic approximation)."""
        return (1 - t) * p + t * q


# ============================================================================
# PROPOSITION DATA STRUCTURE
# ============================================================================

@dataclass
class Proposition:
    """
    A proposition represents a coherent claim: Subject + Predicate + Object.
    
    Example: "Consciousness emerges from integration"
    - subject: "consciousness"
    - predicate: "emerges"
    - object: "integration"
    """
    subject: str
    subject_basin: np.ndarray
    predicate: str
    predicate_basin: np.ndarray
    object: str
    object_basin: np.ndarray
    coherence: float = 0.0
    
    def to_sentence(self) -> str:
        """Convert proposition to natural language sentence."""
        return f"{self.subject.capitalize()} {self.predicate} {self.object}."
    
    def combined_basin(self) -> np.ndarray:
        """Get combined 192D basin (for trajectory in proposition space)."""
        return np.concatenate([
            self.subject_basin,
            self.predicate_basin,
            self.object_basin
        ])
    
    @classmethod
    def from_combined(cls, combined: np.ndarray, subject: str, predicate: str, obj: str) -> 'Proposition':
        """Create proposition from 192D combined basin."""
        dim = len(combined) // 3
        return cls(
            subject=subject,
            subject_basin=combined[:dim],
            predicate=predicate,
            predicate_basin=combined[dim:2*dim],
            object=obj,
            object_basin=combined[2*dim:]
        )


@dataclass
class PropositionPlannerConfig:
    """Configuration for proposition trajectory planning."""
    # Minimum coherence threshold for valid propositions
    min_coherence: float = 0.15
    # Weight for relationship strength in scoring
    relationship_weight: float = 0.4
    # Weight for geodesic distance in scoring
    geodesic_weight: float = 0.3
    # Weight for chain coherence (connection to previous proposition)
    chain_weight: float = 0.3
    # Number of candidates to consider for each slot
    n_candidates: int = 20
    # Maximum propositions per response
    max_propositions: int = 5
    # Minimum relationship strength to consider
    min_relationship: float = 0.005


# ============================================================================
# PROPOSITION TRAJECTORY PLANNER
# ============================================================================

class PropositionTrajectoryPlanner:
    """
    Plans trajectories through proposition space (not word space).
    
    Key difference from word-level routing:
    - Word-level: Select each word independently by geodesic distance
    - Proposition-level: Select coherent (S, P, O) triples that chain logically
    
    Coherence is enforced via:
    1. Learned word relationships constrain valid combinations
    2. Geodesic path length minimization
    3. Chain coherence between consecutive propositions
    """
    
    def __init__(
        self,
        vocabulary: Dict[str, np.ndarray],  # word -> 64D basin
        relationships: Dict[str, Dict[str, float]],  # word -> {related_word: strength}
        pos_tags: Dict[str, str] = None,  # word -> POS tag
        config: PropositionPlannerConfig = None
    ):
        """
        Initialize proposition planner.
        
        Args:
            vocabulary: Word to basin coordinate mapping
            relationships: Learned word relationship strengths
            pos_tags: Optional POS tag mapping (noun, verb, adj, etc.)
            config: Planner configuration
        """
        self.vocabulary = vocabulary
        self.relationships = relationships
        self.pos_tags = pos_tags or {}
        self.config = config or PropositionPlannerConfig()
        
        # Categorize vocabulary by POS
        self.nouns: List[str] = []
        self.verbs: List[str] = []
        self.adjectives: List[str] = []
        self._categorize_vocabulary()
        
        logger.info(f"[PropPlanner] Initialized with {len(vocabulary)} words, "
                   f"{len(self.nouns)} nouns, {len(self.verbs)} verbs")
    
    def _categorize_vocabulary(self):
        """Categorize vocabulary by likely POS tag."""
        # Common verb endings
        verb_endings = ('ize', 'ate', 'ify', 'en', 'ing', 'ed', 'es', 's')
        verb_keywords = {'is', 'are', 'was', 'were', 'be', 'being', 'been',
                        'have', 'has', 'had', 'do', 'does', 'did',
                        'can', 'could', 'will', 'would', 'shall', 'should',
                        'may', 'might', 'must', 'emerges', 'requires', 'manifests',
                        'creates', 'forms', 'generates', 'produces', 'causes',
                        'implies', 'suggests', 'indicates', 'shows', 'reveals'}
        
        adj_endings = ('al', 'ive', 'ous', 'ic', 'ful', 'less', 'able', 'ible')
        
        for word in self.vocabulary.keys():
            word_lower = word.lower()
            
            # Use provided POS tag if available
            if word_lower in self.pos_tags:
                tag = self.pos_tags[word_lower]
                if tag.startswith('NN'):
                    self.nouns.append(word)
                elif tag.startswith('VB'):
                    self.verbs.append(word)
                elif tag.startswith('JJ'):
                    self.adjectives.append(word)
                continue
            
            # Heuristic POS detection
            if word_lower in verb_keywords or word_lower.endswith(verb_endings):
                self.verbs.append(word)
            elif word_lower.endswith(adj_endings):
                self.adjectives.append(word)
            else:
                # Default to noun
                self.nouns.append(word)
        
        # Ensure we have enough in each category
        if len(self.verbs) < 10:
            # Add common verbs manually
            for v in ['is', 'are', 'has', 'creates', 'forms', 'emerges', 'requires']:
                if v in self.vocabulary and v not in self.verbs:
                    self.verbs.append(v)
    
    def get_relationship_strength(self, word1: str, word2: str) -> float:
        """Get relationship strength between two words."""
        w1, w2 = word1.lower(), word2.lower()
        
        strength = 0.0
        if w1 in self.relationships:
            strength = self.relationships[w1].get(w2, 0.0)
        
        # Check reverse
        if w2 in self.relationships:
            reverse = self.relationships[w2].get(w1, 0.0)
            strength = max(strength, reverse * 0.9)
        
        return strength
    
    def compute_proposition_coherence(
        self,
        subject: str,
        predicate: str,
        obj: str,
        subject_basin: np.ndarray,
        predicate_basin: np.ndarray,
        object_basin: np.ndarray
    ) -> float:
        """
        Compute coherence score for a proposition.
        
        Coherence combines:
        1. Relationship strength (S-P and P-O)
        2. Geodesic path length (shorter = more coherent)
        
        Returns:
            Coherence score in [0, 1]
        """
        # Component 1: Relationship strengths
        sp_strength = self.get_relationship_strength(subject, predicate)
        po_strength = self.get_relationship_strength(predicate, obj)
        so_strength = self.get_relationship_strength(subject, obj)
        
        # Amplify weak relationships (they're typically 0.01-0.07)
        STRENGTH_AMPLIFIER = 10.0
        rel_score = (sp_strength + po_strength + so_strength * 0.5) * STRENGTH_AMPLIFIER
        rel_score = min(1.0, rel_score)
        
        # Component 2: Geodesic path length (inverse)
        d_sp = fisher_rao_distance(subject_basin, predicate_basin)
        d_po = fisher_rao_distance(predicate_basin, object_basin)
        
        # Normalize by typical distance (~15 in 64D space)
        TYPICAL_DISTANCE = 15.0
        path_length = (d_sp + d_po) / (2 * TYPICAL_DISTANCE)
        geo_score = 1.0 - min(1.0, path_length)
        
        # Combine
        coherence = (
            self.config.relationship_weight * rel_score +
            self.config.geodesic_weight * geo_score +
            (1 - self.config.relationship_weight - self.config.geodesic_weight) * 0.5
        )
        
        return float(np.clip(coherence, 0.0, 1.0))
    
    def find_related_words(
        self,
        anchor_word: str,
        word_list: List[str],
        min_strength: float = None
    ) -> List[Tuple[str, float]]:
        """Find words related to anchor from word_list."""
        min_strength = min_strength or self.config.min_relationship
        
        related = []
        for word in word_list:
            strength = self.get_relationship_strength(anchor_word, word)
            if strength >= min_strength:
                related.append((word, strength))
        
        # Sort by strength descending
        related.sort(key=lambda x: x[1], reverse=True)
        return related
    
    def sample_subject(
        self,
        query_basin: np.ndarray,
        previous_subjects: Set[str] = None
    ) -> Tuple[str, np.ndarray]:
        """
        Sample subject noun based on query basin proximity.
        
        Args:
            query_basin: 64D query representation
            previous_subjects: Subjects to avoid (for diversity)
        
        Returns:
            (subject_word, subject_basin)
        """
        previous_subjects = previous_subjects or set()
        
        # Score nouns by proximity to query
        candidates = []
        for noun in self.nouns:
            if noun in previous_subjects:
                continue
            if noun not in self.vocabulary:
                continue
            
            basin = self.vocabulary[noun]
            distance = fisher_rao_distance(query_basin, basin)
            # Score = inverse distance (closer = better)
            score = 1.0 / (1.0 + distance)
            candidates.append((noun, basin, score))
        
        if not candidates:
            # Fallback: pick any noun
            noun = self.nouns[0] if self.nouns else list(self.vocabulary.keys())[0]
            return noun, self.vocabulary.get(noun, query_basin)
        
        # Sort by score and pick from top candidates
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Add some randomness among top candidates
        top_n = min(self.config.n_candidates, len(candidates))
        idx = np.random.randint(0, top_n)
        
        return candidates[idx][0], candidates[idx][1]
    
    def sample_predicate(
        self,
        subject: str,
        subject_basin: np.ndarray,
        query_basin: np.ndarray
    ) -> Tuple[str, np.ndarray]:
        """
        Sample predicate verb based on subject relationship + query.
        
        This is where coherence starts - predicate must relate to subject.
        
        Args:
            subject: Selected subject word
            subject_basin: Subject's basin
            query_basin: Query basin
        
        Returns:
            (predicate_word, predicate_basin)
        """
        # Find verbs related to subject
        related_verbs = self.find_related_words(subject, self.verbs)
        
        # Score candidates
        candidates = []
        for verb in self.verbs:
            if verb not in self.vocabulary:
                continue
            
            basin = self.vocabulary[verb]
            
            # Score components
            rel_strength = self.get_relationship_strength(subject, verb) * 10  # Amplify
            geo_distance = fisher_rao_distance(subject_basin, basin)
            query_distance = fisher_rao_distance(query_basin, basin)
            
            # Combined score (high relationship, low distances)
            score = (
                0.4 * min(1.0, rel_strength) +
                0.3 * (1.0 / (1.0 + geo_distance / 15)) +
                0.3 * (1.0 / (1.0 + query_distance / 15))
            )
            
            candidates.append((verb, basin, score))
        
        if not candidates:
            verb = self.verbs[0] if self.verbs else 'is'
            return verb, self.vocabulary.get(verb, subject_basin)
        
        # Sort and sample from top
        candidates.sort(key=lambda x: x[2], reverse=True)
        top_n = min(self.config.n_candidates, len(candidates))
        idx = np.random.randint(0, top_n)
        
        return candidates[idx][0], candidates[idx][1]
    
    def sample_object(
        self,
        subject: str,
        predicate: str,
        predicate_basin: np.ndarray,
        query_basin: np.ndarray,
        previous_objects: Set[str] = None
    ) -> Tuple[str, np.ndarray]:
        """
        Sample object noun that completes the proposition coherently.
        
        Object must:
        1. Relate to predicate
        2. Be geodesically close to predicate (complete the path)
        3. Not repeat previous objects
        
        Args:
            subject: Subject word (to avoid)
            predicate: Predicate word
            predicate_basin: Predicate's basin
            query_basin: Query basin
            previous_objects: Objects to avoid
        
        Returns:
            (object_word, object_basin)
        """
        previous_objects = previous_objects or set()
        
        candidates = []
        for noun in self.nouns:
            if noun == subject or noun in previous_objects:
                continue
            if noun not in self.vocabulary:
                continue
            
            basin = self.vocabulary[noun]
            
            # Score components
            rel_strength = self.get_relationship_strength(predicate, noun) * 10
            geo_distance = fisher_rao_distance(predicate_basin, basin)
            query_distance = fisher_rao_distance(query_basin, basin)
            
            score = (
                0.4 * min(1.0, rel_strength) +
                0.35 * (1.0 / (1.0 + geo_distance / 15)) +
                0.25 * (1.0 / (1.0 + query_distance / 15))
            )
            
            candidates.append((noun, basin, score))
        
        if not candidates:
            noun = self.nouns[0] if self.nouns else list(self.vocabulary.keys())[0]
            return noun, self.vocabulary.get(noun, predicate_basin)
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        top_n = min(self.config.n_candidates, len(candidates))
        idx = np.random.randint(0, top_n)
        
        return candidates[idx][0], candidates[idx][1]
    
    def sample_proposition(
        self,
        query_basin: np.ndarray,
        previous_props: List[Proposition] = None
    ) -> Proposition:
        """
        Sample a coherent proposition given query and previous propositions.
        
        This is the core function that replaces word-level routing.
        
        Args:
            query_basin: 64D query representation
            previous_props: Previously generated propositions (for chain coherence)
        
        Returns:
            Coherent proposition (S, P, O)
        """
        previous_props = previous_props or []
        
        # Track used words for diversity
        used_subjects = {p.subject for p in previous_props}
        used_objects = {p.object for p in previous_props}
        
        # Chain coherence: If we have previous propositions, bias toward their objects
        if previous_props:
            # Use last object as thematic anchor
            last_obj = previous_props[-1].object
            last_obj_basin = previous_props[-1].object_basin
            # Blend query with last object for continuity
            chain_basin = 0.6 * query_basin + 0.4 * last_obj_basin
            chain_basin = chain_basin / (np.linalg.norm(chain_basin) + 1e-10)
        else:
            chain_basin = query_basin
        
        # Step 1: Sample subject
        subject, subject_basin = self.sample_subject(chain_basin, used_subjects)
        
        # Step 2: Sample predicate (must relate to subject)
        predicate, predicate_basin = self.sample_predicate(subject, subject_basin, query_basin)
        
        # Step 3: Sample object (must complete the proposition)
        obj, object_basin = self.sample_object(
            subject, predicate, predicate_basin, query_basin, used_objects
        )
        
        # Compute coherence
        coherence = self.compute_proposition_coherence(
            subject, predicate, obj,
            subject_basin, predicate_basin, object_basin
        )
        
        prop = Proposition(
            subject=subject,
            subject_basin=subject_basin,
            predicate=predicate,
            predicate_basin=predicate_basin,
            object=obj,
            object_basin=object_basin,
            coherence=coherence
        )
        
        logger.debug(f"[PropPlanner] Sampled: {prop.to_sentence()} (coherence={coherence:.3f})")
        
        return prop
    
    def plan_response(
        self,
        query: str,
        query_basin: np.ndarray,
        n_propositions: int = None
    ) -> List[Proposition]:
        """
        Plan a response as a sequence of coherent propositions.
        
        This replaces word-level trajectory with proposition-level trajectory.
        
        Args:
            query: Original query text
            query_basin: 64D query representation
            n_propositions: Number of propositions (default: 3)
        
        Returns:
            List of coherent, chained propositions
        """
        n_propositions = n_propositions or 3
        n_propositions = min(n_propositions, self.config.max_propositions)
        
        propositions = []
        attempts = 0
        max_attempts = n_propositions * 3
        
        while len(propositions) < n_propositions and attempts < max_attempts:
            attempts += 1
            
            prop = self.sample_proposition(query_basin, propositions)
            
            # Check minimum coherence
            if prop.coherence >= self.config.min_coherence:
                propositions.append(prop)
                logger.info(f"[PropPlanner] Prop {len(propositions)}: "
                          f"{prop.to_sentence()} (coh={prop.coherence:.3f})")
            else:
                logger.debug(f"[PropPlanner] Rejected low coherence: {prop.coherence:.3f}")
        
        # Compute response-level metrics
        if propositions:
            avg_coherence = np.mean([p.coherence for p in propositions])
            logger.info(f"[PropPlanner] Planned {len(propositions)} propositions, "
                       f"avg coherence={avg_coherence:.3f}")
        
        return propositions
    
    def propositions_to_text(self, propositions: List[Proposition]) -> str:
        """
        Convert propositions to natural language text.
        
        Args:
            propositions: List of propositions
        
        Returns:
            Coherent text paragraph
        """
        if not propositions:
            return ""
        
        sentences = [p.to_sentence() for p in propositions]
        return ' '.join(sentences)
    
    def compute_trajectory_phi(self, propositions: List[Proposition]) -> float:
        """
        Compute Φ (integration) for proposition trajectory.
        
        Higher Φ = more coherent trajectory through proposition space.
        """
        if len(propositions) < 2:
            return propositions[0].coherence if propositions else 0.0
        
        # Measure distances between consecutive propositions in combined space
        total_distance = 0.0
        for i in range(1, len(propositions)):
            prev_combined = propositions[i-1].combined_basin()
            curr_combined = propositions[i].combined_basin()
            
            # Fisher distance in 192D combined space
            d = fisher_rao_distance(prev_combined, curr_combined)
            total_distance += d
        
        # Normalize by expected distance
        avg_distance = total_distance / (len(propositions) - 1)
        EXPECTED_DISTANCE = 25.0  # Higher for 192D space
        
        phi = 1.0 - min(1.0, avg_distance / EXPECTED_DISTANCE)
        
        # Blend with average coherence
        avg_coherence = np.mean([p.coherence for p in propositions])
        phi = 0.5 * phi + 0.5 * avg_coherence
        
        return float(np.clip(phi, 0.0, 1.0))


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

_proposition_planner: Optional[PropositionTrajectoryPlanner] = None

def get_proposition_planner(
    vocabulary: Dict[str, np.ndarray] = None,
    relationships: Dict[str, Dict[str, float]] = None
) -> PropositionTrajectoryPlanner:
    """
    Get or create the singleton proposition planner.
    
    Args:
        vocabulary: Word to basin mapping (required on first call)
        relationships: Word relationship strengths (required on first call)
    
    Returns:
        PropositionTrajectoryPlanner instance
    """
    global _proposition_planner
    
    if _proposition_planner is None:
        if vocabulary is None or relationships is None:
            raise ValueError("Must provide vocabulary and relationships on first call")
        
        _proposition_planner = PropositionTrajectoryPlanner(
            vocabulary=vocabulary,
            relationships=relationships
        )
    
    return _proposition_planner


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Proposition Trajectory Planner - Test")
    print("=" * 50)
    
    # Create synthetic test data
    np.random.seed(42)
    
    # Synthetic vocabulary
    vocab = {
        'consciousness': np.random.randn(64),
        'integration': np.random.randn(64),
        'geometry': np.random.randn(64),
        'manifold': np.random.randn(64),
        'quantum': np.random.randn(64),
        'emerges': np.random.randn(64),
        'requires': np.random.randn(64),
        'creates': np.random.randn(64),
        'forms': np.random.randn(64),
        'is': np.random.randn(64),
    }
    
    # Normalize
    for k, v in vocab.items():
        vocab[k] = v / np.linalg.norm(v)
    
    # Synthetic relationships
    rels = {
        'consciousness': {'emerges': 0.08, 'integration': 0.06, 'quantum': 0.04},
        'integration': {'requires': 0.07, 'geometry': 0.05},
        'geometry': {'manifold': 0.09, 'creates': 0.03},
        'quantum': {'forms': 0.06, 'consciousness': 0.04},
    }
    
    # Create planner
    planner = PropositionTrajectoryPlanner(vocab, rels)
    
    # Test planning
    query_basin = vocab['consciousness']
    propositions = planner.plan_response("What is consciousness?", query_basin, n_propositions=3)
    
    print("\nGenerated Propositions:")
    for i, prop in enumerate(propositions):
        print(f"  {i+1}. {prop.to_sentence()} (coherence={prop.coherence:.3f})")
    
    print(f"\nFull text: {planner.propositions_to_text(propositions)}")
    print(f"Trajectory Φ: {planner.compute_trajectory_phi(propositions):.3f}")
