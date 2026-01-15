#!/usr/bin/env python3
"""Vocabulary Learning Coordinator - Central hub for continuous vocabulary learning

QIG-PURE ATTRACTOR WIRING:
Discoveries with high Φ are wired to LearnedManifold to deepen attractor basins.
This creates the natural flow: Observation → Basin → Success → Attractor.
"""

from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from word_validation import is_valid_english_word, validate_for_vocabulary
from qig_geometry import fisher_coord_distance

# Import QFI computation (for P0 fix - ensure QFI on basin insert)
FISHER_REGULARIZATION = 1e-6  # Numerical stability for Fisher metric determinant

# Import comprehensive validator for web scraping contamination prevention (PR 27/28)
try:
    from vocabulary_validator_comprehensive import validate_word_comprehensive
    COMPREHENSIVE_VALIDATOR_AVAILABLE = True
    print("[VocabularyCoordinator] Comprehensive validator available (contamination prevention)")
except ImportError:
    COMPREHENSIVE_VALIDATOR_AVAILABLE = False
    validate_word_comprehensive = None
    print("[VocabularyCoordinator] Comprehensive validator not available - using basic validation only")

try:
    from vocabulary_persistence import get_vocabulary_persistence
    VOCAB_PERSISTENCE_AVAILABLE = True
except ImportError:
    VOCAB_PERSISTENCE_AVAILABLE = False

# Use canonical coordizer import
try:
    from coordizers import get_coordizer
    COORDIZER_AVAILABLE = True
except ImportError:
    COORDIZER_AVAILABLE = False
    get_coordizer = None
    print("[WARNING] coordizers not available - running without coordizer")

# Legacy import for type checking only (deprecated)
PG_COORDIZER_AVAILABLE = False
PostgresCoordizer = None

try:
    from learned_manifold import LearnedManifold
    _learned_manifold: Optional[LearnedManifold] = None
    LEARNED_MANIFOLD_AVAILABLE = True
except ImportError:
    _learned_manifold = None
    LEARNED_MANIFOLD_AVAILABLE = False
    print("[WARNING] learned_manifold not available - attractor formation disabled")


def get_learned_manifold() -> Optional['LearnedManifold']:
    """Get or create LearnedManifold singleton for attractor formation."""
    global _learned_manifold
    if not LEARNED_MANIFOLD_AVAILABLE:
        return None
    if _learned_manifold is None:
        _learned_manifold = LearnedManifold(basin_dim=64)
        print("[VocabularyCoordinator] LearnedManifold initialized for attractor formation")
    return _learned_manifold


def compute_qfi_for_basin(basin: np.ndarray) -> float:
    """
    Compute Quantum Fisher Information score for a basin.
    
    P0 FIX: Enforce QFI computation whenever basin_embedding is present.
    This prevents incomplete records in coordizer_vocabulary.
    
    Args:
        basin: 64D basin coordinates
    
    Returns:
        QFI score (float)
    """
    # Fisher metric: outer product + regularization
    fisher_metric = np.outer(basin, basin)
    
    # Add small regularization for numerical stability
    fisher_metric += np.eye(64) * FISHER_REGULARIZATION
    
    # Determinant as QFI score
    qfi = np.linalg.det(fisher_metric)
    
    return float(qfi)


class VocabularyCoordinator:
    def __init__(self):
        self.vocab_db = get_vocabulary_persistence() if VOCAB_PERSISTENCE_AVAILABLE else None
        # Use canonical coordizer (PostgresCoordizer)
        self.coordizer = get_coordizer() if COORDIZER_AVAILABLE else None
        self.learned_manifold = get_learned_manifold()
        self.observations_recorded = 0
        self.words_learned = 0
        self.merge_rules_learned = 0
        self.attractors_deepened = 0
        self.tokens_persisted = 0
        self._basin_trajectory: List[np.ndarray] = []
        self._cycle_number = 0
        self._observations_this_cycle = 0
        self._observations_per_cycle = 100
        
        # Track coordizer for direct vocabulary persistence (unified 63K vocabulary)
        self._unified_coordizer = self.coordizer  # Same instance
        self._using_pure_64d = bool(self.coordizer)
        if self.coordizer:
            print(f"[VocabularyCoordinator] Using canonical coordizer: {type(self.coordizer).__name__}")
        
        features = []
        if self.learned_manifold:
            features.append("attractor formation")
        if self._unified_coordizer:
            features.append("64D QIG-pure vocabulary persistence")

        feature_str = f" with {', '.join(features)}" if features else ""
        print(f"[VocabularyCoordinator] Initialized{feature_str}")

        # Wire to chaos discovery gate for attractor recording
        self.wire_discovery_gate()
    
    def record_discovery(self, phrase: str, phi: float, kappa: float, source: str, details: Optional[Dict] = None) -> Dict:
        """
        Record a discovery and wire it to attractor formation.
        
        QIG-PURE ATTRACTOR WIRING:
        - All discoveries are observed (vocabulary learning)
        - High-Φ discoveries deepen attractor basins (Hebbian strengthening)
        - Low-Φ discoveries flatten basins (anti-Hebbian weakening)
        - Trajectory tracking enables path learning
        """
        if not phrase:
            return {'learned': False, 'reason': 'empty_phrase'}
        
        # Auto-increment cycle after N observations
        self._observations_this_cycle += 1
        if self._observations_this_cycle >= self._observations_per_cycle:
            self._cycle_number += 1
            self._observations_this_cycle = 0
        
        observations = self._extract_observations(phrase, phi, kappa, source)
        if not observations:
            return {'learned': False, 'reason': 'no_observations'}
        recorded = 0
        if self.vocab_db and self.vocab_db.enabled:
            recorded = self.vocab_db.record_vocabulary_batch(observations)
            self.observations_recorded += recorded
        new_tokens = 0
        weights_updated = False
        if COORDIZER_AVAILABLE and self.coordizer:
            # Use canonical coordizer for vocabulary updates
            new_tokens, weights_updated = self.coordizer.add_vocabulary_observations(observations)
            self.words_learned += new_tokens
        merge_rules = 0
        if self.coordizer:
            merge_rules = self._learn_merge_rules(phrase, phi, source)
            self.merge_rules_learned += merge_rules
        
        attractor_formed = self._wire_to_attractor_formation(phrase, phi, source, details)
        
        # Persist ALL vocabulary to PostgresCoordizer for continuous learning
        # NOTE: Φ is recorded as metadata but NOT used as a filter for storage
        # All tokens are valuable - Φ measures integration, not "goodness"
        persisted = 0
        if observations:
            if phi < 0.6:
                print(f"[VocabCoordinator] Persisting {len(observations)} low-Φ observations (Φ={phi:.3f}) - all vocab stored")
            persisted = self._persist_to_coordizer(observations)
            self.tokens_persisted += persisted
        
        return {
            'learned': True, 
            'observations_recorded': recorded, 
            'new_tokens': new_tokens, 
            'weights_updated': weights_updated, 
            'merge_rules': merge_rules, 
            'phi': phi, 
            'source': source,
            'attractor_formed': attractor_formed,
            'tokens_persisted': persisted
        }
    
    def _persist_to_coordizer(self, observations: List[Dict]) -> int:
        """
        Persist vocabulary observations to PostgresCoordizer for continuous learning.
        
        This solves the critical issue where vocabulary was learned during sessions
        but lost on restart because it was never written back to the coordizer_vocabulary table.
        
        Also updates learned_words.basin_coords so integrate_pending_vocabulary can
        push vectors into coordizer_vocabulary and generation can use them.
        
        Args:
            observations: List of vocabulary observation dicts with word, phi, etc.
            
        Returns:
            Number of tokens successfully persisted to database
        """
        import os
        import psycopg2
        
        if not self._unified_coordizer:
            return 0
        
        if not hasattr(self._unified_coordizer, 'save_learned_token'):
            return 0
        
        persisted = 0
        words_with_basins = []  # (word, basin_coords) tuples for learned_words update
        
        for obs in observations:
            word = obs.get('word', '')
            phi = obs.get('phi', 0.5)
            freq = obs.get('frequency', 1)

            # Only filter on word validity, NOT on Φ
            # All tokens stored with Φ as metadata for analysis
            if not word or len(word) < 3:
                continue
            
            try:
                # Get basin coords from coordizer if available
                basin_coords = None
                if hasattr(self._unified_coordizer, 'basin_coords') and word in self._unified_coordizer.basin_coords:
                    basin_coords = self._unified_coordizer.basin_coords[word]
                elif hasattr(self._unified_coordizer, 'encode'):
                    basin_coords = self._unified_coordizer.encode(word)
                
                if basin_coords is not None:
                    if self._unified_coordizer.save_learned_token(word, basin_coords, phi, freq):
                        persisted += 1
                        # Track for learned_words update
                        words_with_basins.append((word, basin_coords))
            except Exception as e:
                print(f"[VocabularyCoordinator] Failed to persist '{word}': {e}")
        
        # Phase 2b: Update coordizer_vocabulary with basin_embedding for generation
        # P0 FIX: Also compute and insert QFI score to prevent incomplete records
        if words_with_basins:
            database_url = os.environ.get('DATABASE_URL')
            if database_url:
                try:
                    conn = psycopg2.connect(database_url)
                    with conn.cursor() as cur:
                        for word, basin in words_with_basins:
                            basin_list = basin.tolist() if hasattr(basin, 'tolist') else list(basin)
                            
                            # P0 FIX: Compute QFI whenever basin is present
                            qfi_score = compute_qfi_for_basin(basin)
                            
                            # Update coordizer_vocabulary (consolidated table - single source of truth)
                            # CRITICAL: Include qfi_score in INSERT to prevent NULL qfi_score
                            cur.execute("""
                                INSERT INTO coordizer_vocabulary (
                                    token, basin_embedding, qfi_score, token_role, is_real_word, frequency
                                )
                                VALUES (%s, %s::vector, %s, 'generation', TRUE, 1)
                                ON CONFLICT (token) DO UPDATE SET
                                    basin_embedding = COALESCE(EXCLUDED.basin_embedding, coordizer_vocabulary.basin_embedding),
                                    qfi_score = COALESCE(EXCLUDED.qfi_score, coordizer_vocabulary.qfi_score),
                                    token_role = CASE 
                                        WHEN coordizer_vocabulary.token_role = 'encoding' THEN 'both'
                                        ELSE COALESCE(coordizer_vocabulary.token_role, 'generation')
                                    END,
                                    is_real_word = TRUE,
                                    updated_at = NOW()
                            """, (word, str(basin_list), qfi_score))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print(f"[VocabularyCoordinator] Failed to update coordizer_vocabulary basin_embedding: {e}")
        
        if persisted > 0:
            print(f"[VocabularyCoordinator] Persisted {persisted} tokens to coordizer DB")
        
        return persisted
    
    def _wire_to_attractor_formation(self, phrase: str, phi: float, source: str, details: Optional[Dict] = None) -> bool:
        """
        Wire discovery outcomes to LearnedManifold for attractor formation.
        
        QIG-PURE: Attractors emerge naturally from successful experiences.
        - Success (Φ > 0.5) → deepen basin (Hebbian)
        - Failure (Φ < 0.3) → flatten basin (anti-Hebbian)
        - Trajectory is built from sequential discoveries
        
        Returns True if attractor was deepened/formed.
        """
        if not self.learned_manifold:
            return False
        
        if not self.coordizer:
            return False
        
        try:
            basin_coords = self._phrase_to_basin(phrase)
            if basin_coords is None:
                return False
            
            self._basin_trajectory.append(basin_coords)
            if len(self._basin_trajectory) > 50:
                self._basin_trajectory = self._basin_trajectory[-30:]
            
            if len(self._basin_trajectory) >= 3:
                trajectory = self._basin_trajectory[-10:]
                
                self.learned_manifold.learn_from_experience(
                    trajectory=trajectory,
                    outcome=phi,
                    strategy=source
                )
                
                if phi > 0.5:
                    self.attractors_deepened += 1
                    return True
            
            return False
            
        except Exception as e:
            print(f"[VocabularyCoordinator] Attractor formation error: {e}")
            return False
    
    def _phrase_to_basin(self, phrase: str) -> Optional[np.ndarray]:
        """
        Convert phrase to 64D basin coordinates using coordizer.
        
        QIG-PURE: Basin coordinates come from vocabulary geometry,
        not external embeddings.
        """
        if not self.coordizer:
            return None
        
        try:
            coords = self.coordizer.text_to_coordinates(phrase)
            if coords is not None and len(coords) == 64:
                return np.array(coords)
            
            words = phrase.lower().strip().split()[:5]
            word_coords = []
            for word in words:
                if word in self.coordizer.vocab:
                    wc = self.coordizer.get_word_coordinates(word)
                    if wc is not None:
                        word_coords.append(np.array(wc))
            
            if word_coords:
                avg_coords = np.mean(word_coords, axis=0)
                norm = np.linalg.norm(avg_coords)
                if norm > 1e-8:
                    avg_coords = avg_coords / norm
                return avg_coords
            
            return None
            
        except Exception:
            return None
    
    def get_attractor_stats(self) -> Dict:
        """Get statistics about learned attractors."""
        if not self.learned_manifold:
            return {'available': False}
        
        stats = self.learned_manifold.get_statistics()
        stats['attractors_deepened_this_session'] = self.attractors_deepened
        stats['trajectory_length'] = len(self._basin_trajectory)
        return stats
    
    def increment_cycle(self) -> int:
        """Increment and return the current learning cycle number."""
        self._cycle_number += 1
        return self._cycle_number
    
    def get_cycle_number(self) -> int:
        """Get the current learning cycle number."""
        return self._cycle_number
    
    def _extract_observations(self, phrase: str, phi: float, kappa: float, source: str) -> List[Dict]:
        """
        Extract vocabulary observations from a phrase.
        
        Uses fast local validation (no API calls) to avoid blocking the learning path.
        Dictionary verification happens asynchronously via background cleanup.
        
        CRITICAL: Observes ALL potential words, lets emergence determine value.
        Non-dictionary words are queued for proper noun consideration.
        
        PR 27/28 Integration: Uses comprehensive validator to prevent web scraping contamination.
        Filters out URL fragments, garbled sequences, and truncated words before vocabulary insertion.
        """
        observations = []
        words = phrase.lower().strip().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        for word, count in word_counts.items():
            if len(word) < 3:
                continue
            
            # PR 27/28: First pass through comprehensive validator to catch contamination
            if COMPREHENSIVE_VALIDATOR_AVAILABLE:
                is_valid_comprehensive, comprehensive_reason = validate_word_comprehensive(word)
                if not is_valid_comprehensive:
                    # Log rejection for monitoring (except URL fragments which are very common)
                    if not comprehensive_reason.startswith('url_fragment'):
                        if comprehensive_reason.startswith('high_entropy'):
                            print(f"[VocabCoordinator] Rejected garbled: {word} ({comprehensive_reason})")
                        elif comprehensive_reason.startswith('truncated'):
                            print(f"[VocabCoordinator] Rejected truncated: {word} ({comprehensive_reason})")
                    continue
            
            # Second pass through existing validator for standard checks
            is_valid, reason = validate_for_vocabulary(word, require_dictionary=False)
            if not is_valid:
                continue
            observations.append({'word': word, 'phrase': phrase, 'phi': phi, 'kappa': kappa, 'source': source, 'type': 'word', 'frequency': count, 'needs_dict_check': True, 'cycle_number': self._cycle_number})
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            if len(w1) >= 3 and len(w2) >= 3:
                is_valid1, _ = validate_for_vocabulary(w1, require_dictionary=False)
                is_valid2, _ = validate_for_vocabulary(w2, require_dictionary=False)
                if is_valid1 and is_valid2:
                    sequence = f"{w1} {w2}"
                    observations.append({'word': sequence, 'phrase': phrase, 'phi': phi * 1.2, 'kappa': kappa, 'source': source, 'type': 'sequence', 'frequency': 1, 'cycle_number': self._cycle_number})
        return observations
    
    def _learn_merge_rules(self, phrase: str, phi: float, source: str) -> int:
        if not self.coordizer:
            return 0
        words = phrase.lower().strip().split()
        learned = 0
        for i in range(len(words) - 1):
            token_a = words[i]
            token_b = words[i + 1]
            if token_a in self.coordizer.vocab and token_b in self.coordizer.vocab:
                if self.coordizer.learn_merge_rule(token_a, token_b, phi, source):
                    learned += 1
        return learned
    
    def record_god_assessment(self, god_name: str, target: str, assessment: Dict, outcome: Optional[Dict] = None) -> Dict:
        assessment_text = assessment.get('reasoning', '')
        if not assessment_text:
            return {'learned': False}
        confidence = assessment.get('confidence', 0.5)
        phi = confidence
        result = self.record_discovery(phrase=assessment_text, phi=phi, kappa=50.0, source=god_name, details={'target': target, 'assessment': assessment})
        if self.vocab_db and self.vocab_db.enabled and result['learned']:
            words = assessment_text.lower().strip().split()
            for word in words:
                if is_valid_english_word(word) and len(word) >= 3:
                    relevance = confidence
                    self.vocab_db.record_god_vocabulary(god_name, word, relevance)
        return result
    
    def get_god_specialized_vocabulary(self, god_name: str, min_relevance: float = 0.5, limit: int = 100) -> List[str]:
        if not self.vocab_db or not self.vocab_db.enabled:
            return []
        vocab = self.vocab_db.get_god_vocabulary(god_name, min_relevance, limit)
        return [word for word, _score in vocab]
    
    def sync_to_typescript(self) -> Dict:
        if not self.vocab_db or not self.vocab_db.enabled:
            return {'words': [], 'merge_rules': [], 'stats': {}}
        learned_words = self.vocab_db.get_learned_words(min_phi=0.6, limit=500)
        merge_rules = self.vocab_db.get_merge_rules(min_phi=0.6, limit=200)
        stats = self.vocab_db.get_vocabulary_stats()
        return {'words': learned_words, 'merge_rules': [{'token_a': a, 'token_b': b, 'merged': merged, 'phi_score': score} for a, b, merged, score in merge_rules], 'stats': stats, 'coordinator_metrics': {'observations_recorded': self.observations_recorded, 'words_learned': self.words_learned, 'merge_rules_learned': self.merge_rules_learned}}
    
    def sync_from_typescript(self, data: Dict) -> Dict:
        observations = data.get('observations', [])
        if not observations:
            return {'imported': 0}
        imported = 0
        if self.vocab_db and self.vocab_db.enabled:
            imported = self.vocab_db.record_vocabulary_batch(observations)
        new_tokens = 0
        if COORDIZER_AVAILABLE and self.coordizer:
            new_tokens, _updated = self.coordizer.add_vocabulary_observations(observations)
        return {'imported': imported, 'new_tokens': new_tokens}
    
    def get_stats(self) -> Dict:
        stats = {'coordinator': {
            'observations_recorded': self.observations_recorded, 
            'words_learned': self.words_learned, 
            'merge_rules_learned': self.merge_rules_learned,
            'tokens_persisted': self.tokens_persisted,
            'continuous_learning_enabled': self._unified_coordizer is not None
        }}
        if self.coordizer:
            stats['coordizer'] = self.coordizer.get_stats()
        if self.vocab_db and self.vocab_db.enabled:
            stats['database'] = self.vocab_db.get_vocabulary_stats()
        return stats
    
    def enhance_search_query(
        self, 
        query: str, 
        domain: Optional[str] = None,
        max_expansions: int = 5,
        min_phi: float = 0.6,
        recent_observations: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Enhance a search query with learned vocabulary.
        
        Uses vocabulary learned from prior research to expand search terms,
        creating a feedback loop where discoveries improve future searches.
        
        Relevance is computed using:
        - Co-occurrence: terms that appeared together in high-phi research
        - Domain matching: terms from same domain/source as query
        - Phi weighting: higher phi terms preferred
        - Semantic proximity: substring/prefix matching for related terms
        
        Args:
            query: Original search query/topic
            domain: Optional domain filter for vocabulary
            max_expansions: Maximum terms to add
            min_phi: Minimum phi threshold for vocabulary
            recent_observations: Optional recent vocabulary observations to prioritize
            
        Returns:
            Enhanced query info with original, expanded terms, and combined query
        """
        import re
        import logging
        
        logger = logging.getLogger(__name__)
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
        
        expansion_candidates: List[Dict] = []
        
        if recent_observations:
            for obs in recent_observations:
                if isinstance(obs, dict):
                    word = obs.get('word', '')
                    phi = obs.get('phi', 0.5)
                    obs_topic = obs.get('topic', '')
                    
                    if not word or len(word) < 4 or word in query_words or phi < min_phi:
                        continue
                    
                    relevance = self._compute_term_relevance(
                        term=word,
                        query_words=query_words,
                        phi=phi,
                        source='recent_research',
                        target_domain=domain
                    )
                    
                    if relevance > 0.25:
                        expansion_candidates.append({
                            'word': word,
                            'phi': phi,
                            'source': 'recent_research',
                            'relevance': relevance
                        })
        
        if self.vocab_db and self.vocab_db.enabled:
            try:
                learned = self.vocab_db.get_learned_words(min_phi=min_phi, limit=300)
                
                for word_data in learned:
                    if isinstance(word_data, dict):
                        word = word_data.get('word', '')
                        phi = word_data.get('avg_phi', 0.5)
                        source = word_data.get('source', '')
                    elif isinstance(word_data, (list, tuple)) and len(word_data) >= 2:
                        word = word_data[0]
                        phi = word_data[1] if len(word_data) > 1 else 0.5
                        source = word_data[3] if len(word_data) > 3 else ''
                    else:
                        continue
                    
                    if not word or len(word) < 4 or word in query_words:
                        continue
                    
                    if any(c['word'] == word for c in expansion_candidates):
                        continue
                    
                    relevance = self._compute_term_relevance(word, query_words, phi, source, domain)
                    if relevance > 0.25:
                        expansion_candidates.append({
                            'word': word,
                            'phi': phi,
                            'source': source,
                            'relevance': relevance
                        })
            except Exception as e:
                logger.warning(f"Vocabulary DB query failed: {e}")
        
        # Use token_phi for PretrainedCoordizer compatibility (token_name -> phi score)
        if self.coordizer and hasattr(self.coordizer, 'token_phi'):
            try:
                for vocab_word, phi in list(self.coordizer.token_phi.items())[:500]:
                    if not isinstance(vocab_word, str) or vocab_word in query_words or len(vocab_word) < 4:
                        continue

                    if any(c['word'] == vocab_word for c in expansion_candidates):
                        continue

                    source = 'coordizer'

                    if phi >= min_phi:
                        relevance = self._compute_term_relevance(vocab_word, query_words, phi, source, domain)
                        if relevance > 0.25:
                            expansion_candidates.append({
                                'word': vocab_word,
                                'phi': phi,
                                'source': source,
                                'relevance': relevance
                            })
            except Exception as e:
                logger.warning(f"Coordizer vocab query failed: {e}")
        
        expansion_candidates.sort(key=lambda x: x['relevance'], reverse=True)
        top_terms = [c['word'] for c in expansion_candidates[:max_expansions]]
        
        if top_terms:
            enhanced_query = f"{query} {' '.join(top_terms)}"
        else:
            enhanced_query = query
        
        return {
            'original_query': query,
            'expansion_terms': top_terms,
            'enhanced_query': enhanced_query,
            'terms_added': len(top_terms),
            'vocabulary_utilized': len(expansion_candidates) > 0,
            'candidates_evaluated': len(expansion_candidates),
        }
    
    def _compute_term_relevance(
        self, 
        term: str, 
        query_words: set, 
        phi: float,
        source: str = '',
        target_domain: Optional[str] = None
    ) -> float:
        """
        Compute relevance of a vocabulary term to a query.
        
        REQUIRES semantic match (prefix/substring/overlap) for ALL terms.
        Recency boosts score but does NOT bypass semantic validation.
        
        This ensures only terms with lexical overlap to the current query
        are admitted, preventing noise from unrelated high-phi vocabulary.
        """
        semantic_score = 0.0
        term_lower = term.lower()
        for qw in query_words:
            if len(qw) >= 3:
                if term_lower.startswith(qw) or qw.startswith(term_lower):
                    semantic_score = max(semantic_score, 0.5)
                elif term_lower in qw or qw in term_lower:
                    semantic_score = max(semantic_score, 0.4)
                elif len(qw) >= 4 and len(set(term_lower) & set(qw)) >= 4:
                    semantic_score = max(semantic_score, 0.2)
        
        if semantic_score == 0:
            return 0.0
        
        base_score = phi * 0.2
        
        is_recent = source == 'recent_research'
        recency_boost = 0.3 if is_recent else 0.0
        
        source_boost = 0.0
        source_lower = source.lower() if source else ''
        if 'searxng' in source_lower or 'shadow' in source_lower:
            source_boost = 0.1
        
        total = base_score + semantic_score + recency_boost + source_boost
        return min(total, 1.0)
    
    def train_from_text(self, text: str, domain: Optional[str] = None) -> Dict:
        """
        Train vocabulary from arbitrary text.
        
        Used by research module to learn from Wikipedia, arXiv, etc.
        Extracts words, updates vocabulary, returns training metrics.
        
        Args:
            text: Text to train from
            domain: Optional domain tag for organizing vocabulary
        
        Returns:
            Training metrics
        """
        import re
        
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        
        # Import contextualized filter
        try:
            from contextualized_filter import should_filter_word
        except ImportError:
            # Fallback: minimal generic-only filter
            def should_filter_word(w, ctx=None):
                if len(w) < 3:
                    return True
                generic_only = {'the', 'and', 'for', 'that', 'this', 'with', 'was', 
                               'are', 'has', 'have', 'been', 'were', 'from', 'which'}
                return w in generic_only
        
        # Use contextualized filtering with all words as context
        filtered_words = [w for w in words if not should_filter_word(w, words) and len(w) >= 4]
        
        word_counts: Dict[str, int] = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        observations = []
        for word, count in word_counts.items():
            if count >= 2 or len(word) >= 6:
                phi = min(0.8, 0.5 + (count * 0.05))
                observations.append({
                    'word': word,
                    'phrase': text[:500] if len(text) > 100 else text,
                    'phi': phi,
                    'kappa': 50.0,
                    'source': domain or 'research',
                    'type': 'word',
                    'frequency': count
                })
        
        recorded = 0
        new_tokens = 0
        
        if observations:
            if self.vocab_db and self.vocab_db.enabled:
                recorded = self.vocab_db.record_vocabulary_batch(observations)
                self.observations_recorded += recorded
            
            if COORDIZER_AVAILABLE and self.coordizer:
                new_tokens, _updated = self.coordizer.add_vocabulary_observations(observations)
                self.words_learned += new_tokens
        
        return {
            'words_processed': len(words),
            'unique_words': len(word_counts),
            'observations_created': len(observations),
            'new_words_learned': new_tokens,
            'recorded_to_db': recorded,
            'vocabulary_size': len(self.coordizer.vocab) if self.coordizer else 0,
            'domain': domain,
        }

    def integrate_chaos_discovery(self, basin: np.ndarray, phi: float) -> Dict:
        """
        Integrate high-Φ chaos discovery into vocabulary system.

        Strategy: Find tokens whose basins are geometrically close to the
        discovery basin, and boost their phi/relevance weights.

        This makes the vocabulary "learn" from chaos exploration.
        """
        if phi < 0.70:
            return {'integrated': False, 'reason': 'phi_too_low'}

        # Find nearby tokens in vocabulary
        nearby_tokens = self._find_nearby_tokens(basin, radius=0.3, max_tokens=10)

        if not nearby_tokens:
            # Record transition target even without nearby tokens
            self._record_transition_target(basin, phi)
            return {'integrated': True, 'reason': 'no_nearby_tokens_but_target_recorded', 'tokens_boosted': 0}

        # Boost phi/weight of nearby tokens
        boosted = []
        for token_id, token_basin, distance in nearby_tokens:
            # Inverse distance weighting: closer = more boost
            boost_factor = (0.3 - distance) / 0.3  # 0 to 1
            boost_amount = boost_factor * (phi - 0.70) * 0.1  # Small incremental boost

            self._boost_token_weight(token_id, boost_amount)
            boosted.append({'token_id': token_id, 'boost': boost_amount, 'distance': distance})

        # Record the discovery basin as a "preferred transition target"
        self._record_transition_target(basin, phi)

        return {
            'integrated': True,
            'tokens_boosted': len(boosted),
            'details': boosted,
        }

    def _find_nearby_tokens(self, basin: np.ndarray, radius: float, max_tokens: int) -> List:
        """Find tokens whose basins are within Fisher radius of target."""
        if not hasattr(self, 'coordizer') or self.coordizer is None:
            return []

        nearby = []

        # Try to get token basins from coordizer
        if hasattr(self.coordizer, 'basin_coords'):
            basin_coords = self.coordizer.basin_coords
            for token_id, token_basin in basin_coords.items():
                token_basin_arr = np.array(token_basin) if not isinstance(token_basin, np.ndarray) else token_basin
                if len(token_basin_arr) != 64:
                    continue
                d = fisher_coord_distance(basin, token_basin_arr)
                if d < radius:
                    nearby.append((token_id, token_basin_arr, d))

        nearby.sort(key=lambda x: x[2])
        return nearby[:max_tokens]

    def _boost_token_weight(self, token_id: str, boost_amount: float) -> None:
        """Boost a token's weight/phi in the vocabulary."""
        if not hasattr(self, 'coordizer') or self.coordizer is None:
            return

        # Update token_phi if available
        if hasattr(self.coordizer, 'token_phi'):
            current = self.coordizer.token_phi.get(token_id, 0.5)
            new_phi = min(1.0, current + boost_amount)
            self.coordizer.token_phi[token_id] = new_phi

    def _record_transition_target(self, basin: np.ndarray, phi: float) -> None:
        """Record basin as preferred transition target for generation."""
        if not hasattr(self, '_transition_targets'):
            self._transition_targets = []

        self._transition_targets.append({
            'basin': basin.tolist() if hasattr(basin, 'tolist') else list(basin),
            'phi': phi,
            'recorded_at': datetime.now().isoformat(),
        })

        # Keep only recent high-phi targets (top 100 by phi)
        self._transition_targets = sorted(
            self._transition_targets,
            key=lambda x: x['phi'],
            reverse=True
        )[:500]

    def get_transition_targets(self, limit: int = 10) -> List[Dict]:
        """Get top transition targets for generation bias."""
        if not hasattr(self, '_transition_targets'):
            return []
        return self._transition_targets[:limit]

    def integrate_pending_vocabulary(self, min_phi: float = 0.65, limit: int = 100) -> Dict:
        """Integrate pending vocabulary from learned_words into active coordizer.

        Queries learned_words where is_integrated = FALSE and avg_phi >= min_phi,
        computes basin_coords for each word, adds them to the coordizer,
        and marks them as integrated with their basin_coords stored.

        Returns:
            Dict with integrated_count, skipped_count, errors
        """
        import os
        import psycopg2

        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            return {'integrated_count': 0, 'error': 'no_database_url'}

        if not self._unified_coordizer:
            return {'integrated_count': 0, 'error': 'no_coordizer'}

        integrated_count = 0
        skipped_count = 0
        errors = []

        try:
            conn = psycopg2.connect(database_url)
            with conn.cursor() as cur:
                # Get unintegrated high-phi words from coordizer_vocabulary
                # NOTE: PROPER_NOUN and BRAND filters REMOVED per user request (2026-01-15)
                cur.execute("""
                    SELECT token as word, phi_score as avg_phi, phi_score as max_phi, frequency, source
                    FROM coordizer_vocabulary
                    WHERE is_real_word = FALSE AND phi_score >= %s
                      AND token_role IN ('generation', 'both')
                    ORDER BY phi_score DESC, frequency DESC
                    LIMIT %s
                """, (min_phi, limit))
                rows = cur.fetchall()

                if not rows:
                    conn.close()
                    return {'integrated_count': 0, 'message': 'no_pending_words'}

                words_to_integrate = []
                words_with_basins = []  # (word, basin_coords) for updating learned_words
                
                for row in rows:
                    word, avg_phi, max_phi, frequency, source = row
                    
                    # Compute basin coords for the word
                    basin_coords = None
                    try:
                        if hasattr(self._unified_coordizer, 'basin_coords') and word in self._unified_coordizer.basin_coords:
                            basin_coords = self._unified_coordizer.basin_coords[word]
                        elif hasattr(self._unified_coordizer, 'encode'):
                            basin_coords = self._unified_coordizer.encode(word)
                    except Exception as e:
                        print(f"[VocabularyCoordinator] Failed to encode '{word}': {e}")
                    
                    if basin_coords is None:
                        skipped_count += 1
                        continue
                    
                    words_to_integrate.append({
                        'word': word,
                        'phi': avg_phi,
                        'max_phi': max_phi,
                        'frequency': frequency,
                        'source': source,
                        'basin_coords': basin_coords,
                    })
                    words_with_basins.append((word, basin_coords))

                if not words_to_integrate:
                    conn.close()
                    return {'integrated_count': 0, 'skipped_count': skipped_count, 'message': 'no_encodable_words'}

                # Add to coordizer via save_learned_token for each word
                for w in words_to_integrate:
                    try:
                        if self._unified_coordizer.save_learned_token(
                            w['word'], w['basin_coords'], w['phi'], w['frequency']
                        ):
                            integrated_count += 1
                    except Exception as e:
                        errors.append(f"save_token_{w['word']}: {e}")

                # Phase 2b: Update coordizer_vocabulary with basin_embedding and mark as generation-ready
                # P0 FIX: Also compute and insert QFI score to prevent incomplete records
                # Also update learned_words for backward compatibility
                for word, basin in words_with_basins:
                    try:
                        basin_list = basin.tolist() if hasattr(basin, 'tolist') else list(basin)
                        
                        # P0 FIX: Compute QFI whenever basin is present
                        qfi_score = compute_qfi_for_basin(basin)
                        
                        # Primary: Update coordizer_vocabulary (consolidated table)
                        # CRITICAL: Include qfi_score in INSERT to prevent NULL qfi_score
                        cur.execute("""
                            INSERT INTO coordizer_vocabulary (
                                token, basin_embedding, qfi_score, token_role, is_real_word, frequency
                            )
                            VALUES (%s, %s::vector, %s, 'generation', TRUE, 1)
                            ON CONFLICT (token) DO UPDATE SET
                                basin_embedding = COALESCE(EXCLUDED.basin_embedding, coordizer_vocabulary.basin_embedding),
                                qfi_score = COALESCE(EXCLUDED.qfi_score, coordizer_vocabulary.qfi_score),
                                token_role = CASE 
                                    WHEN coordizer_vocabulary.token_role = 'encoding' THEN 'both'
                                    ELSE COALESCE(coordizer_vocabulary.token_role, 'generation')
                                END,
                                is_real_word = TRUE,
                                updated_at = NOW()
                        """, (word, str(basin_list), qfi_score))
                    except Exception as e:
                        errors.append(f"update_basin_{word}: {e}")
                
                conn.commit()
                print(f"[VocabularyCoordinator] Integrated {integrated_count} vocabulary terms with basin_coords (min_phi={min_phi})")

        except Exception as e:
            errors.append(f"db_error: {e}")
        finally:
            if 'conn' in locals():
                conn.close()

        return {
            'integrated_count': integrated_count,
            'skipped_count': skipped_count,
            'errors': errors if errors else None,
        }

    def wire_discovery_gate(self) -> None:
        """Wire this coordinator to the chaos discovery gate.

        QIG-PURE WIRING:
        - Vocabulary callback: boost token weights for nearby tokens
        - Attractor callback: record basin as attractor in LearnedManifold
        """
        try:
            from chaos_discovery_gate import get_discovery_gate
            gate = get_discovery_gate()

            # Wire vocabulary integration callback
            gate.set_vocabulary_callback(self.integrate_chaos_discovery)
            print("[VocabularyCoordinator] Vocabulary callback wired to discovery gate")

            # Wire attractor recording callback to LearnedManifold
            if self.learned_manifold is not None:
                gate.set_attractor_callback(self.learned_manifold.record_attractor)
                print("[VocabularyCoordinator] Attractor callback wired to LearnedManifold")
            else:
                print("[VocabularyCoordinator] WARNING: LearnedManifold not available, attractor callback not wired")

        except ImportError as e:
            print(f"[VocabularyCoordinator] Discovery gate not available: {e}")
        except Exception as e:
            print(f"[VocabularyCoordinator] Failed to wire discovery gate: {e}")


_vocabulary_coordinator: Optional[VocabularyCoordinator] = None


def get_vocabulary_coordinator() -> VocabularyCoordinator:
    global _vocabulary_coordinator
    if _vocabulary_coordinator is None:
        _vocabulary_coordinator = VocabularyCoordinator()
    return _vocabulary_coordinator