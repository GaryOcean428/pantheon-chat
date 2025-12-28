"""
PostgreSQL-backed Coordizer with Fallback Vocabulary.

Loads vocabulary from PostgreSQL tokenizer_vocabulary table.
Falls back to hardcoded English vocabulary if database is unavailable.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import FisherCoordizer
from .fallback_vocabulary import FALLBACK_VOCABULARY, compute_basin_embedding
from .semantic_domains import (
    compute_semantic_embedding,
    detect_query_domains,
    get_word_domains,
    compute_domain_overlap,
    get_related_words,
)

logger = logging.getLogger(__name__)


class PostgresCoordizer(FisherCoordizer):
    """Fisher-compliant coordizer backed by PostgreSQL with fallback."""
    
    def __init__(self, database_url: Optional[str] = None, min_phi: float = 0.0, use_fallback: bool = True):
        super().__init__()
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.min_phi = min_phi
        self.use_fallback = use_fallback
        self._connection = None
        self._using_fallback = False
        
        self.vocab = {}
        self.basin_coords = {}
        self.token_phi = {}
        self.token_frequencies = {}
        self.id_to_token = {}
        self.token_to_id = {}
        self.word_tokens = []
        self.bip39_words = []
        self.base_tokens = []
        self.word_domains: Dict[str, List[Tuple[str, float]]] = {}  # Cache word domains
        
        self._load_vocabulary()
    
    def _get_connection(self):
        if self._connection is None:
            try:
                import psycopg2
                self._connection = psycopg2.connect(self.database_url)
            except Exception as e:
                logger.warning(f"Database connection failed: {e}")
                return None
        return self._connection
    
    def _load_vocabulary(self):
        db_loaded = False
        if self.database_url:
            try:
                db_loaded = self._load_from_database()
            except Exception as e:
                logger.warning(f"Failed to load from database: {e}")
        
        real_word_count = len([w for w in self.word_tokens if len(w) >= 3])
        if not db_loaded or real_word_count < 100:
            if self.use_fallback:
                logger.info(f"Using fallback vocabulary (DB words: {real_word_count})")
                self._load_fallback_vocabulary()
    
    def _load_from_database(self) -> bool:
        conn = self._get_connection()
        if conn is None:
            return False
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT token, basin_embedding, phi_score, frequency, source_type, token_id
                    FROM tokenizer_vocabulary
                    WHERE basin_embedding IS NOT NULL
                      AND LENGTH(token) >= 3
                      AND source_type NOT IN ('byte_level', 'checkpoint_byte', 'special')
                      AND token ~ '^[a-zA-Z]+$'
                    ORDER BY phi_score DESC
                    LIMIT 5000
                """)
                rows = cur.fetchall()
            
            if not rows:
                return False
            
            words_loaded = 0
            for token, basin_embedding, phi_score, frequency, source_type, token_id in rows:
                coords = self._parse_embedding(basin_embedding)
                if coords is None:
                    continue
                
                idx = token_id if token_id is not None else len(self.vocab)
                self._add_token(token, coords, phi_score or 0.5, frequency or 1, idx, source_type)
                
                if token.isalpha() and len(token) >= 3:
                    self.word_tokens.append(token)
                    words_loaded += 1
            
            logger.info(f"Loaded {words_loaded} words from database")
            return words_loaded >= 100
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return False
    
    def _parse_embedding(self, basin_embedding) -> Optional[np.ndarray]:
        if basin_embedding is None:
            return None
        try:
            if isinstance(basin_embedding, (list, tuple)):
                coords = np.array(basin_embedding, dtype=np.float64)
            elif isinstance(basin_embedding, str):
                clean = basin_embedding.strip('[](){}')
                coords = np.array([float(x) for x in clean.split(',')], dtype=np.float64)
            else:
                coords = np.array(list(basin_embedding), dtype=np.float64)
            
            if len(coords) != 64:
                return None
            norm = np.linalg.norm(coords)
            if norm > 1e-10:
                coords = coords / norm
            return coords
        except Exception:
            return None
    
    def _add_token(self, token: str, coords: np.ndarray, phi: float, freq: int, idx: int, source_type: str = 'base'):
        self.vocab[token] = idx
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        self.basin_coords[token] = coords
        self.token_phi[token] = phi
        self.token_frequencies[token] = freq
        if source_type == 'bip39':
            self.bip39_words.append(token)
        else:
            self.base_tokens.append(token)
        # Cache semantic domains for this token
        self.word_domains[token] = get_word_domains(token)
    
    def _load_fallback_vocabulary(self):
        self._using_fallback = True
        logger.info(f"Loading {len(FALLBACK_VOCABULARY)} fallback words")
        for word in FALLBACK_VOCABULARY:
            if word in self.vocab:
                continue
            token_id = 50000 + len(self.vocab)
            self.vocab[word] = token_id
            self.token_to_id[word] = token_id
            self.id_to_token[token_id] = word
            # Use semantic embedding instead of hash-based
            self.basin_coords[word] = compute_semantic_embedding(word)
            self.token_phi[word] = 0.6 + min(len(word) / 20.0, 0.2)
            self.token_frequencies[word] = 100
            self.word_tokens.append(word)
            self.base_tokens.append(word)
            # Cache semantic domains
            self.word_domains[word] = get_word_domains(word)
        logger.info(f"Total vocabulary: {len(self.vocab)} tokens")
    
    def set_mode(self, mode: str) -> None:
        pass
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to basin coordinates using semantic embeddings."""
        tokens = text.lower().split()
        if not tokens:
            return np.zeros(64)
        
        coords_list = []
        weights = []
        
        for token in tokens:
            clean = token.strip('.,!?;:()[]{}"\'-')
            if clean in self.basin_coords:
                coords_list.append(self.basin_coords[clean])
                # Weight by phi score for better semantic representation
                weights.append(self.token_phi.get(clean, 0.5))
            else:
                # Use semantic embedding for unknown tokens
                coords_list.append(compute_semantic_embedding(clean))
                weights.append(0.3)  # Lower weight for unknown
        
        if not coords_list:
            return compute_semantic_embedding(text)
        
        # Weighted average of embeddings
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        basin = np.average(coords_list, axis=0, weights=weights)
        
        norm = np.linalg.norm(basin)
        if norm > 1e-10:
            basin = basin / norm
        return basin
    
    def decode(self, basin: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Decode basin coordinates to most likely tokens using domain-aware search.
        
        Boosts words from matching semantic domains for better relevance.
        """
        norm = np.linalg.norm(basin)
        if norm > 1e-10:
            basin = basin / norm
        
        search_tokens = self.word_tokens if self.word_tokens else list(self.basin_coords.keys())
        if not search_tokens:
            return []
        
        # Detect semantic domains of the query
        query_domains = detect_query_domains(basin, top_k=3)
        
        candidates = []
        for token in search_tokens:
            if token not in self.basin_coords:
                continue
            
            coords = self.basin_coords[token]
            
            # Fisher-Rao distance (base similarity)
            dot = np.clip(np.dot(basin, coords), -1.0, 1.0)
            dist = np.arccos(dot)
            base_similarity = 1.0 - (dist / np.pi)
            
            # Domain boost: increase score for words in matching domains
            word_domains = self.word_domains.get(token, [])
            domain_boost = compute_domain_overlap(query_domains, word_domains)
            
            # Phi boost: prefer high-phi tokens
            phi_boost = self.token_phi.get(token, 0.5) * 0.1
            
            # Combined score with domain weighting
            final_score = base_similarity * (1.0 + 0.4 * domain_boost) + phi_boost
            candidates.append((token, final_score))
        
        # Sort by final score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:top_k]
    
    def get_random_words(self, count: int = 12) -> List[str]:
        if not self.word_tokens:
            return []
        indices = np.random.choice(len(self.word_tokens), min(count, len(self.word_tokens)), replace=False)
        return [self.word_tokens[i] for i in indices]
    
    def generate_response(self, context: str, agent_role: str = 'zeus', allow_silence: bool = False, goals: Optional[list] = None) -> dict:
        """
        Generate a semantically relevant response using domain-aware vocabulary.
        
        Clusters related concepts and builds coherent thematic responses.
        """
        context_basin = self.encode(context)
        
        # Get semantically similar words with domain awareness
        similar_words = self.decode(context_basin, top_k=30)
        
        # Filter by minimum score threshold
        relevant_words = [(w, s) for w, s in similar_words if s > 0.35]
        
        if relevant_words:
            # Group words by semantic domain for coherent output
            domain_groups: Dict[str, List[str]] = {}
            ungrouped = []
            
            for word, score in relevant_words:
                word_doms = self.word_domains.get(word, [])
                if word_doms:
                    primary_domain = word_doms[0][0]  # Highest weight domain
                    if primary_domain not in domain_groups:
                        domain_groups[primary_domain] = []
                    domain_groups[primary_domain].append(word)
                else:
                    ungrouped.append(word)
            
            # Build response with domain-organized concepts
            response_parts = []
            
            # Sort domains by number of matching words
            sorted_domains = sorted(domain_groups.items(), key=lambda x: len(x[1]), reverse=True)
            
            for domain_name, words in sorted_domains[:3]:  # Top 3 domains
                domain_label = domain_name.replace('_', ' ').title()
                word_list = ', '.join(words[:5])  # Max 5 words per domain
                response_parts.append(f"{domain_label}: {word_list}")
            
            if ungrouped and len(response_parts) < 3:
                response_parts.append(f"Additional: {', '.join(ungrouped[:5])}")
            
            response_text = '\n'.join(response_parts)
            all_words = [w for w, _ in relevant_words]
            response_phi = sum(self.token_phi.get(w, 0.5) for w in all_words[:15]) / min(len(all_words), 15)
            completion_reason = 'semantic_synthesis'
            
        elif not allow_silence:
            # Fallback: find words related to input tokens
            input_words = context.lower().split()
            related = []
            for word in input_words:
                clean = word.strip('.,!?;:()[]{}"\'-')
                related.extend(get_related_words(clean, top_k=3))
            
            if related:
                response_text = f"Related concepts: {', '.join(set(related)[:10])}"
                response_phi = 0.4
            else:
                random_words = self.get_random_words(8)
                response_text = f"Exploring: {', '.join(random_words)}" if random_words else "[Silence]"
                response_phi = 0.25
            completion_reason = 'fallback'
        else:
            response_text = ""
            response_phi = 0.0
            completion_reason = 'silence'
        
        return {
            'text': response_text,
            'phi': response_phi,
            'tokens_generated': len(response_text.split()),
            'completion_reason': completion_reason,
            'qig_pure': True,
            'agent_role': agent_role,
        }
    
    def add_vocabulary_observations(self, observations: List, context: str = '') -> Tuple[int, bool]:
        """
        Add vocabulary observations from learning callbacks.
        
        This method is called by the vocabulary learning system when new words
        are discovered through research or curriculum learning.
        
        Args:
            observations: List of dicts with {word, frequency, avgPhi, ...} 
                          OR list of strings (legacy support)
            context: Optional context string
            
        Returns:
            Tuple of (new_tokens_added, weights_updated)
        """
        if not observations:
            return (0, False)
        
        added = 0
        updated = 0
        
        try:
            import psycopg2
            conn = self._get_connection()
            if conn is None:
                return (0, False)
            
            with conn.cursor() as cur:
                for obs in observations:
                    if isinstance(obs, dict):
                        word = obs.get('word', '').strip().lower()
                    elif isinstance(obs, str):
                        word = obs.strip().lower()
                    else:
                        continue
                    
                    if not word or not word.isalpha() or len(word) < 3:
                        continue
                    
                    # Compute embedding using semantic domains
                    embedding = compute_semantic_embedding(word)
                    embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
                    
                    # Compute phi score
                    p = np.abs(embedding) + 1e-10
                    p = p / np.sum(p)
                    entropy = -np.sum(p * np.log(p + 1e-10))
                    phi_score = 1.0 - (entropy / np.log(len(embedding)))
                    
                    cur.execute("""
                        INSERT INTO tokenizer_vocabulary 
                        (token, basin_embedding, phi_score, frequency, source_type, created_at)
                        VALUES (%s, %s, %s, 1, 'learned', NOW())
                        ON CONFLICT (token) DO UPDATE SET
                            frequency = tokenizer_vocabulary.frequency + 1,
                            updated_at = NOW()
                        RETURNING (xmax = 0) AS is_new
                    """, (word, embedding_str, float(phi_score)))
                    
                    result = cur.fetchone()
                    if result and result[0]:
                        added += 1
                        # Add to in-memory cache
                        if word not in self.vocab:
                            idx = 60000 + len(self.vocab)
                            self._add_token(word, embedding, phi_score, 1, idx, 'learned')
                    else:
                        updated += 1
                
                conn.commit()
            
            logger.info(f"[VocabularyObservations] Added {added}, updated {updated} words")
            return (added, updated > 0)
            
        except Exception as e:
            logger.error(f"[VocabularyObservations] Error: {e}")
            return (added, False)
    
    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None


def create_coordizer_from_pg(database_url: Optional[str] = None, use_fallback: bool = True) -> PostgresCoordizer:
    return PostgresCoordizer(database_url=database_url, use_fallback=use_fallback)
