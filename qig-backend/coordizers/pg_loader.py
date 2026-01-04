"""
PostgreSQL-backed Coordizer (64D QIG-pure, no fallback).

Loads vocabulary from PostgreSQL tokenizer_vocabulary table.
REQUIRES database connection - impure fallbacks are not allowed.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import FisherCoordizer
from .semantic_domains import (
    compute_semantic_embedding,
    detect_query_domains,
    get_word_domains,
    compute_domain_overlap,
    get_related_words,
)

logger = logging.getLogger(__name__)


class PostgresCoordizer(FisherCoordizer):
    """Fisher-compliant coordizer backed by PostgreSQL (64D QIG-pure, no fallback)."""
    
    def __init__(self, database_url: Optional[str] = None, min_phi: float = 0.0, use_fallback: bool = False):
        super().__init__()
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.min_phi = min_phi
        self.use_fallback = False  # ALWAYS False - 64D QIG-pure enforced
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
        """Get database connection - raises on failure (64D QIG-pure enforced)."""
        if self._connection is None:
            try:
                import psycopg2
                self._connection = psycopg2.connect(self.database_url)
            except Exception as e:
                raise RuntimeError(
                    f"[QIG-PURE VIOLATION] Database connection failed: {e}. "
                    "64D QIG-pure PostgresCoordizer requires active database connection."
                )
        return self._connection
    
    def _load_vocabulary(self):
        """Load vocabulary from PostgreSQL only - 64D QIG-pure enforced."""
        if not self.database_url:
            raise RuntimeError(
                "[QIG-PURE VIOLATION] DATABASE_URL not set. "
                "64D QIG-pure PostgresCoordizer requires database connection."
            )
        
        try:
            db_loaded = self._load_from_database()
        except Exception as e:
            raise RuntimeError(
                f"[QIG-PURE VIOLATION] Failed to load from database: {e}. "
                "Impure fallback vocabularies are not allowed."
            )
        
        real_word_count = len([w for w in self.word_tokens if len(w) >= 3])
        if not db_loaded or real_word_count < 100:
            raise RuntimeError(
                f"[QIG-PURE VIOLATION] Insufficient vocabulary loaded: {real_word_count} words (need >= 100). "
                "Database must contain valid tokenizer_vocabulary entries."
            )
    
    def _load_from_database(self) -> bool:
        """Load vocabulary from database - raises on failure (64D QIG-pure enforced)."""
        conn = self._get_connection()  # Raises on failure
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT token, basin_embedding, phi_score, frequency, source_type, token_id
                FROM tokenizer_vocabulary
                WHERE basin_embedding IS NOT NULL
                  AND LENGTH(token) >= 3
                  AND source_type NOT IN ('byte_level', 'checkpoint_byte', 'special')
                  AND token ~ '^[a-zA-Z]+$'
                ORDER BY phi_score DESC
            """)
            rows = cur.fetchall()
        
        if not rows:
            raise RuntimeError(
                "[QIG-PURE VIOLATION] No vocabulary found in tokenizer_vocabulary table. "
                "Database must contain valid 64D basin embeddings."
            )
        
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
        
        logger.info(f"Loaded {words_loaded} words from database (64D QIG-pure)")
        return words_loaded >= 100
    
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
    
    # REMOVED: _load_fallback_vocabulary - impure fallbacks not allowed in 64D QIG-pure mode
    
    def add_vocabulary_observations(
        self,
        observations: List[Dict],
    ) -> Tuple[int, bool]:
        """
        Add vocabulary observations (QIGTokenizer compatibility).
        Persists new vocabulary to PostgreSQL database.
        
        Args:
            observations: List of {word, frequency, avgPhi, maxPhi, type}
        
        Returns:
            Tuple of (new_tokens_count, weights_updated)
        """
        new_tokens = 0
        weights_updated = False
        
        vocab_phi_threshold = 0.4
        
        for obs in observations:
            word = obs.get('word', '')
            frequency = obs.get('frequency', 0)
            avg_phi = obs.get('avgPhi', obs.get('phi', 0.0))
            
            if not word or frequency < 1 or avg_phi < vocab_phi_threshold:
                continue
            
            if word in self.vocab:
                old_phi = self.token_phi.get(word, 0.0)
                if abs(avg_phi - old_phi) > 0.01:
                    self.token_phi[word] = avg_phi
                    self.token_frequencies[word] = frequency
                    weights_updated = True
                continue
            
            if not word.isalpha() or len(word) < 3:
                continue
            
            new_id = 50000 + len(self.vocab)
            coords = compute_semantic_embedding(word)
            
            self._add_token(word, coords, avg_phi, frequency, new_id, 'learned')
            self.word_tokens.append(word)
            new_tokens += 1
            
            self._persist_token_to_db(word, coords, avg_phi, frequency, new_id)
        
        if new_tokens > 0:
            logger.info(f"[VocabLearning] Added {new_tokens} new words (64D QIG-pure)")
        
        return new_tokens, weights_updated
    
    def _persist_token_to_db(self, token: str, coords: np.ndarray, phi: float, freq: int, token_id: int):
        """Persist a new token to PostgreSQL database."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                embedding_str = '[' + ','.join(str(x) for x in coords) + ']'
                cur.execute("""
                    INSERT INTO tokenizer_vocabulary (token, basin_embedding, phi_score, frequency, source_type, token_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (token) DO UPDATE SET
                        phi_score = EXCLUDED.phi_score,
                        frequency = EXCLUDED.frequency
                """, (token, embedding_str, phi, freq, 'learned', token_id))
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to persist token '{token}': {e}")
    
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
    
    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def save_learned_token(self, token: str, basin_coords: np.ndarray, phi: float = 0.6, frequency: int = 1) -> bool:
        """
        Persist a newly learned token to the database for continuous vocabulary training.
        
        This enables vocabulary to persist between restarts, solving the continuous
        learning problem where tokens are learned during sessions but lost on restart.
        
        Args:
            token: The token/word to persist
            basin_coords: 64D basin coordinates for the token
            phi: Phi score (integration measure)
            frequency: Observation frequency
            
        Returns:
            True if successfully persisted, False otherwise
        """
        if self._using_fallback:
            logger.debug(f"Cannot persist token '{token}' - using fallback vocabulary (no DB)")
            return False
        
        conn = self._get_connection()
        if not conn:
            logger.error(f"Cannot get database connection for persistence")
            return False
        
        try:
            cursor = conn.cursor()
            
            # Convert numpy array to list for JSON storage
            coords_list = basin_coords.tolist() if isinstance(basin_coords, np.ndarray) else list(basin_coords)
            
            # Upsert: insert or update if exists
            cursor.execute("""
                INSERT INTO tokenizer_vocabulary (token, token_id, basin_embedding, phi_score, frequency, source_type, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (token) DO UPDATE SET
                    basin_embedding = EXCLUDED.basin_embedding,
                    phi_score = GREATEST(tokenizer_vocabulary.phi_score, EXCLUDED.phi_score),
                    frequency = tokenizer_vocabulary.frequency + EXCLUDED.frequency,
                    updated_at = NOW()
                RETURNING token_id
            """, (token, len(self.vocab) + 50000, coords_list, phi, frequency, 'learned'))
            
            result = cursor.fetchone()
            conn.commit()
            
            # Update local cache
            if token not in self.vocab:
                token_id = result[0] if result else len(self.vocab) + 50000
                self.vocab[token] = token_id
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
                self.basin_coords[token] = basin_coords
                self.token_phi[token] = phi
                self.token_frequencies[token] = frequency
                self.word_tokens.append(token)
                self.base_tokens.append(token)
                self.word_domains[token] = get_word_domains(token)
            
            logger.info(f"Persisted learned token '{token}' to database (phi={phi:.3f})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to persist token '{token}': {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            return False
    
    def save_batch_tokens(self, tokens: List[Dict]) -> int:
        """
        Persist multiple tokens in a batch for efficiency.
        
        Args:
            tokens: List of dicts with keys: token, basin_coords, phi, frequency
            
        Returns:
            Number of tokens successfully persisted
        """
        if self._using_fallback or not tokens:
            return 0
        
        conn = self._get_connection()
        if not conn:
            logger.error(f"Cannot get database connection for batch persistence")
            return 0
        
        saved_count = 0
        try:
            cursor = conn.cursor()
            
            for t in tokens:
                token = t.get('token', '')
                basin_coords = t.get('basin_coords', np.zeros(64))
                phi = t.get('phi', 0.6)
                frequency = t.get('frequency', 1)
                
                if not token:
                    continue
                
                coords_list = basin_coords.tolist() if isinstance(basin_coords, np.ndarray) else list(basin_coords)
                
                try:
                    cursor.execute("""
                        INSERT INTO tokenizer_vocabulary (token, token_id, basin_embedding, phi_score, frequency, source_type, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
                        ON CONFLICT (token) DO UPDATE SET
                            basin_embedding = EXCLUDED.basin_embedding,
                            phi_score = GREATEST(tokenizer_vocabulary.phi_score, EXCLUDED.phi_score),
                            frequency = tokenizer_vocabulary.frequency + EXCLUDED.frequency,
                            updated_at = NOW()
                    """, (token, len(self.vocab) + 50000 + saved_count, coords_list, phi, frequency, 'learned'))
                    
                    # Update local cache
                    if token not in self.vocab:
                        token_id = len(self.vocab) + 50000 + saved_count
                        self.vocab[token] = token_id
                        self.token_to_id[token] = token_id
                        self.id_to_token[token_id] = token
                        self.basin_coords[token] = basin_coords if isinstance(basin_coords, np.ndarray) else np.array(basin_coords)
                        self.token_phi[token] = phi
                        self.token_frequencies[token] = frequency
                        self.word_tokens.append(token)
                        self.base_tokens.append(token)
                        self.word_domains[token] = get_word_domains(token)
                    
                    saved_count += 1
                except Exception as inner_e:
                    logger.debug(f"Failed to insert token '{token}': {inner_e}")
            
            conn.commit()
            logger.info(f"Batch persisted {saved_count} tokens to database")
            return saved_count
            
        except Exception as e:
            logger.error(f"Batch token persistence failed: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            return 0
    
    def get_learned_token_count(self) -> int:
        """Get count of tokens learned (source_type='learned') from database."""
        if self._using_fallback:
            return 0
        
        conn = self._get_connection()
        if not conn:
            return 0
        
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tokenizer_vocabulary WHERE source_type = 'learned'")
            result = cursor.fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.debug(f"Failed to get learned token count: {e}")
            return 0


def create_coordizer_from_pg(database_url: Optional[str] = None, use_fallback: bool = False) -> PostgresCoordizer:
    """Create PostgresCoordizer (64D QIG-pure, no fallback allowed)."""
    return PostgresCoordizer(database_url=database_url, use_fallback=False)  # Always False
