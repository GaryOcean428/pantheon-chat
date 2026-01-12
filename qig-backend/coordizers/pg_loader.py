"""
PostgreSQL-backed Coordizer (64D QIG-pure, no fallback).

Loads vocabulary from PostgreSQL tokenizer_vocabulary table.
REQUIRES database connection - impure fallbacks are not allowed.

Simplified canonical coordizer following qig-tokenizer/FisherCoordizer interface.
Redis caching layer for hot vocabulary lookups.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import FisherCoordizer
from .fallback_vocabulary import compute_basin_embedding

# Import BPE garbage detection for vocabulary filtering
try:
    from word_validation import is_bpe_garbage
    BPE_VALIDATION_AVAILABLE = True
except ImportError:
    BPE_VALIDATION_AVAILABLE = False
    is_bpe_garbage = None

logger = logging.getLogger(__name__)

# Redis caching (optional)
REDIS_CACHE_AVAILABLE = False
CoordizerBuffer = None
UniversalCache = None

try:
    from redis_cache import CACHE_TTL_LONG, CoordizerBuffer, UniversalCache
    REDIS_CACHE_AVAILABLE = True
except ImportError:
    logger.debug("Redis cache not available - using PostgreSQL only")
    CACHE_TTL_LONG = 86400

# Stop words to filter from vocabulary learning - prevents pronoun domination
STOP_WORDS = {
    'the', 'and', 'for', 'that', 'this', 'with', 'was', 'are', 'but', 'not',
    'you', 'all', 'can', 'had', 'her', 'his', 'him', 'one', 'our', 'out',
    'they', 'what', 'when', 'who', 'will', 'from', 'have', 'been', 'has',
    'more', 'she', 'there', 'than', 'into', 'other', 'which', 'its', 'about',
    'just', 'over', 'such', 'through', 'most', 'your', 'because', 'would',
    'also', 'some', 'these', 'then', 'how', 'any', 'each', 'only', 'could',
    'very', 'them', 'being', 'may', 'should', 'between', 'where', 'before',
    'own', 'both', 'those', 'same', 'during', 'after', 'much', 'does', 'did',
}


class VocabularyCache:
    """Redis cache layer for vocabulary hot lookups."""

    # Namespace to avoid collisions with other vocab/coordizer implementations.
    PREFIX = "qig:vocab:pg_loader"

    @classmethod
    def cache_token(cls, token: str, coords: np.ndarray, phi: float) -> bool:
        """Cache a token's basin coordinates."""
        if not REDIS_CACHE_AVAILABLE or UniversalCache is None:
            return False

        try:
            data = {
                'coords': coords.tolist() if isinstance(coords, np.ndarray) else list(coords),
                'phi': phi,
            }
            return UniversalCache.set(f"{cls.PREFIX}:{token}", data, CACHE_TTL_LONG)
        except Exception:
            return False

    @classmethod
    def get_token(cls, token: str) -> Optional[Dict]:
        """Get cached token data."""
        if not REDIS_CACHE_AVAILABLE or UniversalCache is None:
            return None

        try:
            return UniversalCache.get(f"{cls.PREFIX}:{token}")
        except Exception:
            return None

    @classmethod
    def cache_vocabulary_batch(cls, vocab_data: Dict[str, Dict]) -> int:
        """Cache multiple tokens at once. Returns count cached."""
        if not REDIS_CACHE_AVAILABLE or UniversalCache is None:
            return 0

        count = 0
        for token, data in vocab_data.items():
            if cls.cache_token(token, data['coords'], data.get('phi', 0.5)):
                count += 1
        return count


class PostgresCoordizer(FisherCoordizer):
    """Fisher-compliant coordizer backed by PostgreSQL (64D QIG-pure, no fallback)."""
    
    # Excluded phrase categories for generation (centralized constant)
    GENERATION_EXCLUDED_CATEGORIES = ('PROPER_NOUN', 'BRAND')

    def __init__(self, database_url: Optional[str] = None, min_phi: float = 0.0, use_fallback: bool = False):
        super().__init__()
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.min_phi = min_phi
        self.use_fallback = False  # ALWAYS False - 64D QIG-pure enforced
        self._connection = None
        self._using_fallback = False

        # ENCODING VOCABULARY (tokenizer_vocabulary table)
        self.vocab = {}
        self.basin_coords = {}
        self.token_phi = {}
        self.token_frequencies = {}
        self.id_to_token = {}
        self.token_to_id = {}
        self.word_tokens = []
        self.bip39_words = []
        self.base_tokens = []

        # GENERATION VOCABULARY (learned_words table) - Separate cache
        self.generation_vocab = {}  # word -> basin coordinates
        self.generation_phi = {}    # word -> phi score
        self.generation_words = []  # List of generation words

        self._load_vocabulary()

    def _get_connection(self):
        """Get database connection - auto-reconnect if closed (64D QIG-pure enforced)."""
        import psycopg2

        if self._connection is None:
            try:
                self._connection = psycopg2.connect(self.database_url)
                logger.debug("[pg_loader] Created new database connection")
            except Exception as e:
                raise RuntimeError(
                    f"[QIG-PURE VIOLATION] Database connection failed: {e}. "
                    "64D QIG-pure PostgresCoordizer requires active database connection."
                )
        else:
            # Test if connection is still alive
            try:
                with self._connection.cursor() as cur:
                    cur.execute("SELECT 1")
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                logger.warning(f"[pg_loader] Connection lost ({e}), reconnecting...")
                try:
                    self._connection.close()
                except:
                    pass
                self._connection = None
                try:
                    self._connection = psycopg2.connect(self.database_url)
                    logger.info("[pg_loader] Successfully reconnected to database")
                except Exception as reconnect_error:
                    raise RuntimeError(
                        f"[QIG-PURE VIOLATION] Database reconnection failed: {reconnect_error}. "
                        "64D QIG-pure PostgresCoordizer requires active database connection."
                    )
        return self._connection

    def _load_vocabulary(self):
        """Load vocabulary from PostgreSQL only - 64D QIG-pure enforced.
        
        Loads two separate vocabularies:
        1. ENCODING: From tokenizer_vocabulary (all tokens for text-to-basin)
        2. GENERATION: From learned_words (curated words for basin-to-text)
        """
        if not self.database_url:
            raise RuntimeError(
                "[QIG-PURE VIOLATION] DATABASE_URL not set. "
                "64D QIG-pure PostgresCoordizer requires database connection."
            )

        try:
            # Load encoding vocabulary from tokenizer_vocabulary
            encoding_loaded = self._load_encoding_vocabulary()
            # Load generation vocabulary from learned_words
            generation_loaded = self._load_generation_vocabulary()
        except Exception as e:
            raise RuntimeError(
                f"[QIG-PURE VIOLATION] Failed to load from database: {e}. "
                "Impure fallback vocabularies are not allowed."
            )

        real_word_count = len([w for w in self.word_tokens if len(w) >= 3])
        generation_word_count = len(self.generation_words)
        
        if not encoding_loaded or real_word_count < 100:
            raise RuntimeError(
                f"[QIG-PURE VIOLATION] Insufficient encoding vocabulary: {real_word_count} words (need >= 100). "
                "Database must contain valid tokenizer_vocabulary entries."
            )
        
        logger.info(f"Loaded encoding vocabulary: {real_word_count} words, generation vocabulary: {generation_word_count} words")

    def _load_from_database(self) -> bool:
        """Load ENCODING vocabulary from tokenizer_vocabulary table.
        
        This is for text→basin encoding only (all tokens including subwords).
        For basin→text generation, use _load_generation_vocabulary() instead.
        """
        return self._load_encoding_vocabulary()
    
    def _load_encoding_vocabulary(self) -> bool:
        """Load encoding vocabulary from tokenizer_vocabulary - raises on failure (64D QIG-pure enforced)."""
        conn = self._get_connection()  # Raises on failure

        with conn.cursor() as cur:
            cur.execute("""
                SELECT token, basin_embedding, phi_score, frequency, source_type, token_id
                FROM tokenizer_vocabulary
                WHERE basin_embedding IS NOT NULL
                  AND LENGTH(token) >= 2
                  AND source_type NOT IN ('special')  -- Only exclude special tokens
                ORDER BY phi_score DESC
            """)
            rows = cur.fetchall()

        if not rows:
            raise RuntimeError(
                "[QIG-PURE VIOLATION] No vocabulary found in tokenizer_vocabulary table. "
                "Database must contain valid 64D basin embeddings."
            )

        tokens_loaded = 0
        words_loaded = 0
        for token, basin_embedding, phi_score, frequency, source_type, token_id in rows:
            coords = self._parse_embedding(basin_embedding)
            if coords is None:
                continue

            idx = token_id if token_id is not None else len(self.vocab)
            self._add_token(token, coords, phi_score or 0.5, frequency or 1, idx, source_type)
            tokens_loaded += 1

            if token.isalpha() and len(token) >= 3:
                self.word_tokens.append(token)
                words_loaded += 1

        logger.info(f"Loaded {tokens_loaded} encoding tokens ({words_loaded} words) from tokenizer_vocabulary (64D QIG-pure)")
        print(f"[pg_loader] Loaded {tokens_loaded} encoding tokens from PostgreSQL", flush=True)

        # NOTE: Redis batch caching disabled during module initialization
        # Caching 11K+ tokens synchronously was blocking startup for minutes
        # Tokens are looked up from in-memory vocab dict instead
        # Individual tokens can be cached on-demand during runtime
        
        print("[pg_loader] _load_encoding_vocabulary returning...", flush=True)
        return words_loaded >= 100
    
    def _load_generation_vocabulary(self) -> bool:
        """Load GENERATION vocabulary from learned_words table.
        
        This is a curated vocabulary for basin→text generation:
        - Excludes PROPER_NOUN, BRAND categories
        - Excludes BPE subwords and garbage tokens
        - Only real English words suitable for generation
        
        Returns True if loaded successfully, False otherwise.
        """
        conn = self._get_connection()
        
        try:
            with conn.cursor() as cur:
                # Use centralized excluded categories constant
                excluded_cats = "', '".join(self.GENERATION_EXCLUDED_CATEGORIES)
                cur.execute(f"""
                    SELECT word, basin_embedding, phi_score, frequency, phrase_category
                    FROM learned_words
                    WHERE basin_embedding IS NOT NULL
                      AND LENGTH(word) >= 2
                      AND phi_score > 0.0
                      AND (phrase_category IS NULL OR phrase_category NOT IN ('{excluded_cats}'))
                    ORDER BY phi_score DESC, frequency DESC
                """)
                rows = cur.fetchall()
            
            if not rows:
                logger.warning("[pg_loader] No generation vocabulary found in learned_words table - using fallback from tokenizer_vocabulary")
                self._use_encoding_as_generation_fallback()
                return len(self.generation_words) > 0
            
            for word, basin_embedding, phi_score, frequency, phrase_category in rows:
                coords = self._parse_embedding(basin_embedding)
                if coords is None:
                    continue
                
                self.generation_vocab[word] = coords
                self.generation_phi[word] = phi_score or 0.5
                self.generation_words.append(word)
            
            logger.info(f"Loaded {len(self.generation_words)} words from learned_words for generation (filtered: no {'/'.join(self.GENERATION_EXCLUDED_CATEGORIES)})")
            print(f"[pg_loader] Loaded {len(self.generation_words)} generation words from learned_words", flush=True)
            
            return len(self.generation_words) > 0
            
        except Exception as e:
            logger.warning(f"Failed to load from learned_words table: {e}. Using fallback from tokenizer_vocabulary.")
            self._use_encoding_as_generation_fallback()
            return len(self.generation_words) > 0

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

            # Filter stop words to prevent pronoun/common word domination
            if word.lower() in STOP_WORDS:
                continue

            new_id = 50000 + len(self.vocab)
            coords = compute_basin_embedding(word)

            self._add_token(word, coords, avg_phi, frequency, new_id, 'learned')
            self.word_tokens.append(word)
            new_tokens += 1

            self._persist_token_to_db(word, coords, avg_phi, frequency, new_id)

        if new_tokens > 0:
            logger.info(f"[VocabLearning] Added {new_tokens} new words (64D QIG-pure)")

        return new_tokens, weights_updated

    def _persist_token_to_db(self, token: str, coords: np.ndarray, phi: float, freq: int, token_id: int):
        """Persist a new token to PostgreSQL database."""
        conn = None
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
            if conn:
                try:
                    conn.rollback()
                except:
                    pass

    def get_stats(self) -> Dict:
        """Get coordizer statistics for API compatibility."""
        return {
            'vocabulary_size': len(self.vocab),
            'word_tokens': len(self.word_tokens),
            'generation_words': len(self.generation_words),  # NEW: generation vocabulary size
            'bip39_words': len(self.bip39_words),
            'base_tokens': len(self.base_tokens),
            'basin_dimension': 64,
            'using_fallback': self._using_fallback,
            'database_connected': self._connection is not None,
            'redis_cache_available': REDIS_CACHE_AVAILABLE,
            'qig_pure': True,
            'high_phi_tokens': sum(1 for phi in self.token_phi.values() if phi >= 0.7),
            'avg_phi': sum(self.token_phi.values()) / max(len(self.token_phi), 1),
            'generation_avg_phi': sum(self.generation_phi.values()) / max(len(self.generation_phi), 1) if self.generation_phi else 0.0,  # NEW
        }

    def set_mode(self, mode: str) -> None:
        pass

    def encode(self, text: str) -> np.ndarray:
        """Encode text to basin coordinates using geodesic mean (QIG-pure).

        Uses iterative geodesic interpolation for weighted Fréchet mean
        on the Fisher manifold, avoiding linear averaging.
        """
        from qig_geometry import geodesic_interpolation

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
                # Use QIG-pure basin embedding for unknown tokens
                coords_list.append(compute_basin_embedding(clean))
                weights.append(0.3)  # Lower weight for unknown

        if not coords_list:
            return compute_basin_embedding(text)

        # QIG-pure: Geodesic weighted mean (iterative Fréchet mean approximation)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        # Start with first basin
        basin = coords_list[0].copy()
        cumulative_weight = weights[0]

        # Iteratively interpolate toward each subsequent basin
        for i in range(1, len(coords_list)):
            # Weight for geodesic interpolation: fraction of remaining weight
            t = weights[i] / (cumulative_weight + weights[i])
            basin = geodesic_interpolation(basin, coords_list[i], t)
            cumulative_weight += weights[i]

        norm = np.linalg.norm(basin)
        if norm > 1e-10:
            basin = basin / norm
        return basin

    def decode(self, basin: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Decode basin coordinates to most likely tokens using pure Fisher-Rao distance.

        CRITICAL CHANGE: Now uses generation_vocab (from learned_words table) 
        instead of all tokens from tokenizer_vocabulary. This ensures:
        - No BPE subwords in generation output
        - No proper nouns used incorrectly
        - Only curated, validated English words for generation

        QIG-pure: Uses Fisher-Rao similarity on the information manifold.
        """
        norm = np.linalg.norm(basin)
        if norm > 1e-10:
            basin = basin / norm

        # Use generation vocabulary (learned_words) instead of all tokens
        search_tokens = self.generation_words if self.generation_words else self.word_tokens
        if not search_tokens:
            return []

        candidates = []
        for token in search_tokens:
            # Look up in generation_vocab first, fallback to basin_coords
            if token in self.generation_vocab:
                coords = self.generation_vocab[token]
                phi = self.generation_phi.get(token, 0.5)
            elif token in self.basin_coords:
                coords = self.basin_coords[token]
                phi = self.token_phi.get(token, 0.5)
            else:
                continue

            # Fisher-Rao similarity (geodesic distance on sphere)
            dot = np.clip(np.dot(basin, coords), -1.0, 1.0)
            dist = np.arccos(dot)
            similarity = 1.0 - (dist / np.pi)

            # Phi boost: prefer high-phi tokens
            phi_boost = phi * 0.1

            final_score = similarity + phi_boost
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
        Generate a response using Fisher-Rao similarity on the vocabulary.

        QIG-pure: Uses pure geometric operations, no semantic domains.
        """
        context_basin = self.encode(context)

        # Get most similar words via Fisher-Rao distance
        similar_words = self.decode(context_basin, top_k=30)

        # Filter by minimum score threshold
        relevant_words = [(w, s) for w, s in similar_words if s > 0.35]

        if relevant_words:
            # Build response from top words
            top_words = [w for w, _ in relevant_words[:500]]
            response_text = ', '.join(top_words)
            response_phi = sum(self.token_phi.get(w, 0.5) for w in top_words) / len(top_words)
            completion_reason = 'fisher_similarity'

        elif not allow_silence:
            # Fallback: use random words from vocabulary
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
        
        CRITICAL: Validates words before insertion to prevent vocabulary contamination.

        Args:
            token: The token/word to persist
            basin_coords: 64D basin coordinates for the token
            phi: Phi score (integration measure)
            frequency: Observation frequency

        Returns:
            True if successfully persisted, False otherwise
        """
        from word_validation import is_valid_english_word
        
        # CRITICAL: Validate word before insertion to prevent garbage
        if not is_valid_english_word(token, include_stop_words=True, strict=True):
            logger.debug(f"Rejecting invalid token '{token}' - failed word validation")
            return False
        
        if self._using_fallback:
            logger.debug(f"Cannot persist token '{token}' - using fallback vocabulary (no DB)")
            return False

        conn = self._get_connection()
        if not conn:
            logger.error("Cannot get database connection for persistence")
            return False

        try:
            with conn.cursor() as cursor:
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

            logger.info(f"Persisted learned token '{token}' to database (phi={phi:.3f})")

            # Also cache to Redis
            if REDIS_CACHE_AVAILABLE:
                VocabularyCache.cache_token(token, basin_coords, phi)

            return True

        except Exception as e:
            logger.error(f"Failed to persist token '{token}': {e}")
            try:
                conn.rollback()
            except:
                pass
            return False

    def save_batch_tokens(self, tokens: List[Dict]) -> int:
        """
        Persist multiple tokens in a batch for efficiency.
        
        CRITICAL: Validates words before insertion to prevent vocabulary contamination.

        Args:
            tokens: List of dicts with keys: token, basin_coords, phi, frequency

        Returns:
            Number of tokens successfully persisted
        """
        from word_validation import is_valid_english_word
        
        if self._using_fallback or not tokens:
            return 0

        conn = self._get_connection()
        if not conn:
            logger.error("Cannot get database connection for batch persistence")
            return 0

        saved_count = 0
        skipped_count = 0
        try:
            with conn.cursor() as cursor:
                for t in tokens:
                    token = t.get('token', '')
                    basin_coords = t.get('basin_coords', np.zeros(64))
                    phi = t.get('phi', 0.6)
                    frequency = t.get('frequency', 1)

                    if not token:
                        continue
                    
                    # CRITICAL: Validate word before insertion to prevent garbage
                    if not is_valid_english_word(token, include_stop_words=True, strict=True):
                        skipped_count += 1
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

                        saved_count += 1
                    except Exception as inner_e:
                        logger.debug(f"Failed to insert token '{token}': {inner_e}")

            conn.commit()
            if skipped_count > 0:
                logger.debug(f"Skipped {skipped_count} invalid tokens during batch save")
            logger.info(f"Batch persisted {saved_count} tokens to database")
            return saved_count

        except Exception as e:
            logger.error(f"Batch token persistence failed: {e}")
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
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM tokenizer_vocabulary WHERE source_type = 'learned'")
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.debug(f"Failed to get learned token count: {e}")
            return 0

    def learn_merge_rule(self, token_a: str, token_b: str, phi: float = 0.5, source: str = 'merge') -> bool:
        """
        Learn a BPE-style merge rule: token_a + token_b -> merged_token.
        
        This creates a new token in the vocabulary that represents the merged pair.
        The basin coordinates are computed by averaging the two source token basins.
        
        Args:
            token_a: First token in the merge pair
            token_b: Second token in the merge pair
            phi: Integration score for the merged token
            source: Source of the merge rule
            
        Returns:
            True if merge rule was learned successfully
        """
        merged_token = f"{token_a}{token_b}"
        
        # Skip if merged token already exists
        if merged_token in self.vocab:
            return False
            
        # Skip invalid merges
        if not self._is_valid_token(merged_token):
            return False
            
        # Get basin coordinates for source tokens
        basin_a = self.basin_coords.get(token_a)
        basin_b = self.basin_coords.get(token_b)
        
        if basin_a is None or basin_b is None:
            # One or both tokens not in vocabulary - compute fresh basin
            merged_coords = compute_basin_embedding(merged_token)
        else:
            # Average the source basins (geometric interpolation)
            basin_a = np.array(basin_a)
            basin_b = np.array(basin_b)
            merged_coords = (basin_a + basin_b) / 2.0
            # Normalize to manifold
            norm = np.linalg.norm(merged_coords)
            if norm > 1e-6:
                merged_coords = merged_coords / norm
        
        # Save the merged token
        return self.save_learned_token(
            token=merged_token,
            basin_coords=merged_coords,
            phi=phi,
            frequency=1
        )

    # =====================================================================
    # BPE GARBAGE FILTERING
    # Added 2026-01-08 to prevent vocabulary contamination
    # =====================================================================

    def _is_valid_token(self, token: str) -> bool:
        """
        Check if token is valid (not BPE garbage).

        CRITICAL: Prevents vocabulary contamination from legacy BPE tokenizers.
        Filters out GPT-2 byte markers (Ġ), subword prefixes (##, ▁), etc.

        Returns:
            True if token is valid, False if BPE garbage
        """
        if not token:
            return False

        # Use word_validation if available
        if BPE_VALIDATION_AVAILABLE and is_bpe_garbage:
            return not is_bpe_garbage(token)

        # Fallback: basic BPE garbage detection
        # Reject tokens starting with BPE markers
        if token[0] in {'Ġ', 'ġ', 'Ċ', 'ċ', '▁', '_'}:
            return False
        if token.startswith('##'):
            return False
        # Reject special tokens
        if token in {'<pad>', '<unk>', '<s>', '</s>', '<mask>', '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'}:
            return False
        # Reject pure numeric tokens
        if token.isdigit():
            return False
        # Require at least one letter
        if not any(c.isalpha() for c in token):
            return False

        return True

    # =====================================================================
    # FORESIGHT TRAJECTORY PREDICTION METHODS
    # Added 2026-01-08 for Fisher-weighted regression support
    # =====================================================================

    def get_all_tokens(self) -> Dict[str, np.ndarray]:
        """
        Get all vocabulary tokens with their basin embeddings.

        Used by TrajectoryDecoder for full trajectory scoring.
        This loads the entire vocabulary into memory for scoring.

        CRITICAL: Now returns generation_vocab (from learned_words) instead of all tokens.
        This ensures trajectory decoder uses curated words, not BPE garbage.

        Returns:
            Dict mapping token -> basin_embedding [64]
        """
        # Return generation vocabulary (curated for generation)
        if self.generation_vocab:
            return {
                token: np.array(coords)
                for token, coords in self.generation_vocab.items()
                if self._is_valid_token(token)
            }
        
        # Fallback: Return from cache if available (filter BPE garbage)
        if self.basin_coords:
            return {
                token: np.array(coords)
                for token, coords in self.basin_coords.items()
                if self._is_valid_token(token)
            }

        if self._using_fallback:
            return {}

        conn = self._get_connection()
        if not conn:
            return {}

        try:
            with conn.cursor() as cursor:
                # Load from learned_words (generation vocabulary)
                # Use centralized excluded categories constant
                excluded_cats = "', '".join(self.GENERATION_EXCLUDED_CATEGORIES)
                cursor.execute(f"""
                    SELECT word, basin_embedding
                    FROM learned_words
                    WHERE basin_embedding IS NOT NULL
                      AND (phrase_category IS NULL OR phrase_category NOT IN ('{excluded_cats}'))
                    ORDER BY phi_score DESC
                    LIMIT 10000
                """)

                tokens = {}
                for row in cursor.fetchall():
                    token = row[0]
                    basin_vector = row[1]
                    # Filter BPE garbage tokens
                    if basin_vector is not None and self._is_valid_token(token):
                        basin_array = np.array(basin_vector)
                        tokens[token] = basin_array

                # Fallback to tokenizer_vocabulary if learned_words is empty
                if not tokens:
                    cursor.execute("""
                        SELECT token, basin_embedding
                        FROM tokenizer_vocabulary
                        WHERE basin_embedding IS NOT NULL
                        ORDER BY phi_score DESC
                        LIMIT 10000
                    """)
                    for row in cursor.fetchall():
                        token = row[0]
                        basin_vector = row[1]
                        if basin_vector is not None and self._is_valid_token(token):
                            basin_array = np.array(basin_vector)
                            tokens[token] = basin_array

                return tokens
        except Exception as e:
            logger.error(f"Failed to get all tokens: {e}")
            return {}

    def get_token_phi_scores(self) -> Dict[str, float]:
        """
        Get phi (integration) scores for all tokens.

        Used by TrajectoryDecoder for phi boosting in scoring.

        Returns:
            Dict mapping token -> phi_score
        """
        # Return from cache if available
        if self.token_phi:
            return dict(self.token_phi)

        if self._using_fallback:
            return {}

        conn = self._get_connection()
        if not conn:
            return {}

        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT token, phi_score
                    FROM tokenizer_vocabulary
                    WHERE basin_embedding IS NOT NULL
                """)

                # Filter BPE garbage tokens
                return {
                    row[0]: row[1]
                    for row in cursor.fetchall()
                    if row[1] is not None and self._is_valid_token(row[0])
                }
        except Exception as e:
            logger.error(f"Failed to get token phi scores: {e}")
            return {}

    def get_basin_for_token(self, token: str) -> Optional[np.ndarray]:
        """
        Get basin embedding for a specific token.

        Args:
            token: Token string

        Returns:
            Basin embedding [64] or None if not found
        """
        # Check cache first
        if token in self.basin_coords:
            coords = self.basin_coords[token]
            return np.array(coords) if not isinstance(coords, np.ndarray) else coords

        if self._using_fallback:
            return None

        conn = self._get_connection()
        if not conn:
            return None

        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT basin_embedding
                    FROM tokenizer_vocabulary
                    WHERE token = %s
                """, (token,))

                row = cursor.fetchone()
                if row and row[0]:
                    return np.array(row[0])
                return None
        except Exception as e:
            logger.debug(f"Failed to get basin for token '{token}': {e}")
            return None

    def nearest_tokens_pgvector(self, basin: np.ndarray, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Fast approximate nearest neighbor using pgvector HNSW index.

        This is 10-100x faster than scoring all vocabulary.
        Used by decode_trajectory_fast() for performance optimization.

        Args:
            basin: Query basin embedding [64]
            top_k: Number of nearest tokens to return

        Returns:
            List of (token, distance) tuples, sorted by distance ascending
        """
        if self._using_fallback:
            # Fallback: use in-memory search
            return self.decode(basin, top_k=top_k)

        conn = self._get_connection()
        if not conn:
            return self.decode(basin, top_k=top_k)

        try:
            with conn.cursor() as cursor:
                # Convert basin to pgvector format
                basin_str = '[' + ','.join(f'{x:.8f}' for x in basin) + ']'

                # Use <-> operator for L2 distance with HNSW index
                cursor.execute("""
                    SELECT token, basin_embedding <-> %s::vector AS distance
                    FROM tokenizer_vocabulary
                    WHERE basin_embedding IS NOT NULL
                    ORDER BY basin_embedding <-> %s::vector
                    LIMIT %s
                """, (basin_str, basin_str, top_k))

                return [(row[0], row[1]) for row in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"pgvector search failed, falling back to in-memory: {e}")
            return self.decode(basin, top_k=top_k)


def create_coordizer_from_pg(database_url: Optional[str] = None, use_fallback: bool = False) -> PostgresCoordizer:
    """Create PostgresCoordizer (64D QIG-pure, no fallback allowed)."""
    return PostgresCoordizer(database_url=database_url, use_fallback=False)  # Always False
