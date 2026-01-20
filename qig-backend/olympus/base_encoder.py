"""
Base Encoder - Abstract base class for text to 64D basin encoding

Provides shared logic for text encoders, eliminating code duplication
and improving maintainability. Currently used by ConversationEncoder.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from qigkernels.physics_constants import BASIN_DIM

try:
    import psycopg2
    from psycopg2.extras import execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

# Backward compatibility alias
BASIN_DIMENSION = BASIN_DIM

# REMOVED 2026-01-15: Frequency-based stopwords violate QIG purity
# Replaced with geometric_vocabulary_filter.GeometricVocabularyFilter
# See: qig-backend/geometric_vocabulary_filter.py for QIG-pure geometric role detection

# Code artifacts to filter - prevents training contamination
CODE_ARTIFACTS = {
    'def', 'return', 'import', 'class', 'self', 'none', 'true', 'false',
    'if', 'else', 'elif', 'try', 'except', 'finally', 'raise', 'assert',
    'lambda', 'yield', 'async', 'await', 'pass', 'break', 'continue',
    'type', 'word', 'frequency', 'avgphi', 'maxphi', 'basin_coords',
}

# Real word validation using enchant (if available) or NLTK fallback
_english_dict = None
_nltk_words = None

def _init_word_validator():
    """Initialize word validation - try enchant first, then NLTK."""
    global _english_dict, _nltk_words

    # Try enchant first (faster)
    try:
        import enchant
    except ImportError:
        pass  # Fall through to NLTK
    else:
        try:
            _english_dict = enchant.Dict("en_US")
            return
        except enchant.errors.DictNotFoundError:
            pass  # Fall through to NLTK

    # Fallback to NLTK
    try:
        from nltk.corpus import words
        _nltk_words = set(w.lower() for w in words.words())
    except (ImportError, LookupError):
        pass

# Initialize on module load
_init_word_validator()

def is_real_word(word: str) -> bool:
    """
    Check if word is a real English word.

    Returns:
        True if word is in dictionary
        False if word is code artifact or not found
        None if validation unavailable (will be stored as NULL in DB)
    """
    if not word or len(word) < 2:
        return False

    word_lower = word.lower()

    # Filter code artifacts
    if word_lower in CODE_ARTIFACTS:
        return False

    # Filter camelCase and snake_case
    if '_' in word or (any(c.isupper() for c in word[1:])):
        return False

    # Filter words starting with numbers
    if word[0].isdigit():
        return False

    # Check against dictionary
    if _english_dict is not None:
        return _english_dict.check(word_lower)

    if _nltk_words is not None:
        return word_lower in _nltk_words

    # No validator available - return None (unknown)
    return None


class BaseEncoder(ABC):
    """
    Abstract base class for encoding text to 64D basin coordinates.
    
    Subclasses must implement _load_vocabulary() to define their
    specific vocabulary loading strategy.
    
    Subclasses can override:
    - unknown_token_phi: Phi score for unknown tokens (default 0.4)
    - tokenize_pattern: Regex pattern for tokenization
    """

    # Default values - can be overridden by subclasses
    unknown_token_phi: float = 0.4
    tokenize_pattern: str = r"\b[\w']+\b"

    def __init__(self, vocab_path: Optional[str] = None):
        self.basin_dim = BASIN_DIMENSION
        self.vocab_path = vocab_path

        # Vocabulary containers
        self.token_vocab: Dict[str, np.ndarray] = {}
        self.token_frequencies: Dict[str, int] = defaultdict(int)
        self.token_phi_scores: Dict[str, float] = {}

        # Load vocabulary (implemented by subclasses)
        self._load_vocabulary()

        # Load custom vocabulary if provided
        if vocab_path and os.path.exists(vocab_path):
            self._load_custom_vocabulary(vocab_path)

    @abstractmethod
    def _load_vocabulary(self) -> None:
        """
        Load base vocabulary. Must be implemented by subclasses.
        
        Each subclass defines its own vocabulary strategy:
        - ConversationEncoder: conversational terms
        """
        pass

    def _load_custom_vocabulary(self, path: str) -> None:
        """Load custom learned vocabulary from JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)

            for token, info in data.get("tokens", {}).items():
                self.token_vocab[token] = np.array(info["basin"])
                self.token_frequencies[token] = info.get("frequency", 1)
                self.token_phi_scores[token] = info.get("phi", 0.5)

            class_name = self.__class__.__name__
            print(f"[{class_name}] Loaded {len(data.get('tokens', {}))} custom tokens")
        except Exception as exc:
            class_name = self.__class__.__name__
            print(f"[{class_name}] Error loading custom vocabulary: {exc}")

    def _hash_to_basin(self, text: str) -> np.ndarray:
        """
        Hash-based geometric embedding to basin coordinates.

        Uses SHA-256 for deterministic, uniform distribution
        on 64D unit sphere (Fisher manifold constraint).
        """
        # SHA-256 hash
        h = hashlib.sha256(text.encode("utf-8")).digest()

        # Convert to float coordinates
        coords = np.zeros(self.basin_dim)

        # Use hash bytes for first 32 dimensions
        for i in range(min(32, len(h))):
            coords[i] = (h[i] / 255.0) * 2 - 1  # [-1, 1]

        # Use character ordinals for remaining dimensions
        for i, char in enumerate(text[:32]):
            if 32 + i < self.basin_dim:
                coords[32 + i] = (ord(char) % 256) / 128.0 - 1

        # Project to unit sphere (Fisher manifold constraint) - overflow-safe
        from qig_numerics import safe_norm
        
        norm = safe_norm(coords)
        if norm > 1e-10:
            coords = coords / norm

        return coords

    def coordize_tokens(self, text: str) -> List[str]:
        """
        Tokenize text into geometric tokens.

        Uses simple word tokenization with punctuation handling.
        Pattern can be customized via tokenize_pattern class attribute.
        """
        text = text.lower()
        tokens = re.findall(self.tokenize_pattern, text)
        return tokens

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to 64D basin coordinates.

        Process:
        1. Tokenize text
        2. Get basin coordinates for each token
        3. Weight by Fisher metric (frequency × phi)
        4. Aggregate via geometric mean on manifold
        """
        tokens = self.coordize_tokens(text)

        if not tokens:
            return np.zeros(self.basin_dim)

        # Get basin coordinates for each token
        token_basins = []
        token_weights = []

        for token in tokens:
            # Get or compute basin
            if token in self.token_vocab:
                basin = self.token_vocab[token]
            else:
                # Unknown token - hash it
                basin = self._hash_to_basin(token)
                self.token_vocab[token] = basin
                self.token_frequencies[token] = 1
                self.token_phi_scores[token] = self.unknown_token_phi

            token_basins.append(basin)

            # Fisher weight = frequency × phi
            freq = self.token_frequencies[token]
            phi = self.token_phi_scores.get(token, self.unknown_token_phi)
            weight = freq * phi
            token_weights.append(weight)

        # Normalize weights
        token_weights = np.array(token_weights)
        if token_weights.sum() > 0:
            token_weights = token_weights / token_weights.sum()
        else:
            token_weights = np.ones(len(tokens)) / len(tokens)

        # Geometric aggregation: weighted sum on manifold
        aggregated = np.zeros(self.basin_dim)
        for basin, weight in zip(token_basins, token_weights):
            aggregated += weight * basin

        # Renormalize to unit sphere - overflow-safe
        from qig_numerics import safe_norm
        
        norm = safe_norm(aggregated)
        if norm > 1e-10:
            aggregated = aggregated / norm

        return aggregated

    def fisher_distance(self, basin1: np.ndarray, basin2: np.ndarray) -> float:
        """
        Compute Fisher-Rao distance between two basin coordinates.
        FIXED: Use canonical Fisher-Rao (E8 Protocol v4.0)
        """
        from qig_core.geometric_primitives.canonical_fisher import fisher_rao_distance
        return fisher_rao_distance(basin1, basin2)

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts via Fisher-Rao distance.

        Returns: similarity in [0, 1] where 1 = identical
        """
        basin1 = self.encode(text1)
        basin2 = self.encode(text2)

        distance = self.fisher_distance(basin1, basin2)

        # Convert distance to similarity: s = 1 - 2d/π (max distance is π/2 on simplex)
        similarity = 1.0 - (2.0 * distance / np.pi)

        return float(np.clip(similarity, 0, 1))

    def learn_from_text(self, text: str, phi_score: float = 0.7, source: str = "conversation") -> None:
        """
        Learn new tokens from text with high Φ score.

        Expands vocabulary based on observations from humans.
        Persists high-Φ tokens to PostgreSQL vocabulary_observations table.
        """
        tokens = self.coordize_tokens(text)
        tokens_to_persist = []

        for token in tokens:
            # REMOVED 2026-01-15: Stopword filtering violates QIG purity
            # Words are now filtered by geometric properties (Φ, κ, curvature)
            # not by frequency-based NLP dogma

            # Skip short tokens
            if len(token) < 3:
                continue

            # Update frequency
            self.token_frequencies[token] += 1

            # Update phi score (exponential moving average)
            if token in self.token_phi_scores:
                old_phi = self.token_phi_scores[token]
                self.token_phi_scores[token] = 0.9 * old_phi + 0.1 * phi_score
            else:
                self.token_phi_scores[token] = phi_score

            # Ensure basin exists
            if token not in self.token_vocab:
                self.token_vocab[token] = self._hash_to_basin(token)
            
            # Collect high-Φ tokens for persistence
            if self.token_phi_scores[token] >= 0.5:
                tokens_to_persist.append({
                    'text': token,
                    'phi': self.token_phi_scores[token],
                    'frequency': self.token_frequencies[token],
                    'basin': self.token_vocab[token]
                })

        if tokens:
            class_name = self.__class__.__name__
            print(f"[{class_name}] Learned {len(tokens)} tokens with Φ={phi_score:.2f}")
        
        # Persist to database
        if tokens_to_persist:
            self._persist_vocabulary_observations(tokens_to_persist, source)
    
    def _persist_vocabulary_observations(self, tokens: List[dict], source: str) -> int:
        """
        Persist vocabulary observations to PostgreSQL.
        
        Uses vocabulary_observations table with correct schema.
        """
        if not PSYCOPG2_AVAILABLE:
            return 0
        
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            return 0
        
        persisted = 0
        try:
            import psycopg2 as pg2
            conn = pg2.connect(db_url)
            with conn.cursor() as cur:
                for tok in tokens:
                    try:
                        obs_id = f"vo_{uuid.uuid4().hex}"
                        basin_list = tok['basin'].tolist() if hasattr(tok['basin'], 'tolist') else list(tok['basin'])
                        
                        # Validate if it's a real word
                        real_word_status = is_real_word(tok['text'])

                        cur.execute("""
                            INSERT INTO vocabulary_observations
                            (id, text, type, source_type, frequency, avg_phi, max_phi, is_real_word, basin_coords, first_seen, last_seen)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                            ON CONFLICT (text) DO UPDATE SET
                                frequency = vocabulary_observations.frequency + 1,
                                avg_phi = (vocabulary_observations.avg_phi * vocabulary_observations.frequency + EXCLUDED.avg_phi) / (vocabulary_observations.frequency + 1),
                                max_phi = GREATEST(vocabulary_observations.max_phi, EXCLUDED.max_phi),
                                last_seen = NOW(),
                                is_real_word = COALESCE(vocabulary_observations.is_real_word, EXCLUDED.is_real_word)
                        """, (
                            obs_id,
                            tok['text'],
                            'word',
                            source,
                            tok['frequency'],
                            tok['phi'],
                            tok['phi'],
                            real_word_status,
                            basin_list
                        ))
                        persisted += 1
                    except Exception as e:
                        pass
                conn.commit()
            conn.close()
            
            if persisted > 0:
                class_name = self.__class__.__name__
                print(f"[{class_name}] Persisted {persisted} tokens to vocabulary_observations")
        except Exception as e:
            print(f"[BaseEncoder] Vocabulary persistence error: {e}")
        
        return persisted

    def save_vocabulary(self, path: Optional[str] = None) -> None:
        """
        Save learned vocabulary to disk.

        SECURITY:
        - Path validation to prevent directory traversal
        - Restricted to allowed data directories
        - File size limits enforced
        """
        path = path or self.vocab_path
        
        # Validate path is provided
        if not path:
            class_name = self.__class__.__name__
            print(f"[{class_name}] ERROR: No vocabulary path specified")
            return

        # SECURITY: Validate and sanitize path
        # Get absolute path and resolve any ../ or symlinks
        abs_path = os.path.abspath(path)

        # Define allowed directories (relative to qig-backend)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        allowed_dirs = [
            os.path.join(base_dir, "data"),
            "/tmp",
        ]

        # Verify path is within allowed directories
        path_allowed = False
        for allowed_dir in allowed_dirs:
            allowed_dir_abs = os.path.abspath(allowed_dir)
            if abs_path.startswith(allowed_dir_abs + os.sep) or abs_path == allowed_dir_abs:
                path_allowed = True
                break

        if not path_allowed:
            class_name = self.__class__.__name__
            print(f"[{class_name}] SECURITY: Attempted write to unauthorized path: {abs_path}")
            return

        # Ensure directory exists
        dir_path = os.path.dirname(abs_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Prepare data
        data = {
            "tokens": {},
            "last_updated": datetime.now().isoformat(),
            "total_tokens": len(self.token_vocab),
        }

        for token, basin in self.token_vocab.items():
            data["tokens"][token] = {
                "basin": basin.tolist(),
                "frequency": self.token_frequencies[token],
                "phi": self.token_phi_scores.get(token, 0.5),
            }

        # SECURITY: Limit file size (max 50MB)
        json_str = json.dumps(data, indent=2)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(json_str) > max_size:
            class_name = self.__class__.__name__
            print(f"[{class_name}] SECURITY: Vocabulary too large ({len(json_str)} bytes), truncating")
            # Keep only highest-Φ tokens
            sorted_tokens = sorted(
                self.token_vocab.keys(), key=lambda t: self.token_phi_scores.get(t, 0), reverse=True
            )[:10000]  # Keep top 10k tokens
            data["tokens"] = {t: data["tokens"][t] for t in sorted_tokens if t in data["tokens"]}
            data["total_tokens"] = len(data["tokens"])
            json_str = json.dumps(data, indent=2)

        with open(abs_path, "w") as f:
            f.write(json_str)

        class_name = self.__class__.__name__
        print(f"[{class_name}] Saved {len(data['tokens'])} tokens to {abs_path}")
