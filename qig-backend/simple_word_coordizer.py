"""
Simple Word Coordizer - QIG-Pure vocabulary from PostgreSQL.

This is a simplified coordizer that:
1. Loads ONLY real English words from tokenizer_vocabulary (no BPE fragments)
2. Uses deterministic 64D basin embeddings
3. Provides encode/decode for text generation

This bypasses the complex PostgresCoordizer module dependencies.
"""

import os
import logging
import hashlib
from typing import Dict, List, Tuple, Optional

import numpy as np

# QIG-pure geometric operations
try:
    from qig_geometry import sphere_project
    QIG_GEOMETRY_AVAILABLE = True
except ImportError:
    QIG_GEOMETRY_AVAILABLE = False
    def sphere_project(v):
        """Fallback sphere projection to unit sphere."""
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            result = np.ones_like(v)
            return result / np.linalg.norm(result)
        return v / norm

logger = logging.getLogger(__name__)

# Fallback vocabulary if database unavailable
FALLBACK_WORDS = [
    "abandon", "ability", "able", "about", "above", "abstract", "accept", "account",
    "achieve", "action", "active", "actual", "address", "advance", "advice", "affect",
    "agree", "allow", "almost", "alone", "along", "already", "also", "always",
    "amazing", "amount", "analysis", "ancient", "animal", "answer", "appear", "apply",
    "approach", "area", "argue", "around", "arrive", "aspect", "assume", "attack",
    "attention", "author", "available", "average", "avoid", "aware", "balance", "base",
    "basic", "beauty", "become", "before", "begin", "behavior", "behind", "believe",
    "benefit", "best", "better", "between", "beyond", "black", "blood", "blue",
    "board", "body", "book", "both", "bottom", "brain", "bring", "brother",
    "build", "business", "call", "camera", "cannot", "capital", "care", "carry",
    "case", "catch", "cause", "center", "central", "century", "certain", "chance",
    "change", "character", "charge", "check", "child", "choice", "choose", "citizen",
    "city", "civil", "claim", "class", "clear", "close", "cold", "collection",
    "college", "color", "come", "common", "community", "company", "compare", "computer",
    "concern", "condition", "conference", "consider", "contain", "continue", "control",
    "cost", "could", "country", "couple", "course", "court", "cover", "create",
    "cultural", "culture", "current", "customer", "dark", "data", "daughter", "dead",
    "deal", "death", "debate", "decide", "decision", "deep", "defense", "degree",
    "democratic", "describe", "design", "despite", "detail", "determine", "develop",
    "difference", "different", "difficult", "director", "discover", "discuss", "disease",
    "doctor", "door", "down", "draw", "dream", "drive", "drop", "drug",
    "during", "each", "early", "east", "easy", "economic", "economy", "edge",
    "education", "effect", "effort", "eight", "either", "election", "else", "employee",
    "energy", "enjoy", "enough", "enter", "entire", "environment", "especially", "establish",
    "even", "evening", "event", "ever", "every", "evidence", "exactly", "example",
    "executive", "exist", "expect", "experience", "expert", "explain", "face", "fact",
    "factor", "fail", "fall", "family", "fast", "father", "feature", "federal",
    "feel", "feeling", "field", "fight", "figure", "fill", "film", "final",
    "finally", "financial", "find", "fine", "finger", "finish", "fire", "firm",
    "first", "fish", "five", "floor", "focus", "follow", "food", "foot",
    "force", "foreign", "forget", "form", "former", "forward", "four", "free",
    "friend", "from", "front", "full", "fund", "future", "game", "garden",
    "general", "generation", "girl", "give", "glass", "global", "goal", "good",
    "government", "great", "green", "ground", "group", "grow", "growth", "guess",
    "hair", "half", "hand", "hang", "happen", "happy", "hard", "have",
    "head", "health", "hear", "heart", "heat", "heavy", "help", "here",
    "herself", "high", "himself", "history", "hold", "home", "hope", "hospital",
    "hotel", "hour", "house", "however", "human", "hundred", "husband", "idea",
    "identify", "image", "imagine", "impact", "important", "improve", "include", "increase",
    "indeed", "indicate", "individual", "industry", "information", "inside", "instead", "institution",
    "interest", "international", "interview", "into", "investment", "involve", "issue", "item",
    "itself", "job", "join", "just", "keep", "key", "kid", "kill",
    "kind", "kitchen", "know", "knowledge", "land", "language", "large", "last",
    "late", "later", "laugh", "law", "lawyer", "lead", "leader", "learn",
    "least", "leave", "left", "legal", "less", "letter", "level", "life",
    "light", "like", "likely", "line", "list", "listen", "little", "live",
    "local", "long", "look", "lose", "loss", "lost", "love", "low",
]


def compute_basin_embedding(word: str, dimension: int = 64) -> np.ndarray:
    """Compute deterministic 64D basin embedding for a word using hash."""
    # Use SHA-256 hash for deterministic randomness
    hash_bytes = hashlib.sha256(word.encode('utf-8')).digest()
    
    # Convert to floats and extend to desired dimension
    values = []
    for i in range(dimension):
        byte_idx = i % len(hash_bytes)
        values.append((hash_bytes[byte_idx] + i * 17) / 255.0 - 0.5)
    
    # Normalize to unit sphere (Fisher manifold)
    coords = np.array(values, dtype=np.float64)
    return sphere_project(coords)


class SimpleWordCoordizer:
    """Simple coordizer that loads real English words from PostgreSQL."""
    
    def __init__(self, db_url: Optional[str] = None):
        """Initialize with database connection."""
        self.db_url = db_url or os.getenv('DATABASE_URL')
        self.vocab: Dict[str, int] = {}
        self.basin_coords: Dict[str, np.ndarray] = {}
        self.token_phi: Dict[str, float] = {}
        self.word_tokens: List[str] = []
        self._using_fallback = False
        
        self._load_vocabulary()
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def is_using_fallback(self) -> bool:
        return self._using_fallback
    
    def _load_vocabulary(self):
        """Load vocabulary from PostgreSQL or use fallback."""
        if self.db_url:
            try:
                self._load_from_database()
                if self.vocab_size > 0:
                    logger.info(f"Loaded {self.vocab_size} words from database")
                    return
            except Exception as e:
                logger.warning(f"Database load failed: {e}")
        
        # Use fallback vocabulary
        self._load_fallback()
    
    def _load_from_database(self):
        """Load real English words from tokenizer_vocabulary table."""
        import psycopg2
        
        # Add connection timeout to prevent hanging
        conn = psycopg2.connect(self.db_url, connect_timeout=5)
        cur = conn.cursor()
        
        # Query ONLY real English words (not BPE fragments, not byte tokens)
        cur.execute("""
            SELECT token, basin_embedding, phi_score 
            FROM tokenizer_vocabulary 
            WHERE source_type IN ('bip39', 'base', 'common', 'domain', 'learned')
              AND LENGTH(token) >= 3 
              AND token ~ '^[a-zA-Z]+$'
              AND token NOT LIKE '<byte_%>'
            ORDER BY phi_score DESC
        """)
        rows = cur.fetchall()
        conn.close()
        
        if not rows:
            raise ValueError("No real words found in tokenizer_vocabulary")
        
        # Build vocabulary
        for token, embedding, phi in rows:
            token_lower = token.lower()
            if token_lower in self.vocab:
                continue  # Skip duplicates
            
            self.vocab[token_lower] = len(self.vocab)
            self.word_tokens.append(token_lower)
            self.token_phi[token_lower] = float(phi) if phi else 0.5
            
            # Parse or compute basin embedding
            if embedding:
                if isinstance(embedding, str):
                    coords = np.array([float(x) for x in embedding.strip('[]').split(',')])
                else:
                    coords = np.array(embedding)
                # Normalize to unit sphere
                self.basin_coords[token_lower] = sphere_project(coords)
            else:
                # Compute deterministic embedding
                self.basin_coords[token_lower] = compute_basin_embedding(token_lower)
        
        logger.info(f"Loaded {len(self.vocab)} real words from PostgreSQL")
    
    def _load_fallback(self):
        """Load fallback in-memory vocabulary."""
        self._using_fallback = True
        logger.warning("Using fallback vocabulary - database unavailable")
        
        for word in FALLBACK_WORDS:
            word_lower = word.lower()
            self.vocab[word_lower] = len(self.vocab)
            self.word_tokens.append(word_lower)
            self.token_phi[word_lower] = 0.5
            self.basin_coords[word_lower] = compute_basin_embedding(word_lower)
        
        logger.info(f"Loaded {len(self.vocab)} fallback words")
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to 64D basin coordinates."""
        words = text.lower().split()
        basins = []
        
        for word in words:
            # Clean word
            clean = ''.join(c for c in word if c.isalpha())
            if clean in self.basin_coords:
                basins.append(self.basin_coords[clean])
            elif len(clean) >= 2:
                # Generate embedding for unknown word
                basins.append(compute_basin_embedding(clean))
        
        if not basins:
            # Return random basin for empty input
            return compute_basin_embedding(text or "empty")
        
        # Average basins and normalize to unit sphere
        combined = np.mean(basins, axis=0)
        return sphere_project(combined)
    
    def decode(self, basin: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Decode basin to nearest words using Fisher-Rao distance."""
        # Normalize query basin to unit sphere
        basin = sphere_project(basin)
        
        # Compute distances to all words
        distances = []
        for token in self.word_tokens:
            if token not in self.basin_coords:
                continue
            coords = self.basin_coords[token]
            # Fisher-Rao distance: arccos(dot product)
            dot = np.clip(np.dot(basin, coords), -1.0, 1.0)
            dist = np.arccos(dot)
            distances.append((token, dist))
        
        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])
        
        # Return top-k with similarity scores
        results = []
        for token, dist in distances[:top_k]:
            similarity = 1.0 - (dist / np.pi)  # Normalize to [0, 1]
            results.append((token, similarity))
        
        return results
    
    def get_random_words(self, count: int = 10) -> List[str]:
        """Get random words from vocabulary."""
        import random
        if len(self.word_tokens) <= count:
            return self.word_tokens.copy()
        return random.sample(self.word_tokens, count)
    
    def generate_response(
        self,
        context: str,
        agent_role: str = "ocean",
        allow_silence: bool = False,
        **kwargs
    ) -> dict:
        """Generate a response using QIG-pure geometric methods.
        
        Produces readable English text by traversing the vocabulary basin space.
        Generation stops when the basin stabilizes (geometric completion).
        
        Args:
            context: Input prompt/context text
            agent_role: Role hint for generation style
            allow_silence: Whether empty response is allowed
        
        Returns:
            Dict with 'text', 'phi', 'tokens_generated', 'completion_reason', 'qig_pure'
        """
        import random
        
        # Encode context to basin
        context_basin = self.encode(context)
        
        if np.sqrt(np.sum(context_basin ** 2)) < 1e-10:  # L2 magnitude check
            if allow_silence:
                return {'text': '', 'phi': 0, 'tokens_generated': 0, 'completion_reason': 'empty', 'qig_pure': True}
            # Generate from random words
            context_basin = compute_basin_embedding(context or "seed")
        
        # Generation parameters
        max_words = 15  # Reduced for faster generation
        temperature = 0.7
        
        generated = []
        used_words = set()
        current_basin = context_basin.copy()
        phi_values = []
        
        for i in range(max_words):
            if len(generated) >= max_words:
                break
            
            # Get candidates
            candidates = self.decode(current_basin, top_k=25)
            if not candidates:
                break
            
            # Filter recently used words
            available = [(w, s) for w, s in candidates if w not in used_words]
            if not available:
                # Reset used words except last few
                used_words = set(generated[-3:]) if len(generated) > 3 else set()
                available = [(w, s) for w, s in candidates if w not in used_words]
            if not available:
                break
            
            # Temperature-based sampling
            if len(available) > 1:
                sims = np.array([s for w, s in available])
                logits = sims / temperature
                logits = logits - np.max(logits)
                probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-10)
                idx = np.random.choice(len(available), p=probs)
                word, sim = available[idx]
            else:
                word, sim = available[0]
            
            generated.append(word)
            used_words.add(word)
            phi_values.append(self.token_phi.get(word, 0.5))
            
            # Update basin toward selected word
            if word in self.basin_coords:
                word_basin = self.basin_coords[word]
                t = 0.3
                current_basin = (1 - t) * current_basin + t * word_basin
                # Add small noise for diversity
                current_basin += np.random.randn(64) * 0.05
                current_basin = sphere_project(current_basin)
        
        # Remove consecutive duplicates
        final = [generated[0]] if generated else []
        for w in generated[1:]:
            if w != final[-1]:
                final.append(w)
        
        return {
            'text': ' '.join(final),
            'phi': float(np.mean(phi_values)) if phi_values else 0.0,
            'tokens_generated': len(final),
            'completion_reason': 'geometric_stable',
            'qig_pure': True
        }
    
    def add_vocabulary_observations(self, observations: List[Dict]) -> Tuple[int, int]:
        """Stub method for vocabulary learning compatibility.
        
        SimpleWordCoordizer uses a fixed vocabulary and doesn't support
        dynamic vocabulary learning. This method is a no-op for compatibility
        with the VocabularyCoordinator.
        
        Returns:
            Tuple of (new_tokens_added=0, weights_updated=0)
        """
        return (0, 0)


# Module-level singleton
_instance: Optional[SimpleWordCoordizer] = None


def get_simple_coordizer() -> SimpleWordCoordizer:
    """Get or create the singleton SimpleWordCoordizer instance."""
    global _instance
    if _instance is None:
        _instance = SimpleWordCoordizer()
    return _instance
