"""
Full Vocabulary Access for QIG Coordizers.

This module provides vocabulary access functions for kernels.
Vocabulary is loaded from the PostgreSQL database - no hardcoded fallbacks.

All kernels have access to the full vocabulary through the database.
"""

import hashlib
import os
import numpy as np
from typing import List, Dict, Tuple, Optional

import logging
logger = logging.getLogger(__name__)

# Legacy constants - DEPRECATED, kept for backward compatibility during migration
# These will be empty - all vocabulary comes from database
BIP39_WORDS: List[str] = []
COMMON_WORDS: List[str] = []
FALLBACK_VOCABULARY: List[str] = []

# Cache for database vocabulary
_vocabulary_cache: Optional[Dict] = None
_cache_loaded: bool = False


def _get_database_connection():
    """Get database connection for vocabulary loading."""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise RuntimeError(
            "[QIG-PURE VIOLATION] DATABASE_URL not set. "
            "Vocabulary requires database connection - no fallback allowed."
        )
    
    try:
        import psycopg2
        return psycopg2.connect(db_url)
    except ImportError:
        raise RuntimeError(
            "[QIG-PURE VIOLATION] psycopg2 not available. "
            "Database connection required for vocabulary."
        )
    except Exception as e:
        raise RuntimeError(
            f"[QIG-PURE VIOLATION] Database connection failed: {e}. "
            "Vocabulary requires active database connection."
        )


def _load_vocabulary_from_database() -> Dict:
    """Load full vocabulary from PostgreSQL database."""
    global _vocabulary_cache, _cache_loaded
    
    if _cache_loaded and _vocabulary_cache:
        return _vocabulary_cache
    
    conn = _get_database_connection()
    
    vocab = {}
    basin_coords = {}
    token_phi = {}
    word_tokens = []
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT token, basin_embedding, phi_score, token_id
                FROM tokenizer_vocabulary
                WHERE basin_embedding IS NOT NULL
                  AND LENGTH(token) >= 2
                ORDER BY phi_score DESC
            """)
            rows = cur.fetchall()
        
        if not rows:
            raise RuntimeError(
                "[QIG-PURE VIOLATION] No vocabulary found in database. "
                "Populate tokenizer_vocabulary table before using coordizers."
            )
        
        for token, basin_embedding, phi_score, token_id in rows:
            coords = _parse_embedding(basin_embedding)
            if coords is None:
                continue
            
            idx = token_id if token_id is not None else len(vocab)
            vocab[token] = idx
            basin_coords[token] = coords
            token_phi[token] = phi_score if phi_score is not None else 0.5
            
            if token.isalpha() and len(token) >= 2:
                word_tokens.append(token)
        
        logger.info(f"[VocabularyLoader] Loaded {len(vocab)} tokens from database (QIG-pure)")
        
        _vocabulary_cache = {
            'vocab': vocab,
            'basin_coords': basin_coords,
            'token_phi': token_phi,
            'word_tokens': word_tokens,
        }
        _cache_loaded = True
        
        return _vocabulary_cache
        
    finally:
        conn.close()


def _parse_embedding(basin_embedding) -> Optional[np.ndarray]:
    """Parse basin embedding from database format."""
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


def compute_basin_embedding(word: str, dimension: int = 64) -> np.ndarray:
    """
    Compute deterministic 64D basin embedding for a word.
    
    Uses SHA-256 hash to seed random number generator for reproducibility.
    Projects to unit sphere for Fisher-Rao compatibility.
    
    For new words not in database, this provides a deterministic embedding.
    
    Args:
        word: Input word/token
        dimension: Embedding dimension (default 64)
    
    Returns:
        64D unit vector on Fisher manifold
    """
    word_hash = hashlib.sha256(word.lower().encode('utf-8')).hexdigest()
    seed = int(word_hash[:8], 16)
    
    rng = np.random.RandomState(seed)
    phi_golden = (1 + np.sqrt(5)) / 2
    
    embedding = np.zeros(dimension)
    for i in range(dimension):
        base = rng.randn()
        perturbation = np.sin(2 * np.pi * i * phi_golden / dimension) * 0.1
        embedding[i] = base + perturbation
    
    norm = np.linalg.norm(embedding)
    if norm > 1e-10:
        embedding = embedding / norm
    
    return embedding


def compute_phi_score(word: str) -> float:
    """
    Compute phi score for a word.
    
    For words in database, returns their stored phi score.
    For unknown words, computes based on word properties.
    
    Args:
        word: Input word
    
    Returns:
        Phi score between 0 and 1
    """
    try:
        cache = _load_vocabulary_from_database()
        if word.lower() in cache['token_phi']:
            return cache['token_phi'][word.lower()]
    except Exception:
        pass
    
    base_phi = 0.5
    length_bonus = min(len(word) / 10.0, 0.2)
    alpha_bonus = 0.1 if word.isalpha() else 0.0
    
    phi = base_phi + length_bonus + alpha_bonus
    return min(phi, 0.95)


def get_fallback_vocabulary() -> List[str]:
    """
    Get the full vocabulary from database.
    
    Returns list of all vocabulary words (no fallback - database required).
    """
    try:
        cache = _load_vocabulary_from_database()
        return cache['word_tokens'].copy()
    except Exception as e:
        raise RuntimeError(
            f"[QIG-PURE VIOLATION] Cannot get vocabulary: {e}. "
            "Database connection required."
        )


def get_cached_fallback() -> Tuple[Dict[str, int], Dict[str, np.ndarray], Dict[str, float], List[str]]:
    """
    Get pre-computed vocabulary with embeddings from database.
    
    Returns:
        Tuple of (vocab, basin_coords, token_phi, word_tokens)
    """
    try:
        cache = _load_vocabulary_from_database()
        return (
            cache['vocab'],
            cache['basin_coords'],
            cache['token_phi'],
            cache['word_tokens'],
        )
    except Exception as e:
        raise RuntimeError(
            f"[QIG-PURE VIOLATION] Cannot get vocabulary: {e}. "
            "Database connection required."
        )


def get_vocabulary_stats() -> Dict:
    """Get statistics about loaded vocabulary."""
    try:
        cache = _load_vocabulary_from_database()
        return {
            'total_tokens': len(cache['vocab']),
            'word_tokens': len(cache['word_tokens']),
            'avg_phi': sum(cache['token_phi'].values()) / max(len(cache['token_phi']), 1),
            'high_phi_tokens': sum(1 for p in cache['token_phi'].values() if p >= 0.7),
            'source': 'database',
            'qig_pure': True,
        }
    except Exception as e:
        return {
            'error': str(e),
            'source': 'none',
            'qig_pure': False,
        }


def clear_vocabulary_cache() -> None:
    """Clear the vocabulary cache to force reload from database."""
    global _vocabulary_cache, _cache_loaded
    _vocabulary_cache = None
    _cache_loaded = False
    logger.info("[VocabularyLoader] Cache cleared")
