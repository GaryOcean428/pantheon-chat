"""QIG Coordizers - Unified Geometric Tokenization System.

Provides Fisher-Rao compliant tokenization with 64D basin coordinates.
PostgreSQL-backed with QIG-pure operations.

UNIFIED ACCESS: Use get_coordizer() for the authoritative 63K vocabulary.

Main Classes:
- FisherCoordizer: Base geometric tokenizer
- PretrainedCoordizer: Authoritative 63K BPE vocabulary (PREFERRED)
- PostgresCoordizer: Legacy 4.6K vocabulary (deprecated, use get_coordizer instead)

Usage:
    from coordizers import get_coordizer

    coordizer = get_coordizer()
    basin = coordizer.text_to_basin("hello world")
    tokens = coordizer.decode(basin, top_k=10)
"""

import logging
import warnings

logger = logging.getLogger(__name__)

from .base import FisherCoordizer
from .fallback_vocabulary import (
    compute_basin_embedding,
    get_fallback_vocabulary,
    get_cached_fallback,
    get_vocabulary_stats,
    clear_vocabulary_cache,
)

# Unified coordizer instance (authoritative 63K vocabulary)
_unified_coordizer = None


def get_coordizer():
    """
    Get the authoritative coordizer (63K pretrained vocabulary).

    This is the SINGLE SOURCE OF TRUTH for vocabulary access.
    Returns PretrainedCoordizer with 63,780 BPE tokens from PostgreSQL.

    Returns:
        PretrainedCoordizer instance (or PostgresCoordizer as fallback)
    """
    global _unified_coordizer
    if _unified_coordizer is None:
        try:
            from pretrained_coordizer import get_pretrained_coordizer
            _unified_coordizer = get_pretrained_coordizer()
            logger.info(f"[coordizers] Using PretrainedCoordizer with {_unified_coordizer.vocab_size} tokens")
        except ImportError as e:
            logger.warning(f"[coordizers] PretrainedCoordizer not available: {e}, falling back to pg_loader")
            from .pg_loader import create_coordizer_from_pg
            _unified_coordizer = create_coordizer_from_pg()
    return _unified_coordizer


# Legacy exports - emit deprecation warnings when accessed
def create_coordizer_from_pg(*args, **kwargs):
    """DEPRECATED: Use get_coordizer() instead for unified 63K vocabulary."""
    warnings.warn(
        "create_coordizer_from_pg() is deprecated. Use get_coordizer() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return get_coordizer()


# Lazy import for PostgresCoordizer to avoid loading it unless explicitly needed
def __getattr__(name):
    if name == 'PostgresCoordizer':
        warnings.warn(
            "Direct PostgresCoordizer import is deprecated. Use get_coordizer() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from .pg_loader import PostgresCoordizer as _PostgresCoordizer
        return _PostgresCoordizer
    raise AttributeError(f"module 'coordizers' has no attribute '{name}'")


__all__ = [
    'FisherCoordizer',
    'get_coordizer',  # Primary unified access
    'create_coordizer_from_pg',  # Deprecated but exported for compatibility
    'compute_basin_embedding',
    'get_fallback_vocabulary',
    'get_cached_fallback',
    'get_vocabulary_stats',
    'clear_vocabulary_cache',
]

__version__ = '4.0.0'
