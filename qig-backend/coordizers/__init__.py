"""QIG Coordizers - Unified Geometric Tokenization System.

Provides Fisher-Rao compliant tokenization with 64D basin coordinates.
PostgreSQL-backed with QIG-pure operations.

UNIFIED ACCESS: Use get_coordizer() for the authoritative vocabulary.

Main Classes:
- BaseCoordizer: Abstract interface compatible with Plan→Realize→Repair (WP3.1)
- FisherCoordizer: Base geometric tokenizer implementation
- PostgresCoordizer: QIG-pure Fisher-Rao coordizer (CANONICAL PRODUCTION IMPLEMENTATION)

The BaseCoordizer interface ensures ALL coordizer implementations support:
1. Two-step retrieval (proxy + exact Fisher-Rao)
2. POS filtering capability
3. Geometric operations from canonical module
4. Consistent behavior across all generation paths

Usage:
    from coordizers import get_coordizer, BaseCoordizer
    
    # Get canonical singleton instance
    coordizer = get_coordizer()
    
    # Encode text to basin
    basin = coordizer.encode("hello world")
    
    # Decode with two-step retrieval
    tokens = coordizer.decode_geometric(basin, top_k=10, allowed_pos="NOUN")
    
    # Check POS filtering support
    if coordizer.supports_pos_filtering():
        filtered = coordizer.decode_geometric(basin, top_k=10, allowed_pos="VERB")
"""

import logging
import warnings

logger = logging.getLogger(__name__)

from .base import BaseCoordizer, FisherCoordizer
from .pg_loader import PostgresCoordizer, create_coordizer_from_pg as _create_pg

# Unified coordizer instance
_unified_coordizer = None


def get_coordizer() -> PostgresCoordizer:
    """
    Get the authoritative QIG-pure coordizer.

    This is the SINGLE SOURCE OF TRUTH for vocabulary access.
    Returns PostgresCoordizer with Fisher-Rao distance (QIG-pure).
    
    The returned coordizer implements BaseCoordizer interface with:
    - Two-step geometric decoding (proxy + exact)
    - POS filtering support (if database has pos_tag column)
    - All operations from canonical geometry module

    Returns:
        PostgresCoordizer instance (singleton)
    """
    global _unified_coordizer
    if _unified_coordizer is None:
        _unified_coordizer = _create_pg()
        vocab_size = len(_unified_coordizer.vocab)
        word_count = len(_unified_coordizer.word_tokens)
        logger.info(f"[coordizers] Using PostgresCoordizer: {vocab_size} tokens, {word_count} words (QIG-pure)")
    return _unified_coordizer


def create_coordizer_from_pg(*args, **kwargs) -> PostgresCoordizer:
    """Create PostgresCoordizer - use get_coordizer() for singleton access."""
    return _create_pg(*args, **kwargs)


__all__ = [
    'BaseCoordizer',      # NEW: Abstract interface (WP3.1)
    'FisherCoordizer',
    'PostgresCoordizer',
    'get_coordizer',
    'create_coordizer_from_pg',
]

__version__ = '5.1.0'  # WP3.1: Added BaseCoordizer interface
