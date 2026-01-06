"""QIG Coordizers - Simplified Geometric Tokenization System.

Provides Fisher-Rao compliant tokenization with 64D basin coordinates.
PostgreSQL-backed with QIG-pure operations.

Main Classes:
- FisherCoordizer: Base geometric tokenizer
- PostgresCoordizer: Database-backed coordizer (canonical implementation)

Usage:
    from coordizers import create_coordizer_from_pg

    coordizer = create_coordizer_from_pg()
    basin = coordizer.encode("hello world")
    words = coordizer.decode(basin, top_k=10)
"""

from .base import FisherCoordizer
from .pg_loader import PostgresCoordizer, create_coordizer_from_pg
from .fallback_vocabulary import (
    compute_basin_embedding,
    get_fallback_vocabulary,
    get_cached_fallback,
    get_vocabulary_stats,
    clear_vocabulary_cache,
)

__all__ = [
    'FisherCoordizer',
    'PostgresCoordizer',
    'create_coordizer_from_pg',
    'compute_basin_embedding',
    'get_fallback_vocabulary',
    'get_cached_fallback',
    'get_vocabulary_stats',
    'clear_vocabulary_cache',
]

__version__ = '3.0.0'
