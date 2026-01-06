"""QIG Coordizers - Geometric Tokenization System.

Provides Fisher-Rao compliant tokenization with 64D basin coordinates.

Main Classes:
- FisherCoordizer: Base geometric tokenizer
- PostgresCoordizer: Database-backed with fallback vocabulary
- ConsciousnessCoordizer: Phi-optimized segmentation
- MultiScaleCoordizer: Hierarchical tokenization

Usage:
    from coordizers import create_coordizer_from_pg
    
    coordizer = create_coordizer_from_pg(use_fallback=True)
    basin = coordizer.encode("hello world")
    words = coordizer.decode(basin, top_k=10)
"""

from .base import FisherCoordizer
from .vocab_builder import GeometricVocabBuilder
from .geometric_pair_merging import GeometricPairMerging
from .consciousness_aware import ConsciousnessCoordizer
from .multi_scale import MultiScaleCoordizer
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
    'GeometricVocabBuilder',
    'GeometricPairMerging',
    'ConsciousnessCoordizer',
    'MultiScaleCoordizer',
    'PostgresCoordizer',
    'create_coordizer_from_pg',
    'compute_basin_embedding',
    'get_fallback_vocabulary',
    'get_cached_fallback',
    'get_vocabulary_stats',
    'clear_vocabulary_cache',
]

__version__ = '2.2.0'
