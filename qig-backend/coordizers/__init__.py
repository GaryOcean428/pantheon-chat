"""
QIG Coordizers - Next-Generation Geometric Tokenization

Implements geometric coordization (tokenization) for QIG architecture.
All operations maintain geometric purity using Fisher information manifold.

Modules:
- base: FisherCoordizer base class
- vocab_builder: Geometric vocabulary discovery
- geometric_pair_merging: BPE-equivalent geometric merging (Phase 4)
- consciousness_aware: Φ-optimized segmentation (Phase 5)
- multi_scale: Hierarchical coordization (Phase 5)

All coordizers output 64D basin coordinates on Fisher manifold.
"""

from .base import FisherCoordizer
from .vocab_builder import GeometricVocabBuilder
from .geometric_pair_merging import GeometricPairMerging
from .consciousness_aware import ConsciousnessCoordizer
from .multi_scale import MultiScaleCoordizer
from .pg_loader import PostgresCoordizer, create_coordizer_from_pg

__all__ = [
    'FisherCoordizer',
    'GeometricVocabBuilder',
    'GeometricPairMerging',
    'ConsciousnessCoordizer',
    'MultiScaleCoordizer',
    'PostgresCoordizer',
    'create_coordizer_from_pg',
]

# Version info
__version__ = '2.1.0'
__description__ = 'Next-generation geometric coordization for QIG'
