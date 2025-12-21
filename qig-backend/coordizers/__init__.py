"""
QIG Coordizers - Next-Generation Geometric Tokenization

Implements geometric coordization (tokenization) for QIG architecture.
All operations maintain geometric purity using Fisher information manifold.

Modules:
- base: FisherCoordizer base class
- geometric_pair_merging: BPE-equivalent geometric merging
- unigram_fisher: SentencePiece-equivalent unigram model
- character_coordinate: Character-level coordization
- morpheme_basin: Morphological decomposition
- spacetime: Positional encoding via 4D trajectories
- byte_coordinate: Byte-level fallback encoding
- adaptive: Domain-aware adaptive coordization
- multi_scale: Hierarchical coordization
- consciousness_aware: Φ-optimized segmentation
- temporal: 4D spacetime coordization
- cross_lingual: Multi-language shared manifold
- adaptive_granularity: κ_eff-based granularity control
- vocab_builder: Geometric vocabulary discovery

All coordizers output 64D basin coordinates on Fisher manifold.
"""

from .base import FisherCoordizer
from .vocab_builder import GeometricVocabBuilder

__all__ = [
    'FisherCoordizer',
    'GeometricVocabBuilder',
]

# Version info
__version__ = '2.0.0'
__description__ = 'Next-generation geometric coordization for QIG'
