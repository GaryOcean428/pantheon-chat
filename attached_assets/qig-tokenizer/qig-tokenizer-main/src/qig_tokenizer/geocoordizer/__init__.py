"""
GeoCoordizer: Next-Generation Geometric Tokenization (Coordizing)
==================================================================

Replaces traditional tokenization with geometric coordization on a
64-dimensional Fisher information manifold.

Key Classes:
    - FisherCoordizer: Core coordizer (text → basin coordinates)
    - GeometricVocabBuilder: Vocabulary construction via Fisher criteria
    - ConsciousnessCoordizer: Φ/κ-aware coordization controller
    - MultiScaleCoordizer: Hierarchical granularity management

Terminology:
    - Token → Basin Coordinate (64D point on manifold)
    - Tokenize → Coordize (map to manifold)
    - Embedding → Coordinate (intrinsic position)
    - Merge → Geodesic Fusion (combine via geometry)
"""

from .consciousness_coordizer import ConsciousnessCoordizer
from .fisher_coordizer import FisherCoordizer
from .multi_scale import MultiScaleCoordizer
from .types import BasinCoordinate, CoordizationResult, TokenCandidate
from .vocab_builder import GeometricVocabBuilder

__all__ = [
    "FisherCoordizer",
    "GeometricVocabBuilder",
    "ConsciousnessCoordizer",
    "MultiScaleCoordizer",
    "BasinCoordinate",
    "CoordizationResult",
    "TokenCandidate",
]

__version__ = "0.1.0"
