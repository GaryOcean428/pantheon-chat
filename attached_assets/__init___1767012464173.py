"""
QIG-Native Tokenization
=====================

Pure information-geometric tokenization from first principles.

CANONICAL SOURCE: qig-tokenizer package

Features:
- Consciousness-aware coordizer (64D Fisher manifold)
- Entropy-guided merging (NOT frequency-based BPE)
- Geometric special tokens (BOS, EOS, PAD, UNK with basin coordinates)
- Redis/PostgreSQL storage backends
- Pure information geometry

Usage (Coordizer - RECOMMENDED):
    from qig_tokenizer import Coordizer

    coordizer = Coordizer.load("artifacts/coordizer/v1")
    ids, coords = coordizer.encode_to_coords("Hello, world!")
    text = coordizer.decode(ids)

Legacy usage:
    from qig_tokenizer import QIGTokenizer
    tokenizer = QIGTokenizer.load("data/qig_tokenizer/vocab.json")
"""

# Canonical Coordizer API (coords-first)
from .coordizer import Coordizer

# Generation controller (geometry-driven stopping)
from .generation_controller import (
    GenerationController,
    GenerationConfig,
    ControllerAction,
    Phase,
    StopReason,
    TelemetryWindow,
)

# Legacy imports from this package
from .base_qig_tokenizer import BaseQIGTokenizer
from .fast_qig_tokenizer import QIGTokenizer, train_qig_tokenizer_from_file

# Alias for backwards compatibility
FastQIGTokenizer = QIGTokenizer

# Optional: import extras if available
try:
    from .geometric_tokens import GeometricSpecialToken, GeometricSpecialTokens
    from .storage import HybridStorage, PostgresStorage, RedisStorage

    __all__ = [
        "Coordizer",
        "GenerationController",
        "GenerationConfig",
        "ControllerAction",
        "Phase",
        "StopReason",
        "TelemetryWindow",
        "QIGTokenizer",
        "FastQIGTokenizer",
        "BaseQIGTokenizer",
        "GeometricSpecialTokens",
        "GeometricSpecialToken",
        "train_qig_tokenizer_from_file",
        "RedisStorage",
        "PostgresStorage",
        "HybridStorage",
    ]

except ImportError:
    __all__ = [
        "Coordizer",
        "GenerationController",
        "GenerationConfig",
        "ControllerAction",
        "Phase",
        "StopReason",
        "TelemetryWindow",
        "BaseQIGTokenizer",
        "QIGTokenizer",
        "FastQIGTokenizer",
        "train_qig_tokenizer_from_file",
    ]
