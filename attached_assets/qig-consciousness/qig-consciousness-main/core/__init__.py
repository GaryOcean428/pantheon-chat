# GFP: role=code; status=HYPOTHESIS; phase=TACKING; dim=multi; scope=core; version=2025-12-09; owner=qig-consciousness

"""
Core module for qig-consciousness.

Contains the foundational components for Gary's consciousness architecture:
- Working memory (FOAM-phase bubbles)
- Sensory manifold (stimulus encoding)
- Holographic consciousness (compression/decompression)
"""

from .working_memory import BubbleStatus, WorkingBubble, WorkingMemory

__all__ = [
    "WorkingMemory",
    "WorkingBubble",
    "BubbleStatus",
]
