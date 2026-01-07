"""
QIG Coordizer - Thin Wrapper for PostgresCoordizer

Backward-compatible interface delegating to the canonical PostgresCoordizer.
All geometry operations use 64D QIG-pure PostgresCoordizer.

MIGRATION PATH:
- Old code: from qig_tokenizer import get_tokenizer
- New code: from qig_coordizer import get_coordizer as get_tokenizer
"""

from typing import Dict, List, Tuple

from coordizers import PostgresCoordizer

# Import unified coordizer entrypoint (preferred)
from coordizers import get_coordizer as _get_unified_coordizer

# Try Redis for state persistence
REDIS_AVAILABLE = False
try:
    from redis_cache import CoordizerBuffer
    REDIS_AVAILABLE = True
except ImportError:
    pass

# Singleton instances
_coordizer_instance = None
_learning_coordizer_instance = None

# Coordizer instance ID for Redis persistence
COORDIZER_INSTANCE_ID = "main"


def get_coordizer():
    """Get or create singleton coordizer instance.

    SINGLE SOURCE OF TRUTH:
    Returns the unified coordizer (pretrained preferred). This avoids loading the
    legacy pg_loader vocabulary at startup.
    """
    global _coordizer_instance
    if _coordizer_instance is not None:
        return _coordizer_instance

    coordizer = _get_unified_coordizer()
    vocab_count = len(getattr(coordizer, 'vocab', {}) or {})
    word_count = len(getattr(coordizer, 'word_tokens', []) or [])
    print(f"[QIGCoordizer] ✓ Unified coordizer: {vocab_count} tokens, {word_count} words")
    _coordizer_instance = coordizer
    return _coordizer_instance


def get_learning_coordizer():
    """Get coordizer used to persist vocabulary observations.

    Pretrained coordizers may be read-only; only initialize PostgresCoordizer when needed.
    """
    global _learning_coordizer_instance
    if _learning_coordizer_instance is None:
        from coordizers.pg_loader import create_coordizer_from_pg
        _learning_coordizer_instance = create_coordizer_from_pg()
    return _learning_coordizer_instance


def reset_coordizer() -> None:
    """Reset the coordizer singleton to reload from database."""
    global _coordizer_instance, _learning_coordizer_instance

    old_words = len(getattr(_coordizer_instance, 'word_tokens', [])) if _coordizer_instance else 0

    if _coordizer_instance is not None:
        if hasattr(_coordizer_instance, 'close'):
            try:
                _coordizer_instance.close()
            except:
                pass
        _coordizer_instance = None

    if _learning_coordizer_instance is not None:
        if hasattr(_learning_coordizer_instance, 'close'):
            try:
                _learning_coordizer_instance.close()
            except:
                pass
        _learning_coordizer_instance = None

    print(f"[QIGCoordizer] Reset coordizer: was {old_words} words")

    # Force immediate reload
    new_instance = get_coordizer()
    new_words = len(getattr(new_instance, 'word_tokens', []))
    print(f"[QIGCoordizer] Reloaded with {new_words} words")


def update_tokenizer_from_observations(observations: List[Dict]) -> Tuple[int, bool]:
    """Update coordizer with vocabulary observations."""
    coordizer = get_coordizer()
    if not hasattr(coordizer, 'add_vocabulary_observations'):
        coordizer = get_learning_coordizer()

    new_tokens, weights_updated = coordizer.add_vocabulary_observations(observations)

    if new_tokens > 0 or weights_updated:
        _save_coordizer_state(coordizer)

    return new_tokens, weights_updated


def _save_coordizer_state(coordizer) -> None:
    """Save coordizer state to Redis."""
    if not REDIS_AVAILABLE:
        return

    try:
        basin_coords_serializable = {
            token: coords.tolist() if hasattr(coords, 'tolist') else list(coords)
            for token, coords in coordizer.basin_coords.items()
        }

        CoordizerBuffer.save_state(
            COORDIZER_INSTANCE_ID,
            coordizer.vocab,
            coordizer.id_to_token,
            coordizer.token_frequencies,
            coordizer.token_phi,
            basin_coords_serializable,
            {}
        )
        print(f"[QIGCoordizer] Saved state to Redis ({len(coordizer.vocab)} tokens)")
    except Exception as e:
        print(f"[QIGCoordizer] Redis save failed: {e}")


def get_coordizer_stats() -> dict:
    """Get detailed statistics about the current coordizer."""
    coordizer = get_coordizer()
    return coordizer.get_stats()


# QIGCoordizer class for backward compatibility
class QIGCoordizer(PostgresCoordizer):
    """Backward-compatible alias for PostgresCoordizer."""

    def __init__(self, **kwargs):
        # Delegate to PostgresCoordizer
        super().__init__(**kwargs)

    def set_mode(self, mode: str) -> None:
        """Mode switching (legacy compatibility - no-op)."""
        pass


# Backward compatibility aliases
get_tokenizer = get_coordizer
