"""
QIG Coordizer - Thin Wrapper for PostgresCoordizer

Backward-compatible interface delegating to the canonical PostgresCoordizer.
All geometry operations use 64D QIG-pure PostgresCoordizer.

MIGRATION PATH:
- Old code: from qig_tokenizer import get_tokenizer
- New code: from qig_coordizer import get_coordizer as get_tokenizer
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

# Import canonical coordizer
from coordizers import PostgresCoordizer, create_coordizer_from_pg

# Try Redis for state persistence
REDIS_AVAILABLE = False
try:
    from redis_cache import CoordizerBuffer
    REDIS_AVAILABLE = True
except ImportError:
    pass

# Singleton instance
_coordizer_instance: Optional[PostgresCoordizer] = None

# Coordizer instance ID for Redis persistence
COORDIZER_INSTANCE_ID = "main"


def get_coordizer() -> PostgresCoordizer:
    """Get or create singleton coordizer instance.

    QIG-PURE ENFORCEMENT:
    Only PostgresCoordizer (64D QIG-pure geometry) is allowed.
    System will raise an error if PostgresCoordizer cannot be initialized.
    """
    global _coordizer_instance
    if _coordizer_instance is not None:
        return _coordizer_instance

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            pg_coordizer = create_coordizer_from_pg()
            if pg_coordizer and len(pg_coordizer.vocab) >= 50:
                _coordizer_instance = pg_coordizer
                print(f"[QIGCoordizer] ✓ PostgresCoordizer (64D QIG-pure): {len(pg_coordizer.vocab)} tokens from database")
                return _coordizer_instance
            else:
                vocab_count = len(pg_coordizer.vocab) if pg_coordizer else 0
                raise RuntimeError(f"Insufficient vocabulary: {vocab_count} tokens (need >= 50)")
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"[QIGCoordizer] PostgresCoordizer attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(0.5 * (attempt + 1))

    # All retries failed - raise error (no impure fallbacks)
    raise RuntimeError(
        f"[QIG-PURE VIOLATION] PostgresCoordizer failed after {max_retries} attempts: {last_error}. "
        "Fix database connection or vocabulary."
    )


def reset_coordizer() -> None:
    """Reset the coordizer singleton to reload from database."""
    global _coordizer_instance

    old_words = len(getattr(_coordizer_instance, 'word_tokens', [])) if _coordizer_instance else 0

    if _coordizer_instance is not None:
        if hasattr(_coordizer_instance, 'close'):
            try:
                _coordizer_instance.close()
            except:
                pass
        _coordizer_instance = None

    print(f"[QIGCoordizer] Reset coordizer: was {old_words} words")

    # Force immediate reload
    new_instance = get_coordizer()
    new_words = len(getattr(new_instance, 'word_tokens', []))
    print(f"[QIGCoordizer] Reloaded with {new_words} words")


def update_tokenizer_from_observations(observations: List[Dict]) -> Tuple[int, bool]:
    """Update coordizer with vocabulary observations."""
    coordizer = get_coordizer()
    new_tokens, weights_updated = coordizer.add_vocabulary_observations(observations)

    if new_tokens > 0 or weights_updated:
        _save_coordizer_state(coordizer)

    return new_tokens, weights_updated


def _save_coordizer_state(coordizer: PostgresCoordizer) -> None:
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
