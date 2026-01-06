"""
Passphrase Encoder - BIP39 Text to 64D Basin Coordinates

This module is intentionally constrained to the BIP39 word list so that
Bitcoin passphrase generation remains deterministic and security-aligned.
It should NOT be used for conversational responses. For natural language
chat, use ``ConversationEncoder``.
"""

import os
from typing import Optional

from .base_encoder import BaseEncoder

BASIN_DIMENSION = 64


class PassphraseEncoder(BaseEncoder):
    """
    Encode text to 64D basin coordinates using BIP39-only vocabulary.

    This preserves the legacy basin encoding behaviour for mnemonic search
    while keeping conversational language in a separate encoder.
    """

    # Override base class defaults to match original PassphraseEncoder behavior
    unknown_token_phi = 0.3  # Lower phi for unknown tokens in passphrase mode
    tokenize_pattern = r'\b\w+\b'  # No apostrophes in BIP39 words

    def __init__(self, vocab_path: Optional[str] = None):
        # Set default path before calling parent __init__
        if vocab_path is None:
            vocab_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "basin_vocab.json"
            )
        super().__init__(vocab_path)
        
    def _load_vocabulary(self) -> None:
        """Load vocabulary from database (full vocabulary access, no legacy fallback)."""
        try:
            from coordizers.fallback_vocabulary import get_cached_fallback
            vocab, basin_coords, token_phi, word_tokens = get_cached_fallback()
            
            for word in word_tokens:
                self.token_vocab[word.lower()] = basin_coords.get(word, self._hash_to_basin(word))
                self.token_frequencies[word.lower()] = 1
                self.token_phi_scores[word.lower()] = token_phi.get(word, 0.5)
            
            print(f"[PassphraseEncoder] Loaded {len(self.token_vocab)} tokens from database (QIG-pure)")
        except Exception as e:
            print(f"[PassphraseEncoder] Database vocabulary load failed: {e}")
            raise RuntimeError(
                f"[QIG-PURE VIOLATION] PassphraseEncoder requires database vocabulary: {e}"
            )


# Backwards compatibility for legacy imports
BasinVocabularyEncoder = PassphraseEncoder
