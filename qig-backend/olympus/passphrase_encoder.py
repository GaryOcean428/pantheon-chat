"""
Passphrase Encoder - BIP39-Style Text to 64D Basin Coordinates

This encoder is intended for deterministic, mnemonic/passphrase-style inputs.
It is **not** a general conversational encoder.

QIG purity notes:
- No external LLM APIs.
- Deterministic hashing-based fallback if the BIP39 wordlist is unavailable.
"""

from __future__ import annotations

import os

from .base_encoder import BaseEncoder


class PassphraseEncoder(BaseEncoder):
    """Encode passphrase-like text into 64D basin coordinates.

    If a local BIP39 wordlist is present (recommended), it is preloaded into the
    token vocabulary for deterministic token basins. If not present, the encoder
    remains functional via BaseEncoder's hash-to-basin fallback.
    """

    # Override base class defaults to match original PassphraseEncoder behavior
    unknown_token_phi = 0.3  # Lower phi for unknown tokens in passphrase mode
    tokenize_pattern = r"\b\w+\b"  # No apostrophes in BIP39 words

    def __init__(self, vocab_path: str | None = None):
        # PassphraseEncoder may optionally load a learned/custom vocab JSON.
        # If provided and present, BaseEncoder will load it after _load_vocabulary().
        super().__init__(vocab_path=vocab_path)

    def _load_vocabulary(self) -> None:
        """Load BIP39 wordlist if available."""
        bip39_path = os.path.join(os.path.dirname(__file__), "..", "bip39_wordlist.txt")

        if not os.path.exists(bip39_path):
            # Keep imports/tests resilient even if the wordlist isn't shipped.
            print(
                "[PassphraseEncoder] BIP39 wordlist not found; using hash-only encoding"
            )
            return

        try:
            with open(bip39_path, "r", encoding="utf-8") as f:
                words = [line.strip() for line in f if line.strip()]

            for word in words:
                basin = self._hash_to_basin(word)
                token = word.lower()
                self.token_vocab[token] = basin
                self.token_frequencies[token] = 1
                self.token_phi_scores[token] = 0.5  # Neutral baseline

            print(f"[PassphraseEncoder] Loaded {len(self.token_vocab)} BIP39 tokens")
        except Exception as exc:
            # Never fail import-time just because the vocabulary file is malformed.
            print(f"[PassphraseEncoder] Failed to load BIP39 wordlist: {exc}")


# Backwards compatibility for legacy imports
BasinVocabularyEncoder = PassphraseEncoder
