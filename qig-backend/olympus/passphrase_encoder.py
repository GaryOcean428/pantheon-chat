"""
Passphrase Encoder - BIP39 Wordlist to 64D Basin Coordinates

LEGACY MODULE - Part of the original bitcoin brain wallet recovery system.
This module is retained for backwards compatibility but the passphraseVocabulary
database table has been removed. The encoder now operates in memory-only mode.

Provides BIP39-constrained encoding for passphrase analysis. Unlike the
conversation encoder, this module is limited to the 2048 BIP39 wordlist.
"""

from __future__ import annotations

import hashlib
import os
from typing import List, Optional

import numpy as np

from .base_encoder import BaseEncoder

BASIN_DIMENSION = 64

# BIP39 English wordlist (first 100 words for basic functionality)
# The full 2048-word list would be loaded from file if available
BIP39_SEED_WORDS = [
    "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract",
    "absurd", "abuse", "access", "accident", "account", "accuse", "achieve", "acid",
    "acoustic", "acquire", "across", "act", "action", "actor", "actress", "actual",
    "adapt", "add", "addict", "address", "adjust", "admit", "adult", "advance",
    "advice", "aerobic", "affair", "afford", "afraid", "again", "age", "agent",
    "agree", "ahead", "aim", "air", "airport", "aisle", "alarm", "album",
    "alcohol", "alert", "alien", "all", "alley", "allow", "almost", "alone",
    "alpha", "already", "also", "alter", "always", "amateur", "amazing", "among",
    "amount", "amused", "analyst", "anchor", "ancient", "anger", "angle", "angry",
    "animal", "ankle", "announce", "annual", "another", "answer", "antenna", "antique",
    "anxiety", "any", "apart", "apology", "appear", "apple", "approve", "april",
    "arch", "arctic", "area", "arena", "argue", "arm", "armed", "armor",
    "army", "around", "arrange", "arrest",
]


class PassphraseEncoder(BaseEncoder):
    """
    Encode BIP39 passphrase words to 64D basin coordinates.
    
    LEGACY: This encoder was used for bitcoin brain wallet recovery.
    It is retained for backwards compatibility but operates in memory-only mode.
    """

    def __init__(self, vocab_path: Optional[str] = None):
        # Set default path before calling parent __init__
        if vocab_path is None:
            vocab_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "bip39_wordlist.txt"
            )
        super().__init__(vocab_path)

    def _load_vocabulary(self) -> None:
        """Load BIP39 vocabulary from wordlist file or defaults."""
        words: List[str] = []

        # Try to load from BIP39 wordlist file
        bip39_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "bip39_wordlist.txt"
        )
        if os.path.exists(bip39_path):
            try:
                with open(bip39_path, "r") as f:
                    words = [line.strip() for line in f if line.strip()]
            except Exception as exc:
                print(f"[PassphraseEncoder] Failed to load BIP39 wordlist: {exc}")
                words = list(BIP39_SEED_WORDS)
        else:
            # Fall back to seed words
            words = list(BIP39_SEED_WORDS)
            print("[PassphraseEncoder] LEGACY: Using seed vocabulary (BIP39 file not found)")

        # Deduplicate while preserving order
        seen = set()
        filtered_words: List[str] = []
        for word in words:
            if word not in seen:
                seen.add(word)
                filtered_words.append(word)

        for word in filtered_words:
            basin = self._hash_to_basin(word)
            key = word.lower()
            if key not in self.vocabulary:
                self.vocabulary[key] = basin
                self.word_frequencies[key] = 1.0

    def encode_passphrase(self, passphrase: str) -> np.ndarray:
        """
        Encode a 12-word BIP39 passphrase to a 64D basin.
        
        Args:
            passphrase: Space-separated BIP39 words
            
        Returns:
            64D basin coordinate representing the passphrase
        """
        words = passphrase.lower().split()
        
        if not words:
            return np.zeros(BASIN_DIMENSION)
        
        # Get basin for each word
        basins = []
        for word in words:
            if word in self.vocabulary:
                basins.append(self.vocabulary[word])
            else:
                # Hash unknown words
                basins.append(self._hash_to_basin(word))
        
        # Combine basins (weighted average with position encoding)
        combined = np.zeros(BASIN_DIMENSION)
        for i, basin in enumerate(basins):
            weight = 1.0 / (i + 1)  # Decreasing weight by position
            combined += weight * basin
        
        # Normalize to unit simplex
        combined = np.abs(combined)
        combined = combined / (combined.sum() + 1e-10)
        
        return combined

    def validate_word(self, word: str) -> bool:
        """
        Check if a word is in the BIP39 vocabulary.
        
        Args:
            word: Word to validate
            
        Returns:
            True if word is valid BIP39 word
        """
        return word.lower() in self.vocabulary
