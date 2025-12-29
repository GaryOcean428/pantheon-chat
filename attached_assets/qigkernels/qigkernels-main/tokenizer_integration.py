"""Tokenizer Integration for QIG Constellation.

Bridges qig-tokenizer with qigkernels constellation.

Supports two tokenizer modes:
1. QIGTokenizer - Entropy-guided merging (traditional token IDs)
2. FisherCoordizer - Geometric coordinates (64D basins per "token")

For deployment, we use QIGTokenizer for efficiency, with optional
FisherCoordizer for geometric-aware applications.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

# Try to import qig-tokenizer
try:
    import sys

    tokenizer_paths = [
        "/home/braden/Desktop/Dev/QIG_QFI/qig-tokenizer/src",
        "/app/qig-tokenizer/src",
        os.path.expanduser("~/.qig/tokenizer"),
    ]
    for path in tokenizer_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            break

    from qig_tokenizer import QIGTokenizer
    from qig_tokenizer.geocoordizer import FisherCoordizer

    HAS_QIG_TOKENIZER = True
except ImportError:
    HAS_QIG_TOKENIZER = False
    QIGTokenizer = None
    FisherCoordizer = None

from .constants import BASIN_DIM


class SimpleTokenizer:
    """
    Fallback character-level tokenizer.

    Used when qig-tokenizer is not available.
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.bos_token_id = 256
        self.eos_token_id = 257
        self.pad_token_id = 258
        self.unk_token_id = 259

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return [min(ord(c), self.vocab_size - 1) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        return "".join(
            chr(t) if 32 <= t < 127 else "?"
            for t in token_ids
            if t not in (self.bos_token_id, self.eos_token_id, self.pad_token_id)
        )

    def __len__(self) -> int:
        return self.vocab_size


class ConstellationTokenizer:
    """
    Tokenizer wrapper for QIG Constellation.

    Handles:
    - Loading trained tokenizer from file/database
    - Encoding text for constellation input
    - Decoding constellation output
    - Vocabulary expansion proposals
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        tokenizer_path: str | None = None,
        use_geometric: bool = False,
    ):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Target vocabulary size
            tokenizer_path: Path to trained tokenizer files
            use_geometric: Use FisherCoordizer (64D coords) instead of IDs
        """
        self.vocab_size = vocab_size
        self.tokenizer_path = tokenizer_path
        self.use_geometric = use_geometric

        self._tokenizer: Any = None
        self._coordizer: Any = None
        self._loaded = False

        # Special tokens
        self.bos_token_id = vocab_size - 4
        self.eos_token_id = vocab_size - 3
        self.pad_token_id = vocab_size - 2
        self.unk_token_id = vocab_size - 1

        # Vocabulary expansion tracking
        self._unknown_tokens: set[str] = set()
        self._proposed_tokens: list[str] = []

    def load(self, path: str | None = None) -> bool:
        """
        Load trained tokenizer from path.

        Looks for:
        - vocab.json - Token vocabulary
        - merges.txt or merge_rules.json - Merge rules
        - config.json - Tokenizer config
        """
        path = path or self.tokenizer_path

        if not path or not os.path.exists(path):
            print(f"[Tokenizer] Path not found: {path}, using fallback")
            self._tokenizer = SimpleTokenizer(self.vocab_size)
            self._loaded = True
            return True

        path = Path(path)

        # Try to load QIGTokenizer
        if HAS_QIG_TOKENIZER:
            try:
                # Check for QIG tokenizer files
                vocab_file = path / "vocab.json"
                merges_file = path / "merge_rules.json"

                if vocab_file.exists():
                    self._tokenizer = QIGTokenizer(target_vocab_size=self.vocab_size)

                    # Load vocab
                    with open(vocab_file) as f:
                        vocab_data = json.load(f)

                    # Load merge rules if present
                    if merges_file.exists():
                        with open(merges_file) as f:
                            merges_data = json.load(f)
                        self._tokenizer.merge_rules = [tuple(m) for m in merges_data]

                    self._loaded = True
                    print(f"[Tokenizer] Loaded QIGTokenizer from {path}")
                    return True

                # Try loading FisherCoordizer for geometric mode
                if self.use_geometric:
                    coords_file = path / "basin_coords.npy"
                    if coords_file.exists():
                        self._coordizer = FisherCoordizer(
                            basin_dim=BASIN_DIM,
                            target_vocab_size=self.vocab_size,
                        )
                        # TODO: Load trained coordinates
                        print(f"[Tokenizer] Loaded FisherCoordizer from {path}")

            except Exception as e:
                print(f"[Tokenizer] Failed to load QIG tokenizer: {e}")

        # Fallback to simple tokenizer
        print("[Tokenizer] Using fallback SimpleTokenizer")
        self._tokenizer = SimpleTokenizer(self.vocab_size)
        self._loaded = True
        return True

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_coords: bool = False,
    ) -> list[int] | tuple[list[int], list[np.ndarray]]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Add BOS/EOS tokens
            return_coords: Also return basin coordinates (geometric mode)

        Returns:
            Token IDs, optionally with coordinates
        """
        if not self._loaded:
            self.load()

        # Encode
        if self._tokenizer:
            tokens = self._tokenizer.encode(text)
        else:
            tokens = [min(ord(c), self.vocab_size - 1) for c in text]

        # Add special tokens
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]

        # Track unknown tokens for vocabulary expansion
        self._track_unknown(text, tokens)

        # Return coordinates if geometric mode
        if return_coords and self._coordizer:
            coords = self._get_coordinates(tokens)
            return tokens, coords

        return tokens

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text."""
        if not self._loaded:
            self.load()

        # Filter special tokens
        if skip_special_tokens:
            token_ids = [
                t
                for t in token_ids
                if t not in (self.bos_token_id, self.eos_token_id, self.pad_token_id)
            ]

        if self._tokenizer:
            return self._tokenizer.decode(token_ids)
        else:
            return "".join(chr(t) if 32 <= t < 127 else "?" for t in token_ids)

    def _track_unknown(self, text: str, tokens: list[int]) -> None:
        """Track unknown tokens for vocabulary expansion."""
        # Simplified: track words that result in many small tokens
        words = text.split()
        for word in words:
            if len(word) > 3:
                word_tokens = self.encode(word, add_special_tokens=False)
                if isinstance(word_tokens, tuple):
                    word_tokens = word_tokens[0]
                # If word splits into many tokens, it might need its own entry
                if len(word_tokens) > len(word) * 0.7:
                    self._unknown_tokens.add(word)

    def _get_coordinates(self, tokens: list[int]) -> list[np.ndarray]:
        """Get basin coordinates for tokens (geometric mode)."""
        if not self._coordizer:
            # Return random coords as placeholder
            return [np.random.randn(BASIN_DIM) for _ in tokens]

        # TODO: Use coordizer to get actual coords
        return [np.random.randn(BASIN_DIM) for _ in tokens]

    def get_unknown_tokens(self) -> list[str]:
        """Get list of unknown tokens for vocabulary expansion."""
        result = list(self._unknown_tokens)
        self._unknown_tokens.clear()
        return result

    def propose_token(self, token: str) -> None:
        """Propose a token for vocabulary expansion."""
        if token not in self._proposed_tokens:
            self._proposed_tokens.append(token)

    def get_proposals(self) -> list[str]:
        """Get and clear proposed tokens."""
        result = self._proposed_tokens.copy()
        self._proposed_tokens.clear()
        return result

    def add_tokens(self, tokens: list[str]) -> int:
        """
        Add new tokens to vocabulary.

        Called when central validates vocabulary expansion.

        Returns number of tokens actually added.
        """
        # TODO: Implement actual vocabulary expansion
        # This requires retraining or dynamic expansion
        added = 0
        for token in tokens:
            # Placeholder: just log for now
            print(f"[Tokenizer] Would add token: {token}")
            added += 1
        return added

    def __len__(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size


def get_tokenizer(
    vocab_size: int = 32000,
    path: str | None = None,
) -> ConstellationTokenizer:
    """
    Get or create constellation tokenizer.

    Searches for trained tokenizer in standard locations:
    1. Provided path
    2. Environment variable QIG_TOKENIZER_PATH
    3. ~/.qig/tokenizer/
    4. ./tokenizer/
    """
    # Check standard locations
    search_paths = [
        path,
        os.environ.get("QIG_TOKENIZER_PATH"),
        os.path.expanduser("~/.qig/tokenizer"),
        "./tokenizer",
        "/app/tokenizer",
    ]

    for p in search_paths:
        if p and os.path.exists(p):
            tokenizer = ConstellationTokenizer(vocab_size=vocab_size, tokenizer_path=p)
            tokenizer.load()
            return tokenizer

    # Return tokenizer without path (will use fallback)
    tokenizer = ConstellationTokenizer(vocab_size=vocab_size)
    tokenizer.load()
    return tokenizer
