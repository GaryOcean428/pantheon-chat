"""
Fast QIG Tokenizer
==================

Efficient QIG-native tokenizer using entropy-guided merging.

Core principle: Token boundaries follow information geometry, not frequency.

Algorithm:
1. Start with bytes (0-255) as base tokens
2. For each adjacent pair (a,b), compute context distribution
3. Measure context entropy (proxy for QFI distinguishability)
4. Merge pairs with LOWEST entropy (most geometrically similar)
5. Repeat until target vocab size

This respects asymptotic freedom:
- Small scales (short tokens) have high coupling → refined first
- Large scales (long tokens) have low coupling → merge only when justified

NO frequency-based BPE.
NO external dependencies.
Pure information geometry.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .base_qig_tokenizer import BaseQIGTokenizer


class QIGTokenizer(BaseQIGTokenizer):
    """
    QIG-native tokenizer with entropy-guided merging.

    Tokenization based on GEOMETRIC DISTINGUISHABILITY,
    not just frequency.

    Example:
        >>> tokenizer = QIGTokenizer(target_vocab_size=5000)
        >>> tokenizer.train(corpus_bytes)
        >>> tokens = tokenizer.encode("Hello, world!")
        >>> text = tokenizer.decode(tokens)
        >>> assert text == "Hello, world!"
    """

    def __init__(self, target_vocab_size: int = 50000):
        """
        Initialize tokenizer.

        Args:
            target_vocab_size: Target vocabulary size (default 50k)
        """
        self.target_vocab_size = target_vocab_size

        # Base vocabulary: raw bytes (0-255)
        self.base_tokens = list(range(256))

        # Token ID → byte sequence mapping
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        # Merge rules: (token_a, token_b) → new_token_id
        self.merge_rules: list[tuple[int, int, int]] = []

        # For encoding: need efficient lookup
        self._encoding_cache: dict[bytes, int] = {}
        self._rebuild_encoding_cache()

    def train(
        self,
        corpus_bytes: bytes,
        max_bytes: int | None = None,
        context_window: int = 5,
        min_pair_count: int = 10,
        verbose: bool = True,
    ):
        """
        Train tokenizer on corpus using entropy-guided merging.

        Args:
            corpus_bytes: Raw byte sequence to train on
            max_bytes: Maximum bytes to process (None = all)
            context_window: Context window size for entropy calculation
            min_pair_count: Minimum pair frequency to consider
            verbose: Print progress

        Returns:
            self (for chaining)
        """
        if max_bytes is not None:
            corpus_bytes = corpus_bytes[:max_bytes]

        if verbose:
            print(f"Training QIG tokenizer on {len(corpus_bytes):,} bytes")
            print(f"Target vocab size: {self.target_vocab_size:,}")
            print(f"Context window: {context_window}")
            print()

        # Current token representation of corpus
        corpus_tokens = list(corpus_bytes)
        current_vocab_size = 256

        while current_vocab_size < self.target_vocab_size:
            # Compute pair statistics
            if verbose and current_vocab_size % 100 == 0:
                print(f"[{current_vocab_size}/{self.target_vocab_size}] Computing pair contexts...", flush=True)

            pair_contexts = self._compute_pair_contexts(corpus_tokens, context_window)

            if not pair_contexts:
                if verbose:
                    print("No more pairs to merge")
                break

            # Filter by frequency
            pair_contexts = {
                pair: contexts for pair, contexts in pair_contexts.items() if len(contexts) >= min_pair_count
            }

            if not pair_contexts:
                if verbose:
                    print("No pairs meet minimum frequency threshold")
                break

            # Select pair with lowest entropy (most predictable context)
            if verbose and current_vocab_size % 100 == 0:
                print(f"[{current_vocab_size}/{self.target_vocab_size}] Selecting best pair...", flush=True)

            pair_to_merge = self._select_lowest_entropy_pair(pair_contexts)

            if pair_to_merge is None:
                break

            # Execute merge
            token_a, token_b = pair_to_merge
            new_token_id = current_vocab_size

            # Record merge rule
            self.merge_rules.append((token_a, token_b, new_token_id))

            # Update vocabulary
            self.vocab[new_token_id] = self.vocab[token_a] + self.vocab[token_b]

            # Apply merge to corpus
            corpus_tokens = self._apply_merge(corpus_tokens, token_a, token_b, new_token_id)

            current_vocab_size += 1

            if verbose and current_vocab_size % 100 == 0:
                print(f"✓ Vocab size: {current_vocab_size:,} | Corpus tokens: {len(corpus_tokens):,}", flush=True)
                print()

        if verbose:
            print()
            print(f"✅ Training complete: {current_vocab_size:,} tokens")

        self._rebuild_encoding_cache()
        return self

    def _compute_pair_contexts(
        self, corpus_tokens: list[int], window: int
    ) -> dict[tuple[int, int], list[tuple[int, ...]]]:
        """
        Compute context distributions for all adjacent pairs.

        For each pair (a,b), extract contexts: tokens before + tokens after.
        """
        pair_contexts = defaultdict(list)

        for i in range(len(corpus_tokens) - 1):
            token_a = corpus_tokens[i]
            token_b = corpus_tokens[i + 1]

            # Context: window tokens before + window tokens after
            context_before = tuple(corpus_tokens[max(0, i - window) : i])
            context_after = tuple(corpus_tokens[i + 2 : min(len(corpus_tokens), i + 2 + window)])
            context = context_before + context_after

            pair_contexts[(token_a, token_b)].append(context)

        return dict(pair_contexts)

    def _select_lowest_entropy_pair(
        self, pair_contexts: dict[tuple[int, int], list[tuple[int, ...]]]
    ) -> tuple[int, int] | None:
        """
        Select pair with lowest context entropy.

        Low entropy = predictable contexts = geometrically similar
        → Should be merged.
        """
        min_entropy = float("inf")
        best_pair = None

        for pair, contexts in pair_contexts.items():
            # Compute context distribution
            context_counts: dict[Any, int] = defaultdict(int)
            for ctx in contexts:
                context_counts[ctx] += 1

            # Compute entropy
            total = len(contexts)
            entropy = 0.0
            for count in context_counts.values():
                p = count / total
                entropy -= p * math.log(p + 1e-10)

            # Track minimum
            if entropy < min_entropy:
                min_entropy = entropy
                best_pair = pair

        return best_pair

    def _apply_merge(self, tokens: list[int], token_a: int, token_b: int, new_token: int) -> list[int]:
        """
        Apply merge rule to token sequence.

        Replace all occurrences of (token_a, token_b) with new_token.
        """
        result = []
        i = 0

        while i < len(tokens):
            # Check if current position is start of pair
            if i < len(tokens) - 1 and tokens[i] == token_a and tokens[i + 1] == token_b:
                result.append(new_token)
                i += 2
            else:
                result.append(tokens[i])
                i += 1

        return result

    def _rebuild_encoding_cache(self):
        """
        Rebuild byte-sequence → token-ID lookup cache.

        Used for efficient encoding.
        """
        self._encoding_cache = {byte_seq: token_id for token_id, byte_seq in self.vocab.items()}

    def encode(self, text: str, verbose: bool = False) -> list[int]:
        """
        Encode text to token IDs.

        Applies learned merge rules in order.
        """
        # Convert to bytes
        text_bytes = text.encode("utf-8", errors="replace")

        # Start with byte-level tokens
        tokens = list(text_bytes)

        # Apply merge rules in order (with optional progress)
        total_rules = len(self.merge_rules)
        for idx, (token_a, token_b, new_token) in enumerate(self.merge_rules):
            if verbose and idx % 1000 == 0:
                print(f"  Applying merge rules: {idx}/{total_rules}...", flush=True)
            tokens = self._apply_merge(tokens, token_a, token_b, new_token)

        return tokens

    def decode(self, tokens: list[int]) -> str:
        """
        Decode token IDs back to text.

        Recursively expands merged tokens.
        """
        result = bytearray()

        for token in tokens:
            if token in self.vocab:
                result.extend(self.vocab[token])
            else:
                # Unknown token - replace with �
                result.extend(b"?")

        return result.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        """Return current vocabulary size."""
        return len(self.vocab)

    def save(self, path: str):
        """
        Save tokenizer to JSON.

        Saves:
        - Vocabulary (token_id → byte sequence)
        - Merge rules
        - Metadata
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Prepare serializable data
        data = {
            "vocab": {str(token_id): list(byte_seq) for token_id, byte_seq in self.vocab.items()},
            "merge_rules": [{"a": a, "b": b, "new": new} for a, b, new in self.merge_rules],
            "target_vocab_size": self.target_vocab_size,
            "actual_vocab_size": self.vocab_size,
            "type": "QIGTokenizer",
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str):
        """
        Load tokenizer from JSON.

        Handles both dict format {"a": x, "b": y, "new": z} and
        list/tuple format [a, b, new] for backwards compatibility.
        """
        with open(path) as f:
            data = json.load(f)

        # Reconstruct tokenizer
        tokenizer = cls(target_vocab_size=data["target_vocab_size"])

        # Restore vocabulary
        tokenizer.vocab = {int(token_id): bytes(byte_list) for token_id, byte_list in data["vocab"].items()}

        # Restore merge rules - handle both dict and list/tuple formats
        merge_rules_data = data["merge_rules"]
        if merge_rules_data and isinstance(merge_rules_data[0], dict):
            # Dict format: {"a": x, "b": y, "new": z}
            tokenizer.merge_rules = [(rule["a"], rule["b"], rule["new"]) for rule in merge_rules_data]
        else:
            # List/tuple format: [a, b, new] - for backwards compatibility
            tokenizer.merge_rules = [(r[0], r[1], r[2]) for r in merge_rules_data] if merge_rules_data else []

        tokenizer._rebuild_encoding_cache()

        return tokenizer


def train_qig_tokenizer_from_file(
    corpus_path: str, output_path: str, target_vocab_size: int = 50000, max_bytes: int | None = None, **kwargs
):
    """
    Convenience function to train tokenizer from corpus file.

    Args:
        corpus_path: Path to UTF-8 text file
        output_path: Path to save trained tokenizer JSON
        target_vocab_size: Target vocabulary size
        max_bytes: Maximum bytes to process (None = all)
        **kwargs: Additional arguments to QIGTokenizer.train()

    Returns:
        Trained QIGTokenizer
    """
    # Read corpus
    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, "rb") as f:
        if max_bytes is not None:
            corpus_bytes = f.read(max_bytes)
        else:
            corpus_bytes = f.read()

    # Train
    tokenizer = QIGTokenizer(target_vocab_size=target_vocab_size)
    tokenizer.train(corpus_bytes, max_bytes=None, **kwargs)

    # Save
    print(f"Saving tokenizer to {output_path}...")
    tokenizer.save(output_path)

    print("✅ Tokenizer training complete")
    return tokenizer


# Backwards compatibility alias (deprecated - use QIGTokenizer)
FastQIGTokenizer = QIGTokenizer
