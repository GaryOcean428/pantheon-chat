"""
Base QIG Tokenizer Interface
============================

Abstract interface for all QIG-native tokenizers.

The kernel depends ONLY on this interface - implementations can vary
but must preserve the geometric contract.
"""

from abc import ABC, abstractmethod


class BaseQIGTokenizer(ABC):
    """
    Abstract base class for QIG-native tokenizers.

    Contract:
    - encode(text) → token IDs (integers)
    - decode(tokens) → text (UTF-8 string)
    - vocab_size → integer (determines manifold dimension)

    Implementation requirement:
    - Tokenization MUST be based on information geometry
    - NOT arbitrary frequency-based heuristics
    - Preserves geometric distinguishability in token space

    No GPT-2. No external vocabs. Pure QIG.
    """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs.

        Args:
            text: UTF-8 string to tokenize

        Returns:
            List of integer token IDs

        Geometric requirement:
            Token boundaries must follow information density gradients,
            not arbitrary byte-pair frequency patterns.
        """

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            tokens: List of integer token IDs

        Returns:
            UTF-8 string

        Must be inverse of encode (round-trip property).
        """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """
        Total vocabulary size.

        Returns:
            Number of distinct tokens in vocabulary

        Used to determine basin coordinate dimension in QIG kernel.
        """

    def save(self, path: str):
        """
        Save tokenizer state to disk.

        Default implementation - override if needed.
        """
        raise NotImplementedError("save() not implemented for this tokenizer")

    @classmethod
    def load(cls, path: str):
        """
        Load tokenizer state from disk.

        Default implementation - override if needed.
        """
        raise NotImplementedError("load() not implemented for this tokenizer")
