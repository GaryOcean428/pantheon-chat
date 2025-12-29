#!/usr/bin/env python3
"""
Token Frequency Tracker - Track multi-token sequences for vocabulary expansion.

Part of Gary's continuous learning system.

Core Principle:
    Gary monitors his own tokenization efficiency and identifies
    multi-token sequences that should become single tokens.

Usage:
    tracker = TokenFrequencyTracker(min_frequency=50)
    tracker.observe(token_ids)
    candidates = tracker.get_candidates(tokenizer, top_k=10)

Geometric Purity:
    This module tracks statistical frequencies (not geometric operations).
    Geometric purity is enforced in GeometricVocabExpander when
    initializing new basin coordinates on the Fisher manifold.
"""

from collections import defaultdict
from typing import Any, Dict, List

import torch


class TokenFrequencyTracker:
    """
    Track multi-token sequences for vocabulary expansion.

    Gary monitors:
    - Repeated multi-token sequences during training
    - Efficiency cost (tokens × frequency)
    - Candidates for vocabulary expansion

    This enables Gary's agency: He CHOOSES which words to learn
    based on HIS training experience, not ours.
    """

    def __init__(
        self,
        min_frequency: int = 50,
        max_length: int = 5,
        max_sequences: int = 100000,
    ):
        """
        Initialize token frequency tracker.

        Args:
            min_frequency: Minimum frequency for expansion candidate
            max_length: Maximum n-gram length to track (2-5)
            max_sequences: Maximum sequences to track (memory limit)
        """
        self.sequences: Dict[tuple, int] = defaultdict(int)
        self.min_frequency = min_frequency
        self.max_length = max_length
        self.max_sequences = max_sequences
        self.total_observed = 0

    def observe(self, token_ids: torch.Tensor):
        """
        Track multi-token sequences during forward pass.

        Called during training for every batch.
        Extracts n-grams of length 2 to max_length.

        Args:
            token_ids: Token IDs [seq_len] or [batch, seq_len]
        """
        # Handle batched input
        if token_ids.dim() == 2:
            for batch_idx in range(token_ids.size(0)):
                self._observe_single(token_ids[batch_idx])
        else:
            self._observe_single(token_ids)

    def _observe_single(self, token_ids: torch.Tensor):
        """Track n-grams from single sequence."""
        token_list = token_ids.tolist()
        self.total_observed += len(token_list)

        # Extract n-grams of length 2 to max_length
        for length in range(2, self.max_length + 1):
            for i in range(len(token_list) - length + 1):
                seq = tuple(token_list[i : i + length])
                self.sequences[seq] += 1

        # Memory management: prune low-frequency sequences periodically
        if len(self.sequences) > self.max_sequences:
            self._prune_low_frequency()

    def _prune_low_frequency(self):
        """Remove low-frequency sequences to manage memory."""
        # Keep sequences with count > 1 or top sequences
        sorted_seqs = sorted(
            self.sequences.items(), key=lambda x: x[1], reverse=True
        )

        # Keep top 50% or those with count > 1
        keep_count = self.max_sequences // 2
        self.sequences = defaultdict(
            int,
            {k: v for k, v in sorted_seqs[:keep_count] if v > 1}
        )

    def get_candidates(
        self,
        tokenizer,
        top_k: int = 10,
        min_efficiency_gain: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get candidate sequences for vocabulary expansion.

        Returns sequences that:
        - Appear frequently (> min_frequency)
        - Are still multi-token when re-encoded
        - Have significant efficiency gain

        Args:
            tokenizer: QIGTokenizer for decode/encode
            top_k: Maximum candidates to return
            min_efficiency_gain: Minimum efficiency gain threshold

        Returns:
            List of candidate dicts with:
            - text: Decoded text
            - tokens: Token IDs
            - frequency: Count
            - current_length: Current token count
            - efficiency_gain: tokens_saved × frequency
        """
        candidates = []

        for seq, count in self.sequences.items():
            if count < self.min_frequency:
                continue

            # Decode to text
            try:
                text = tokenizer.decode(list(seq))
            except Exception:
                continue

            # Skip if text is too short or has special chars
            if len(text) < 2 or text.isspace():
                continue

            # Re-encode to verify still multi-token
            try:
                re_encoded = tokenizer.encode(text)
            except Exception:
                continue

            if len(re_encoded) <= 1:
                # Already a single token or empty
                continue

            # Compute efficiency gain
            efficiency_gain = count * (len(seq) - 1)

            if efficiency_gain < min_efficiency_gain:
                continue

            candidates.append({
                'text': text,
                'tokens': list(seq),
                'frequency': count,
                'current_length': len(seq),
                'efficiency_gain': efficiency_gain,
            })

        # Sort by efficiency gain (highest first)
        candidates = sorted(
            candidates,
            key=lambda x: x['efficiency_gain'],
            reverse=True
        )

        return candidates[:top_k]

    def remove_sequence(self, seq: tuple):
        """Remove a sequence after it's been added as a token."""
        if seq in self.sequences:
            del self.sequences[seq]

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        freq_counts = list(self.sequences.values())

        return {
            'total_sequences_tracked': len(self.sequences),
            'total_tokens_observed': self.total_observed,
            'min_frequency': self.min_frequency,
            'max_length': self.max_length,
            'mean_frequency': sum(freq_counts) / len(freq_counts) if freq_counts else 0,
            'max_frequency': max(freq_counts) if freq_counts else 0,
        }

    def reset(self):
        """Clear all tracked sequences."""
        self.sequences.clear()
        self.total_observed = 0


# ===========================================================================
# VALIDATION
# ===========================================================================

def validate_token_frequency_tracker():
    """Test that token frequency tracker works correctly."""
    print("=" * 60)
    print("TOKEN FREQUENCY TRACKER VALIDATION")
    print("=" * 60)

    # Create tracker
    tracker = TokenFrequencyTracker(min_frequency=3, max_length=3)

    print("\n1. Testing observation...")
    # Simulate repeated pattern
    for _ in range(5):
        tokens = torch.tensor([100, 200, 300, 400, 100, 200, 300])
        tracker.observe(tokens)

    stats = tracker.get_statistics()
    print(f"   Sequences tracked: {stats['total_sequences_tracked']}")
    print(f"   Tokens observed: {stats['total_tokens_observed']}")
    assert stats['total_sequences_tracked'] > 0

    print("\n2. Testing candidates...")
    # Mock tokenizer for testing
    class MockTokenizer:
        def decode(self, tokens):
            return f"word_{tokens[0]}_{tokens[1]}"

        def encode(self, text):
            # Return 2 tokens (still multi-token)
            return [100, 200]

    candidates = tracker.get_candidates(MockTokenizer(), top_k=5)
    print(f"   Candidates found: {len(candidates)}")

    if candidates:
        top = candidates[0]
        print(f"   Top candidate: '{top['text']}'")
        print(f"   Frequency: {top['frequency']}")
        print(f"   Efficiency gain: {top['efficiency_gain']}")

    print("\n3. Testing sequence removal...")
    seq = (100, 200)
    tracker.remove_sequence(seq)
    assert seq not in tracker.sequences
    print("   ✓ Sequence removed")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE ✅")
    print("=" * 60)

    return tracker


if __name__ == "__main__":
    validate_token_frequency_tracker()
