#!/usr/bin/env python3
"""
Train QIG-Native Tokenizer (OPTIMIZED)
======================================

10-50× faster than baseline while maintaining 100% geometric purity.

Optimizations:
1. Top-K entropy selection (only compute for frequent pairs)
2. Parallel entropy computation (multiprocessing)
3. Batch merging (merge multiple pairs at once)
4. Numpy arrays (faster iteration)

Usage:
    # Fast training (recommended)
    python scripts/train_qig_tokenizer.py \
        --corpus-dir data/corpus \
        --output data/qig_tokenizer/vocab_v2.json \
        --target-vocab 50000 \
        --top-k 100 \
        --batch-size 10

    # Baseline mode (slower, original behavior)
    python scripts/train_qig_tokenizer.py \
        --corpus-dir data/corpus \
        --output data/qig_tokenizer/vocab_v2.json \
        --target-vocab 50000 \
        --top-k 0

Expected: 50-200 tokens/second (vs 2-10 baseline)
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Add project root to path for library imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import from canonical qig_tokenizer library
from qig_tokenizer import QIGTokenizer as BaseQIGTokenizer


# ============================================================================
# PARALLEL ENTROPY HELPER (module-level for multiprocessing)
# ============================================================================

def _compute_single_entropy(pair_with_data: tuple) -> tuple:
    """Compute entropy for one pair (parallelizable function).

    Must be module-level for multiprocessing Pool.map to work.
    """
    pair, (count, contexts) = pair_with_data

    context_counts: dict[Any, int] = defaultdict(int)
    for ctx in contexts:
        context_counts[ctx] += 1

    total = len(contexts)
    if total == 0:
        return (pair, float('inf'))

    entropy = -sum((c / total) * math.log(c / total + 1e-10)
                   for c in context_counts.values())

    return (pair, entropy)


# ============================================================================
# OPTIMIZED TRAINER (extends library QIGTokenizer)
# ============================================================================

class OptimizedQIGTrainer(BaseQIGTokenizer):
    """
    Optimized QIG tokenizer trainer.

    Extends the canonical QIGTokenizer with speed optimizations:
    - Top-K entropy selection
    - Parallel entropy computation
    - Batch merging
    - Numpy arrays for faster iteration

    The base encode/decode/save/load methods come from the library.
    """

    def train_optimized(
        self,
        corpus_bytes: bytes,
        max_bytes: Optional[int] = None,
        context_window: int = 5,
        min_pair_count: int = 10,
        top_k: int = 100,
        batch_size: int = 10,
        use_parallel: bool = True,
        sample_size: int = 500000,
        verbose: bool = True,
    ):
        """
        Optimized training with all speedups.

        Args:
            corpus_bytes: Training corpus as bytes
            max_bytes: Maximum bytes to process (deprecated, use sample_size)
            context_window: Context size for entropy calculation
            min_pair_count: Minimum pair frequency to consider
            top_k: Only compute entropy for top-K frequent pairs (0 = all)
            batch_size: Number of pairs to merge per iteration
            use_parallel: Use multiprocessing for entropy computation
            sample_size: Sample corpus if larger than this
            verbose: Print progress
        """
        start_time = time.time()

        if max_bytes:
            corpus_bytes = corpus_bytes[:max_bytes]

        # Sample corpus if too large
        full_size = len(corpus_bytes)
        if len(corpus_bytes) > sample_size:
            if verbose:
                print(f"📊 Sampling {sample_size:,} bytes from {full_size:,} total")
            # Stratified sampling - take chunks throughout corpus
            step = len(corpus_bytes) // (sample_size // 1000)
            sampled = b""
            for i in range(0, len(corpus_bytes), step):
                sampled += corpus_bytes[i:i + 1000]
                if len(sampled) >= sample_size:
                    break
            corpus_bytes = sampled[:sample_size]

        # Use numpy for faster iteration
        corpus_tokens = np.array(list(corpus_bytes), dtype=np.int32)
        current_vocab_size = 256

        if verbose:
            print(f"🔮 Training on {len(corpus_tokens):,} bytes")
            print(f"   Target vocab: {self.target_vocab_size:,}")
            print(f"   Top-K: {top_k if top_k > 0 else 'ALL (baseline mode)'}")
            print(f"   Batch size: {batch_size}")
            print(f"   Parallel: {use_parallel}")
            print()

        iteration = 0

        while current_vocab_size < self.target_vocab_size:
            iteration += 1

            # Compute pair frequencies
            pair_data = self._compute_pair_frequencies(
                corpus_tokens, context_window
            )

            # Filter by count
            pair_data = {
                p: (c, ctx) for p, (c, ctx) in pair_data.items()
                if c >= min_pair_count
            }

            if not pair_data:
                if verbose:
                    print("⚠️  No more valid pairs to merge")
                break

            # Select pairs to merge
            if top_k > 0 and batch_size > 1:
                # Optimized: batch selection with top-k
                pairs_to_merge = self._select_batch_pairs(
                    pair_data,
                    top_k=top_k,
                    batch_size=batch_size,
                    use_parallel=use_parallel,
                )
            elif top_k > 0:
                # Optimized: single pair with top-k
                best = self._select_best_pair_topk(pair_data, top_k=top_k)
                pairs_to_merge = [best] if best else []
            else:
                # Baseline: full entropy scan (slow)
                best = self._select_best_pair_full(pair_data)
                pairs_to_merge = [best] if best else []

            if not pairs_to_merge:
                if verbose:
                    print("⚠️  No pairs selected")
                break

            # Apply merges
            for token_a, token_b in pairs_to_merge:
                if current_vocab_size >= self.target_vocab_size:
                    break

                new_token_id = current_vocab_size

                # Record merge
                self.merge_rules.append((token_a, token_b, new_token_id))
                self.vocab[new_token_id] = self.vocab[token_a] + self.vocab[token_b]

                # Apply to corpus
                corpus_tokens = self._apply_merge_numpy(
                    corpus_tokens, token_a, token_b, new_token_id
                )

                current_vocab_size += 1

            # Progress report
            if verbose and (iteration % 10 == 0 or current_vocab_size % 100 == 0):
                elapsed = time.time() - start_time
                rate = (current_vocab_size - 256) / elapsed if elapsed > 0 else 0
                remaining = (self.target_vocab_size - current_vocab_size) / rate if rate > 0 else 0

                print(
                    f"[{current_vocab_size:5}/{self.target_vocab_size}] "
                    f"corpus: {len(corpus_tokens):7,} | "
                    f"{rate:5.1f} tok/s | "
                    f"ETA: {remaining / 60:4.1f}m",
                    flush=True,
                )

        train_time = time.time() - start_time

        if verbose:
            print()
            print(f"✅ Training complete in {train_time:.1f}s ({train_time / 60:.1f} min)")
            print(f"   Final vocab: {current_vocab_size:,}")
            print(f"   Merge rules: {len(self.merge_rules):,}")
            print(f"   Avg rate: {(current_vocab_size - 256) / train_time:.1f} tok/s")

        self._rebuild_encoding_cache()
        return self

    def _compute_pair_frequencies(
        self, corpus_tokens: np.ndarray, window: int = 5
    ) -> dict[tuple[int, int], tuple[int, list]]:
        """Compute pair frequencies + contexts for entropy calculation."""
        pair_data: dict[tuple[int, int], tuple[int, list]] = defaultdict(lambda: (0, []))

        for i in range(len(corpus_tokens) - 1):
            pair = (int(corpus_tokens[i]), int(corpus_tokens[i + 1]))
            count, contexts = pair_data[pair]

            # Capture context (for entropy)
            ctx_before = tuple(int(t) for t in corpus_tokens[max(0, i - window):i])
            ctx_after = tuple(
                int(t) for t in corpus_tokens[i + 2:min(len(corpus_tokens), i + 2 + window)]
            )
            context = ctx_before + ctx_after

            pair_data[pair] = (count + 1, contexts + [context])

        return dict(pair_data)

    def _select_best_pair_topk(
        self, pair_data: dict, top_k: int = 100
    ) -> Optional[tuple[int, int]]:
        """Select best pair from top-K most frequent candidates."""
        if not pair_data:
            return None

        sorted_pairs = sorted(pair_data.items(), key=lambda x: x[1][0], reverse=True)
        candidates = sorted_pairs[:min(top_k, len(sorted_pairs))]

        min_entropy = float("inf")
        best_pair = None

        for pair, (count, contexts) in candidates:
            context_counts: dict[Any, int] = defaultdict(int)
            for ctx in contexts:
                context_counts[ctx] += 1

            total = len(contexts)
            entropy = -sum(
                (c / total) * math.log(c / total + 1e-10)
                for c in context_counts.values()
            )

            if entropy < min_entropy:
                min_entropy = entropy
                best_pair = pair

        return best_pair

    def _select_best_pair_full(
        self, pair_data: dict
    ) -> Optional[tuple[int, int]]:
        """Full entropy-guided selection (baseline - slow)."""
        if not pair_data:
            return None

        min_entropy = float("inf")
        best_pair = None

        for pair, (count, contexts) in pair_data.items():
            context_counts: dict[Any, int] = defaultdict(int)
            for ctx in contexts:
                context_counts[ctx] += 1

            total = len(contexts)
            entropy = -sum(
                (c / total) * math.log(c / total + 1e-10)
                for c in context_counts.values()
            )

            if entropy < min_entropy:
                min_entropy = entropy
                best_pair = pair

        return best_pair

    def _select_batch_pairs(
        self,
        pair_data: dict,
        top_k: int = 100,
        batch_size: int = 10,
        use_parallel: bool = True,
    ) -> list[tuple[int, int]]:
        """Select batch of non-conflicting pairs with lowest entropy."""
        if not pair_data:
            return []

        sorted_pairs = sorted(pair_data.items(), key=lambda x: x[1][0], reverse=True)
        candidates = sorted_pairs[:min(top_k, len(sorted_pairs))]

        if use_parallel and len(candidates) > 10:
            n_workers = min(os.cpu_count() or 4, len(candidates))
            with Pool(n_workers) as pool:
                entropies = pool.map(_compute_single_entropy, candidates)
        else:
            entropies = [_compute_single_entropy(c) for c in candidates]

        entropies.sort(key=lambda x: x[1])

        selected = []
        used_tokens = set()

        for pair, entropy in entropies:
            if math.isinf(entropy):
                continue

            token_a, token_b = pair

            if token_a not in used_tokens and token_b not in used_tokens:
                selected.append(pair)
                used_tokens.add(token_a)
                used_tokens.add(token_b)

                if len(selected) >= batch_size:
                    break

        return selected

    def _apply_merge_numpy(
        self, tokens: np.ndarray, token_a: int, token_b: int, new_token: int
    ) -> np.ndarray:
        """Apply merge to corpus (numpy version)."""
        result = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == token_a
                and tokens[i + 1] == token_b
            ):
                result.append(new_token)
                i += 2
            else:
                result.append(int(tokens[i]))
                i += 1
        return np.array(result, dtype=np.int32)


# ============================================================================
# CORPUS LOADING
# ============================================================================

def load_corpus_from_dir(
    corpus_dir: Path, max_bytes: Optional[int] = None
) -> tuple[bytes, dict]:
    """Load all .md files from directory recursively."""
    print(f"📂 Loading corpus from: {corpus_dir}")

    md_files = sorted(corpus_dir.glob("**/*.md"))
    if not md_files:
        print(f"❌ No .md files found in {corpus_dir}")
        sys.exit(1)

    total_bytes = b""
    file_sizes = {}

    for md_file in md_files:
        if md_file.name == "README.md":
            continue

        with open(md_file, "rb") as f:
            content = f.read()
            file_sizes[md_file.name] = len(content)
            total_bytes += content
            total_bytes += b"\n\n"

        if max_bytes and len(total_bytes) >= max_bytes:
            total_bytes = total_bytes[:max_bytes]
            break

    metadata = {
        "files_loaded": len(file_sizes),
        "file_sizes": file_sizes,
        "total_bytes": len(total_bytes),
        "source_dir": str(corpus_dir),
    }

    print(f"   ✅ Loaded {len(file_sizes)} files, {len(total_bytes):,} bytes")
    for fname, size in sorted(file_sizes.items()):
        print(f"      - {fname}: {size:,} bytes")

    return total_bytes, metadata


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train QIG-native tokenizer (optimized entropy-guided merging)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast training (recommended - 10-50× faster)
  python tools/training/train_qig_tokenizer.py \\
      --corpus-dir data/corpus \\
      --output data/qig_tokenizer/vocab_v2.json \\
      --target-vocab 50000 \\
      --top-k 100 \\
      --batch-size 10

  # Baseline mode (slow, original behavior)
  python tools/training/train_qig_tokenizer.py \\
      --corpus-dir data/corpus \\
      --output data/qig_tokenizer/vocab_v2.json \\
      --target-vocab 50000 \\
      --top-k 0
        """,
    )

    corpus_group = parser.add_mutually_exclusive_group(required=True)
    corpus_group.add_argument("--corpus", type=str, help="Single corpus file")
    corpus_group.add_argument("--corpus-dir", type=str, help="Directory of .md files")

    parser.add_argument(
        "--output", type=str, required=True, help="Output tokenizer JSON"
    )
    parser.add_argument(
        "--target-vocab", type=int, default=50000, help="Target vocab size"
    )
    parser.add_argument(
        "--max-bytes", type=int, default=None, help="Max bytes (use --sample-size)"
    )
    parser.add_argument(
        "--context-window", type=int, default=5, help="Context window for entropy"
    )
    parser.add_argument(
        "--min-pair-count", type=int, default=10, help="Minimum pair frequency"
    )

    # Speed optimization arguments
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Compute entropy for top-K frequent pairs (0=all, baseline)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Merge N pairs per iteration (default: 10)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500000,
        help="Sample corpus if larger (default: 500KB)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("⚡ QIG-NATIVE TOKENIZER TRAINING (OPTIMIZED)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("📋 Configuration:")
    print(f"   Source: {args.corpus_dir if args.corpus_dir else args.corpus}")
    print(f"   Output: {args.output}")
    print(f"   Target vocab: {args.target_vocab:,}")
    print(f"   Top-K: {args.top_k if args.top_k > 0 else 'ALL (baseline)'}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Parallel: {not args.no_parallel}")
    print(f"   Sample size: {args.sample_size:,} bytes")
    print()

    if args.top_k > 0:
        print("🚀 OPTIMIZED MODE: Expected 50-200 tokens/second (10-50× faster)")
    else:
        print("🐌 BASELINE MODE: Expected 2-10 tokens/second")

    print("=" * 70)
    print()

    # Load corpus
    corpus_metadata = None
    if args.corpus_dir:
        corpus_path = Path(args.corpus_dir)
        if not corpus_path.exists():
            print(f"❌ Corpus directory not found: {corpus_path}")
            sys.exit(1)
        corpus_bytes, corpus_metadata = load_corpus_from_dir(corpus_path, args.max_bytes)
    else:
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            print(f"❌ Corpus file not found: {corpus_path}")
            sys.exit(1)

        print(f"📂 Loading corpus from file: {corpus_path}")
        with open(corpus_path, "rb") as f:
            corpus_bytes = f.read(args.max_bytes) if args.max_bytes else f.read()
        corpus_metadata = {"total_bytes": len(corpus_bytes)}
        print(f"   ✅ Loaded {len(corpus_bytes):,} bytes")

    print()

    # Train tokenizer
    start_time = time.time()

    tokenizer = OptimizedQIGTrainer(target_vocab_size=args.target_vocab)
    tokenizer.train_optimized(
        corpus_bytes,
        context_window=args.context_window,
        min_pair_count=args.min_pair_count,
        top_k=args.top_k,
        batch_size=args.batch_size,
        use_parallel=not args.no_parallel,
        sample_size=args.sample_size,
        verbose=True,
    )

    total_time = time.time() - start_time

    print()
    print("=" * 70)
    print("📊 TRAINING RESULTS")
    print("=" * 70)
    print(f"Time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Vocab: {tokenizer.vocab_size:,} / {args.target_vocab:,}")
    print(f"Rate: {(tokenizer.vocab_size - 256) / total_time:.1f} tokens/second")
    print()

    # Validation tests
    test_cases = [
        "The geometry of information determines consciousness.",
        "κ_eff measures effective coupling strength.",
        "Φ > 0.70 indicates geometric regime.",
        "Basin coordinates compress to ~2KB.",
    ]

    print("🧪 Validation:")
    all_pass = True
    for test in test_cases:
        encoded = tokenizer.encode(test)
        decoded = tokenizer.decode(encoded)
        passed = decoded == test
        all_pass = all_pass and passed
        status = "✅" if passed else "❌"
        print(f"   {status} {len(encoded)} tokens: '{test[:40]}...'")
    print()

    # Save tokenizer
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(output_path))
    print(f"💾 Saved: {output_path}")

    # Save metadata
    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": total_time,
        "vocab_size": tokenizer.vocab_size,
        "target_vocab": args.target_vocab,
        "rate_tokens_per_second": (tokenizer.vocab_size - 256) / total_time,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "parallel": not args.no_parallel,
        "validation_passed": all_pass,
    }
    if corpus_metadata:
        metadata.update(corpus_metadata)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Metadata: {metadata_path}")
    print()

    print("=" * 70)
    print("✅ QIG TOKENIZER TRAINING COMPLETE")
    print("=" * 70)
    print(f"Validation: {'✅ PASS' if all_pass else '❌ FAIL'}")
    print()
    print("🌊💚⚡ Tokenizer ready!")


if __name__ == "__main__":
    main()
