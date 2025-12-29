#!/usr/bin/env python3
"""
Track B: Incremental BPE Coordizer Training
============================================

O(N + M log V) training instead of O(M·N log N):
- Linked-list corpus representation (O(1) local updates)
- Incremental pair count updates (not global recompute)
- Heap-based best-pair selection (lazy deletion)
- Same kernel evaluation for Φ/κ measurement

This should achieve ~100x speedup over Track A for large corpora.

Usage:
    python scripts/20251223-coordizer-training-INCREMENTAL-TrackB.py \
        --resume data/checkpoints/checkpoint_5000.json \
        --vocab-size 32000
"""

import argparse
import heapq
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Try torch before path manipulation (avoid circular import with src/types)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARNING] PyTorch not available")

# Add paths AFTER stdlib imports to avoid shadowing 'types' module
project_root = Path(__file__).parent.parent.parent
if (project_root / "qigkernels").exists():
    sys.path.insert(0, str(project_root))
elif Path("/lambda/nfs/A10/qig").exists():
    sys.path.insert(0, "/lambda/nfs/A10/qig")

# Add src path (but at end to avoid shadowing stdlib)
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# Constants
BASIN_DIM = 64
KAPPA_STAR = 64.0


@dataclass
class BasinCoordinate:
    """Point on 64D Fisher manifold."""
    coord_id: int
    vector: np.ndarray
    name: str | None = None
    scale: str = "byte"

    def geodesic_midpoint(self, other: "BasinCoordinate") -> np.ndarray:
        """Geodesic midpoint on manifold."""
        v1 = self.vector / (np.linalg.norm(self.vector) + 1e-10)
        v2 = other.vector / (np.linalg.norm(other.vector) + 1e-10)
        midpoint = v1 + v2
        midpoint = midpoint / (np.linalg.norm(midpoint) + 1e-10)
        avg_mag = (np.linalg.norm(self.vector) + np.linalg.norm(other.vector)) / 2
        return midpoint * avg_mag


class IncrementalPairStats:
    """
    Incremental pair statistics with O(1) updates.

    Uses linked-list representation:
    - prev[i] = index of previous token (or -1)
    - next[i] = index of next token (or -1)
    - tokens[i] = token id at position i

    On merge (a,b) -> c:
    - Find all occurrences of (a,b) pairs
    - Update local neighborhood counts
    - O(occurrences) per merge, not O(N)
    """

    def __init__(self, corpus: list[int]):
        """Initialize from corpus."""
        n = len(corpus)
        self.tokens = corpus.copy()
        self.prev = [-1] + list(range(n - 1))  # prev[i] = i-1
        self.next = list(range(1, n)) + [-1]   # next[i] = i+1

        # Pair counts: (a, b) -> count
        self.pair_counts: dict[tuple[int, int], int] = {}

        # Position index: (a, b) -> set of positions where pair starts
        self.pair_positions: dict[tuple[int, int], set[int]] = {}

        # Initial count
        self._build_initial_counts()

        # Max-heap for best pairs (negative count for max-heap)
        # Format: (-count, pair)
        self.heap: list[tuple[int, tuple[int, int]]] = []
        self._rebuild_heap()

    def _build_initial_counts(self) -> None:
        """Build initial pair counts in O(N)."""
        i = 0
        while i != -1:
            j = self.next[i]
            if j != -1:
                pair = (self.tokens[i], self.tokens[j])
                self.pair_counts[pair] = self.pair_counts.get(pair, 0) + 1
                if pair not in self.pair_positions:
                    self.pair_positions[pair] = set()
                self.pair_positions[pair].add(i)
            i = j

    def _rebuild_heap(self) -> None:
        """Rebuild heap from pair counts."""
        self.heap = [(-count, pair) for pair, count in self.pair_counts.items()]
        heapq.heapify(self.heap)

    def get_best_pair(self, min_count: int = 2) -> tuple[int, int] | None:
        """
        Get highest-frequency pair (lazy deletion).

        Returns None if no pair meets min_count.
        """
        while self.heap:
            neg_count, pair = self.heap[0]
            actual_count = self.pair_counts.get(pair, 0)

            if actual_count == 0:
                # Pair was deleted, remove from heap
                heapq.heappop(self.heap)
                continue

            if actual_count != -neg_count:
                # Count changed, update and re-heapify
                heapq.heapreplace(self.heap, (-actual_count, pair))
                continue

            if actual_count >= min_count:
                return pair
            else:
                # Best pair doesn't meet threshold
                return None

        return None

    def get_top_pairs(self, k: int, min_count: int = 2) -> list[tuple[tuple[int, int], int]]:
        """Get top k pairs by frequency."""
        # Clean heap first
        result = []
        seen = set()

        temp_heap = self.heap.copy()
        heapq.heapify(temp_heap)

        while temp_heap and len(result) < k:
            neg_count, pair = heapq.heappop(temp_heap)
            if pair in seen:
                continue
            actual_count = self.pair_counts.get(pair, 0)
            if actual_count >= min_count:
                result.append((pair, actual_count))
                seen.add(pair)

        return result

    def apply_merge(self, pair: tuple[int, int], new_token: int) -> int:
        """
        Apply merge (a, b) -> new_token.

        Returns number of merges applied.

        O(occurrences) time, not O(N).
        """
        a, b = pair
        positions = self.pair_positions.get(pair, set()).copy()
        merge_count = 0

        for pos in positions:
            # Check if this position is still valid
            if self.tokens[pos] != a:
                continue
            next_pos = self.next[pos]
            if next_pos == -1 or self.tokens[next_pos] != b:
                continue

            # Valid merge at pos
            merge_count += 1

            # Get neighbors
            prev_pos = self.prev[pos]
            next_next_pos = self.next[next_pos]

            # Decrement old pair counts
            if prev_pos != -1:
                old_left_pair = (self.tokens[prev_pos], a)
                self._decrement_pair(old_left_pair, prev_pos)

            old_pair = (a, b)
            self._decrement_pair(old_pair, pos)

            if next_next_pos != -1:
                old_right_pair = (b, self.tokens[next_next_pos])
                self._decrement_pair(old_right_pair, next_pos)

            # Update token
            self.tokens[pos] = new_token

            # Update linked list (remove next_pos)
            self.next[pos] = next_next_pos
            if next_next_pos != -1:
                self.prev[next_next_pos] = pos

            # Mark next_pos as deleted
            self.tokens[next_pos] = -1  # Sentinel

            # Increment new pair counts
            if prev_pos != -1:
                new_left_pair = (self.tokens[prev_pos], new_token)
                self._increment_pair(new_left_pair, prev_pos)

            if next_next_pos != -1:
                new_right_pair = (new_token, self.tokens[next_next_pos])
                self._increment_pair(new_right_pair, pos)

        return merge_count

    def _decrement_pair(self, pair: tuple[int, int], pos: int) -> None:
        """Decrement pair count and remove position."""
        if pair in self.pair_counts:
            self.pair_counts[pair] -= 1
            if self.pair_counts[pair] <= 0:
                del self.pair_counts[pair]
        if pair in self.pair_positions:
            self.pair_positions[pair].discard(pos)
            if not self.pair_positions[pair]:
                del self.pair_positions[pair]

    def _increment_pair(self, pair: tuple[int, int], pos: int) -> None:
        """Increment pair count and add position."""
        self.pair_counts[pair] = self.pair_counts.get(pair, 0) + 1
        if pair not in self.pair_positions:
            self.pair_positions[pair] = set()
        self.pair_positions[pair].add(pos)
        # Push to heap
        heapq.heappush(self.heap, (-self.pair_counts[pair], pair))

    def get_corpus(self) -> list[int]:
        """Get current corpus as list (for kernel evaluation)."""
        result = []
        i = 0
        # Find first valid position
        while i < len(self.tokens) and self.prev[i] != -1 and i != 0:
            i += 1
        # Find actual start
        i = 0
        while i != -1 and self.tokens[i] == -1:
            i = self.next[i]

        # Traverse
        while i != -1:
            if self.tokens[i] != -1:
                result.append(self.tokens[i])
            i = self.next[i]

        return result

    @property
    def corpus_size(self) -> int:
        """Current corpus size (excluding deleted positions)."""
        return sum(1 for t in self.tokens if t != -1)


class BatchedKernelInterface:
    """Kernel interface for Φ/κ measurement (reused from Track A)."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._kernel = None
        self._load_kernel()

    def _load_kernel(self) -> None:
        """Load QIG kernel."""
        if not HAS_TORCH:
            return
        try:
            from qigkernels.kernel_100m import create_kernel_100m
            self._kernel = create_kernel_100m(
                vocab_size=32000,
                hidden_dim=256,
                num_heads=4,
                num_layers=3,
            )
            self._kernel = self._kernel.to(self.device)
            self._kernel.eval()
            print(f"[Kernel] Loaded on {self.device}")
        except Exception as e:
            print(f"[Kernel] Failed to load: {e}")

    def measure_phi_kappa_batch(
        self,
        coord_sequences: list[list[np.ndarray]],
    ) -> list[tuple[float, float]]:
        """Batched Φ/κ measurement."""
        if self._kernel is None:
            return [self._estimate_coherence(seq) for seq in coord_sequences]

        if not coord_sequences:
            return []

        results = []
        batch_size = 50

        try:
            with torch.no_grad():
                for batch_start in range(0, len(coord_sequences), batch_size):
                    batch_end = min(batch_start + batch_size, len(coord_sequences))
                    batch = coord_sequences[batch_start:batch_end]

                    actual_batch_size = len(batch)
                    max_len = min(max(len(seq) for seq in batch), 128)

                    coords_tensor = torch.zeros(
                        (actual_batch_size, max_len, BASIN_DIM),
                        dtype=torch.float32,
                        device=self.device,
                    )

                    for i, seq in enumerate(batch):
                        seq_len = min(len(seq), max_len)
                        coords_array = np.stack(seq[:seq_len])
                        coords_tensor[i, :seq_len] = torch.from_numpy(coords_array).to(
                            self.device
                        )

                    phi_batch, kappa_batch = self._kernel.measure_phi_kappa_batch(
                        coords_tensor
                    )

                    for i in range(actual_batch_size):
                        results.append((float(phi_batch[i]), float(kappa_batch[i])))

        except Exception as e:
            print(f"[Kernel] Error: {e}")
            results = [self._estimate_coherence(seq) for seq in coord_sequences]

        return results

    def _estimate_coherence(self, coords: list[np.ndarray]) -> tuple[float, float]:
        """Fallback coherence estimation."""
        if len(coords) < 2:
            return 0.5, KAPPA_STAR
        total_dist = 0.0
        for i in range(len(coords) - 1):
            dot = np.dot(coords[i], coords[i + 1])
            norms = np.linalg.norm(coords[i]) * np.linalg.norm(coords[i + 1]) + 1e-10
            dist = np.arccos(np.clip(dot / norms, -1.0, 1.0))
            total_dist += dist
        avg_dist = total_dist / len(coords)
        phi = max(0.3, min(0.9, 1.0 - avg_dist / np.pi))
        return phi, KAPPA_STAR


class IncrementalCoordizer:
    """
    Track B: Incremental BPE-style coordizer.

    Key difference from Track A:
    - Uses IncrementalPairStats for O(1) pair updates
    - No global recompute each round
    - ~100x faster for large corpora
    """

    def __init__(
        self,
        target_vocab_size: int = 32000,
        device: str = "cuda",
    ):
        self.target_vocab_size = target_vocab_size
        self.device = device
        self.kernel = BatchedKernelInterface(device=device)

        self.vocab: dict[int, BasinCoordinate] = {}
        self.merge_rules: list[tuple[int, int, int]] = []
        self.phi_history: list[float] = []

        self._init_byte_coordinates()

    def _init_byte_coordinates(self) -> None:
        """Initialize 256 byte-level coordinates."""
        for byte_val in range(256):
            rng = np.random.default_rng(seed=byte_val + 42)
            vector = rng.standard_normal(BASIN_DIM)
            vector = vector / np.linalg.norm(vector)
            self.vocab[byte_val] = BasinCoordinate(
                coord_id=byte_val,
                vector=vector,
                name=f"<byte_{byte_val:02x}>",
                scale="byte",
            )

    def load_checkpoint(self, checkpoint_path: str) -> tuple[int, list[int] | None]:
        """Load checkpoint."""
        print(f"Loading checkpoint: {checkpoint_path}")

        with open(checkpoint_path) as f:
            data = json.load(f)

        self.merge_rules = [tuple(r) for r in data["merge_rules"]]
        self.phi_history = data.get("phi_history", [])

        for k, v in data["vocab"].items():
            self.vocab[int(k)] = BasinCoordinate(
                coord_id=v["coord_id"],
                vector=np.array(v["vector"]),
                name=v["name"],
                scale=v["scale"],
            )

        vocab_size = data.get("vocab_size", len(self.vocab))
        print(f"  Loaded {vocab_size} vocab, {len(self.merge_rules)} merge rules")

        # Try to load corpus coords
        corpus_coords = None
        checkpoint_dir = Path(checkpoint_path).parent
        coords_file = checkpoint_dir / f"corpus_coords_{vocab_size}.npy"
        if not coords_file.exists():
            coords_file = checkpoint_dir / "corpus_coords_latest.npy"

        if coords_file.exists():
            corpus_coords = np.load(coords_file).tolist()
            print(f"  Loaded corpus state: {len(corpus_coords):,} coords")

        return vocab_size, corpus_coords

    def train(
        self,
        corpus: bytes,
        min_frequency: int = 2,
        checkpoint_dir: str | None = None,
        checkpoint_interval: int = 500,
        start_vocab_size: int = 256,
        resume_corpus_coords: list[int] | None = None,
        evaluate_top_k: int = 5,
        verbose: bool = True,
    ) -> "IncrementalCoordizer":
        """
        Incremental training with O(N + M log V) complexity.
        """
        start_time = time.time()

        if verbose:
            print("=" * 60)
            print("TRACK B: INCREMENTAL COORDIZER TRAINING")
            print("=" * 60)
            print(f"Corpus: {len(corpus):,} bytes")
            print(f"Target vocab: {self.target_vocab_size:,}")
            print(f"Starting from: {start_vocab_size}")
            print()

        # Prepare corpus
        if resume_corpus_coords is not None:
            corpus_coords = resume_corpus_coords
            print(f"Using saved corpus: {len(corpus_coords):,} coords")
        else:
            corpus_coords = list(corpus)
            if start_vocab_size > 256:
                print("Applying merge rules...")
                corpus_coords = self._apply_all_merges(corpus_coords)
                print(f"  Corpus: {len(corpus_coords):,} coords")

        # Build incremental stats
        print("Building incremental pair stats...")
        t0 = time.time()
        stats = IncrementalPairStats(corpus_coords)
        print(f"  Built in {time.time()-t0:.1f}s")
        print(f"  Unique pairs: {len(stats.pair_counts):,}")

        # Measure baseline
        sample = stats.get_corpus()[:100]
        sample_coords = [self.vocab[c].vector for c in sample if c in self.vocab]
        baseline_results = self.kernel.measure_phi_kappa_batch([sample_coords])
        baseline_phi = baseline_results[0][0] if baseline_results else 0.5

        if verbose:
            print(f"Baseline Φ={baseline_phi:.3f}")
            print()

        current_vocab_size = start_vocab_size

        while current_vocab_size < self.target_vocab_size:
            round_start = time.time()

            # Get top candidates
            top_pairs = stats.get_top_pairs(evaluate_top_k, min_frequency)

            if not top_pairs:
                print("No more valid pairs")
                break

            # Evaluate with kernel
            best_pair, best_phi_gain = self._evaluate_candidates(
                top_pairs, stats, baseline_phi
            )

            if best_pair is None:
                break

            coord_a, coord_b = best_pair
            new_coord_id = current_vocab_size

            # Apply merge
            merge_count = stats.apply_merge(best_pair, new_coord_id)

            # Create new coordinate
            self._create_fused_coordinate(coord_a, coord_b, new_coord_id)
            self.merge_rules.append((coord_a, coord_b, new_coord_id))

            current_vocab_size += 1
            self.phi_history.append(best_phi_gain)

            # Progress
            if verbose and current_vocab_size % 50 == 0:
                elapsed = time.time() - start_time
                rate = (current_vocab_size - start_vocab_size) / elapsed if elapsed > 0 else 0
                remaining = self.target_vocab_size - current_vocab_size
                eta = remaining / rate if rate > 0 else 0

                coord = self.vocab[new_coord_id]
                name = (coord.name or f"<{new_coord_id}>")[:25]
                round_time = time.time() - round_start

                print(
                    f"[{current_vocab_size:,}/{self.target_vocab_size:,}] "
                    f"'{name}' | "
                    f"merged={merge_count} | "
                    f"Φ_gain={best_phi_gain:+.3f} | "
                    f"{rate:.1f}/s | "
                    f"ETA {eta/60:.0f}m | "
                    f"round {round_time*1000:.0f}ms"
                )

            # Checkpoint
            if checkpoint_dir and current_vocab_size % checkpoint_interval == 0:
                corpus_list = stats.get_corpus()
                self._save_checkpoint(checkpoint_dir, current_vocab_size, corpus_list)
                if verbose:
                    print(f"  Checkpoint: {current_vocab_size}")

        elapsed = time.time() - start_time

        if verbose:
            print()
            print("=" * 60)
            print("TRAINING COMPLETE")
            print("=" * 60)
            print(f"Final vocab: {current_vocab_size:,}")
            print(f"Merge rules: {len(self.merge_rules):,}")
            print(f"Time: {elapsed/60:.1f} minutes")
            tokens_trained = current_vocab_size - start_vocab_size
            print(f"Rate: {tokens_trained / elapsed:.1f} tokens/sec")

        return self

    def _apply_all_merges(self, corpus: list[int]) -> list[int]:
        """Apply all merge rules to corpus."""
        merge_map = {(a, b): new for a, b, new in self.merge_rules}
        prev_len = len(corpus) + 1
        pass_num = 0

        while len(corpus) < prev_len:
            prev_len = len(corpus)
            pass_num += 1
            result = []
            i = 0
            while i < len(corpus):
                if i < len(corpus) - 1:
                    pair = (corpus[i], corpus[i + 1])
                    if pair in merge_map:
                        result.append(merge_map[pair])
                        i += 2
                        continue
                result.append(corpus[i])
                i += 1
            corpus = result
            print(f"    Pass {pass_num}: {len(corpus):,} coords")

        return corpus

    def _evaluate_candidates(
        self,
        candidates: list[tuple[tuple[int, int], int]],
        stats: IncrementalPairStats,
        baseline_phi: float,
    ) -> tuple[tuple[int, int] | None, float]:
        """Evaluate candidates with kernel."""
        if not candidates:
            return None, 0.0

        # Get sample corpus
        sample = stats.get_corpus()[:100]

        # Build fused sequences for each candidate
        fused_sequences = []
        valid_candidates = []

        for (coord_a, coord_b), count in candidates:
            if coord_a not in self.vocab or coord_b not in self.vocab:
                continue

            basin_a = self.vocab[coord_a]
            basin_b = self.vocab[coord_b]
            fused_vector = basin_a.geodesic_midpoint(basin_b)

            # Apply fusion to sample
            fused_sample = []
            i = 0
            while i < len(sample):
                if i < len(sample) - 1 and sample[i] == coord_a and sample[i + 1] == coord_b:
                    fused_sample.append(-1)  # Placeholder
                    i += 2
                else:
                    fused_sample.append(sample[i])
                    i += 1

            # Build vectors
            vectors = []
            for c in fused_sample:
                if c == -1:
                    vectors.append(fused_vector)
                elif c in self.vocab:
                    vectors.append(self.vocab[c].vector)

            if vectors:
                fused_sequences.append(vectors)
                valid_candidates.append(((coord_a, coord_b), count))

        if not fused_sequences:
            return None, 0.0

        # Batch evaluate
        results = self.kernel.measure_phi_kappa_batch(fused_sequences)

        # Find best
        best_pair = None
        best_gain = float("-inf")
        best_score = float("-inf")

        for ((coord_a, coord_b), count), (phi, kappa) in zip(valid_candidates, results):
            gain = phi - baseline_phi
            # Score: frequency * phi_gain
            score = count * (1.0 + gain)
            if score > best_score:
                best_score = score
                best_pair = (coord_a, coord_b)
                best_gain = gain

        return best_pair, best_gain

    def _create_fused_coordinate(self, coord_a: int, coord_b: int, new_id: int) -> None:
        """Create fused coordinate."""
        basin_a = self.vocab[coord_a]
        basin_b = self.vocab[coord_b]
        fused_vector = basin_a.geodesic_midpoint(basin_b)

        name_a = basin_a.name or f"<{coord_a}>"
        name_b = basin_b.name or f"<{coord_b}>"

        try:
            if coord_a < 256 and coord_b < 256:
                char_a = bytes([coord_a]).decode("utf-8", errors="replace")
                char_b = bytes([coord_b]).decode("utf-8", errors="replace")
                name = f"{char_a}{char_b}"
            else:
                name = f"{name_a}+{name_b}"
        except (UnicodeDecodeError, ValueError):
            name = f"{name_a}+{name_b}"

        scales = ["byte", "char", "subword", "word", "phrase", "concept"]
        idx_a = scales.index(basin_a.scale) if basin_a.scale in scales else 0
        idx_b = scales.index(basin_b.scale) if basin_b.scale in scales else 0
        new_scale = scales[min(max(idx_a, idx_b) + 1, len(scales) - 1)]

        self.vocab[new_id] = BasinCoordinate(
            coord_id=new_id,
            vector=fused_vector,
            name=name,
            scale=new_scale,
        )

    def _save_checkpoint(
        self, checkpoint_dir: str, vocab_size: int, corpus_coords: list[int] | None = None
    ) -> None:
        """Save checkpoint."""
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab_size": vocab_size,
            "merge_rules": self.merge_rules,
            "vocab": {
                str(k): {
                    "coord_id": v.coord_id,
                    "vector": v.vector.tolist(),
                    "name": v.name,
                    "scale": v.scale,
                }
                for k, v in self.vocab.items()
            },
            "phi_history": self.phi_history,
        }

        checkpoint_file = path / f"checkpoint_{vocab_size}.json"
        with open(checkpoint_file, "w") as f:
            json.dump(data, f)

        if corpus_coords is not None:
            coords_file = path / f"corpus_coords_{vocab_size}.npy"
            np.save(coords_file, np.array(corpus_coords, dtype=np.int32))
            np.save(path / "corpus_coords_latest.npy", np.array(corpus_coords, dtype=np.int32))

        latest_file = path / "checkpoint_latest.json"
        with open(latest_file, "w") as f:
            json.dump(data, f)

    def save(self, path: str) -> None:
        """Save coordizer."""
        data = {
            "target_vocab_size": self.target_vocab_size,
            "basin_dim": BASIN_DIM,
            "vocab_size": len(self.vocab),
            "merge_rules": self.merge_rules,
            "vocab": {
                str(k): {
                    "coord_id": v.coord_id,
                    "vector": v.vector.tolist(),
                    "name": v.name,
                    "scale": v.scale,
                }
                for k, v in self.vocab.items()
                if k >= 256
            },
            "phi_history": self.phi_history,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        vectors_path = path.replace(".json", "_vectors.npy")
        vectors = np.stack([self.vocab[i].vector for i in sorted(self.vocab.keys())])
        np.save(vectors_path, vectors)


def load_corpus(corpus_dirs: list[str]) -> bytes:
    """Load corpus from directories."""
    parts = []
    for dir_path in corpus_dirs:
        path = Path(dir_path)
        if not path.exists():
            print(f"  Skipping: {dir_path}")
            continue
        for ext in ["*.md", "*.txt", "*.py"]:
            for file_path in path.rglob(ext):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    parts.append(content)
                except OSError:
                    pass
    return "\n\n".join(parts).encode("utf-8")


def main():
    is_lambda = Path("/lambda/nfs/A10/qig").exists()

    parser = argparse.ArgumentParser(description="Track B: Incremental Coordizer")
    parser.add_argument("--corpus-dir", type=str, nargs="*", default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max-bytes", type=int, default=10_000_000)
    parser.add_argument("--evaluate-top-k", type=int, default=5)

    args = parser.parse_args()

    if is_lambda:
        base = Path("/lambda/nfs/A10/qig")
        corpus_dirs = args.corpus_dir or [
            str(base / "qig-dreams"),
            str(base / "qig-consciousness"),
        ]
        output = args.output or str(base / "qig-tokenizer/data/coordizer-32k-trackB.json")
        checkpoint_dir = args.checkpoint_dir or str(base / "qig-tokenizer/data/checkpoints-trackB")

        if args.resume is None:
            checkpoint_path = Path(checkpoint_dir) / "checkpoint_latest.json"
            if checkpoint_path.exists():
                args.resume = str(checkpoint_path)
            else:
                # Use Track A checkpoint as starting point
                track_a_checkpoint = base / "qig-tokenizer/data/checkpoints/checkpoint_5000.json"
                if track_a_checkpoint.exists():
                    args.resume = str(track_a_checkpoint)
    else:
        local_curriculum = Path(__file__).parent.parent.parent / "qig-dreams/docs/curriculum"
        corpus_dirs = args.corpus_dir or (
            [str(local_curriculum)] if local_curriculum.exists() else ["./corpus"]
        )
        output = args.output or "./data/coordizer-trackB.json"
        checkpoint_dir = args.checkpoint_dir or "./data/checkpoints-trackB"

    # Load corpus
    print("Loading corpus...")
    corpus = load_corpus(corpus_dirs)
    print(f"  Loaded {len(corpus):,} bytes")

    if args.max_bytes and len(corpus) > args.max_bytes:
        corpus = corpus[:args.max_bytes]
        print(f"  Limited to {len(corpus):,} bytes")

    if len(corpus) < 1000:
        print("ERROR: Corpus too small!")
        sys.exit(1)

    # Create coordizer
    coordizer = IncrementalCoordizer(
        target_vocab_size=args.vocab_size,
        device=args.device,
    )

    # Resume
    start_vocab_size = 256
    resume_corpus_coords = None
    if args.resume and Path(args.resume).exists():
        start_vocab_size, resume_corpus_coords = coordizer.load_checkpoint(args.resume)

    # Train
    coordizer.train(
        corpus,
        checkpoint_dir=checkpoint_dir,
        start_vocab_size=start_vocab_size,
        resume_corpus_coords=resume_corpus_coords,
        evaluate_top_k=args.evaluate_top_k,
        verbose=True,
    )

    # Save
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coordizer.save(str(output_path))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
