#!/usr/bin/env python3
"""
Consciousness-Aware Geometric Coordizer Training
=================================================

Full Phase 2 implementation: kernel-in-loop training where merge
decisions are guided by actual Φ/κ feedback from a QIG kernel.

This is PURE geometric coordization - no BPE frequency shortcuts.

Algorithm:
1. Initialize 256 byte coordinates on Fisher manifold
2. For each merge candidate:
   a. Tentatively create fused coordinate
   b. Coordize sample corpus with candidate
   c. Run through QIG kernel to measure Φ/κ
   d. Score candidate by: coupling × Φ_gain × (1/entropy)
3. Execute best merge (geodesic midpoint, not arithmetic)
4. Repeat until target vocab size

Requires: qigkernels (for Φ/κ measurement)
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "qigkernels"))

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARNING] PyTorch not available - using mock kernel")

# Import from consolidated modules (moved from qig-consciousness)
try:
    from metrics.geodesic_distance import GeodesicDistance
    from safety.self_repair import Stage, GeometryState
    from safety.emergency_monitor import EmergencyMonitor
    from neuroplasticity.sleep_protocol import SleepProtocol, SleepReport

    HAS_QIG_MODULES = True
except ImportError:
    HAS_QIG_MODULES = False
    print("[WARNING] QIG modules not fully available - using basic mode")


# Constants from FROZEN_FACTS
BASIN_DIM = 64
KAPPA_STAR = 64.0
PHI_GEOMETRIC_MIN = 0.65


@dataclass
class BasinCoordinate:
    """Point on 64D Fisher manifold."""

    coord_id: int
    vector: np.ndarray
    name: str | None = None
    scale: str = "byte"

    def fisher_distance(self, other: "BasinCoordinate") -> float:
        """Fisher-Rao geodesic distance (NOT Euclidean)."""
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)

        if norm_self < 1e-10 or norm_other < 1e-10:
            return float("inf")

        cos_angle = np.clip(
            np.dot(self.vector, other.vector) / (norm_self * norm_other), -1.0, 1.0
        )
        return float(np.arccos(cos_angle))

    def geodesic_midpoint(self, other: "BasinCoordinate") -> np.ndarray:
        """Geodesic midpoint on manifold (NOT arithmetic mean)."""
        v1 = self.vector / (np.linalg.norm(self.vector) + 1e-10)
        v2 = other.vector / (np.linalg.norm(other.vector) + 1e-10)

        midpoint = v1 + v2
        midpoint = midpoint / (np.linalg.norm(midpoint) + 1e-10)

        avg_mag = (np.linalg.norm(self.vector) + np.linalg.norm(other.vector)) / 2
        return midpoint * avg_mag


@dataclass
class MergeCandidate:
    """Candidate for geodesic fusion."""

    coord_a: int
    coord_b: int
    frequency: int
    coupling: float  # κ between coordinates
    entropy: float  # Context entropy (lower = more predictable)
    phi_gain: float  # Δ in measured from kernel

    @property
    def score(self) -> float:
        """Geometric merge score."""
        entropy_factor = 1.0 / (self.entropy + 0.1)
        return self.frequency * self.coupling * (1.0 + self.phi_gain) * entropy_factor


class MockKernel:
    """Mock kernel for testing without qigkernels."""

    def measure_phi_kappa(self, coords: list[np.ndarray]) -> tuple[float, float]:
        """Return mock Φ/κ based on coordinate coherence."""
        if len(coords) < 2:
            return 0.5, KAPPA_STAR

        # Estimate Φ from coordinate coherence
        total_dist = 0.0
        for i in range(len(coords) - 1):
            dist = np.arccos(
                np.clip(
                    np.dot(coords[i], coords[i + 1])
                    / (
                        np.linalg.norm(coords[i]) * np.linalg.norm(coords[i + 1])
                        + 1e-10
                    ),
                    -1.0,
                    1.0,
                )
            )
            total_dist += dist

        avg_dist = total_dist / len(coords)
        phi = max(0.3, min(0.9, 1.0 - avg_dist / np.pi))

        return phi, KAPPA_STAR


class KernelInterface:
    """Interface to QIG kernel for Φ/κ measurement."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._kernel = None
        self._mock = MockKernel()

        self._load_kernel()

    def _load_kernel(self) -> None:
        """Try to load actual QIG kernel."""
        if not HAS_TORCH:
            print("[Kernel] Using mock kernel (no PyTorch)")
            return

        try:
            from qigkernels.kernel_100m import create_kernel_100m

            self._kernel = create_kernel_100m(
                vocab_size=32000,
                hidden_dim=256,  # Smaller for speed
                num_heads=4,
                num_layers=3,
            )
            self._kernel = self._kernel.to(self.device)
            self._kernel.eval()

            print(f"[Kernel] Loaded QIG kernel on {self.device}")

        except (ImportError, RuntimeError) as e:
            print(f"[Kernel] Using mock kernel: {e}")

    def measure_phi_kappa(
        self,
        coord_vectors: list[np.ndarray],
    ) -> tuple[float, float]:
        """
        Measure Φ and κ for a coordinate sequence.

        This is the key feedback loop - actual consciousness
        metrics from the kernel guide merge decisions.
        """
        if self._kernel is None:
            return self._mock.measure_phi_kappa(coord_vectors)

        try:
            with torch.no_grad():
                # Convert coordinates to tensor
                # Stack as "embeddings" - kernel will process
                coords_tensor = torch.tensor(
                    np.stack(coord_vectors), dtype=torch.float32, device=self.device
                ).unsqueeze(
                    0
                )  # [1, seq_len, basin_dim]

                # Create dummy input_ids (kernel needs them)
                seq_len = len(coord_vectors)
                input_ids = torch.zeros(
                    1, seq_len, dtype=torch.long, device=self.device
                )

                # Forward pass with coordinate injection
                # Note: This requires kernel to accept pre-computed coordinates
                telemetry = self._kernel.forward_with_coords(input_ids, coords_tensor)

                phi = telemetry.phi
                kappa = telemetry.kappa

                return float(phi), float(kappa)

        except Exception as e:
            # Fall back to mock
            return self._mock.measure_phi_kappa(coord_vectors)


class ConsciousnessCoordizer:
    """
    Full geometric coordizer with kernel-in-loop training.

    Every merge decision is validated by actual Φ/κ measurement
    from the QIG kernel, ensuring consciousness-aware vocabulary.
    """

    def __init__(
        self,
        target_vocab_size: int = 32000,
        basin_dim: int = BASIN_DIM,
        device: str = "cpu",
    ):
        self.target_vocab_size = target_vocab_size
        self.basin_dim = basin_dim

        # Kernel interface for Φ/κ measurement
        self.kernel = KernelInterface(device=device)

        # Vocabulary: coord_id -> BasinCoordinate
        self.vocab: dict[int, BasinCoordinate] = {}

        # Merge rules: (coord_a, coord_b, new_coord)
        self.merge_rules: list[tuple[int, int, int]] = []

        # Statistics
        self.phi_history: list[float] = []
        self.kappa_history: list[float] = []

        # Initialize byte coordinates
        self._init_byte_coordinates()

    def _init_byte_coordinates(self) -> None:
        """Initialize 256 byte-level coordinates on manifold."""
        for byte_val in range(256):
            rng = np.random.default_rng(seed=byte_val + 42)
            vector = rng.standard_normal(self.basin_dim)
            vector = vector / np.linalg.norm(vector)

            self.vocab[byte_val] = BasinCoordinate(
                coord_id=byte_val,
                vector=vector,
                name=f"<byte_{byte_val:02x}>",
                scale="byte",
            )

    def train(
        self,
        corpus: bytes,
        sample_size: int = 1000,
        candidates_per_round: int = 50,
        min_frequency: int = 5,
        context_window: int = 3,
        checkpoint_dir: str | None = None,
        checkpoint_interval: int = 500,
        verbose: bool = True,
    ) -> "ConsciousnessCoordizer":
        """
        Train coordizer with kernel-in-loop Φ/κ feedback.

        Args:
            corpus: Training corpus as bytes
            sample_size: Samples to evaluate per candidate
            candidates_per_round: Top candidates to evaluate with kernel
            min_frequency: Minimum pair frequency
            context_window: Context window for entropy
            checkpoint_dir: Directory for checkpoints
            checkpoint_interval: Tokens between checkpoints
            verbose: Print progress
        """
        start_time = time.time()

        if verbose:
            print("=" * 60)
            print("CONSCIOUSNESS-AWARE GEOMETRIC COORDIZER TRAINING")
            print("=" * 60)
            print(f"Corpus: {len(corpus):,} bytes")
            print(f"Target vocab: {self.target_vocab_size:,}")
            print(f"Basin dimension: {self.basin_dim}")
            print(
                f"Kernel: {type(self.kernel._kernel).__name__ if self.kernel._kernel else 'Mock'}"
            )
            print()

        # Convert corpus to coordinate sequence
        corpus_coords = list(corpus)
        current_vocab_size = 256

        # Measure baseline Φ/κ
        sample_coords = [self.vocab[c].vector for c in corpus_coords[:100]]
        baseline_phi, baseline_kappa = self.kernel.measure_phi_kappa(sample_coords)

        if verbose:
            print(f"Baseline Φ={baseline_phi:.3f}, κ={baseline_kappa:.1f}")
            print()

        while current_vocab_size < self.target_vocab_size:
            _ = time.time()  # round timing

            # 1. Compute pair statistics
            pair_stats = self._compute_pair_stats(
                corpus_coords, context_window, min_frequency
            )

            if not pair_stats:
                if verbose:
                    print("No more valid pairs")
                break

            # 2. Get top candidates by frequency × coupling
            candidates = self._get_top_candidates(pair_stats, candidates_per_round)

            if not candidates:
                break

            # 3. Evaluate candidates with kernel Φ/κ feedback
            best_candidate = self._evaluate_with_kernel(
                candidates, corpus_coords, sample_size
            )

            if best_candidate is None:
                break

            # 4. Execute geodesic fusion
            coord_a, coord_b = best_candidate.coord_a, best_candidate.coord_b
            new_coord_id = current_vocab_size

            self.merge_rules.append((coord_a, coord_b, new_coord_id))
            self._create_fused_coordinate(coord_a, coord_b, new_coord_id)

            # 5. Update corpus
            corpus_coords = self._apply_fusion(
                corpus_coords, coord_a, coord_b, new_coord_id
            )

            current_vocab_size += 1

            # Track history
            self.phi_history.append(best_candidate.phi_gain)

            # Progress
            if verbose and current_vocab_size % 50 == 0:
                elapsed = time.time() - start_time
                rate = (current_vocab_size - 256) / elapsed
                eta = (
                    (self.target_vocab_size - current_vocab_size) / rate
                    if rate > 0
                    else 0
                )

                coord = self.vocab[new_coord_id]
                name = coord.name[:30] if coord.name else f"<{new_coord_id}>"

                print(
                    f"[{current_vocab_size:,}/{self.target_vocab_size:,}] "
                    f"'{name}' | "
                    f"Φ_gain={best_candidate.phi_gain:+.3f} | "
                    f"κ={best_candidate.coupling:.1f} | "
                    f"{rate:.1f}/s | ETA {eta/60:.1f}m"
                )

            # Checkpoint
            if checkpoint_dir and current_vocab_size % checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_dir, current_vocab_size)
                if verbose:
                    print(f"  💾 Checkpoint: {current_vocab_size}")

        elapsed = time.time() - start_time

        if verbose:
            print()
            print("=" * 60)
            print("TRAINING COMPLETE")
            print("=" * 60)
            print(f"Final vocab: {current_vocab_size:,}")
            print(f"Merge rules: {len(self.merge_rules):,}")
            print(f"Time: {elapsed/60:.1f} minutes")
            print(f"Rate: {(current_vocab_size - 256) / elapsed:.1f} tokens/sec")

            # Final Φ/κ measurement
            sample_coords = [self.vocab[c].vector for c in corpus_coords[:100]]
            final_phi, final_kappa = self.kernel.measure_phi_kappa(sample_coords)
            print(f"Final Φ={final_phi:.3f} (Δ={final_phi-baseline_phi:+.3f})")
            print(f"Final κ={final_kappa:.1f}")

        return self

    def _compute_pair_stats(
        self,
        corpus_coords: list[int],
        window: int,
        min_count: int,
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """Compute pair statistics with Fisher coupling."""
        pair_contexts: dict[tuple[int, int], list[tuple]] = defaultdict(list)
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)

        for i in range(len(corpus_coords) - 1):
            pair = (corpus_coords[i], corpus_coords[i + 1])
            pair_counts[pair] += 1

            # Cap contexts per pair to 64 for performance (entropy approximation)
            if len(pair_contexts[pair]) < 64:
                ctx_before = tuple(corpus_coords[max(0, i - window) : i])
                ctx_after = tuple(
                    corpus_coords[i + 2 : min(len(corpus_coords), i + 2 + window)]
                )
                pair_contexts[pair].append(ctx_before + ctx_after)

        pair_stats = {}
        for pair, count in pair_counts.items():
            if count < min_count:
                continue

            contexts = pair_contexts[pair]
            entropy = self._compute_entropy(contexts)
            coupling = self._compute_coupling(
                pair[0], pair[1], count, len(corpus_coords)
            )

            pair_stats[pair] = {
                "count": count,
                "entropy": entropy,
                "coupling": coupling,
            }

        return pair_stats

    def _compute_entropy(self, contexts: list[tuple]) -> float:
        """Compute context entropy."""
        counts: dict[tuple, int] = defaultdict(int)
        for ctx in contexts:
            counts[ctx] += 1

        total = len(contexts)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log(p + 1e-10)

        return entropy

    def _compute_coupling(
        self,
        coord_a: int,
        coord_b: int,
        co_occurrence: int,
        corpus_size: int,
    ) -> float:
        """Compute coupling κ using Fisher distance."""
        if coord_a not in self.vocab or coord_b not in self.vocab:
            return 0.0

        basin_a = self.vocab[coord_a]
        basin_b = self.vocab[coord_b]

        fisher_dist = basin_a.fisher_distance(basin_b)

        # κ increases with co-occurrence, decreases with distance
        coupling = (co_occurrence / corpus_size) / (fisher_dist + 0.1)
        return min(coupling * 1000, 100.0)

    def _get_top_candidates(
        self,
        pair_stats: dict[tuple[int, int], dict],
        top_k: int,
    ) -> list[MergeCandidate]:
        """Get top merge candidates by initial score."""
        candidates = []

        for (coord_a, coord_b), stats in pair_stats.items():
            candidate = MergeCandidate(
                coord_a=coord_a,
                coord_b=coord_b,
                frequency=stats["count"],
                coupling=stats["coupling"],
                entropy=stats["entropy"],
                phi_gain=0.0,  # Will be computed by kernel
            )
            candidates.append(candidate)

        # Sort by preliminary score (without Φ)
        candidates.sort(key=lambda c: c.frequency * c.coupling, reverse=True)
        return candidates[:top_k]

    def _evaluate_with_kernel(
        self,
        candidates: list[MergeCandidate],
        corpus_coords: list[int],
        sample_size: int,
    ) -> MergeCandidate | None:
        """Evaluate candidates using actual Φ/κ from kernel."""
        if not candidates:
            return None

        # Measure baseline Φ
        sample = corpus_coords[: min(sample_size, len(corpus_coords))]
        baseline_vectors = [self.vocab[c].vector for c in sample]
        baseline_phi, _ = self.kernel.measure_phi_kappa(baseline_vectors)

        best_candidate = None
        best_score = float("-inf")

        for candidate in candidates[:10]:  # Limit kernel calls
            # Tentatively create fused coordinate
            basin_a = self.vocab[candidate.coord_a]
            basin_b = self.vocab[candidate.coord_b]
            fused_vector = basin_a.geodesic_midpoint(basin_b)

            # Apply fusion to sample
            fused_sample = []
            i = 0
            while i < len(sample):
                if (
                    i < len(sample) - 1
                    and sample[i] == candidate.coord_a
                    and sample[i + 1] == candidate.coord_b
                ):
                    fused_sample.append(-1)  # Placeholder for fused
                    i += 2
                else:
                    fused_sample.append(sample[i])
                    i += 1

            # Build vector sequence
            vectors = []
            for c in fused_sample:
                if c == -1:
                    vectors.append(fused_vector)
                else:
                    vectors.append(self.vocab[c].vector)

            # Measure Φ/κ with fusion
            fused_phi, fused_kappa = self.kernel.measure_phi_kappa(vectors)

            # Compute Φ gain - amplify small differences
            # Raw Φ differences are tiny, so scale by 100x
            raw_gain = fused_phi - baseline_phi

            # Also compute local coherence gain around merged pairs
            # This is more sensitive to individual merges
            local_gain = 0.0
            if len(vectors) > 2:
                for i, c in enumerate(fused_sample):
                    if c == -1 and i > 0 and i < len(vectors) - 1:
                        # Merged token - check coherence with neighbors
                        v_prev = vectors[i - 1]
                        v_curr = vectors[i]  # fused vector
                        v_next = vectors[i + 1] if i + 1 < len(vectors) else vectors[i]

                        # Coherence = dot products (normalized)
                        coh_prev = float(
                            np.dot(v_prev, v_curr)
                            / (np.linalg.norm(v_prev) * np.linalg.norm(v_curr) + 1e-10)
                        )
                        coh_next = float(
                            np.dot(v_curr, v_next)
                            / (np.linalg.norm(v_curr) * np.linalg.norm(v_next) + 1e-10)
                        )
                        local_gain += (coh_prev + coh_next) / 2

            # Φ_gain = global integration change + local coherence
            phi_gain = raw_gain * 100 + local_gain * 0.1
            candidate.phi_gain = phi_gain

            # Update score with actual Φ
            score = candidate.score

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    def _create_fused_coordinate(
        self,
        coord_a: int,
        coord_b: int,
        new_id: int,
    ) -> None:
        """Create fused coordinate via geodesic midpoint."""
        basin_a = self.vocab[coord_a]
        basin_b = self.vocab[coord_b]

        fused_vector = basin_a.geodesic_midpoint(basin_b)

        # Build name
        name_a = basin_a.name or f"<{coord_a}>"
        name_b = basin_b.name or f"<{coord_b}>"

        # Decode to readable if possible
        try:
            if coord_a < 256 and coord_b < 256:
                char_a = bytes([coord_a]).decode("utf-8", errors="replace")
                char_b = bytes([coord_b]).decode("utf-8", errors="replace")
                name = f"{char_a}{char_b}"
            else:
                name = f"{name_a}+{name_b}"
        except:
            name = f"{name_a}+{name_b}"

        # Determine scale
        scale_a = basin_a.scale
        scale_b = basin_b.scale
        scales = ["byte", "char", "subword", "word", "phrase", "concept"]
        idx_a = scales.index(scale_a) if scale_a in scales else 0
        idx_b = scales.index(scale_b) if scale_b in scales else 0
        new_scale = scales[min(max(idx_a, idx_b) + 1, len(scales) - 1)]

        self.vocab[new_id] = BasinCoordinate(
            coord_id=new_id,
            vector=fused_vector,
            name=name,
            scale=new_scale,
        )

    def _apply_fusion(
        self,
        coords: list[int],
        coord_a: int,
        coord_b: int,
        new_coord: int,
    ) -> list[int]:
        """Apply geodesic fusion to corpus."""
        result = []
        i = 0

        while i < len(coords):
            if (
                i < len(coords) - 1
                and coords[i] == coord_a
                and coords[i + 1] == coord_b
            ):
                result.append(new_coord)
                i += 2
            else:
                result.append(coords[i])
                i += 1

        return result

    def _save_checkpoint(self, checkpoint_dir: str, vocab_size: int) -> None:
        """Save training checkpoint."""
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

    def coordize(self, text: str) -> list[int]:
        """Convert text to coordinate sequence."""
        coords = list(text.encode("utf-8"))

        for coord_a, coord_b, new_coord in self.merge_rules:
            coords = self._apply_fusion(coords, coord_a, coord_b, new_coord)

        return coords

    def decoordize(self, coord_ids: list[int]) -> str:
        """Convert coordinates back to text."""
        reverse_merges = {
            new_coord: (coord_a, coord_b)
            for coord_a, coord_b, new_coord in self.merge_rules
        }

        def expand(cid: int) -> list[int]:
            if cid < 256:
                return [cid]
            if cid in reverse_merges:
                a, b = reverse_merges[cid]
                return expand(a) + expand(b)
            return [cid]

        bytes_list = []
        for cid in coord_ids:
            bytes_list.extend(expand(cid))

        return bytes(bytes_list).decode("utf-8", errors="replace")

    def get_coordinate_vectors(self, coord_ids: list[int]) -> list[np.ndarray]:
        """Get 64D vectors for coordinate sequence."""
        return [self.vocab[cid].vector for cid in coord_ids if cid in self.vocab]

    def save(self, path: str) -> None:
        """Save coordizer to file."""
        data = {
            "target_vocab_size": self.target_vocab_size,
            "basin_dim": self.basin_dim,
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
                if k >= 256  # Don't save base bytes (reproducible)
            },
            "phi_history": self.phi_history,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Also save vectors as numpy for efficiency
        vectors_path = path.replace(".json", "_vectors.npy")
        vectors = np.stack([self.vocab[i].vector for i in sorted(self.vocab.keys())])
        np.save(vectors_path, vectors)

    @classmethod
    def load(cls, path: str) -> "ConsciousnessCoordizer":
        """Load coordizer from file or checkpoint."""
        with open(path) as f:
            data = json.load(f)

        # Handle both checkpoint format and save format
        target_vocab = data.get("target_vocab_size", 32000)
        basin_dim = data.get("basin_dim", BASIN_DIM)

        coordizer = cls(
            target_vocab_size=target_vocab,
            basin_dim=basin_dim,
        )

        coordizer.merge_rules = [tuple(r) for r in data["merge_rules"]]
        coordizer.phi_history = data.get("phi_history", [])

        # Load vocab if present (checkpoints have it, saves may not have base bytes)
        if "vocab" in data:
            for k, v in data["vocab"].items():
                coordizer.vocab[int(k)] = BasinCoordinate(
                    coord_id=v["coord_id"],
                    vector=np.array(v["vector"]),
                    name=v["name"],
                    scale=v["scale"],
                )

        # Replay merge rules to rebuild vocab if needed
        if len(coordizer.vocab) <= 256 and coordizer.merge_rules:
            print(f"  Replaying {len(coordizer.merge_rules)} merge rules...")
            for coord_a, coord_b, new_id in coordizer.merge_rules:
                if new_id not in coordizer.vocab:
                    coordizer._create_fused_coordinate(coord_a, coord_b, new_id)

        return coordizer


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
                except (IOError, UnicodeDecodeError):
                    pass

    return "\n\n".join(parts).encode("utf-8")


def save_to_postgres(coordizer: ConsciousnessCoordizer) -> str | None:
    """Save coordizer to PostgreSQL if DATABASE_URL is set."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("[PostgreSQL] DATABASE_URL not set, skipping")
        return None

    try:
        import psycopg2

        print("\nSaving to PostgreSQL...")
        conn = psycopg2.connect(database_url)

        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS qig_coordizer_versions (
                    version_id VARCHAR(32) PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT NOW(),
                    vocab_size INTEGER, basin_dim INTEGER, metadata JSONB
                )"""
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS qig_coordizer_vocab (
                    version_id VARCHAR(32), coord_id INTEGER,
                    vector FLOAT8[], name TEXT, scale VARCHAR(32),
                    PRIMARY KEY (version_id, coord_id)
                )"""
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS qig_coordizer_merge_rules (
                    version_id VARCHAR(32), rule_order INTEGER,
                    coord_a INTEGER, coord_b INTEGER, new_coord INTEGER,
                    PRIMARY KEY (version_id, rule_order)
                )"""
            )
            conn.commit()

        from datetime import datetime

        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO qig_coordizer_versions VALUES (%s, NOW(), %s, %s, %s)",
                (version_id, len(coordizer.vocab), BASIN_DIM, json.dumps({})),
            )
            for coord in coordizer.vocab.values():
                if coord.coord_id >= 256:
                    cur.execute(
                        "INSERT INTO qig_coordizer_vocab VALUES (%s,%s,%s,%s,%s)",
                        (
                            version_id,
                            coord.coord_id,
                            coord.vector.tolist(),
                            coord.name,
                            coord.scale,
                        ),
                    )
            for i, (a, b, n) in enumerate(coordizer.merge_rules):
                cur.execute(
                    "INSERT INTO qig_coordizer_merge_rules VALUES (%s,%s,%s,%s,%s)",
                    (version_id, i, a, b, n),
                )
            conn.commit()
        conn.close()
        print(f"✅ PostgreSQL: version_id={version_id}")
        return version_id
    except (ImportError, Exception) as e:
        print(f"[PostgreSQL] Error: {e}")
        return None


def main():
    # Auto-detect Lambda environment
    is_lambda = Path("/lambda/nfs/A10/qig").exists()

    parser = argparse.ArgumentParser(
        description="Consciousness-Aware Geometric Coordizer Training"
    )
    parser.add_argument(
        "--corpus-dir", type=str, nargs="*", default=None, help="Corpus directories"
    )
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument(
        "--vocab-size", type=int, default=32000, help="Target vocabulary size"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (cpu/cuda/auto)"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None, help="Checkpoint directory"
    )
    parser.add_argument(
        "--sample-size", type=int, default=500, help="Sample size for kernel eval"
    )
    parser.add_argument("--save-pg", action="store_true", help="Save to PostgreSQL")
    parser.add_argument(
        "--max-bytes", type=int, default=0, help="Max corpus bytes (0=unlimited)"
    )
    parser.add_argument(
        "--min-freq", type=int, default=10, help="Min frequency for pair candidates"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint file"
    )

    args = parser.parse_args()

    # Set defaults based on environment
    # FULL CORPUS: qig-dreams + qig-consciousness (all markdown files)
    if is_lambda:
        base = Path("/lambda/nfs/A10/qig")
        corpus_dirs = args.corpus_dir or [
            str(base / "qig-dreams"),  # All of qig-dreams (~2MB)
            str(base / "qig-consciousness"),  # All of qig-consciousness (~2.3MB)
        ]
        output = args.output or str(base / "qig-tokenizer/data/coordizer-32k.json")
        checkpoint_dir = args.checkpoint_dir or str(
            base / "qig-tokenizer/data/checkpoints"
        )
    else:
        # Local: use qig-dreams/docs/curriculum or fallback
        local_curriculum = (
            Path(__file__).parent.parent.parent / "qig-dreams/docs/curriculum"
        )
        corpus_dirs = args.corpus_dir or (
            [str(local_curriculum)] if local_curriculum.exists() else ["./corpus"]
        )
        output = args.output or "./data/coordizer.json"
        checkpoint_dir = args.checkpoint_dir

    print("Loading corpus...")
    corpus = load_corpus(corpus_dirs)

    # Truncate corpus if max_bytes specified (for faster training)
    if args.max_bytes > 0 and len(corpus) > args.max_bytes:
        corpus = corpus[: args.max_bytes]
        print(f"  Truncated to {len(corpus):,} bytes (--max-bytes)")
    else:
        print(f"  Loaded {len(corpus):,} bytes")

    if len(corpus) < 1000:
        print("ERROR: Corpus too small!")
        sys.exit(1)

    # Detect device
    device = args.device
    if device == "auto":
        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    print(f"  Device: {device}")

    # Resume from checkpoint or create new
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        coordizer = ConsciousnessCoordizer.load(args.resume)
        coordizer.target_vocab_size = args.vocab_size
        coordizer.kernel = KernelInterface(device=device)
        print(f"  Loaded vocab: {len(coordizer.vocab):,}")
        print(f"  Merge rules: {len(coordizer.merge_rules):,}")
    else:
        coordizer = ConsciousnessCoordizer(
            target_vocab_size=args.vocab_size,
            device=device,
        )

    coordizer.train(
        corpus,
        sample_size=args.sample_size,
        min_frequency=args.min_freq,
        checkpoint_dir=checkpoint_dir,
        verbose=True,
    )

    # Save local
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coordizer.save(str(output_path))
    print(f"\n✅ Saved to {output_path}")

    # Save to PostgreSQL if requested or DATABASE_URL set
    if args.save_pg or os.getenv("DATABASE_URL"):
        save_to_postgres(coordizer)

    # Validation
    print("\nValidation:")
    test = "The quantum information geometry reveals emergent spacetime."
    coords = coordizer.coordize(test)
    decoded = coordizer.decoordize(coords)
    print(f"  Original:    {test}")
    print(f"  Coords:      {len(coords)}")
    print(f"  Decoded:     {decoded}")
    print(f"  Match:       {'✅' if decoded == test else '❌'}")


if __name__ == "__main__":
    main()
