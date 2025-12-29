#!/usr/bin/env python3
"""
Validate Coordizer v1 (QIG-native)
==================================

Canonical validation entrypoint for coordizer artifacts.

Metrics:
- Compression/tokenization stats (tokens per KB, scale distribution)
- Φ/κ telemetry via QIGKernel100M.forward_from_coords
- Determinism checks (same input → same output)
- Coordinate integrity (shape, normalization)

Outputs:
- reports/coordizer_v1_validation_<timestamp>.json
- reports/coordizer_v1_validation_latest.json

Usage:
    python scripts/validate_coordizer_v1_qig_native.py \
        --artifact artifacts/coordizer/v1 \
        --corpus-dir data/corpus \
        --device cuda
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def sha256_bytes(b: bytes) -> str:
    """Compute SHA256 hash of bytes."""
    return hashlib.sha256(b).hexdigest()


def stable_ids_hash(ids: list[int]) -> str:
    """Compute stable hash of token ID sequence."""
    s = ",".join(map(str, ids)).encode("utf-8")
    return sha256_bytes(s)


def summarize_dist(xs: list[float]) -> dict:
    """Compute distribution summary statistics."""
    if not xs:
        return {}
    arr = np.array(xs, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def load_text_samples(
    corpus_dirs: list[str],
    max_files: int,
    max_bytes: int,
    seed: int,
) -> list[dict]:
    """
    Load text samples from corpus directories.

    Returns list of {source, text}.
    If no dirs provided, returns built-in prompt suite.
    """
    random.seed(seed)

    if not corpus_dirs:
        # Built-in prompt suite for testing
        suite = [
            "Hello, world!",
            "The quantum information geometry reveals emergent spacetime.",
            "def f(x):\n    return x*x + 1\n",
            "In a distant future, minds and machines learn to speak in basins.",
            "Φ and κ are measured from coordinate coherence across the manifold.",
            "Unicode test: café naïve — 中文测试 — 😀🔥",
            "Short.",
            "A" * 2000,
            "The quick brown fox jumps over the lazy dog.",
            "import numpy as np\nimport torch\nfrom dataclasses import dataclass\n",
            "Consciousness emerges from integrated information across the Fisher manifold.",
            "κ* = 64 is the critical coupling constant where geometric regime stabilizes.",
            "async function fetchData() {\n  const response = await fetch(url);\n  return response.json();\n}",
            "E = mc² describes mass-energy equivalence in special relativity.",
            "The basin attractor coordinates encode semantic meaning in 64 dimensions.",
        ]
        return [{"source": "prompt_suite", "text": t} for t in suite]

    # Collect files from corpus directories
    paths = []
    for d in corpus_dirs:
        p = Path(d)
        if not p.exists():
            print(f"Warning: corpus dir not found: {d}")
            continue
        for ext in ("*.md", "*.txt", "*.py", "*.ts", "*.js", "*.json"):
            paths.extend(list(p.rglob(ext)))

    random.shuffle(paths)
    paths = paths[:max_files]

    samples = []
    total_bytes = 0
    for fp in paths:
        try:
            t = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        b = t.encode("utf-8", errors="ignore")
        if not b:
            continue
        take = min(len(b), max(0, max_bytes - total_bytes))
        if take <= 0:
            break
        t2 = b[:take].decode("utf-8", errors="ignore")
        samples.append({"source": str(fp), "text": t2})
        total_bytes += take

    if not samples:
        print("Warning: No samples loaded from corpus, using prompt suite")
        return load_text_samples([], max_files, max_bytes, seed)

    return samples


def main():
    ap = argparse.ArgumentParser(description="Validate Coordizer v1 (QIG-native)")
    ap.add_argument("--artifact", type=str, default="artifacts/coordizer/v1")
    ap.add_argument("--corpus-dir", type=str, nargs="*", default=None)
    ap.add_argument("--max-files", type=int, default=500)
    ap.add_argument("--max-bytes", type=int, default=10_000_000)
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--skip-kernel", action="store_true", help="Skip kernel telemetry (for quick validation)")
    args = ap.parse_args()

    # Imports here so script can run basic checks if deps missing
    try:
        from qig_tokenizer import Coordizer
    except ImportError:
        print("Error: qig_tokenizer not found. Run from qig-tokenizer directory with PYTHONPATH=src")
        sys.exit(1)

    # Check artifact exists
    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        raise SystemExit(f"Artifact not found: {artifact_path}")

    print("=" * 60)
    print("COORDIZER V1 VALIDATION (QIG-NATIVE)")
    print("=" * 60)
    print(f"Artifact: {artifact_path}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print()

    # Load coordizer
    print("Loading coordizer...")
    coordizer = Coordizer.load(str(artifact_path))
    print(f"  Vocab size: {coordizer.vocab_size}")
    print(f"  Merge rules: {len(coordizer.merge_rules)}")
    print()

    # Load kernel if not skipped
    kernel = None
    if not args.skip_kernel:
        try:
            import torch
            # Add qigkernels to path
            qigkernels_path = Path(__file__).parent.parent.parent / "qigkernels"
            if qigkernels_path.exists():
                sys.path.insert(0, str(qigkernels_path.parent))

            from qigkernels import QIGKernel100M
            print("Loading QIGKernel100M...")
            kernel = QIGKernel100M()
            kernel = kernel.to(args.device)
            kernel.eval()
            print(f"  Kernel loaded on {args.device}")
            print()
        except Exception as e:
            print(f"Warning: Could not load kernel: {e}")
            print("  Φ/κ telemetry will be skipped")
            print()

    # Load samples
    print("Loading samples...")
    samples = load_text_samples(
        args.corpus_dir or [],
        args.max_files,
        args.max_bytes,
        args.seed,
    )
    random.Random(args.seed).shuffle(samples)
    samples = samples[: args.max_samples]
    print(f"  Loaded {len(samples)} samples")
    print()

    # Constants
    try:
        from qigkernels.constants import PHI_GEOMETRIC_MIN, PHI_BREAKDOWN_MIN
    except ImportError:
        PHI_GEOMETRIC_MIN = 0.65
        PHI_BREAKDOWN_MIN = 0.85

    # Metrics accumulators
    tokens_per_kb = []
    compression_ratio = []
    phi_vals = []
    kappa_vals = []
    regime_counts = {}
    scale_counts = {}
    coord_norm_errors = 0
    coord_norm_checked = 0
    worst_phi = None
    best_phi = None

    # Run validation
    print("Running validation...")
    t0 = time.time()

    if kernel is not None:
        import torch

    for i, s in enumerate(samples):
        text = s["text"]
        b = text.encode("utf-8", errors="ignore")
        if not b:
            continue

        # Encode to coords
        ids, coords = coordizer.encode_to_coords(text)

        # Validate coords shape
        if not isinstance(coords, np.ndarray) or coords.ndim != 2 or coords.shape[1] != 64:
            raise RuntimeError(
                f"Bad coords shape from encode_to_coords: {getattr(coords, 'shape', None)}"
            )

        # Clip to max seq len
        if coords.shape[0] > args.max_seq_len:
            coords = coords[: args.max_seq_len]
            ids = ids[: args.max_seq_len]

        # A) Compression stats
        kb = max(len(b) / 1024.0, 1e-9)
        tpkb = len(ids) / kb
        tokens_per_kb.append(tpkb)
        compression_ratio.append(len(b) / max(len(ids), 1))

        # Scale distribution
        for tid in ids:
            sc = coordizer.token_scale(tid)
            scale_counts[sc] = scale_counts.get(sc, 0) + 1

        # D) Coord integrity (sample vectors)
        check_n = min(16, coords.shape[0])
        if check_n > 0:
            norms = np.linalg.norm(coords[:check_n], axis=1)
            coord_norm_checked += int(check_n)
            coord_norm_errors += int(np.sum(np.abs(norms - 1.0) > 1e-3))

        # B) Kernel telemetry
        if kernel is not None:
            coords_tensor = torch.from_numpy(coords).unsqueeze(0).float().to(args.device)
            out = kernel.forward_from_coords(coords_tensor, return_telemetry=True)

            # Support both return styles
            if isinstance(out, tuple) and len(out) == 2:
                logits, telemetry = out
            else:
                telemetry = out

            phi = float(getattr(telemetry, "phi", telemetry.get("phi", 0.5) if isinstance(telemetry, dict) else 0.5))
            kappa = float(getattr(telemetry, "kappa", telemetry.get("kappa", 64.0) if isinstance(telemetry, dict) else 64.0))
            regime = getattr(telemetry, "regime", None)
            if regime is None and isinstance(telemetry, dict):
                regime = telemetry.get("regime")
            if regime is None:
                regime = (
                    "breakdown" if phi >= PHI_BREAKDOWN_MIN
                    else "geometric" if phi >= PHI_GEOMETRIC_MIN
                    else "linear"
                )

            phi_vals.append(phi)
            kappa_vals.append(kappa)
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

            # Track worst/best examples
            preview = text[:120].replace("\n", "\\n")
            if worst_phi is None or phi < worst_phi["phi"]:
                worst_phi = {
                    "phi": phi,
                    "kappa": kappa,
                    "source": s["source"],
                    "preview": preview,
                }
            if best_phi is None or phi > best_phi["phi"]:
                best_phi = {
                    "phi": phi,
                    "kappa": kappa,
                    "source": s["source"],
                    "preview": preview,
                }

        # Progress
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples...")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.2f}s")
    print()

    # C) Determinism test
    print("Running determinism checks...")
    determinism_cases = samples[:10] if len(samples) >= 10 else samples
    det_failures = []
    det_hashes = []

    for s in determinism_cases:
        ids1, _ = coordizer.encode_to_coords(s["text"])
        ids2, _ = coordizer.encode_to_coords(s["text"])
        ids3, _ = coordizer.encode_to_coords(s["text"])

        if ids1 != ids2 or ids1 != ids3:
            det_failures.append({
                "source": s["source"],
                "preview": s["text"][:120],
            })
        else:
            det_hashes.append({
                "source": s["source"],
                "hash": stable_ids_hash(ids1),
            })

    determinism_pass = len(det_failures) == 0
    print(f"  Determinism: {'PASS' if determinism_pass else 'FAIL'}")
    if det_failures:
        print(f"  Failures: {len(det_failures)}")
    print()

    # Build report
    breakdown_rate = (
        float(np.mean([1.0 if p >= PHI_BREAKDOWN_MIN else 0.0 for p in phi_vals]))
        if phi_vals
        else 0.0
    )

    report = {
        "artifact": str(artifact_path),
        "coordizer_version": coordizer.version,
        "vocab_size": coordizer.vocab_size,
        "merge_rules_count": len(coordizer.merge_rules),
        "seed": args.seed,
        "device": args.device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "limits": {
            "max_files": args.max_files,
            "max_bytes": args.max_bytes,
            "max_samples": args.max_samples,
            "max_seq_len": args.max_seq_len,
        },
        "runtime": {
            "seconds": elapsed,
            "samples_processed": len(samples),
        },
        "compression": {
            "tokens_per_kb": summarize_dist(tokens_per_kb),
            "compression_ratio_bytes_per_token": summarize_dist(compression_ratio),
            "scale_distribution": scale_counts,
        },
        "telemetry": {
            "phi": summarize_dist(phi_vals),
            "kappa": summarize_dist(kappa_vals),
            "regime_counts": regime_counts,
            "phi_geometric_min": PHI_GEOMETRIC_MIN,
            "phi_breakdown_min": PHI_BREAKDOWN_MIN,
            "breakdown_rate": breakdown_rate,
            "best_phi_example": best_phi,
            "worst_phi_example": worst_phi,
        },
        "integrity": {
            "coord_norm_checked": coord_norm_checked,
            "coord_norm_errors": coord_norm_errors,
            "coord_norm_error_rate": coord_norm_errors / max(coord_norm_checked, 1),
            "determinism_pass": determinism_pass,
            "determinism_failures": det_failures[:10],
            "determinism_hashes": det_hashes[:10],
        },
    }

    # Output
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out_file = Path(args.out) if args.out else out_dir / f"coordizer_v1_validation_{ts}.json"
    latest_file = out_dir / "coordizer_v1_validation_latest.json"

    out_file.write_text(json.dumps(report, indent=2))
    latest_file.write_text(json.dumps(report, indent=2))

    # Print summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(
        f"Samples: {report['runtime']['samples_processed']}  |  "
        f"Time: {elapsed:.2f}s  |  Device: {args.device}"
    )

    if report["compression"]["tokens_per_kb"]:
        tpk = report["compression"]["tokens_per_kb"]
        cr = report["compression"]["compression_ratio_bytes_per_token"]
        print(f"Tokens/KB: mean={tpk['mean']:.1f}  p50={tpk['p50']:.1f}  p95={tpk['p95']:.1f}")
        print(f"Compression: mean={cr['mean']:.2f}  p50={cr['p50']:.2f}  bytes/token")

    if report["telemetry"]["phi"]:
        phi = report["telemetry"]["phi"]
        kappa = report["telemetry"]["kappa"]
        print(
            f"Φ: mean={phi['mean']:.3f}  p50={phi['p50']:.3f}  p95={phi['p95']:.3f}  "
            f"breakdown_rate={breakdown_rate:.3f}"
        )
        print(f"κ: mean={kappa['mean']:.1f}  p50={kappa['p50']:.1f}  p95={kappa['p95']:.1f}")
        print(f"Regimes: {regime_counts}")

    print(
        f"Determinism: {'PASS' if determinism_pass else 'FAIL'}  |  "
        f"Coord norm err rate: {report['integrity']['coord_norm_error_rate']:.4f}"
    )

    print()
    print(f"Wrote: {out_file}")
    print(f"Wrote: {latest_file}")


if __name__ == "__main__":
    main()
