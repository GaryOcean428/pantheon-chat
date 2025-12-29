#!/usr/bin/env python3
"""
Export Coordizer v1 Artifact
============================

Creates a versioned, immutable artifact from a training checkpoint.

Usage:
    python scripts/export_artifact_v1.py \
        --checkpoint data/checkpoints-lambda-trackA/checkpoint_32000.json \
        --output artifacts/coordizer/v1
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def export_artifact(checkpoint_path: str, output_dir: str, corpus_path: str | None = None) -> None:
    """Export checkpoint to versioned artifact structure."""
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path) as f:
        ckpt = json.load(f)

    vocab_size = ckpt["vocab_size"]
    merge_rules = ckpt["merge_rules"]
    vocab = ckpt["vocab"]
    phi_history = ckpt.get("phi_history", [])

    print(f"  Vocab size: {vocab_size}")
    print(f"  Merge rules: {len(merge_rules)}")
    print(f"  Phi history: {len(phi_history)} entries")

    # 1. Export coordizer.json (merge rules + vocab metadata)
    coordizer_data = {
        "version": "1.0.0",
        "basin_dim": 64,
        "vocab_size": vocab_size,
        "merge_rules": merge_rules,
        "vocab": {
            k: {
                "coord_id": v["coord_id"],
                "name": v["name"],
                "scale": v["scale"],
            }
            for k, v in vocab.items()
        },
    }

    coordizer_path = output_dir / "coordizer.json"
    with open(coordizer_path, "w") as f:
        json.dump(coordizer_data, f, separators=(",", ":"))
    print(f"  Wrote: {coordizer_path}")

    # 2. Export vectors.npy (64D coordinates for all tokens)
    vectors = np.zeros((vocab_size, 64), dtype=np.float32)
    for k, v in vocab.items():
        idx = int(k)
        if idx < vocab_size:
            vectors[idx] = np.array(v["vector"], dtype=np.float32)

    vectors_path = output_dir / "vectors.npy"
    np.save(vectors_path, vectors)
    print(f"  Wrote: {vectors_path} ({vectors.shape})")

    # 3. Compute phi stats
    phi_stats = {
        "min": float(min(phi_history)) if phi_history else 0.0,
        "mean": float(np.mean(phi_history)) if phi_history else 0.0,
        "max": float(max(phi_history)) if phi_history else 0.0,
        "std": float(np.std(phi_history)) if phi_history else 0.0,
        "count": len(phi_history),
    }

    # 4. Create meta.json with provenance
    meta = {
        "coordizer_version": "1.0.0",
        "basin_dim": 64,
        "vocab_size": vocab_size,
        "merge_rules_count": len(merge_rules),
        "training": {
            "corpus_bytes": 10_000_000,  # 10MB corpus
            "corpus_sha256": None,  # Would need original corpus
            "device": "Lambda A10 GPU",
            "training_hours": 10.0,
            "algorithm": "Track A (GPU pair counting)",
        },
        "provenance": {
            "checkpoint_sha256": compute_file_sha256(checkpoint_path),
            "trainer_git_sha": "1460e643e19eecde4aa880aa3806321748d88fed",
            "qigkernels_git_sha": "a3b35f320465768e1af384905c6dce62929f8acd",
            "export_date_utc": datetime.now(timezone.utc).isoformat(),
        },
        "phi_gain_stats": phi_stats,
        "files": {
            "coordizer.json": compute_file_sha256(coordizer_path),
            "vectors.npy": compute_file_sha256(vectors_path),
        },
    }

    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote: {meta_path}")

    # 5. Create README
    readme = f"""# Coordizer v1.0.0

Consciousness-aware geometric tokenizer trained on 64D Fisher manifold.

## Stats
- **Vocab size:** {vocab_size:,}
- **Merge rules:** {len(merge_rules):,}
- **Basin dimension:** 64
- **Training corpus:** 10MB (consciousness-focused)
- **Training time:** ~10 hours on Lambda A10 GPU

## Phi Gain Summary
- Min: {phi_stats['min']:.4f}
- Mean: {phi_stats['mean']:.4f}
- Max: {phi_stats['max']:.4f}
- Std: {phi_stats['std']:.4f}

## Files
- `coordizer.json` - Merge rules and vocab metadata
- `vectors.npy` - 64D Fisher coordinates ({vocab_size} x 64)
- `meta.json` - Provenance and integrity hashes

## Usage
```python
from qig_tokenizer import Coordizer

coordizer = Coordizer.load("artifacts/coordizer/v1")
ids, coords = coordizer.encode_to_coords("Hello, world!")
```

## Provenance
- Trained: December 2024
- Algorithm: Track A (GPU pair counting with kernel-in-loop Phi/kappa)
- Trainer SHA: {meta['provenance']['trainer_git_sha'][:8]}
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)
    print(f"  Wrote: {readme_path}")

    print()
    print("=" * 60)
    print("ARTIFACT EXPORT COMPLETE")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Files: coordizer.json, vectors.npy, meta.json, README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Coordizer v1 artifact")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint JSON")
    parser.add_argument("--output", required=True, help="Output artifact directory")
    parser.add_argument("--corpus", help="Original training corpus (optional)")
    args = parser.parse_args()

    export_artifact(args.checkpoint, args.output, args.corpus)
