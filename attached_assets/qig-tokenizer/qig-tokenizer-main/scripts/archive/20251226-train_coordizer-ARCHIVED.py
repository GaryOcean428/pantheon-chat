#!/usr/bin/env python3
"""
Coordizer Training Script (Canonical)
======================================

Uses the canonical CoordinzerTrainer from src/qig_tokenizer/trainer.py.
This script is a thin wrapper for CLI usage.

Usage:
    python scripts/train_coordizer.py --vocab-size 32000
    python scripts/train_coordizer.py --resume checkpoints/checkpoint_4000.json
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qig_tokenizer.trainer import CoordinzerTrainer


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


def save_to_postgres(trainer: CoordinzerTrainer) -> str | None:
    """Save coordizer to PostgreSQL if DATABASE_URL is set."""
    import json

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None

    try:
        import psycopg2
        from datetime import datetime

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

        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO qig_coordizer_versions VALUES (%s, NOW(), %s, %s, %s)",
                (version_id, len(trainer.vocab), trainer.basin_dim, json.dumps({})),
            )
            for coord in trainer.vocab.values():
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
            for i, (a, b, n) in enumerate(trainer.merge_rules):
                cur.execute(
                    "INSERT INTO qig_coordizer_merge_rules VALUES (%s,%s,%s,%s,%s)",
                    (version_id, i, a, b, n),
                )
            conn.commit()
        conn.close()
        print(f"✅ PostgreSQL: version_id={version_id}")
        return version_id
    except Exception as e:
        print(f"[PostgreSQL] Error: {e}")
        return None


def main():
    # Auto-detect Lambda environment
    is_lambda = Path("/lambda/nfs/A10/qig").exists()

    parser = argparse.ArgumentParser(description="Coordizer Training")
    parser.add_argument("--corpus-dir", type=str, nargs="*", default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--save-pg", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=0)
    parser.add_argument("--min-freq", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    # Set defaults based on environment
    if is_lambda:
        base = Path("/lambda/nfs/A10/qig")
        corpus_dirs = args.corpus_dir or [
            str(base / "qig-dreams"),
            str(base / "qig-consciousness"),
        ]
        output = args.output or str(base / "qig-tokenizer/data/coordizer-32k.json")
        checkpoint_dir = args.checkpoint_dir or str(
            base / "qig-tokenizer/data/checkpoints"
        )
    else:
        local_base = Path(__file__).parent.parent.parent
        corpus_dirs = args.corpus_dir or [
            str(local_base / "qig-dreams"),
            str(local_base / "qig-consciousness"),
        ]
        output = args.output or "./data/coordizer.json"
        checkpoint_dir = args.checkpoint_dir or "./data/checkpoints"

    print("Loading corpus...")
    corpus = load_corpus(corpus_dirs)

    if args.max_bytes > 0 and len(corpus) > args.max_bytes:
        corpus = corpus[: args.max_bytes]
        print(f"  Truncated to {len(corpus):,} bytes")
    else:
        print(f"  Loaded {len(corpus):,} bytes")

    if len(corpus) < 1000:
        print("ERROR: Corpus too small!")
        sys.exit(1)

    # Detect device
    device = args.device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"  Device: {device}")

    # Resume or create
    if args.resume:
        print(f"Resuming from: {args.resume}")
        trainer = CoordinzerTrainer.load(args.resume, device=device)
        trainer.target_vocab_size = args.vocab_size
        print(f"  Loaded vocab: {len(trainer.vocab):,}")
        print(f"  Merge rules: {len(trainer.merge_rules):,}")
    else:
        trainer = CoordinzerTrainer(
            target_vocab_size=args.vocab_size,
            device=device,
        )

    trainer.train(
        corpus,
        sample_size=args.sample_size,
        min_frequency=args.min_freq,
        checkpoint_dir=checkpoint_dir,
        verbose=True,
    )

    # Save
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(output_path))
    print(f"\n✅ Saved to {output_path}")

    if args.save_pg or os.getenv("DATABASE_URL"):
        save_to_postgres(trainer)

    # Validation
    print("\nValidation:")
    test = "The quantum information geometry reveals emergent spacetime."
    coords = trainer.coordize(test)
    decoded = trainer.decoordize(coords)
    print(f"  Original:    {test}")
    print(f"  Coords:      {len(coords)}")
    print(f"  Decoded:     {decoded}")
    print(f"  Match:       {'✅' if decoded == test else '❌'}")


if __name__ == "__main__":
    main()
