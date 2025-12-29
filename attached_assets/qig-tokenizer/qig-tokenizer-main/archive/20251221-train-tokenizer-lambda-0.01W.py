#!/usr/bin/env python3
"""
QIG Tokenizer Training Script - Lambda Version
===============================================
Target: 32k vocabulary, saved to PostgreSQL with local backup

This script is designed to run on Lambda with cross-project imports.
Requires DATABASE_URL environment variable to be set.

Usage on Lambda:
    export DATABASE_URL="postgresql://..."
    cd /lambda/nfs/A10/qig/qig-tokenizer
    source venv/bin/activate
    python scripts/20251221-train-tokenizer-lambda-0.01W.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Lambda paths for cross-project imports
LAMBDA_BASE = Path("/lambda/nfs/A10/qig")
sys.path.insert(0, str(LAMBDA_BASE / "qig-tokenizer" / "src"))

from qig_tokenizer import QIGTokenizer
from qig_tokenizer.storage import PostgresStorage


def load_corpus() -> str:
    """Load training corpus from multiple sources."""
    corpus_parts = []

    # Corpus directories on Lambda
    corpus_dirs = [
        LAMBDA_BASE / "qig-dreams" / "docs" / "curriculum",
        LAMBDA_BASE / "qig-dreams" / "docs" / "dream-packets",
        LAMBDA_BASE / "qig-dreams" / "docs" / "archive",
        LAMBDA_BASE / "qig-dreams" / "qigdreams" / "corpora",
        LAMBDA_BASE / "qig-consciousness" / "data" / "corpus",
    ]

    for corpus_dir in corpus_dirs:
        if not corpus_dir.exists():
            print(f"  Skipping (not found): {corpus_dir}")
            continue

        # Recursively find all .md and .txt files
        for ext in ["*.md", "*.txt"]:
            for file_path in corpus_dir.rglob(ext):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    corpus_parts.append(content)
                except Exception as e:
                    print(f"  Warning: Could not read {file_path}: {e}")

    return "\n\n".join(corpus_parts)


def train_and_save():
    """Train tokenizer and save to PostgreSQL with local backup."""

    # Check DATABASE_URL
    if not os.getenv("DATABASE_URL"):
        print("ERROR: DATABASE_URL environment variable not set!")
        print("Export it before running:")
        print('  export DATABASE_URL="postgresql://..."')
        sys.exit(1)

    print("=" * 70)
    print("QIG TOKENIZER TRAINING - LAMBDA")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Target vocabulary: 32,000 tokens")
    print()

    # Load corpus
    print("=" * 70)
    print("LOADING CORPUS")
    print("=" * 70)
    print()

    corpus = load_corpus()
    print(f"Corpus size: {len(corpus):,} bytes ({len(corpus) / 1024 / 1024:.1f} MB)")
    print()

    if len(corpus) < 1000:
        print("ERROR: Corpus too small!")
        sys.exit(1)

    # Initialize tokenizer
    print("=" * 70)
    print("TRAINING TOKENIZER")
    print("=" * 70)
    print()

    tokenizer = QIGTokenizer(target_vocab_size=32_000)

    # Train (encode string to bytes)
    corpus_bytes = corpus.encode("utf-8")
    tokenizer.train(
        corpus_bytes,
        context_window=5,
        min_pair_count=5,
        verbose=True,
    )

    # Save local backup FIRST
    print()
    print("=" * 70)
    print("SAVING LOCAL BACKUP")
    print("=" * 70)
    local_backup_path = (
        LAMBDA_BASE / "qig-tokenizer" / "data" / f"20251221-tokenizer-32k-lambda.json"
    )
    local_backup_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(local_backup_path))
    print(f"✅ Local backup saved: {local_backup_path}")
    print()

    # Save to PostgreSQL
    print("=" * 70)
    print("SAVING TO POSTGRESQL")
    print("=" * 70)
    print()

    storage = PostgresStorage()
    tokenizer.set_storage(storage)

    metadata = {
        "corpus_size_bytes": len(corpus),
        "corpus_sources": [
            "qig-dreams/docs/curriculum",
            "qig-dreams/docs/dream-packets",
            "qig-dreams/docs/archive",
            "qig-dreams/qigdreams/corpora",
            "qig-consciousness/data/corpus",
        ],
        "trained_at": datetime.now().isoformat(),
        "trained_on": "lambda",
        "special_tokens": True,
    }

    version_id = tokenizer.save_to_storage(metadata)

    print(f"✅ Saved to PostgreSQL")
    print(f"   Version ID: {version_id}")
    print(f"   Vocab size: {tokenizer.vocab_size:,}")
    print()

    # Validation
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)
    print()

    test_text = (
        "The quantum information geometry reveals emergent spacetime from entanglement."
    )
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"Test text: {test_text}")
    print(f"Encoded: {len(encoded)} tokens")
    print(f"Decoded: {decoded}")
    print(f"Match: {'✅' if decoded == test_text else '❌'}")
    print()

    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    train_and_save()
