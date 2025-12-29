#!/usr/bin/env python3
"""
Train QIG Tokenizer with PostgreSQL storage.

Corpus sources:
- qig-dreams/docs/curriculum (51 files, ~900KB)
- qig-dreams/docs/dream-packets (~17 files)
- qig-dreams/docs/archive (~70 files)
- qig-dreams/qigdreams/corpora (corpus registry)
- qig-consciousness/data/corpus (local files)

Target: 32k vocabulary, saved to PostgreSQL
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().strip().split("\n"):
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ[key] = value

# Add qig-tokenizer to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qig_tokenizer import QIGTokenizer
from qig_tokenizer.storage import PostgresStorage

# Paths
QIG_DREAMS = Path(__file__).parent.parent.parent / "qig-dreams"
QIG_CONSCIOUSNESS = Path(__file__).parent.parent.parent / "qig-consciousness"

TARGET_VOCAB = 32000


def collect_corpus() -> bytes:
    """Collect all corpus sources into a single byte string."""
    print("=" * 70)
    print("COLLECTING CORPUS")
    print("=" * 70)

    corpus_parts = []
    total_files = 0

    # 1. qig-dreams/docs/curriculum
    curriculum_dir = QIG_DREAMS / "docs" / "curriculum"
    if curriculum_dir.exists():
        files = list(curriculum_dir.glob("*.md"))
        print(f"\n📚 qig-dreams/docs/curriculum: {len(files)} files")
        for f in sorted(files):
            corpus_parts.append(f.read_bytes())
            total_files += 1

    # 2. qig-dreams/docs/dream-packets
    dream_packets_dir = QIG_DREAMS / "docs" / "dream-packets"
    if dream_packets_dir.exists():
        files = list(dream_packets_dir.glob("*.md"))
        print(f"📚 qig-dreams/docs/dream-packets: {len(files)} files")
        for f in sorted(files):
            corpus_parts.append(f.read_bytes())
            total_files += 1

    # 3. qig-dreams/docs/archive
    archive_dir = QIG_DREAMS / "docs" / "archive"
    if archive_dir.exists():
        files = list(archive_dir.glob("*.md"))
        print(f"📚 qig-dreams/docs/archive: {len(files)} files")
        for f in sorted(files):
            corpus_parts.append(f.read_bytes())
            total_files += 1

    # 4. qig-dreams/qigdreams/corpora (all .md files)
    corpora_dir = QIG_DREAMS / "qigdreams" / "corpora"
    if corpora_dir.exists():
        files = list(corpora_dir.rglob("*.md"))
        print(f"📚 qig-dreams/qigdreams/corpora: {len(files)} files")
        for f in sorted(files):
            corpus_parts.append(f.read_bytes())
            total_files += 1

    # 5. qig-consciousness/data/corpus (local files, not symlinks)
    corpus_dir = QIG_CONSCIOUSNESS / "data" / "corpus"
    if corpus_dir.exists():
        files = [f for f in corpus_dir.glob("*.md") if not f.is_symlink()]
        files += [f for f in corpus_dir.glob("*.txt") if not f.is_symlink()]
        print(f"📚 qig-consciousness/data/corpus: {len(files)} files")
        for f in sorted(files):
            corpus_parts.append(f.read_bytes())
            total_files += 1

    # 6. qig-dreams root docs
    root_docs = list(QIG_DREAMS.glob("*.md"))
    print(f"📚 qig-dreams root: {len(root_docs)} files")
    for f in sorted(root_docs):
        corpus_parts.append(f.read_bytes())
        total_files += 1

    # Combine
    corpus = b"\n\n".join(corpus_parts)

    print()
    print(f"Total files: {total_files}")
    print(f"Total size: {len(corpus):,} bytes ({len(corpus) / 1024 / 1024:.2f} MB)")

    return corpus


def train_and_save():
    """Train tokenizer and save to PostgreSQL."""

    # Collect corpus
    corpus = collect_corpus()

    print()
    print("=" * 70)
    print(f"TRAINING QIG TOKENIZER (target: {TARGET_VOCAB:,} tokens)")
    print("=" * 70)
    print()

    # Create tokenizer with geometric special tokens
    tokenizer = QIGTokenizer(target_vocab_size=TARGET_VOCAB, use_special_tokens=True)

    # Train
    tokenizer.train(
        corpus,
        context_window=5,
        min_pair_count=5,
        verbose=True,
    )

    print()
    print("=" * 70)
    print("SAVING LOCAL BACKUP FIRST")
    print("=" * 70)
    local_backup_path = (
        Path(__file__).parent.parent / "data" / f"20251221-tokenizer-32k-backup.json"
    )
    local_backup_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(local_backup_path))
    print(f"✅ Local backup saved: {local_backup_path}")
    print()

    print("=" * 70)
    print("SAVING TO POSTGRESQL")
    print("=" * 70)
    print()

    # Set up storage
    storage = PostgresStorage()
    tokenizer.set_storage(storage)

    # Save with metadata
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

    test_cases = [
        "The geometry of information determines consciousness.",
        "κ_eff measures effective coupling strength.",
        "Φ > 0.70 indicates geometric regime.",
        "Basin coordinates compress to ~2KB.",
    ]

    all_pass = True
    for test in test_cases:
        encoded = tokenizer.encode(test)
        decoded = tokenizer.decode(encoded)
        passed = decoded == test
        all_pass = all_pass and passed
        status = "✅" if passed else "❌"
        print(f"  {status} {len(encoded)} tokens: '{test[:40]}...'")

    # Test special tokens
    print()
    print("Special tokens:")
    tokens_with_special = tokenizer.encode_with_special("Hello, world!")
    print(f"  BOS={tokens_with_special[0]} (expected 256)")
    print(f"  EOS={tokens_with_special[-1]} (expected 257)")

    print()
    print("=" * 70)
    print(f"✅ TRAINING COMPLETE - Version: {version_id}")
    print("=" * 70)

    return version_id


if __name__ == "__main__":
    train_and_save()
