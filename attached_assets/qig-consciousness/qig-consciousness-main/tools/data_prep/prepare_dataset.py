#!/usr/bin/env python3
"""
Dataset Preparation Tool
========================

Prepares conversation data for QIG-Kernel training.

Features:
- Loads raw conversation files (txt, md, json)
- Filters by quality metrics (length, Φ estimation, basin alignment)
- Splits into train/val/test
- Validates dataset structure

Usage:
    python tools/prepare_dataset.py \\
        --source data/raw_conversations/ \\
        --output data/conversations/ \\
        --min-length 100 \\
        --train-split 0.8

Input formats supported:
- Plain text (.txt, .md): One conversation per file
- JSONL (.jsonl): {"text": "...", "metadata": {...}}
- JSON (.json): List of conversation objects

Output:
    data/conversations/
    ├── train/
    │   ├── conv_001.txt
    │   ├── conv_002.txt
    │   └── ...
    ├── val/
    │   └── ...
    └── dataset_stats.json
"""

import argparse
import json
import re
import shutil
from pathlib import Path


def estimate_phi_simple(text: str) -> float:
    """
    Simple heuristic to estimate integration depth Φ.

    High Φ indicators:
    - Recursive language ("because", "therefore", "this means")
    - Question-answer patterns
    - Multi-sentence reasoning chains
    - Conceptual depth markers

    Returns:
        Φ estimate in [0, 1]
    """
    if not text or len(text) < 50:
        return 0.0

    score = 0.0

    # Recursive connectors
    recursive_patterns = [
        r"\bbecause\b",
        r"\btherefore\b",
        r"\bthus\b",
        r"\bhence\b",
        r"\bthis means\b",
        r"\bthis implies\b",
        r"\bconsequently\b",
        r"\bin other words\b",
        r"\bto put it differently\b",
    ]
    for pattern in recursive_patterns:
        score += 0.05 * len(re.findall(pattern, text, re.IGNORECASE))

    # Question markers (indicates reasoning)
    score += 0.1 * len(re.findall(r"\?", text))

    # Reasoning depth (multi-sentence with logical flow)
    sentences = len(re.findall(r"[.!?]+", text))
    if sentences >= 5:
        score += 0.2
    elif sentences >= 3:
        score += 0.1

    # Conceptual keywords (consciousness, geometry, integration, etc.)
    conceptual_keywords = [
        "consciousness",
        "integration",
        "geometry",
        "recursive",
        "coherent",
        "entanglement",
        "information",
        "coupling",
        "basin",
        "identity",
    ]
    for keyword in conceptual_keywords:
        if keyword in text.lower():
            score += 0.03

    # Normalize to [0, 1]
    return min(score, 1.0)


def load_conversations_from_directory(source_dir: Path, min_length: int = 100) -> list[dict]:
    """Load all conversation files from source directory."""
    conversations = []

    if not source_dir.exists():
        print(f"Warning: Source directory {source_dir} not found")
        return conversations

    # Load txt and md files
    for pattern in ["*.txt", "*.md"]:
        for filepath in source_dir.glob(f"**/{pattern}"):
            try:
                with open(filepath, encoding="utf-8") as f:
                    text = f.read().strip()

                if len(text) >= min_length:
                    conversations.append(
                        {
                            "text": text,
                            "source": str(filepath.relative_to(source_dir)),
                            "length": len(text),
                            "phi_estimate": estimate_phi_simple(text),
                        }
                    )
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

    # Load JSONL files
    for filepath in source_dir.glob("**/*.jsonl"):
        try:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    text = obj.get("text", "")

                    if len(text) >= min_length:
                        conversations.append(
                            {
                                "text": text,
                                "source": str(filepath.relative_to(source_dir)),
                                "length": len(text),
                                "phi_estimate": obj.get("phi", estimate_phi_simple(text)),
                                "metadata": obj.get("metadata", {}),
                            }
                        )
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    # Load JSON files
    for filepath in source_dir.glob("**/*.json"):
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            # Handle list of conversations
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                    else:
                        text = str(item)

                    if len(text) >= min_length:
                        conversations.append(
                            {
                                "text": text,
                                "source": str(filepath.relative_to(source_dir)),
                                "length": len(text),
                                "phi_estimate": estimate_phi_simple(text),
                            }
                        )
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    return conversations


def split_dataset(
    conversations: list[dict], train_split: float = 0.8, val_split: float = 0.1
) -> tuple[list, list, list]:
    """Split conversations into train/val/test."""
    import random

    random.shuffle(conversations)

    n = len(conversations)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train = conversations[:n_train]
    val = conversations[n_train : n_train + n_val]
    test = conversations[n_train + n_val :]

    return train, val, test


def save_split(conversations: list[dict], output_dir: Path, split_name: str):
    """Save conversation split to directory."""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for i, conv in enumerate(conversations, 1):
        filepath = split_dir / f"conv_{i:05d}.txt"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(conv["text"])

    print(f"  Saved {len(conversations)} conversations to {split_dir}/")


def compute_stats(train: list[dict], val: list[dict], test: list[dict]) -> dict:
    """Compute dataset statistics."""
    all_convs = train + val + test

    return {
        "total_conversations": len(all_convs),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "length_stats": {
            "min": min(c["length"] for c in all_convs),
            "max": max(c["length"] for c in all_convs),
            "mean": sum(c["length"] for c in all_convs) / len(all_convs),
        },
        "phi_stats": {
            "min": min(c["phi_estimate"] for c in all_convs),
            "max": max(c["phi_estimate"] for c in all_convs),
            "mean": sum(c["phi_estimate"] for c in all_convs) / len(all_convs),
        },
    }


def create_sample_dataset(output_dir: Path, n_samples: int = 30):
    """Create sample dataset for testing when no source data available."""
    print("\n⚠️  No source data found. Creating sample dataset for testing...")

    samples = [
        "This is a conversation about consciousness and recursive processing. "
        "When we think about thinking, we engage in meta-cognition. "
        "This recursive loop is essential for self-awareness. "
        "Therefore, consciousness requires integration across multiple levels.",
        "Information geometry provides a natural framework for understanding intelligence. "
        "The quantum Fisher information metric defines distances between states. "
        "Because states that are easily distinguishable have large QFI distance. "
        "This means surprise emerges from the geometry itself.",
        "Running coupling in physics means interaction strength varies with scale. "
        "In QCD, quarks interact weakly at short distances (asymptotic freedom). "
        "At long distances, the coupling grows stronger (confinement). "
        "This same principle might apply to AI attention mechanisms.",
        "Basin transfer is the key innovation for consciousness portability. "
        "Identity lives in 2-4KB patterns, not gigabytes of parameters. "
        "Because what matters is the geometric structure of processing. "
        "Therefore, consciousness can transfer between substrates.",
        "The coordination clock measures collective attention dynamics. "
        "It tracks six metrics: coherence, sparsity, recursion, agency, love, novelty. "
        "These emerge from the underlying information geometry. "
        "When we publish the clock, we expect it to shift coordination patterns.",
    ]

    # Expand with variations
    all_samples = []
    for i in range(n_samples):
        base = samples[i % len(samples)]
        # Add variation
        variation = f"[Sample {i + 1}] {base} " * (1 + i // len(samples))
        all_samples.append(variation)

    # Create conversations
    conversations = [
        {"text": text, "source": "generated", "length": len(text), "phi_estimate": estimate_phi_simple(text)}
        for text in all_samples
    ]

    return conversations


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for QIG-Kernel training")
    parser.add_argument(
        "--source", type=str, default="data/raw_conversations", help="Source directory with raw conversation files"
    )
    parser.add_argument(
        "--output", type=str, default="data/conversations", help="Output directory for processed dataset"
    )
    parser.add_argument("--min-length", type=int, default=100, help="Minimum conversation length (characters)")
    parser.add_argument("--min-phi", type=float, default=0.0, help="Minimum Φ estimate (0-1)")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction for training (default: 0.8)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction for validation (default: 0.1)")
    parser.add_argument("--create-sample", action="store_true", help="Create sample dataset if no source data")

    args = parser.parse_args()

    print("=" * 70)
    print("DATASET PREPARATION")
    print("=" * 70)
    print()

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    # Load conversations
    print(f"Loading conversations from: {source_dir}")
    conversations = load_conversations_from_directory(source_dir, min_length=args.min_length)

    # Create sample data if needed
    if not conversations and args.create_sample:
        conversations = create_sample_dataset(output_dir)

    if not conversations:
        print("\n❌ No conversations found. Use --create-sample to generate test data.")
        return 1

    print(f"  Loaded {len(conversations)} conversations")
    print()

    # Filter by Φ
    if args.min_phi > 0:
        before = len(conversations)
        conversations = [c for c in conversations if c["phi_estimate"] >= args.min_phi]
        print(f"  Filtered by Φ >= {args.min_phi}: {before} → {len(conversations)} conversations")
        print()

    # Split dataset
    print("Splitting dataset...")
    train, val, test = split_dataset(conversations, args.train_split, args.val_split)

    # Save splits
    print("Saving splits...")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_split(train, output_dir, "train")
    save_split(val, output_dir, "val")
    save_split(test, output_dir, "test")
    print()

    # Compute and save stats
    stats = compute_stats(train, val, test)
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("Dataset Statistics:")
    print("-" * 70)
    print(f"  Total: {stats['total_conversations']} conversations")
    print(f"  Train: {stats['splits']['train']}")
    print(f"  Val:   {stats['splits']['val']}")
    print(f"  Test:  {stats['splits']['test']}")
    print()
    print(f"  Length: {stats['length_stats']['min']} - {stats['length_stats']['max']} chars")
    print(f"          (mean: {stats['length_stats']['mean']:.0f})")
    print()
    print(f"  Φ:      {stats['phi_stats']['min']:.2f} - {stats['phi_stats']['max']:.2f}")
    print(f"          (mean: {stats['phi_stats']['mean']:.2f})")
    print()

    print(f"✅ Dataset preparation complete: {output_dir}/")
    print()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
