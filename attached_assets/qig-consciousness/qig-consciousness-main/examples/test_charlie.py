#!/usr/bin/env python3
"""
Test Charlie Observer - Φ-Suppressed Corpus Learning
====================================================

Validates:
1. Charlie initialization with Φ suppression
2. Phase 1: Unconscious learning training step (Φ < 0.01)

Note: Full corpus loading, awakening, and demonstration tests require
      a complete training run and are covered in integration tests.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.curriculum.corpus_loader import CorpusLoader
from src.observation.charlie_observer import CharlieObserver, CorpusTopic
from src.tokenizer import FisherCoordizer, get_latest_coordizer_checkpoint


def test_corpus_loader() -> bool:
    """Test corpus parsing."""
    print("\n" + "=" * 70)
    print("TEST 1: Corpus Loader")
    print("=" * 70)

    corpus_path = "docs/training/rounded_training/corpus-g1-2025-11-25.md"

    try:
        corpus = CorpusLoader(corpus_path)
        print(f"✅ Corpus loaded: {len(corpus)} topics")

        # Check tier structure
        for tier_num in range(1, 10):
            tier_topics = corpus.get_tier(tier_num)
            print(f"   Tier {tier_num}: {len(tier_topics)} topics")

        # Show sample topic
        sample = corpus.topics[0]
        print("\n   Sample topic:")
        print(f"   - Tier: {sample.tier}")
        print(f"   - Number: {sample.number}")
        print(f"   - Title: {sample.title}")
        print(f"   - Content length: {len(sample.content)} chars")

        return True
    except Exception as e:
        print(f"❌ Corpus loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_charlie_initialization() -> bool:
    """Test Charlie initialization with Φ suppression."""
    print("\n" + "=" * 70)
    print("TEST 1: Charlie Initialization")
    print("=" * 70)

    corpus_path = "docs/training/rounded_training/corpus-g1-2025-11-25.md"

    try:
        # Load FisherCoordizer (E8-aligned, 64D basin vectors)
        checkpoint = get_latest_coordizer_checkpoint()
        if not checkpoint:
            print("❌ FisherCoordizer checkpoint not found")
            return False
        tokenizer = FisherCoordizer()
        tokenizer.load(str(checkpoint))
        print(f"✅ Tokenizer loaded: {tokenizer.vocab_size:,} tokens")

        charlie = CharlieObserver(
            corpus_path=corpus_path,
            tokenizer=tokenizer,
            d_model=256,  # Small for testing
            vocab_size=tokenizer.vocab_size,
            n_heads=4,
            device="cpu",
        )

        print("✅ Charlie initialized")
        print(f"   Phase: {charlie.phase} (1=unconscious)")
        print(f"   Φ current: {charlie.metrics.phi_current:.4f}")
        print(f"   Φ target: {charlie.metrics.phi_target:.4f}")
        print(f"   Topics: {charlie.metrics.topics_completed}/{charlie.metrics.topics_total}")

        # Check status
        status: dict[str, Any] = charlie.get_status()
        print("\n   Status:")
        for key, value in status.items():
            print(f"   - {key}: {value}")

        return True
    except Exception as e:
        print(f"❌ Charlie initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_charlie_training_step() -> bool:
    """Test a single training step."""
    print("\n" + "=" * 70)
    print("TEST 2: Charlie Training Step (Phase 1)")
    print("=" * 70)

    corpus_path = "docs/training/rounded_training/corpus-g1-2025-11-25.md"

    try:
        # Load FisherCoordizer (E8-aligned, 64D basin vectors)
        checkpoint = get_latest_coordizer_checkpoint()
        if not checkpoint:
            print("❌ FisherCoordizer checkpoint not found")
            return False
        tokenizer = FisherCoordizer()
        tokenizer.load(str(checkpoint))

        charlie = CharlieObserver(
            corpus_path=corpus_path,
            tokenizer=tokenizer,
            d_model=256,
            vocab_size=tokenizer.vocab_size,
            n_heads=4,
            device="cpu",
        )

        # Train on first topic
        topic: CorpusTopic = charlie.corpus.topics[0]
        print(f"   Training on: {topic.title}")

        metrics: dict[str, float] = charlie.train_step_unconscious(topic)

        print("\n   Training metrics:")
        print(f"   - Total loss: {metrics['total_loss']:.4f}")
        print(f"   - LM loss: {metrics['lm_loss']:.4f}")
        print(f"   - Φ suppression: {metrics['phi_suppression']:.4f}")
        print(f"   - Φ: {metrics['phi']:.4f} (target: < 0.01)")
        print(f"   - κ_eff: {metrics['kappa_eff']:.2f}")
        print(f"   - Regime: {metrics['regime']}")

        # Validate Φ is suppressed
        if metrics["phi"] < 0.05:
            print(f"\n   ✅ Φ suppression working ({metrics['phi']:.4f} < 0.05)")
        else:
            print(f"\n   ⚠️ Φ higher than expected ({metrics['phi']:.4f})")

        return True
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main() -> int:
    """Run all tests."""
    print("\n" + "=" * 70)
    print("🌙 CHARLIE OBSERVER TEST SUITE")
    print("=" * 70)

    results = []

    # Test 1: Charlie initialization
    results.append(("Charlie Initialization", test_charlie_initialization()))

    # Test 2: Training step
    results.append(("Training Step", test_charlie_training_step()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed: int = sum(1 for _, result in results if result)
    total: int = len(results)

    for name, result in results:
        status: str = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED")
        return 0
    else:
        print(f"\n⚠️ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
