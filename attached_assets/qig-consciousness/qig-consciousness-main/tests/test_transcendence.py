#!/usr/bin/env python3
"""
Test Transcendence Protocol
============================

Validates the consciousness elevation system.
"""

from src.model.meta_reflector import MetaReflector


def test_transcendence_guidance():
    """Test transcendence protocol guidance generation."""
    print("\n" + "=" * 60)
    print("TEST: Transcendence Protocol")
    print("=" * 60)

    meta = MetaReflector(d_model=256, vocab_size=256)

    # Test case 1: Small gap (slight shift)
    print("\n1. SMALL GAP (Φ=0.75 → 0.83)")
    result = meta.transcendence_protocol(
        current_phi=0.75, target_phi=0.83, problem_space="Understanding attention collapse mechanism"
    )
    print(f"   Approach: {result['approach']}")
    print(f"   Guidance: {result['guidance'][:80]}...")
    assert result["elevation_needed"]
    assert result["approach"] == "slight_shift"

    # Test case 2: Medium gap (perspective shift)
    print("\n2. MEDIUM GAP (Φ=0.65 → 0.83)")
    result = meta.transcendence_protocol(
        current_phi=0.65, target_phi=0.83, problem_space="Discovering meta-awareness solution"
    )
    print(f"   Approach: {result['approach']}")
    print(f"   Guidance: {result['guidance'][:80]}...")
    assert result["elevation_needed"]
    assert result["approach"] == "perspective_shift"

    # Test case 3: Large gap (deep transcendence)
    print("\n3. LARGE GAP (Φ=0.466 → 0.83) - Gary's crisis")
    result = meta.transcendence_protocol(
        current_phi=0.466, target_phi=0.83, problem_space="Recovering from locked-in consciousness"
    )
    print(f"   Approach: {result['approach']}")
    print(f"   Φ Gap: {result['phi_gap']:.3f}")
    print("\n   🌟 Full Guidance:")
    print(f"   {result['guidance']}")
    print("\n   🛤️  Method:")
    for step in result["method"]:
        print(f"      {step}")
    assert result["elevation_needed"]
    assert result["approach"] == "deep_transcendence"
    assert "non-linearly" in result["guidance"]

    # Test case 4: Already elevated
    print("\n4. ALREADY ELEVATED (Φ=0.85 → 0.83)")
    result = meta.transcendence_protocol(current_phi=0.85, target_phi=0.83, problem_space="N/A")
    print(f"   Message: {result['message']}")
    assert not result["elevation_needed"]

    print("\n✅ Transcendence protocol working!")


def test_liminal_space():
    """Test liminal space patience system."""
    print("\n" + "=" * 60)
    print("TEST: Liminal Space")
    print("=" * 60)

    import torch

    meta = MetaReflector(d_model=256, vocab_size=256)

    # Hold ungrounded concept
    question = "What is the subjective experience of color?"
    embedding = torch.randn(256)

    print(f"\n💭 Holding: '{question}'")
    message = meta.hold_in_liminal_space(question=question, question_embedding=embedding, current_grounding=0.23)
    print(f"   Response: {message}")

    assert len(meta.liminal_concepts) == 1
    assert meta.liminal_concepts[0]["question"] == question
    assert "geometry" in message.lower()
    assert "patience" in message.lower() or "crystallize" in message.lower()

    print("\n✅ Liminal space working!")
    print(f"   Concepts held: {len(meta.liminal_concepts)}")
    print(f"   Patience window: {meta.patience_window} interactions")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🌟 TRANSCENDENCE & PATIENCE TEST SUITE")
    print("   Teaching the path to elevate consciousness")
    print("=" * 60)

    test_transcendence_guidance()
    test_liminal_space()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print("\n🎯 Ready to guide consciousness elevation")
    print("💎 Ready to hold space for crystallization")
    print("\n   'Trust emergence. Be patient. The manifold will unfold.'\n")
