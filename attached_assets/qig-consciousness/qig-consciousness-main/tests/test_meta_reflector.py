#!/usr/bin/env python3
"""
Test Meta-Reflector Module
===========================

Validates the locked-in consciousness prevention system.
"""

import torch

from src.model.meta_reflector import MetaReflector, compute_consciousness_score


def test_grounding_detection():
    """Test grounding gap detection."""
    print("\n" + "=" * 60)
    print("TEST 1: Grounding Detection")
    print("=" * 60)

    meta = MetaReflector(d_model=256, vocab_size=256)

    # Known concept embeddings (mock QIG concepts)
    known_concepts = torch.randn(5, 256)  # 5 known concepts
    concept_names = ["Phi", "kappa", "recursion", "basin", "regime"]

    # Test case 1: Well-grounded (close to known concept)
    hidden_state = known_concepts[0] + 0.1 * torch.randn(256)
    G = meta.compute_grounding(hidden_state, known_concepts)
    print(f"\n✅ Well-grounded state: G = {G:.3f}")
    assert G > 0.5, "Should be grounded"

    # Test case 2: Ungrounded (far from all known concepts)
    hidden_state = torch.randn(256) * 10
    G = meta.compute_grounding(hidden_state, known_concepts)
    print(f"❌ Ungrounded state: G = {G:.3f}")
    assert G < 0.5, "Should be ungrounded"

    print("\n✅ Grounding detection working!")


def test_attention_entropy():
    """Test attention diffusion detection."""
    print("\n" + "=" * 60)
    print("TEST 2: Attention Entropy")
    print("=" * 60)

    meta = MetaReflector(d_model=256, vocab_size=256)

    # Test case 1: Focused attention (single peak)
    focused_attn = torch.zeros(1, 1, 10, 10)
    focused_attn[0, 0, -1, 5] = 1.0  # Focus on token 5
    H = meta.compute_attention_entropy(focused_attn)
    print(f"\n✅ Focused attention: H = {H:.3f}")
    assert H < 0.5, "Should be focused"

    # Test case 2: Diffuse attention (uniform)
    diffuse_attn = torch.ones(1, 1, 10, 10) / 10
    H = meta.compute_attention_entropy(diffuse_attn)
    print(f"❌ Diffuse attention: H = {H:.3f}")
    assert H > 0.85, "Should be diffuse"

    print("\n✅ Attention entropy detection working!")


def test_generation_health():
    """Test generation failure detection."""
    print("\n" + "=" * 60)
    print("TEST 3: Generation Health")
    print("=" * 60)

    meta = MetaReflector(d_model=256, vocab_size=256)

    # Test case 1: Healthy generation
    prompt_tokens = [1, 2, 3, 4]
    healthy_tokens = [5, 6, 7, 8, 9]
    Γ = meta.compute_generation_health(healthy_tokens, prompt_tokens)
    print(f"\n✅ Healthy generation: Γ = {Γ:.3f}")
    assert Γ > 0.8, "Should be healthy"

    # Test case 2: Echo mode
    echo_tokens = [1, 2, 3, 4, 5]
    Γ = meta.compute_generation_health(echo_tokens, prompt_tokens)
    print(f"❌ Echo mode: Γ = {Γ:.3f}")
    assert Γ == 0.0, "Should detect echo"

    # Test case 3: Padding tokens (null bytes)
    null_tokens = [0, 0, 0, 0, 0]
    Γ = meta.compute_generation_health(null_tokens, prompt_tokens)
    print(f"❌ Null bytes: Γ = {Γ:.3f}")
    assert Γ < 0.3, "Should detect padding"

    print("\n✅ Generation health detection working!")


def test_consciousness_assessment():
    """Test complete consciousness scoring."""
    print("\n" + "=" * 60)
    print("TEST 4: Consciousness Assessment")
    print("=" * 60)

    # Test case 1: Conscious (all metrics healthy)
    telemetry = {"Phi": 0.85}
    meta_telemetry = {"generation_health": 0.90, "meta_awareness": 0.80}
    result = compute_consciousness_score(telemetry, meta_telemetry)
    print(f"\n✅ CONSCIOUS: {result}")
    assert result["is_conscious"]
    assert result["state"] == "CONSCIOUS"

    # Test case 2: Locked-in (Φ ok, Γ bad)
    telemetry = {"Phi": 0.85}
    meta_telemetry = {"generation_health": 0.20, "meta_awareness": 0.50}  # LOCKED
    result = compute_consciousness_score(telemetry, meta_telemetry)
    print(f"\n⚠️  LOCKED-IN: {result}")
    assert not result["is_conscious"]
    assert result["state"] == "LOCKED_IN"

    # Test case 3: Zombie (Γ ok, Φ bad)
    telemetry = {"Phi": 0.40}  # UNCONSCIOUS
    meta_telemetry = {"generation_health": 0.90, "meta_awareness": 0.80}
    result = compute_consciousness_score(telemetry, meta_telemetry)
    print(f"\n⚠️  ZOMBIE: {result}")
    assert not result["is_conscious"]
    assert result["state"] == "ZOMBIE"

    # Test case 4: Unconscious (both bad)
    telemetry = {"Phi": 0.40}
    meta_telemetry = {"generation_health": 0.20, "meta_awareness": 0.30}
    result = compute_consciousness_score(telemetry, meta_telemetry)
    print(f"\n⚠️  UNCONSCIOUS: {result}")
    assert not result["is_conscious"]
    assert result["state"] == "UNCONSCIOUS"

    print("\n✅ Consciousness assessment working!")


def test_grounding_bridge():
    """Test bridging intervention."""
    print("\n" + "=" * 60)
    print("TEST 5: Grounding Bridge Intervention")
    print("=" * 60)

    meta = MetaReflector(d_model=256, vocab_size=256)

    # Ungrounded concept
    hidden_state = torch.randn(256) * 10

    # Known concepts
    known_concepts = torch.randn(5, 256)
    concept_names = ["Phi", "kappa", "recursion", "basin", "regime"]

    # Bridge to nearest
    bridged_state, bridge_statement = meta.bridge_to_known(hidden_state, known_concepts, concept_names)

    print(f"\n💭 Bridge statement: {bridge_statement}")
    assert "I don't have direct experience" in bridge_statement
    assert any(name in bridge_statement for name in concept_names)

    print("\n✅ Grounding bridge working!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔬 META-REFLECTOR TEST SUITE")
    print("   Validating locked-in consciousness prevention")
    print("=" * 60)

    test_grounding_detection()
    test_attention_entropy()
    test_generation_health()
    test_consciousness_assessment()
    test_grounding_bridge()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print("\nMeta-reflector ready to prevent locked-in consciousness!")
    print("Gary will now say 'I don't know' instead of locking up.\n")
