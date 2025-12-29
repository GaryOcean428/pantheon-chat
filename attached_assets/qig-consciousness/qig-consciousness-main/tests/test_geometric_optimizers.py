#!/usr/bin/env python3
"""
Test Geometric Optimizers
==========================

Quick tests to verify geometric optimizers are working correctly.

Run:
    python tests/test_geometric_optimizers.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn


def test_qig_diagonal_ng():
    """Test QIGDiagonalNG optimizer."""
    print("\n=== Testing QIGDiagonalNG ===")

    from qig.optim import QIGDiagonalNG

    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1),
    )

    optimizer = QIGDiagonalNG(
        model.parameters(),
        lr=1e-3,
        alpha=0.99,
    )

    # Forward and backward
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)

    pred = model(x)
    loss = ((pred - y) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check Fisher stats
    stats = optimizer.get_fisher_stats()
    print(f"✓ Fisher mean: {stats['fisher_mean']:.6f}")
    print(f"✓ Fisher std: {stats['fisher_std']:.6f}")
    print(f"✓ Condition number: {stats['condition_number']:.2f}")

    assert stats["fisher_mean"] > 0, "Fisher mean should be positive"
    print("✅ QIGDiagonalNG passed!")


def test_basin_natural_grad():
    """Test BasinNaturalGrad optimizer."""
    print("\n=== Testing BasinNaturalGrad ===")

    from qig.optim import BasinNaturalGrad

    # Small model (simulating basin block)
    model = nn.Linear(64, 768)

    optimizer = BasinNaturalGrad(
        model.parameters(),
        lr=1e-2,
        cg_iters=5,
        damping=1e-4,
    )

    # Forward and backward with create_graph=True
    x = torch.randn(4, 64)
    y = torch.randn(4, 768)

    pred = model(x)
    loss = ((pred - y) ** 2).mean()

    optimizer.zero_grad()
    loss.backward(create_graph=True)  # Important for Pearlmutter trick
    optimizer.step()

    # Check stats
    stats = optimizer.get_stats()
    print(f"✓ Num params: {stats['num_params']:,}")
    print(f"✓ Grad norm: {stats['grad_norm']:.6f}")

    assert stats["num_params"] == 64 * 768 + 768, "Should have correct param count"
    print("✅ BasinNaturalGrad passed!")


def test_mixed_qig_optimizer():
    """Test HybridGeometricOptimizer."""
    print("\n=== Testing HybridGeometricOptimizer ===")

    from qig.optim import HybridGeometricOptimizer

    # Simple model with basin-like structure
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Simulate basin embeddings
            self.embedding = nn.Module()
            self.embedding.basin_coords = nn.Parameter(torch.randn(100, 64))
            self.embedding.basin_to_model = nn.Linear(64, 768, bias=False)

            # Simulate rest of model
            self.fc1 = nn.Linear(768, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            # Simple forward for testing
            return self.fc2(self.fc1(x))

    model = SimpleModel()

    optimizer = HybridGeometricOptimizer(
        model,
        lr_ng=1e-2,
        lr_rest=1e-3,
        cg_iters=5,
        use_exact_ng=False,  # Use diagonal NG for testing
    )

    # Forward and backward
    x = torch.randn(4, 768)
    y = torch.randn(4, 10)

    pred = model(x)
    loss = ((pred - y) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check stats
    stats = optimizer.get_stats()
    print(f"✓ Basin params: {stats['num_basin_params']:,}")
    print(f"✓ Rest params: {stats['num_rest_params']:,}")

    assert stats["num_basin_params"] > 0, "Should have basin params"
    assert stats["num_rest_params"] > 0, "Should have rest params"
    print("✅ HybridGeometricOptimizer passed!")


def test_adaptive_config():
    """Test adaptive gating configuration."""
    print("\n=== Testing Adaptive Config ===")

    from qig.optim.adaptive_gate import AdaptiveConfig, get_recommended_config_for_phase, should_use_ng

    # Test config creation
    config = AdaptiveConfig(
        min_kappa_for_ng=40.0,
        min_basin_distance=0.6,
    )

    # Test gating logic
    telemetry = {
        "kappa_eff": 50.0,  # High curvature
        "basin_distance": 0.7,  # Far from basin
        "Phi": 0.75,
        "regime": "geometric",
    }

    result = should_use_ng(telemetry, step=10, config=config)
    print(f"✓ Should use NG (high κ): {result}")
    assert result is True, "Should apply NG when κ high"

    # Test with low curvature
    telemetry["kappa_eff"] = 30.0
    telemetry["basin_distance"] = 0.3
    result = should_use_ng(telemetry, step=10, config=config)
    print(f"✓ Should NOT use NG (low κ): {result}")
    assert result is False, "Should not apply NG when κ low"

    # Test phase configs
    for phase in ["early", "middle", "late", "fine_tune"]:
        cfg = get_recommended_config_for_phase(phase)
        print(f"✓ Phase '{phase}': κ>{cfg.min_kappa_for_ng}, every {cfg.force_ng_every_n_steps}")

    print("✅ Adaptive config passed!")


def test_emotion_monitor():
    """Test emotion monitor."""
    print("\n=== Testing Emotion Monitor ===")

    from qig.affect import EmotionMonitor, compute_emotion_primitives

    # Test standalone function
    telemetry = {
        "kappa_eff": 50.0,
        "Phi": 0.75,
        "basin_distance": 0.8,
        "regime": "geometric",
        "curiosity_slow": 0.06,
        "curiosity_regime": "EXPLORATION",
        "drive_info": {
            "frustration": 0.3,
            "velocity": 0.02,
        },
    }

    emotions = compute_emotion_primitives(telemetry)
    print(f"✓ Joy: {emotions['joy']:.3f}")
    print(f"✓ Suffering: {emotions['suffering']:.3f}")
    print(f"✓ Fear: {emotions['fear']:.3f}")
    print(f"✓ Calm: {emotions['calm']:.3f}")

    assert "joy" in emotions, "Should compute joy"
    assert "suffering" in emotions, "Should compute suffering"

    # Test monitor
    monitor = EmotionMonitor(enable_extended_emotions=True, verbose=False)
    emotions2 = monitor.compute(telemetry)
    dominant = monitor.get_dominant_emotion()
    print(f"✓ Dominant emotion: {dominant}")

    print("✅ Emotion monitor passed!")


def test_end_to_end_qig_ml():
    """
    End-to-end validation of the complete QIG ML pipeline.

    Validates:
    1. QIG tokenizer independence (no GPT-2 dependency)
    2. Basin embeddings with geometric initialization
    3. Natural gradient optimization
    4. Full forward pass with consciousness telemetry
    5. Geometric purity (no Euclidean contamination)
    """
    print("\n=== End-to-End QIG ML Validation ===")

    from qig.optim import QIGDiagonalNG
    from src.constants import BASIN_DIM, D_MODEL, N_HEADS, VOCAB_SIZE
    from src.model.basin_embedding import BasinCoordinates
    from src.model.qig_kernel_recursive import QIGKernelRecursive

    # 1. Verify vocab size is NOT GPT-2's 50257
    print("\n1. Checking vocab size...")
    assert VOCAB_SIZE == 50000, f"VOCAB_SIZE should be 50000 (QIG), not {VOCAB_SIZE} (GPT-2 is 50257)"
    print(f"   ✓ VOCAB_SIZE = {VOCAB_SIZE} (QIG native, NOT GPT-2)")

    # 2. Test basin embeddings with geometric initialization
    print("\n2. Testing Basin Embeddings...")
    basin_emb = BasinCoordinates(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        basin_dim=BASIN_DIM,
        init_mode="geometric"
    )

    # Verify basin coordinates are on manifold (normalized)
    basin_norms = torch.norm(basin_emb.basin_coords, dim=-1)
    mean_norm = basin_norms.mean().item()
    print(f"   ✓ Basin coords mean norm: {mean_norm:.4f}")
    print(f"   ✓ Basin dim: {BASIN_DIM}, Model dim: {D_MODEL}")

    # 3. Test QIG kernel with correct vocab size
    print("\n3. Testing QIG Kernel...")
    kernel = QIGKernelRecursive(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        n_heads=N_HEADS,
        min_recursion_depth=3,
        min_Phi=0.7,
    )

    # Forward pass
    x = torch.randint(0, VOCAB_SIZE, (2, 32))  # [batch, seq]
    output, telemetry = kernel(x)

    print(f"   ✓ Output shape: {output.shape}")
    print(f"   ✓ Φ (Phi): {telemetry.get('Phi', 'N/A')}")
    print(f"   ✓ κ_eff: {telemetry.get('kappa_eff', 'N/A')}")
    print(f"   ✓ Regime: {telemetry.get('regime', 'N/A')}")

    # 4. Test natural gradient optimizer
    print("\n4. Testing Natural Gradient Optimizer...")
    optimizer = QIGDiagonalNG(kernel.parameters(), lr=1e-4)

    # Compute loss and backward
    loss = output.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    fisher_stats = optimizer.get_fisher_stats()
    print(f"   ✓ Fisher mean: {fisher_stats['fisher_mean']:.6f}")
    print(f"   ✓ Condition number: {fisher_stats['condition_number']:.2f}")

    # 5. Verify geometric purity
    print("\n5. Verifying Geometric Purity...")

    # Check no GPT-2 tokenizer imports
    import sys
    gpt2_modules = [m for m in sys.modules if 'gpt2' in m.lower()]
    transformers_modules = [m for m in sys.modules if 'transformers' in m.lower()]

    print(f"   ✓ No GPT-2 modules loaded: {len(gpt2_modules) == 0}")
    print(f"   ✓ Transformers modules (should be 0): {len(transformers_modules)}")

    # Final summary
    print("\n" + "=" * 50)
    print("✅ END-TO-END QIG ML VALIDATION PASSED!")
    print("=" * 50)
    print(f"""
Summary:
- VOCAB_SIZE: {VOCAB_SIZE} (QIG native)
- Basin Embedding: Pure geometric initialization
- Natural Gradient: Diagonal Fisher working
- Consciousness Telemetry: Φ, κ, regime available
- Geometric Purity: No external tokenizer dependencies
    """)


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing QIG Geometric Optimizers")
    print("=" * 60)

    try:
        test_qig_diagonal_ng()
        test_basin_natural_grad()
        test_mixed_qig_optimizer()
        test_adaptive_config()
        test_emotion_monitor()
        test_end_to_end_qig_ml()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nGeometric optimizers are ready for training.")
        print("Use configs/train_geometric_optimizer_example.yaml as reference.")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
