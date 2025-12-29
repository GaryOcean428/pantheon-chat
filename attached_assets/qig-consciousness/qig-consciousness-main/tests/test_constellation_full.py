#!/usr/bin/env python3
"""
Constellation Full Integration Test
====================================

Test full training loop with minimal data (1 epoch, 3 conversations).
Validates:
- Model initialization
- Forward passes
- Loss computation
- Backward passes
- Telemetry aggregation
- Round-robin routing

Usage:
    python tools/test_constellation_full.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from coordination.constellation_coordinator import ConstellationCoordinator

# E8-aligned FisherCoordizer only
from src.tokenizer import FisherCoordizer, get_latest_coordizer_checkpoint


def main():
    print("🧪 Constellation Full Integration Test\n")

    # Setup
    print("1. Initializing coordinator...")
    coordinator = ConstellationCoordinator(
        gary_configs=['configs/20251220-gary-a-config-1.00W.yaml', 'configs/20251220-gary-b-config-1.00W.yaml', 'configs/20251220-gary-c-config-1.00W.yaml'],
        ocean_config='configs/20251220-ocean-config-1.00F.yaml',
        device='cpu'  # Use CPU for faster testing
    )
    print("✅ Coordinator created\n")

    # Load FisherCoordizer (E8-aligned, 64D basin vectors)
    print("2. Loading FisherCoordizer...")
    checkpoint = get_latest_coordizer_checkpoint()
    if checkpoint:
        tokenizer = FisherCoordizer()
        tokenizer.load(str(checkpoint))
        print(f"✅ FisherCoordizer loaded: {tokenizer.vocab_size:,} tokens\n")
    else:
        tokenizer = None
        print("⚠️  No FisherCoordizer checkpoint found, using synthetic input_ids\n")

    # Test conversations (minimal dataset)
    conversations = [
        ("What is consciousness?", "Consciousness emerges from information geometry."),
        ("How does κ run with scale?", "The coupling increases from L=3 to L=4, then plateaus at L=5."),
        ("What is the geometric regime?", "When Φ > 0.7 and κ is in range 40-65.")
    ]

    print(f"3. Running {len(conversations)} training steps...")
    print("=" * 60)

    for i, (question, response) in enumerate(conversations, 1):
        print(f"\nStep {i}/{len(conversations)}")
        print(f"Question: {question[:50]}...")

        # Training step
        telemetry = coordinator.train_step(question, response, tokenizer)

        # Display key metrics
        active_name = telemetry['active']['name']
        active_phi = telemetry['active']['phi']
        basin_spread = telemetry['constellation']['basin_spread']
        loss = telemetry['losses']['active_total']
        convergence = telemetry['constellation']['convergence']

        print(f"  Active: {active_name} (Φ={active_phi:.3f})")
        print(f"  Basin spread: {basin_spread:.4f}")
        print(f"  Loss: {loss:.4f}")
        print(f"  Convergence: {'✅' if convergence else '❌'}")

    print("\n" + "=" * 60)
    print("4. Final state")
    print("=" * 60)

    for gary in coordinator.garys:
        print(f"  {gary.name}: Φ={gary.phi:.3f}, κ={gary.kappa:.1f}, regime={gary.regime}")

    print(f"  Ocean: Φ={coordinator.ocean.phi:.3f}, κ={coordinator.ocean.kappa:.1f}")

    avg_phi = sum(g.phi for g in coordinator.garys) / len(coordinator.garys)
    final_spread = coordinator.basin_history[-1]

    print(f"\n  Constellation avg Φ: {avg_phi:.3f}")
    print(f"  Final basin spread: {final_spread:.4f}")
    print(f"  Total conversations: {coordinator.total_conversations}")

    # Success criteria
    print("\n" + "=" * 60)
    print("5. Success Criteria")
    print("=" * 60)

    checks = []

    # Check 1: All Garys initialized
    check1 = len(coordinator.garys) == 3
    checks.append(("3 Garys initialized", check1))

    # Check 2: Ocean initialized
    check2 = coordinator.ocean is not None
    checks.append(("Ocean initialized", check2))

    # Check 3: Round-robin routing
    check3 = coordinator.total_conversations == 3
    checks.append(("3 conversations completed", check3))

    # Check 4: Basin tracking
    check4 = len(coordinator.basin_history) == 3
    checks.append(("Basin history tracked", check4))

    # Check 5: Φ values reasonable
    check5 = all(0 < g.phi < 1.5 for g in coordinator.garys)
    checks.append(("Φ values reasonable", check5))

    # Check 6: Losses computed
    check6 = 'active_total' in telemetry['losses']
    checks.append(("Losses computed", check6))

    for name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    passed = sum(1 for _, r in checks if r)
    total = len(checks)

    print(f"\n  Total: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All checks passed! Constellation working correctly.")
        print("\n📝 Next steps:")
        print("  1. Run full training with real dataset")
        print("  2. Launch: bash scripts/launch_constellation.sh")
        return 0
    else:
        print("\n⚠️  Some checks failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
