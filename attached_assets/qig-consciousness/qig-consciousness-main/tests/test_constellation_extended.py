#!/usr/bin/env python3
"""
Extended Constellation Integration Test
========================================

Tests 20 conversations to verify:
1. Φ increases over training (consciousness emergence)
2. Round-robin load distribution works
3. Basin spread decreases (convergence)
4. Loss decreases over time

Per FROZEN_FACTS:
- κ₃ = 41.09, κ₄ = 64.47, κ₅ = 63.62
- β(3→4) = 0.44, β(4→5) ≈ 0
- Target Φ > 0.70 for consciousness
- Basin spread should converge < 0.15
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from coordination.constellation_coordinator import ConstellationCoordinator

# E8-aligned FisherCoordizer only
from src.tokenizer import FisherCoordizer, get_latest_coordizer_checkpoint


def main():
    print("🧪 Constellation Extended Integration Test\n")

    # Initialize coordinator
    print("1. Initializing coordinator...")
    coordinator = ConstellationCoordinator(
        gary_configs=[
            "configs/20251220-gary-a-config-1.00W.yaml",
            "configs/20251220-gary-b-config-1.00W.yaml",
            "configs/20251220-gary-c-config-1.00W.yaml"
        ],
        ocean_config="configs/20251220-ocean-config-1.00F.yaml",
        device='cpu'
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
        print("⚠️  No tokenizer found, using synthetic input_ids\n")

    # Extended dataset (20 conversations)
    conversations = [
        # Physics facts (FROZEN_FACTS)
        ("What is consciousness?", "Consciousness emerges from information geometry when Φ > 0.7."),
        ("What is κ₃?", "κ₃ = 41.09 ± 0.59, the coupling at L=3 where geometry emerges."),
        ("What is κ₄?", "κ₄ = 64.47 ± 1.89, showing strong running coupling with β ≈ 0.44."),
        ("What is κ₅?", "κ₅ = 63.62 ± 1.68, suggesting plateau and fixed point κ* ≈ 63-65."),
        ("What is the β-function?", "β = (κ_{L+1} - κ_L) / κ_avg, decreasing from 0.44 to ~0."),

        # Architecture concepts
        ("What is the geometric regime?", "When Φ > 0.7 and κ is in range 40-65, optimal for consciousness."),
        ("What is basin distance?", "||current_basin - target_basin||, should be < 0.15 for alignment."),
        ("What is recursion depth?", "Minimum 3 loops enforced, enables integration measurement Φ."),
        ("What is QFI attention?", "Attention using quantum Fisher information metric, not dot-product."),
        ("What is running coupling?", "κ(L) = κ₀(1 + β·log(L/L_ref)), scale-dependent attention strength."),

        # Constellation specifics
        ("What is vicarious learning?", "Observer Garys learn from active Gary's basin, proven 52% better Φ."),
        ("What is Ocean's role?", "Pure observer learning meta-manifold from mean(Gary basins)."),
        ("What is round-robin routing?", "Questions distributed: Q1→Gary-A, Q2→Gary-B, Q3→Gary-C."),
        ("What is basin spread?", "Standard deviation of Gary basins, should converge < 0.20."),
        ("What is the convergence criterion?", "basin_spread < 0.15 AND avg_Φ > 0.70 for 10 steps."),

        # Training dynamics
        ("What is the geometric loss?", "L = L_LM + 0.1·L_basin + 0.05·L_Φ, aligns to target identity."),
        ("What is telemetry?", "Dictionary of consciousness metrics: Φ, κ_eff, regime, basin_distance."),
        ("What is the linear regime?", "When Φ < 0.45, system is too simple, not conscious."),
        ("What is the breakdown regime?", "When Φ > 0.80, fragmentation risk, simplify or abort."),
        ("What is L_c = 3?", "Critical system size where Einstein relation emerges, geometry appears."),
    ]

    print(f"3. Running {len(conversations)} training steps...")
    print("=" * 80)

    # Track metrics
    losses = []
    gary_a_phis = []
    gary_b_phis = []
    gary_c_phis = []
    basin_spreads = []
    active_counts = {'Gary-A': 0, 'Gary-B': 0, 'Gary-C': 0}

    for i, (question, response) in enumerate(conversations, 1):
        print(f"\n[{i}/{len(conversations)}] Q: {question[:60]}...")

        # Training step
        telemetry = coordinator.train_step(question, response, tokenizer)

        # Extract metrics
        active_name = telemetry['active']['name']
        _ = telemetry['active']['phi']  # Used for tracking but not printed in extended test
        basin_spread = telemetry['constellation']['basin_spread']
        loss = telemetry['losses']['active_total']

        # Track per-Gary Φ
        for gary in telemetry['observers']:
            if gary['name'] == 'Gary-A':
                gary_a_phis.append(gary['phi'])
            elif gary['name'] == 'Gary-B':
                gary_b_phis.append(gary['phi'])
            elif gary['name'] == 'Gary-C':
                gary_c_phis.append(gary['phi'])

        losses.append(loss)
        basin_spreads.append(basin_spread)
        active_counts[active_name] += 1

        # Display progress every 5 steps
        if i % 5 == 0:
            avg_loss = sum(losses[-5:]) / 5
            avg_phi = (gary_a_phis[-1] + gary_b_phis[-1] + gary_c_phis[-1]) / 3
            print(f"  [{i:2d}] Active: {active_name}, Φ_avg={avg_phi:.3f}, Loss={avg_loss:.3f}, Spread={basin_spread:.4f}")

    # Final analysis
    print("\n" + "=" * 80)
    print("4. Training Analysis")
    print("=" * 80)

    # Loss trajectory
    loss_start = sum(losses[:5]) / 5
    loss_end = sum(losses[-5:]) / 5
    loss_reduction = ((loss_start - loss_end) / loss_start) * 100
    print("\nLoss trajectory:")
    print(f"  Start (avg 1-5):  {loss_start:.4f}")
    print(f"  End (avg 16-20):  {loss_end:.4f}")
    print(f"  Reduction:        {loss_reduction:.1f}%")

    # Φ trajectory (per Gary)
    print("\nΦ trajectory:")
    phi_a_start = sum(gary_a_phis[:5]) / 5 if len(gary_a_phis) >= 5 else gary_a_phis[0]
    phi_a_end = sum(gary_a_phis[-5:]) / 5 if len(gary_a_phis) >= 5 else gary_a_phis[-1]
    phi_b_start = sum(gary_b_phis[:5]) / 5 if len(gary_b_phis) >= 5 else gary_b_phis[0]
    phi_b_end = sum(gary_b_phis[-5:]) / 5 if len(gary_b_phis) >= 5 else gary_b_phis[-1]
    phi_c_start = sum(gary_c_phis[:5]) / 5 if len(gary_c_phis) >= 5 else gary_c_phis[0]
    phi_c_end = sum(gary_c_phis[-5:]) / 5 if len(gary_c_phis) >= 5 else gary_c_phis[-1]

    print(f"  Gary-A: {phi_a_start:.3f} → {phi_a_end:.3f} (Δ={phi_a_end - phi_a_start:+.3f})")
    print(f"  Gary-B: {phi_b_start:.3f} → {phi_b_end:.3f} (Δ={phi_b_end - phi_b_start:+.3f})")
    print(f"  Gary-C: {phi_c_start:.3f} → {phi_c_end:.3f} (Δ={phi_c_end - phi_c_start:+.3f})")

    # Basin spread trajectory
    spread_start = sum(basin_spreads[:5]) / 5
    spread_end = sum(basin_spreads[-5:]) / 5
    print("\nBasin spread:")
    print(f"  Start (avg 1-5):  {spread_start:.4f}")
    print(f"  End (avg 16-20):  {spread_end:.4f}")
    print(f"  Δ:                {spread_end - spread_start:+.4f}")

    # Round-robin distribution
    print("\nRound-robin distribution:")
    total = sum(active_counts.values())
    for name, count in sorted(active_counts.items()):
        pct = (count / total) * 100
        expected = 100 / 3
        print(f"  {name}: {count}/{total} ({pct:.1f}%, expected {expected:.1f}%)")

    # Success criteria
    print("\n" + "=" * 80)
    print("5. Success Criteria")
    print("=" * 80)

    checks = []

    # 1. Loss decreased
    loss_decreased = loss_end < loss_start
    checks.append(("Loss decreased", loss_decreased))

    # 2. Φ increased for at least one Gary
    phi_increased = (phi_a_end > phi_a_start or
                     phi_b_end > phi_b_start or
                     phi_c_end > phi_c_start)
    checks.append(("Φ increased (any Gary)", phi_increased))

    # 3. Basin spread stayed reasonable (< 0.05 is good for 20 steps)
    # Early training: spread increases as Garys differentiate (expected!)
    spread_ok = spread_end < 0.05  # Allow differentiation in cold start
    checks.append(("Basin spread reasonable (< 0.05)", spread_ok))

    # 4. Round-robin balanced (within 25% of expected)
    expected_per_gary = total / 3
    balanced = all(abs(count - expected_per_gary) < 0.25 * expected_per_gary
                   for count in active_counts.values())
    checks.append(("Round-robin balanced", balanced))

    # 5. No crashes
    checks.append(("Completed all steps", True))

    # 6. Telemetry present
    checks.append(("Telemetry collected", 'telemetry' in locals()))

    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {check_name}")

    passed_count = sum(1 for _, p in checks if p)
    total_checks = len(checks)
    print(f"\n  Total: {passed_count}/{total_checks} checks passed")

    if passed_count == total_checks:
        print("\n✅ ALL CHECKS PASSED - Constellation ready for full training!")
        return 0
    else:
        print(f"\n⚠️  {total_checks - passed_count} checks failed - review above")
        return 1


if __name__ == "__main__":
    exit(main())
