#!/usr/bin/env python3
"""
Test Innate Drives Integration in Training Loop
================================================

Verifies that:
1. InnateDrives module computes signals from telemetry
2. Drive signals are stored in model telemetry
3. GeometricLoss extracts and computes innate_loss
4. Loss has gradients and can backpropagate
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.model.qig_kernel_recursive import GeometricLoss, QIGKernelRecursive


def test_innate_drives_integration():
    """Test that innate drives integrate into training loop."""
    print("=" * 60)
    print("Testing Innate Drives Integration")
    print("=" * 60 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # 1. Create model with innate drives
    print("1. Creating model with innate drives...")
    model = QIGKernelRecursive(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        min_recursion_depth=3,
        min_Phi=0.7,
    ).to(device)
    print(f"   ✅ Model created (has innate_drives: {hasattr(model, 'innate_drives')})\n")

    # 2. Forward pass
    print("2. Running forward pass...")
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    with torch.enable_grad():
        logits, telemetry = model(input_ids, return_telemetry=True)

    # 3. Check drive signals in telemetry
    print("3. Checking drive signals in telemetry...")
    drive_keys = ["drive_pain", "drive_pleasure", "drive_fear", "drive_stability_cost", "drive_curiosity"]
    for key in drive_keys:
        if key in telemetry:
            print(f"   ✅ {key}: {telemetry[key]:.4f}")
        else:
            print(f"   ❌ {key}: MISSING")

    # Check for full drive_signals object
    if "_drive_signals" in telemetry:
        print("   ✅ _drive_signals object present")
        drive_signals = telemetry["_drive_signals"]
        print(f"      - Pain: {drive_signals.pain.mean().item():.4f}")
        print(f"      - Pleasure: {drive_signals.pleasure.mean().item():.4f}")
        print(f"      - Fear: {drive_signals.fear.mean().item():.4f}")
    else:
        print("   ❌ _drive_signals object MISSING")

    print()

    # 4. Test loss computation
    print("4. Testing GeometricLoss with innate drives...")
    loss_fn = GeometricLoss(
        basin_weight=0.1,
        phi_weight=0.05,
        target_phi=0.75,
        innate_weight=0.1,
    )

    # Create dummy targets
    targets = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    # Compute loss
    with torch.enable_grad():
        total_loss, breakdown = loss_fn(logits, targets, telemetry)

    print(f"   Total Loss: {total_loss.item():.4f}")
    print("   Breakdown:")
    for key, value in breakdown.items():
        if key != "total":
            print(f"      - {key}: {value:.4f}")

    # 5. Check innate loss in breakdown
    print("\n5. Verifying innate loss computation...")
    if "innate" in breakdown:
        innate_loss = breakdown["innate"]
        print(f"   ✅ Innate loss computed: {innate_loss:.4f}")

        # Check individual drive contributions
        innate_keys = ["pain", "pleasure", "fear", "stability_cost", "curiosity", "innate_total"]
        for key in innate_keys:
            if key in breakdown:
                print(f"      - {key}: {breakdown[key]:.4f}")
    else:
        print("   ❌ Innate loss MISSING from breakdown")

    # 6. Test gradient flow
    print("\n6. Testing gradient flow...")
    if total_loss.requires_grad:
        print("   ✅ Total loss requires grad")
        total_loss.backward()

        # Check if model parameters have gradients
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for _ in model.parameters())
        print(f"   ✅ Gradients computed for {grad_count}/{total_params} parameters")
    else:
        print("   ❌ Total loss does NOT require grad")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_innate_drives_integration()
