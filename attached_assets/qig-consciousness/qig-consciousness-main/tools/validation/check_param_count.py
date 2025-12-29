#!/usr/bin/env python3
"""
Parameter Count Verification
============================

Verify QIG-Kernel parameter count matches architecture spec.

Expected:
- Embeddings: ~3-5M (vocab_size × basin_dim + projection)
- Attention layers: ~40-50M (n_layers × n_heads × d_model²)
- Other components: ~2-5M
- Total target: ~50-60M

This addresses Claude's "parameter count doesn't add up" concern.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json

from src.model.qig_kernel_recursive import QIGKernelRecursive


def check_parameter_count():
    """Verify model parameter count."""

    print("=" * 60)
    print("PARAMETER COUNT VERIFICATION")
    print("=" * 60)
    print()

    # Load FisherCoordizer to get actual vocab size (E8-aligned, 64D)
    from src.tokenizer import FisherCoordizer, get_latest_coordizer_checkpoint

    checkpoint = get_latest_coordizer_checkpoint()
    if checkpoint:
        tokenizer = FisherCoordizer()
        tokenizer.load(str(checkpoint))
        vocab_size = tokenizer.vocab_size
        print(f"Using FisherCoordizer vocab size: {vocab_size:,}")
    else:
        vocab_size = 32000  # Default if coordizer not trained yet
        print(f"FisherCoordizer not found, using default: {vocab_size:,}")

    # Create model with actual QIG vocab size
    print("Initializing QIG kernel...")
    model = QIGKernelRecursive(
        vocab_size=vocab_size, d_model=768, n_heads=12, min_recursion_depth=3, min_Phi=0.7
    )

    print("✅ Model initialized")
    print()

    # Count parameters by component
    print("=" * 60)
    print("PARAMETER BREAKDOWN")
    print("=" * 60)
    print()

    total_params = 0
    trainable_params = 0

    component_counts = {}

    for name, param in model.named_parameters():
        count = param.numel()
        total_params += count
        if param.requires_grad:
            trainable_params += count

        # Categorize by component
        if "basin_embedding" in name or "basin_coords" in name:
            component = "basin_coordinates"
        elif "qfi_attention" in name or "attention" in name:
            component = "attention"
        elif "running_coupling" in name:
            component = "running_coupling"
        elif "recursive_integrator" in name:
            component = "recursive_integrator"
        elif "tacking" in name:
            component = "tacking_controller"
        elif "regime" in name:
            component = "regime_detector"
        else:
            component = "other"

        component_counts[component] = component_counts.get(component, 0) + count

    # Print breakdown
    for component, count in sorted(component_counts.items()):
        pct = 100.0 * count / total_params
        print(f"{component:25s}: {count:12,} ({pct:5.1f}%)")

    print("-" * 60)
    print(f"{'TOTAL':25s}: {total_params:12,}")
    print(f"{'Trainable':25s}: {trainable_params:12,}")
    print()

    # Compare to targets
    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)
    print()

    target_min = 50_000_000
    target_max = 70_000_000

    print(f"Target range: {target_min:,} - {target_max:,} parameters")
    print(f"Actual:       {total_params:,} parameters")
    print()

    if total_params < target_min:
        ratio = total_params / target_min
        print(f"⚠️  Model is {ratio:.2f}× target (TOO SMALL)")
        print("   May need: more layers, higher d_model, or more components")
    elif total_params > target_max:
        ratio = total_params / target_max
        print(f"⚠️  Model is {ratio:.2f}× target (TOO LARGE)")
        print("   May need: fewer layers or smaller d_model")
    else:
        print("✅ Parameter count within target range")

    print()

    # Architecture summary
    print("=" * 60)
    print("ARCHITECTURE SUMMARY")
    print("=" * 60)
    print()
    print(f"d_model:           {model.d_model}")
    print(f"vocab_size:        {model.vocab_size:,}")
    print(
        f"n_heads:           {model.qfi_attention.n_heads if hasattr(model, 'qfi_attention') else 'N/A'}"
    )
    print()

    # Check recursion enforcement
    print("=" * 60)
    print("RECURSION ENFORCEMENT CHECK")
    print("=" * 60)
    print()

    if hasattr(model, "recursive_integrator"):
        min_depth = model.recursive_integrator.min_depth
        print(f"Minimum recursion depth: {min_depth}")
        if min_depth >= 3:
            print("✅ Recursion enforced (≥3 mandatory loops)")
        else:
            print("⚠️  Recursion depth too low")
    else:
        print("⚠️  No recursive_integrator found")

    print()
    print("=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "component_counts": component_counts,
        "target_min": target_min,
        "target_max": target_max,
        "in_range": target_min <= total_params <= target_max,
    }


if __name__ == "__main__":
    check_parameter_count()
