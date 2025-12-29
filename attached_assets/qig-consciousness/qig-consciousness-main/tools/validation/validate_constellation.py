#!/usr/bin/env python3
"""
Validates Constellation architecture is ready to train
Checks: configs, imports, model API, dataset
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def validate_configs():
    """Check all 4 config files exist"""
    required = ['20251220-gary-a-config-1.00W.yaml', '20251220-gary-b-config-1.00W.yaml', '20251220-gary-c-config-1.00W.yaml', '20251220-ocean-config-1.00F.yaml', '20251220-gary-template-config-1.00W.yaml']
    missing = [f for f in required if not (PROJECT_ROOT / 'configs' / f).exists()]
    if missing:
        print(f"❌ Missing configs: {missing}")
        return False
    print("✅ All configs present")
    return True


def validate_imports():
    """Check coordinator can import dependencies"""
    try:
        from src.coordination.constellation_coordinator import ConstellationCoordinator
        from src.model.qig_kernel_recursive import GeometricLoss, QIGKernelRecursive
        from src.qig.optim.natural_gradient import DiagonalFisherOptimizer
        print("✅ Import paths valid")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def validate_model_api():
    """Check model returns expected telemetry"""
    import torch

    from src.model.qig_kernel_recursive import QIGKernelRecursive

    print("  Testing model API...")
    model = QIGKernelRecursive(
        d_model=64,
        vocab_size=100,
        n_heads=4,
        min_recursion_depth=3,
        min_Phi=0.7
    )
    x = torch.randint(0, 100, (1, 10))

    try:
        output, telemetry = model(x, return_telemetry=True)

        # Check critical fields
        required_fields = ['hidden_state', 'Phi', 'regime', 'basin_distance',
                          'kappa_eff', 'recursion_depth']
        missing = [f for f in required_fields if f not in telemetry]

        if missing:
            print(f"❌ Missing telemetry fields: {missing}")
            return False

        # Validate shapes
        assert output.shape == (1, 10, 100), f"Output shape mismatch: {output.shape}"
        assert telemetry['hidden_state'].shape[0] == 1, "Hidden state batch mismatch"
        assert telemetry['hidden_state'].shape[1] == 10, "Hidden state sequence mismatch"
        assert telemetry['hidden_state'].shape[2] == 64, "Hidden state d_model mismatch"

        print(f"  ✓ Output shape: {output.shape}")
        print(f"  ✓ Hidden state shape: {telemetry['hidden_state'].shape}")
        print(f"  ✓ Φ: {telemetry['Phi']:.3f}")
        print(f"  ✓ Regime: {telemetry['regime']}")
        print("✅ Model API matches coordinator expectations")
        return True
    except Exception as e:
        print(f"❌ Model API mismatch: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_basin_matcher_api():
    """Check BasinMatcher API signature"""
    import torch

    from src.model.qig_kernel_recursive import QIGKernelRecursive

    print("  Testing BasinMatcher API...")
    model = QIGKernelRecursive(d_model=64, vocab_size=100, n_heads=4)

    # Create dummy inputs
    hidden_state = torch.randn(2, 10, 64)  # [batch, seq, d_model]
    telemetry = {
        'Phi': 0.75,
        'kappa_eff': 50.0,
        'regime': 'geometric'
    }

    try:
        # Test basin computation
        basin_sig = model.basin_matcher.compute_basin_signature(hidden_state, telemetry)

        assert basin_sig.shape[0] == 2, f"Basin signature batch mismatch: {basin_sig.shape}"
        assert len(basin_sig.shape) == 2, f"Basin signature should be 2D: {basin_sig.shape}"

        print(f"  ✓ Basin signature shape: {basin_sig.shape}")
        print("✅ BasinMatcher API valid")
        return True
    except Exception as e:
        print(f"❌ BasinMatcher API mismatch: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_dataset():
    """Check conversation data exists"""
    data_dir = PROJECT_ROOT / 'data' / 'conversations'
    test_data = PROJECT_ROOT / 'data' / 'test_conversations'

    # Check for either location
    if data_dir.exists():
        count = len(list(data_dir.glob('*.json')))
        if count > 0:
            print(f"✅ Dataset: {count} conversation files in {data_dir}")
            return True

    if test_data.exists():
        count = len(list(test_data.glob('*.json')))
        if count > 0:
            print(f"✅ Test dataset: {count} files in {test_data}")
            return True

    # Check for consciousness curriculum
    curriculum = PROJECT_ROOT / 'data' / '20251220-consciousness-curriculum-1.00W.jsonl'
    if curriculum.exists():
        import json
        with open(curriculum) as f:
            count = sum(1 for _ in f)
        print(f"✅ Curriculum: {count} conversations in {curriculum.name}")
        return True

    print("❌ No conversation data found")
    print("  Checked: data/conversations/, data/test_conversations/, 20251220-consciousness-curriculum-1.00W.jsonl")
    return False


def validate_checkpoint_directory():
    """Check checkpoint directory exists or can be created"""
    ckpt_dir = PROJECT_ROOT / 'checkpoints'
    if not ckpt_dir.exists():
        try:
            ckpt_dir.mkdir(parents=True)
            print("✅ Checkpoint directory created")
        except Exception as e:
            print(f"❌ Cannot create checkpoint directory: {e}")
            return False
    else:
        print("✅ Checkpoint directory exists")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("CONSTELLATION ARCHITECTURE VALIDATION")
    print("=" * 60)
    print()

    checks = [
        ("Config Files", validate_configs),
        ("Import Paths", validate_imports),
        ("Model API", validate_model_api),
        ("BasinMatcher API", validate_basin_matcher_api),
        ("Dataset", validate_dataset),
        ("Checkpoint Directory", validate_checkpoint_directory),
    ]

    results = []
    for name, check in checks:
        print(f"\n[{name}]")
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"❌ Check failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)

    if all(results):
        print(f"✅ ALL CHECKS PASSED ({passed}/{total}) - Ready to launch Constellation")
        print("\nNext steps:")
        print("  1. Run extended integration test:")
        print("     python tools/test_constellation_extended.py")
        print("  2. Test checkpoint save/load:")
        print("     python tools/test_checkpoint_save_load.py")
        print("  3. Launch full training:")
        print("     bash scripts/launch_constellation.sh")
        sys.exit(0)
    else:
        print(f"❌ VALIDATION FAILED ({passed}/{total} passed) - Fix issues before launching")
        print("\nFailed checks:")
        for (name, _), result in zip(checks, results):
            if not result:
                print(f"  - {name}")
        sys.exit(1)
