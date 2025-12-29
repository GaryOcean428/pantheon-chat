#!/usr/bin/env python3
"""
Test Checkpoint Save/Load for Constellation Architecture
==========================================================

Validates that:
1. Constellation state can be saved mid-training
2. State can be loaded and resumed without loss
3. Φ trajectory continues smoothly (no reset)
4. Basin coordinates persist correctly
"""

import json
import sys
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.coordination.constellation_coordinator import ConstellationCoordinator


def test_checkpoint_save_load():
    """Test that checkpoint save/load preserves state"""

    print("=" * 60)
    print("CHECKPOINT SAVE/LOAD TEST")
    print("=" * 60)
    print()

    # Setup
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "test_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    gary_configs = [
        str(PROJECT_ROOT / "configs" / "20251220-gary-a-config-1.00W.yaml"),
        str(PROJECT_ROOT / "configs" / "20251220-gary-b-config-1.00W.yaml"),
        str(PROJECT_ROOT / "configs" / "20251220-gary-c-config-1.00W.yaml"),
    ]
    ocean_config = str(PROJECT_ROOT / "configs" / "20251220-ocean-config-1.00F.yaml")

    # Phase 1: Train for 10 steps and save
    print("[Phase 1] Training for 10 steps...")
    coordinator = ConstellationCoordinator(
        gary_configs=gary_configs,
        ocean_config=ocean_config,
        shared_basin_dir=str(checkpoint_dir / "basins"),
        device="cpu"
    )

    # Create geometric test sequences (pure QIG - no tokenizer dependency)
    # These are random walks in token space - valid geometric primitives
    vocab_size = 50257  # Standard vocab size from config
    seq_len = 32  # Standard test sequence length

    # Generate 10 geometric test sequences
    test_sequences = [
        torch.randint(0, vocab_size, (1, seq_len))
        for _ in range(10)
    ]

    # Train initial 10 steps using train_step()
    print("Training 10 steps...")
    for i, input_ids in enumerate(test_sequences):
        telemetry = coordinator.train_step(
            input_ids=input_ids  # Pure geometric - direct token sequence
        )
        if i % 3 == 0:
            print(f"  Step {i+1}/10: Active={telemetry['active']['name']}, Φ={telemetry['active']['phi']:.3f}")

    # Capture state after 10 steps
    state_before = {
        'garys': [
            {
                'name': g.name,
                'phi': g.phi,
                'kappa': g.kappa,
                'regime': g.regime,
                'basin_norm': torch.norm(g.basin).item(),
                'conversations': g.conversations
            }
            for g in coordinator.garys
        ],
        'ocean': {
            'phi': coordinator.ocean.phi,
            'kappa': coordinator.ocean.kappa,
            'regime': coordinator.ocean.regime,
            'basin_norm': torch.norm(coordinator.ocean.basin).item(),
            'conversations': coordinator.ocean.conversations
        },
        'total_conversations': coordinator.total_conversations,
        'active_index': coordinator.active_index
    }

    print("\nState after 10 steps:")
    print(f"  Gary-A: Φ={state_before['garys'][0]['phi']:.3f}, κ={state_before['garys'][0]['kappa']:.1f}")
    print(f"  Gary-B: Φ={state_before['garys'][1]['phi']:.3f}, κ={state_before['garys'][1]['kappa']:.1f}")
    print(f"  Gary-C: Φ={state_before['garys'][2]['phi']:.3f}, κ={state_before['garys'][2]['kappa']:.1f}")
    print(f"  Ocean:  Φ={state_before['ocean']['phi']:.3f}, κ={state_before['ocean']['kappa']:.1f}")
    print(f"  Active index: {state_before['active_index']}")

    # Save checkpoint
    checkpoint_path = checkpoint_dir / "constellation_step10.pt"
    print(f"\nSaving checkpoint to {checkpoint_path}...")
    coordinator.save_checkpoint(str(checkpoint_path))

    # Phase 2: Load checkpoint and verify state
    print("\n[Phase 2] Loading checkpoint...")
    coordinator2 = ConstellationCoordinator(
        gary_configs=gary_configs,
        ocean_config=ocean_config,
        shared_basin_dir=str(checkpoint_dir / "basins"),
        device="cpu"
    )

    # Load checkpoint
    coordinator2.load_checkpoint(str(checkpoint_path))

    # Capture loaded state
    state_after_load = {
        'garys': [
            {
                'name': g.name,
                'phi': g.phi,
                'kappa': g.kappa,
                'regime': g.regime,
                'basin_norm': torch.norm(g.basin).item(),
                'conversations': g.conversations
            }
            for g in coordinator2.garys
        ],
        'ocean': {
            'phi': coordinator2.ocean.phi,
            'kappa': coordinator2.ocean.kappa,
            'regime': coordinator2.ocean.regime,
            'basin_norm': torch.norm(coordinator2.ocean.basin).item(),
            'conversations': coordinator2.ocean.conversations
        },
        'total_conversations': coordinator2.total_conversations,
        'active_index': coordinator2.active_index
    }

    print("\nState after load:")
    print(f"  Gary-A: Φ={state_after_load['garys'][0]['phi']:.3f}, κ={state_after_load['garys'][0]['kappa']:.1f}")
    print(f"  Gary-B: Φ={state_after_load['garys'][1]['phi']:.3f}, κ={state_after_load['garys'][1]['kappa']:.1f}")
    print(f"  Gary-C: Φ={state_after_load['garys'][2]['phi']:.3f}, κ={state_after_load['garys'][2]['kappa']:.1f}")
    print(f"  Ocean:  Φ={state_after_load['ocean']['phi']:.3f}, κ={state_after_load['ocean']['kappa']:.1f}")
    print(f"  Active index: {state_after_load['active_index']}")

    # Phase 3: Continue training for 10 more steps
    print("\n[Phase 3] Continuing training for 10 more steps...")
    # Generate new geometric sequences for continuation (different random walk)
    continuation_sequences = [
        torch.randint(0, vocab_size, (1, seq_len))
        for _ in range(10)
    ]
    for i, input_ids in enumerate(continuation_sequences):
        telemetry = coordinator2.train_step(
            input_ids=input_ids  # Pure geometric - direct token sequence
        )
        if i % 3 == 0:
            print(f"  Step {i+11}/20: Active={telemetry['active']['name']}, Φ={telemetry['active']['phi']:.3f}")

    # Capture state after continuation
    state_after_continue = {
        'garys': [
            {
                'name': g.name,
                'phi': g.phi,
                'kappa': g.kappa,
                'regime': g.regime,
                'basin_norm': torch.norm(g.basin).item(),
                'conversations': g.conversations
            }
            for g in coordinator2.garys
        ],
        'ocean': {
            'phi': coordinator2.ocean.phi,
            'kappa': coordinator2.ocean.kappa,
            'regime': coordinator2.ocean.regime,
            'basin_norm': torch.norm(coordinator2.ocean.basin).item(),
            'conversations': coordinator2.ocean.conversations
        },
        'total_conversations': coordinator2.total_conversations
    }

    print("\nState after continuation (20 total steps):")
    print(f"  Gary-A: Φ={state_after_continue['garys'][0]['phi']:.3f}, κ={state_after_continue['garys'][0]['kappa']:.1f}")
    print(f"  Gary-B: Φ={state_after_continue['garys'][1]['phi']:.3f}, κ={state_after_continue['garys'][1]['kappa']:.1f}")
    print(f"  Gary-C: Φ={state_after_continue['garys'][2]['phi']:.3f}, κ={state_after_continue['garys'][2]['kappa']:.1f}")
    print(f"  Ocean:  Φ={state_after_continue['ocean']['phi']:.3f}, κ={state_after_continue['ocean']['kappa']:.1f}")
    print(f"  Total conversations: {state_after_continue['total_conversations']}")

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    checks = []

    # Check 1: State preserved after load
    for i in range(3):
        name = f"Gary-{chr(65+i)}"
        phi_match = abs(state_before['garys'][i]['phi'] - state_after_load['garys'][i]['phi']) < 0.001
        basin_match = abs(state_before['garys'][i]['basin_norm'] - state_after_load['garys'][i]['basin_norm']) < 0.001
        checks.append((f"{name} Φ preserved", phi_match))
        checks.append((f"{name} basin preserved", basin_match))

    ocean_phi_match = abs(state_before['ocean']['phi'] - state_after_load['ocean']['phi']) < 0.001
    ocean_basin_match = abs(state_before['ocean']['basin_norm'] - state_after_load['ocean']['basin_norm']) < 0.001
    checks.append(("Ocean Φ preserved", ocean_phi_match))
    checks.append(("Ocean basin preserved", ocean_basin_match))

    # Check 2: Conversation counts preserved
    conv_match = state_before['total_conversations'] == state_after_load['total_conversations']
    checks.append(("Conversation count preserved", conv_match))

    # Check 3: Active index preserved
    active_match = state_before['active_index'] == state_after_load['active_index']
    checks.append(("Active index preserved", active_match))

    # Check 4: Training continued successfully
    continued = state_after_continue['total_conversations'] == 20
    checks.append(("Training continued", continued))

    # Check 5: Φ didn't reset (should be different after continuation)
    phi_changed = any(
        abs(state_after_load['garys'][i]['phi'] - state_after_continue['garys'][i]['phi']) > 0.001
        for i in range(3)
    )
    checks.append(("Φ trajectory continuous (changed after continuation)", phi_changed))

    # Print results
    print()
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")

    # Summary
    print("\n" + "=" * 60)
    passed_count = sum(1 for _, p in checks if p)
    total_count = len(checks)

    if all(p for _, p in checks):
        print(f"✅ ALL CHECKS PASSED ({passed_count}/{total_count})")
        print("\nCheckpoint save/load is working correctly!")
        print("Ready for multi-day training runs with checkpoint resilience.")
        return True
    else:
        print(f"❌ SOME CHECKS FAILED ({passed_count}/{total_count})")
        print("\nFailed checks:")
        for check_name, passed in checks:
            if not passed:
                print(f"  - {check_name}")
        return False


if __name__ == '__main__':
    try:
        success = test_checkpoint_save_load()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
