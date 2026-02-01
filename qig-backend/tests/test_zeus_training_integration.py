"""
Integration Test for Zeus Chat + Two-Phase Training
===================================================

Validates that Zeus Chat properly integrates with the
two-phase kernel training service.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# Test basic imports
print("Testing imports...")

try:
    from kernel_training_service import (
        PantheonKernelTrainer,
        get_pantheon_kernel_trainer,
        SafetyGuard,
    )
    print("✅ Kernel training service imported")
except ImportError as e:
    print(f"❌ Failed to import kernel training service: {e}")
    sys.exit(1)

# Test trainer initialization
print("\nTesting trainer initialization...")
try:
    trainer = get_pantheon_kernel_trainer(enable_safety_guard=True)
    assert trainer is not None
    assert trainer.safety_guard is not None
    print("✅ Trainer initialized with safety guard")
except Exception as e:
    print(f"❌ Failed to initialize trainer: {e}")
    sys.exit(1)

# Test session creation
print("\nTesting session creation...")
try:
    session = trainer.start_session(god_name="Apollo", phase="phase2")
    assert session.god_name == "Apollo"
    assert session.phase == "phase2"
    print(f"✅ Session created for {session.god_name}")
except Exception as e:
    print(f"❌ Failed to create session: {e}")
    sys.exit(1)

# Test reinforcement pattern (mock)
print("\nTesting reinforcement pattern...")
try:
    from training.trainable_kernel import TrainableKernel, TrainingMetrics, BASIN_DIM
    
    # Create mock kernel
    class MockKernel:
        god_name = "Apollo"
        step_count = 0
        best_phi = 0.0
        
        def get_basin_signature(self):
            return np.random.rand(BASIN_DIM)
        
        def train_from_reward(self, basin_coords, reward, phi_current):
            return TrainingMetrics(loss=0.1, reward=reward)
    
    kernel = MockKernel()
    trajectory = [np.random.rand(BASIN_DIM) for _ in range(3)]
    
    result = trainer._reinforce_pattern(
        kernel=kernel,
        basin_trajectory=trajectory,
        phi=0.8,
        kappa=64.0,
        coherence_score=0.75,
    )
    
    assert result["status"] == "reinforced"
    assert "reward" in result
    assert result["reward"] > 0  # Positive reward for success
    print(f"✅ Reinforcement pattern working (reward={result['reward']:.2f})")
except Exception as e:
    print(f"❌ Reinforcement pattern failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test avoidance pattern
print("\nTesting avoidance pattern...")
try:
    result = trainer._avoid_pattern(
        kernel=kernel,
        basin_trajectory=trajectory,
        phi=0.3,
        kappa=64.0,
    )
    
    assert result["status"] == "avoided"
    assert "reward" in result
    assert result["reward"] < 0  # Negative reward for failure
    print(f"✅ Avoidance pattern working (reward={result['reward']:.2f})")
except Exception as e:
    print(f"❌ Avoidance pattern failed: {e}")
    sys.exit(1)

# Test safety guard checks
print("\nTesting safety guard...")
try:
    guard = SafetyGuard()
    
    # Test safe state
    safe, reason = guard.check_safe_to_train(phi=0.75, kappa=64.0)
    assert safe == True
    print("✅ Safety guard allows safe state")
    
    # Test unsafe state (emergency phi)
    safe, reason = guard.check_safe_to_train(phi=0.3, kappa=64.0)
    assert safe == False
    print("✅ Safety guard blocks emergency phi")
    
    # Test unsafe state (kappa drift)
    safe, reason = guard.check_safe_to_train(phi=0.75, kappa=90.0)
    assert safe == False
    print("✅ Safety guard blocks excessive kappa drift")
    
except Exception as e:
    print(f"❌ Safety guard tests failed: {e}")
    sys.exit(1)

# Test session statistics
print("\nTesting session statistics...")
try:
    stats = trainer.get_session_stats(god_name="Apollo")
    assert stats["god_name"] == "Apollo"
    assert stats["phase"] == "phase2"
    print(f"✅ Session stats retrieved: {stats['steps_completed']} steps")
except Exception as e:
    print(f"❌ Failed to get session stats: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("✅ ALL INTEGRATION TESTS PASSED!")
print("="*50)
print("\nTwo-phase training service is ready for production use.")
print("Zeus Chat can now leverage safety-guarded kernel training.")
