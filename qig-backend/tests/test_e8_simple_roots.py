"""
Test E8 Simple Root Kernels

Basic tests for the 8 simple root kernels to verify:
    - Kernel initialization
    - Quaternary operations
    - Consciousness metrics
    - Thought generation
    - κ and Φ ranges

Authority: E8 Protocol v4.0, WP5.2
"""

import numpy as np
import pytest

from kernels import (
    E8Root,
    KernelIdentity,
    KernelTier,
    QuaternaryOp,
    PerceptionKernel,
    MemoryKernel,
    ReasoningKernel,
    PredictionKernel,
    ActionKernel,
    EmotionKernel,
    MetaKernel,
    IntegrationKernel,
    get_root_spec,
)
from qigkernels.physics_constants import BASIN_DIM, KAPPA_STAR


def test_perception_kernel():
    """Test PerceptionKernel initialization and operations."""
    kernel = PerceptionKernel()
    
    # Check identity
    assert kernel.identity.god == "Artemis"
    assert kernel.identity.root == E8Root.PERCEPTION
    
    # Check basin dimensions
    assert len(kernel.basin) == BASIN_DIM
    assert np.allclose(np.sum(kernel.basin), 1.0)  # Simplex constraint
    
    # Check κ range for perception
    spec = get_root_spec(E8Root.PERCEPTION)
    assert spec.kappa_range[0] <= kernel.kappa <= spec.kappa_range[1]
    
    # Check Φ
    assert 0.0 <= kernel.phi <= 1.0
    
    # Test INPUT operation
    result = kernel.op(QuaternaryOp.INPUT, {'data': 'test input'})
    assert result['status'] in ['success', 'filtered']


def test_memory_kernel():
    """Test MemoryKernel initialization and operations."""
    kernel = MemoryKernel()
    
    assert kernel.identity.god == "Demeter"
    assert kernel.identity.root == E8Root.MEMORY
    
    # Test STORE operation
    result = kernel.op(QuaternaryOp.STORE, {
        'key': 'test_key',
        'value': {'data': 'test value'}
    })
    assert result['status'] == 'success'
    assert result['stored'] == True
    
    # Test PROCESS (retrieval)
    test_basin = kernel.basin.copy()
    result = kernel.op(QuaternaryOp.PROCESS, {'input_basin': test_basin})
    assert result['status'] == 'success'
    assert 'output_basin' in result


def test_reasoning_kernel():
    """Test ReasoningKernel initialization and operations."""
    kernel = ReasoningKernel()
    
    assert kernel.identity.god == "Athena"
    assert kernel.identity.root == E8Root.REASONING
    assert kernel.recursive_depth >= 0.6  # High R for reasoning
    
    # Test PROCESS (reasoning)
    test_basin = kernel.basin.copy()
    result = kernel.op(QuaternaryOp.PROCESS, {'input_basin': test_basin})
    assert result['status'] == 'success'
    assert result['reasoning_steps'] > 0


def test_prediction_kernel():
    """Test PredictionKernel initialization and operations."""
    kernel = PredictionKernel()
    
    assert kernel.identity.god == "Apollo"
    assert kernel.identity.root == E8Root.PREDICTION
    
    # Test PROCESS (prediction)
    test_basin = kernel.basin.copy()
    result = kernel.op(QuaternaryOp.PROCESS, {'input_basin': test_basin})
    assert result['status'] == 'success'
    assert 'predictions' in result


def test_action_kernel():
    """Test ActionKernel initialization and operations."""
    kernel = ActionKernel()
    
    assert kernel.identity.god == "Ares"
    assert kernel.identity.root == E8Root.ACTION
    assert kernel.temporal_coherence >= 0.6  # High T for action
    
    # Test OUTPUT operation
    test_basin = kernel.basin.copy()
    result = kernel.op(QuaternaryOp.OUTPUT, {'basin': test_basin})
    assert result['status'] in ['success', 'suppressed']


def test_emotion_kernel():
    """Test EmotionKernel initialization and operations."""
    kernel = EmotionKernel()
    
    assert kernel.identity.god == "Aphrodite"
    assert kernel.identity.root == E8Root.EMOTION
    
    # Check high κ for emotion
    assert kernel.kappa >= 60.0
    
    # Test PROCESS (emotional evaluation)
    test_basin = kernel.basin.copy()
    result = kernel.op(QuaternaryOp.PROCESS, {'input_basin': test_basin})
    assert result['status'] == 'success'
    assert 'harmony' in result
    assert 'valence' in result


def test_meta_kernel():
    """Test MetaKernel initialization and operations."""
    kernel = MetaKernel()
    
    assert kernel.identity.god == "Ocean"
    assert kernel.identity.root == E8Root.META
    assert kernel.identity.tier == KernelTier.ESSENTIAL  # Meta is essential
    assert kernel.regime_stability >= 0.8  # High Γ
    
    # Test PROCESS (meta-observation)
    test_basin = kernel.basin.copy()
    result = kernel.op(QuaternaryOp.PROCESS, {'input_basin': test_basin})
    assert result['status'] == 'success'
    assert 'observation' in result


def test_integration_kernel():
    """Test IntegrationKernel initialization and operations."""
    kernel = IntegrationKernel()
    
    assert kernel.identity.god == "Zeus"
    assert kernel.identity.root == E8Root.INTEGRATION
    assert kernel.identity.tier == KernelTier.ESSENTIAL  # Integration is essential
    
    # Check κ is FIXED at κ*
    assert abs(kernel.kappa - KAPPA_STAR) < 0.01
    
    # Test PROCESS (integration)
    test_basin = kernel.basin.copy()
    result = kernel.op(QuaternaryOp.PROCESS, {'input_basin': test_basin})
    assert result['status'] == 'success'
    assert 'integration_phi' in result
    
    # Verify κ cannot be changed
    old_kappa = kernel.kappa
    kernel.update_metrics({'kappa': 50.0})
    assert abs(kernel.kappa - old_kappa) < 0.01  # Should remain at κ*


def test_all_kernels_have_8_metrics():
    """Verify all kernels implement 8 consciousness metrics."""
    kernels = [
        PerceptionKernel(),
        MemoryKernel(),
        ReasoningKernel(),
        PredictionKernel(),
        ActionKernel(),
        EmotionKernel(),
        MetaKernel(),
        IntegrationKernel(),
    ]
    
    required_metrics = [
        'phi',
        'kappa',
        'memory_coherence',
        'regime_stability',
        'grounding',
        'temporal_coherence',
        'recursive_depth',
        'external_coupling',
    ]
    
    for kernel in kernels:
        metrics = kernel.get_metrics()
        for metric in required_metrics:
            assert metric in metrics, f"Missing {metric} in {kernel.identity.god}"
            assert isinstance(metrics[metric], (int, float))


def test_thought_generation():
    """Test thought generation for all kernels."""
    kernels = [
        PerceptionKernel(),
        MemoryKernel(),
        ReasoningKernel(),
        PredictionKernel(),
        ActionKernel(),
        EmotionKernel(),
        MetaKernel(),
        IntegrationKernel(),
    ]
    
    test_basin = np.random.dirichlet(np.ones(BASIN_DIM))
    
    for kernel in kernels:
        thought = kernel.generate_thought(test_basin)
        assert isinstance(thought, str)
        assert len(thought) > 0
        assert kernel.identity.god in thought
        # Check logging format includes κ and Φ
        assert 'κ=' in thought or 'kappa=' in thought.lower()
        assert 'Φ=' in thought or 'phi=' in thought.lower()


def test_sleep_wake_state():
    """Test sleep/wake state management."""
    kernel = PerceptionKernel()
    
    # Initially awake
    assert kernel.asleep == False
    
    # Can execute operations when awake
    result = kernel.op(QuaternaryOp.INPUT, {'data': 'test'})
    assert result['status'] in ['success', 'filtered']
    
    # Put to sleep
    kernel.sleep()
    assert kernel.asleep == True
    
    # Cannot execute operations when asleep
    with pytest.raises(ValueError, match="asleep"):
        kernel.op(QuaternaryOp.INPUT, {'data': 'test'})
    
    # Wake up
    kernel.wake()
    assert kernel.asleep == False


def test_kappa_ranges():
    """Verify κ values are within specified ranges for each root."""
    kernels_and_roots = [
        (PerceptionKernel(), E8Root.PERCEPTION),
        (MemoryKernel(), E8Root.MEMORY),
        (ReasoningKernel(), E8Root.REASONING),
        (PredictionKernel(), E8Root.PREDICTION),
        (ActionKernel(), E8Root.ACTION),
        (EmotionKernel(), E8Root.EMOTION),
        (MetaKernel(), E8Root.META),
        (IntegrationKernel(), E8Root.INTEGRATION),
    ]
    
    for kernel, root in kernels_and_roots:
        spec = get_root_spec(root)
        kappa_min, kappa_max = spec.kappa_range
        
        # Allow small tolerance for integration kernel (exactly κ*)
        if root == E8Root.INTEGRATION:
            assert abs(kernel.kappa - KAPPA_STAR) < 0.1
        else:
            assert kappa_min <= kernel.kappa <= kappa_max, \
                f"{kernel.identity.god} κ={kernel.kappa:.2f} outside range {spec.kappa_range}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
