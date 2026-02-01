"""
E8 Simple Roots Kernel Layer - Usage Examples

This file demonstrates how to use the 8 simple root kernels (Layer 8 of E8 hierarchy).

Authority: E8 Protocol v4.0, WP5.2
"""

import numpy as np
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
)


def example_1_basic_kernel_creation():
    """Example 1: Create and inspect a kernel."""
    print("=" * 60)
    print("Example 1: Basic Kernel Creation")
    print("=" * 60)
    
    # Create a perception kernel
    kernel = PerceptionKernel()
    
    print(f"Created: {kernel.identity.god}")
    print(f"Root: {kernel.identity.root.value}")
    print(f"Tier: {kernel.identity.tier.value}")
    print(f"κ: {kernel.kappa:.2f}")
    print(f"Φ: {kernel.phi:.2f}")
    print(f"Basin dim: {len(kernel.basin)}")
    print(f"Asleep: {kernel.asleep}")
    print()


def example_2_quaternary_operations():
    """Example 2: Execute quaternary operations."""
    print("=" * 60)
    print("Example 2: Quaternary Operations")
    print("=" * 60)
    
    # Create kernels
    perception = PerceptionKernel()
    memory = MemoryKernel()
    reasoning = ReasoningKernel()
    action = ActionKernel()
    
    # INPUT: Perceive external data
    print("1. INPUT (Perception):")
    result = perception.op(QuaternaryOp.INPUT, {'data': 'hello world'})
    print(f"   Status: {result['status']}")
    print()
    
    # STORE: Save to memory
    print("2. STORE (Memory):")
    result = memory.op(QuaternaryOp.STORE, {
        'key': 'greeting',
        'value': {'text': 'hello world', 'basin': perception.basin}
    })
    print(f"   Status: {result['status']}")
    print(f"   Memory count: {result['memory_count']}")
    print()
    
    # PROCESS: Reason about input
    print("3. PROCESS (Reasoning):")
    result = reasoning.op(QuaternaryOp.PROCESS, {
        'input_basin': perception.basin
    })
    print(f"   Status: {result['status']}")
    print(f"   Reasoning steps: {result['reasoning_steps']}")
    print()
    
    # OUTPUT: Generate action
    print("4. OUTPUT (Action):")
    result = action.op(QuaternaryOp.OUTPUT, {
        'basin': reasoning.basin
    })
    print(f"   Status: {result['status']}")
    if result['status'] == 'success':
        print(f"   Thought: {result['thought'][:60]}...")
    print()


def example_3_consciousness_metrics():
    """Example 3: Monitor consciousness metrics."""
    print("=" * 60)
    print("Example 3: Consciousness Metrics (8 E8 Metrics)")
    print("=" * 60)
    
    kernel = IntegrationKernel()
    metrics = kernel.get_metrics()
    
    print(f"Kernel: {kernel.identity.god}")
    print(f"Φ (Integration):         {metrics['phi']:.3f}")
    print(f"κ (Coupling):            {metrics['kappa']:.2f}")
    print(f"M (Memory Coherence):    {metrics['memory_coherence']:.3f}")
    print(f"Γ (Regime Stability):    {metrics['regime_stability']:.3f}")
    print(f"G (Grounding):           {metrics['grounding']:.3f}")
    print(f"T (Temporal Coherence):  {metrics['temporal_coherence']:.3f}")
    print(f"R (Recursive Depth):     {metrics['recursive_depth']:.3f}")
    print(f"C (External Coupling):   {metrics['external_coupling']:.3f}")
    print()


def example_4_thought_generation():
    """Example 4: Generate thoughts from all 8 kernels."""
    print("=" * 60)
    print("Example 4: Thought Generation (All 8 Kernels)")
    print("=" * 60)
    
    # Create test basin
    test_basin = np.random.dirichlet(np.ones(64))
    
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
    
    for kernel in kernels:
        thought = kernel.generate_thought(test_basin)
        print(f"{thought[:70]}...")
    print()


def example_5_sleep_wake_cycle():
    """Example 5: Hemisphere sleep/wake cycle."""
    print("=" * 60)
    print("Example 5: Sleep/Wake Cycle")
    print("=" * 60)
    
    kernel = PerceptionKernel()
    
    print(f"Initial state: asleep={kernel.asleep}")
    
    # Can operate when awake
    result = kernel.op(QuaternaryOp.INPUT, {'data': 'test'})
    print(f"Operation while awake: {result['status']}")
    
    # Put to sleep
    kernel.sleep()
    print(f"After sleep(): asleep={kernel.asleep}")
    
    # Cannot operate when asleep
    try:
        kernel.op(QuaternaryOp.INPUT, {'data': 'test'})
    except ValueError as e:
        print(f"Operation while asleep: ERROR - {e}")
    
    # Wake up
    kernel.wake()
    print(f"After wake(): asleep={kernel.asleep}")
    print()


def example_6_integration_kernel():
    """Example 6: Integration kernel synthesizing multiple inputs."""
    print("=" * 60)
    print("Example 6: Integration Kernel (κ* Fixed Point)")
    print("=" * 60)
    
    integration = IntegrationKernel()
    
    print(f"Integration kernel: {integration.identity.god}")
    print(f"κ = {integration.kappa:.2f} (fixed at κ*)")
    print(f"Φ = {integration.phi:.2f}")
    print()
    
    # Process multiple inputs
    print("Processing 4 kernel inputs:")
    for i in range(4):
        test_basin = np.random.dirichlet(np.ones(64))
        result = integration.op(QuaternaryOp.PROCESS, {'input_basin': test_basin})
        print(f"  Input {i+1}: Φ={result['integration_phi']:.3f}, "
              f"kernel_count={result['kernel_count']}")
    
    # Verify κ is still fixed
    print(f"\nFinal κ: {integration.kappa:.2f} (still fixed at κ*)")
    
    # Try to change κ (will fail)
    print("\nAttempting to change κ...")
    integration.update_metrics({'kappa': 50.0})
    print(f"After update attempt: κ = {integration.kappa:.2f} (unchanged)")
    print()


def example_7_specialized_behaviors():
    """Example 7: Specialized kernel behaviors."""
    print("=" * 60)
    print("Example 7: Specialized Kernel Behaviors")
    print("=" * 60)
    
    # Perception: Signal filtering
    perception = PerceptionKernel()
    perception.set_signal_threshold(0.5)  # Increase threshold
    print(f"1. Perception: signal_threshold={perception.signal_threshold}")
    
    # Memory: Consolidation
    memory = MemoryKernel()
    for i in range(5):
        memory.op(QuaternaryOp.STORE, {
            'key': f'item_{i}',
            'value': {'data': f'data_{i}'}
        })
    print(f"2. Memory: stored {len(memory.memory_store)} items")
    
    # Reasoning: Inference depth
    reasoning = ReasoningKernel()
    reasoning.set_inference_depth(5)
    print(f"3. Reasoning: inference_depth={reasoning.inference_depth}")
    
    # Prediction: Horizon
    prediction = PredictionKernel()
    prediction.set_prediction_horizon(10)
    print(f"4. Prediction: prediction_horizon={prediction.prediction_horizon}")
    
    # Action: Threshold
    action = ActionKernel()
    action.set_action_threshold(0.6)
    print(f"5. Action: action_threshold={action.action_threshold}")
    
    # Emotion: Harmony
    emotion = EmotionKernel()
    test_basin = np.random.dirichlet(np.ones(64))
    result = emotion.op(QuaternaryOp.PROCESS, {'input_basin': test_basin})
    print(f"6. Emotion: harmony={result['harmony']:.3f}, "
          f"valence={result['valence']:.3f}")
    
    # Meta: Observations
    meta = MetaKernel()
    result = meta.op(QuaternaryOp.PROCESS, {'input_basin': test_basin})
    print(f"7. Meta: observation_count={result['observation_count']}")
    
    # Integration: Multi-kernel
    integration = IntegrationKernel()
    for _ in range(3):
        integration.op(QuaternaryOp.PROCESS, {
            'input_basin': np.random.dirichlet(np.ones(64))
        })
    print(f"8. Integration: integrated {len(integration.kernel_basins)} kernels")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("E8 SIMPLE ROOTS KERNEL LAYER - USAGE EXAMPLES")
    print("=" * 60)
    print()
    
    example_1_basic_kernel_creation()
    example_2_quaternary_operations()
    example_3_consciousness_metrics()
    example_4_thought_generation()
    example_5_sleep_wake_cycle()
    example_6_integration_kernel()
    example_7_specialized_behaviors()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
