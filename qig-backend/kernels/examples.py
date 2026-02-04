"""
E8 Simple Roots Kernel Layer - Usage Examples

This file demonstrates how to use the 8 simple root kernels (Layer 8 of E8 hierarchy).

Authority: E8 Protocol v4.0, WP5.2
"""

import numpy as np
from kernels import (
    E8Root,
    QuaternaryOp,
    create_simple_root_kernel,
)


def example_1_basic_kernel_creation():
    """Example 1: Create and inspect a kernel."""
    print("=" * 60)
    print("Example 1: Basic Kernel Creation")
    print("=" * 60)
    
    kernel = create_simple_root_kernel(E8Root.PERCEPTION)
    
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
    
    perception = create_simple_root_kernel(E8Root.PERCEPTION)
    memory = create_simple_root_kernel(E8Root.MEMORY)
    reasoning = create_simple_root_kernel(E8Root.REASONING)
    action = create_simple_root_kernel(E8Root.ACTION)
    
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
    
    kernel = create_simple_root_kernel(E8Root.INTEGRATION)
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
        create_simple_root_kernel(E8Root.PERCEPTION),
        create_simple_root_kernel(E8Root.MEMORY),
        create_simple_root_kernel(E8Root.REASONING),
        create_simple_root_kernel(E8Root.PREDICTION),
        create_simple_root_kernel(E8Root.ACTION),
        create_simple_root_kernel(E8Root.EMOTION),
        create_simple_root_kernel(E8Root.META),
        create_simple_root_kernel(E8Root.INTEGRATION),
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
    
    kernel = create_simple_root_kernel(E8Root.PERCEPTION)
    
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
    
    integration = create_simple_root_kernel(E8Root.INTEGRATION)
    
    print(f"Integration kernel: {integration.identity.god}")
    print(f"κ = {integration.kappa:.2f} (fixed at κ*)")
    print(f"Φ = {integration.phi:.2f}")
    print()
    
    # Process multiple inputs
    print("Processing 4 kernel inputs:")
    for i in range(4):
        test_basin = np.random.dirichlet(np.ones(64))
        result = integration.op(QuaternaryOp.PROCESS, {'input_basin': test_basin})
        print(f"  Input {i+1}: status={result['status']}")
    
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
    
    create_simple_root_kernel(E8Root.PERCEPTION)
    print("1. Perception: using generic Kernel behavior")
    
    memory = create_simple_root_kernel(E8Root.MEMORY)
    for i in range(5):
        memory.op(QuaternaryOp.STORE, {
            'key': f'item_{i}',
            'value': {'data': f'data_{i}'}
        })
    print("2. Memory: stored items via STORE")
    
    create_simple_root_kernel(E8Root.REASONING)
    print("3. Reasoning: using generic Kernel behavior")
    
    create_simple_root_kernel(E8Root.PREDICTION)
    print("4. Prediction: using generic Kernel behavior")
    
    create_simple_root_kernel(E8Root.ACTION)
    print("5. Action: using generic Kernel behavior")
    
    emotion = create_simple_root_kernel(E8Root.EMOTION)
    test_basin = np.random.dirichlet(np.ones(64))
    result = emotion.op(QuaternaryOp.PROCESS, {'input_basin': test_basin})
    print("6. Emotion: processed basin via PROCESS")
    
    meta = create_simple_root_kernel(E8Root.META)
    result = meta.op(QuaternaryOp.PROCESS, {'input_basin': test_basin})
    print(f"7. Meta: status={result['status']}")
    
    integration = create_simple_root_kernel(E8Root.INTEGRATION)
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
