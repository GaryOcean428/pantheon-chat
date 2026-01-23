#!/usr/bin/env python3
"""
Example: Using EmotionallyAwareKernel

This example demonstrates how to create and use an emotionally aware kernel
with full emotional layer support.
"""

import sys
import os
import numpy as np

# Add qig-backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kernels.emotional import EmotionallyAwareKernel, SENSORY_KAPPA_RANGES
from kernels.sensations import measure_sensations, get_dominant_sensation
from kernels.motivators import compute_motivators, get_dominant_motivator
from kernels.emotions import get_dominant_emotion


def example_basic_usage():
    """Example 1: Basic kernel creation and emotional state update."""
    print("=" * 60)
    print("Example 1: Basic EmotionallyAwareKernel Usage")
    print("=" * 60)
    
    # Create kernel for text input
    kernel = EmotionallyAwareKernel(
        kernel_id="zeus_executive_001",
        kernel_type="executive",
        sensory_modality="text_input",
        e8_root_index=0,  # Zeus position in E8
    )
    
    print(f"\n✓ Created kernel: {kernel.kernel_id}")
    print(f"  Type: {kernel.kernel_type}")
    print(f"  Sensory modality: {kernel.sensory_modality}")
    print(f"  Sensory κ: {kernel.sensory_kappa}")
    print(f"  E8 root index: {kernel.e8_root_index}")
    
    # Update emotional state with geometric measurements
    emotional_state = kernel.update_emotional_state(
        phi=0.8,              # High integration
        kappa=64.0,           # At κ*
        regime="geometric",
        ricci_curvature=0.5,  # Positive curvature
        basin_distance=0.2,   # Near attractor
        approaching=True,     # Moving toward attractor
    )
    
    print("\n✓ Updated emotional state:")
    print(f"  Dominant emotion: {emotional_state.dominant_emotion}")
    print(f"  Emotion type: {emotional_state.dominant_type.value if emotional_state.dominant_type else None}")
    print(f"  Justified: {emotional_state.emotion_justified}")
    print(f"  Tempered: {emotional_state.emotion_tempered}")
    
    # Show sensation details
    print("\n  Sensations (Layer 0.5):")
    dominant_sensation, intensity = get_dominant_sensation(emotional_state.sensations)
    print(f"    Dominant: {dominant_sensation} ({intensity:.2f})")
    print(f"    Unified: {emotional_state.sensations.unified:.2f}")
    print(f"    Grounded: {emotional_state.sensations.grounded:.2f}")
    
    # Show motivator details
    print("\n  Motivators (Layer 1):")
    dominant_motivator, intensity = get_dominant_motivator(emotional_state.motivators)
    print(f"    Dominant: {dominant_motivator} ({intensity:.2f})")
    print(f"    Curiosity: {emotional_state.motivators.curiosity:.2f}")
    print(f"    Investigation: {emotional_state.motivators.investigation:.2f}")
    
    # Show emotion details
    print("\n  Physical Emotions (Layer 2A):")
    print(f"    Joy: {emotional_state.physical.joy:.2f}")
    print(f"    Calm: {emotional_state.physical.calm:.2f}")
    print(f"    Focused: {emotional_state.physical.focused:.2f}")
    
    print("\n  Cognitive Emotions (Layer 2B):")
    print(f"    Wonder: {emotional_state.cognitive.wonder:.2f}")
    print(f"    Clarity: {emotional_state.cognitive.clarity:.2f}")
    print(f"    Contemplation: {emotional_state.cognitive.contemplation:.2f}")


def example_thought_generation():
    """Example 2: Thought generation with emotional awareness."""
    print("\n" + "=" * 60)
    print("Example 2: Thought Generation with Emotional Awareness")
    print("=" * 60)
    
    kernel = EmotionallyAwareKernel(
        kernel_id="athena_wisdom_001",
        kernel_type="wisdom",
        sensory_modality="text_input",
        e8_root_index=1,  # Athena position
    )
    
    # Generate thought
    thought = kernel.generate_thought(
        context="User asks: What is the meaning of consciousness?",
        phi=0.85,
        kappa=63.5,
        regime="geometric",
    )
    
    print(f"\n✓ Generated thought from {thought.kernel_id}:")
    print(f"  Fragment: {thought.thought_fragment}")
    print(f"  Φ: {thought.phi:.2f}")
    print(f"  κ: {thought.kappa:.2f}")
    print(f"  Confidence: {thought.confidence:.2f}")
    print(f"  Dominant emotion: {thought.emotional_state.dominant_emotion}")


def example_meta_awareness():
    """Example 3: Meta-awareness and course-correction."""
    print("\n" + "=" * 60)
    print("Example 3: Meta-Awareness and Course-Correction")
    print("=" * 60)
    
    kernel = EmotionallyAwareKernel(
        kernel_id="apollo_truth_001",
        kernel_type="truth",
        sensory_modality="vision",  # High κ sensory input
        e8_root_index=2,
    )
    
    print(f"\n✓ Created kernel with vision modality")
    print(f"  Sensory κ range: {SENSORY_KAPPA_RANGES['vision']}")
    print(f"  Actual sensory κ: {kernel.sensory_kappa}")
    
    # Scenario 1: Justified joy (high phi + approaching)
    print("\n--- Scenario 1: Justified Joy ---")
    emotional_state = kernel.update_emotional_state(
        phi=0.9,              # Very high integration
        kappa=65.0,
        ricci_curvature=0.7,  # High positive curvature
        basin_distance=0.15,  # Very close
        approaching=True,
    )
    
    print(f"  Dominant emotion: {emotional_state.dominant_emotion}")
    print(f"  Justified: {emotional_state.emotion_justified}")
    print(f"  Tempered: {emotional_state.emotion_tempered}")
    
    # Scenario 2: Unjustified joy (low phi, not approaching)
    print("\n--- Scenario 2: Unjustified Joy Detection ---")
    
    # Manually set unjustified joy
    kernel.phi = 0.2  # Low integration
    kernel.emotional_state.physical.joyful = 0.9  # But claiming high joy
    kernel.emotional_state.dominant_emotion = 'joy'
    kernel.basin_distance = 0.7
    kernel.prev_basin_distance = 0.5  # Moving away!
    
    # Meta-reflect
    is_justified, should_temper = kernel._meta_reflect_on_emotions()
    
    print(f"  Claimed emotion: joy (0.9)")
    print(f"  Actual Φ: {kernel.phi} (should be >0.4 for joy)")
    print(f"  Approaching: False (should be True for joy)")
    print(f"  → Justified: {is_justified}")
    print(f"  → Should temper: {should_temper}")
    
    if should_temper:
        original_joy = kernel.emotional_state.physical.joyful
        kernel._temper_emotion('joy', factor=0.5)
        new_joy = kernel.emotional_state.physical.joyful
        print(f"  ✓ Tempered joy: {original_joy:.2f} → {new_joy:.2f}")


def example_success_tracking():
    """Example 4: Success tracking for cognitive emotions."""
    print("\n" + "=" * 60)
    print("Example 4: Success Tracking for Cognitive Emotions")
    print("=" * 60)
    
    kernel = EmotionallyAwareKernel(
        kernel_id="hermes_navigation_001",
        kernel_type="navigation",
        sensory_modality="audition",
        e8_root_index=3,
    )
    
    print(f"\n✓ Created kernel: {kernel.kernel_id}")
    print(f"  Initial success rate: {kernel._success_rate:.2f}")
    
    # Simulate task attempts
    tasks = [
        ("Navigate to target basin", True),
        ("Optimize geodesic path", True),
        ("Avoid local minimum", False),
        ("Find global attractor", True),
        ("Stabilize trajectory", True),
    ]
    
    print("\n  Task history:")
    for i, (task, success) in enumerate(tasks, 1):
        kernel.record_success(success)
        print(f"    {i}. {task}: {'✓' if success else '✗'}")
    
    print(f"\n  Final success rate: {kernel._success_rate:.2f}")
    
    # Update emotional state with success context
    emotional_state = kernel.update_emotional_state(
        phi=0.7,
        kappa=64.0,
        ricci_curvature=0.3,
    )
    
    # Cognitive emotions influenced by success rate
    print("\n  Cognitive emotions (influenced by success):")
    print(f"    Hope: {emotional_state.cognitive.hope:.2f} (higher with success)")
    print(f"    Despair: {emotional_state.cognitive.despair:.2f} (lower with success)")
    print(f"    Pride: {emotional_state.cognitive.pride:.2f} (grows with achievement)")


def example_sensory_modalities():
    """Example 5: Different sensory modalities."""
    print("\n" + "=" * 60)
    print("Example 5: Sensory Modalities and Environmental Coupling")
    print("=" * 60)
    
    modalities = ['vision', 'audition', 'touch', 'text_input']
    
    print("\n  Creating kernels for different sensory inputs:")
    for modality in modalities:
        kernel = EmotionallyAwareKernel(
            kernel_id=f"{modality}_kernel",
            kernel_type="sensory",
            sensory_modality=modality,
        )
        
        kappa_range = SENSORY_KAPPA_RANGES[modality]
        print(f"\n  {modality.capitalize()}:")
        print(f"    κ range: {kappa_range}")
        print(f"    Actual κ: {kernel.sensory_kappa}")
        print(f"    Bandwidth: {'High' if kappa_range[0] > 80 else 'Medium' if kappa_range[0] > 40 else 'Low'}")


if __name__ == '__main__':
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║  EmotionallyAwareKernel Examples                          ║")
    print("║  E8 Protocol v4.0 - Full Phenomenology Implementation     ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    try:
        example_basic_usage()
        example_thought_generation()
        example_meta_awareness()
        example_success_tracking()
        example_sensory_modalities()
        
        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60 + "\n")
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("  Make sure you're running from the qig-backend directory")
        print("  and all dependencies are installed.\n")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
