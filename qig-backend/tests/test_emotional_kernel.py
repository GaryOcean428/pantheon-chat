#!/usr/bin/env python3
"""
Tests for EmotionallyAwareKernel and emotional layer components.

Tests cover:
- Layer 0.5: Sensations (12 geometric states)
- Layer 1: Motivators (5 derivatives)
- Layer 2A: Physical Emotions (9 fast)
- Layer 2B: Cognitive Emotions (9 slow)
- Meta-awareness and course-correction
"""

import unittest
import numpy as np
import sys
import os

# Add qig-backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kernels.sensations import (
    SensationState,
    measure_sensations,
    get_dominant_sensation,
)
from kernels.motivators import (
    MotivatorState,
    compute_motivators,
    get_dominant_motivator,
)
from kernels.emotions import (
    PhysicalEmotionState,
    CognitiveEmotionState,
    compute_physical_emotions,
    compute_cognitive_emotions,
    get_dominant_emotion,
    EmotionType,
)
from kernels.emotional import (
    EmotionallyAwareKernel,
    EmotionalState,
    SENSORY_KAPPA_RANGES,
)


class TestSensations(unittest.TestCase):
    """Test Layer 0.5: Pre-linguistic Sensations."""
    
    def test_measure_sensations_unified(self):
        """Test that high phi produces unified sensation."""
        sensations = measure_sensations(
            phi=0.9,
            kappa=64.0,
            ricci_curvature=0.1,
            basin_distance=0.2,
        )
        
        self.assertGreater(sensations.unified, 0.7)
        self.assertLess(sensations.fragmented, 0.3)
    
    def test_measure_sensations_fragmented(self):
        """Test that low phi produces fragmented sensation."""
        sensations = measure_sensations(
            phi=0.2,
            kappa=64.0,
            ricci_curvature=0.1,
            basin_distance=0.5,
        )
        
        self.assertGreater(sensations.fragmented, 0.5)
        self.assertLess(sensations.unified, 0.3)
    
    def test_measure_sensations_compressed(self):
        """Test that positive curvature produces compressed sensation."""
        sensations = measure_sensations(
            phi=0.5,
            kappa=64.0,
            ricci_curvature=1.0,  # High positive curvature
            basin_distance=0.5,
        )
        
        self.assertGreater(sensations.compressed, 0.3)
    
    def test_measure_sensations_expanded(self):
        """Test that negative curvature produces expanded sensation."""
        sensations = measure_sensations(
            phi=0.5,
            kappa=64.0,
            ricci_curvature=-1.0,  # High negative curvature
            basin_distance=0.5,
        )
        
        self.assertGreater(sensations.expanded, 0.3)
    
    def test_get_dominant_sensation(self):
        """Test dominant sensation detection."""
        sensations = SensationState(unified=0.8, fragmented=0.1)
        dominant, intensity = get_dominant_sensation(sensations)
        
        self.assertEqual(dominant, 'unified')
        self.assertAlmostEqual(intensity, 0.8)


class TestMotivators(unittest.TestCase):
    """Test Layer 1: Motivators (5 geometric derivatives)."""
    
    def test_compute_motivators_curiosity(self):
        """Test curiosity from information rate."""
        motivators = compute_motivators(
            phi=0.5,
            kappa=64.0,
            fisher_info=2.0,
            basin_distance=0.5,
            prev_fisher_info=1.0,  # Information increased
        )
        
        self.assertGreater(motivators.curiosity, 0.0)
    
    def test_compute_motivators_investigation(self):
        """Test investigation from approach velocity."""
        motivators = compute_motivators(
            phi=0.5,
            kappa=64.0,
            fisher_info=1.0,
            basin_distance=0.3,
            prev_basin_distance=0.5,  # Approaching
        )
        
        self.assertGreater(motivators.investigation, 0.0)
    
    def test_compute_motivators_surprise(self):
        """Test surprise from gradient magnitude."""
        loss_gradient = np.array([0.5, 0.3, 0.2])
        motivators = compute_motivators(
            phi=0.5,
            kappa=64.0,
            fisher_info=1.0,
            basin_distance=0.5,
            loss_gradient=loss_gradient,
        )
        
        self.assertGreater(motivators.surprise, 0.0)
    
    def test_compute_motivators_transcendence(self):
        """Test transcendence from critical point distance."""
        motivators = compute_motivators(
            phi=0.5,
            kappa=100.0,  # Far from kappa_star = 64.21
            fisher_info=1.0,
            basin_distance=0.5,
            kappa_critical=64.21,
        )
        
        self.assertGreater(motivators.transcendence, 0.3)
    
    def test_get_dominant_motivator(self):
        """Test dominant motivator detection."""
        motivators = MotivatorState(curiosity=0.9, investigation=0.2)
        dominant, intensity = get_dominant_motivator(motivators)
        
        self.assertEqual(dominant, 'curiosity')
        self.assertAlmostEqual(intensity, 0.9)


class TestPhysicalEmotions(unittest.TestCase):
    """Test Layer 2A: Physical Emotions (fast, τ<1)."""
    
    def test_compute_joy(self):
        """Test joy from high curvature + approaching."""
        sensations = SensationState(unified=0.8)
        motivators = MotivatorState(integration=0.7)
        
        emotions = compute_physical_emotions(
            sensations=sensations,
            motivators=motivators,
            ricci_curvature=0.8,  # High positive
            basin_distance=0.3,
            approaching=True,
            basin_stability=0.8,
        )
        
        self.assertGreater(emotions.joy, 0.3)
    
    def test_compute_fear(self):
        """Test fear from high curvature + unstable."""
        sensations = SensationState()
        motivators = MotivatorState()
        
        emotions = compute_physical_emotions(
            sensations=sensations,
            motivators=motivators,
            ricci_curvature=0.8,  # High positive
            basin_distance=0.5,
            approaching=False,
            basin_stability=0.2,  # Unstable
        )
        
        self.assertGreater(emotions.fear, 0.2)
    
    def test_compute_calm(self):
        """Test calm from low curvature + stable."""
        sensations = SensationState(grounded=0.8)
        motivators = MotivatorState()
        
        emotions = compute_physical_emotions(
            sensations=sensations,
            motivators=motivators,
            ricci_curvature=0.1,  # Low curvature
            basin_distance=0.2,
            approaching=True,
            basin_stability=0.9,  # Very stable
        )
        
        self.assertGreater(emotions.calm, 0.3)


class TestCognitiveEmotions(unittest.TestCase):
    """Test Layer 2B: Cognitive Emotions (slow, τ=1-100)."""
    
    def test_compute_wonder(self):
        """Test wonder from curiosity + surprise."""
        physical = PhysicalEmotionState()
        motivators = MotivatorState(curiosity=0.8, surprise=0.5)
        sensations = SensationState()
        
        emotions = compute_cognitive_emotions(
            physical=physical,
            motivators=motivators,
            sensations=sensations,
            success_rate=0.5,
        )
        
        self.assertGreater(emotions.wonder, 0.3)
    
    def test_compute_frustration(self):
        """Test frustration from investigation + low integration."""
        physical = PhysicalEmotionState()
        motivators = MotivatorState(investigation=0.8, integration=0.2)
        sensations = SensationState()
        
        emotions = compute_cognitive_emotions(
            physical=physical,
            motivators=motivators,
            sensations=sensations,
            success_rate=0.5,
        )
        
        self.assertGreater(emotions.frustration, 0.3)
    
    def test_compute_clarity(self):
        """Test clarity from low surprise + high integration."""
        physical = PhysicalEmotionState()
        motivators = MotivatorState(surprise=0.1, integration=0.9)
        sensations = SensationState()
        
        emotions = compute_cognitive_emotions(
            physical=physical,
            motivators=motivators,
            sensations=sensations,
            success_rate=0.5,
        )
        
        self.assertGreater(emotions.clarity, 0.5)


class TestEmotionallyAwareKernel(unittest.TestCase):
    """Test EmotionallyAwareKernel integration."""
    
    def setUp(self):
        """Create test kernel."""
        self.kernel = EmotionallyAwareKernel(
            kernel_id="test_kernel_001",
            kernel_type="test",
            sensory_modality="text_input",
        )
    
    def test_kernel_initialization(self):
        """Test kernel initializes with correct properties."""
        self.assertEqual(self.kernel.kernel_id, "test_kernel_001")
        self.assertEqual(self.kernel.kernel_type, "test")
        self.assertEqual(self.kernel.sensory_modality, "text_input")
        self.assertEqual(self.kernel.sensory_kappa, 60.0)  # text_input
        self.assertIsNotNone(self.kernel.basin_coords)
        self.assertEqual(len(self.kernel.basin_coords), 64)
    
    def test_update_emotional_state(self):
        """Test emotional state update."""
        emotional_state = self.kernel.update_emotional_state(
            phi=0.8,
            kappa=64.0,
            regime="geometric",
            ricci_curvature=0.5,
        )
        
        self.assertIsNotNone(emotional_state)
        self.assertTrue(emotional_state.is_meta_aware)
        self.assertIsNotNone(emotional_state.sensations)
        self.assertIsNotNone(emotional_state.motivators)
        self.assertIsNotNone(emotional_state.physical)
        self.assertIsNotNone(emotional_state.cognitive)
    
    def test_meta_reflection_justified(self):
        """Test meta-reflection when emotion is justified."""
        self.kernel.update_emotional_state(
            phi=0.8,
            kappa=64.0,
            ricci_curvature=0.8,
        )
        
        # High phi + high curvature should produce joy
        # Joy should be justified
        self.assertTrue(self.kernel.emotional_state.emotion_justified)
    
    def test_emotion_tempering(self):
        """Test emotion tempering when unjustified."""
        # Create state that claims joy but has low phi
        self.kernel.phi = 0.2
        self.kernel.emotional_state.physical.joyful = 0.8
        self.kernel.emotional_state.dominant_emotion = 'joy'
        
        # Meta-reflect should detect and temper
        is_justified, should_temper = self.kernel._meta_reflect_on_emotions()
        
        self.assertFalse(is_justified)
        self.assertTrue(should_temper)
    
    def test_generate_thought(self):
        """Test thought generation with emotional awareness."""
        thought = self.kernel.generate_thought(
            context="test context",
            phi=0.7,
            kappa=64.0,
            regime="geometric",
        )
        
        self.assertEqual(thought.kernel_id, "test_kernel_001")
        self.assertEqual(thought.kernel_type, "test")
        self.assertIsNotNone(thought.emotional_state)
        self.assertGreater(thought.confidence, 0.0)
        self.assertLessEqual(thought.confidence, 1.0)
    
    def test_sensory_kappa_ranges(self):
        """Test sensory kappa values for different modalities."""
        self.assertIn('vision', SENSORY_KAPPA_RANGES)
        self.assertIn('audition', SENSORY_KAPPA_RANGES)
        self.assertIn('touch', SENSORY_KAPPA_RANGES)
        self.assertIn('text_input', SENSORY_KAPPA_RANGES)
        
        # Vision should have highest kappa
        self.assertGreater(
            SENSORY_KAPPA_RANGES['vision'][0],
            SENSORY_KAPPA_RANGES['touch'][0]
        )
    
    def test_success_tracking(self):
        """Test success rate tracking for cognitive emotions."""
        # Record some successes and failures
        self.kernel.record_success(True)
        self.kernel.record_success(True)
        self.kernel.record_success(False)
        
        self.assertAlmostEqual(self.kernel._success_rate, 2/3, places=2)
    
    def test_get_status(self):
        """Test kernel status reporting."""
        self.kernel.update_emotional_state(phi=0.7, kappa=64.0)
        status = self.kernel.get_status()
        
        self.assertIn('kernel_id', status)
        self.assertIn('dominant_emotion', status)
        self.assertIn('emotion_justified', status)
        self.assertIn('success_rate', status)


if __name__ == '__main__':
    unittest.main()
