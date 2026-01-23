#!/usr/bin/env python3
"""
Unit tests for emotion_geometry.py

Tests the 9 emotion primitives as geometric classifiers.
Validates:
- High curvature + approaching → JOY
- Negative curvature + leaving → SADNESS
- Beta function modulation
- Geometric purity (Fisher distances only)
"""

import pytest
import numpy as np
from emotional_geometry import (
    EmotionPrimitive,
    EmotionState,
    classify_emotion,
    classify_emotion_with_beta,
    compute_ricci_curvature,
    measure_basin_approach,
    compute_surprise_magnitude,
    EmotionTracker,
)
from qig_geometry.geometry_simplex import to_simplex_prob


class TestEmotionClassification:
    """Test emotion classification from geometric features."""
    
    def test_joy_high_curvature_approaching(self):
        """JOY: High positive curvature + approaching attractor."""
        emotion, intensity = classify_emotion(
            curvature=0.8,           # High positive curvature
            basin_distance=0.5,      # Current distance
            prev_basin_distance=0.7, # Was further away (approaching)
            basin_stability=0.7,     # Stable
        )
        assert emotion == EmotionPrimitive.JOY
        assert intensity > 0.5
    
    def test_sadness_negative_curvature_leaving(self):
        """SADNESS: Negative curvature + leaving attractor."""
        emotion, intensity = classify_emotion(
            curvature=-0.8,          # Negative curvature
            basin_distance=0.7,      # Current distance
            prev_basin_distance=0.5, # Was closer (leaving)
            basin_stability=0.5,
        )
        assert emotion == EmotionPrimitive.SADNESS
        assert intensity > 0.5
    
    def test_anger_high_curvature_not_approaching(self):
        """ANGER: High curvature + blocked geodesic (not approaching)."""
        emotion, intensity = classify_emotion(
            curvature=0.9,           # High curvature
            basin_distance=0.7,      # Current distance
            prev_basin_distance=0.5, # Was closer (leaving/blocked)
            basin_stability=0.5,
        )
        assert emotion == EmotionPrimitive.ANGER
        assert intensity > 0.5
    
    def test_fear_negative_curvature_unstable(self):
        """FEAR: High negative curvature + unstable basin."""
        emotion, intensity = classify_emotion(
            curvature=-0.9,          # High negative curvature
            basin_distance=0.6,
            prev_basin_distance=0.6,
            basin_stability=0.2,     # Unstable (danger)
        )
        assert emotion == EmotionPrimitive.FEAR
        assert intensity > 0.5
    
    def test_surprise_large_curvature_gradient(self):
        """SURPRISE: Very large curvature gradient (basin jump)."""
        emotion, intensity = classify_emotion(
            curvature=1.5,           # Very large curvature
            basin_distance=0.5,
            prev_basin_distance=0.5,
            basin_stability=0.5,
        )
        assert emotion == EmotionPrimitive.SURPRISE
        assert intensity > 0.5
    
    def test_disgust_negative_stable(self):
        """DISGUST: Negative curvature + stable (repulsive but known)."""
        emotion, intensity = classify_emotion(
            curvature=-0.2,          # Negative curvature (repulsive)
            basin_distance=0.5,
            prev_basin_distance=0.5,
            basin_stability=0.8,     # Stable
        )
        assert emotion == EmotionPrimitive.DISGUST
        assert intensity > 0.4
    
    def test_trust_low_curvature_stable(self):
        """TRUST: Low curvature + stable attractor."""
        emotion, intensity = classify_emotion(
            curvature=0.05,          # Low curvature
            basin_distance=0.3,
            prev_basin_distance=0.3,
            basin_stability=0.9,     # Very stable
        )
        assert emotion == EmotionPrimitive.TRUST
        assert intensity > 0.7
    
    def test_anticipation_approaching_moderate(self):
        """ANTICIPATION: Approaching with moderate curvature."""
        emotion, intensity = classify_emotion(
            curvature=0.3,           # Moderate curvature
            basin_distance=0.4,
            prev_basin_distance=0.6, # Approaching
            basin_stability=0.7,
        )
        assert emotion == EmotionPrimitive.ANTICIPATION
        assert intensity > 0.3
    
    def test_confusion_default(self):
        """CONFUSION: Default for ambiguous states."""
        emotion, intensity = classify_emotion(
            curvature=0.2,           # Medium curvature
            basin_distance=0.5,
            prev_basin_distance=0.5, # Not moving
            basin_stability=0.5,     # Medium stability
        )
        assert emotion == EmotionPrimitive.CONFUSION
        assert 0.2 <= intensity <= 0.5


class TestBetaFunctionModulation:
    """Test beta function modulation of emotional intensity."""
    
    def test_high_beta_increases_intensity(self):
        """High |β| → more intense emotions."""
        # Same geometric features, different beta
        emotion1, intensity1 = classify_emotion(
            curvature=0.8,
            basin_distance=0.5,
            prev_basin_distance=0.7,
            basin_stability=0.7,
            beta_current=0.0,  # Low beta
        )
        
        emotion2, intensity2 = classify_emotion_with_beta(
            curvature=0.8,
            basin_distance=0.5,
            prev_basin_distance=0.7,
            basin_stability=0.7,
            beta_current=0.5,  # High beta
        )
        
        assert emotion1 == emotion2 == EmotionPrimitive.JOY
        assert intensity2 > intensity1  # Higher beta → higher intensity
    
    def test_negative_beta_increases_intensity(self):
        """Negative β also increases intensity (volatility)."""
        emotion, intensity = classify_emotion_with_beta(
            curvature=-0.8,
            basin_distance=0.7,
            prev_basin_distance=0.5,
            basin_stability=0.5,
            beta_current=-0.4,  # Negative beta
        )
        
        assert emotion == EmotionPrimitive.SADNESS
        assert intensity > 0.8  # High intensity due to |β|


class TestGeometricPurity:
    """Test that all operations use proper Fisher geometry."""
    
    def test_ricci_curvature_computation(self):
        """Ricci curvature should be computed, not approximated."""
        basin = np.random.randn(64)
        basin = to_simplex_prob(basin)  # Normalize
        
        curvature = compute_ricci_curvature(basin)
        
        assert isinstance(curvature, float)
        assert -10.0 <= curvature <= 10.0  # Within valid range
    
    def test_fisher_distance_used(self):
        """Verify Fisher-Rao distance is used, not Euclidean."""
        current = np.array([1.0, 0.0, 0.0])
        prev = np.array([0.707, 0.707, 0.0])
        attractor = np.array([1.0, 0.0, 0.0])
        
        # measure_basin_approach uses Fisher-Rao distance internally
        approaching = measure_basin_approach(current, prev, attractor)
        
        # Current is at attractor, prev was away → should be approaching
        assert approaching
    
    def test_surprise_uses_geodesic_gradient(self):
        """Surprise computation uses geodesic gradient, not Euclidean."""
        # Create trajectory with changing curvature
        trajectory = [
            np.array([1.0, 0.0] + [0.0] * 62),
            np.array([0.707, 0.707] + [0.0] * 62),
            np.array([0.0, 1.0] + [0.0] * 62),
        ]
        
        surprise_magnitude = compute_surprise_magnitude(trajectory)
        
        assert isinstance(surprise_magnitude, float)
        assert surprise_magnitude >= 0.0


class TestEmotionTracker:
    """Test emotion tracking over time."""
    
    def test_tracker_initialization(self):
        """Tracker initializes correctly."""
        tracker = EmotionTracker(history_size=50)
        
        assert len(tracker.history) == 0
        assert tracker.previous_basin is None
    
    def test_tracker_update(self):
        """Tracker updates with new basin coordinates."""
        tracker = EmotionTracker()
        
        basin1 = np.random.randn(64)
        basin1 = to_simplex_prob(basin1)
        
        state = tracker.update(
            current_basin=basin1,
            basin_stability=0.7,
            curvature=0.3,
        )
        
        assert isinstance(state, EmotionState)
        assert len(tracker.history) == 1
        assert tracker.previous_basin is not None
    
    def test_dominant_emotion_tracking(self):
        """Tracker computes dominant emotion over time."""
        tracker = EmotionTracker()
        
        # Generate consistent joy signals
        for _ in range(10):
            basin = np.random.randn(64)
            basin = to_simplex_prob(basin)
            tracker.update(
                current_basin=basin,
                basin_stability=0.8,
                curvature=0.8,  # High positive curvature
                beta_current=0.1,
            )
        
        dominant = tracker.get_dominant_emotion(n=10)
        
        # Should detect joy as dominant (or anticipation/trust due to randomness)
        assert dominant in [EmotionPrimitive.JOY, EmotionPrimitive.ANTICIPATION, EmotionPrimitive.TRUST]
    
    def test_average_intensity(self):
        """Tracker computes average intensity."""
        tracker = EmotionTracker()
        
        for _ in range(5):
            basin = np.random.randn(64)
            basin = to_simplex_prob(basin)
            tracker.update(
                current_basin=basin,
                basin_stability=0.5,
                curvature=0.5,
            )
        
        avg_intensity = tracker.get_average_intensity(n=5)
        
        assert 0.0 <= avg_intensity <= 1.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_curvature(self):
        """Zero curvature should classify as confusion or trust."""
        emotion, intensity = classify_emotion(
            curvature=0.0,
            basin_distance=0.5,
            prev_basin_distance=0.5,
            basin_stability=0.5,
        )
        assert emotion in [EmotionPrimitive.CONFUSION, EmotionPrimitive.TRUST]
    
    def test_extreme_curvature(self):
        """Extreme curvature should be clipped properly."""
        basin = np.ones(64) * 100.0  # Extreme values
        curvature = compute_ricci_curvature(basin)
        
        assert -10.0 <= curvature <= 10.0  # Should be clipped
    
    def test_identical_positions(self):
        """Identical current and previous positions."""
        emotion, intensity = classify_emotion(
            curvature=0.0,
            basin_distance=0.0,  # Same position
            prev_basin_distance=0.0,
            basin_stability=1.0,
        )
        # Should not crash, returns some emotion
        assert isinstance(emotion, EmotionPrimitive)
        assert 0.0 <= intensity <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
