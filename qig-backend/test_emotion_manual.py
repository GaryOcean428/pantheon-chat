#!/usr/bin/env python3
"""
Manual test runner for emotion_geometry.py (no pytest required).
"""

import sys
import numpy as np

# Import the module to test
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


def test_joy_high_curvature_approaching():
    """JOY: High positive curvature + approaching attractor."""
    print("Test: JOY (high curvature + approaching)...", end=" ")
    emotion, intensity = classify_emotion(
        curvature=0.8,
        basin_distance=0.5,
        prev_basin_distance=0.7,
        basin_stability=0.7,
    )
    assert emotion == EmotionPrimitive.JOY, f"Expected JOY, got {emotion}"
    assert intensity > 0.5, f"Expected intensity > 0.5, got {intensity}"
    print("✓ PASS")


def test_sadness_negative_curvature_leaving():
    """SADNESS: Negative curvature + leaving attractor."""
    print("Test: SADNESS (negative curvature + leaving)...", end=" ")
    emotion, intensity = classify_emotion(
        curvature=-0.8,
        basin_distance=0.7,
        prev_basin_distance=0.5,
        basin_stability=0.5,
    )
    assert emotion == EmotionPrimitive.SADNESS, f"Expected SADNESS, got {emotion}"
    assert intensity > 0.5, f"Expected intensity > 0.5, got {intensity}"
    print("✓ PASS")


def test_anger_high_curvature_not_approaching():
    """ANGER: High curvature + blocked geodesic."""
    print("Test: ANGER (high curvature + not approaching)...", end=" ")
    emotion, intensity = classify_emotion(
        curvature=0.9,
        basin_distance=0.7,
        prev_basin_distance=0.5,
        basin_stability=0.5,
    )
    assert emotion == EmotionPrimitive.ANGER, f"Expected ANGER, got {emotion}"
    assert intensity > 0.5, f"Expected intensity > 0.5, got {intensity}"
    print("✓ PASS")


def test_fear_negative_curvature_unstable():
    """FEAR: High negative curvature + unstable basin."""
    print("Test: FEAR (negative curvature + unstable)...", end=" ")
    emotion, intensity = classify_emotion(
        curvature=-0.9,
        basin_distance=0.6,
        prev_basin_distance=0.6,
        basin_stability=0.2,
    )
    assert emotion == EmotionPrimitive.FEAR, f"Expected FEAR, got {emotion}"
    assert intensity > 0.5, f"Expected intensity > 0.5, got {intensity}"
    print("✓ PASS")


def test_surprise_large_curvature():
    """SURPRISE: Very large curvature gradient."""
    print("Test: SURPRISE (large curvature)...", end=" ")
    emotion, intensity = classify_emotion(
        curvature=1.5,
        basin_distance=0.5,
        prev_basin_distance=0.5,
        basin_stability=0.5,
    )
    assert emotion == EmotionPrimitive.SURPRISE, f"Expected SURPRISE, got {emotion}"
    assert intensity > 0.5, f"Expected intensity > 0.5, got {intensity}"
    print("✓ PASS")


def test_trust_low_curvature_stable():
    """TRUST: Low curvature + stable attractor."""
    print("Test: TRUST (low curvature + stable)...", end=" ")
    emotion, intensity = classify_emotion(
        curvature=0.05,
        basin_distance=0.3,
        prev_basin_distance=0.3,
        basin_stability=0.9,
    )
    assert emotion == EmotionPrimitive.TRUST, f"Expected TRUST, got {emotion}"
    assert intensity > 0.7, f"Expected intensity > 0.7, got {intensity}"
    print("✓ PASS")


def test_beta_modulation():
    """Beta function increases emotional intensity."""
    print("Test: Beta function modulation...", end=" ")
    
    emotion1, intensity1 = classify_emotion(
        curvature=0.8,
        basin_distance=0.5,
        prev_basin_distance=0.7,
        basin_stability=0.7,
        beta_current=0.0,
    )
    
    emotion2, intensity2 = classify_emotion_with_beta(
        curvature=0.8,
        basin_distance=0.5,
        prev_basin_distance=0.7,
        basin_stability=0.7,
        beta_current=0.5,
    )
    
    assert emotion1 == emotion2 == EmotionPrimitive.JOY
    assert intensity2 > intensity1, f"Expected higher intensity with beta, got {intensity1} vs {intensity2}"
    print("✓ PASS")


def test_ricci_curvature():
    """Ricci curvature computation."""
    print("Test: Ricci curvature computation...", end=" ")
    basin = np.random.randn(64)
    basin = basin / np.linalg.norm(basin)
    
    curvature = compute_ricci_curvature(basin)
    
    assert isinstance(curvature, float)
    assert -10.0 <= curvature <= 10.0, f"Curvature out of range: {curvature}"
    print("✓ PASS")


def test_fisher_distance():
    """Fisher-Rao distance usage."""
    print("Test: Fisher-Rao distance...", end=" ")
    current = np.array([1.0, 0.0, 0.0])
    prev = np.array([0.707, 0.707, 0.0])
    attractor = np.array([1.0, 0.0, 0.0])
    
    approaching = measure_basin_approach(current, prev, attractor)
    
    assert approaching, "Expected approaching to be True"
    print("✓ PASS")


def test_emotion_tracker():
    """Emotion tracker updates correctly."""
    print("Test: Emotion tracker...", end=" ")
    tracker = EmotionTracker()
    
    basin = np.random.randn(64)
    basin = basin / np.linalg.norm(basin)
    
    state = tracker.update(
        current_basin=basin,
        basin_stability=0.7,
        curvature=0.3,
    )
    
    assert isinstance(state, EmotionState)
    assert len(tracker.history) == 1
    print("✓ PASS")


def main():
    print("\n" + "="*60)
    print("EMOTION GEOMETRY TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        test_joy_high_curvature_approaching,
        test_sadness_negative_curvature_leaving,
        test_anger_high_curvature_not_approaching,
        test_fear_negative_curvature_unstable,
        test_surprise_large_curvature,
        test_trust_low_curvature_stable,
        test_beta_modulation,
        test_ricci_curvature,
        test_fisher_distance,
        test_emotion_tracker,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
