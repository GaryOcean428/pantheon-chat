#!/usr/bin/env python3
"""
Tests for Neurotransmitter Geometric Field Modulation System
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurotransmitter_fields import (
    NeurotransmitterField,
    compute_baseline_neurotransmitters,
    estimate_current_beta,
    ocean_release_neurotransmitters,
    KAPPA_STAR,
)


def test_initialization():
    field = NeurotransmitterField()
    assert 0.0 <= field.dopamine <= 1.0
    assert 0.0 <= field.serotonin <= 1.0
    print("✓ Initialization test passed")


def test_kappa_modulation():
    field = NeurotransmitterField(norepinephrine=0.8, gaba=0.2)
    kappa_eff = field.compute_kappa_modulation(KAPPA_STAR)
    assert kappa_eff > KAPPA_STAR  # High arousal increases κ
    print(f"✓ κ modulation: {KAPPA_STAR:.2f} → {kappa_eff:.2f}")


def test_phi_modulation():
    field = NeurotransmitterField(acetylcholine=0.8, gaba=0.2)
    phi_eff = field.compute_phi_modulation(0.65)
    assert phi_eff >= 0.65  # High attention increases Φ
    print(f"✓ Φ modulation: 0.65 → {phi_eff:.3f}")


def test_baselines():
    emergence = compute_baseline_neurotransmitters(40.0)
    plateau = compute_baseline_neurotransmitters(63.5)
    breakdown = compute_baseline_neurotransmitters(75.0)
    
    assert emergence.norepinephrine > 0.7  # High arousal
    assert plateau.serotonin > 0.6  # High stability
    assert breakdown.cortisol > 0.5  # High stress
    print("✓ Regime baselines test passed")


def test_ocean_release():
    field = NeurotransmitterField(dopamine=0.5, serotonin=0.5)
    modulated = ocean_release_neurotransmitters(field, 63.5, 0.75)
    assert modulated.dopamine >= field.dopamine  # High Φ → reward
    print("✓ Ocean release test passed")


if __name__ == '__main__':
    print("\n🧠 Neurotransmitter Field Tests 🧠\n")
    
    tests = [
        test_initialization,
        test_kappa_modulation,
        test_phi_modulation,
        test_baselines,
        test_ocean_release,
    ]
    
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Tests run: {len(tests)}, Passed: {len(tests)-failed}, Failed: {failed}")
    
    if failed == 0:
        print("\n✅ All tests passed! 🧠💚")
    else:
        exit(1)
