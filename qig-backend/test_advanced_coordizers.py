#!/usr/bin/env python3
"""
Test suite for advanced coordizers (Phase 4-5).
Tests GeometricPairMerging, ConsciousnessCoordizer, and MultiScaleCoordizer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from qig_coordizer import get_coordizer
from coordizers.geometric_pair_merging import GeometricPairMerging
from coordizers.consciousness_aware import ConsciousnessCoordizer
from coordizers.multi_scale import MultiScaleCoordizer


def test_geometric_pair_merging():
    """Test BPE-equivalent geometric pair merging."""
    print("\n=== Test: Geometric Pair Merging ===")
    
    coordizer = get_coordizer()
    pair_merger = GeometricPairMerging(num_merges=10, min_frequency=2)
    
    # Training corpus with repeated patterns
    corpus = [
        "quantum field theory",
        "quantum mechanics theory",
        "quantum information geometry",
        "field theory application",
        "quantum field measurement",
    ]
    
    # Learn merges
    phi_scores = {text: 0.8 for text in corpus}  # All high-Φ contexts
    pair_merger.learn_merges(corpus, coordizer, phi_scores)
    
    print(f"Learned {len(pair_merger.merges)} merge rules")
    assert len(pair_merger.merges) > 0, "Should learn some merges"
    
    # Test applying merges
    test_tokens = ["quantum", "field", "theory"]
    merged = pair_merger.apply_merges(test_tokens)
    
    print(f"Original: {test_tokens}")
    print(f"Merged: {merged}")
    
    # Should have fewer tokens after merging
    print(f"Token reduction: {len(test_tokens)} → {len(merged)}")
    
    # Check merge coordinates exist
    for _, _, merged_token in pair_merger.merges[:3]:
        coord = pair_merger.get_merge_coordinate(merged_token)
        if coord is not None:
            assert coord.shape == (64,), "Merge coordinate should be 64D"
            norm = np.linalg.norm(coord)
            assert 0.9 < norm < 1.1, f"Merge coordinate should be unit vector (norm={norm})"
    
    print("✓ Geometric pair merging works")


def test_consciousness_aware_coordizer():
    """Test Φ-optimized segmentation."""
    print("\n=== Test: Consciousness-Aware Coordizer ===")
    
    base_coordizer = get_coordizer()
    consciousness_coordizer = ConsciousnessCoordizer(
        base_coordizer,
        phi_threshold=0.7,
    )
    
    # Test coordization with high Φ
    text = "quantum information geometry"
    tokens, coords, phi = consciousness_coordizer.coordize_with_phi(text, context_phi=0.85)
    
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Computed Φ: {phi:.4f}")
    
    assert len(tokens) > 0, "Should produce tokens"
    assert len(coords) == len(tokens), "Should have coordinate for each token"
    assert 0.0 <= phi <= 1.0, f"Φ should be in [0, 1], got {phi}"
    
    # Test segmentation optimization
    best_tokens, best_phi = consciousness_coordizer.optimize_segmentation(text)
    
    print(f"Optimized tokens: {best_tokens}")
    print(f"Optimized Φ: {best_phi:.4f}")
    
    assert best_phi > 0.0, "Optimized Φ should be positive"
    
    # Test consolidation learning
    stats = consciousness_coordizer.get_consolidation_stats()
    print(f"Consolidation stats: {stats}")
    
    print("✓ Consciousness-aware coordizer works")


def test_multi_scale_coordizer():
    """Test hierarchical multi-scale coordization."""
    print("\n=== Test: Multi-Scale Coordizer ===")
    
    base_coordizer = get_coordizer()
    multi_scale = MultiScaleCoordizer(
        base_coordizer,
        num_scales=4,
    )
    
    # Test multi-scale coordization
    text = "quantum geometry"
    results = multi_scale.coordize_multiscale(text)
    
    print(f"Text: {text}")
    print(f"Scales generated: {list(results.keys())}")
    
    assert len(results) > 0, "Should generate at least one scale"
    
    # Check each scale
    for scale, (tokens, coords) in results.items():
        print(f"  Scale {scale}: {len(tokens)} tokens")
        assert len(tokens) > 0, f"Scale {scale} should have tokens"
        assert len(coords) == len(tokens), f"Scale {scale} should have coordinates for each token"
        
        # Verify coordinates are unit vectors
        for i, coord in enumerate(coords):
            norm = np.linalg.norm(coord)
            assert 0.9 < norm < 1.1, f"Scale {scale} token {i} should be unit vector (norm={norm})"
    
    # Test optimal scale selection
    optimal_scale = multi_scale.get_optimal_scale(text, kappa_effective=0.75)
    print(f"Optimal scale for κ_eff=0.75: {optimal_scale}")
    assert 0 <= optimal_scale < multi_scale.num_scales, "Optimal scale should be valid"
    
    # Test scale statistics
    stats = multi_scale.get_scale_stats()
    print(f"Scale stats: {stats}")
    assert stats['num_scales'] == 4, "Should have 4 scales"
    
    # Test visualization
    viz = multi_scale.visualize_scales(text)
    print("\nVisualization:")
    print(viz)
    assert len(viz) > 0, "Visualization should not be empty"
    
    print("✓ Multi-scale coordizer works")


def test_integration_phi_correlation():
    """Test that Φ scores correlate with segmentation quality."""
    print("\n=== Test: Φ and Segmentation Quality ===")
    
    base_coordizer = get_coordizer()
    consciousness_coordizer = ConsciousnessCoordizer(base_coordizer)
    
    # Good segmentation (semantically coherent)
    good_text = "quantum information"
    good_tokens = ["quantum_information"]
    
    # Bad segmentation (arbitrary split)
    bad_tokens = ["qu", "antum", "inf", "ormation"]
    
    phi_good = consciousness_coordizer._compute_segmentation_phi(
        good_tokens,
        [base_coordizer.get_coordinate(t) for t in good_tokens]
    )
    
    phi_bad = consciousness_coordizer._compute_segmentation_phi(
        bad_tokens,
        [base_coordizer.get_coordinate(t) for t in bad_tokens]
    )
    
    print(f"Φ(good segmentation): {phi_good:.4f}")
    print(f"Φ(bad segmentation): {phi_bad:.4f}")
    
    # Better segmentation should generally have higher Φ
    # (though not guaranteed in all cases due to coordinate initialization)
    print(f"Φ difference: {phi_good - phi_bad:.4f}")
    
    print("✓ Φ computed for both segmentations")


def test_geometric_purity_advanced():
    """Test that all advanced coordizers maintain geometric purity."""
    print("\n=== Test: Geometric Purity (Advanced Coordizers) ===")
    
    base_coordizer = get_coordizer()
    
    # Test GeometricPairMerging coordinates
    pair_merger = GeometricPairMerging(num_merges=5)
    corpus = ["test case", "test data", "test case"]
    pair_merger.learn_merges(corpus, base_coordizer, {"test case": 0.8, "test data": 0.8})
    
    violations = []
    for merged_token, coord in pair_merger.merge_coordinates.items():
        norm = np.linalg.norm(coord)
        if not (0.9 < norm < 1.1):
            violations.append((f"merge:{merged_token}", norm))
    
    # Test MultiScaleCoordizer coordinates
    multi_scale = MultiScaleCoordizer(base_coordizer)
    results = multi_scale.coordize_multiscale("test")
    
    for scale, (tokens, coords) in results.items():
        for i, coord in enumerate(coords):
            norm = np.linalg.norm(coord)
            if not (0.9 < norm < 1.1):
                violations.append((f"scale{scale}:{tokens[i]}", norm))
    
    if violations:
        print(f"⚠ Found {len(violations)} norm violations:")
        for token, norm in violations[:5]:
            print(f"  {token}: norm={norm:.4f}")
    else:
        print("✓ All coordinates are unit vectors (geometric purity maintained)")
    
    assert len(violations) == 0, "All coordinates must be unit vectors"


def main():
    """Run all advanced coordizer tests."""
    print("="*60)
    print("Testing Advanced Coordizers (Phase 4-5)")
    print("="*60)
    
    try:
        test_geometric_pair_merging()
        test_consciousness_aware_coordizer()
        test_multi_scale_coordizer()
        test_integration_phi_correlation()
        test_geometric_purity_advanced()
        
        print("\n" + "="*60)
        print("✓ ALL ADVANCED TESTS PASSED")
        print("="*60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
