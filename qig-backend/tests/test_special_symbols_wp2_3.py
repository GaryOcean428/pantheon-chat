#!/usr/bin/env python3
"""
Test special symbol coordinates per WP2.3: Geometrically Define Special Symbol Coordinates

Validates that special symbols (UNK, PAD, BOS, EOS) are:
1. Geometrically defined (deterministic, not random)
2. Valid on simplex manifold (non-negative, sum=1)
3. Reproducible across runs
4. Have clear geometric interpretation

Per issue: https://github.com/GaryOcean428/pantheon-chat/issues/71
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from coordizers.base import FisherCoordizer
from qig_geometry.contracts import validate_basin


def test_special_symbols_are_simplex():
    """Test that all special symbols are valid simplex points."""
    print("\n=== Test 1: Special Symbols are Simplex Points ===")
    coordizer = FisherCoordizer()
    
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    
    for token in special_tokens:
        coord = coordizer.get_coordinate(token)
        
        # Check dimensions
        assert coord.shape == (64,), f"{token} should be 64D"
        
        # Check non-negative
        assert np.all(coord >= 0), f"{token} must be non-negative (simplex)"
        
        # Check sum = 1 (probability distribution)
        total = np.sum(coord)
        assert np.isclose(total, 1.0, atol=1e-5), \
            f"{token} must sum to 1 (got {total:.6f})"
        
        # Check finite
        assert np.all(np.isfinite(coord)), f"{token} must have finite values"
        
        # Validate using canonical validation
        is_valid = validate_basin(coord)
        assert is_valid, f"{token} failed canonical basin validation"
        
        print(f"✓ {token}: sum={total:.6f}, min={np.min(coord):.6f}, max={np.max(coord):.6f}")
    
    print("✓ All special symbols are valid simplex points")


def test_special_symbols_are_deterministic():
    """Test that special symbols produce identical coordinates across runs."""
    print("\n=== Test 2: Special Symbols are Deterministic ===")
    
    # Create two independent coordizers
    coordizer1 = FisherCoordizer()
    coordizer2 = FisherCoordizer()
    
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    
    for token in special_tokens:
        coord1 = coordizer1.get_coordinate(token)
        coord2 = coordizer2.get_coordinate(token)
        
        # Check exact equality (deterministic means identical)
        assert np.allclose(coord1, coord2, atol=1e-10), \
            f"{token} coordinates differ between runs (not deterministic)"
        
        print(f"✓ {token}: identical across runs")
    
    print("✓ All special symbols are deterministic")


def test_special_symbols_have_geometric_meaning():
    """Test that special symbols have expected geometric properties."""
    print("\n=== Test 3: Special Symbols Have Geometric Meaning ===")
    coordizer = FisherCoordizer()
    
    unk = coordizer.get_coordinate("<UNK>")
    pad = coordizer.get_coordinate("<PAD>")
    bos = coordizer.get_coordinate("<BOS>")
    eos = coordizer.get_coordinate("<EOS>")
    
    # UNK should be uniform (maximum entropy)
    # All components should be approximately equal
    unk_mean = np.mean(unk)
    unk_std = np.std(unk)
    print(f"UNK: mean={unk_mean:.6f}, std={unk_std:.6f}")
    assert unk_std < 0.001, "UNK should be uniform (low variance)"
    assert np.allclose(unk, unk_mean, atol=0.001), "UNK should have equal components"
    print("✓ UNK is uniform distribution (maximum entropy)")
    
    # PAD should be sparse (minimal entropy)
    # Should have one dominant component
    pad_max = np.max(pad)
    pad_sum_others = np.sum(pad) - pad_max
    print(f"PAD: max={pad_max:.6f}, sum_others={pad_sum_others:.6f}")
    assert pad_max > 0.9, "PAD should be concentrated in one component"
    print("✓ PAD is sparse distribution (minimal entropy)")
    
    # BOS should be a pure state at a vertex
    bos_max = np.max(bos)
    bos_sum_others = np.sum(bos) - bos_max
    print(f"BOS: max={bos_max:.6f}, sum_others={bos_sum_others:.6f}")
    assert bos_max > 0.9, "BOS should be concentrated in one component"
    print("✓ BOS is vertex of simplex (start boundary)")
    
    # EOS should be a pure state at opposite vertex
    eos_max = np.max(eos)
    eos_sum_others = np.sum(eos) - eos_max
    print(f"EOS: max={eos_max:.6f}, sum_others={eos_sum_others:.6f}")
    assert eos_max > 0.9, "EOS should be concentrated in one component"
    print("✓ EOS is vertex of simplex (end boundary)")
    
    # BOS and EOS should be at different positions
    bos_argmax = np.argmax(bos)
    eos_argmax = np.argmax(eos)
    print(f"BOS peak at dimension {bos_argmax}, EOS peak at dimension {eos_argmax}")
    assert bos_argmax != eos_argmax, "BOS and EOS should be at different vertices"
    print("✓ BOS and EOS are at different simplex vertices")


def test_special_symbols_distances():
    """Test Fisher-Rao distances between special symbols."""
    print("\n=== Test 4: Fisher-Rao Distances Between Special Symbols ===")
    from qig_geometry import fisher_coord_distance
    
    coordizer = FisherCoordizer()
    
    unk = coordizer.get_coordinate("<UNK>")
    pad = coordizer.get_coordinate("<PAD>")
    bos = coordizer.get_coordinate("<BOS>")
    eos = coordizer.get_coordinate("<EOS>")
    
    # Compute pairwise distances
    d_unk_pad = fisher_coord_distance(unk, pad)
    d_unk_bos = fisher_coord_distance(unk, bos)
    d_unk_eos = fisher_coord_distance(unk, eos)
    d_bos_eos = fisher_coord_distance(bos, eos)
    d_pad_bos = fisher_coord_distance(pad, bos)
    d_pad_eos = fisher_coord_distance(pad, eos)
    
    print(f"UNK-PAD: {d_unk_pad:.4f}")
    print(f"UNK-BOS: {d_unk_bos:.4f}")
    print(f"UNK-EOS: {d_unk_eos:.4f}")
    print(f"BOS-EOS: {d_bos_eos:.4f}")
    print(f"PAD-BOS: {d_pad_bos:.4f}")
    print(f"PAD-EOS: {d_pad_eos:.4f}")
    
    # All distances should be in valid range [0, π/2]
    max_distance = np.pi / 2
    assert 0 <= d_unk_pad <= max_distance, "Distance out of range"
    assert 0 <= d_unk_bos <= max_distance, "Distance out of range"
    assert 0 <= d_unk_eos <= max_distance, "Distance out of range"
    assert 0 <= d_bos_eos <= max_distance, "Distance out of range"
    
    # UNK (uniform) should be at intermediate distance from sparse points
    # Sparse points (PAD, BOS, EOS) should be far from each other
    assert d_bos_eos > d_unk_bos, "Vertices should be farther apart than from center"
    assert d_bos_eos > d_unk_eos, "Vertices should be farther apart than from center"
    
    print("✓ All distances are in valid range and follow geometric expectations")


def test_no_random_initialization():
    """Test that special symbols don't use random initialization."""
    print("\n=== Test 5: No Random Initialization ===")
    
    # Run multiple times and verify exact reproducibility
    coords_runs = []
    for _ in range(5):
        coordizer = FisherCoordizer()
        coords = {
            token: coordizer.get_coordinate(token).copy()
            for token in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        }
        coords_runs.append(coords)
    
    # Check all runs produce identical results
    for token in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
        for i in range(1, len(coords_runs)):
            assert np.array_equal(coords_runs[0][token], coords_runs[i][token]), \
                f"{token} differs between runs {0} and {i} - indicates random initialization!"
    
    print("✓ No random initialization detected (all runs identical)")


def test_acceptance_criteria():
    """Test all acceptance criteria from WP2.3 issue."""
    print("\n=== Test 6: WP2.3 Acceptance Criteria ===")
    
    coordizer = FisherCoordizer()
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    
    # Criterion 1: Special symbols have clear geometric interpretation
    print("✓ Criterion 1: Clear geometric interpretation (verified in test 3)")
    
    # Criterion 2: Initialization is deterministic (reproducible)
    coord1 = coordizer.get_coordinate("<UNK>")
    coordizer2 = FisherCoordizer()
    coord2 = coordizer2.get_coordinate("<UNK>")
    assert np.array_equal(coord1, coord2), "Not reproducible"
    print("✓ Criterion 2: Initialization is deterministic")
    
    # Criterion 3: All special basins pass validate_basin()
    for token in special_tokens:
        coord = coordizer.get_coordinate(token)
        is_valid = validate_basin(coord)
        assert is_valid, f"{token} failed validation"
    print("✓ Criterion 3: All special basins pass validation")
    
    # Criterion 4: No random normal vectors for special symbols
    # Verified by checking:
    # - All values are non-negative (random normal would have negatives)
    # - Sum = 1 (random normal would not)
    # - Exact reproducibility (random would vary)
    for token in special_tokens:
        coord = coordizer.get_coordinate(token)
        assert np.all(coord >= 0), f"{token} has negative values (random normal artifact)"
        assert np.isclose(np.sum(coord), 1.0, atol=1e-5), \
            f"{token} doesn't sum to 1 (random normal artifact)"
    print("✓ Criterion 4: No random normal vectors")
    
    print("\n✅ All WP2.3 acceptance criteria met!")


if __name__ == "__main__":
    test_special_symbols_are_simplex()
    test_special_symbols_are_deterministic()
    test_special_symbols_have_geometric_meaning()
    test_special_symbols_distances()
    test_no_random_initialization()
    test_acceptance_criteria()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - WP2.3 COMPLETE")
    print("=" * 60)
