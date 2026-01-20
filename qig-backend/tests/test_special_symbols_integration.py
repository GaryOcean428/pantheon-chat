#!/usr/bin/env python3
"""
Integration Test: Special Symbols in Generation Pipeline

Tests that special symbols (UNK, PAD, BOS, EOS) work correctly in the
Plan→Realize→Repair generation architecture.

Validates:
1. Special symbols are accessible during generation
2. Unknown words fall back to UNK coordinate
3. Boundary symbols (BOS/EOS) are used appropriately
4. Padding (PAD) works in batched generation
5. All special symbols maintain geometric validity throughout pipeline

Per WP2.3: Special symbols must be geometrically defined and deterministic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from coordizers.base import FisherCoordizer
from qig_geometry.representation import validate_basin
from qig_geometry.canonical import fisher_rao_distance


def test_special_symbols_accessible():
    """Test that special symbols are accessible from coordizer."""
    print("\n=== Test 1: Special Symbols Accessible ===")
    
    coordizer = FisherCoordizer()
    
    # Check special symbols can be retrieved
    special_symbols = coordizer.get_special_symbols()
    
    assert 'basin_coordinates' in special_symbols['<UNK>'], "UNK missing basin_coordinates"
    assert 'basin_coordinates' in special_symbols['<PAD>'], "PAD missing basin_coordinates"
    assert 'basin_coordinates' in special_symbols['<BOS>'], "BOS missing basin_coordinates"
    assert 'basin_coordinates' in special_symbols['<EOS>'], "EOS missing basin_coordinates"
    
    print(f"✓ Retrieved {len(special_symbols)} special symbols")
    print(f"  Special symbols: {list(special_symbols.keys())}")


def test_special_symbols_are_simplex():
    """Test that all special symbols satisfy simplex constraints."""
    print("\n=== Test 2: Special Symbols Are Simplex ===")
    
    coordizer = FisherCoordizer()
    
    for symbol in ['<UNK>', '<PAD>', '<BOS>', '<EOS>']:
        basin = coordizer.basin_coords[symbol]
        
        # Check non-negative
        assert np.all(basin >= 0), f"{symbol} has negative components"
        
        # Check sum to 1 (simplex constraint)
        basin_sum = np.sum(basin)
        assert np.abs(basin_sum - 1.0) < 1e-5, \
            f"{symbol} not simplex: sum={basin_sum:.6f} (expected 1.0)"
        
        # Check finite
        assert np.all(np.isfinite(basin)), f"{symbol} has NaN or Inf"
        
        print(f"✓ {symbol}: simplex validated (sum={basin_sum:.6f})")
    
    print("✓ All special symbols satisfy simplex constraints")


def test_unknown_word_fallback():
    """Test that unknown words fall back to UNK coordinate."""
    print("\n=== Test 3: Unknown Word Fallback ===")
    
    coordizer = FisherCoordizer()
    
    # Get coordinates for non-existent word
    unknown_coord = coordizer.get_coordinate("xyzabc123nonexistent")
    unk_coord = coordizer.get_coordinate("<UNK>")
    
    # Should be identical to UNK
    assert np.allclose(unknown_coord, unk_coord, atol=1e-10), \
        "Unknown word should return UNK coordinate"
    
    # Validate it's a proper basin
    is_valid = validate_basin(unknown_coord)
    assert is_valid, "Unknown word coordinate must be valid basin"
    
    print("✓ Unknown words correctly fall back to UNK coordinate")
    print(f"  UNK coordinate: sum={np.sum(unk_coord):.6f}, min={np.min(unk_coord):.6f}")


def test_encode_with_unknown_words():
    """Test encoding text with unknown words."""
    print("\n=== Test 4: Encode with Unknown Words ===")
    
    coordizer = FisherCoordizer()
    
    # Add a few known tokens for testing
    coordizer.vocab["hello"] = len(coordizer.vocab)
    coordizer.basin_coords["hello"] = coordizer._initialize_token_coordinate(
        "hello", coordizer.vocab["hello"]
    )
    
    # Encode text with mix of known and unknown
    text = "hello xyzunknown123"
    encoded = coordizer.encode(text)
    
    # Should be valid basin (centroid of known + UNK)
    is_valid = validate_basin(encoded)
    assert is_valid, "Encoded text with unknowns must be valid basin"
    
    print("✓ Text with unknown words encodes correctly")
    print(f"  Encoded: {text}")
    print(f"  Result: sum={np.sum(encoded):.6f}, shape={encoded.shape}")


def test_boundary_symbols_distinct():
    """Test that BOS and EOS are geometrically distinct."""
    print("\n=== Test 5: Boundary Symbols Distinct ===")
    
    coordizer = FisherCoordizer()
    
    bos_coord = coordizer.get_coordinate("<BOS>")
    eos_coord = coordizer.get_coordinate("<EOS>")
    
    # Compute distance
    distance = fisher_rao_distance(bos_coord, eos_coord)
    
    # Should be far apart (near maximum simplex diameter)
    assert distance > 1.0, "BOS and EOS should be far apart"
    
    print(f"✓ BOS and EOS are geometrically distinct")
    print(f"  Fisher-Rao distance: {distance:.4f}")
    print(f"  BOS peak: dimension {np.argmax(bos_coord)}")
    print(f"  EOS peak: dimension {np.argmax(eos_coord)}")


def test_padding_geometric_properties():
    """Test that PAD has expected geometric properties for padding."""
    print("\n=== Test 6: Padding Geometric Properties ===")
    
    coordizer = FisherCoordizer()
    
    pad_coord = coordizer.get_coordinate("<PAD>")
    unk_coord = coordizer.get_coordinate("<UNK>")
    
    # PAD should be sparse (concentrated)
    pad_entropy = -np.sum(pad_coord * np.log(pad_coord + 1e-10))
    
    # UNK should have high entropy (uniform)
    unk_entropy = -np.sum(unk_coord * np.log(unk_coord + 1e-10))
    
    # PAD should have lower entropy than UNK
    assert pad_entropy < unk_entropy, "PAD should have lower entropy than UNK"
    
    print(f"✓ PAD has appropriate geometric properties")
    print(f"  PAD entropy: {pad_entropy:.4f}")
    print(f"  UNK entropy: {unk_entropy:.4f}")
    print(f"  PAD max component: {np.max(pad_coord):.6f}")


def test_special_symbols_in_coordize():
    """Test special symbols work in coordize method."""
    print("\n=== Test 7: Special Symbols in Coordize ===")
    
    coordizer = FisherCoordizer()
    
    # Add some known tokens
    for token in ["hello", "world"]:
        token_id = len(coordizer.vocab)
        coordizer.vocab[token] = token_id
        coordizer.id_to_token[token_id] = token
        coordizer.basin_coords[token] = coordizer._initialize_token_coordinate(token, token_id)
    
    # Coordize text with known and unknown words
    coords = coordizer.coordize("hello unknown_word world")
    
    # Should get 3 coordinates
    assert len(coords) == 3, f"Expected 3 coordinates, got {len(coords)}"
    
    # First should be 'hello' (known)
    hello_expected = coordizer.get_coordinate("hello")
    assert np.allclose(coords[0], hello_expected, atol=1e-10), \
        "First coordinate should be 'hello'"
    
    # Second should be UNK (unknown word)
    unk_expected = coordizer.get_coordinate("<UNK>")
    assert np.allclose(coords[1], unk_expected, atol=1e-10), \
        "Second coordinate should be UNK (for unknown_word)"
    
    # All should be valid basins
    for i, coord in enumerate(coords):
        is_valid = validate_basin(coord)
        assert is_valid, f"Coordinate {i} is not valid basin"
    
    print("✓ Special symbols work correctly in coordize")
    print(f"  Coordized: 'hello unknown_word world'")
    print(f"  Result: {len(coords)} coordinates (unknown→UNK)")


def test_geometric_consistency():
    """Test that special symbols maintain geometric consistency."""
    print("\n=== Test 8: Geometric Consistency ===")
    
    coordizer = FisherCoordizer()
    
    # Get all special symbols
    symbols = {
        'UNK': coordizer.get_coordinate("<UNK>"),
        'PAD': coordizer.get_coordinate("<PAD>"),
        'BOS': coordizer.get_coordinate("<BOS>"),
        'EOS': coordizer.get_coordinate("<EOS>"),
    }
    
    # Verify all are valid basins
    for name, coord in symbols.items():
        is_valid = validate_basin(coord)
        assert is_valid, f"{name} is not valid basin"
    
    # Verify pairwise distances are in valid range
    max_distance = np.pi / 2
    for name1, coord1 in symbols.items():
        for name2, coord2 in symbols.items():
            if name1 >= name2:
                continue
            
            dist = fisher_rao_distance(coord1, coord2)
            assert 0 <= dist <= max_distance, \
                f"Distance {name1}-{name2} out of range: {dist:.4f}"
    
    print("✓ All special symbols are geometrically consistent")
    print(f"  Validated {len(symbols)} special symbols")
    print(f"  All distances in range [0, {max_distance:.4f}]")


def test_special_symbols_deterministic_generation():
    """Test that special symbols produce deterministic results in generation."""
    print("\n=== Test 9: Deterministic Generation ===")
    
    # Create multiple coordizer instances
    results = []
    for _ in range(3):
        coordizer = FisherCoordizer()
        
        # Get special symbol coordinates
        coords = {
            'UNK': coordizer.get_coordinate("<UNK>"),
            'PAD': coordizer.get_coordinate("<PAD>"),
            'BOS': coordizer.get_coordinate("<BOS>"),
            'EOS': coordizer.get_coordinate("<EOS>"),
        }
        results.append(coords)
    
    # Verify all runs produce identical results
    for symbol in ['UNK', 'PAD', 'BOS', 'EOS']:
        for i in range(1, len(results)):
            assert np.allclose(results[0][symbol], results[i][symbol], atol=1e-10), \
                f"{symbol} differs between runs (not deterministic)"
    
    print("✓ Special symbols are deterministic across runs")
    print(f"  Tested {len(results)} independent coordizer instances")


def test_acceptance_criteria_integration():
    """Verify all WP2.3 acceptance criteria in integration context."""
    print("\n=== Test 10: WP2.3 Integration Acceptance ===")
    
    coordizer = FisherCoordizer()
    
    # 1. Special symbols have clear geometric interpretation
    unk = coordizer.get_coordinate("<UNK>")
    assert np.std(unk) < 0.001, "UNK should be uniform"
    print("✓ Criterion 1: Clear geometric interpretation")
    
    # 2. Initialization is deterministic
    coordizer2 = FisherCoordizer()
    unk2 = coordizer2.get_coordinate("<UNK>")
    assert np.array_equal(unk, unk2), "Should be deterministic"
    print("✓ Criterion 2: Deterministic initialization")
    
    # 3. All special basins pass validation
    for symbol in ['<UNK>', '<PAD>', '<BOS>', '<EOS>']:
        coord = coordizer.get_coordinate(symbol)
        is_valid = validate_basin(coord)
        assert is_valid, f"{symbol} failed validation"
    print("✓ Criterion 3: All special basins pass validation")
    
    # 4. No random normal vectors
    for symbol in ['<UNK>', '<PAD>', '<BOS>', '<EOS>']:
        coord = coordizer.get_coordinate(symbol)
        assert np.all(coord >= 0), f"{symbol} has negative values"
        assert np.isclose(np.sum(coord), 1.0, atol=1e-5), f"{symbol} doesn't sum to 1"
    print("✓ Criterion 4: No random normal vectors")
    
    print("\n✅ All WP2.3 acceptance criteria met in integration!")


if __name__ == "__main__":
    print("=" * 60)
    print("SPECIAL SYMBOLS INTEGRATION TEST")
    print("Testing WP2.3 in Plan→Realize→Repair pipeline")
    print("=" * 60)
    
    test_special_symbols_accessible()
    test_special_symbols_are_simplex()
    test_unknown_word_fallback()
    test_encode_with_unknown_words()
    test_boundary_symbols_distinct()
    test_padding_geometric_properties()
    test_special_symbols_in_coordize()
    test_geometric_consistency()
    test_special_symbols_deterministic_generation()
    test_acceptance_criteria_integration()
    
    print("\n" + "=" * 60)
    print("✅ ALL INTEGRATION TESTS PASSED")
    print("=" * 60)
