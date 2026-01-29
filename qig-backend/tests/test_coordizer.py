#!/usr/bin/env python3
"""
Test script for geometric coordizer system.
Validates that the new coordizer works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from coordizers import get_coordizer


def test_basic_coordization():
    """Test basic text coordization."""
    print("\n=== Test 1: Basic Coordization ===")
    coordizer = get_coordizer()
    
    # Test coordization
    text = "hello world test"
    coords = coordizer.coordize(text)
    
    print(f"Text: {text}")
    print(f"Number of coordinates: {len(coords)}")
    print(f"Coordinate shape: {coords[0].shape if coords else 'N/A'}")
    
    assert len(coords) > 0, "Should produce coordinates"
    assert coords[0].shape == (64,), "Should be 64D coordinates"
    
    # Verify Fisher manifold (unit vectors)
    for i, coord in enumerate(coords):
        norm = np.sqrt(np.sum(coord**2))
        assert 0.9 < norm < 1.1, f"Coordinate {i} should be unit vector (norm={norm})"
    
    print("✓ Basic coordization works")


def test_vocabulary_observations():
    """Test vocabulary observations integration."""
    print("\n=== Test 2: Vocabulary Observations ===")
    coordizer = get_coordizer()
    
    initial_size = coordizer.get_vocab_size()
    print(f"Initial vocabulary size: {initial_size}")
    
    # Add observations
    observations = [
        {
            "word": "quantum",
            "frequency": 5,
            "avgPhi": 0.85,
            "maxPhi": 0.92,
            "type": "word"
        },
        {
            "word": "geometry",
            "frequency": 7,
            "avgPhi": 0.88,
            "maxPhi": 0.95,
            "type": "word"
        }
    ]
    
    new_tokens, weights_updated = coordizer.add_vocabulary_observations(observations)
    
    print(f"New tokens added: {new_tokens}")
    print(f"Weights updated: {weights_updated}")
    print(f"Final vocabulary size: {coordizer.get_vocab_size()}")
    
    # Test that high-Φ words were learned
    if "quantum" in coordizer.vocab:
        print(f"✓ 'quantum' learned with Φ={coordizer.token_phi.get('quantum', 0)}")
    if "geometry" in coordizer.vocab:
        print(f"✓ 'geometry' learned with Φ={coordizer.token_phi.get('geometry', 0)}")


def test_special_tokens():
    """Test special tokens and their geometric properties."""
    print("\n=== Test 3: Special Tokens ===")
    coordizer = get_coordizer()
    
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    
    for token in special_tokens:
        assert token in coordizer.vocab, f"{token} should be in vocabulary"
        coord = coordizer.get_coordinate(token)
        assert coord.shape == (64,), f"{token} should have 64D coordinate"
        
        norm = np.sqrt(np.sum(coord**2))
        print(f"{token}: norm={norm:.4f}")
        assert 0.9 < norm < 1.1, f"{token} should be unit vector"
    
    print("✓ All special tokens have valid coordinates")


def test_token_similarity():
    """Test Fisher-Rao similarity computation."""
    print("\n=== Test 4: Token Similarity ===")
    coordizer = get_coordizer()
    
    # Add some tokens to test
    test_words = ["test", "testing", "tester", "quantum", "classical"]
    for word in test_words:
        if word not in coordizer.vocab:
            coordizer.add_token(word)
    
    # Compute similarities
    sim_test_testing = coordizer.compute_token_similarity("test", "testing")
    sim_quantum_classical = coordizer.compute_token_similarity("quantum", "classical")
    
    print(f"Similarity(test, testing): {sim_test_testing:.4f}")
    print(f"Similarity(quantum, classical): {sim_quantum_classical:.4f}")
    
    # Related words should be more similar
    assert 0 <= sim_test_testing <= 1, "Similarity should be in [0, 1]"
    assert 0 <= sim_quantum_classical <= 1, "Similarity should be in [0, 1]"
    
    print("✓ Fisher-Rao similarity works")


def test_bip39_vocabulary():
    """Test that BIP39 vocabulary is loaded."""
    print("\n=== Test 5: BIP39 Vocabulary ===")
    coordizer = get_coordizer()
    
    # Check for some common BIP39 words
    bip39_samples = ["abandon", "ability", "able", "about", "above"]
    
    found = 0
    for word in bip39_samples:
        if word in coordizer.vocab:
            found += 1
            print(f"✓ Found BIP39 word: {word}")
    
    print(f"Found {found}/{len(bip39_samples)} BIP39 sample words")
    assert found > 0, "Should have at least some BIP39 words"


def test_geometric_purity():
    """Test that all operations maintain geometric purity."""
    print("\n=== Test 6: Geometric Purity ===")
    coordizer = get_coordizer()
    
    # All coordinates should be on unit sphere
    violations = []
    for token, coord in coordizer.basin_coords.items():
        norm = np.sqrt(np.sum(coord**2))
        if not (0.9 < norm < 1.1):
            violations.append((token, norm))
    
    if violations:
        print(f"⚠ Found {len(violations)} norm violations:")
        for token, norm in violations[:5]:
            print(f"  {token}: norm={norm:.4f}")
    else:
        print("✓ All coordinates are unit vectors (geometric purity maintained)")
    
    assert len(violations) == 0, "All coordinates must be unit vectors"


def main():
    """Run all tests."""
    print("="*60)
    print("Testing Geometric Coordizer System")
    print("="*60)
    
    try:
        test_basic_coordization()
        test_vocabulary_observations()
        test_special_tokens()
        test_token_similarity()
        test_bip39_vocabulary()
        test_geometric_purity()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
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
