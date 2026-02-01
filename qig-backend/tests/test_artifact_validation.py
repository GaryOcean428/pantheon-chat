#!/usr/bin/env python3
"""
Test Suite for CoordizerArtifactV1 Validation

Tests artifact format validation, versioning, and provenance tracking.
Validates geometric constraints and round-trip save/load operations.
"""

import sys
import os
from qig_geometry import to_simplex_prob

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import tempfile
import shutil
import numpy as np
from datetime import datetime, timezone

from coordizers.base import FisherCoordizer
from artifact_validation import (
    validate_artifact,
    validate_artifact_from_file,
    detect_artifact_version,
    ArtifactValidator
)


def test_valid_artifact_validation():
    """Test validation of a valid CoordizerArtifactV1 artifact."""
    print("\n=== Test 1: Valid Artifact Validation ===")
    
    # Create a minimal valid artifact
    artifact = {
        "version": "1.0",
        "basin_dim": 64,
        "symbols": ["hello", "world", "test"],
        "basin_coords": [
            list(np.random.randn(64) / 10 + 1),  # Near unit norm
            list(np.random.randn(64) / 10 + 1),
            list(np.random.randn(64) / 10 + 1)
        ],
        "phi_scores": [0.7, 0.6, 0.5],
        "special_symbols": {
            "UNK": {
                "token": "<UNK>",
                "token_id": 0,
                "basin_coord": list(np.random.randn(64) / 10 + 1),
                "phi_score": 0.5,
                "frequency": 0
            },
            "PAD": {
                "token": "<PAD>",
                "token_id": 1,
                "basin_coord": list(np.random.randn(64) / 10 + 1),
                "phi_score": 0.0,
                "frequency": 0
            },
            "BOS": {
                "token": "<BOS>",
                "token_id": 2,
                "basin_coord": list(np.random.randn(64) / 10 + 1),
                "phi_score": 0.3,
                "frequency": 0
            },
            "EOS": {
                "token": "<EOS>",
                "token_id": 3,
                "basin_coord": list(np.random.randn(64) / 10 + 1),
                "phi_score": 0.3,
                "frequency": 0
            }
        },
        "provenance": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "geometry_version": "a" * 40,  # Valid SHA-1 format
            "hyperparameters": {
                "vocab_size": 4096,
                "coordinate_dim": 64,
                "min_frequency": 2
            }
        },
        "validation": {
            "passes_simplex_check": True,
            "fisher_rao_identity_verified": True,
            "dimension_consistent": True,
            "unit_norm_verified": True,
            "no_nan_inf": True,
            "special_symbols_deterministic": True,
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_errors": []
        }
    }
    
    # Normalize coordinates to unit sphere
    for i in range(len(artifact["basin_coords"])):
        coord = np.array(artifact["basin_coords"][i])
        artifact["basin_coords"][i] = (to_simplex_prob(coord)).tolist()
    
    for symbol_key in artifact["special_symbols"]:
        coord = np.array(artifact["special_symbols"][symbol_key]["basin_coord"])
        artifact["special_symbols"][symbol_key]["basin_coord"] = (to_simplex_prob(coord)).tolist()
    
    result = validate_artifact(artifact)
    
    print(f"Validation result: {result['valid']}")
    print(f"Errors: {result['errors']}")
    print(f"Warnings: {result['warnings']}")
    print(f"Checks passed: {sum(1 for v in result['checks'].values() if v)}/{len(result['checks'])}")
    
    assert result['valid'], f"Valid artifact should pass validation. Errors: {result['errors']}"
    assert len(result['errors']) == 0, "Should have no errors"
    print("✓ Valid artifact passes validation")


def test_missing_required_fields():
    """Test validation fails for missing required fields."""
    print("\n=== Test 2: Missing Required Fields ===")
    
    # Artifact missing 'version' field
    artifact = {
        "basin_dim": 64,
        "symbols": ["test"]
    }
    
    result = validate_artifact(artifact)
    
    print(f"Validation result: {result['valid']}")
    print(f"Errors: {result['errors']}")
    
    assert not result['valid'], "Should fail validation"
    assert any("Missing required field" in err for err in result['errors']), \
        "Should report missing fields"
    print("✓ Missing fields detected correctly")


def test_invalid_version():
    """Test validation fails for invalid version."""
    print("\n=== Test 3: Invalid Version ===")
    
    artifact = {
        "version": "2.0",  # Invalid version
        "basin_dim": 64,
        "symbols": ["test"],
        "basin_coords": [[1.0] * 64],
        "phi_scores": [0.5],
        "special_symbols": {
            "UNK": {"token": "<UNK>", "token_id": 0, "basin_coord": [1.0] * 64},
            "PAD": {"token": "<PAD>", "token_id": 1, "basin_coord": [1.0] * 64},
            "BOS": {"token": "<BOS>", "token_id": 2, "basin_coord": [1.0] * 64},
            "EOS": {"token": "<EOS>", "token_id": 3, "basin_coord": [1.0] * 64}
        },
        "provenance": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "geometry_version": "a" * 40,
            "hyperparameters": {"coordinate_dim": 64}
        },
        "validation": {
            "passes_simplex_check": True,
            "fisher_rao_identity_verified": True,
            "dimension_consistent": True
        }
    }
    
    result = validate_artifact(artifact)
    
    print(f"Validation result: {result['valid']}")
    print(f"Errors: {result['errors']}")
    
    assert not result['valid'], "Should fail validation"
    assert any("Invalid version" in err for err in result['errors']), \
        "Should report invalid version"
    print("✓ Invalid version detected correctly")


def test_dimension_mismatch():
    """Test validation fails for dimension mismatches."""
    print("\n=== Test 4: Dimension Mismatch ===")
    
    artifact = {
        "version": "1.0",
        "basin_dim": 64,
        "symbols": ["test1", "test2"],
        "basin_coords": [[1.0] * 64],  # Only 1 coordinate for 2 symbols
        "phi_scores": [0.5, 0.6],
        "special_symbols": {
            "UNK": {"token": "<UNK>", "token_id": 0, "basin_coord": [1.0] * 64},
            "PAD": {"token": "<PAD>", "token_id": 1, "basin_coord": [1.0] * 64},
            "BOS": {"token": "<BOS>", "token_id": 2, "basin_coord": [1.0] * 64},
            "EOS": {"token": "<EOS>", "token_id": 3, "basin_coord": [1.0] * 64}
        },
        "provenance": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "geometry_version": "a" * 40,
            "hyperparameters": {"coordinate_dim": 64}
        },
        "validation": {
            "passes_simplex_check": True,
            "fisher_rao_identity_verified": True,
            "dimension_consistent": True
        }
    }
    
    result = validate_artifact(artifact)
    
    print(f"Validation result: {result['valid']}")
    print(f"Errors: {result['errors']}")
    
    assert not result['valid'], "Should fail validation"
    assert any("Dimension mismatch" in err for err in result['errors']), \
        "Should report dimension mismatch"
    print("✓ Dimension mismatch detected correctly")


def test_non_unit_norm():
    """Test validation fails for non-unit-normalized coordinates."""
    print("\n=== Test 5: Non-Unit Norm Coordinates ===")
    
    artifact = {
        "version": "1.0",
        "basin_dim": 64,
        "symbols": ["test"],
        "basin_coords": [[10.0] * 64],  # Not unit normalized
        "phi_scores": [0.5],
        "special_symbols": {
            "UNK": {"token": "<UNK>", "token_id": 0, "basin_coord": [1.0] * 64},
            "PAD": {"token": "<PAD>", "token_id": 1, "basin_coord": [1.0] * 64},
            "BOS": {"token": "<BOS>", "token_id": 2, "basin_coord": [1.0] * 64},
            "EOS": {"token": "<EOS>", "token_id": 3, "basin_coord": [1.0] * 64}
        },
        "provenance": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "geometry_version": "a" * 40,
            "hyperparameters": {"coordinate_dim": 64}
        },
        "validation": {
            "passes_simplex_check": True,
            "fisher_rao_identity_verified": True,
            "dimension_consistent": True
        }
    }
    
    result = validate_artifact(artifact)
    
    print(f"Validation result: {result['valid']}")
    print(f"Errors: {result['errors']}")
    
    assert not result['valid'], "Should fail validation"
    assert any("not unit-normalized" in err for err in result['errors']), \
        "Should report non-unit norm"
    print("✓ Non-unit norm detected correctly")


def test_save_load_roundtrip():
    """Test save and load round-trip with validation."""
    print("\n=== Test 6: Save/Load Round-Trip ===")
    
    # Create a coordizer with some vocabulary
    coordizer = FisherCoordizer(vocab_size=100, coordinate_dim=64)
    
    # Add some tokens
    test_tokens = ["hello", "world", "test", "quantum", "fisher"]
    for token in test_tokens:
        coordizer.add_token(token)
    
    # Create temp directory for artifact
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save artifact with metadata
        metadata = {
            "training_corpus": "test corpus",
            "corpus_size": 1000,
            "created_by": "test_suite",
            "description": "Test artifact for validation"
        }
        
        coordizer.save(temp_dir, metadata=metadata)
        print(f"Saved artifact to: {temp_dir}")
        
        # Verify files exist
        assert os.path.exists(os.path.join(temp_dir, "vocab.json"))
        assert os.path.exists(os.path.join(temp_dir, "basin_coords.npy"))
        assert os.path.exists(os.path.join(temp_dir, "coord_tokens.json"))
        
        # Load vocab.json and validate structure
        with open(os.path.join(temp_dir, "vocab.json"), "r") as f:
            artifact_data = json.load(f)
        
        print(f"Artifact version: {artifact_data.get('version')}")
        print(f"Basin dimension: {artifact_data.get('basin_dim')}")
        print(f"Number of symbols: {len(artifact_data.get('symbols', []))}")
        print(f"Provenance created_at: {artifact_data.get('provenance', {}).get('created_at')}")
        
        assert artifact_data['version'] == '1.0', "Should have version 1.0"
        assert artifact_data['basin_dim'] == 64, "Should have basin_dim 64"
        assert 'provenance' in artifact_data, "Should have provenance"
        assert 'validation' in artifact_data, "Should have validation"
        
        # Load artifact back
        coordizer2 = FisherCoordizer(vocab_size=100, coordinate_dim=64)
        coordizer2.load(temp_dir, validate=True)
        
        print(f"Loaded vocabulary size: {len(coordizer2.vocab)}")
        
        # Verify tokens
        for token in test_tokens:
            assert token in coordizer2.vocab, f"Token {token} should be in loaded vocab"
            coord1 = coordizer.basin_coords[token]
            coord2 = coordizer2.basin_coords[token]
            assert np.allclose(coord1, coord2), f"Coordinates for {token} should match"
        
        print("✓ Save/load round-trip successful")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_version_detection():
    """Test artifact version detection."""
    print("\n=== Test 7: Version Detection ===")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create v1.0 artifact
        coordizer = FisherCoordizer(vocab_size=100, coordinate_dim=64)
        coordizer.add_token("test")
        coordizer.save(temp_dir)
        
        # Detect version
        version = detect_artifact_version(temp_dir)
        print(f"Detected version: {version}")
        
        assert version == '1.0', f"Should detect version 1.0, got {version}"
        print("✓ Version detection works correctly")
        
    finally:
        shutil.rmtree(temp_dir)


def test_reject_unversioned():
    """Test that unversioned artifacts are rejected."""
    print("\n=== Test 8: Reject Unversioned Artifacts ===")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create artifact without version field
        vocab_data = {
            "vocab": {"test": 0},
            "id_to_token": {"0": "test"},
            "token_frequency": {"test": 1},
            "token_phi": {"test": 0.5}
            # Missing: version, basin_dim, symbols, special_symbols, provenance, validation
        }
        
        with open(os.path.join(temp_dir, "vocab.json"), "w") as f:
            json.dump(vocab_data, f)
        
        # Create dummy files
        np.save(os.path.join(temp_dir, "basin_coords.npy"), np.random.randn(1, 64))
        with open(os.path.join(temp_dir, "coord_tokens.json"), "w") as f:
            json.dump(["test"], f)
        
        # Try to load - should fail
        coordizer = FisherCoordizer()
        try:
            coordizer.load(temp_dir)
            assert False, "Should have raised RuntimeError for unversioned artifact"
        except RuntimeError as e:
            error_msg = str(e)
            print(f"Error message: {error_msg}")
            assert "Unversioned artifact" in error_msg or "Missing required keys" in error_msg, \
                "Should report unversioned artifact"
            print("✓ Unversioned artifact rejected correctly")
        
    finally:
        shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all validation tests."""
    print("="*60)
    print("CoordizerArtifactV1 Validation Test Suite")
    print("="*60)
    
    tests = [
        test_valid_artifact_validation,
        test_missing_required_fields,
        test_invalid_version,
        test_dimension_mismatch,
        test_non_unit_norm,
        test_save_load_roundtrip,
        test_version_detection,
        test_reject_unversioned
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Test error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
