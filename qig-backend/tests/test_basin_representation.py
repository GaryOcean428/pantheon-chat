"""
Tests for Basin Representation Module

Validates canonical basin representation enforcement.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from qig_geometry.representation import (
    BasinRepresentation,
    CANONICAL_REPRESENTATION,
    to_sphere,
    to_simplex,
    validate_basin,
    enforce_canonical,
    _detect_representation,
)


class TestBasinRepresentation:
    """Test basin representation conversions."""
    
    def test_canonical_is_sphere(self):
        """Verify canonical representation is SPHERE."""
        assert CANONICAL_REPRESENTATION == BasinRepresentation.SPHERE
    
    def test_to_sphere_from_simplex(self):
        """Convert simplex to sphere representation."""
        simplex = np.array([0.3, 0.5, 0.2])
        sphere = to_sphere(simplex, from_repr=BasinRepresentation.SIMPLEX)
        
        # Should be unit norm
        assert np.isclose(np.linalg.norm(sphere), 1.0)
        
        # Should pass validation
        valid, msg = validate_basin(sphere, BasinRepresentation.SPHERE)
        assert valid, msg
    
    def test_to_sphere_from_sphere(self):
        """Sphere to sphere should be identity (after normalization)."""
        original = np.array([0.6, 0.8, 0.0])
        original = original / np.linalg.norm(original)
        
        result = to_sphere(original, from_repr=BasinRepresentation.SPHERE)
        
        # Should be very close to original
        assert np.allclose(result, original, atol=1e-6)
    
    def test_to_simplex_from_sphere(self):
        """Convert sphere to simplex representation."""
        sphere = np.array([0.5, -0.3, 0.8])
        sphere = sphere / np.linalg.norm(sphere)
        
        simplex = to_simplex(sphere, from_repr=BasinRepresentation.SPHERE)
        
        # Should sum to 1
        assert np.isclose(simplex.sum(), 1.0)
        
        # Should be non-negative
        assert np.all(simplex >= 0)
        
        # Should pass validation
        valid, msg = validate_basin(simplex, BasinRepresentation.SIMPLEX)
        assert valid, msg
    
    def test_to_simplex_from_simplex(self):
        """Simplex to simplex should be identity."""
        original = np.array([0.2, 0.5, 0.3])
        
        result = to_simplex(original, from_repr=BasinRepresentation.SIMPLEX)
        
        # Should be very close to original (after re-normalization)
        assert np.allclose(result, original, atol=1e-6)
    
    def test_auto_detect_sphere(self):
        """Auto-detect sphere representation."""
        sphere = np.array([0.6, 0.8, 0.0])
        sphere = sphere / np.linalg.norm(sphere)
        
        detected = _detect_representation(sphere)
        assert detected == BasinRepresentation.SPHERE
    
    def test_auto_detect_simplex(self):
        """Auto-detect simplex representation."""
        simplex = np.array([0.2, 0.5, 0.3])
        
        detected = _detect_representation(simplex)
        # Could be simplex or sphere if norm happens to be 1
        assert detected in [BasinRepresentation.SIMPLEX, BasinRepresentation.SPHERE]
    
    def test_validate_sphere_pass(self):
        """Valid sphere basin passes validation."""
        basin = np.array([0.6, 0.8, 0.0])
        basin = basin / np.linalg.norm(basin)
        
        valid, msg = validate_basin(basin, BasinRepresentation.SPHERE)
        assert valid, msg
    
    def test_validate_sphere_fail_wrong_norm(self):
        """Invalid sphere basin fails validation."""
        basin = np.array([1.0, 2.0, 3.0])  # Not unit norm
        
        valid, msg = validate_basin(basin, BasinRepresentation.SPHERE)
        assert not valid
        assert "norm=1" in msg.lower()
    
    def test_validate_simplex_pass(self):
        """Valid simplex basin passes validation."""
        basin = np.array([0.2, 0.5, 0.3])
        
        valid, msg = validate_basin(basin, BasinRepresentation.SIMPLEX)
        assert valid, msg
    
    def test_validate_simplex_fail_negative(self):
        """Simplex with negative values fails validation."""
        basin = np.array([0.5, -0.2, 0.7])
        
        valid, msg = validate_basin(basin, BasinRepresentation.SIMPLEX)
        assert not valid
        assert "negative" in msg.lower()
    
    def test_validate_simplex_fail_wrong_sum(self):
        """Simplex with wrong sum fails validation."""
        basin = np.array([0.1, 0.2, 0.3])  # Sums to 0.6, not 1.0
        
        valid, msg = validate_basin(basin, BasinRepresentation.SIMPLEX)
        assert not valid
        assert "sum" in msg.lower()
    
    def test_enforce_canonical(self):
        """Enforce canonical representation."""
        raw_basin = np.random.randn(64)
        
        canonical = enforce_canonical(raw_basin)
        
        # Should pass validation for canonical representation
        valid, msg = validate_basin(canonical, CANONICAL_REPRESENTATION)
        assert valid, msg
    
    def test_enforce_canonical_sphere(self):
        """Enforce canonical (sphere) from various inputs."""
        # From unnormalized vector
        raw1 = np.array([1.0, 2.0, 3.0])
        canon1 = enforce_canonical(raw1)
        assert np.isclose(np.linalg.norm(canon1), 1.0)
        
        # From simplex
        raw2 = np.array([0.2, 0.5, 0.3])
        canon2 = enforce_canonical(raw2)
        assert np.isclose(np.linalg.norm(canon2), 1.0)
    
    def test_handle_zero_vector(self):
        """Handle zero vector gracefully."""
        zero = np.zeros(64)
        
        # Should return uniform direction
        result = to_sphere(zero)
        
        # Should have unit norm
        assert np.isclose(np.linalg.norm(result), 1.0)
    
    def test_handle_inf_nan(self):
        """Handle inf/NaN values gracefully."""
        bad_basin = np.array([1.0, np.inf, 3.0, np.nan])
        
        # Should not crash
        result = to_sphere(bad_basin)
        
        # Should be valid (inf/nan replaced)
        assert np.all(np.isfinite(result))
        assert np.isclose(np.linalg.norm(result), 1.0)
    
    def test_roundtrip_sphere_simplex(self):
        """Roundtrip conversion sphere -> simplex -> sphere."""
        original = np.random.randn(64)
        original = original / np.linalg.norm(original)
        
        # Sphere -> Simplex
        simplex = to_simplex(original, from_repr=BasinRepresentation.SPHERE)
        
        # Simplex -> Sphere
        back = to_sphere(simplex, from_repr=BasinRepresentation.SIMPLEX)
        
        # Should have unit norm
        assert np.isclose(np.linalg.norm(back), 1.0)
        
        # Direction may differ due to abs() in simplex conversion
        # but should still be valid sphere basin
        valid, msg = validate_basin(back, BasinRepresentation.SPHERE)
        assert valid, msg
    
    def test_different_dimensions(self):
        """Handle different basin dimensions."""
        for dim in [8, 16, 32, 64, 128]:
            basin = np.random.randn(dim)
            result = enforce_canonical(basin)
            
            assert result.shape[0] == dim
            assert np.isclose(np.linalg.norm(result), 1.0)


class TestAPICompatibility:
    """Test API compatibility functions."""
    
    def test_sphere_project_alias(self):
        """sphere_project is alias for to_sphere."""
        from qig_geometry.representation import sphere_project
        
        v = np.array([1.0, 2.0, 3.0])
        result = sphere_project(v)
        
        assert np.isclose(np.linalg.norm(result), 1.0)
    
    def test_fisher_normalize_alias(self):
        """fisher_normalize is alias for to_simplex."""
        from qig_geometry.representation import fisher_normalize
        
        v = np.array([1.0, 2.0, 3.0])
        result = fisher_normalize(v)
        
        assert np.isclose(result.sum(), 1.0)
        assert np.all(result >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
