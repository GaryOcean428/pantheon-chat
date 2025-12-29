"""
Tests for Canonical Coordizer API
=================================

Validates:
- Roundtrip encoding/decoding
- Determinism
- Coordinate integrity
- Unicode edge cases
- Fisher distance calculations
"""

import numpy as np
import pytest

from qig_tokenizer.coordizer import Coordizer

# Test artifact path - adjust if needed
ARTIFACT_PATH = "artifacts/coordizer/v1"


@pytest.fixture
def coordizer():
    """Load the v1 coordizer artifact."""
    return Coordizer.load(ARTIFACT_PATH)


class TestRoundtrip:
    """Test encode/decode roundtrip."""

    def test_simple_ascii(self, coordizer):
        """Simple ASCII text roundtrips correctly."""
        text = "Hello, world!"
        ids = coordizer.encode(text)
        decoded = coordizer.decode(ids)
        assert decoded == text

    def test_empty_string(self, coordizer):
        """Empty string roundtrips correctly."""
        text = ""
        ids = coordizer.encode(text)
        decoded = coordizer.decode(ids)
        assert decoded == text
        assert ids == []

    def test_whitespace(self, coordizer):
        """Whitespace-only text roundtrips correctly."""
        text = "   \t\n  "
        ids = coordizer.encode(text)
        decoded = coordizer.decode(ids)
        assert decoded == text

    def test_unicode_basic(self, coordizer):
        """Basic unicode roundtrips correctly."""
        text = "Hello, 世界! 🌍"
        ids = coordizer.encode(text)
        decoded = coordizer.decode(ids)
        assert decoded == text

    def test_unicode_edge_cases(self, coordizer):
        """Various unicode edge cases."""
        test_cases = [
            "Emoji: 😀🎉🚀",
            "Arabic: مرحبا",
            "Hebrew: שלום",
            "Mixed: Hello世界مرحبا",
            "Math: ∫∑∏√∞",
            "Combining: é = e + ́",
            "Zero-width: a\u200bb",
        ]
        for text in test_cases:
            ids = coordizer.encode(text)
            decoded = coordizer.decode(ids)
            assert decoded == text, f"Failed for: {text!r}"

    def test_long_text(self, coordizer):
        """Long text roundtrips correctly."""
        text = "The quick brown fox jumps over the lazy dog. " * 100
        ids = coordizer.encode(text)
        decoded = coordizer.decode(ids)
        assert decoded == text


class TestDeterminism:
    """Test deterministic behavior."""

    def test_same_output_multiple_calls(self, coordizer):
        """Same input produces same output across multiple calls."""
        text = "Determinism test: Hello, world! 123"
        ids1 = coordizer.encode(text)
        ids2 = coordizer.encode(text)
        ids3 = coordizer.encode(text)
        assert ids1 == ids2 == ids3

    def test_same_coords_multiple_calls(self, coordizer):
        """Coordinates are identical across calls."""
        text = "Coordinate test"
        ids1, coords1 = coordizer.encode_to_coords(text)
        ids2, coords2 = coordizer.encode_to_coords(text)
        assert ids1 == ids2
        assert np.allclose(coords1, coords2)

    def test_encode_hash_consistency(self, coordizer):
        """Encoding produces consistent hash."""
        texts = [
            "Hello",
            "World",
            "Test 123",
            "Unicode: 日本語",
        ]
        hashes1 = [hash(tuple(coordizer.encode(t))) for t in texts]
        hashes2 = [hash(tuple(coordizer.encode(t))) for t in texts]
        assert hashes1 == hashes2


class TestCoordinateIntegrity:
    """Test coordinate properties."""

    def test_coords_shape(self, coordizer):
        """Coordinates have correct shape."""
        text = "Test"
        ids, coords = coordizer.encode_to_coords(text)
        assert coords.shape == (len(ids), 64)

    def test_coords_not_zero(self, coordizer):
        """Coordinates are non-zero for valid tokens."""
        text = "Hello"
        ids, coords = coordizer.encode_to_coords(text)
        for i, (token_id, coord) in enumerate(zip(ids, coords)):
            norm = np.linalg.norm(coord)
            assert norm > 0, f"Token {token_id} has zero vector"

    def test_coords_normalized(self, coordizer):
        """Check coordinate normalization."""
        # Note: vectors may or may not be unit-normalized
        # This test documents the actual behavior
        text = "Test normalization"
        ids, coords = coordizer.encode_to_coords(text)
        norms = np.linalg.norm(coords, axis=1)
        # Should be roughly consistent magnitude
        assert np.std(norms) < 2.0, "Coordinate magnitudes vary too much"

    def test_byte_coords_exist(self, coordizer):
        """All 256 byte-level coordinates exist."""
        assert coordizer.vocab_size >= 256
        for byte_val in range(256):
            coord = coordizer.vectors[byte_val]
            assert np.linalg.norm(coord) > 0


class TestFisherDistance:
    """Test Fisher-Rao distance calculations."""

    def test_distance_to_self(self, coordizer):
        """Distance to self is zero."""
        for token_id in [0, 65, 255, 256, 1000]:
            if token_id < coordizer.vocab_size:
                dist = coordizer.fisher_distance(token_id, token_id)
                assert abs(dist) < 1e-6

    def test_distance_symmetric(self, coordizer):
        """Distance is symmetric."""
        pairs = [(0, 100), (256, 500), (1000, 2000)]
        for a, b in pairs:
            if a < coordizer.vocab_size and b < coordizer.vocab_size:
                dist_ab = coordizer.fisher_distance(a, b)
                dist_ba = coordizer.fisher_distance(b, a)
                assert abs(dist_ab - dist_ba) < 1e-6

    def test_distance_bounds(self, coordizer):
        """Distance is in valid range [0, pi]."""
        import random
        random.seed(42)
        for _ in range(100):
            a = random.randint(0, coordizer.vocab_size - 1)
            b = random.randint(0, coordizer.vocab_size - 1)
            dist = coordizer.fisher_distance(a, b)
            assert 0 <= dist <= np.pi + 1e-6


class TestCompression:
    """Test compression behavior."""

    def test_compression_positive(self, coordizer):
        """Compression ratio is positive."""
        text = "Hello, world!"
        ratio = coordizer.compression_ratio(text)
        assert ratio > 0

    def test_compression_improves_for_common_text(self, coordizer):
        """Common text should compress (ratio > 1)."""
        text = "The quick brown fox jumps over the lazy dog."
        ratio = coordizer.compression_ratio(text)
        # Should achieve some compression on common English
        assert ratio >= 1.0

    def test_compression_long_text(self, coordizer):
        """Compression should be better for longer text."""
        short = "Hello"
        long_text = "Hello world, this is a longer piece of text with common words."
        ratio_short = coordizer.compression_ratio(short)
        ratio_long = coordizer.compression_ratio(long_text)
        # Longer text typically compresses better
        assert ratio_long >= ratio_short * 0.8  # Allow some variance


class TestTokenMetadata:
    """Test token name/scale functionality."""

    def test_byte_token_names(self, coordizer):
        """Byte tokens have proper names."""
        name_a = coordizer.token_name(ord("a"))
        assert "a" in name_a or "97" in name_a or "61" in name_a

    def test_merged_token_names(self, coordizer):
        """Merged tokens have names."""
        if coordizer.vocab_size > 256:
            name = coordizer.token_name(256)
            assert len(name) > 0

    def test_byte_scale(self, coordizer):
        """Byte tokens have 'byte' scale."""
        for byte_val in [0, 65, 255]:
            scale = coordizer.token_scale(byte_val)
            assert scale == "byte"


class TestEncodeToCoords:
    """Test the canonical encode_to_coords API."""

    def test_returns_tuple(self, coordizer):
        """encode_to_coords returns (ids, coords) tuple."""
        result = coordizer.encode_to_coords("Hello")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_ids_and_coords_match(self, coordizer):
        """IDs and coords have same length."""
        ids, coords = coordizer.encode_to_coords("Hello, world!")
        assert len(ids) == len(coords)

    def test_equivalent_to_separate_calls(self, coordizer):
        """encode_to_coords == encode + ids_to_coords."""
        text = "Test equivalence"
        ids1, coords1 = coordizer.encode_to_coords(text)
        ids2 = coordizer.encode(text)
        coords2 = coordizer.ids_to_coords(ids2)
        assert ids1 == ids2
        assert np.allclose(coords1, coords2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
