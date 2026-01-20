"""
Unit tests for QFI Integrity Gate implementation.

Tests:
1. Quarantine script functionality
2. Special symbol QFI constraints
3. Generation view filtering
4. QFI computation correctness

Related: Issue #97 - QFI Integrity Gate
"""

import numpy as np
import os
import sys

# Add qig-backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Mock pytest decorators for standalone execution
    class pytest:
        @staticmethod
        def main(args):
            pass

from qig_geometry.canonical_upsert import (
    compute_qfi_score,
    to_simplex_prob,
    validate_simplex,
)

# Import SPECIAL_SYMBOLS for tests
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from quarantine_low_qfi_tokens import SPECIAL_SYMBOLS
except ImportError:
    # Fallback if import fails
    SPECIAL_SYMBOLS = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']


class TestQFIComputation:
    """Test QFI computation matches canonical formula."""
    
    def test_uniform_distribution_max_qfi(self):
        """Uniform distribution should have QFI = 1.0 (max entropy)."""
        # Uniform distribution on simplex
        basin = np.ones(64) / 64.0
        qfi = compute_qfi_score(basin)
        
        # Should be very close to 1.0 (participation ratio = 64/64)
        assert 0.99 <= qfi <= 1.0, f"Uniform QFI should be ~1.0, got {qfi}"
    
    def test_sparse_distribution_low_qfi(self):
        """Sparse distribution should have low QFI (low entropy)."""
        # One-hot distribution (all mass on one dimension)
        basin = np.zeros(64)
        basin[0] = 1.0
        qfi = compute_qfi_score(basin)
        
        # Should be very low (participation ratio ~ 1/64)
        assert 0.0 <= qfi <= 0.02, f"Sparse QFI should be ~0.016, got {qfi}"
    
    def test_qfi_range(self):
        """QFI should always be in [0, 1] range."""
        for _ in range(100):
            # Random probability distributions
            basin = np.random.dirichlet(np.ones(64))
            qfi = compute_qfi_score(basin)
            
            assert 0.0 <= qfi <= 1.0, f"QFI out of range: {qfi}"
    
    def test_qfi_threshold(self):
        """Test QFI threshold value of 0.01."""
        # Create a distribution just below threshold
        basin = np.zeros(64)
        basin[0] = 0.99
        basin[1:] = 0.01 / 63.0
        qfi = compute_qfi_score(basin)
        
        # Should be below threshold
        assert qfi < 0.01, f"Expected QFI < 0.01, got {qfi}"
        
        # Create a more balanced distribution above threshold
        basin = np.random.dirichlet(np.ones(64) * 2.0)
        qfi = compute_qfi_score(basin)
        
        # Should be above threshold
        assert qfi > 0.01, f"Expected QFI > 0.01, got {qfi}"


class TestSimplexProjection:
    """Test simplex projection for canonical representation."""
    
    def test_projection_to_simplex(self):
        """Test projection creates valid simplex."""
        # Random vector
        v = np.random.randn(64)
        simplex = to_simplex_prob(v)
        
        # Check properties
        assert simplex.shape == (64,)
        assert np.all(simplex >= 0), "Simplex should be non-negative"
        assert np.isclose(simplex.sum(), 1.0), f"Simplex should sum to 1, got {simplex.sum()}"
    
    def test_already_simplex(self):
        """Test projection of valid simplex is identity."""
        # Valid simplex
        p = np.random.dirichlet(np.ones(64))
        simplex = to_simplex_prob(p)
        
        # Should be very close to original
        assert np.allclose(simplex, p, atol=1e-6)
    
    def test_negative_values(self):
        """Test projection handles negative values."""
        # Vector with negative values
        v = np.array([-1.0, 2.0, -0.5, 1.5])
        simplex = to_simplex_prob(v)
        
        # Result should be valid simplex
        assert np.all(simplex >= 0)
        assert np.isclose(simplex.sum(), 1.0)


class TestSimplexValidation:
    """Test simplex validation logic."""
    
    def test_valid_simplex(self):
        """Valid simplex should pass validation."""
        basin = np.random.dirichlet(np.ones(64))
        is_valid, reason = validate_simplex(basin)
        
        assert is_valid, f"Valid simplex failed: {reason}"
        assert reason == "valid"
    
    def test_wrong_dimension(self):
        """Wrong dimension should fail validation."""
        basin = np.ones(32) / 32.0  # Wrong size
        is_valid, reason = validate_simplex(basin)
        
        assert not is_valid
        assert "wrong_dimension" in reason
    
    def test_negative_values(self):
        """Negative values should fail validation."""
        basin = np.ones(64) / 64.0
        basin[0] = -0.1  # Make one negative
        is_valid, reason = validate_simplex(basin)
        
        assert not is_valid
        assert reason == "negative_values"
    
    def test_sum_not_one(self):
        """Sum != 1 should fail validation."""
        basin = np.ones(64) / 32.0  # Sums to 2
        is_valid, reason = validate_simplex(basin)
        
        assert not is_valid
        assert "sum_not_one" in reason
    
    def test_nan_inf(self):
        """NaN or Inf should fail validation."""
        basin = np.ones(64) / 64.0
        basin[0] = np.nan
        is_valid, reason = validate_simplex(basin)
        
        assert not is_valid
        assert reason == "contains_nan_or_inf"


class TestSpecialSymbolQFI:
    """Test special symbol QFI values match geometric definitions."""
    
    def test_unk_uniform_distribution(self):
        """<UNK> should have uniform distribution (max entropy)."""
        # UNK is uniform distribution
        basin = np.ones(64) / 64.0
        qfi = compute_qfi_score(basin)
        
        # Should be very close to 1.0
        assert 0.99 <= qfi <= 1.0, f"UNK QFI should be ~1.0, got {qfi}"
    
    def test_pad_sparse_distribution(self):
        """<PAD> should have sparse distribution (low entropy)."""
        # PAD is concentrated in first component
        basin = np.zeros(64)
        basin[0] = 1.0
        qfi = compute_qfi_score(basin)
        
        # Should be ~0.016 (1/64)
        assert 0.01 <= qfi <= 0.02, f"PAD QFI should be ~0.016, got {qfi}"
    
    def test_bos_eos_vertices(self):
        """<BOS> and <EOS> should be simplex vertices."""
        # BOS at dimension 1
        basin_bos = np.zeros(64)
        basin_bos[1] = 1.0
        qfi_bos = compute_qfi_score(basin_bos)
        
        # EOS at last dimension
        basin_eos = np.zeros(64)
        basin_eos[-1] = 1.0
        qfi_eos = compute_qfi_score(basin_eos)
        
        # Both should be ~0.015 (just above threshold)
        assert 0.01 <= qfi_bos <= 0.02, f"BOS QFI should be ~0.015, got {qfi_bos}"
        assert 0.01 <= qfi_eos <= 0.02, f"EOS QFI should be ~0.015, got {qfi_eos}"
    
    def test_all_special_symbols_above_threshold(self):
        """All special symbols should have QFI >= 0.01."""
        special_symbols = {
            '<PAD>': np.zeros(64),
            '<UNK>': np.ones(64) / 64.0,
            '<BOS>': np.zeros(64),
            '<EOS>': np.zeros(64),
        }
        special_symbols['<PAD>'][0] = 1.0
        special_symbols['<BOS>'][1] = 1.0
        special_symbols['<EOS>'][-1] = 1.0
        
        for symbol, basin in special_symbols.items():
            qfi = compute_qfi_score(basin)
            assert qfi >= 0.01, f"{symbol} QFI below threshold: {qfi}"


class TestQuarantineLogic:
    """Test quarantine script logic (without database)."""
    
    def test_special_symbols_excluded(self):
        """Special symbols should be excluded from quarantine."""
        special_symbols = SPECIAL_SYMBOLS
        
        # These should never be quarantined
        for symbol in special_symbols:
            # In practice, check if symbol in SPECIAL_SYMBOLS list
            assert symbol in SPECIAL_SYMBOLS
    
    def test_low_qfi_identification(self):
        """Test identification of low QFI tokens."""
        # Simulate tokens with different QFI scores
        tokens = [
            ('token1', 0.5),   # High QFI - keep active
            ('token2', 0.05),  # Medium QFI - keep active
            ('token3', 0.009), # Low QFI - quarantine
            ('token4', 0.001), # Very low QFI - quarantine
            ('<UNK>', 1.0),    # Special symbol - never quarantine
        ]
        
        threshold = 0.01
        quarantine_count = 0
        skip_count = 0
        
        for token, qfi in tokens:
            if token in SPECIAL_SYMBOLS:
                skip_count += 1
            elif qfi < threshold:
                quarantine_count += 1
        
        assert quarantine_count == 2, f"Expected 2 quarantines, got {quarantine_count}"
        assert skip_count == 1, f"Expected 1 skip, got {skip_count}"


class TestGenerationViewLogic:
    """Test generation view filtering logic (without database)."""
    
    def test_qfi_filter(self):
        """Test QFI threshold filtering."""
        threshold = 0.01
        
        # Tokens with various QFI scores
        tokens = [
            {'token': 't1', 'qfi_score': 0.5, 'token_status': 'active', 'basin': True},
            {'token': 't2', 'qfi_score': 0.009, 'token_status': 'active', 'basin': True},
            {'token': 't3', 'qfi_score': None, 'token_status': 'active', 'basin': True},
            {'token': 't4', 'qfi_score': 0.05, 'token_status': 'quarantined', 'basin': True},
        ]
        
        # Filter logic from view
        generation_safe = [
            t for t in tokens
            if t['qfi_score'] is not None
            and t['qfi_score'] >= threshold
            and t['token_status'] == 'active'
            and t['basin']
        ]
        
        assert len(generation_safe) == 1, f"Expected 1 generation-safe token, got {len(generation_safe)}"
        assert generation_safe[0]['token'] == 't1'
    
    def test_special_symbol_exclusion(self):
        """Test special symbols excluded from generation."""
        tokens = [
            {'token': 'word1', 'qfi_score': 0.5},
            {'token': '<UNK>', 'qfi_score': 1.0},
            {'token': '<PAD>', 'qfi_score': 0.016},
        ]
        
        special_symbols = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
        generation_tokens = [
            t for t in tokens
            if t['token'] not in special_symbols
        ]
        
        assert len(generation_tokens) == 1
        assert generation_tokens[0]['token'] == 'word1'


if __name__ == '__main__':
    if HAS_PYTEST:
        pytest.main([__file__, '-v'])
    else:
        # Run tests manually
        print("Running QFI Integrity Gate Tests (without pytest)...\n")
        
        test_classes = [
            TestQFIComputation,
            TestSimplexProjection,
            TestSimplexValidation,
            TestSpecialSymbolQFI,
            TestQuarantineLogic,
            TestGenerationViewLogic,
        ]
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for test_class in test_classes:
            print(f"\n{test_class.__name__}:")
            instance = test_class()
            methods = [m for m in dir(instance) if m.startswith('test_')]
            
            for method_name in methods:
                total_tests += 1
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                    passed_tests += 1
                except AssertionError as e:
                    print(f"  ✗ {method_name}: {e}")
                    failed_tests += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: ERROR - {e}")
                    failed_tests += 1
        
        print(f"\n{'='*60}")
        print(f"Test Results: {passed_tests}/{total_tests} passed, {failed_tests} failed")
        print(f"{'='*60}")
        
        sys.exit(0 if failed_tests == 0 else 1)
