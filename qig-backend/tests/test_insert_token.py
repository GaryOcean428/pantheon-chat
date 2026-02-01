#!/usr/bin/env python3
"""
Unit tests for canonical token insertion pathway.

Tests the insert_token() function and related validation logic.

Source: E8 Protocol Issue #97 (Issue-01: QFI Integrity Gate)
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vocabulary.insert_token import (
    insert_token,
    validate_token_integrity,
    TokenRecord,
    _fallback_compute_qfi,
    _fallback_to_simplex,
    _fallback_validate_simplex,
)


class TestSimplexValidation(unittest.TestCase):
    """Test simplex validation functions."""
    
    def test_fallback_to_simplex_valid(self):
        """Test simplex projection with valid input."""
        v = np.array([1, 2, 3, 4], dtype=np.float64)
        result = _fallback_to_simplex(v)
        
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(np.sum(result), 1.0, places=10)
        self.assertTrue(np.all(result >= 0))
    
    def test_fallback_to_simplex_negative(self):
        """Test simplex projection with negative input (should take abs)."""
        v = np.array([-1, 2, -3, 4], dtype=np.float64)
        result = _fallback_to_simplex(v)
        
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(np.sum(result), 1.0, places=10)
        self.assertTrue(np.all(result >= 0))
    
    def test_fallback_validate_simplex_valid(self):
        """Test validation of valid simplex."""
        p = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        is_valid, reason = _fallback_validate_simplex(p)
        
        self.assertTrue(is_valid)
        self.assertEqual(reason, "valid_simplex")
    
    def test_fallback_validate_simplex_negative(self):
        """Test validation rejects negative values."""
        p = np.array([0.5, -0.1, 0.4, 0.2], dtype=np.float64)
        is_valid, reason = _fallback_validate_simplex(p)
        
        self.assertFalse(is_valid)
        self.assertIn("negative", reason)
    
    def test_fallback_validate_simplex_wrong_sum(self):
        """Test validation rejects wrong sum."""
        p = np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float64)
        is_valid, reason = _fallback_validate_simplex(p)
        
        self.assertFalse(is_valid)
        self.assertIn("sum", reason)
    
    def test_fallback_validate_simplex_nan(self):
        """Test validation rejects NaN values."""
        p = np.array([0.5, np.nan, 0.3, 0.2], dtype=np.float64)
        is_valid, reason = _fallback_validate_simplex(p)
        
        self.assertFalse(is_valid)
        self.assertIn("nan", reason.lower())


class TestQFIComputation(unittest.TestCase):
    """Test QFI computation."""
    
    def test_fallback_compute_qfi_uniform(self):
        """Test QFI computation for uniform distribution."""
        # Uniform distribution should have high QFI (maximum entropy)
        basin = np.ones(64, dtype=np.float64) / 64
        qfi = _fallback_compute_qfi(basin)
        
        self.assertGreater(qfi, 0.0)
        self.assertLessEqual(qfi, 1.0)
        self.assertAlmostEqual(qfi, 1.0, places=2)  # Close to 1 for uniform
    
    def test_fallback_compute_qfi_peaked(self):
        """Test QFI computation for peaked distribution."""
        # Peaked distribution should have low QFI (low entropy)
        basin = np.zeros(64, dtype=np.float64)
        basin[0] = 1.0
        qfi = _fallback_compute_qfi(basin)
        
        self.assertGreater(qfi, 0.0)
        self.assertLessEqual(qfi, 1.0)
        self.assertLess(qfi, 0.1)  # Much less than 1 for peaked
    
    def test_fallback_compute_qfi_range(self):
        """Test QFI is always in [0, 1] range."""
        for _ in range(10):
            basin = np.random.rand(64)
            basin = basin / basin.sum()  # Normalize to simplex
            qfi = _fallback_compute_qfi(basin)
            
            self.assertGreaterEqual(qfi, 0.0)
            self.assertLessEqual(qfi, 1.0)


class TestTokenInsertion(unittest.TestCase):
    """Test token insertion logic (mocked database)."""
    
    @patch('vocabulary.insert_token.get_db_connection')
    def test_insert_token_valid_basin(self, mock_get_conn):
        """Test inserting token with valid basin."""
        # Mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock cursor results
        mock_cursor.fetchone.side_effect = [
            None,  # Token doesn't exist
            {'updated_at': '2026-01-20 00:00:00'}  # Updated timestamp
        ]
        
        # Create valid 64D basin
        basin = np.random.rand(64)
        basin = basin / basin.sum()
        
        # Insert token
        record = insert_token(
            "test_token",
            basin,
            token_role="test",
            frequency=1,
            is_real_word=True
        )
        
        # Verify record
        self.assertEqual(record.token, "test_token")
        self.assertIsNotNone(record.qfi_score)
        self.assertGreater(record.qfi_score, 0.0)
        self.assertTrue(record.is_generation_eligible)
        self.assertEqual(len(record.basin_embedding), 64)
    
    def test_insert_token_wrong_dimension(self):
        """Test inserting token with wrong dimension raises error."""
        basin = np.random.rand(32)  # Wrong dimension
        
        with self.assertRaises(ValueError) as context:
            insert_token("test_token", basin)
        
        self.assertIn("64D", str(context.exception))
    
    def test_insert_token_nan_basin(self):
        """Test inserting token with NaN values raises error."""
        basin = np.random.rand(64)
        basin[0] = np.nan
        
        with self.assertRaises(ValueError) as context:
            insert_token("test_token", basin)
        
        self.assertIn("NaN", str(context.exception))
    
    def test_insert_token_inf_basin(self):
        """Test inserting token with Inf values raises error."""
        basin = np.random.rand(64)
        basin[0] = np.inf
        
        with self.assertRaises(ValueError) as context:
            insert_token("test_token", basin)
        
        self.assertIn("Inf", str(context.exception))


class TestTokenValidation(unittest.TestCase):
    """Test token integrity validation (mocked database)."""
    
    @patch('vocabulary.insert_token.get_db_connection')
    def test_validate_token_integrity_valid(self, mock_get_conn):
        """Test validating a token with valid data."""
        # Mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock cursor result
        basin = np.random.rand(64)
        basin = basin / basin.sum()
        mock_cursor.fetchone.return_value = {
            'basin_embedding': basin.tolist(),
            'qfi_score': 0.5,
            'is_generation_eligible': True,
            'is_real_word': True
        }
        
        # Validate token
        result = validate_token_integrity("test_token")
        
        # Verify result
        self.assertTrue(result['exists'])
        self.assertTrue(result['has_basin'])
        self.assertTrue(result['has_qfi'])
        self.assertTrue(result['is_generation_eligible'])
        self.assertEqual(len(result['issues']), 0)
    
    @patch('vocabulary.insert_token.get_db_connection')
    def test_validate_token_integrity_missing_qfi(self, mock_get_conn):
        """Test validating a token with missing QFI."""
        # Mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock cursor result
        basin = np.random.rand(64)
        basin = basin / basin.sum()
        mock_cursor.fetchone.return_value = {
            'basin_embedding': basin.tolist(),
            'qfi_score': None,
            'is_generation_eligible': False,
            'is_real_word': True
        }
        
        # Validate token
        result = validate_token_integrity("test_token")
        
        # Verify result
        self.assertTrue(result['exists'])
        self.assertTrue(result['has_basin'])
        self.assertFalse(result['has_qfi'])
        self.assertIn('Missing qfi_score', result['issues'])
    
    @patch('vocabulary.insert_token.get_db_connection')
    def test_validate_token_integrity_not_found(self, mock_get_conn):
        """Test validating a token that doesn't exist."""
        # Mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock cursor result
        mock_cursor.fetchone.return_value = None
        
        # Validate token
        result = validate_token_integrity("nonexistent_token")
        
        # Verify result
        self.assertFalse(result['exists'])
        self.assertIn('Token not found', result['issues'][0])


if __name__ == '__main__':
    # Set DATABASE_URL to dummy value to avoid connection attempts
    os.environ['DATABASE_URL'] = 'postgresql://dummy:dummy@localhost/dummy'
    
    unittest.main()
