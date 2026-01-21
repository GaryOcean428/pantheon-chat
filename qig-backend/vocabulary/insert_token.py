#!/usr/bin/env python3
"""
Canonical Token Insertion Pathway - SINGLE SOURCE OF TRUTH

This module provides the ONLY valid pathway for inserting tokens into
the coordizer_vocabulary table. All code paths MUST route through insert_token().

GEOMETRIC PURITY REQUIREMENTS:
- Basin MUST be 64D probability simplex (non-negative, sum=1)
- QFI score MUST be computed before insertion
- All geometric validation enforced at insertion time
- No direct INSERTs to coordizer_vocabulary allowed

Source: E8 Protocol Issue #97 (Issue-01: QFI Integrity Gate)
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Import canonical simplex operations
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from qig_geometry.geometry_simplex import (
        to_simplex_prob,
        validate_simplex,
    )
    SIMPLEX_OPS_AVAILABLE = True
except ImportError:
    SIMPLEX_OPS_AVAILABLE = False
    logger.warning("[insert_token] qig_geometry not available - using fallback validation")


# Import canonical QFI computation
try:
    from qig_geometry.canonical_upsert import compute_qfi_score
    QFI_COMPUTE_AVAILABLE = True
except ImportError:
    QFI_COMPUTE_AVAILABLE = False
    logger.warning("[insert_token] canonical_upsert not available - using fallback QFI")


# Database connection
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("[insert_token] psycopg2 not available - database operations disabled")


BASIN_DIMENSION = 64

# Numerical stability constants
EPSILON_SMALL = 1e-10  # For probability clipping and zero checks
EPSILON_TINY = 1e-12   # For simplex normalization
SIMPLEX_TOLERANCE = 1e-6  # For simplex validation


@dataclass
class TokenRecord:
    """Result of token insertion operation."""
    token: str
    basin_embedding: list
    qfi_score: float
    token_role: Optional[str]
    frequency: int
    is_real_word: bool
    is_generation_eligible: bool
    updated_at: datetime
    created: bool  # True if newly created, False if updated


def _fallback_compute_qfi(basin: np.ndarray) -> float:
    """
    Fallback QFI computation using participation ratio.
    
    QFI = exp(H(p)) / n where H(p) is Shannon entropy.
    This is the CANONICAL formula - produces values in [0, 1].
    """
    # Project to simplex probabilities
    v = np.abs(basin) + EPSILON_SMALL
    p = v / v.sum()
    
    # Compute Shannon entropy
    positive_probs = p[p > EPSILON_SMALL]
    if len(positive_probs) == 0:
        return 0.0
    
    entropy = -np.sum(positive_probs * np.log(positive_probs + EPSILON_SMALL))
    
    # Participation ratio = exp(entropy) / dimension
    n_dim = len(basin)
    effective_dim = np.exp(entropy)
    qfi_score = effective_dim / n_dim
    
    return float(np.clip(qfi_score, 0.0, 1.0))


def _fallback_to_simplex(v: np.ndarray) -> np.ndarray:
    """Fallback simplex projection."""
    v = np.asarray(v, dtype=np.float64).flatten()
    w = np.abs(v) + EPSILON_TINY
    return w / w.sum()


def _fallback_validate_simplex(p: np.ndarray, tolerance: float = SIMPLEX_TOLERANCE) -> tuple:
    """Fallback simplex validation."""
    if p is None or p.size == 0:
        return False, "empty_vector"
    
    p = np.asarray(p, dtype=np.float64).flatten()
    
    if not np.all(np.isfinite(p)):
        return False, "contains_nan_or_inf"
    
    if np.min(p) < -tolerance:
        return False, f"negative_values_min={np.min(p):.6f}"
    
    total = np.sum(p)
    if abs(total - 1.0) > tolerance:
        return False, f"sum_not_one_{total:.6f}"
    
    return True, "valid_simplex"


def get_db_connection():
    """Get PostgreSQL connection."""
    if not DB_AVAILABLE:
        raise RuntimeError("Database not available - psycopg2 not installed")
    
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable not set")
    
    return psycopg2.connect(database_url)


def insert_token(
    token: str,
    basin: np.ndarray,
    token_role: Optional[str] = None,
    frequency: int = 1,
    is_real_word: bool = True,
) -> TokenRecord:
    """
    Canonical token insertion - ONLY pathway for adding vocabulary.
    
    This function is the SINGLE SOURCE OF TRUTH for token insertion.
    ALL code paths that add tokens to the vocabulary MUST route through here.
    
    Steps:
    1. Validate basin is 64D
    2. Project basin to simplex (canonical representation)
    3. Compute QFI score (using quantum_fisher_information)
    4. Set flags (is_real_word, token_role)
    5. Write row atomically to database
    6. Return record with qfi_score populated
    
    Args:
        token: Token string to insert
        basin: 64D basin coordinates (will be projected to simplex)
        token_role: Optional geometric role (e.g., "basin_center", "boundary_crosser")
        frequency: Initial frequency count
        is_real_word: Whether this is a real word (vs. BPE artifact)
        
    Returns:
        TokenRecord with insertion results
        
    Raises:
        ValueError: if basin is not 64D or contains NaN/Inf
        RuntimeError: if database operation fails
        
    Example:
        >>> basin = np.random.rand(64)
        >>> record = insert_token("example", basin, token_role="noun")
        >>> assert record.qfi_score is not None
        >>> assert record.qfi_score > 0
    """
    # Step 1: Validate basin dimension
    if not isinstance(basin, np.ndarray):
        basin = np.array(basin, dtype=np.float64)
    
    basin = basin.flatten()
    
    if len(basin) != BASIN_DIMENSION:
        raise ValueError(
            f"Basin must be {BASIN_DIMENSION}D, got {len(basin)}D"
        )
    
    if not np.all(np.isfinite(basin)):
        raise ValueError(
            f"Basin contains NaN or Inf values"
        )
    
    # Step 2: Project to simplex (canonical representation)
    if SIMPLEX_OPS_AVAILABLE:
        basin_simplex = to_simplex_prob(basin)
        is_valid, reason = validate_simplex(basin_simplex)
    else:
        basin_simplex = _fallback_to_simplex(basin)
        is_valid, reason = _fallback_validate_simplex(basin_simplex)
    
    if not is_valid:
        raise ValueError(
            f"Simplex validation failed: {reason}"
        )
    
    # Step 3: Compute QFI score
    if QFI_COMPUTE_AVAILABLE:
        qfi_score = compute_qfi_score(basin_simplex)
    else:
        qfi_score = _fallback_compute_qfi(basin_simplex)
    
    if not np.isfinite(qfi_score):
        raise ValueError(
            f"QFI computation produced invalid value: {qfi_score}"
        )
    
    # Step 4: Determine generation eligibility
    # Token is eligible if it has valid QFI and is a real word
    is_generation_eligible = np.isfinite(qfi_score) and is_real_word
    
    # Step 5: Write to database atomically
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check if token already exists
            cur.execute(
                "SELECT token FROM coordizer_vocabulary WHERE token = %s",
                (token,)
            )
            exists = cur.fetchone() is not None
            
            # Prepare basin as PostgreSQL array
            basin_list = basin_simplex.tolist()
            
            # Upsert token
            cur.execute("""
                INSERT INTO coordizer_vocabulary (
                    token,
                    basin_embedding,
                    qfi_score,
                    token_role,
                    frequency,
                    is_real_word,
                    is_generation_eligible,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (token) DO UPDATE SET
                    basin_embedding = EXCLUDED.basin_embedding,
                    qfi_score = EXCLUDED.qfi_score,
                    token_role = COALESCE(EXCLUDED.token_role, coordizer_vocabulary.token_role),
                    frequency = coordizer_vocabulary.frequency + EXCLUDED.frequency,
                    is_real_word = EXCLUDED.is_real_word,
                    is_generation_eligible = EXCLUDED.is_generation_eligible,
                    updated_at = NOW()
                RETURNING updated_at
            """, (
                token,
                basin_list,
                float(qfi_score),
                token_role,
                frequency,
                is_real_word,
                is_generation_eligible
            ))
            
            result = cur.fetchone()
            updated_at = result['updated_at'] if result else datetime.now()
            
            conn.commit()
            
            logger.info(
                f"[insert_token] {'Updated' if exists else 'Inserted'} token: {token} "
                f"(qfi={qfi_score:.4f}, role={token_role}, eligible={is_generation_eligible})"
            )
            
            return TokenRecord(
                token=token,
                basin_embedding=basin_list,
                qfi_score=float(qfi_score),
                token_role=token_role,
                frequency=frequency,
                is_real_word=is_real_word,
                is_generation_eligible=is_generation_eligible,
                updated_at=updated_at,
                created=not exists
            )
    
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"[insert_token] Failed to insert token {token}: {e}")
        raise RuntimeError(f"Token insertion failed: {e}") from e
    
    finally:
        if conn:
            conn.close()


def validate_token_integrity(token: str) -> dict:
    """
    Validate that a token in the database has proper QFI and geometric properties.
    
    Args:
        token: Token string to validate
        
    Returns:
        Dict with validation results:
        - has_basin: bool
        - has_qfi: bool
        - is_generation_eligible: bool
        - qfi_score: float or None
        - basin_valid: bool
        - issues: list of issue strings
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    basin_embedding,
                    qfi_score,
                    is_generation_eligible,
                    is_real_word
                FROM coordizer_vocabulary
                WHERE token = %s
            """, (token,))
            
            row = cur.fetchone()
            
            if not row:
                return {
                    'exists': False,
                    'issues': ['Token not found in vocabulary']
                }
            
            issues = []
            
            # Check basin
            has_basin = row['basin_embedding'] is not None
            if not has_basin:
                issues.append('Missing basin_embedding')
            
            # Check QFI
            has_qfi = row['qfi_score'] is not None
            if not has_qfi:
                issues.append('Missing qfi_score')
            
            # Validate basin if present
            basin_valid = False
            if has_basin:
                basin = np.array(row['basin_embedding'], dtype=np.float64)
                if SIMPLEX_OPS_AVAILABLE:
                    basin_valid, reason = validate_simplex(basin)
                else:
                    basin_valid, reason = _fallback_validate_simplex(basin)
                
                if not basin_valid:
                    issues.append(f'Invalid basin: {reason}')
            
            # Check generation eligibility consistency
            is_eligible = row['is_generation_eligible']
            is_real = row['is_real_word']
            
            if is_eligible and not has_qfi:
                issues.append('Generation-eligible but missing QFI')
            
            if is_eligible and not has_basin:
                issues.append('Generation-eligible but missing basin')
            
            if is_eligible and not is_real:
                issues.append('Generation-eligible but not marked as real word')
            
            return {
                'exists': True,
                'has_basin': has_basin,
                'has_qfi': has_qfi,
                'is_generation_eligible': is_eligible,
                'qfi_score': row['qfi_score'],
                'basin_valid': basin_valid,
                'is_real_word': is_real,
                'issues': issues
            }
    
    except Exception as e:
        logger.error(f"[validate_token_integrity] Failed to validate token {token}: {e}")
        return {
            'exists': False,
            'error': str(e),
            'issues': [f'Validation error: {e}']
        }
    
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    # Simple test
    import sys
    
    def test_insert():
        """Test token insertion."""
        # Create a random 64D basin
        basin = np.random.rand(64)
        
        # Insert token
        record = insert_token(
            "test_token",
            basin,
            token_role="test",
            frequency=1,
            is_real_word=True
        )
        
        print(f"Inserted token: {record.token}")
        print(f"QFI score: {record.qfi_score:.4f}")
        print(f"Generation eligible: {record.is_generation_eligible}")
        print(f"Created new: {record.created}")
        
        # Validate
        validation = validate_token_integrity("test_token")
        print(f"Validation: {validation}")
    
    # Run test if DATABASE_URL is set
    if os.environ.get('DATABASE_URL'):
        test_insert()
    else:
        print("DATABASE_URL not set - skipping test", file=sys.stderr)
