#!/usr/bin/env python3
"""
Canonical Token Insertion Module
=================================

SINGLE SOURCE OF TRUTH for inserting tokens into coordizer_vocabulary.

All code that adds tokens MUST use this module. Direct INSERT statements are forbidden.

This ensures:
1. QFI score is computed before insertion (required for generation eligibility)
2. Basin coordinates are validated (simplex: non-negative, sum=1, dim=64)
3. Consistent insertion logic across the codebase
4. Audit trail for all vocabulary additions

Usage:
    from vocabulary.insert_token import canonical_insert_token
    
    result = canonical_insert_token(
        word="example",
        basin_coords=np.array([...]),  # 64D simplex
        source_type="learned",
        frequency=1,
        db_conn=conn
    )
    
    if result['success']:
        print(f"Inserted token_id={result['token_id']}")
    else:
        print(f"Error: {result['error']}")

Author: Copilot AI Agent
Date: 2026-01-20
Issue: GaryOcean428/pantheon-chat#97 (E8 Protocol Issue-01)
Reference: docs/10-e8-protocol/issues/20260119-issue-97-qfi-integrity-gate-remediation-1.00W.md
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
import psycopg2
from psycopg2.extensions import connection as PgConnection

logger = logging.getLogger(__name__)

# Import canonical QFI computation
try:
    from qig_geometry import compute_qfi_score
except ImportError:
    logger.warning("qig_geometry not available, using fallback QFI computation")
    
    def compute_qfi_score(basin: np.ndarray) -> float:
        """Fallback QFI computation using participation ratio."""
        v = np.abs(basin) + 1e-10
        p = v / v.sum()
        positive_probs = p[p > 1e-10]
        if len(positive_probs) == 0:
            return 0.0
        entropy = -np.sum(positive_probs * np.log(positive_probs + 1e-10))
        effective_dim = np.exp(entropy)
        qfi_score = effective_dim / len(basin)
        return float(np.clip(qfi_score, 0.0, 1.0))


BASIN_DIM = 64
MIN_QFI = 0.0
MAX_QFI = 1.0


def validate_basin(basin: np.ndarray) -> Dict[str, Any]:
    """
    Validate basin coordinates are valid simplex.
    
    Requirements:
    - Non-negative values
    - Sum to 1 (within tolerance)
    - Correct dimension (64D)
    
    Args:
        basin: Basin coordinates to validate
        
    Returns:
        Dict with 'valid' (bool) and 'error' (str) keys
    """
    try:
        if basin is None:
            return {'valid': False, 'error': 'Basin is None'}
        
        basin = np.asarray(basin, dtype=np.float64).flatten()
        
        if len(basin) != BASIN_DIM:
            return {
                'valid': False,
                'error': f'Invalid dimension: {len(basin)} (expected {BASIN_DIM})'
            }
        
        if np.any(basin < 0):
            return {
                'valid': False,
                'error': f'Negative values in basin: min={basin.min()}'
            }
        
        total = basin.sum()
        if not np.isclose(total, 1.0, atol=1e-3):
            return {
                'valid': False,
                'error': f'Basin does not sum to 1: sum={total}'
            }
        
        if not np.all(np.isfinite(basin)):
            return {
                'valid': False,
                'error': 'Basin contains inf or nan'
            }
        
        return {'valid': True, 'error': None}
        
    except Exception as e:
        return {'valid': False, 'error': f'Validation exception: {str(e)}'}


def canonical_insert_token(
    word: str,
    basin_coords: np.ndarray,
    db_conn: PgConnection,
    source_type: str = "learned",
    frequency: int = 1,
    token_id: Optional[int] = None,
    phi_score: Optional[float] = None,
    force: bool = False
) -> Dict[str, Any]:
    """
    Canonical token insertion with QFI computation and validation.
    
    This is the ONLY way to insert tokens into coordizer_vocabulary.
    All other code must call this function.
    
    Steps:
    1. Validate word (non-empty, reasonable length)
    2. Validate basin coordinates (simplex)
    3. Compute QFI score from basin
    4. Insert or update record in database
    5. Return result with token_id
    
    Args:
        word: Token string (1-50 chars)
        basin_coords: 64D simplex coordinates
        db_conn: PostgreSQL connection
        source_type: Source of token ('learned', 'curriculum', 'manual')
        frequency: Initial frequency count
        token_id: Optional token_id for deterministic insertion
        phi_score: Optional Φ score (computed if not provided)
        force: If True, skip some validation (use with caution)
        
    Returns:
        Dict with:
            - success (bool): Whether insertion succeeded
            - token_id (int): Database ID of inserted token
            - qfi_score (float): Computed QFI score
            - error (str): Error message if failed
    """
    try:
        # Validate word
        if not word or not isinstance(word, str):
            return {'success': False, 'error': 'Invalid word: must be non-empty string'}
        
        word = word.strip()
        if len(word) < 1 or len(word) > 50:
            return {'success': False, 'error': f'Invalid word length: {len(word)} (must be 1-50)'}
        
        # Validate basin
        validation = validate_basin(basin_coords)
        if not validation['valid'] and not force:
            return {'success': False, 'error': f'Basin validation failed: {validation["error"]}'}
        
        # Normalize basin to ensure exact simplex
        basin = np.asarray(basin_coords, dtype=np.float64).flatten()
        basin = np.maximum(basin, 0) + 1e-10
        basin = basin / basin.sum()
        
        # Compute QFI score (REQUIRED for generation eligibility)
        qfi_score = compute_qfi_score(basin)
        
        if not (MIN_QFI <= qfi_score <= MAX_QFI) and not force:
            return {
                'success': False,
                'error': f'Invalid QFI score: {qfi_score} (must be in [0, 1])'
            }
        
        # Compute phi_score if not provided (fallback to QFI)
        if phi_score is None:
            phi_score = qfi_score
        
        # Format basin for PostgreSQL vector type
        basin_str = '[' + ','.join(str(x) for x in basin) + ']'
        
        # Insert or update token
        with db_conn.cursor() as cursor:
            if token_id is not None:
                # Deterministic insertion with specific token_id
                cursor.execute("""
                    INSERT INTO coordizer_vocabulary (
                        token_id, token, basin_embedding, qfi_score, phi_score,
                        frequency, source_type, token_status, created_at, updated_at
                    ) VALUES (
                        %s, %s, %s::vector, %s, %s,
                        %s, %s, 'active', NOW(), NOW()
                    )
                    ON CONFLICT (token_id) DO UPDATE SET
                        token = EXCLUDED.token,
                        basin_embedding = EXCLUDED.basin_embedding,
                        qfi_score = COALESCE(EXCLUDED.qfi_score, coordizer_vocabulary.qfi_score),
                        phi_score = COALESCE(EXCLUDED.phi_score, coordizer_vocabulary.phi_score),
                        frequency = coordizer_vocabulary.frequency + EXCLUDED.frequency,
                        updated_at = NOW()
                    RETURNING token_id
                """, (token_id, word, basin_str, qfi_score, phi_score, frequency, source_type))
            else:
                # Natural insertion with auto-generated token_id
                cursor.execute("""
                    INSERT INTO coordizer_vocabulary (
                        token, basin_embedding, qfi_score, phi_score,
                        frequency, source_type, token_status, created_at, updated_at
                    ) VALUES (
                        %s, %s::vector, %s, %s,
                        %s, %s, 'active', NOW(), NOW()
                    )
                    ON CONFLICT (token) DO UPDATE SET
                        basin_embedding = COALESCE(EXCLUDED.basin_embedding, coordizer_vocabulary.basin_embedding),
                        qfi_score = COALESCE(EXCLUDED.qfi_score, coordizer_vocabulary.qfi_score),
                        phi_score = COALESCE(EXCLUDED.phi_score, coordizer_vocabulary.phi_score),
                        frequency = coordizer_vocabulary.frequency + EXCLUDED.frequency,
                        updated_at = NOW()
                    RETURNING token_id
                """, (word, basin_str, qfi_score, phi_score, frequency, source_type))
            
            result = cursor.fetchone()
            if result:
                inserted_token_id = result[0]
            else:
                # Fallback: fetch existing token_id
                cursor.execute("SELECT token_id FROM coordizer_vocabulary WHERE token = %s", (word,))
                result = cursor.fetchone()
                inserted_token_id = result[0] if result else None
        
        db_conn.commit()
        
        logger.info(f"Inserted token '{word}' with token_id={inserted_token_id}, qfi={qfi_score:.4f}")
        
        return {
            'success': True,
            'token_id': inserted_token_id,
            'qfi_score': qfi_score,
            'phi_score': phi_score,
            'word': word
        }
        
    except psycopg2.IntegrityError as e:
        db_conn.rollback()
        logger.error(f"Integrity error inserting token '{word}': {e}")
        return {'success': False, 'error': f'Database integrity error: {str(e)}'}
    
    except Exception as e:
        db_conn.rollback()
        logger.error(f"Error inserting token '{word}': {e}")
        return {'success': False, 'error': f'Insertion failed: {str(e)}'}


def batch_insert_tokens(
    tokens: list,
    db_conn: PgConnection,
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Insert multiple tokens in batches.
    
    Args:
        tokens: List of dicts with keys: word, basin_coords, source_type, frequency
        db_conn: PostgreSQL connection
        batch_size: Number of tokens per batch
        
    Returns:
        Dict with:
            - success (bool): Overall success
            - inserted (int): Number of tokens inserted
            - failed (int): Number of tokens failed
            - errors (list): List of error messages
    """
    inserted = 0
    failed = 0
    errors = []
    
    for i in range(0, len(tokens), batch_size):
        batch = tokens[i:i+batch_size]
        
        for token_data in batch:
            result = canonical_insert_token(
                word=token_data['word'],
                basin_coords=token_data['basin_coords'],
                db_conn=db_conn,
                source_type=token_data.get('source_type', 'learned'),
                frequency=token_data.get('frequency', 1),
                token_id=token_data.get('token_id'),
                phi_score=token_data.get('phi_score')
            )
            
            if result['success']:
                inserted += 1
            else:
                failed += 1
                errors.append(f"{token_data['word']}: {result['error']}")
        
        # Commit after each batch
        try:
            db_conn.commit()
        except Exception as e:
            db_conn.rollback()
            logger.error(f"Batch commit failed: {e}")
    
    return {
        'success': failed == 0,
        'inserted': inserted,
        'failed': failed,
        'errors': errors
    }
