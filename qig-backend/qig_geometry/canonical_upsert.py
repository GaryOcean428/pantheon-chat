"""
Canonical Token Upsert - SLEEP-PACKET Section 4D Implementation

This is the SINGLE SOURCE OF TRUTH for all coordizer_vocabulary writes.
All INSERT/UPDATE operations MUST route through upsert_token().

ENFORCES:
- Basin projected to simplex (∑p_i = 1, p_i ≥ 0)
- QFI computed when basin is present
- Quarantine if QFI computation fails
- Active tokens require non-null qfi_score
"""

import os
import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None
    RealDictCursor = None
    PSYCOPG2_AVAILABLE = False


def to_simplex_prob(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Project vector to probability simplex (canonical storage form).
    
    SLEEP-PACKET Section 4A: Explicit simplex projection, no autodetect.
    
    Args:
        v: Any vector (amplitudes, embeddings, etc.)
        eps: Small constant for numerical stability
        
    Returns:
        Simplex probabilities: p_i ≥ 0, ∑p_i = 1
    """
    v = np.asarray(v, dtype=np.float64)
    v = np.abs(v) + eps
    return v / v.sum()


def compute_qfi_score(basin: np.ndarray) -> float:
    """
    Compute Quantum Fisher Information score for a simplex basin.
    
    Uses participation ratio (effective dimension) which is geometrically proper:
    QFI = exp(H(p)) / n where H(p) is Shannon entropy.
    
    Args:
        basin: 64D simplex probabilities
        
    Returns:
        QFI score in [0, 1]
    """
    p = to_simplex_prob(basin)
    
    positive_probs = p[p > 1e-10]
    if len(positive_probs) == 0:
        return 0.0
    
    entropy = -np.sum(positive_probs * np.log(positive_probs + 1e-10))
    n_dim = len(basin)
    effective_dim = np.exp(entropy)
    qfi_score = effective_dim / n_dim
    
    return float(np.clip(qfi_score, 0.0, 1.0))


def validate_simplex(basin: np.ndarray, tolerance: float = 1e-6) -> Tuple[bool, str]:
    """
    Validate basin is a proper simplex probability distribution.
    
    Args:
        basin: Basin coordinates
        tolerance: Numerical tolerance for sum check
        
    Returns:
        (is_valid, reason)
    """
    if basin is None:
        return False, "basin_is_none"
    
    basin = np.asarray(basin, dtype=np.float64)
    
    if len(basin) != 64:
        return False, f"wrong_dimension_{len(basin)}"
    
    if np.any(basin < -tolerance):
        return False, "negative_values"
    
    if not np.isfinite(basin).all():
        return False, "contains_nan_or_inf"
    
    prob_sum = basin.sum()
    if abs(prob_sum - 1.0) > tolerance:
        return False, f"sum_not_one_{prob_sum:.6f}"
    
    return True, "valid"


def upsert_token(
    token: str,
    basin: Optional[np.ndarray] = None,
    token_role: str = "encode",
    token_status: str = "active",
    is_real_word: Optional[bool] = None,
    frequency: int = 0,
    source: str = "canonical_upsert",
    phrase_category: Optional[str] = None,
    force_qfi: bool = True
) -> Dict[str, Any]:
    """
    Canonical token upsert - THE ONLY DB WRITE PATH for coordizer_vocabulary.
    
    SLEEP-PACKET Section 4D Implementation:
    1. Basin projected to simplex
    2. QFI computed when basin present
    3. Quarantine if QFI fails
    4. Active tokens require non-null qfi_score
    
    Args:
        token: The token string (unique key)
        basin: 64D basin coordinates (will be projected to simplex)
        token_role: encode|generate|quarantine
        token_status: active|quarantined|deprecated
        is_real_word: Whether token is a real English word
        frequency: Usage frequency
        source: Source of this token (max 32 chars)
        phrase_category: Classification category
        force_qfi: If True and basin exists, QFI must be computed
        
    Returns:
        {
            'success': bool,
            'token': str,
            'qfi_score': float or None,
            'status': str,
            'reason': str
        }
    """
    result = {
        'success': False,
        'token': token,
        'qfi_score': None,
        'status': token_status,
        'reason': ''
    }
    
    if not token or not isinstance(token, str):
        result['reason'] = 'invalid_token'
        return result
    
    if not PSYCOPG2_AVAILABLE:
        result['reason'] = 'psycopg2_not_available'
        return result
    
    connection_string = os.getenv('DATABASE_URL')
    if not connection_string:
        result['reason'] = 'no_database_url'
        return result
    
    simplex_basin = None
    qfi_score = None
    
    if basin is not None:
        basin = np.asarray(basin, dtype=np.float64)
        simplex_basin = to_simplex_prob(basin)
        is_valid, validation_reason = validate_simplex(simplex_basin)
        
        if not is_valid:
            logger.warning(f"Basin validation failed for '{token}': {validation_reason}")
            token_role = 'quarantine'  # Use token_role for quarantine (schema uses token_role, not token_status)
            token_status = 'quarantined'
            result['status'] = 'quarantined'
            result['reason'] = f'basin_invalid_{validation_reason}'
        else:
            qfi_score = compute_qfi_score(simplex_basin)
            result['qfi_score'] = qfi_score
            
            if qfi_score < 0.01:
                logger.warning(f"Low QFI score for '{token}': {qfi_score:.4f}")
                token_role = 'quarantine'  # Use token_role for quarantine
                token_status = 'quarantined'
                result['status'] = 'quarantined'
                result['reason'] = 'qfi_too_low'
    
    if token_status == 'active' and basin is not None and qfi_score is None:
        token_role = 'quarantine'  # Use token_role for quarantine
        token_status = 'quarantined'
        result['status'] = 'quarantined'
        result['reason'] = 'active_requires_qfi'
    
    source = source[:32] if source and len(source) > 32 else (source or 'unknown')
    
    try:
        conn = psycopg2.connect(connection_string)
        try:
            with conn.cursor() as cur:
                basin_list = simplex_basin.tolist() if simplex_basin is not None else None
                
                cur.execute("""
                    INSERT INTO coordizer_vocabulary (
                        token, basin_embedding, qfi_score, token_role, 
                        is_real_word, frequency, phrase_category,
                        created_at, updated_at
                    ) VALUES (
                        %s, %s::vector, %s, %s,
                        %s, %s, %s,
                        NOW(), NOW()
                    )
                    ON CONFLICT (token) DO UPDATE SET
                        basin_embedding = COALESCE(EXCLUDED.basin_embedding, coordizer_vocabulary.basin_embedding),
                        qfi_score = COALESCE(EXCLUDED.qfi_score, coordizer_vocabulary.qfi_score),
                        token_role = EXCLUDED.token_role,
                        is_real_word = COALESCE(EXCLUDED.is_real_word, coordizer_vocabulary.is_real_word),
                        frequency = GREATEST(coordizer_vocabulary.frequency, EXCLUDED.frequency),
                        phrase_category = COALESCE(EXCLUDED.phrase_category, coordizer_vocabulary.phrase_category),
                        updated_at = NOW()
                    RETURNING token, qfi_score, token_role
                """, (
                    token, basin_list, qfi_score, token_role,
                    is_real_word, frequency, phrase_category
                ))
                
                row = cur.fetchone()
                conn.commit()
                
                result['success'] = True
                result['reason'] = 'upserted'
                if row:
                    result['qfi_score'] = row[1]
                    result['status'] = row[2] if row[2] else token_status
                
                logger.debug(f"Canonical upsert: '{token}' qfi={qfi_score:.4f if qfi_score else 'null'}")
                
        finally:
            conn.close()
            
    except Exception as e:
        result['reason'] = f'db_error_{str(e)[:100]}'
        logger.error(f"Canonical upsert failed for '{token}': {e}")
    
    return result


def batch_upsert_tokens(
    tokens: list,
    basins: Optional[list] = None,
    token_role: str = "encode",
    source: str = "batch_canonical"
) -> Dict[str, Any]:
    """
    Batch version of canonical upsert for efficiency.
    
    Args:
        tokens: List of token strings
        basins: Optional list of basin coordinates (same length as tokens)
        token_role: Default role for all tokens
        source: Source identifier
        
    Returns:
        {
            'success_count': int,
            'quarantine_count': int,
            'error_count': int,
            'results': list of individual results
        }
    """
    if basins is None:
        basins = [None] * len(tokens)
    
    if len(tokens) != len(basins):
        return {
            'success_count': 0,
            'quarantine_count': 0,
            'error_count': len(tokens),
            'results': [],
            'reason': 'tokens_basins_length_mismatch'
        }
    
    results = []
    success_count = 0
    quarantine_count = 0
    error_count = 0
    
    for token, basin in zip(tokens, basins):
        result = upsert_token(
            token=token,
            basin=basin,
            token_role=token_role,
            source=source
        )
        results.append(result)
        
        if result['success']:
            if result['status'] == 'quarantined':
                quarantine_count += 1
            else:
                success_count += 1
        else:
            error_count += 1
    
    return {
        'success_count': success_count,
        'quarantine_count': quarantine_count,
        'error_count': error_count,
        'results': results
    }


__all__ = [
    'to_simplex_prob',
    'compute_qfi_score', 
    'validate_simplex',
    'upsert_token',
    'batch_upsert_tokens',
]
