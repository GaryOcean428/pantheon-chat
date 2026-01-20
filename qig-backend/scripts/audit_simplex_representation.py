#!/usr/bin/env python3
"""
Simplex Representation Audit Script
====================================

Audits all basin coordinates in the database to ensure they are valid simplices.

Checks:
1. All values non-negative
2. Sum to 1 (within tolerance)
3. Correct dimension (64D)
4. No inf or nan values

Reports violations and can optionally fix them by renormalizing.

Usage:
    python audit_simplex_representation.py [--fix] [--report output.txt]

Args:
    --fix: Automatically fix violations by renormalizing to simplex
    --report: Output file for audit report
    --strict: Use stricter tolerance for sum check

Author: Copilot AI Agent
Date: 2026-01-20
Issue: GaryOcean428/pantheon-chat#98 (E8 Protocol Issue-02)
Reference: docs/10-e8-protocol/issues/20260119-issue-98-strict-simplex-representation-remediation-1.00W.md
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import psycopg2
from psycopg2.extensions import connection as PgConnection

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Validation parameters
BASIN_DIM = 64
SIMPLEX_SUM_TOLERANCE = 1e-6
STRICT_TOLERANCE = 1e-9


def parse_basin(basin_str: str) -> np.ndarray:
    """
    Parse basin from PostgreSQL vector format.
    
    Args:
        basin_str: Basin string from database
        
    Returns:
        Basin as numpy array
    """
    basin_str = basin_str.strip()
    if basin_str.startswith('['):
        basin_str = basin_str[1:-1]
    elif basin_str.startswith('{'):
        basin_str = basin_str[1:-1]
    
    coords = [float(x.strip()) for x in basin_str.split(',')]
    return np.array(coords, dtype=np.float64)


def validate_simplex(basin: np.ndarray, strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate that basin is a valid simplex.
    
    Args:
        basin: Basin coordinates
        strict: Use stricter tolerance
        
    Returns:
        (is_valid, error_messages) tuple
    """
    errors = []
    
    # Check dimension
    if len(basin) != BASIN_DIM:
        errors.append(f"Invalid dimension: {len(basin)} (expected {BASIN_DIM})")
    
    # Check non-negative
    if np.any(basin < 0):
        min_val = basin.min()
        neg_count = np.sum(basin < 0)
        errors.append(f"Negative values: {neg_count} values, min={min_val:.6e}")
    
    # Check sum to 1
    total = basin.sum()
    tolerance = STRICT_TOLERANCE if strict else SIMPLEX_SUM_TOLERANCE
    if not np.isclose(total, 1.0, atol=tolerance):
        errors.append(f"Sum not 1: sum={total:.9f} (tolerance={tolerance})")
    
    # Check finite
    if not np.all(np.isfinite(basin)):
        inf_count = np.sum(np.isinf(basin))
        nan_count = np.sum(np.isnan(basin))
        errors.append(f"Non-finite values: {inf_count} inf, {nan_count} nan")
    
    is_valid = len(errors) == 0
    return (is_valid, errors)


def renormalize_to_simplex(basin: np.ndarray) -> np.ndarray:
    """
    Fix basin coordinates by renormalizing to simplex.
    
    Args:
        basin: Basin coordinates (possibly invalid)
        
    Returns:
        Valid simplex coordinates
    """
    # Clip negative values to zero
    basin = np.maximum(basin, 0)
    
    # Add small epsilon for numerical stability
    basin = basin + 1e-12
    
    # Renormalize
    basin = basin / basin.sum()
    
    return basin


def audit_vocabulary_basins(
    db_conn: PgConnection,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Audit all basins in coordizer_vocabulary.
    
    Args:
        db_conn: PostgreSQL connection
        strict: Use stricter validation
        
    Returns:
        Dict with audit results
    """
    total_tokens = 0
    valid_tokens = 0
    invalid_tokens = 0
    violations = []
    
    with db_conn.cursor() as cursor:
        cursor.execute("""
            SELECT token_id, token, basin_embedding
            FROM coordizer_vocabulary
            WHERE basin_embedding IS NOT NULL
        """)
        
        for row in cursor.fetchall():
            token_id, token, basin_str = row
            total_tokens += 1
            
            try:
                basin = parse_basin(basin_str)
                is_valid, errors = validate_simplex(basin, strict=strict)
                
                if is_valid:
                    valid_tokens += 1
                else:
                    invalid_tokens += 1
                    violations.append({
                        'token_id': token_id,
                        'token': token,
                        'errors': errors,
                        'basin': basin
                    })
                    
                    if invalid_tokens <= 10:  # Log first 10 violations
                        logger.warning(f"Invalid basin for token_id={token_id} '{token}': {'; '.join(errors)}")
                
            except Exception as e:
                invalid_tokens += 1
                violations.append({
                    'token_id': token_id,
                    'token': token,
                    'errors': [f'Parse error: {str(e)}'],
                    'basin': None
                })
                logger.error(f"Error parsing basin for token_id={token_id}: {e}")
            
            if total_tokens % 1000 == 0:
                logger.info(f"Audited {total_tokens} tokens...")
    
    return {
        'total': total_tokens,
        'valid': valid_tokens,
        'invalid': invalid_tokens,
        'violations': violations
    }


def fix_violations(
    db_conn: PgConnection,
    violations: List[Dict]
) -> Dict[str, int]:
    """
    Fix simplex violations by renormalizing.
    
    Args:
        db_conn: PostgreSQL connection
        violations: List of violation records
        
    Returns:
        Dict with counts: fixed, failed
    """
    fixed = 0
    failed = 0
    
    with db_conn.cursor() as cursor:
        for violation in violations:
            token_id = violation['token_id']
            basin = violation['basin']
            
            if basin is None:
                logger.error(f"Cannot fix token_id={token_id}: basin is None")
                failed += 1
                continue
            
            try:
                # Renormalize to simplex
                fixed_basin = renormalize_to_simplex(basin)
                
                # Validate fix
                is_valid, errors = validate_simplex(fixed_basin)
                if not is_valid:
                    logger.error(f"Fix failed for token_id={token_id}: {'; '.join(errors)}")
                    failed += 1
                    continue
                
                # Update database
                basin_str = '[' + ','.join(str(x) for x in fixed_basin) + ']'
                cursor.execute("""
                    UPDATE coordizer_vocabulary
                    SET basin_embedding = %s::vector,
                        updated_at = NOW()
                    WHERE token_id = %s
                """, (basin_str, token_id))
                
                fixed += 1
                logger.debug(f"Fixed basin for token_id={token_id}")
                
            except Exception as e:
                logger.error(f"Error fixing token_id={token_id}: {e}")
                failed += 1
    
    db_conn.commit()
    
    return {'fixed': fixed, 'failed': failed}


def generate_report(audit_results: Dict, output_file: str = None):
    """
    Generate human-readable audit report.
    
    Args:
        audit_results: Results from audit_vocabulary_basins
        output_file: Optional output file path
    """
    lines = []
    lines.append("=" * 80)
    lines.append("SIMPLEX REPRESENTATION AUDIT REPORT")
    lines.append("=" * 80)
    lines.append(f"Total tokens audited: {audit_results['total']}")
    lines.append(f"Valid simplices: {audit_results['valid']} ({audit_results['valid']/audit_results['total']*100:.1f}%)")
    lines.append(f"Invalid simplices: {audit_results['invalid']} ({audit_results['invalid']/audit_results['total']*100:.1f}%)")
    lines.append("")
    
    if audit_results['invalid'] > 0:
        lines.append("\nVIOLATIONS BY TYPE:")
        lines.append("-" * 80)
        
        # Group violations by error type
        error_types = {}
        for v in audit_results['violations']:
            for error in v['errors']:
                # Extract error type (first part before colon)
                error_type = error.split(':')[0]
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append(v)
        
        for error_type, violations in sorted(error_types.items()):
            lines.append(f"\n{error_type} ({len(violations)} tokens)")
            for v in violations[:10]:  # Show first 10
                lines.append(f"  - token_id={v['token_id']} '{v['token']}': {'; '.join(v['errors'])}")
            if len(violations) > 10:
                lines.append(f"  ... and {len(violations) - 10} more")
    
    lines.append("\n" + "=" * 80)
    
    report = "\n".join(lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        logger.info(f"Report written to {output_file}")
    else:
        print(report)


def main():
    parser = argparse.ArgumentParser(description='Audit simplex representation in coordizer_vocabulary')
    parser.add_argument('--fix', action='store_true', help='Fix violations by renormalizing')
    parser.add_argument('--report', type=str, help='Output file for audit report')
    parser.add_argument('--strict', action='store_true', help='Use stricter tolerance')
    parser.add_argument('--db-url', type=str, help='Database URL')
    
    args = parser.parse_args()
    
    # Get database URL
    db_url = args.db_url or os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("ERROR: Database URL not provided. Set DATABASE_URL or use --db-url")
        sys.exit(1)
    
    # Connect to database
    try:
        conn = psycopg2.connect(db_url)
        logger.info("Connected to database")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)
    
    # Run audit
    logger.info("Starting simplex representation audit...")
    audit_results = audit_vocabulary_basins(conn, strict=args.strict)
    
    logger.info(f"Audit complete: {audit_results['valid']} valid, {audit_results['invalid']} invalid")
    
    # Generate report
    generate_report(audit_results, args.report)
    
    # Fix violations if requested
    if args.fix and audit_results['invalid'] > 0:
        logger.info("Fixing violations...")
        fix_results = fix_violations(conn, audit_results['violations'])
        logger.info(f"Fixed: {fix_results['fixed']}, Failed: {fix_results['failed']}")
    
    conn.close()


if __name__ == '__main__':
    main()
