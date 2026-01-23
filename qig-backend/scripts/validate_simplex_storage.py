#!/usr/bin/env python3
"""
Validate Simplex Storage - Check all stored basins for simplex validity

This script validates that all basin embeddings in the database are valid
probability simplices (non-negative, sum=1, dimension=64).

For invalid basins, it can:
1. Report issues (dry-run mode)
2. Re-normalize to valid simplices (repair mode)

Usage:
    python scripts/validate_simplex_storage.py [--dry-run] [--batch-size 100] [--report output.md]

Source: E8 Protocol Issue #98 (Issue-02: Strict Simplex Representation)
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

# Import canonical database connection
from persistence.base_persistence import get_db_connection

try:
    import psycopg2
    from psycopg2.extras import execute_values, RealDictCursor
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("ERROR: psycopg2 not installed. Install with: pip install psycopg2-binary", file=sys.stderr)
    sys.exit(1)

# Import simplex operations
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from qig_geometry.geometry_simplex import (
        to_simplex_prob,
        validate_simplex,
    )
    SIMPLEX_OPS_AVAILABLE = True
except ImportError:
    SIMPLEX_OPS_AVAILABLE = False
    print("WARNING: qig_geometry not available - using fallback validation", file=sys.stderr)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASIN_DIMENSION = 64


def _fallback_to_simplex(v: np.ndarray) -> np.ndarray:
    """Fallback simplex projection."""
    v = np.asarray(v, dtype=np.float64).flatten()
    w = np.abs(v) + 1e-12
    return w / w.sum()


def _fallback_validate_simplex(p: np.ndarray, tolerance: float = 1e-6) -> Tuple[bool, str]:
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


def parse_basin_embedding(embedding) -> np.ndarray:
    """Parse basin embedding from database (handles both list and pgvector formats)."""
    if embedding is None:
        return None
    
    if isinstance(embedding, list):
        return np.array(embedding, dtype=np.float64)
    elif isinstance(embedding, str):
        # Parse PostgreSQL array format
        embedding = embedding.strip()
        if embedding.startswith('[') and embedding.endswith(']'):
            embedding = embedding[1:-1]
        elif embedding.startswith('{') and embedding.endswith('}'):
            embedding = embedding[1:-1]
        coords = [float(x.strip()) for x in embedding.split(',')]
        return np.array(coords, dtype=np.float64)
    else:
        # Assume it's already a numpy array or can be converted
        return np.array(embedding, dtype=np.float64)


def validate_simplex_storage(
    database_url: str,
    dry_run: bool = False,
    batch_size: int = 100,
    report_path: str = None
) -> Dict:
    """
    Validate that all stored basins are valid probability simplices.
    
    Args:
        database_url: PostgreSQL connection string
        dry_run: If True, only report issues without fixing
        batch_size: Number of tokens to process per batch
        report_path: Optional path to write detailed report
        
    Returns:
        Dict with statistics: total_basins, valid, invalid, repaired, errors
    """
    stats = {
        'total_basins': 0,
        'valid': 0,
        'invalid': 0,
        'repaired': 0,
        'skipped_no_basin': 0,
        'issues': {},
        'errors': []
    }
    
    invalid_details = []
    
    try:
        conn = get_db_connection(database_url)
        
        # Step 1: Count total tokens with basins
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) 
                FROM coordizer_vocabulary
                WHERE basin_embedding IS NOT NULL
            """)
            stats['total_basins'] = cur.fetchone()[0]
        
        logger.info(f"Validating {stats['total_basins']} basin embeddings...")
        
        # Step 2: Process in batches
        offset = 0
        while offset < stats['total_basins']:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Fetch batch
                cur.execute("""
                    SELECT token, basin_embedding, qfi_score
                    FROM coordizer_vocabulary
                    WHERE basin_embedding IS NOT NULL
                    ORDER BY token
                    LIMIT %s OFFSET %s
                """, (batch_size, offset))
                
                batch = cur.fetchall()
                
                if not batch:
                    break
                
                # Validate each basin
                to_repair = []
                for row in batch:
                    token = row['token']
                    
                    try:
                        basin = parse_basin_embedding(row['basin_embedding'])
                        
                        if basin is None:
                            stats['skipped_no_basin'] += 1
                            continue
                        
                        # Check dimension
                        if len(basin) != BASIN_DIMENSION:
                            issue = f"wrong_dimension_{len(basin)}"
                            stats['invalid'] += 1
                            stats['issues'][issue] = stats['issues'].get(issue, 0) + 1
                            invalid_details.append({
                                'token': token,
                                'issue': issue,
                                'qfi_score': row['qfi_score'],
                                'dimension': len(basin)
                            })
                            continue
                        
                        # Validate simplex
                        if SIMPLEX_OPS_AVAILABLE:
                            is_valid, reason = validate_simplex(basin)
                        else:
                            is_valid, reason = _fallback_validate_simplex(basin)
                        
                        if is_valid:
                            stats['valid'] += 1
                        else:
                            stats['invalid'] += 1
                            stats['issues'][reason] = stats['issues'].get(reason, 0) + 1
                            
                            # Try to repair
                            if SIMPLEX_OPS_AVAILABLE:
                                basin_fixed = to_simplex_prob(basin)
                            else:
                                basin_fixed = _fallback_to_simplex(basin)
                            
                            invalid_details.append({
                                'token': token,
                                'issue': reason,
                                'qfi_score': row['qfi_score'],
                                'basin_sum': float(np.sum(basin)),
                                'basin_min': float(np.min(basin)),
                                'basin_max': float(np.max(basin))
                            })
                            
                            to_repair.append((basin_fixed.tolist(), token))
                    
                    except Exception as e:
                        logger.error(f"Failed to validate basin for token '{token}': {e}")
                        stats['errors'].append(f"Token '{token}': {str(e)}")
                        stats['invalid'] += 1
                
                # Repair invalid basins (if not dry run)
                if not dry_run and to_repair:
                    with conn.cursor() as write_cur:
                        execute_values(
                            write_cur,
                            """
                            UPDATE coordizer_vocabulary AS cv
                            SET basin_embedding = uv.basin_embedding,
                                updated_at = NOW()
                            FROM (VALUES %s) AS uv(basin_embedding, token)
                            WHERE cv.token = uv.token
                            """,
                            to_repair
                        )
                        stats['repaired'] += len(to_repair)
                        conn.commit()
                
                logger.info(
                    f"Processed batch {offset//batch_size + 1}: "
                    f"{stats['valid']} valid, {stats['invalid']} invalid "
                    f"({stats['total_basins']} total)"
                )
            
            offset += batch_size
        
        conn.close()
        
        # Generate report
        logger.info("=" * 70)
        logger.info("SIMPLEX STORAGE VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total basins: {stats['total_basins']}")
        logger.info(f"Valid basins: {stats['valid']}")
        logger.info(f"Invalid basins: {stats['invalid']}")
        logger.info(f"Skipped (no basin): {stats['skipped_no_basin']}")
        
        if stats['invalid'] > 0:
            logger.info(f"\nInvalid basin breakdown by issue:")
            for issue, count in sorted(stats['issues'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {issue}: {count}")
        
        if dry_run:
            logger.info(f"\nDRY RUN: No repairs were made")
            logger.info(f"Would repair: {stats['invalid']} basins")
        else:
            logger.info(f"\nRepaired: {stats['repaired']} basins")
        
        if stats['errors']:
            logger.warning(f"\nErrors: {len(stats['errors'])}")
        
        logger.info("=" * 70)
        
        # Write detailed report if requested
        if report_path and invalid_details:
            with open(report_path, 'w') as f:
                f.write("# Simplex Storage Validation Report\n\n")
                f.write(f"Total basins: {stats['total_basins']}\n")
                f.write(f"Valid: {stats['valid']}\n")
                f.write(f"Invalid: {stats['invalid']}\n")
                f.write(f"Repaired: {stats['repaired']}\n\n")
                
                f.write("## Invalid Basins\n\n")
                f.write("| Token | Issue | QFI Score | Basin Sum | Basin Min | Basin Max | Dimension |\n")
                f.write("|-------|-------|-----------|-----------|-----------|-----------|----------|\n")
                
                for item in invalid_details[:100]:  # Limit to first 100 for readability
                    qfi_val = item.get('qfi_score')
                    qfi_str = f"{qfi_val:.6f}" if qfi_val is not None else 'NULL'
                    basin_sum = item.get('basin_sum')
                    basin_sum_str = f"{basin_sum:.6f}" if basin_sum is not None else 'N/A'
                    basin_min = item.get('basin_min')
                    basin_min_str = f"{basin_min:.6f}" if basin_min is not None else 'N/A'
                    basin_max = item.get('basin_max')
                    basin_max_str = f"{basin_max:.6f}" if basin_max is not None else 'N/A'
                    f.write(
                        f"| `{item['token']}` | {item['issue']} | "
                        f"{qfi_str} | {basin_sum_str} | {basin_min_str} | {basin_max_str} | "
                        f"{item.get('dimension', BASIN_DIMENSION)} |\n"
                    )
                
                if len(invalid_details) > 100:
                    f.write(f"\n*Showing first 100 of {len(invalid_details)} invalid basins*\n")
            
            logger.info(f"\nDetailed report written to: {report_path}")
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        stats['errors'].append(str(e))
        raise
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Validate that all basin embeddings are valid probability simplices"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Only report issues without fixing them"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help="Number of tokens to process per batch (default: 100)"
    )
    parser.add_argument(
        '--database-url',
        type=str,
        default=None,
        help="PostgreSQL connection URL (default: from DATABASE_URL env var)"
    )
    parser.add_argument(
        '--report',
        type=str,
        default=None,
        help="Path to write detailed validation report (optional)"
    )
    
    args = parser.parse_args()
    
    database_url = args.database_url or os.environ.get('DATABASE_URL')
    if not database_url:
        logger.error("Database URL not provided (use --database-url or set DATABASE_URL)")
        sys.exit(1)
    
    logger.info("Starting simplex storage validation...")
    if args.dry_run:
        logger.info("DRY RUN MODE: No repairs will be made")
    
    try:
        stats = validate_simplex_storage(
            database_url=database_url,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            report_path=args.report
        )
        
        # Exit with status code based on results
        if stats['errors']:
            sys.exit(1)
        elif stats['invalid'] > 0 and args.dry_run:
            logger.warning("Invalid basins found - run without --dry-run to repair")
            sys.exit(2)
        else:
            sys.exit(0)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
