#!/usr/bin/env python3
"""
Backfill QFI Scores for Existing Tokens

P1 FIX: Compute and populate qfi_score for all tokens in coordizer_vocabulary
that have basin_embedding but NULL qfi_score.

QFI (Quantum Fisher Information) computation:
- Fisher metric: outer product + regularization
- Determinant as QFI score
- Measures geometric distinguishability of basin coordinates

Usage:
    python3 scripts/backfill_qfi_scores.py [--dry-run] [--batch-size 100]
"""

import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def compute_qfi(basin: np.ndarray) -> float:
    """
    Compute Quantum Fisher Information score for a basin.

    Uses participation ratio (effective dimension) which is geometrically proper:
    QFI = exp(H(p)) / n where H(p) is Shannon entropy.

    This is the CANONICAL QFI formula - produces values in [0, 1].

    Args:
        basin: 64D basin coordinates

    Returns:
        QFI score in [0, 1]
    """
    # Project to simplex probabilities
    v = np.abs(basin) + 1e-10
    p = v / v.sum()

    # Compute Shannon entropy
    positive_probs = p[p > 1e-10]
    if len(positive_probs) == 0:
        return 0.0

    entropy = -np.sum(positive_probs * np.log(positive_probs + 1e-10))

    # Participation ratio = exp(entropy) / dimension
    n_dim = len(basin)
    effective_dim = np.exp(entropy)
    qfi_score = effective_dim / n_dim

    return float(np.clip(qfi_score, 0.0, 1.0))


def parse_embedding(embedding_str: str) -> Optional[np.ndarray]:
    """Parse basin embedding from PostgreSQL vector format."""
    try:
        # Handle both formats: '[1,2,3]' and '{1,2,3}'
        embedding_str = embedding_str.strip()
        if embedding_str.startswith('[') and embedding_str.endswith(']'):
            embedding_str = embedding_str[1:-1]
        elif embedding_str.startswith('{') and embedding_str.endswith('}'):
            embedding_str = embedding_str[1:-1]
        
        coords = [float(x.strip()) for x in embedding_str.split(',')]
        
        if len(coords) != 64:
            logger.warning(f"Invalid basin dimension: {len(coords)} (expected 64)")
            return None
        
        return np.array(coords, dtype=np.float64)
    except Exception as e:
        logger.warning(f"Failed to parse embedding: {e}")
        return None


def backfill_qfi_scores(
    database_url: str,
    dry_run: bool = False,
    batch_size: int = 100
) -> dict:
    """
    Backfill qfi_score for all tokens with basin_embedding but NULL qfi_score.
    
    Args:
        database_url: PostgreSQL connection string
        dry_run: If True, only report what would be done (no database writes)
        batch_size: Number of tokens to process per batch
    
    Returns:
        Dict with statistics: total_tokens, updated_tokens, skipped_tokens, errors
    """
    stats = {
        'total_tokens': 0,
        'updated_tokens': 0,
        'skipped_tokens': 0,
        'errors': []
    }
    
    try:
        conn = psycopg2.connect(database_url)
        
        # Step 1: Count tokens needing backfill
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) 
                FROM coordizer_vocabulary
                WHERE basin_embedding IS NOT NULL
                  AND qfi_score IS NULL
            """)
            stats['total_tokens'] = cur.fetchone()[0]
        
        logger.info(f"Found {stats['total_tokens']} tokens with basin but NULL qfi_score")
        
        if stats['total_tokens'] == 0:
            logger.info("No tokens need backfilling. All done!")
            return stats
        
        if dry_run:
            logger.info("DRY RUN: Would process {stats['total_tokens']} tokens")
            return stats
        
        # Step 2: Process in batches
        offset = 0
        while offset < stats['total_tokens']:
            with conn.cursor() as cur:
                # Fetch batch
                cur.execute("""
                    SELECT token, basin_embedding
                    FROM coordizer_vocabulary
                    WHERE basin_embedding IS NOT NULL
                      AND qfi_score IS NULL
                    ORDER BY token
                    LIMIT %s OFFSET %s
                """, (batch_size, offset))
                
                batch = cur.fetchall()
                
                if not batch:
                    break
                
                # Compute QFI for each token in batch
                updates = []
                for token, basin_embedding in batch:
                    try:
                        basin = parse_embedding(basin_embedding)
                        if basin is None:
                            stats['skipped_tokens'] += 1
                            stats['errors'].append(f"Invalid basin for token '{token}'")
                            continue
                        
                        qfi = compute_qfi(basin)
                        updates.append((qfi, token))
                    except Exception as e:
                        stats['skipped_tokens'] += 1
                        stats['errors'].append(f"Failed to compute QFI for '{token}': {e}")
                        logger.warning(f"Failed to compute QFI for '{token}': {e}")
                
                # Batch update
                if updates:
                    execute_values(
                        cur,
                        """
                        UPDATE coordizer_vocabulary AS cv
                        SET qfi_score = uv.qfi_score, updated_at = NOW()
                        FROM (VALUES %s) AS uv(qfi_score, token)
                        WHERE cv.token = uv.token
                        """,
                        updates
                    )
                    stats['updated_tokens'] += len(updates)
                    conn.commit()
                    
                    logger.info(
                        f"Processed batch {offset//batch_size + 1}: "
                        f"Updated {len(updates)} tokens "
                        f"({stats['updated_tokens']}/{stats['total_tokens']})"
                    )
            
            offset += batch_size
        
        logger.info(
            f"Backfill complete: "
            f"{stats['updated_tokens']} updated, "
            f"{stats['skipped_tokens']} skipped"
        )
        
        if stats['errors']:
            logger.warning(f"Encountered {len(stats['errors'])} errors (see log for details)")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        stats['errors'].append(str(e))
        raise
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Backfill qfi_score for existing tokens with basin_embedding"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be done without making changes"
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
    
    args = parser.parse_args()
    
    database_url = args.database_url or os.environ.get('DATABASE_URL')
    if not database_url:
        logger.error("Database URL not provided (use --database-url or set DATABASE_URL)")
        sys.exit(1)
    
    logger.info("Starting QFI score backfill...")
    if args.dry_run:
        logger.info("DRY RUN MODE: No changes will be made")
    
    try:
        stats = backfill_qfi_scores(
            database_url=database_url,
            dry_run=args.dry_run,
            batch_size=args.batch_size
        )
        
        logger.info("=" * 60)
        logger.info("BACKFILL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total tokens found:    {stats['total_tokens']}")
        logger.info(f"Tokens updated:        {stats['updated_tokens']}")
        logger.info(f"Tokens skipped:        {stats['skipped_tokens']}")
        logger.info(f"Errors:                {len(stats['errors'])}")
        logger.info("=" * 60)
        
        if stats['errors']:
            logger.warning("First 10 errors:")
            for error in stats['errors'][:10]:
                logger.warning(f"  - {error}")
        
        sys.exit(0 if not stats['errors'] else 1)
        
    except Exception as e:
        logger.error(f"Backfill failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
