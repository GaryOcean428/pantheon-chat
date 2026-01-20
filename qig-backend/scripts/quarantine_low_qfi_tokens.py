#!/usr/bin/env python3
"""
Quarantine Low QFI Tokens

P0 CRITICAL: Identify and quarantine tokens with QFI scores below threshold.
This script moves low-quality tokens out of the active generation vocabulary.

QFI (Quantum Fisher Information) threshold: 0.01
- Tokens with qfi_score < 0.01 are quarantined
- Sets token_role = 'quarantine' and token_status = 'quarantined'
- Excludes special symbols (PAD, UNK, BOS, EOS) from quarantine

Usage:
    python3 scripts/quarantine_low_qfi_tokens.py [--dry-run] [--threshold 0.01] [--batch-size 100]
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Special symbols that should never be quarantined (needed for coordizer)
# These are the canonical special symbols defined in coordizers/base.py
SPECIAL_SYMBOLS = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']


def identify_low_qfi_tokens(
    database_url: str,
    threshold: float = 0.01,
    batch_size: int = 100
) -> Dict[str, int]:
    """
    Identify tokens with QFI scores below threshold.
    
    Args:
        database_url: PostgreSQL connection string
        threshold: QFI score threshold (default: 0.01)
        batch_size: Number of tokens to process per batch
    
    Returns:
        Dict with statistics: total_low_qfi, quarantined, skipped_special, errors
    """
    stats = {
        'total_low_qfi': 0,
        'quarantined': 0,
        'skipped_special': 0,
        'already_quarantined': 0,
        'errors': []
    }
    
    try:
        conn = psycopg2.connect(database_url)
        
        # Step 1: Count tokens with low QFI
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) 
                FROM coordizer_vocabulary
                WHERE qfi_score IS NOT NULL
                  AND qfi_score < %s
                  AND token_status != 'quarantined'
            """, (threshold,))
            stats['total_low_qfi'] = cur.fetchone()[0]
        
        logger.info(f"Found {stats['total_low_qfi']} tokens with qfi_score < {threshold}")
        
        if stats['total_low_qfi'] == 0:
            logger.info("No tokens need quarantining. All done!")
            return stats
        
        # Step 2: Fetch and process in batches
        offset = 0
        while offset < stats['total_low_qfi']:
            with conn.cursor() as cur:
                # Fetch batch
                cur.execute("""
                    SELECT token, qfi_score, token_status, token_role
                    FROM coordizer_vocabulary
                    WHERE qfi_score IS NOT NULL
                      AND qfi_score < %s
                      AND token_status != 'quarantined'
                    ORDER BY token
                    LIMIT %s OFFSET %s
                """, (threshold, batch_size, offset))
                
                batch = cur.fetchall()
                
                if not batch:
                    break
                
                # Process each token
                quarantine_list = []
                for token, qfi_score, token_status, token_role in batch:
                    # Skip special symbols
                    if token in SPECIAL_SYMBOLS:
                        stats['skipped_special'] += 1
                        logger.info(f"Skipping special symbol: {token}")
                        continue
                    
                    # Skip already quarantined
                    if token_status == 'quarantined' or token_role == 'quarantine':
                        stats['already_quarantined'] += 1
                        continue
                    
                    quarantine_list.append((token, qfi_score))
                
                # Batch quarantine
                if quarantine_list:
                    execute_values(
                        cur,
                        """
                        UPDATE coordizer_vocabulary AS cv
                        SET token_role = 'quarantine',
                            token_status = 'quarantined',
                            updated_at = NOW()
                        FROM (VALUES %s) AS qv(token, qfi_score)
                        WHERE cv.token = qv.token
                        """,
                        quarantine_list
                    )
                    stats['quarantined'] += len(quarantine_list)
                    conn.commit()
                    
                    logger.info(
                        f"Quarantined batch {offset//batch_size + 1}: "
                        f"{len(quarantine_list)} tokens "
                        f"({stats['quarantined']}/{stats['total_low_qfi']})"
                    )
                    
                    # Log first few tokens in batch for visibility
                    if len(quarantine_list) > 0:
                        sample = quarantine_list[:3]
                        for token, qfi in sample:
                            logger.debug(f"  - '{token}' (qfi={qfi:.4f})")
            
            offset += batch_size
        
        # Step 3: Log impact on generation vocabulary
        with conn.cursor() as cur:
            # Count active tokens remaining
            cur.execute("""
                SELECT COUNT(*) 
                FROM coordizer_vocabulary
                WHERE token_status = 'active'
                  AND basin_embedding IS NOT NULL
                  AND qfi_score IS NOT NULL
                  AND qfi_score >= %s
            """, (threshold,))
            active_count = cur.fetchone()[0]
            
            # Count quarantined tokens
            cur.execute("""
                SELECT COUNT(*) 
                FROM coordizer_vocabulary
                WHERE token_status = 'quarantined'
            """)
            quarantined_total = cur.fetchone()[0]
        
        logger.info(
            f"\n{'='*60}\n"
            f"QUARANTINE SUMMARY\n"
            f"{'='*60}\n"
            f"Tokens quarantined:        {stats['quarantined']}\n"
            f"Special symbols skipped:   {stats['skipped_special']}\n"
            f"Already quarantined:       {stats['already_quarantined']}\n"
            f"Active generation tokens:  {active_count}\n"
            f"Total quarantined tokens:  {quarantined_total}\n"
            f"{'='*60}"
        )
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Quarantine failed: {e}")
        stats['errors'].append(str(e))
        raise
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Quarantine tokens with low QFI scores"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help="QFI score threshold (default: 0.01)"
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
    
    logger.info("=" * 60)
    logger.info("QFI QUARANTINE SCRIPT")
    logger.info("=" * 60)
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Batch size: {args.batch_size}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE: No changes will be made")
        logger.info("=" * 60)
        
        # In dry-run, just count what would be quarantined
        try:
            conn = psycopg2.connect(database_url)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM coordizer_vocabulary
                    WHERE qfi_score IS NOT NULL
                      AND qfi_score < %s
                      AND token_status != 'quarantined'
                """, (args.threshold,))
                count = cur.fetchone()[0]
                
                # Sample tokens
                cur.execute("""
                    SELECT token, qfi_score
                    FROM coordizer_vocabulary
                    WHERE qfi_score IS NOT NULL
                      AND qfi_score < %s
                      AND token_status != 'quarantined'
                    ORDER BY qfi_score
                    LIMIT 10
                """, (args.threshold,))
                samples = cur.fetchall()
            conn.close()
            
            logger.info(f"\nWould quarantine {count} tokens")
            if samples:
                logger.info("\nSample tokens (lowest QFI):")
                for token, qfi in samples:
                    logger.info(f"  - '{token}' (qfi={qfi:.6f})")
            
            sys.exit(0)
            
        except Exception as e:
            logger.error(f"Dry run failed: {e}")
            sys.exit(1)
    
    # Actual quarantine
    logger.info("=" * 60)
    
    try:
        stats = identify_low_qfi_tokens(
            database_url=database_url,
            threshold=args.threshold,
            batch_size=args.batch_size
        )
        
        if stats['errors']:
            logger.warning(f"Encountered {len(stats['errors'])} errors")
            for error in stats['errors'][:10]:
                logger.warning(f"  - {error}")
            sys.exit(1)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Quarantine failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
