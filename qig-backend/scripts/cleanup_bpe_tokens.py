#!/usr/bin/env python3
"""Cleanup BPE Fragments from Tokenizer Vocabulary.

This script removes BPE subword fragments from the coordizer_vocabulary table,
keeping only real English words. BPE fragments cause garbled text output
when used with the PostgresCoordizer.

Usage:
    python cleanup_bpe_tokens.py [--dry-run]

BPE fragments are identified by:
- Containing '+' (merged BPE pieces)
- Containing '<' or '>' (byte tokens)
- Starting with '##' (BERT-style subwords)
- Containing spaces, newlines, or tabs
- Being single characters
- Not being purely alphabetic
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

# Load environment
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_bpe_fragment(token: str) -> bool:
    """Check if a token is a BPE fragment."""
    if len(token) < 2:
        return True
    if '+' in token:
        return True
    if '<' in token or '>' in token:
        return True
    if token.startswith('##'):
        return True
    if ' ' in token or '\n' in token or '\t' in token:
        return True
    if not token.isalpha():
        return True
    return False


def get_db_connection():
    """Get PostgreSQL connection."""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL not set")
    return psycopg2.connect(db_url)


def analyze_vocabulary(conn):
    """Analyze the current vocabulary composition."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM coordizer_vocabulary")
        total = cur.fetchone()[0]
        
        # Count real words (alphabetic, 2+ chars)
        cur.execute("""
            SELECT COUNT(*) FROM coordizer_vocabulary 
            WHERE LENGTH(token) >= 2 
              AND token ~ '^[a-zA-Z]+$'
              AND token NOT LIKE '%%+%%'
              AND token NOT LIKE '<%%>'
        """)
        real_words = cur.fetchone()[0]
        
        # Count BPE fragments
        bpe_count = total - real_words
        
        logger.info(f"Current vocabulary composition:")
        logger.info(f"  Total tokens: {total}")
        logger.info(f"  Real words: {real_words}")
        logger.info(f"  BPE fragments: {bpe_count}")
        
        return total, real_words, bpe_count


def get_bpe_fragments(conn, limit: int = None):
    """Get all BPE fragment tokens from database."""
    with conn.cursor() as cur:
        # Get tokens that don't match the real word pattern
        query = """
            SELECT id, token FROM coordizer_vocabulary 
            WHERE LENGTH(token) < 2 
               OR token !~ '^[a-zA-Z]+$'
               OR token LIKE '%%+%%'
               OR token LIKE '<%%>'
               OR token LIKE '##%%'
        """
        if limit:
            query += f" LIMIT {limit}"
        cur.execute(query)
        return cur.fetchall()


def delete_bpe_fragments(conn, fragment_ids: list, dry_run: bool = True):
    """Delete BPE fragments from database."""
    if not fragment_ids:
        logger.info("No BPE fragments to delete")
        return 0
    
    if dry_run:
        logger.info(f"[DRY RUN] Would delete {len(fragment_ids)} BPE fragments")
        return 0
    
    deleted = 0
    batch_size = 500
    
    with conn.cursor() as cur:
        for i in range(0, len(fragment_ids), batch_size):
            batch = fragment_ids[i:i + batch_size]
            placeholders = ','.join(['%s'] * len(batch))
            cur.execute(f"DELETE FROM coordizer_vocabulary WHERE id IN ({placeholders})", batch)
            deleted += cur.rowcount
            logger.info(f"Deleted batch {i // batch_size + 1}: {cur.rowcount} tokens")
        
        conn.commit()
    
    logger.info(f"Total deleted: {deleted} BPE fragments")
    return deleted


def main():
    parser = argparse.ArgumentParser(description='Cleanup BPE fragments from coordizer_vocabulary')
    parser.add_argument('--dry-run', action='store_true', help='Preview without making changes')
    parser.add_argument('--show-samples', action='store_true', help='Show sample BPE fragments')
    args = parser.parse_args()
    
    try:
        conn = get_db_connection()
        
        # Analyze current state
        total, real_words, bpe_count = analyze_vocabulary(conn)
        
        if bpe_count == 0:
            logger.info("No BPE fragments found - vocabulary is clean!")
            conn.close()
            return
        
        # Get BPE fragments
        fragments = get_bpe_fragments(conn)
        fragment_ids = [f[0] for f in fragments]
        
        # Show samples if requested
        if args.show_samples:
            logger.info(f"\nSample BPE fragments to delete:")
            for fid, token in fragments[:500]:
                logger.info(f"  {token!r}")
        
        # Delete fragments
        deleted = delete_bpe_fragments(conn, fragment_ids, dry_run=args.dry_run)
        
        # Show final state
        if not args.dry_run:
            logger.info("\nFinal vocabulary composition:")
            analyze_vocabulary(conn)
        
        conn.close()
        
        if args.dry_run:
            logger.info(f"\n[DRY RUN] Run without --dry-run to delete {bpe_count} BPE fragments")
        else:
            logger.info(f"\n✅ Cleanup complete! Deleted {deleted} BPE fragments")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
