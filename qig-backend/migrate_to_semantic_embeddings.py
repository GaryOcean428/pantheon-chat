#!/usr/bin/env python3
"""
Migrate Database Embeddings to Semantic Clustering.

This script regenerates all basin_embedding values in the tokenizer_vocabulary
table using the new semantic domain clustering system, so that semantically
related words (like 'consciousness', 'mind', 'awareness') cluster together
in the 64D Fisher manifold.

Usage:
    python migrate_to_semantic_embeddings.py [--dry-run] [--batch-size N]

Options:
    --dry-run       Preview changes without updating database
    --batch-size N  Number of tokens to update per batch (default: 500)
    --verbose       Show detailed progress
"""

import argparse
import logging
import os
import sys
from typing import List, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coordizers.semantic_domains import (
    compute_semantic_embedding,
    get_word_domains,
    SEMANTIC_DOMAINS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_database_connection():
    """Get PostgreSQL connection from environment."""
    load_dotenv(dotenv_path='../.env')
    load_dotenv()  # Also try current directory
    
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL not set in environment")
    
    return psycopg2.connect(db_url)


def fetch_all_tokens(conn) -> List[Tuple[int, str, str]]:
    """Fetch all tokens from tokenizer_vocabulary.
    
    Returns:
        List of (id, token, source_type) tuples
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, token, source_type
            FROM tokenizer_vocabulary
            WHERE token IS NOT NULL
            ORDER BY id
        """)
        return cur.fetchall()


def embedding_to_pg_vector(embedding: np.ndarray) -> str:
    """Convert numpy array to PostgreSQL vector string."""
    return '[' + ','.join(f'{x:.8f}' for x in embedding) + ']'


def update_embeddings_batch(
    conn,
    updates: List[Tuple[int, str]],
    dry_run: bool = False
) -> int:
    """Update a batch of embeddings in the database.
    
    Args:
        conn: Database connection
        updates: List of (id, vector_string) tuples
        dry_run: If True, don't actually update
    
    Returns:
        Number of rows updated
    """
    if dry_run or not updates:
        return len(updates)
    
    with conn.cursor() as cur:
        # Use execute_values for efficient batch update
        query = """
            UPDATE tokenizer_vocabulary AS t
            SET basin_embedding = v.embedding::vector,
                updated_at = CURRENT_TIMESTAMP
            FROM (VALUES %s) AS v(id, embedding)
            WHERE t.id = v.id
        """
        execute_values(cur, query, updates, template="(%s, %s)")
        updated = cur.rowcount
    
    return updated


def get_domain_stats(tokens: List[Tuple[int, str, str]]) -> dict:
    """Compute statistics about domain coverage."""
    stats = {
        'total': len(tokens),
        'with_domains': 0,
        'without_domains': 0,
        'domain_counts': {},
    }
    
    for _, token, _ in tokens:
        domains = get_word_domains(token)
        if domains:
            stats['with_domains'] += 1
            for domain, _ in domains:
                stats['domain_counts'][domain] = stats['domain_counts'].get(domain, 0) + 1
        else:
            stats['without_domains'] += 1
    
    return stats


def migrate_embeddings(
    dry_run: bool = False,
    batch_size: int = 500,
    verbose: bool = False
):
    """Migrate all embeddings to use semantic clustering.
    
    Args:
        dry_run: If True, preview changes without updating
        batch_size: Number of tokens to process per batch
        verbose: Show detailed progress
    """
    logger.info("Starting semantic embedding migration...")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE UPDATE'}")
    
    # Connect to database
    conn = get_database_connection()
    logger.info("Connected to database")
    
    try:
        # Fetch all tokens
        tokens = fetch_all_tokens(conn)
        logger.info(f"Found {len(tokens)} tokens to migrate")
        
        if not tokens:
            logger.warning("No tokens found in database")
            return
        
        # Compute domain statistics
        if verbose:
            stats = get_domain_stats(tokens)
            logger.info(f"Domain coverage: {stats['with_domains']}/{stats['total']} tokens have semantic domains")
            logger.info(f"Domain distribution:")
            for domain, count in sorted(stats['domain_counts'].items(), key=lambda x: -x[1]):
                logger.info(f"  {domain}: {count} tokens")
        
        # Process in batches
        total_updated = 0
        total_errors = 0
        
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i + batch_size]
            updates = []
            
            for token_id, token, source_type in batch:
                try:
                    # Generate semantic embedding
                    embedding = compute_semantic_embedding(token, fallback_hash=True)
                    
                    # Convert to PostgreSQL vector format
                    vector_str = embedding_to_pg_vector(embedding)
                    
                    updates.append((token_id, vector_str))
                    
                    if verbose and len(updates) <= 3:
                        domains = get_word_domains(token)
                        domain_str = ', '.join(d for d, _ in domains[:2]) if domains else 'none'
                        logger.info(f"  {token}: domains=[{domain_str}]")
                        
                except Exception as e:
                    total_errors += 1
                    if verbose:
                        logger.error(f"  Error processing '{token}': {e}")
            
            # Update batch
            if updates:
                updated = update_embeddings_batch(conn, updates, dry_run=dry_run)
                total_updated += updated
            
            # Progress
            progress = min(i + batch_size, len(tokens))
            logger.info(f"Progress: {progress}/{len(tokens)} tokens processed ({100*progress/len(tokens):.1f}%)")
        
        # Commit transaction
        if not dry_run:
            conn.commit()
            logger.info("Transaction committed")
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("MIGRATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total tokens: {len(tokens)}")
        logger.info(f"Updated: {total_updated}")
        logger.info(f"Errors: {total_errors}")
        logger.info(f"Mode: {'DRY RUN (no changes made)' if dry_run else 'LIVE UPDATE (changes committed)'}")
        
        # Verify a few examples
        if not dry_run and verbose:
            logger.info("\nVerification samples:")
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT token, source_type
                    FROM tokenizer_vocabulary
                    WHERE token IN ('consciousness', 'mind', 'quantum', 'happy', 'money')
                """)
                for row in cur.fetchall():
                    logger.info(f"  {row[0]} ({row[1]}): embedding updated")
        
    finally:
        conn.close()
        logger.info("Database connection closed")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate database embeddings to semantic clustering'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without updating database'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=500,
        help='Number of tokens to update per batch (default: 500)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed progress'
    )
    
    args = parser.parse_args()
    
    try:
        migrate_embeddings(
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            verbose=args.verbose
        )
        logger.info("\n✅ Migration completed successfully!")
    except Exception as e:
        logger.error(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
