#!/usr/bin/env python3
"""
Migrate 32K Vocabulary Checkpoint to PostgreSQL coordizer_vocabulary

This script loads the QIG-pure 32K vocabulary checkpoint from
shared/coordizer/checkpoint_32000.json and merges it into the
PostgreSQL coordizer_vocabulary table (the actual vocabulary table
used by the QIG system).

Usage:
    python migrate_checkpoint_to_pg.py [--dry-run] [--batch-size N]

Requires:
    - DATABASE_URL environment variable
    - shared/coordizer/checkpoint_32000.json
    - shared/coordizer/corpus_coords_32000.npy (optional)
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / 'shared' / 'coordizer' / 'checkpoint_32000.json'
CORPUS_COORDS_PATH = PROJECT_ROOT / 'shared' / 'coordizer' / 'corpus_coords_32000.npy'


def normalize_to_unit_sphere(vector: list[float]) -> list[float]:
    """Normalize vector to unit sphere for Fisher manifold."""
    arr = np.array(vector, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < 1e-10:
        # Return uniform distribution on sphere
        arr = np.ones(64) / np.sqrt(64)
    else:
        arr = arr / norm
    return arr.tolist()


def compute_phi_from_vector(vector: list[float]) -> float:
    """Compute approximate phi (integration) score from basin vector.
    
    Uses entropy-based approximation: higher entropy = higher phi.
    """
    arr = np.array(vector, dtype=np.float64)
    arr = np.abs(arr) + 1e-10  # Ensure positive for probability interpretation
    arr = arr / arr.sum()  # Normalize to probability distribution
    entropy = -np.sum(arr * np.log(arr + 1e-10))
    # Normalize to [0, 1] range (max entropy for 64D is ln(64) ≈ 4.16)
    phi = entropy / np.log(64)
    return float(np.clip(phi, 0.0, 1.0))


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load the 32K vocabulary checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded checkpoint with {len(data.get('vocab', {}))} tokens")
    logger.info(f"Scale type: {data.get('scale_type', 'unknown')}")
    
    return data


def load_corpus_coords(corpus_path: Path) -> Optional[np.ndarray]:
    """Load corpus coordinates (optional)."""
    if not corpus_path.exists():
        logger.warning(f"Corpus coordinates not found: {corpus_path}")
        return None
    
    coords = np.load(corpus_path)
    logger.info(f"Loaded corpus coordinates: shape={coords.shape}, dtype={coords.dtype}")
    return coords


# Common BPE suffixes/prefixes that are NOT real words
BPE_FRAGMENTS = {
    'tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ful', 'less', 'ous', 'ive',
    'ing', 'ed', 'er', 'est', 'ly', 'al', 'ity', 'ty', 'ry', 'ary', 'ory',
    'th', 'st', 'nd', 'rd', 'ght', 'ck', 'ch', 'sh', 'wh', 'ph', 'gh',
    'pre', 'pro', 'per', 'con', 'com', 'dis', 'mis', 'sub', 'sup', 'trans',
    'un', 're', 'in', 'im', 'il', 'ir', 'non', 'anti', 'auto', 'bi', 'co',
    '##', '@@', '▁',  # Common BPE markers
}

def is_real_word(token: str) -> bool:
    """Check if token is a real word (not BPE fragment or byte token).
    
    Strict filtering to avoid BPE garble in generation.
    """
    # Byte tokens are not real words
    if token.startswith('<byte_') or token.startswith('<') or token.endswith('>'):
        return False
    
    # BPE markers
    if token.startswith('##') or token.startswith('@@') or token.startswith('▁'):
        return False
    
    # Must be fully alphabetic (allow hyphens and apostrophes for compound words)
    clean_token = token.replace('-', '').replace("'", '').replace("'", '')
    if not clean_token.isalpha():
        return False
    
    # Minimum length for real words (3 chars to exclude fragments like "ed", "ly")
    if len(token) < 3:
        return False
    
    # Maximum length sanity check
    if len(token) > 30:
        return False
    
    # Exclude known BPE fragments (but allow longer real words like "professional")
    if len(token) <= 5 and token.lower() in BPE_FRAGMENTS:
        return False
    
    return True


def transform_vocab_to_records(checkpoint: dict) -> list[dict]:
    """Transform checkpoint vocab to database records for coordizer_vocabulary table."""
    records = []
    seen_tokens = set()  # Track seen tokens to avoid duplicates
    vocab = checkpoint.get('vocab', {})
    merge_rules = checkpoint.get('merge_rules', [])
    scale_type = checkpoint.get('scale_type', 'byte')
    
    # Build merge rules lookup (token_id -> merge info)
    merge_lookup = {}
    for rule in merge_rules:
        if isinstance(rule, dict):
            result_id = rule.get('result')
            if result_id:
                merge_lookup[result_id] = rule
    
    for idx_str, token_data in vocab.items():
        try:
            idx = int(idx_str)
            name = token_data.get('name', f'<token_{idx}>')
            vector = token_data.get('vector', [])
            frequency = token_data.get('frequency', 1)
            
            # Skip if no vector or invalid dimension
            if not vector or len(vector) != 64:
                logger.warning(f"Skipping token {name}: invalid vector length {len(vector) if vector else 0}")
                continue
            
            # Normalize to unit sphere
            basin_embedding = normalize_to_unit_sphere(vector)
            
            # Compute phi score
            phi_score = compute_phi_from_vector(vector)
            
            # Determine source type based on token name pattern
            if name.startswith('<byte_'):
                source_type = 'byte_level'
            elif name in ['<pad>', '<unk>', '<bos>', '<eos>']:
                source_type = 'special'
            elif idx in merge_lookup:
                source_type = 'learned'  # BPE merged tokens
            elif is_real_word(name):
                source_type = 'base'  # Real words
            else:
                source_type = f'checkpoint_{scale_type}'
            
            # Skip duplicate tokens (keep first occurrence)
            if name in seen_tokens:
                logger.debug(f"Skipping duplicate token: {name}")
                continue
            seen_tokens.add(name)
            
            # Compute weight based on frequency and phi
            weight = 1.0 + (phi_score * 0.5) + (min(frequency, 1000) / 1000 * 0.3)
            
            records.append({
                'token': name,
                'token_id': idx,
                'weight': weight,
                'frequency': frequency,
                'phi_score': phi_score,
                'basin_embedding': basin_embedding,
                'source_type': source_type,
            })
            
        except (ValueError, KeyError) as e:
            logger.warning(f"Error processing token {idx_str}: {e}")
            continue
    
    logger.info(f"Transformed {len(records)} unique tokens for database insertion (deduplicated from {len(vocab)})")
    
    # Log source type distribution
    source_counts = {}
    for r in records:
        st = r['source_type']
        source_counts[st] = source_counts.get(st, 0) + 1
    logger.info(f"Source type distribution: {source_counts}")
    
    return records


def get_db_connection():
    """Get PostgreSQL connection from environment."""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    return psycopg2.connect(database_url)


def ensure_table_exists(conn):
    """Ensure coordizer_vocabulary table exists with pgvector extension."""
    with conn.cursor() as cur:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create table if not exists (matching schema.ts definition)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS coordizer_vocabulary (
                id SERIAL PRIMARY KEY,
                token TEXT NOT NULL UNIQUE,
                token_id INTEGER NOT NULL UNIQUE,
                weight DOUBLE PRECISION DEFAULT 1.0,
                frequency INTEGER DEFAULT 1,
                phi_score DOUBLE PRECISION DEFAULT 0,
                basin_embedding vector(64),
                source_type VARCHAR(32) DEFAULT 'base',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create indexes for efficient retrieval
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_tokenizer_vocab_token_id ON coordizer_vocabulary(token_id);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_tokenizer_vocab_phi ON coordizer_vocabulary(phi_score);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_tokenizer_vocab_weight ON coordizer_vocabulary(weight);
        """)
        
        conn.commit()
        logger.info("Table coordizer_vocabulary ensured with indexes")


def insert_tokens_batch(conn, records: list[dict], batch_size: int = 100, dry_run: bool = False):
    """Insert tokens one at a time using upsert to handle conflicts."""
    if dry_run:
        logger.info(f"[DRY RUN] Would insert {len(records)} tokens")
        # Show sample of what would be inserted
        logger.info("Sample records:")
        for r in records[:10]:
            logger.info(f"  {r['token']}: token_id={r['token_id']}, phi={r['phi_score']:.4f}, source={r['source_type']}")
        return
    
    inserted = 0
    updated = 0
    errors = 0
    
    with conn.cursor() as cur:
        for i, r in enumerate(records):
            try:
                # Format vector as PostgreSQL array string
                basin_str = '[' + ','.join(map(str, r['basin_embedding'])) + ']'
                
                # Upsert query - one at a time to avoid batch conflicts
                cur.execute("""
                    INSERT INTO coordizer_vocabulary (
                        token, token_id, weight, frequency, phi_score, basin_embedding, source_type,
                        created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (token) DO UPDATE SET
                        weight = GREATEST(coordizer_vocabulary.weight, EXCLUDED.weight),
                        frequency = GREATEST(coordizer_vocabulary.frequency, EXCLUDED.frequency),
                        phi_score = GREATEST(coordizer_vocabulary.phi_score, EXCLUDED.phi_score),
                        basin_embedding = COALESCE(EXCLUDED.basin_embedding, coordizer_vocabulary.basin_embedding),
                        source_type = EXCLUDED.source_type,
                        updated_at = CURRENT_TIMESTAMP
                """, (r['token'], r['token_id'], r['weight'], r['frequency'], r['phi_score'], basin_str, r['source_type']))
                
                if cur.rowcount > 0:
                    inserted += 1
                else:
                    updated += 1
                    
            except Exception as e:
                errors += 1
                if errors <= 10:  # Only log first 10 errors
                    logger.warning(f"Error inserting token '{r['token']}': {e}")
                continue
            
            # Commit and log progress periodically
            if (i + 1) % batch_size == 0:
                conn.commit()
                logger.info(f"Progress: {i + 1}/{len(records)} tokens processed")
        
        conn.commit()
    
    logger.info(f"Completed: {inserted} inserted, {updated} updated, {errors} errors")


def get_existing_token_count(conn) -> int:
    """Get count of existing tokens in vocabulary."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM coordizer_vocabulary")
        return cur.fetchone()[0]


def verify_migration(conn, expected_count: int):
    """Verify the migration was successful."""
    with conn.cursor() as cur:
        # Count total tokens
        cur.execute("SELECT COUNT(*) FROM coordizer_vocabulary")
        total = cur.fetchone()[0]
        
        # Count by source type
        cur.execute("""
            SELECT source_type, COUNT(*) 
            FROM coordizer_vocabulary 
            GROUP BY source_type 
            ORDER BY COUNT(*) DESC
        """)
        by_source = cur.fetchall()
        
        # Get phi score distribution
        cur.execute("""
            SELECT 
                MIN(phi_score) as min_phi,
                AVG(phi_score) as avg_phi,
                MAX(phi_score) as max_phi
            FROM coordizer_vocabulary
        """)
        phi_stats = cur.fetchone()
        
        # Sample high-phi real words (not byte tokens)
        cur.execute("""
            SELECT token, phi_score, source_type 
            FROM coordizer_vocabulary 
            WHERE source_type IN ('base', 'learned')
              AND LENGTH(token) >= 2
              AND token NOT LIKE '<%%>'
            ORDER BY phi_score DESC 
            LIMIT 20
        """)
        top_phi = cur.fetchall()
        
        # Count real words vs byte tokens
        # Migration 017 (2026-01-19): Count coordizer_vocabulary by source_type instead of learned_words
        cur.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE source_type = 'base') as base_words,
                COUNT(*) FILTER (WHERE source_type = 'learned') as generation_words,
                COUNT(*) FILTER (WHERE source_type = 'byte_level') as byte_tokens,
                COUNT(*) FILTER (WHERE source_type = 'special') as special_tokens
            FROM coordizer_vocabulary
        """)
        word_counts = cur.fetchone()
    
    logger.info("\n" + "="*60)
    logger.info("MIGRATION VERIFICATION")
    logger.info("="*60)
    logger.info(f"Total tokens in database: {total}")
    logger.info(f"Expected: ~{expected_count}")
    logger.info(f"\nTokens by source type:")
    for source, count in by_source:
        logger.info(f"  {source}: {count}")
    logger.info(f"\nWord type breakdown:")
    logger.info(f"  Base words: {word_counts[0]}")
    logger.info(f"  Generation words (learned): {word_counts[1]}")
    logger.info(f"  Byte tokens: {word_counts[2]}")
    logger.info(f"  Special tokens: {word_counts[3]}")
    logger.info(f"\nPhi score distribution:")
    logger.info(f"  Min: {phi_stats[0]:.4f}")
    logger.info(f"  Avg: {phi_stats[1]:.4f}")
    logger.info(f"  Max: {phi_stats[2]:.4f}")
    logger.info(f"\nTop 20 high-phi REAL WORDS (not byte tokens):")
    for token, phi, source in top_phi:
        logger.info(f"  {token}: phi={phi:.4f} ({source})")
    logger.info("="*60)
    
    return total >= expected_count * 0.95  # Allow 5% margin


def main():
    parser = argparse.ArgumentParser(description='Migrate 32K vocabulary checkpoint to PostgreSQL coordizer_vocabulary')
    parser.add_argument('--dry-run', action='store_true', help='Preview without making changes')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for inserts')
    parser.add_argument('--checkpoint', type=str, default=str(CHECKPOINT_PATH), help='Path to checkpoint JSON')
    args = parser.parse_args()
    
    try:
        # Load checkpoint
        checkpoint = load_checkpoint(Path(args.checkpoint))
        
        # Load corpus coords (optional, for reference)
        corpus_coords = load_corpus_coords(CORPUS_COORDS_PATH)
        
        # Transform to records
        records = transform_vocab_to_records(checkpoint)
        
        if not records:
            logger.error("No valid tokens to migrate")
            sys.exit(1)
        
        # Connect to database
        logger.info("Connecting to PostgreSQL...")
        conn = get_db_connection()
        
        # Get existing count
        try:
            existing_count = get_existing_token_count(conn)
            logger.info(f"Existing tokens in coordizer_vocabulary: {existing_count}")
        except psycopg2.errors.UndefinedTable:
            existing_count = 0
            logger.info("Table does not exist yet, will create")
            conn.rollback()
        
        # Ensure table exists
        ensure_table_exists(conn)
        
        # Insert tokens
        logger.info(f"Inserting {len(records)} tokens (batch_size={args.batch_size})...")
        insert_tokens_batch(conn, records, batch_size=args.batch_size, dry_run=args.dry_run)
        
        # Verify migration
        if not args.dry_run:
            success = verify_migration(conn, len(records))
            if success:
                logger.info("\n✅ Migration to coordizer_vocabulary completed successfully!")
            else:
                logger.warning("\n⚠️ Migration completed but verification found discrepancies")
        else:
            logger.info("\n[DRY RUN] No changes made to database")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
