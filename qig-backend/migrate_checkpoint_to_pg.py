#!/usr/bin/env python3
"""
Migrate 32K Vocabulary Checkpoint to PostgreSQL

This script loads the QIG-pure 32K vocabulary checkpoint from
shared/coordizer/checkpoint_32000.json and merges it into the
PostgreSQL qig_vocabulary table.

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
from datetime import datetime
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


def transform_vocab_to_records(checkpoint: dict) -> list[dict]:
    """Transform checkpoint vocab to database records."""
    records = []
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
            
            # Skip if no vector
            if not vector or len(vector) != 64:
                logger.warning(f"Skipping token {name}: invalid vector length {len(vector) if vector else 0}")
                continue
            
            # Normalize to unit sphere
            basin_coords = normalize_to_unit_sphere(vector)
            
            # Compute phi score
            phi_score = compute_phi_from_vector(vector)
            
            # Determine source based on token name pattern
            if name.startswith('<byte_'):
                source = 'byte_level'
                domain = 'universal'
            elif name in ['<pad>', '<unk>', '<bos>', '<eos>']:
                source = 'special'
                domain = 'control'
            elif idx in merge_lookup:
                source = 'bpe_merge'
                domain = 'learned'
            else:
                source = f'checkpoint_{scale_type}'
                domain = 'general'
            
            # Get merge rule if applicable
            token_merge_rules = None
            if idx in merge_lookup:
                token_merge_rules = json.dumps(merge_lookup[idx])
            
            records.append({
                'token': name,
                'frequency': frequency,
                'phi_score': phi_score,
                'basin_coords': basin_coords,
                'source': source,
                'domain': domain,
                'coherence_rank': idx,  # Use index as initial rank
                'merge_rules': token_merge_rules,
                'observation_count': frequency,
                'semantic_group': scale_type,
            })
            
        except (ValueError, KeyError) as e:
            logger.warning(f"Error processing token {idx_str}: {e}")
            continue
    
    logger.info(f"Transformed {len(records)} tokens for database insertion")
    return records


def get_db_connection():
    """Get PostgreSQL connection from environment."""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    return psycopg2.connect(database_url)


def ensure_table_exists(conn):
    """Ensure qig_vocabulary table exists with pgvector extension."""
    with conn.cursor() as cur:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS qig_vocabulary (
                id SERIAL PRIMARY KEY,
                token TEXT UNIQUE NOT NULL,
                frequency INTEGER DEFAULT 1,
                phi_score REAL DEFAULT 0.5,
                basin_coords vector(64),
                source TEXT DEFAULT 'learned',
                domain TEXT DEFAULT 'general',
                coherence_rank INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                merge_rules JSONB,
                observation_count INTEGER DEFAULT 1,
                semantic_group TEXT,
                last_used_at TIMESTAMP
            );
        """)
        
        # Create indexes for efficient retrieval
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_vocab_token ON qig_vocabulary(token);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_vocab_phi ON qig_vocabulary(phi_score DESC);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_vocab_source ON qig_vocabulary(source);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_vocab_domain ON qig_vocabulary(domain);
        """)
        
        # Create vector index for similarity search (IVFFlat)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_vocab_basin_ivfflat 
            ON qig_vocabulary 
            USING ivfflat (basin_coords vector_cosine_ops)
            WITH (lists = 100);
        """)
        
        conn.commit()
        logger.info("Table qig_vocabulary ensured with indexes")


def insert_tokens_batch(conn, records: list[dict], batch_size: int = 1000, dry_run: bool = False):
    """Insert tokens in batches using upsert."""
    if dry_run:
        logger.info(f"[DRY RUN] Would insert {len(records)} tokens")
        return
    
    inserted = 0
    updated = 0
    
    with conn.cursor() as cur:
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            # Prepare values for batch insert
            values = []
            for r in batch:
                # Format vector as PostgreSQL array string
                basin_str = '[' + ','.join(map(str, r['basin_coords'])) + ']'
                values.append((
                    r['token'],
                    r['frequency'],
                    r['phi_score'],
                    basin_str,
                    r['source'],
                    r['domain'],
                    r['coherence_rank'],
                    r['merge_rules'],
                    r['observation_count'],
                    r['semantic_group'],
                ))
            
            # Upsert query - update if token exists, insert if not
            query = """
                INSERT INTO qig_vocabulary (
                    token, frequency, phi_score, basin_coords, source, domain,
                    coherence_rank, merge_rules, observation_count, semantic_group,
                    created_at, updated_at
                )
                VALUES %s
                ON CONFLICT (token) DO UPDATE SET
                    frequency = GREATEST(qig_vocabulary.frequency, EXCLUDED.frequency),
                    phi_score = EXCLUDED.phi_score,
                    basin_coords = EXCLUDED.basin_coords,
                    source = EXCLUDED.source,
                    domain = EXCLUDED.domain,
                    coherence_rank = EXCLUDED.coherence_rank,
                    merge_rules = COALESCE(EXCLUDED.merge_rules, qig_vocabulary.merge_rules),
                    observation_count = qig_vocabulary.observation_count + EXCLUDED.observation_count,
                    semantic_group = EXCLUDED.semantic_group,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            template = """(
                %s, %s, %s, %s::vector, %s, %s, %s, %s::jsonb, %s, %s,
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
            )"""
            
            execute_values(cur, query, values, template=template, page_size=batch_size)
            
            batch_inserted = cur.rowcount
            inserted += len(batch)
            
            logger.info(f"Processed batch {i // batch_size + 1}: {len(batch)} tokens")
        
        conn.commit()
    
    logger.info(f"Completed: {inserted} tokens processed")


def get_existing_token_count(conn) -> int:
    """Get count of existing tokens in vocabulary."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM qig_vocabulary")
        return cur.fetchone()[0]


def verify_migration(conn, expected_count: int):
    """Verify the migration was successful."""
    with conn.cursor() as cur:
        # Count total tokens
        cur.execute("SELECT COUNT(*) FROM qig_vocabulary")
        total = cur.fetchone()[0]
        
        # Count by source
        cur.execute("""
            SELECT source, COUNT(*) 
            FROM qig_vocabulary 
            GROUP BY source 
            ORDER BY COUNT(*) DESC
        """)
        by_source = cur.fetchall()
        
        # Get phi score distribution
        cur.execute("""
            SELECT 
                MIN(phi_score) as min_phi,
                AVG(phi_score) as avg_phi,
                MAX(phi_score) as max_phi
            FROM qig_vocabulary
        """)
        phi_stats = cur.fetchone()
        
        # Sample high-phi tokens
        cur.execute("""
            SELECT token, phi_score, source 
            FROM qig_vocabulary 
            ORDER BY phi_score DESC 
            LIMIT 10
        """)
        top_phi = cur.fetchall()
    
    logger.info("\n" + "="*50)
    logger.info("MIGRATION VERIFICATION")
    logger.info("="*50)
    logger.info(f"Total tokens in database: {total}")
    logger.info(f"Expected: ~{expected_count}")
    logger.info(f"\nTokens by source:")
    for source, count in by_source:
        logger.info(f"  {source}: {count}")
    logger.info(f"\nPhi score distribution:")
    logger.info(f"  Min: {phi_stats[0]:.4f}")
    logger.info(f"  Avg: {phi_stats[1]:.4f}")
    logger.info(f"  Max: {phi_stats[2]:.4f}")
    logger.info(f"\nTop 10 high-phi tokens:")
    for token, phi, source in top_phi:
        logger.info(f"  {token}: phi={phi:.4f} ({source})")
    logger.info("="*50)
    
    return total >= expected_count * 0.95  # Allow 5% margin


def main():
    parser = argparse.ArgumentParser(description='Migrate 32K vocabulary checkpoint to PostgreSQL')
    parser.add_argument('--dry-run', action='store_true', help='Preview without making changes')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for inserts')
    parser.add_argument('--checkpoint', type=str, default=str(CHECKPOINT_PATH), help='Path to checkpoint JSON')
    args = parser.parse_args()
    
    try:
        # Load checkpoint
        checkpoint = load_checkpoint(Path(args.checkpoint))
        
        # Load corpus coords (optional)
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
            logger.info(f"Existing tokens in database: {existing_count}")
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
                logger.info("\n✅ Migration completed successfully!")
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
