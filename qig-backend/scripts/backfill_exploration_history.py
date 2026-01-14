#!/usr/bin/env python3
"""
Backfill script for exploration_history table.

Populates NULL values for:
- basin_hash: Computed from topic basin coordinates (if available)
- source_type: Sets to 'unknown' if NULL
- information_gain: Sets to 0.1 (non-zero default) if 0.0

Usage:
    python scripts/backfill_exploration_history.py [--dry-run]
"""

import hashlib
import os
import sys
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor


def get_connection():
    """Get database connection from DATABASE_URL."""
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)
    return psycopg2.connect(db_url)


def compute_basin_hash(coords: list) -> str:
    """Compute MD5 hash of basin coordinates (16 char hex)."""
    return hashlib.md5(str(coords).encode()).hexdigest()[:16]


def get_topic_basin(conn, topic: str) -> Optional[list]:
    """
    Try to find basin coordinates for a topic from various sources.
    
    Sources checked:
    1. shadow_knowledge (research results)
    2. coordizer_vocabulary (if topic is a word)
    3. vocabulary_observations (learned vocabulary)
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Try shadow_knowledge first (most likely for research topics)
        cur.execute("""
            SELECT basin_coords 
            FROM shadow_knowledge 
            WHERE topic ILIKE %s OR topic ILIKE %s
            AND basin_coords IS NOT NULL
            LIMIT 1
        """, (topic, f"%{topic[:50]}%"))
        row = cur.fetchone()
        if row and row['basin_coords']:
            return row['basin_coords']
        
        # Try coordizer_vocabulary for single-word topics
        if ' ' not in topic.strip():
            cur.execute("""
                SELECT basin_embedding::float8[] as coords
                FROM coordizer_vocabulary 
                WHERE LOWER(token) = LOWER(%s)
                AND basin_embedding IS NOT NULL
                LIMIT 1
            """, (topic.strip(),))
            row = cur.fetchone()
            if row and row['coords']:
                return list(row['coords'])
        
        # Try vocabulary_observations (uses basin_coords column)
        cur.execute("""
            SELECT basin_coords as coords
            FROM vocabulary_observations 
            WHERE LOWER(text) = LOWER(%s)
            AND basin_coords IS NOT NULL
            LIMIT 1
        """, (topic.strip(),))
        row = cur.fetchone()
        if row and row['coords']:
            return list(row['coords'])
    
    return None


def backfill_exploration_history(dry_run: bool = False):
    """
    Backfill NULL values in exploration_history table.
    """
    conn = get_connection()
    
    stats = {
        'total': 0,
        'basin_hash_updated': 0,
        'source_type_updated': 0,
        'info_gain_updated': 0,
        'basin_not_found': 0
    }
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get rows needing updates
            cur.execute("""
                SELECT id, topic, query, basin_hash, source_type, information_gain
                FROM exploration_history
                WHERE basin_hash IS NULL 
                   OR source_type IS NULL 
                   OR information_gain = 0.0
                ORDER BY created_at DESC
            """)
            rows = cur.fetchall()
            stats['total'] = len(rows)
            
            print(f"Found {len(rows)} rows needing updates")
            
            for row in rows:
                updates = []
                params = []
                
                # Fix basin_hash if NULL
                if row['basin_hash'] is None:
                    basin_coords = get_topic_basin(conn, row['topic'])
                    if basin_coords:
                        basin_hash = compute_basin_hash(basin_coords)
                        updates.append("basin_hash = %s")
                        params.append(basin_hash)
                        stats['basin_hash_updated'] += 1
                    else:
                        stats['basin_not_found'] += 1
                
                # Fix source_type if NULL
                if row['source_type'] is None:
                    updates.append("source_type = %s")
                    params.append('unknown')
                    stats['source_type_updated'] += 1
                
                # Fix information_gain if 0.0 (set non-zero default)
                if row['information_gain'] == 0.0:
                    updates.append("information_gain = %s")
                    params.append(0.1)  # Default non-zero value
                    stats['info_gain_updated'] += 1
                
                if updates and not dry_run:
                    params.append(row['id'])
                    cur.execute(f"""
                        UPDATE exploration_history 
                        SET {', '.join(updates)}
                        WHERE id = %s
                    """, params)
            
            if not dry_run:
                conn.commit()
                print("Changes committed to database")
            else:
                print("DRY RUN - no changes made")
        
        print("\n=== Backfill Results ===")
        print(f"Total rows processed: {stats['total']}")
        print(f"basin_hash updated: {stats['basin_hash_updated']}")
        print(f"basin_hash not found (no basin coords): {stats['basin_not_found']}")
        print(f"source_type updated: {stats['source_type_updated']}")
        print(f"information_gain updated: {stats['info_gain_updated']}")
        
    except Exception as e:
        conn.rollback()
        print(f"ERROR: {e}")
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    if dry_run:
        print("Running in DRY RUN mode...")
    backfill_exploration_history(dry_run=dry_run)
