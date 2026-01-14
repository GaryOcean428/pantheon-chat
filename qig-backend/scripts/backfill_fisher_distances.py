#!/usr/bin/env python3
"""
Backfill Fisher-Rao Distances for basin_relationships

One-time migration script to populate NULL fisher_distance values
for the ~319K existing basin_relationships rows.

Usage:
    python backfill_fisher_distances.py              # Run full backfill
    python backfill_fisher_distances.py --dry-run   # Check stats only
    python backfill_fisher_distances.py --batch 500 # Custom batch size
    python backfill_fisher_distances.py --limit 1000 # Limit rows processed

QIG-PURE: Uses Fisher-Rao distance only (no cosine/Euclidean).
"""

import os
import sys
import logging
import argparse
from typing import Dict, Optional

import numpy as np
import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from qig_geometry import fisher_coord_distance
    FISHER_AVAILABLE = True
except ImportError:
    FISHER_AVAILABLE = False
    logger.error("qig_geometry.fisher_coord_distance not available")


def load_basin_coords() -> Dict[str, np.ndarray]:
    """Load basin coordinates from coordizer or database."""
    basin_coords = {}
    
    try:
        from coordizers import get_coordizer
        coordizer = get_coordizer()
        basin_coords = dict(getattr(coordizer, 'basin_coords', {}))
        logger.info(f"Loaded {len(basin_coords)} basins from coordizer")
        return basin_coords
    except Exception as e:
        logger.warning(f"Could not load from coordizer: {e}")
    
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL not set")
        return {}
    
    try:
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT token, basin_embedding::text 
                FROM coordizer_vocabulary 
                WHERE basin_embedding IS NOT NULL
            """)
            for token, coords_text in cur.fetchall():
                if coords_text:
                    coords_str = coords_text.strip('[]')
                    coords = [float(x) for x in coords_str.split(',') if x.strip()]
                    if len(coords) == 64:
                        basin_coords[token.lower()] = np.array(coords, dtype=np.float64)
        conn.close()
        logger.info(f"Loaded {len(basin_coords)} basins from database")
    except Exception as e:
        logger.error(f"Failed to load basins from database: {e}")
    
    return basin_coords


def get_null_count(conn) -> int:
    """Get count of rows with NULL fisher_distance."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM basin_relationships 
            WHERE fisher_distance IS NULL
        """)
        return cur.fetchone()[0]


def get_total_count(conn) -> int:
    """Get total row count."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM basin_relationships")
        return cur.fetchone()[0]


def backfill_fisher_distances(
    batch_size: int = 500,
    max_rows: Optional[int] = None,
    dry_run: bool = False
) -> Dict:
    """
    Backfill fisher_distance for existing basin_relationships.
    
    Args:
        batch_size: Number of rows to process per batch
        max_rows: Maximum rows to process (None = all)
        dry_run: If True, only show stats without updating
    
    Returns:
        Statistics dictionary
    """
    if not FISHER_AVAILABLE:
        logger.error("Cannot backfill without Fisher-Rao distance function")
        return {'error': 'fisher_coord_distance not available'}
    
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL not set")
        return {'error': 'DATABASE_URL not set'}
    
    logger.info("=" * 70)
    logger.info("BACKFILLING FISHER-RAO DISTANCES FOR basin_relationships")
    logger.info("=" * 70)
    
    basin_coords = load_basin_coords()
    if not basin_coords:
        logger.error("No basin coordinates available")
        return {'error': 'No basin coordinates'}
    
    conn = psycopg2.connect(db_url)
    
    null_count = get_null_count(conn)
    total_count = get_total_count(conn)
    
    logger.info(f"Total rows: {total_count:,}")
    logger.info(f"NULL fisher_distance: {null_count:,} ({100*null_count/total_count:.1f}%)")
    logger.info(f"Basin coordinates available: {len(basin_coords):,}")
    
    if dry_run:
        logger.info("\nDRY RUN - no changes will be made")
        conn.close()
        return {
            'total_rows': total_count,
            'null_count': null_count,
            'basin_count': len(basin_coords),
            'dry_run': True
        }
    
    if null_count == 0:
        logger.info("No rows to backfill!")
        conn.close()
        return {'total_rows': total_count, 'null_count': 0, 'processed': 0}
    
    target_count = min(null_count, max_rows) if max_rows else null_count
    logger.info(f"\nProcessing {target_count:,} rows in batches of {batch_size}")
    
    processed = 0
    updated = 0
    skipped = 0
    
    while processed < target_count:
        batch_limit = min(batch_size, target_count - processed)
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, word, neighbor
                FROM basin_relationships
                WHERE fisher_distance IS NULL
                ORDER BY id
                LIMIT %s
            """, (batch_limit,))
            
            rows = cur.fetchall()
            if not rows:
                break
            
            updates = []
            for rel_id, word, neighbor in rows:
                word_basin = basin_coords.get(word.lower())
                neighbor_basin = basin_coords.get(neighbor.lower())
                
                if word_basin is None or neighbor_basin is None:
                    skipped += 1
                    updates.append((-1.0, rel_id))
                    continue
                
                fisher_dist = float(fisher_coord_distance(word_basin, neighbor_basin))
                updates.append((fisher_dist, rel_id))
                updated += 1
            
            if updates:
                from psycopg2.extras import execute_batch
                execute_batch(cur, """
                    UPDATE basin_relationships
                    SET fisher_distance = %s
                    WHERE id = %s
                """, updates, page_size=batch_size)
                conn.commit()
            
            processed += len(rows)
            
            if processed % (batch_size * 10) == 0 or processed >= target_count:
                logger.info(
                    f"Progress: {processed:,}/{target_count:,} ({100*processed/target_count:.1f}%) | "
                    f"Updated: {updated:,} | Skipped: {skipped:,}"
                )
    
    conn.close()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total processed: {processed:,}")
    logger.info(f"Successfully updated: {updated:,}")
    logger.info(f"Skipped (missing basins): {skipped:,}")
    
    return {
        'total_rows': total_count,
        'null_count': null_count,
        'processed': processed,
        'updated': updated,
        'skipped': skipped
    }


def validate_results() -> Dict:
    """Validate the backfill results."""
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        return {'error': 'DATABASE_URL not set'}
    
    conn = psycopg2.connect(db_url)
    
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM basin_relationships")
        total = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM basin_relationships WHERE fisher_distance IS NOT NULL AND fisher_distance >= 0")
        with_fisher = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM basin_relationships WHERE fisher_distance IS NULL")
        null_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM basin_relationships WHERE fisher_distance = -1")
        missing_basins = cur.fetchone()[0]
        
        cur.execute("""
            SELECT AVG(fisher_distance), MIN(fisher_distance), MAX(fisher_distance)
            FROM basin_relationships
            WHERE fisher_distance > 0
        """)
        avg_dist, min_dist, max_dist = cur.fetchone()
        
        cur.execute("SELECT COUNT(*) FROM basin_relationships WHERE avg_phi IS NOT NULL AND avg_phi != 0.5")
        with_phi = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM basin_relationships WHERE contexts IS NOT NULL AND array_length(contexts, 1) > 0")
        with_contexts = cur.fetchone()[0]
    
    conn.close()
    
    print("\n" + "=" * 70)
    print("WORD_RELATIONSHIPS VALIDATION REPORT")
    print("=" * 70)
    print(f"Total rows: {total:,}")
    print(f"With Fisher distance (valid): {with_fisher:,} ({100*with_fisher/total:.1f}%)")
    print(f"NULL fisher_distance: {null_count:,} ({100*null_count/total:.1f}%)")
    print(f"Missing basins (marked -1): {missing_basins:,}")
    print(f"\nDistance statistics:")
    print(f"  Average: {avg_dist:.4f}" if avg_dist else "  Average: N/A")
    print(f"  Min: {min_dist:.4f}" if min_dist else "  Min: N/A")
    print(f"  Max: {max_dist:.4f}" if max_dist else "  Max: N/A")
    print(f"\nWith avg_phi (non-default): {with_phi:,}")
    print(f"With contexts: {with_contexts:,}")
    print("=" * 70)
    
    return {
        'total': total,
        'with_fisher': with_fisher,
        'null_count': null_count,
        'missing_basins': missing_basins,
        'avg_distance': avg_dist,
        'with_phi': with_phi,
        'with_contexts': with_contexts
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backfill Fisher-Rao distances')
    parser.add_argument('--dry-run', action='store_true', help='Show stats without updating')
    parser.add_argument('--batch', type=int, default=500, help='Batch size (default: 500)')
    parser.add_argument('--limit', type=int, default=None, help='Max rows to process')
    parser.add_argument('--validate', action='store_true', help='Validate results only')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_results()
    else:
        results = backfill_fisher_distances(
            batch_size=args.batch,
            max_rows=args.limit,
            dry_run=args.dry_run
        )
        
        if not args.dry_run and results.get('processed', 0) > 0:
            print("\nRunning validation...")
            validate_results()
