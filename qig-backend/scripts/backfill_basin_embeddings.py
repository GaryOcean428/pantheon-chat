#!/usr/bin/env python3
"""
Backfill Basin Embeddings Script
=================================

Regenerates missing/invalid basin embeddings for vocabulary entries.

This script:
1. Finds words with empty/invalid basin_embedding
2. Recomputes basins using QIG-pure pipeline (coordizer)
3. Updates database with valid 64D basins
4. Reports success/failure statistics

Usage:
    # Dry run (show what would be backfilled)
    python qig-backend/scripts/backfill_basin_embeddings.py --limit 100
    
    # Execute backfill
    python qig-backend/scripts/backfill_basin_embeddings.py --execute
    
    # Execute with limit
    python qig-backend/scripts/backfill_basin_embeddings.py --execute --limit 1000

Requirements:
- DATABASE_URL environment variable set
- VocabularyIngestionService available
- Coordizer initialized
"""

import sys
import os
import argparse
from typing import List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[WARNING] tqdm not available - progress bar disabled")

try:
    from vocabulary_ingestion import get_ingestion_service
    from vocabulary_persistence import get_vocabulary_persistence
    SERVICES_AVAILABLE = True
except ImportError as e:
    SERVICES_AVAILABLE = False
    print(f"[ERROR] Required services not available: {e}")
    sys.exit(1)

# Column name whitelist (prevent SQL injection)
BASIN_COLUMN_PRE_MIGRATION = 'basin_embedding'
BASIN_COLUMN_POST_MIGRATION = 'basin_coordinates'
ALLOWED_BASIN_COLUMNS = {BASIN_COLUMN_PRE_MIGRATION, BASIN_COLUMN_POST_MIGRATION}


def find_words_needing_backfill(limit: Optional[int] = None) -> List[Tuple[str, Optional[List[float]]]]:
    """
    Find words with empty/invalid basin embeddings.
    
    Args:
        limit: Optional limit on number of words to return
    
    Returns:
        List of (word, legacy_embedding) tuples
    """
    vp = get_vocabulary_persistence()
    
    if not vp.enabled:
        raise RuntimeError("Database connection not available")
    
    with vp._connect() as conn:
        with conn.cursor() as cur:
            # Check which column exists (migration 010 renames basin_embedding -> basin_coordinates)
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'coordizer_vocabulary' 
                      AND column_name = 'basin_coordinates'
                )
            """)
            has_basin_coordinates = cur.fetchone()[0]
            
            basin_column = BASIN_COLUMN_POST_MIGRATION if has_basin_coordinates else BASIN_COLUMN_PRE_MIGRATION
            
            # Validate column name against whitelist (prevent SQL injection)
            if basin_column not in ALLOWED_BASIN_COLUMNS:
                raise RuntimeError(f"Invalid basin column name: {basin_column}")
            
            # Find words with empty or NULL basins
            # Note: 'embedding' column may not exist after migration 010
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'coordizer_vocabulary' 
                      AND column_name = 'embedding'
                )
            """)
            has_embedding = cur.fetchone()[0]
            
            # Build query with validated column name (safe from SQL injection)
            # basin_column is validated against ALLOWED_BASIN_COLUMNS whitelist above
            if has_embedding:
                if limit:
                    cur.execute(f"""
                        SELECT token, embedding
                        FROM coordizer_vocabulary 
                        WHERE array_length({basin_column}, 1) = 0 
                           OR array_length({basin_column}, 1) IS NULL
                           OR {basin_column} IS NULL
                        ORDER BY token
                        LIMIT %s
                    """, (limit,))
                else:
                    cur.execute(f"""
                        SELECT token, embedding
                        FROM coordizer_vocabulary 
                        WHERE array_length({basin_column}, 1) = 0 
                           OR array_length({basin_column}, 1) IS NULL
                           OR {basin_column} IS NULL
                        ORDER BY token
                    """)
            else:
                if limit:
                    cur.execute(f"""
                        SELECT token, NULL::float8[]
                        FROM coordizer_vocabulary 
                        WHERE array_length({basin_column}, 1) = 0 
                           OR array_length({basin_column}, 1) IS NULL
                           OR {basin_column} IS NULL
                        ORDER BY token
                        LIMIT %s
                    """, (limit,))
                else:
                    cur.execute(f"""
                        SELECT token, NULL::float8[]
                        FROM coordizer_vocabulary 
                        WHERE array_length({basin_column}, 1) = 0 
                           OR array_length({basin_column}, 1) IS NULL
                           OR {basin_column} IS NULL
                        ORDER BY token
                    """)
            
            results = cur.fetchall()
    
    return results


def backfill_basins(dry_run: bool = True, limit: Optional[int] = None) -> Tuple[int, List[Tuple[str, str]]]:
    """
    Regenerate basin embeddings for NULL/invalid entries.
    
    Args:
        dry_run: If True, only show what would be backfilled
        limit: Optional limit on number of words to backfill
    
    Returns:
        Tuple of (success_count, failed_words)
    """
    print("\n" + "="*80)
    print("Basin Embedding Backfill Script")
    print("="*80)
    
    # Find words needing backfill
    print("\n[1/3] Finding words needing basin backfill...")
    results = find_words_needing_backfill(limit)
    
    print(f"Found {len(results)} words needing basin backfill")
    
    if not results:
        print("\n✓ No words need backfill - all basins are valid!")
        return 0, []
    
    if dry_run:
        print("\n🔍 DRY RUN MODE - showing first 10 words:")
        print("-" * 80)
        for i, (word, _) in enumerate(results[:10]):
            print(f"  {i+1:3d}. {word}")
        
        if len(results) > 10:
            print(f"  ... and {len(results) - 10} more")
        
        print("\n" + "="*80)
        print("Run with --execute to perform backfill")
        print("="*80)
        return 0, []
    
    # Execute backfill
    print("\n[2/3] Backfilling basins via QIG pipeline...")
    print("-" * 80)
    
    service = get_ingestion_service()
    success = 0
    failed = []
    
    # Use tqdm if available
    iterator = tqdm(results, desc="Backfilling") if TQDM_AVAILABLE else results
    
    for word, legacy_embedding in iterator:
        try:
            # Regenerate basin using QIG pipeline
            result = service.ingest_word(
                word=word,
                context=None,  # No context available from database
                force_recompute=True,
                source='backfill_script'
            )
            success += 1
            
            if not TQDM_AVAILABLE and success % 100 == 0:
                print(f"  Progress: {success}/{len(results)} words backfilled")
            
        except Exception as e:
            failed.append((word, str(e)))
            if not TQDM_AVAILABLE:
                print(f"\n❌ Failed: {word} - {e}")
    
    # Report results
    print("\n[3/3] Backfill Summary")
    print("="*80)
    print(f"✅ Success: {success}/{len(results)} words")
    
    if failed:
        print(f"❌ Failed: {len(failed)} words")
        print("\nFailed words (first 10):")
        for word, error in failed[:10]:
            print(f"  - {word}: {error[:80]}")
        
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more failures")
    
    print("="*80)
    
    return success, failed


def verify_backfill():
    """
    Verify all basins are populated after backfill.
    
    Returns stats on basin validity.
    """
    print("\n" + "="*80)
    print("Verifying Backfill Completion")
    print("="*80)
    
    vp = get_vocabulary_persistence()
    
    with vp._connect() as conn:
        with conn.cursor() as cur:
            # Check which column exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'coordizer_vocabulary' 
                      AND column_name = 'basin_coordinates'
                )
            """)
            has_basin_coordinates = cur.fetchone()[0]
            
            basin_column = BASIN_COLUMN_POST_MIGRATION if has_basin_coordinates else BASIN_COLUMN_PRE_MIGRATION
            
            # Validate column name against whitelist (prevent SQL injection)
            if basin_column not in ALLOWED_BASIN_COLUMNS:
                raise RuntimeError(f"Invalid basin column name: {basin_column}")
            
            # Build query with validated column name (safe from SQL injection)
            cur.execute(f"""
                SELECT 
                    COUNT(*) FILTER (WHERE array_length({basin_column}, 1) = 64) as valid_basins,
                    COUNT(*) FILTER (WHERE array_length({basin_column}, 1) = 0 OR array_length({basin_column}, 1) IS NULL) as empty_basins,
                    COUNT(*) FILTER (WHERE {basin_column} IS NULL) as null_basins,
                    COUNT(*) as total
                FROM coordizer_vocabulary
            """)
            row = cur.fetchone()
            
            valid, empty, null, total = row
    
    print(f"\nVocabulary Statistics:")
    print(f"  Total entries: {total}")
    print(f"  Valid basins (64D): {valid} ({100*valid/total if total > 0 else 0:.1f}%)")
    print(f"  Empty basins: {empty} ({100*empty/total if total > 0 else 0:.1f}%)")
    print(f"  NULL basins: {null} ({100*null/total if total > 0 else 0:.1f}%)")
    
    if empty + null == 0:
        print("\n✓ All basins are valid!")
        print("Ready to proceed with migration 010 (remove legacy basin column)")
    else:
        print(f"\n⚠ WARNING: {empty + null} basins still need backfill")
        print("Run backfill script again or investigate failed words")
    
    print("="*80)
    
    return valid, empty + null


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Backfill missing basin coordinates in coordizer_vocabulary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (show first 100 words needing backfill)
  python backfill_basin_embeddings.py --limit 100

  # Execute backfill for all words
  python backfill_basin_embeddings.py --execute

  # Execute backfill for first 1000 words
  python backfill_basin_embeddings.py --execute --limit 1000

  # Verify backfill completion
  python backfill_basin_embeddings.py --verify
        """
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually perform backfill (default: dry run)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of words to backfill'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify backfill completion (check stats)'
    )
    
    args = parser.parse_args()
    
    # Check services available
    if not SERVICES_AVAILABLE:
        print("[ERROR] Required services not available")
        sys.exit(1)
    
    # Verify mode
    if args.verify:
        valid, invalid = verify_backfill()
        sys.exit(0 if invalid == 0 else 1)
    
    # Backfill mode
    success, failed = backfill_basins(dry_run=not args.execute, limit=args.limit)
    
    if args.execute and failed:
        sys.exit(1)  # Exit with error if any failed
    
    sys.exit(0)


if __name__ == '__main__':
    main()
