#!/usr/bin/env python3
"""Populate vocabulary observation basin coordinates using coordizer.

Phase 2.2 of NULL column population plan.
Reference: docs/03-technical/20260110-null-column-population-plan-1.00W.md

Uses the PostgresCoordizer to generate 64D basin coordinates for each
word/phrase in vocabulary_observations.
"""

import os
import sys
from pathlib import Path

# Add qig-backend to path for imports
_qig_backend = Path(__file__).parent.parent
if str(_qig_backend) not in sys.path:
    sys.path.insert(0, str(_qig_backend))

BATCH_SIZE = 100


def populate_vocab_basins(limit: int = 0, dry_run: bool = False):
    """Populate vocabulary_observations.basin_coords.

    Args:
        limit: Maximum rows to process (0 = all)
        dry_run: If True, show what would be updated without making changes
    """
    import psycopg2

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        return

    # Import coordizer after path setup
    try:
        from coordizers import get_coordizer
        coordizer = get_coordizer()
        print(f"Coordizer loaded: {len(coordizer.vocab)} tokens")
    except ImportError as e:
        print(f"ERROR: Cannot import coordizer: {e}")
        return
    except Exception as e:
        print(f"ERROR: Failed to initialize coordizer: {e}")
        return

    conn = psycopg2.connect(database_url)
    cur = conn.cursor()

    # Get words needing basin coords
    query = """
        SELECT text FROM vocabulary_observations
        WHERE basin_coords IS NULL
        ORDER BY frequency DESC NULLS LAST
    """
    if limit > 0:
        query += f" LIMIT {limit}"

    cur.execute(query)
    words = [row[0] for row in cur.fetchall()]

    if not words:
        print("No vocabulary observations found needing basin coordinates")
        cur.close()
        conn.close()
        return

    print(f"Found {len(words)} words needing basin coordinates")

    if dry_run:
        print("\nDRY RUN - showing first 10 words:")
        for word in words[:10]:
            try:
                basin = coordizer.encode(word)
                print(f"  '{word}': norm={basin.sum():.4f}, nonzero={(basin != 0).sum()}")
            except Exception as e:
                print(f"  '{word}': ERROR - {e}")
        return

    success = 0
    errors = 0

    for i in range(0, len(words), BATCH_SIZE):
        batch = words[i:i+BATCH_SIZE]

        for word in batch:
            try:
                basin = coordizer.encode(word)
                if basin is not None and len(basin) == 64:
                    cur.execute(
                        """UPDATE vocabulary_observations
                           SET basin_coords = %s
                           WHERE text = %s""",
                        (basin.tolist(), word)
                    )
                    success += 1
                else:
                    errors += 1
            except Exception as e:
                print(f"Error coordizing '{word[:30]}...': {e}")
                errors += 1

        conn.commit()
        processed = min(i + BATCH_SIZE, len(words))
        print(f"Progress: {processed}/{len(words)} (success={success}, errors={errors})")

    print(f"\nCompleted: {success} populated, {errors} errors")

    cur.close()
    conn.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Populate vocabulary basin coordinates')
    parser.add_argument('--limit', type=int, default=0,
                        help='Maximum rows to process (0 = all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be updated without making changes')
    args = parser.parse_args()

    populate_vocab_basins(limit=args.limit, dry_run=args.dry_run)
