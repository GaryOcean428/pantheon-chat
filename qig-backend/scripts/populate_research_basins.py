#!/usr/bin/env python3
"""Populate research request basin coordinates using coordizer.

Phase 2.3 of NULL column population plan.
Reference: docs/03-technical/20260110-null-column-population-plan-1.00W.md

Uses the PostgresCoordizer to generate 64D basin coordinates for each
research topic in research_requests.
"""

import os
import sys
from pathlib import Path

# Add qig-backend to path for imports
_qig_backend = Path(__file__).parent.parent
if str(_qig_backend) not in sys.path:
    sys.path.insert(0, str(_qig_backend))

BATCH_SIZE = 100


def populate_research_basins(limit: int = 0, dry_run: bool = False):
    """Populate research_requests.basin_coords.

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

    # Get research requests needing basin coords
    query = """
        SELECT request_id, topic FROM research_requests
        WHERE basin_coords IS NULL AND topic IS NOT NULL
        ORDER BY created_at DESC
    """
    if limit > 0:
        query += f" LIMIT {limit}"

    cur.execute(query)
    requests = cur.fetchall()

    if not requests:
        print("No research requests found needing basin coordinates")
        cur.close()
        conn.close()
        return

    print(f"Found {len(requests)} research requests needing basin coordinates")

    if dry_run:
        print("\nDRY RUN - showing first 10 topics:")
        for request_id, topic in requests[:10]:
            try:
                # Truncate long topics for display
                display_topic = topic[:50] + "..." if len(topic) > 50 else topic
                basin = coordizer.encode(topic)
                print(f"  [{request_id}] '{display_topic}': norm={basin.sum():.4f}")
            except Exception as e:
                print(f"  [{request_id}] ERROR - {e}")
        return

    success = 0
    errors = 0

    for i in range(0, len(requests), BATCH_SIZE):
        batch = requests[i:i+BATCH_SIZE]

        for request_id, topic in batch:
            try:
                basin = coordizer.encode(topic)
                if basin is not None and len(basin) == 64:
                    cur.execute(
                        """UPDATE research_requests
                           SET basin_coords = %s
                           WHERE request_id = %s""",
                        (basin.tolist(), request_id)
                    )
                    success += 1
                else:
                    errors += 1
            except Exception as e:
                print(f"Error coordizing topic '{topic[:50]}...': {e}")
                errors += 1

        conn.commit()
        processed = min(i + BATCH_SIZE, len(requests))
        print(f"Progress: {processed}/{len(requests)} (success={success}, errors={errors})")

    print(f"\nCompleted: {success} populated, {errors} errors")

    cur.close()
    conn.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Populate research request basin coordinates')
    parser.add_argument('--limit', type=int, default=0,
                        help='Maximum rows to process (0 = all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be updated without making changes')
    args = parser.parse_args()

    populate_research_basins(limit=args.limit, dry_run=args.dry_run)
