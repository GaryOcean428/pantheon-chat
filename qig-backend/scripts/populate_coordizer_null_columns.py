#!/usr/bin/env python3
"""Populate NULL columns in coordizer_vocabulary table.

Phase 2.3 of NULL column population plan.
Reference: docs/03-technical/20260110-null-column-population-plan-1.00W.md

Columns to populate:
- embedding: Copy from basin_embedding (legacy compatibility)
- metadata: JSON with token properties
- scale: Token scale classification (char/word/phrase)
"""

import os
import sys
import json
from pathlib import Path

# Add qig-backend to path for imports
_qig_backend = Path(__file__).parent.parent
if str(_qig_backend) not in sys.path:
    sys.path.insert(0, str(_qig_backend))

BATCH_SIZE = 500


def classify_scale(token: str) -> str:
    """Classify token scale based on characteristics.

    Scale categories:
    - char: Single characters or very short tokens
    - word: Standard words (3-15 chars, single word)
    - phrase: Multi-word or compound tokens
    """
    if len(token) <= 2:
        return 'char'
    elif ' ' in token or '-' in token or len(token) > 15:
        return 'phrase'
    else:
        return 'word'


def generate_metadata(token: str, source_type: str, phi_score: float) -> dict:
    """Generate metadata JSON for a token."""
    return {
        'length': len(token),
        'is_alpha': token.isalpha(),
        'is_lowercase': token.islower(),
        'source': source_type,
        'phi': round(phi_score, 4) if phi_score else 0.5,
        'has_vowels': any(c in token.lower() for c in 'aeiou'),
        'char_classes': {
            'vowels': sum(1 for c in token.lower() if c in 'aeiou'),
            'consonants': sum(1 for c in token.lower() if c.isalpha() and c not in 'aeiou'),
        }
    }


def populate_null_columns(limit: int = 0, dry_run: bool = False):
    """Populate embedding, metadata, and scale columns where NULL.

    Args:
        limit: Maximum rows to process (0 = all)
        dry_run: If True, show what would be updated without making changes
    """
    import psycopg2

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        return

    conn = psycopg2.connect(database_url)
    cur = conn.cursor()

    # Get tokens with NULL columns
    query = """
        SELECT id, token, source_type, phi_score, basin_embedding
        FROM coordizer_vocabulary
        WHERE embedding IS NULL
           OR metadata IS NULL
           OR scale IS NULL
        ORDER BY id
    """
    if limit > 0:
        query += f" LIMIT {limit}"

    cur.execute(query)
    rows = cur.fetchall()

    if not rows:
        print("No coordizer_vocabulary rows found needing NULL column population")
        cur.close()
        conn.close()
        return

    print(f"Found {len(rows)} rows needing NULL column population")

    if dry_run:
        print("\nDRY RUN - showing first 10 rows:")
        for row in rows[:10]:
            id_, token, source_type, phi_score, basin_embedding = row
            scale = classify_scale(token)
            metadata = generate_metadata(token, source_type or 'base', phi_score or 0.5)
            print(f"  id={id_}: token='{token}', scale='{scale}', metadata keys={list(metadata.keys())}")
        return

    success = 0
    errors = 0

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]

        for row in batch:
            id_, token, source_type, phi_score, basin_embedding = row

            try:
                scale = classify_scale(token)
                metadata = generate_metadata(token, source_type or 'base', phi_score or 0.5)
                metadata_json = json.dumps(metadata)

                # Update row - copy basin_embedding to embedding for legacy compatibility
                cur.execute("""
                    UPDATE coordizer_vocabulary
                    SET embedding = basin_embedding,
                        metadata = %s::jsonb,
                        scale = %s
                    WHERE id = %s
                      AND (embedding IS NULL OR metadata IS NULL OR scale IS NULL)
                """, (metadata_json, scale, id_))

                success += 1
            except Exception as e:
                print(f"Error updating id={id_} token='{token[:30]}': {e}")
                errors += 1

        conn.commit()
        processed = min(i + BATCH_SIZE, len(rows))
        print(f"Progress: {processed}/{len(rows)} (success={success}, errors={errors})")

    # Verify results
    cur.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(embedding) as has_embedding,
            COUNT(metadata) as has_metadata,
            COUNT(scale) as has_scale
        FROM coordizer_vocabulary
    """)
    stats = cur.fetchone()
    print(f"\nFinal stats:")
    print(f"  Total rows: {stats[0]}")
    print(f"  With legacy basin column: {stats[1]}")
    print(f"  With metadata: {stats[2]}")
    print(f"  With scale: {stats[3]}")

    print(f"\nCompleted: {success} populated, {errors} errors")

    cur.close()
    conn.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Populate NULL columns in coordizer_vocabulary')
    parser.add_argument('--limit', type=int, default=0,
                        help='Maximum rows to process (0 = all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be updated without making changes')
    args = parser.parse_args()

    populate_null_columns(limit=args.limit, dry_run=args.dry_run)
