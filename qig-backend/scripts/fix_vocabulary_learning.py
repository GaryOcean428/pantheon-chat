#!/usr/bin/env python3
"""Fix vocabulary_learning table data issues.

Issues addressed:
1. related_words array contains the word itself (should only have other words)
2. metadata column is empty (should have learning context)

Reference: User report - "word is same as related_word, metadata is empty"
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add qig-backend to path for imports
_qig_backend = Path(__file__).parent.parent
if str(_qig_backend) not in sys.path:
    sys.path.insert(0, str(_qig_backend))

BATCH_SIZE = 100


def generate_metadata(row: dict) -> dict:
    """Generate metadata for a vocabulary_learning row.

    Args:
        row: Dict with learning_id, word, relationship_type, relationship_strength,
             learned_from, source_kernel, learned_at, related_words, context_words
    """
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'fix_applied': 'vocabulary_learning_null_fix',
        'word_length': len(row.get('word', '')),
        'relationship': {
            'type': row.get('relationship_type', 'unknown'),
            'strength': round(row.get('relationship_strength', 0.0), 4),
        },
        'source': {
            'learned_from': row.get('learned_from', 'unknown'),
            'kernel': row.get('source_kernel', 'unknown'),
        },
        'stats': {
            'related_count': len(row.get('related_words', []) or []),
            'context_count': len(row.get('context_words', []) or []),
        }
    }

    # Add temporal info if learned_at exists
    if row.get('learned_at'):
        metadata['learned_timestamp'] = row['learned_at'].isoformat() if hasattr(row['learned_at'], 'isoformat') else str(row['learned_at'])

    return metadata


def fix_vocabulary_learning(limit: int = 0, dry_run: bool = False):
    """Fix vocabulary_learning table issues.

    Args:
        limit: Maximum rows to process (0 = all)
        dry_run: If True, show what would be updated without making changes
    """
    import psycopg2
    from psycopg2.extras import RealDictCursor

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        return

    conn = psycopg2.connect(database_url)

    # Get rows needing fixes
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'vocabulary_learning'
            )
        """)
        if not cur.fetchone()['exists']:
            print("vocabulary_learning table does not exist")
            conn.close()
            return

        # Get rows where:
        # 1. metadata is NULL or empty
        # 2. OR word is in related_words array
        query = """
            SELECT learning_id, word, relationship_type, relationship_strength,
                   learned_from, source_kernel, learned_at, related_words, context_words,
                   metadata
            FROM vocabulary_learning
            WHERE metadata IS NULL
               OR metadata = '{}'::jsonb
               OR (related_words IS NOT NULL AND word = ANY(related_words))
            ORDER BY learning_id
        """
        if limit > 0:
            query += f" LIMIT {limit}"

        cur.execute(query)
        rows = cur.fetchall()

    if not rows:
        print("No vocabulary_learning rows found needing fixes")
        conn.close()
        return

    print(f"Found {len(rows)} rows needing fixes")

    # Count specific issues
    self_reference_count = sum(1 for r in rows if r.get('related_words') and r['word'] in r['related_words'])
    empty_metadata_count = sum(1 for r in rows if not r.get('metadata') or r.get('metadata') == {})
    print(f"  - Self-reference in related_words: {self_reference_count}")
    print(f"  - Empty/NULL metadata: {empty_metadata_count}")

    if dry_run:
        print("\nDRY RUN - showing first 10 rows:")
        for row in rows[:10]:
            related = row.get('related_words') or []
            has_self = row['word'] in related if related else False
            print(f"  id={row['learning_id']}: word='{row['word']}', "
                  f"self_ref={has_self}, related_count={len(related)}")
        return

    success = 0
    errors = 0

    with conn.cursor() as cur:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i+BATCH_SIZE]

            for row in batch:
                try:
                    # Fix related_words - remove self-reference
                    related_words = row.get('related_words') or []
                    if row['word'] in related_words:
                        related_words = [w for w in related_words if w != row['word']]

                    # Generate metadata
                    row_dict = dict(row)
                    row_dict['related_words'] = related_words  # Use fixed list
                    metadata = generate_metadata(row_dict)
                    metadata_json = json.dumps(metadata)

                    # Update row
                    cur.execute("""
                        UPDATE vocabulary_learning
                        SET related_words = %s,
                            metadata = %s::jsonb
                        WHERE learning_id = %s
                    """, (related_words if related_words else None, metadata_json, row['learning_id']))

                    success += 1
                except Exception as e:
                    print(f"Error fixing learning_id={row['learning_id']}: {e}")
                    errors += 1

            conn.commit()
            processed = min(i + BATCH_SIZE, len(rows))
            print(f"Progress: {processed}/{len(rows)} (success={success}, errors={errors})")

    # Verify results
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(metadata) FILTER (WHERE metadata IS NOT NULL AND metadata != '{}') as has_metadata,
                COUNT(*) FILTER (WHERE related_words IS NOT NULL AND word = ANY(related_words)) as has_self_ref
            FROM vocabulary_learning
        """)
        stats = cur.fetchone()
        print(f"\nFinal stats:")
        print(f"  Total rows: {stats[0]}")
        print(f"  With metadata: {stats[1]}")
        print(f"  With self-reference (should be 0): {stats[2]}")

    print(f"\nCompleted: {success} fixed, {errors} errors")

    conn.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fix vocabulary_learning table issues')
    parser.add_argument('--limit', type=int, default=0,
                        help='Maximum rows to process (0 = all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be updated without making changes')
    args = parser.parse_args()

    fix_vocabulary_learning(limit=args.limit, dry_run=args.dry_run)
