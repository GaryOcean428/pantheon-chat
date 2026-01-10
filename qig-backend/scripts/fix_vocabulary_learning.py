#!/usr/bin/env python3
"""Fix vocabulary_learning table data issues.

Issues addressed:
1. related_words/related_word contains the word itself (should only have other words)
2. metadata column is empty (should have learning context)

Auto-detects schema columns - works with any table structure.
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
    """Generate metadata for a vocabulary_learning row."""
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'fix_applied': 'vocabulary_learning_null_fix',
        'word_length': len(row.get('word', '') or ''),
    }

    # Add relationship info if available
    if row.get('relationship_type') or row.get('relationship_strength'):
        metadata['relationship'] = {
            'type': row.get('relationship_type') or 'unknown',
            'strength': round(float(row.get('relationship_strength') or 0.0), 4),
        }

    # Add source info if available
    if row.get('learned_from') or row.get('source_kernel'):
        metadata['source'] = {
            'learned_from': row.get('learned_from') or 'unknown',
            'kernel': row.get('source_kernel') or 'unknown',
        }

    # Handle related words (array or single value)
    related = row.get('related_words') or row.get('related_word') or []
    if isinstance(related, str):
        related = [related] if related else []
    context = row.get('context_words') or []
    if isinstance(context, str):
        context = [context] if context else []

    metadata['stats'] = {
        'related_count': len(related) if related else 0,
        'context_count': len(context) if context else 0,
    }

    # Add temporal info if available
    if row.get('learned_at'):
        try:
            metadata['learned_timestamp'] = row['learned_at'].isoformat() if hasattr(row['learned_at'], 'isoformat') else str(row['learned_at'])
        except:
            pass

    return metadata


def fix_vocabulary_learning(limit: int = 0, dry_run: bool = False):
    """Fix vocabulary_learning table issues."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        return

    conn = psycopg2.connect(database_url)

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

        # Get actual columns
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'vocabulary_learning'
            ORDER BY ordinal_position
        """)
        columns = [r['column_name'] for r in cur.fetchall()]
        print(f"Table columns: {columns}")

        # Find primary key (id or learning_id)
        pk_col = None
        for candidate in ['id', 'learning_id']:
            if candidate in columns:
                pk_col = candidate
                break
        if not pk_col:
            pk_col = columns[0]
        print(f"Primary key: {pk_col}")

        # Find related words column
        related_col = None
        for candidate in ['related_words', 'related_word']:
            if candidate in columns:
                related_col = candidate
                break
        print(f"Related words column: {related_col}")

        has_metadata = 'metadata' in columns
        print(f"Has metadata column: {has_metadata}")

        if not related_col and not has_metadata:
            print("No fixable columns found")
            conn.close()
            return

        # Build SELECT
        select_cols = [pk_col]
        if 'word' in columns:
            select_cols.append('word')
        for col in ['relationship_type', 'relationship_strength', 'learned_from',
                    'source_kernel', 'learned_at', 'context_words']:
            if col in columns:
                select_cols.append(col)
        if related_col:
            select_cols.append(related_col)
        if has_metadata:
            select_cols.append('metadata')

        # Build WHERE - find rows that need fixing
        where_parts = []
        if has_metadata:
            where_parts.append("(metadata IS NULL OR metadata = '{}'::jsonb)")
        if related_col and 'word' in columns:
            # Check if word equals related_word (for single value) or is in array
            cur.execute(f"SELECT data_type FROM information_schema.columns WHERE table_name = 'vocabulary_learning' AND column_name = '{related_col}'")
            dtype = cur.fetchone()
            if dtype and 'ARRAY' in str(dtype['data_type']).upper():
                where_parts.append(f"({related_col} IS NOT NULL AND word = ANY({related_col}))")
            else:
                where_parts.append(f"({related_col} IS NOT NULL AND word = {related_col})")

        if not where_parts:
            print("No conditions to filter on")
            conn.close()
            return

        query = f"""
            SELECT {', '.join(select_cols)}
            FROM vocabulary_learning
            WHERE {' OR '.join(where_parts)}
            ORDER BY {pk_col}
        """
        if limit > 0:
            query += f" LIMIT {limit}"

        print(f"\nQuery:\n{query}\n")
        cur.execute(query)
        rows = cur.fetchall()

    if not rows:
        print("No rows found needing fixes")
        conn.close()
        return

    print(f"Found {len(rows)} rows needing fixes")

    if dry_run:
        print("\nDRY RUN - showing first 10 rows:")
        for row in rows[:10]:
            related = row.get(related_col) if related_col else None
            word = row.get('word', '')
            if isinstance(related, list):
                has_self = word in related if related else False
            else:
                has_self = word == related if related else False
            print(f"  {pk_col}={row[pk_col]}: word='{word}', related='{related}', self_ref={has_self}")
        conn.close()
        return

    success = 0
    errors = 0

    with conn.cursor() as cur:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i+BATCH_SIZE]

            for row in batch:
                try:
                    row_dict = dict(row)
                    updates = []
                    params = []

                    # Fix related words - remove self-reference
                    if related_col and row.get(related_col):
                        related = row[related_col]
                        word = row.get('word', '')
                        if isinstance(related, list):
                            fixed = [w for w in related if w != word]
                            updates.append(f"{related_col} = %s")
                            params.append(fixed if fixed else None)
                            row_dict[related_col] = fixed
                        elif related == word:
                            updates.append(f"{related_col} = NULL")
                            row_dict[related_col] = None

                    # Generate and set metadata
                    if has_metadata:
                        metadata = generate_metadata(row_dict)
                        updates.append("metadata = %s::jsonb")
                        params.append(json.dumps(metadata))

                    if updates:
                        params.append(row[pk_col])
                        cur.execute(f"""
                            UPDATE vocabulary_learning
                            SET {', '.join(updates)}
                            WHERE {pk_col} = %s
                        """, params)
                        success += 1

                except Exception as e:
                    print(f"Error fixing {pk_col}={row[pk_col]}: {e}")
                    errors += 1

            conn.commit()
            print(f"Progress: {min(i + BATCH_SIZE, len(rows))}/{len(rows)} (success={success}, errors={errors})")

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
