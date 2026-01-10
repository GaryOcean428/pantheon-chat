#!/usr/bin/env python3
"""
Backup missing tables before schema migration to prevent data loss
Creates JSON backups in migrations/backups/
"""
import json
import os
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError('DATABASE_URL not set')

BACKUP_DIR = Path(__file__).parent.parent / 'migrations' / 'backups'
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def backup_table(cursor, table_name):
    """Backup a single table to JSON."""
    print(f"Backing up {table_name}...")

    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    # Convert to JSON-serializable format
    rows_json = []
    for row in rows:
        row_dict = dict(row)
        # Convert datetime objects to ISO format
        for key, value in row_dict.items():
            if hasattr(value, 'isoformat'):
                row_dict[key] = value.isoformat()
        rows_json.append(row_dict)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = BACKUP_DIR / f'{table_name}_{timestamp}.json'

    with open(backup_file, 'w') as f:
        json.dump({
            'table': table_name,
            'timestamp': timestamp,
            'row_count': len(rows_json),
            'data': rows_json
        }, f, indent=2)

    print(f"  ✓ Backed up {len(rows_json)} rows to {backup_file}")
    return len(rows_json)

def main():
    tables_to_backup = [
        'qig_metadata',           # 1 row
        'governance_proposals',    # 955 rows
        'tool_requests',          # 14 rows
        'pattern_discoveries',     # 14 rows
        'vocabulary_stats',        # 15,298 rows
        'federation_peers',        # 1 row
        'passphrase_vocabulary'    # 2,073 rows
    ]

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    total_rows = 0
    try:
        print(f"\nCreating backups in: {BACKUP_DIR}")
        print("=" * 60)

        for table in tables_to_backup:
            try:
                count = backup_table(cursor, table)
                total_rows += count
            except Exception as e:
                print(f"  ✗ Error backing up {table}: {e}")

        print("=" * 60)
        print(f"\n✓ Backup complete: {total_rows} total rows backed up")
        print(f"Backups saved to: {BACKUP_DIR}")

    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    main()
