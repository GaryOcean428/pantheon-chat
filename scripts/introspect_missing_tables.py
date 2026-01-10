#!/usr/bin/env python3
"""
Introspect missing tables from Neon database to reconstruct schema definitions
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError('DATABASE_URL not set')

def introspect_table(cursor, table_name):
    print(f"\n=== {table_name} ===")
    
    # Get column definitions
    cursor.execute("""
        SELECT 
            column_name,
            data_type,
            character_maximum_length,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position;
    """, (table_name,))
    
    columns = cursor.fetchall()
    if not columns:
        print(f"  Table {table_name} not found")
        return
    
    print("Columns:")
    for col in columns:
        nullable = '' if col['is_nullable'] == 'YES' else 'NOT NULL'
        length = f"({col['character_maximum_length']})" if col['character_maximum_length'] else ''
        default = f"DEFAULT {col['column_default']}" if col['column_default'] else ''
        print(f"  {col['column_name']}: {col['data_type']}{length} {nullable} {default}".strip())
    
    # Get indexes
    cursor.execute("""
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = 'public' AND tablename = %s;
    """, (table_name,))
    
    indexes = cursor.fetchall()
    if indexes:
        print("Indexes:")
        for idx in indexes:
            print(f"  {idx['indexname']}: {idx['indexdef']}")
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
    count = cursor.fetchone()
    print(f"Row count: {count['count']}")

def main():
    missing_tables = [
        'qig_metadata',
        'governance_proposals',
        'tool_requests',
        'pattern_discoveries',
        'vocabulary_stats',
        'federation_peers',
        'passphrase_vocabulary'
    ]
    
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        for table in missing_tables:
            try:
                introspect_table(cursor, table)
            except Exception as e:
                print(f"Error introspecting {table}: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    main()
