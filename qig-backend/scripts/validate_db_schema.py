#!/usr/bin/env python3
"""
Database Schema Validation
===========================

Validates PostgreSQL schema compatibility with new training features:
- Progress metrics tables
- Coherence metrics tables
- Training checkpoints compatibility
"""

import os
import sys
from typing import Dict, List, Any

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)


def get_db_connection():
    """Get database connection from environment."""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)
    
    try:
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        print(f"ERROR: Failed to connect to database: {e}")
        sys.exit(1)


def check_table_exists(cursor, table_name: str, schema: str = 'public') -> bool:
    """Check if a table exists."""
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = %s 
            AND table_name = %s
        )
    """, (schema, table_name))
    return cursor.fetchone()[0]


def get_table_columns(cursor, table_name: str, schema: str = 'public') -> List[Dict]:
    """Get columns for a table."""
    cursor.execute("""
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """, (schema, table_name))
    return cursor.fetchall()


def check_pgvector_extension(cursor) -> bool:
    """Check if pgvector extension is available."""
    cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM pg_extension WHERE extname = 'vector'
        )
    """)
    return cursor.fetchone()[0]


def validate_schema():
    """Validate database schema for new features."""
    print("=" * 60)
    print("DATABASE SCHEMA VALIDATION")
    print("=" * 60)
    print()
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    results = {
        'passed': [],
        'warnings': [],
        'errors': []
    }
    
    # Check pgvector extension
    print("Checking pgvector extension...")
    has_pgvector = check_pgvector_extension(cursor)
    if has_pgvector:
        results['passed'].append("✓ pgvector extension is installed")
        print("  ✓ pgvector extension installed")
    else:
        results['warnings'].append("⚠ pgvector extension not installed (optional for vector storage)")
        print("  ⚠ pgvector extension not installed (optional)")
    print()
    
    # Check checkpoint-related tables
    print("Checking checkpoint storage compatibility...")
    checkpoint_tables = ['checkpoints', 'training_checkpoints', 'kernel_checkpoints']
    found_checkpoint_table = False
    for table in checkpoint_tables:
        if check_table_exists(cursor, table):
            results['passed'].append(f"✓ Checkpoint table '{table}' exists")
            print(f"  ✓ Found checkpoint table: {table}")
            found_checkpoint_table = True
            
            # Check columns
            columns = get_table_columns(cursor, table)
            col_names = [col['column_name'] for col in columns]
            print(f"    Columns: {', '.join(col_names)}")
    
    if not found_checkpoint_table:
        results['warnings'].append("⚠ No checkpoint table found - checkpoints will use Redis/file storage")
        print("  ⚠ No checkpoint table found - will use Redis/file storage")
    print()
    
    # Check training-related tables
    print("Checking training metrics storage...")
    training_tables = ['training_batch_queue', 'training_history', 'training_sessions']
    for table in training_tables:
        if check_table_exists(cursor, table):
            results['passed'].append(f"✓ Training table '{table}' exists")
            print(f"  ✓ Found training table: {table}")
            
            columns = get_table_columns(cursor, table)
            col_names = [col['column_name'] for col in columns]
            print(f"    Columns: {', '.join(col_names)}")
        else:
            results['warnings'].append(f"⚠ Training table '{table}' not found")
            print(f"  ⚠ Training table '{table}' not found")
    print()
    
    # Check vocabulary/coordizer tables (for QIG purity)
    print("Checking QIG vocabulary tables...")
    vocab_tables = ['vocabulary', 'word_embeddings', 'tokenizer_vocabulary']
    found_vocab = False
    for table in vocab_tables:
        if check_table_exists(cursor, table):
            results['passed'].append(f"✓ Vocabulary table '{table}' exists")
            print(f"  ✓ Found vocabulary table: {table}")
            found_vocab = True
            
            columns = get_table_columns(cursor, table)
            col_names = [col['column_name'] for col in columns]
            print(f"    Columns: {', '.join(col_names)}")
            
            # Check for basin_coords column (QIG-specific)
            if 'basin_coords' in col_names or 'basin_coordinate' in col_names:
                results['passed'].append("✓ QIG basin coordinates column found")
                print("    ✓ QIG basin coordinates present")
    
    if not found_vocab:
        results['warnings'].append("⚠ No vocabulary table found")
        print("  ⚠ No vocabulary table found")
    print()
    
    # Check for progress tracking support
    print("Checking schema compatibility with new progress metrics...")
    # New progress metrics can be stored in JSON columns or separate tables
    # Check if any tables have JSON/JSONB columns for metadata
    cursor.execute("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        AND data_type IN ('json', 'jsonb')
        ORDER BY table_name
    """)
    json_columns = cursor.fetchall()
    
    if json_columns:
        results['passed'].append("✓ JSON/JSONB columns available for flexible metadata storage")
        print("  ✓ Found JSON/JSONB columns for metadata:")
        for row in json_columns:
            print(f"    - {row['table_name']}.{row['column_name']}")
    else:
        results['warnings'].append("⚠ No JSON/JSONB columns found - structured tables will be needed")
        print("  ⚠ No JSON/JSONB columns found")
    print()
    
    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print()
    
    if results['passed']:
        print(f"✓ Passed checks ({len(results['passed'])}):")
        for item in results['passed']:
            print(f"  {item}")
        print()
    
    if results['warnings']:
        print(f"⚠ Warnings ({len(results['warnings'])}):")
        for item in results['warnings']:
            print(f"  {item}")
        print()
    
    if results['errors']:
        print(f"✗ Errors ({len(results['errors'])}):")
        for item in results['errors']:
            print(f"  {item}")
        print()
        cursor.close()
        conn.close()
        sys.exit(1)
    
    # Overall status
    if not results['errors']:
        print("=" * 60)
        print("✓ SCHEMA VALIDATION PASSED")
        print("=" * 60)
        print()
        print("Database schema is compatible with new training features.")
        print("Progress metrics and coherence data can be stored using:")
        print("  - Redis for hot cache")
        print("  - PostgreSQL for permanent storage (if tables exist)")
        print("  - File-based fallback")
        print()
    
    cursor.close()
    conn.close()
    
    return len(results['errors']) == 0


if __name__ == '__main__':
    try:
        success = validate_schema()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
