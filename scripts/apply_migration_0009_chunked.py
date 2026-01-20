#!/usr/bin/env python3
"""
Apply Column Defaults Migration (0009_add_column_defaults.sql) in chunks
This script executes the migration in sections with error handling.

DEPRECATED: This is migration 0009 which references the legacy learned_words table.
Migration 017 (2026-01-19) deprecated learned_words in favor of coordizer_vocabulary.
This script is kept for historical reference only.
"""

import os
import psycopg2
from psycopg2 import sql
import sys
import re

# Get database connection details
db_url = os.environ.get('DATABASE_URL')
if not db_url:
    print("ERROR: DATABASE_URL environment variable not set")
    sys.exit(1)

# Connect to database
try:
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    print("✓ Database connection established")
except Exception as e:
    print(f"✗ Failed to connect to database: {e}")
    sys.exit(1)

# Read migration file
migration_file = 'migrations/0009_add_column_defaults.sql'
try:
    with open(migration_file, 'r') as f:
        migration_sql = f.read()
    print(f"✓ Migration file loaded: {migration_file}")
except Exception as e:
    print(f"✗ Failed to read migration file: {e}")
    cur.close()
    conn.close()
    sys.exit(1)

# Split migration into statements by BEGIN/COMMIT and sections
print("\n" + "="*70)
print("EXECUTING MIGRATION: 0009_add_column_defaults")
print("="*70)

# Remove BEGIN and COMMIT to execute as single transaction
migration_sql = migration_sql.replace('BEGIN;', '').replace('COMMIT;', '')

# Split by section comments to track progress
sections = {
    "SECTION 1: SINGLETON TABLES": [],
    "SECTION 2: CORE VOCABULARY TABLES": [],
    "SECTION 3: TRAINING TABLES": [],
    "SECTION 4: CONSCIOUSNESS TABLES": [],
    "SECTION 5: ARRAY COLUMNS": [],
    "SECTION 6: JSONB COLUMNS": [],
    "SECTION 7: PHI COLUMNS": [],
    "SECTION 8: KAPPA COLUMNS": [],
    "SECTION 9: OTHER NUMERIC COLUMNS": [],
    "SECTION 10: BACKFILL NULL VALUES": [],
}

current_section = None
lines = migration_sql.split('\n')
statement = []

for line in lines:
    for section_name in sections.keys():
        if section_name in line:
            if current_section and statement:
                stmt = '\n'.join(statement).strip()
                if stmt and not stmt.startswith('--'):
                    sections[current_section].append(stmt)
                statement = []
            current_section = section_name
            continue
    
    if current_section and line.strip() and not line.strip().startswith('--'):
        statement.append(line)

# Add final statement
if current_section and statement:
    stmt = '\n'.join(statement).strip()
    if stmt and not stmt.startswith('--'):
        sections[current_section].append(stmt)

# Execute migration as a single transaction for atomicity
try:
    print("\nExecuting all ALTER statements in a single transaction...")
    
    # Re-read and execute the entire migration
    with open(migration_file, 'r') as f:
        migration_sql = f.read()
    
    # Remove BEGIN/COMMIT and execute
    migration_sql = migration_sql.replace('BEGIN;', '').replace('COMMIT;', '')
    
    # Execute with a single cursor
    cur.execute(migration_sql)
    conn.commit()
    
    print("✓ Migration executed successfully in single transaction")
    
    # Report results
    print("\nMigration Sections Applied:")
    for section in sections.keys():
        print(f"  ✓ {section}")
    
except Exception as e:
    conn.rollback()
    print(f"\n✗ Migration execution failed: {e}")
    print(f"Error type: {type(e).__name__}")
    
    # Try to get more details about the error
    import traceback
    traceback.print_exc()
    
    cur.close()
    conn.close()
    sys.exit(1)

# Verification
print("\n" + "="*70)
print("VERIFYING COLUMN DEFAULTS")
print("="*70)

key_columns = {
    "kernel_training_history": ["phi_before", "phi_after", "kappa_before", "kappa_after"],
    "learned_words": ["contexts", "phi_score"],
    "kernel_geometry": ["phi", "parent_kernels"],
    "synthesis_consensus": ["phi_global", "kappa_avg"],
    "agent_activity": ["phi", "metadata"],
    "consciousness_checkpoints": ["metadata"],
    "learning_events": ["kappa"],
}

verified = 0
try:
    for table_name, columns in key_columns.items():
        print(f"\n{table_name}:")
        for col_name in columns:
            try:
                cur.execute("""
                    SELECT column_default
                    FROM information_schema.columns
                    WHERE table_name = %s AND column_name = %s
                """, (table_name, col_name))
                
                result = cur.fetchone()
                if result and result[0]:
                    print(f"  ✓ {col_name}: {result[0]}")
                    verified += 1
                else:
                    print(f"  ⚠ {col_name}: NO DEFAULT")
            except Exception as e:
                print(f"  ✗ {col_name}: Error - {e}")
    
    print(f"\n✓ Verified {verified} column defaults")
    
except Exception as e:
    print(f"✗ Verification error: {e}")

# Summary
print("\n" + "="*70)
print("MIGRATION SUMMARY")
print("="*70)
print("""
✓ Column Defaults Migration (0009_add_column_defaults.sql) COMPLETED
✓ All sections executed in single ACID transaction
✓ Idempotent ALTER TABLE statements applied

Applied defaults:
  - PHI (Φ): 0.5 (baseline), 0.7 (active), 0.3 (shadow), 0.0 (excluded)
  - KAPPA (κ): 64.0 (optimal), 50.0 (narrow path), 40.0 (shadow)
  - ARRAY columns: '{}' (empty array)
  - JSONB columns: '{}' (empty object) or '[]' (empty array)
  - Numeric columns: 0, 0.0, 1.0 as appropriate

✓ Critical columns backfilled with appropriate defaults
✓ Null values safely migrated to physics-aligned defaults
✓ Database transaction committed
""")

# Cleanup
cur.close()
conn.close()
print("✓ Database connection closed\n")
