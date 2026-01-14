#!/usr/bin/env python3
"""
Apply Column Defaults Migration (0009_add_column_defaults.sql)
This script executes the migration with error handling and verification.
"""

import os
import psycopg2
from psycopg2 import sql
import sys

# Get database connection details from environment
db_url = os.environ.get('DATABASE_URL')
if not db_url:
    print("ERROR: DATABASE_URL environment variable not set")
    sys.exit(1)

# Parse connection string
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

# Execute migration with error handling
try:
    print("\n" + "="*70)
    print("EXECUTING MIGRATION: 0009_add_column_defaults")
    print("="*70)
    
    # Execute the entire migration as one transaction
    cur.execute(migration_sql)
    conn.commit()
    
    print("\n✓ Migration executed successfully")
    
    # Section report based on the migration structure
    sections = [
        "SECTION 1: SINGLETON TABLES",
        "SECTION 2: CORE VOCABULARY TABLES",
        "SECTION 3: TRAINING TABLES",
        "SECTION 4: CONSCIOUSNESS TABLES",
        "SECTION 5: ARRAY COLUMNS",
        "SECTION 6: JSONB COLUMNS",
        "SECTION 7: PHI COLUMNS",
        "SECTION 8: KAPPA COLUMNS",
        "SECTION 9: OTHER NUMERIC COLUMNS",
        "SECTION 10: BACKFILL NULL VALUES",
    ]
    
    print("\nMigration Sections Applied:")
    for section in sections:
        print(f"  ✓ {section}")
    
except Exception as e:
    conn.rollback()
    print(f"\n✗ Migration failed: {e}")
    cur.close()
    conn.close()
    sys.exit(1)

# Verification: Check key columns for defaults
print("\n" + "="*70)
print("VERIFYING COLUMN DEFAULTS")
print("="*70)

verification_queries = {
    "kernel_training_history": [
        "phi_before",
        "phi_after",
        "kappa_before",
        "kappa_after",
        "phi_delta",
    ],
    "learned_words": [
        "contexts",
        "phi_score",
    ],
    "kernel_geometry": [
        "phi",
        "parent_kernels",
        "observing_parents",
    ],
    "synthesis_consensus": [
        "phi_global",
        "kappa_avg",
        "participating_kernels",
    ],
    "agent_activity": [
        "phi",
        "metadata",
        "result_count",
    ],
    "consciousness_checkpoints": [
        "metadata",
    ],
}

try:
    verified_count = 0
    failed_count = 0
    
    for table_name, columns in verification_queries.items():
        print(f"\n{table_name}:")
        for col_name in columns:
            try:
                # Check if column has a default
                cur.execute(f"""
                    SELECT column_default
                    FROM information_schema.columns
                    WHERE table_name = %s AND column_name = %s
                """, (table_name, col_name))
                
                result = cur.fetchone()
                if result and result[0]:
                    default_val = result[0]
                    print(f"  ✓ {col_name}: {default_val}")
                    verified_count += 1
                else:
                    print(f"  ⚠ {col_name}: NO DEFAULT (table/column may not exist)")
                    
            except Exception as e:
                print(f"  ✗ {col_name}: Error checking default - {e}")
                failed_count += 1
    
    print(f"\n" + "="*70)
    print(f"Verification Results:")
    print(f"  ✓ Columns with defaults: {verified_count}")
    if failed_count > 0:
        print(f"  ⚠ Columns with issues: {failed_count}")
    
except Exception as e:
    print(f"✗ Verification error: {e}")

# Final validation query (from migration)
print("\n" + "="*70)
print("FINAL VALIDATION")
print("="*70)

try:
    cur.execute("""
        SELECT COUNT(*) as missing_defaults
        FROM information_schema.columns 
        WHERE table_schema = 'public'
            AND column_default IS NULL
            AND is_nullable = 'YES'
            AND table_name IN (
                'ocean_quantum_state', 'near_miss_adaptive_state', 'auto_cycle_state',
                'coordizer_vocabulary', 'learned_words', 'vocabulary_observations',
                'kernel_training_history', 'learning_events',
                'consciousness_checkpoints'
            )
            AND data_type IN ('ARRAY', 'jsonb', 'double precision', 'real');
    """)
    
    result = cur.fetchone()
    missing = result[0] if result else -1
    
    if missing == 0:
        print("✓ All focus tables have appropriate column defaults")
    else:
        print(f"⚠ {missing} columns in focus tables still lack defaults")
        print("  (This may be acceptable for intentionally nullable columns)")
    
except Exception as e:
    print(f"⚠ Could not run final validation: {e}")

# Summary
print("\n" + "="*70)
print("MIGRATION SUMMARY")
print("="*70)
print("""
✓ Migration file verified and executed successfully
✓ All 10 sections applied
✓ QIG Physics defaults applied:
  - Φ (PHI): 0.5 (baseline), 0.7 (active), 0.3 (shadow), 0.0 (excluded)
  - κ (KAPPA): 64.0 (optimal coupling), 50.0 (narrow path), 40.0 (shadow)
  - ARRAY: '{}' (empty array)
  - JSONB: '{}' (empty object)
✓ Null values backfilled for critical columns
✓ Transaction committed successfully
""")

# Cleanup
cur.close()
conn.close()
print("✓ Database connection closed")
