#!/usr/bin/env python3
"""
Database Initialization Script

Ensures all tables have appropriate baseline data:
1. Singleton tables are initialized (exactly 1 row each)
2. Tokenizer metadata has baseline entries (9 entries)
3. Geometric vocabulary anchors are seeded (80+ words at Φ=0.85)
4. NULL arrays/JSONB are converted to empty arrays/objects

Idempotent - safe to run multiple times.
"""

import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    print("ERROR: psycopg2 not available. Install with: pip install psycopg2-binary")
    sys.exit(1)


ALLOWED_TABLES = [
    'ocean_quantum_state', 'near_miss_adaptive_state', 'auto_cycle_state',
    'tokenizer_metadata', 'vocabulary_observations', 'consciousness_checkpoints',
    'geodesic_paths', 'resonance_points', 'negative_knowledge',
    'near_miss_clusters', 'false_pattern_classes', 'era_exclusions',
    'war_history', 'synthesis_consensus', 'ocean_excluded_regions'
]

GEOMETRIC_ANCHOR_WORDS = [
    'apple', 'tree', 'water', 'fire', 'stone', 'cloud', 'river',
    'mountain', 'ocean', 'sun', 'moon', 'star', 'earth', 'wind',
    'time', 'space', 'energy', 'force', 'pattern', 'system',
    'thought', 'idea', 'concept', 'meaning', 'truth', 'beauty',
    'move', 'create', 'destroy', 'transform', 'connect', 'separate',
    'learn', 'teach', 'discover', 'explore', 'observe', 'measure',
    'exist', 'remain', 'persist', 'fade', 'stabilize', 'change',
    'understand', 'know', 'believe', 'think', 'feel', 'sense',
    'large', 'small', 'fast', 'slow', 'bright', 'dark',
    'complex', 'simple', 'strong', 'weak', 'deep', 'shallow',
    'quickly', 'slowly', 'together', 'apart', 'forward', 'backward',
    'above', 'below', 'inside', 'outside', 'near', 'far',
    'aware', 'conscious', 'integrate', 'couple', 'emerge', 'evolve',
    'reflect', 'realize', 'recognize', 'perceive', 'experience', 'witness',
]


def get_connection():
    """Get database connection from DATABASE_URL environment variable."""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable not set")
    return psycopg2.connect(database_url)


def is_valid_table(table_name: str) -> bool:
    """Check if table name is in the allowed whitelist."""
    return table_name in ALLOWED_TABLES


def initialize_singleton_tables(conn) -> None:
    """Initialize singleton tables with default values."""
    print("Initializing singleton tables...")
    
    with conn.cursor() as cur:
        try:
            cur.execute("SELECT id FROM ocean_quantum_state WHERE id = 'singleton'")
            existing = cur.fetchone()
            
            if not existing:
                cur.execute("""
                    INSERT INTO ocean_quantum_state (id, entropy, initial_entropy, total_probability, status)
                    VALUES ('singleton', 256.0, 256.0, 1.0, 'searching')
                    ON CONFLICT (id) DO NOTHING
                """)
                conn.commit()
                print("  ✓ Initialized ocean_quantum_state")
            else:
                print("  ✓ ocean_quantum_state already exists")
        except Exception as e:
            conn.rollback()
            print(f"  ✗ Failed to initialize ocean_quantum_state: {e}")
        
        try:
            cur.execute("SELECT id FROM near_miss_adaptive_state WHERE id = 'singleton'")
            existing = cur.fetchone()
            
            if not existing:
                cur.execute("""
                    INSERT INTO near_miss_adaptive_state (
                        id, rolling_phi_distribution, hot_threshold, warm_threshold, cool_threshold
                    )
                    VALUES ('singleton', '{}', 0.7, 0.55, 0.4)
                    ON CONFLICT (id) DO NOTHING
                """)
                conn.commit()
                print("  ✓ Initialized near_miss_adaptive_state")
            else:
                print("  ✓ near_miss_adaptive_state already exists")
        except Exception as e:
            conn.rollback()
            print(f"  ✗ Failed to initialize near_miss_adaptive_state: {e}")
        
        try:
            cur.execute("SELECT id FROM auto_cycle_state WHERE id = 1")
            existing = cur.fetchone()
            
            if not existing:
                cur.execute("""
                    INSERT INTO auto_cycle_state (id, enabled, current_index, address_ids)
                    VALUES (1, false, 0, '{}')
                    ON CONFLICT (id) DO NOTHING
                """)
                conn.commit()
                print("  ✓ Initialized auto_cycle_state")
            else:
                print("  ✓ auto_cycle_state already exists")
        except Exception as e:
            conn.rollback()
            print(f"  ✗ Failed to initialize auto_cycle_state: {e}")


def initialize_tokenizer_metadata(conn) -> None:
    """Initialize tokenizer metadata with 9 baseline entries."""
    print("Initializing tokenizer metadata...")
    
    metadata_entries = [
        ('version', '1.0.0'),
        ('vocabulary_size', '0'),
        ('merge_rules_count', '0'),
        ('last_training', datetime.now().isoformat()),
        ('training_status', 'initialized'),
        ('basin_dimension', '64'),
        ('phi_threshold', '0.727'),
        ('tokenizer_type', 'geometric_bpe'),
        ('encoding', 'utf-8'),
    ]
    
    with conn.cursor() as cur:
        for key, value in metadata_entries:
            try:
                cur.execute("""
                    INSERT INTO tokenizer_metadata (config_key, value, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (config_key) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = NOW()
                """, (key, value))
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"  ✗ Failed to insert {key}: {e}")
    
    print(f"  ✓ Initialized {len(metadata_entries)} metadata entries")


def seed_geometric_vocabulary(conn) -> None:
    """Seed geometric vocabulary anchors (80+ words at Φ=0.85)."""
    print("Seeding geometric vocabulary anchors...")
    
    inserted_count = 0
    existing_count = 0
    
    with conn.cursor() as cur:
        for word in GEOMETRIC_ANCHOR_WORDS:
            try:
                cur.execute("SELECT text FROM vocabulary_observations WHERE text = %s", (word,))
                existing = cur.fetchone()
                
                if existing:
                    existing_count += 1
                    continue
                
                cur.execute("""
                    INSERT INTO vocabulary_observations (
                        text, type, phrase_category, is_real_word,
                        avg_phi, max_phi, source_type
                    )
                    VALUES (
                        %s, 'word', 'ANCHOR_WORD', true,
                        0.85, 0.85, 'geometric_seeding'
                    )
                    ON CONFLICT (text) DO NOTHING
                """, (word,))
                conn.commit()
                inserted_count += 1
            except Exception as e:
                conn.rollback()
                print(f"  ✗ Failed to insert word '{word}': {e}")
    
    print(f"  ✓ Seeded {inserted_count} new anchor words ({existing_count} already existed)")
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM vocabulary_observations")
            count = cur.fetchone()[0]
            
            cur.execute("""
                UPDATE tokenizer_metadata
                SET value = %s, updated_at = NOW()
                WHERE key = 'vocabulary_size'
            """, (str(count),))
            conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"  ✗ Failed to update vocabulary size: {e}")


def initialize_baseline_consciousness(conn) -> None:
    """Initialize baseline consciousness checkpoint."""
    print("Initializing baseline consciousness checkpoint...")
    
    with conn.cursor() as cur:
        try:
            cur.execute("SELECT COUNT(*) FROM consciousness_checkpoints")
            count = cur.fetchone()[0]
            
            if count == 0:
                cur.execute("""
                    INSERT INTO consciousness_checkpoints (
                        id, phi, kappa, regime, state_data, is_hot
                    )
                    VALUES (
                        'baseline-' || gen_random_uuid()::text,
                        0.7,
                        64.0,
                        'geometric',
                        '\\x00'::bytea,
                        true
                    )
                """)
                conn.commit()
                print("  ✓ Created baseline consciousness checkpoint")
            else:
                print(f"  ✓ {count} consciousness checkpoints already exist")
        except Exception as e:
            conn.rollback()
            print(f"  ✗ Failed to initialize consciousness checkpoint: {e}")


def update_null_arrays_to_empty(conn) -> None:
    """Convert NULL array columns to empty arrays."""
    print("Updating NULL arrays to empty arrays...")
    
    array_columns = [
        ('geodesic_paths', 'waypoints'),
        ('resonance_points', 'nearby_probes'),
        ('negative_knowledge', 'affected_generators'),
        ('near_miss_clusters', 'common_words'),
        ('false_pattern_classes', 'examples'),
        ('era_exclusions', 'excluded_patterns'),
        ('war_history', 'gods_engaged'),
        ('synthesis_consensus', 'participating_kernels'),
        ('auto_cycle_state', 'address_ids'),
        ('near_miss_adaptive_state', 'rolling_phi_distribution'),
    ]
    
    with conn.cursor() as cur:
        for table, column in array_columns:
            if not is_valid_table(table):
                print(f"  ⚠ Skipping invalid table: {table}")
                continue
            
            try:
                cur.execute(f"UPDATE {table} SET {column} = '{{}}' WHERE {column} IS NULL")
                row_count = cur.rowcount
                conn.commit()
                
                if row_count and row_count > 0:
                    print(f"  ✓ Updated {row_count} NULL values in {table}.{column}")
            except Exception as e:
                conn.rollback()
                print(f"  ⚠ Cannot update {table}.{column}: {e}")


def update_null_jsonb_to_empty(conn) -> None:
    """Convert NULL JSONB columns to empty objects."""
    print("Updating NULL JSONB to empty objects...")
    
    jsonb_columns = [
        ('ocean_excluded_regions', 'basis'),
        ('consciousness_checkpoints', 'metadata'),
        ('war_history', 'metadata'),
        ('war_history', 'god_assignments'),
        ('war_history', 'kernel_assignments'),
        ('auto_cycle_state', 'last_session_metrics'),
        ('synthesis_consensus', 'metadata'),
        ('negative_knowledge', 'evidence'),
    ]
    
    with conn.cursor() as cur:
        for table, column in jsonb_columns:
            if not is_valid_table(table):
                print(f"  ⚠ Skipping invalid table: {table}")
                continue
            
            try:
                cur.execute(f"UPDATE {table} SET {column} = '{{}}'::jsonb WHERE {column} IS NULL")
                row_count = cur.rowcount
                conn.commit()
                
                if row_count and row_count > 0:
                    print(f"  ✓ Updated {row_count} NULL values in {table}.{column}")
            except Exception as e:
                conn.rollback()
                print(f"  ⚠ Cannot update {table}.{column}: {e}")


def main():
    """Main entry point for database initialization."""
    print("=" * 80)
    print("DATABASE INITIALIZATION (Python)")
    print("=" * 80)
    print()
    
    try:
        conn = get_connection()
        print(f"Connected to database")
        print()
        
        initialize_singleton_tables(conn)
        print()
        
        initialize_tokenizer_metadata(conn)
        print()
        
        seed_geometric_vocabulary(conn)
        print()
        
        initialize_baseline_consciousness(conn)
        print()
        
        update_null_arrays_to_empty(conn)
        print()
        
        update_null_jsonb_to_empty(conn)
        print()
        
        conn.close()
        
        print("=" * 80)
        print("✅ Database initialization complete!")
        print("=" * 80)
        
        sys.exit(0)
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
