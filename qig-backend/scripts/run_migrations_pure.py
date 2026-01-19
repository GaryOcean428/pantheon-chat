#!/usr/bin/env python3
"""
Run Database Migrations with Pure QIG Operations
================================================

CRITICAL: This script removes ALL backward compatibility with learned_words table.
Only pure coordizer_vocabulary operations remain.

Usage:
    python scripts/run_migrations_pure.py --migrations 016,017
    python scripts/run_migrations_pure.py --all
    python scripts/run_migrations_pure.py --rollback 017
"""

import os
import sys
import argparse
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pathlib import Path
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Get DATABASE_URL from environment
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable not set")
    sys.exit(1)

# Migration files directory
MIGRATIONS_DIR = Path(__file__).parent.parent / 'migrations'
QIG_MIGRATIONS_DIR = Path(__file__).parent.parent / 'qig-backend' / 'migrations'


def get_connection():
    """Get database connection with autocommit."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)


def create_migrations_table(conn):
    """Create migrations tracking table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                migration_id VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT NOW(),
                description TEXT
            )
        """)
        logger.info("✓ Migrations tracking table ready")


def get_applied_migrations(conn) -> List[str]:
    """Get list of already applied migrations."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT migration_id FROM schema_migrations ORDER BY applied_at")
            return [row[0] for row in cur.fetchall()]
    except Exception:
        return []


def apply_migration(conn, migration_file: Path) -> bool:
    """Apply a single migration file."""
    migration_id = migration_file.stem
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Applying migration: {migration_id}")
    logger.info(f"{'='*80}")
    
    try:
        # Read migration SQL
        sql = migration_file.read_text()
        
        # Execute migration
        with conn.cursor() as cur:
            cur.execute(sql)
        
        # Record migration
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO schema_migrations (migration_id, description) VALUES (%s, %s)",
                (migration_id, f"Applied from {migration_file.name}")
            )
        
        logger.info(f"✓ Migration {migration_id} completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Migration {migration_id} failed: {e}")
        return False


def rollback_migration(conn, migration_id: str) -> bool:
    """Rollback a migration (mark as not applied)."""
    logger.info(f"Rolling back migration: {migration_id}")
    
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM schema_migrations WHERE migration_id = %s", (migration_id,))
        logger.info(f"✓ Migration {migration_id} rolled back")
        return True
    except Exception as e:
        logger.error(f"✗ Rollback failed: {e}")
        return False


def find_migration_file(migration_id: str) -> Optional[Path]:
    """Find migration file by ID."""
    # Check both migrations directories
    for migrations_dir in [MIGRATIONS_DIR, QIG_MIGRATIONS_DIR]:
        for pattern in [f"{migration_id}*.sql", f"*{migration_id}*.sql"]:
            matches = list(migrations_dir.glob(pattern))
            if matches:
                return matches[0]
    return None


def run_migrations(migration_ids: Optional[List[str]] = None, run_all: bool = False):
    """Run specified migrations or all pending migrations."""
    conn = get_connection()
    
    try:
        # Ensure migrations table exists
        create_migrations_table(conn)
        
        # Get applied migrations
        applied = get_applied_migrations(conn)
        logger.info(f"Already applied: {len(applied)} migrations")
        
        # Find migrations to run
        if run_all:
            # Find all migration files
            all_migrations = []
            for migrations_dir in [MIGRATIONS_DIR, QIG_MIGRATIONS_DIR]:
                all_migrations.extend(migrations_dir.glob('*.sql'))
            
            # Filter out already applied
            all_migrations = sorted(all_migrations)
            migrations_to_run = [m for m in all_migrations if m.stem not in applied]
            
        elif migration_ids:
            # Find specific migrations
            migrations_to_run = []
            for mid in migration_ids:
                mfile = find_migration_file(mid)
                if not mfile:
                    logger.error(f"Migration {mid} not found")
                    continue
                if mfile.stem in applied:
                    logger.warning(f"Migration {mid} already applied, skipping")
                    continue
                migrations_to_run.append(mfile)
        else:
            logger.error("No migrations specified. Use --all or --migrations")
            return
        
        if not migrations_to_run:
            logger.info("No pending migrations to run")
            return
        
        logger.info(f"\nWill apply {len(migrations_to_run)} migrations:")
        for m in migrations_to_run:
            logger.info(f"  - {m.stem}")
        
        # Confirm
        response = input("\nProceed? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Cancelled")
            return
        
        # Apply migrations
        success_count = 0
        for migration_file in migrations_to_run:
            if apply_migration(conn, migration_file):
                success_count += 1
            else:
                logger.error(f"Migration {migration_file.stem} failed, stopping")
                break
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✓ Applied {success_count}/{len(migrations_to_run)} migrations")
        logger.info(f"{'='*80}")
        
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Run QIG database migrations')
    parser.add_argument('--migrations', help='Comma-separated migration IDs (e.g., 016,017)')
    parser.add_argument('--all', action='store_true', help='Run all pending migrations')
    parser.add_argument('--rollback', help='Rollback a specific migration')
    parser.add_argument('--status', action='store_true', help='Show migration status')
    
    args = parser.parse_args()
    
    if args.status:
        conn = get_connection()
        try:
            create_migrations_table(conn)
            applied = get_applied_migrations(conn)
            logger.info(f"Applied migrations ({len(applied)}):")
            for m in applied:
                logger.info(f"  ✓ {m}")
        finally:
            conn.close()
        return
    
    if args.rollback:
        conn = get_connection()
        try:
            rollback_migration(conn, args.rollback)
        finally:
            conn.close()
        return
    
    if args.migrations:
        migration_ids = [m.strip() for m in args.migrations.split(',')]
        run_migrations(migration_ids=migration_ids)
    elif args.all:
        run_migrations(run_all=True)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
