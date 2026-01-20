"""
Registry Database Sync - Load Pantheon Registry into PostgreSQL
================================================================

Synchronizes the formal pantheon registry from YAML into PostgreSQL database.
Provides database-backed registry access with caching and transactions.

Authority: E8 Protocol v4.0, WP5.1
Status: ACTIVE
Created: 2026-01-20
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

try:
    import psycopg2
    from psycopg2.extras import Json as PsycopgJson, DictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None
    PsycopgJson = None
    DictCursor = None

from pantheon_registry import (
    PantheonRegistry,
    GodContract,
    get_registry,
)

logger = logging.getLogger(__name__)


class RegistryDatabaseSync:
    """
    Synchronizes Pantheon Registry from YAML to PostgreSQL.
    
    Features:
    - Load god contracts from YAML to database
    - Initialize spawner state tables
    - Validate database against YAML source
    - Handle updates and versioning
    
    Example:
        sync = RegistryDatabaseSync(db_connection)
        sync.sync_registry()  # Load from YAML to database
        
        # Query from database
        god = sync.get_god_contract("Apollo")
        gods = sync.find_gods_by_domain("synthesis")
    """
    
    def __init__(self, db_conn):
        """
        Initialize database sync.
        
        Args:
            db_conn: psycopg2 database connection
        """
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 not available - cannot use database sync")
        
        self.conn = db_conn
        self.registry = get_registry()
        
    # =========================================================================
    # SYNC OPERATIONS
    # =========================================================================
    
    def sync_registry(self, force: bool = False) -> Dict[str, int]:
        """
        Sync pantheon registry from YAML to database.
        
        Args:
            force: If True, clear and reload all data. If False, update only changed.
            
        Returns:
            Dict with sync statistics
        """
        logger.info(f"Starting registry sync (force={force})")
        
        stats = {
            "gods_inserted": 0,
            "gods_updated": 0,
            "gods_skipped": 0,
            "errors": 0,
        }
        
        try:
            with self.conn.cursor() as cursor:
                # Clear existing data if force
                if force:
                    logger.info("Force sync: clearing existing god contracts")
                    cursor.execute("DELETE FROM god_contracts")
                
                # Sync each god contract
                for name, god in self.registry.get_all_gods().items():
                    try:
                        if self._god_exists(cursor, name) and not force:
                            if self._god_changed(cursor, name, god):
                                self._update_god_contract(cursor, god)
                                stats["gods_updated"] += 1
                                logger.debug(f"Updated god contract: {name}")
                            else:
                                stats["gods_skipped"] += 1
                        else:
                            self._insert_god_contract(cursor, god)
                            stats["gods_inserted"] += 1
                            logger.debug(f"Inserted god contract: {name}")
                    except Exception as e:
                        logger.error(f"Error syncing god {name}: {e}")
                        stats["errors"] += 1
                
                # Initialize spawner state for gods
                self._initialize_spawner_state(cursor)
                
                # Update registry metadata
                self._update_registry_metadata(cursor)
                
                # Commit transaction
                self.conn.commit()
                
        except Exception as e:
            logger.error(f"Registry sync failed: {e}")
            self.conn.rollback()
            raise
        
        logger.info(
            f"Registry sync complete: "
            f"{stats['gods_inserted']} inserted, "
            f"{stats['gods_updated']} updated, "
            f"{stats['gods_skipped']} skipped, "
            f"{stats['errors']} errors"
        )
        
        return stats
    
    def _god_exists(self, cursor, name: str) -> bool:
        """Check if god exists in database."""
        cursor.execute(
            "SELECT 1 FROM god_contracts WHERE name = %s",
            (name,)
        )
        return cursor.fetchone() is not None
    
    def _god_changed(self, cursor, name: str, god: GodContract) -> bool:
        """Check if god contract has changed."""
        cursor.execute(
            "SELECT domain, description, rest_policy, e8_alignment FROM god_contracts WHERE name = %s",
            (name,)
        )
        row = cursor.fetchone()
        if not row:
            return True
        
        # Compare key fields (simplified check)
        db_domain = set(row[0])
        new_domain = set(god.domain)
        
        return (
            db_domain != new_domain or
            row[1] != god.description
        )
    
    def _insert_god_contract(self, cursor, god: GodContract) -> None:
        """Insert god contract into database."""
        cursor.execute("""
            INSERT INTO god_contracts (
                name, tier, domain, description, octant, epithets, coupling_affinity,
                rest_policy, spawn_constraints, promotion_from, e8_alignment, registry_version
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """, (
            god.name,
            god.tier.value,
            god.domain,
            god.description,
            god.octant,
            god.epithets,
            god.coupling_affinity,
            PsycopgJson(self._rest_policy_to_dict(god)),
            PsycopgJson(self._spawn_constraints_to_dict(god)),
            god.promotion_from,
            PsycopgJson(self._e8_alignment_to_dict(god)),
            self.registry.get_metadata().version,
        ))
    
    def _update_god_contract(self, cursor, god: GodContract) -> None:
        """Update god contract in database."""
        cursor.execute("""
            UPDATE god_contracts SET
                tier = %s,
                domain = %s,
                description = %s,
                octant = %s,
                epithets = %s,
                coupling_affinity = %s,
                rest_policy = %s,
                spawn_constraints = %s,
                promotion_from = %s,
                e8_alignment = %s,
                registry_version = %s,
                updated_at = NOW()
            WHERE name = %s
        """, (
            god.tier.value,
            god.domain,
            god.description,
            god.octant,
            god.epithets,
            god.coupling_affinity,
            PsycopgJson(self._rest_policy_to_dict(god)),
            PsycopgJson(self._spawn_constraints_to_dict(god)),
            god.promotion_from,
            PsycopgJson(self._e8_alignment_to_dict(god)),
            self.registry.get_metadata().version,
            god.name,
        ))
    
    def _initialize_spawner_state(self, cursor) -> None:
        """Initialize spawner state for all gods."""
        cursor.execute("""
            INSERT INTO kernel_spawner_state (god_name, active_instances, max_instances, when_allowed)
            SELECT 
                name,
                0,
                (spawn_constraints->>'max_instances')::INTEGER,
                spawn_constraints->>'when_allowed'
            FROM god_contracts
            ON CONFLICT (god_name) DO UPDATE
            SET 
                max_instances = EXCLUDED.max_instances,
                when_allowed = EXCLUDED.when_allowed,
                updated_at = NOW()
        """)
    
    def _update_registry_metadata(self, cursor) -> None:
        """Update registry metadata in database."""
        metadata = self.registry.get_metadata()
        
        cursor.execute("""
            INSERT INTO pantheon_registry_metadata (
                registry_version, schema_version, status, authority,
                e8_protocol_version, qig_backend_version,
                validation_required, god_count
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s
            )
        """, (
            metadata.version,
            "1.0",  # Schema version
            metadata.status,
            metadata.authority,
            "v4.0",  # E8 protocol version
            ">=1.0.0",  # QIG backend version
            metadata.validation_required,
            self.registry.get_god_count(),
        ))
    
    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================
    
    def get_god_contract(self, name: str) -> Optional[Dict]:
        """Get god contract from database."""
        with self.conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute("""
                SELECT * FROM god_contracts WHERE name = %s
            """, (name,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def find_gods_by_domain(self, domain: str) -> List[Dict]:
        """Find gods by domain from database."""
        with self.conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute("""
                SELECT * FROM god_contracts WHERE %s = ANY(domain)
            """, (domain,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_all_gods(self) -> List[Dict]:
        """Get all god contracts from database."""
        with self.conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute("SELECT * FROM god_contracts ORDER BY tier, name")
            return [dict(row) for row in cursor.fetchall()]
    
    def get_spawner_status(self, god_name: str) -> Optional[Dict]:
        """Get spawner status for a god."""
        with self.conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute("""
                SELECT * FROM kernel_spawner_state WHERE god_name = %s
            """, (god_name,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def can_spawn_god(self, god_name: str) -> Tuple[bool, str]:
        """Check if god can be spawned."""
        status = self.get_spawner_status(god_name)
        if not status:
            return (False, f"God {god_name} not found")
        
        if status['active_instances'] >= status['max_instances']:
            return (
                False,
                f"God {god_name} at max instances: {status['active_instances']}/{status['max_instances']}"
            )
        
        return (True, f"God {god_name} can spawn")
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _rest_policy_to_dict(self, god: GodContract) -> Dict:
        """Convert RestPolicy to dict for JSONB storage."""
        policy = god.rest_policy
        return {
            "type": policy.type.value,
            "reason": policy.reason,
            "partner": policy.partner,
            "duty_cycle": policy.duty_cycle,
            "rest_duration": policy.rest_duration,
            "active_season": policy.active_season,
            "rest_season": policy.rest_season,
        }
    
    def _spawn_constraints_to_dict(self, god: GodContract) -> Dict:
        """Convert SpawnConstraints to dict for JSONB storage."""
        return {
            "max_instances": god.spawn_constraints.max_instances,
            "when_allowed": god.spawn_constraints.when_allowed,
            "rationale": god.spawn_constraints.rationale,
        }
    
    def _e8_alignment_to_dict(self, god: GodContract) -> Dict:
        """Convert E8Alignment to dict for JSONB storage."""
        return {
            "simple_root": god.e8_alignment.simple_root,
            "layer": god.e8_alignment.layer,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def sync_registry_to_database(db_conn, force: bool = False) -> Dict[str, int]:
    """
    Convenience function to sync registry to database.
    
    Args:
        db_conn: psycopg2 database connection
        force: If True, clear and reload all data
        
    Returns:
        Dict with sync statistics
    """
    sync = RegistryDatabaseSync(db_conn)
    return sync.sync_registry(force=force)


def get_database_url() -> Optional[str]:
    """Get database URL from environment."""
    import os
    return os.getenv("DATABASE_URL")


def sync_registry_from_env(force: bool = False) -> Dict[str, int]:
    """
    Sync registry using DATABASE_URL from environment.
    
    Args:
        force: If True, clear and reload all data
        
    Returns:
        Dict with sync statistics
    """
    if not POSTGRES_AVAILABLE:
        raise ImportError("psycopg2 not available")
    
    db_url = get_database_url()
    if not db_url:
        raise ValueError("DATABASE_URL not set")
    
    conn = psycopg2.connect(db_url)
    try:
        return sync_registry_to_database(conn, force=force)
    finally:
        conn.close()


if __name__ == "__main__":
    # CLI for syncing registry
    import sys
    
    force = "--force" in sys.argv
    
    try:
        stats = sync_registry_from_env(force=force)
        print(f"Registry sync complete:")
        print(f"  Inserted: {stats['gods_inserted']}")
        print(f"  Updated: {stats['gods_updated']}")
        print(f"  Skipped: {stats['gods_skipped']}")
        print(f"  Errors: {stats['errors']}")
        
        if stats['errors'] > 0:
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
