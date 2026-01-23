#!/usr/bin/env python3
"""
M8 Persistence Layer - Database operations for kernel spawning

PostgreSQL persistence for:
- m8_spawn_proposals: Spawn proposals and votes
- m8_spawned_kernels: Spawned kernel profiles and state
- m8_spawn_history: Complete spawn event history
- m8_kernel_awareness: Kernel self-awareness tracking
"""

import os
from contextlib import contextmanager
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    M8_PSYCOPG2_AVAILABLE = True
except ImportError:
    M8_PSYCOPG2_AVAILABLE = False
    print("[M8] psycopg2 not available - PostgreSQL persistence disabled")

class M8SpawnerPersistence:
    """
    PostgreSQL persistence layer for M8 Kernel Spawning data.
    
    Persists:
    - m8_spawn_proposals: Spawn proposals and their votes
    - m8_spawned_kernels: Spawned kernel profiles and state
    - m8_spawn_history: Complete spawn event history
    - m8_kernel_awareness: Kernel self-awareness tracking
    
    Pattern follows ShadowPantheonPersistence for consistency.
    """

    def __init__(self):
        """Initialize persistence layer."""
        self.database_url = os.environ.get('DATABASE_URL')
        self._tables_ensured = False
        
        if not self.database_url:
            print("[M8Persistence] WARNING: DATABASE_URL not set - persistence disabled")
        elif not M8_PSYCOPG2_AVAILABLE:
            print("[M8Persistence] WARNING: psycopg2 not available - persistence disabled")
        else:
            self._ensure_m8_tables()
            print("[M8Persistence] ✓ PostgreSQL persistence enabled")

    @contextmanager
    def _get_db_connection(self):
        """Get a database connection with automatic cleanup."""
        if not self.database_url or not M8_PSYCOPG2_AVAILABLE:
            yield None
            return
        
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"[M8Persistence] Database error: {e}")
            yield None
        finally:
            if conn:
                conn.close()

    def _ensure_m8_tables(self) -> bool:
        """Create M8 spawning tables if they don't exist."""
        if self._tables_ensured:
            return True
        
        with self._get_db_connection() as conn:
            if not conn:
                return False
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS m8_spawn_proposals (
                            proposal_id VARCHAR(64) PRIMARY KEY,
                            proposed_name VARCHAR(128),
                            proposed_domain VARCHAR(256),
                            proposed_element VARCHAR(128),
                            proposed_role VARCHAR(128),
                            reason VARCHAR(64),
                            parent_gods JSONB DEFAULT '[]'::jsonb,
                            votes_for JSONB DEFAULT '[]'::jsonb,
                            votes_against JSONB DEFAULT '[]'::jsonb,
                            abstentions JSONB DEFAULT '[]'::jsonb,
                            status VARCHAR(32) DEFAULT 'pending',
                            metadata JSONB DEFAULT '{}'::jsonb,
                            proposed_at TIMESTAMP DEFAULT NOW(),
                            updated_at TIMESTAMP DEFAULT NOW()
                        );
                        
                        CREATE TABLE IF NOT EXISTS m8_spawned_kernels (
                            kernel_id VARCHAR(64) PRIMARY KEY,
                            god_name VARCHAR(128),
                            domain VARCHAR(256),
                            mode VARCHAR(32),
                            affinity_strength FLOAT8 DEFAULT 0.5,
                            entropy_threshold FLOAT8 DEFAULT 0.5,
                            basin_coords FLOAT8[],
                            parent_gods JSONB DEFAULT '[]'::jsonb,
                            spawn_reason VARCHAR(64),
                            proposal_id VARCHAR(64),
                            genesis_votes JSONB DEFAULT '{}'::jsonb,
                            basin_lineage JSONB DEFAULT '{}'::jsonb,
                            m8_position JSONB,
                            observation_state JSONB DEFAULT '{}'::jsonb,
                            autonomic_state JSONB DEFAULT '{}'::jsonb,
                            profile_metadata JSONB DEFAULT '{}'::jsonb,
                            status VARCHAR(32) DEFAULT 'active',
                            spawned_at TIMESTAMP DEFAULT NOW(),
                            retired_at TIMESTAMP
                        );
                        
                        CREATE TABLE IF NOT EXISTS m8_spawn_history (
                            event_id VARCHAR(64) PRIMARY KEY,
                            event_type VARCHAR(64),
                            kernel_id VARCHAR(64),
                            god_name VARCHAR(128),
                            payload JSONB DEFAULT '{}'::jsonb,
                            occurred_at TIMESTAMP DEFAULT NOW()
                        );
                        
                        CREATE TABLE IF NOT EXISTS m8_kernel_awareness (
                            kernel_id VARCHAR(64) PRIMARY KEY,
                            phi_trajectory JSONB DEFAULT '[]'::jsonb,
                            kappa_trajectory JSONB DEFAULT '[]'::jsonb,
                            curvature_history JSONB DEFAULT '[]'::jsonb,
                            stuck_signals JSONB DEFAULT '[]'::jsonb,
                            geometric_deadends JSONB DEFAULT '[]'::jsonb,
                            research_opportunities JSONB DEFAULT '[]'::jsonb,
                            last_spawn_proposal VARCHAR(64),
                            awareness_updated_at TIMESTAMP DEFAULT NOW()
                        );
                        
                        CREATE TABLE IF NOT EXISTS kernel_evolution_fitness (
                            kernel_id VARCHAR(64) PRIMARY KEY,
                            phi_current FLOAT8 DEFAULT 0.0,
                            phi_gradient FLOAT8 DEFAULT 0.0,
                            phi_velocity FLOAT8 DEFAULT 0.0,
                            kappa_current FLOAT8 DEFAULT 0.0,
                            kappa_stability FLOAT8 DEFAULT 0.0,
                            fisher_diversity FLOAT8 DEFAULT 0.0,
                            geometric_fitness FLOAT8 DEFAULT 0.0,
                            dimensional_state VARCHAR(8) DEFAULT 'D3',
                            evolution_pressure FLOAT8 DEFAULT 0.0,
                            cannibalize_priority FLOAT8 DEFAULT 0.0,
                            merge_affinity JSONB DEFAULT '{}'::jsonb,
                            last_evolution_event VARCHAR(64),
                            fitness_computed_at TIMESTAMP DEFAULT NOW()
                        );
                        
                        CREATE TABLE IF NOT EXISTS kernel_evolution_events (
                            event_id VARCHAR(64) PRIMARY KEY,
                            event_type VARCHAR(32) NOT NULL,
                            source_kernel_id VARCHAR(64),
                            target_kernel_id VARCHAR(64),
                            result_kernel_id VARCHAR(64),
                            geometric_reasoning JSONB DEFAULT '{}'::jsonb,
                            phi_before FLOAT8,
                            phi_after FLOAT8,
                            kappa_before FLOAT8,
                            kappa_after FLOAT8,
                            fisher_distance FLOAT8,
                            fitness_delta FLOAT8,
                            occurred_at TIMESTAMP DEFAULT NOW()
                        );
                        
                        CREATE INDEX IF NOT EXISTS idx_m8_proposals_status ON m8_spawn_proposals(status);
                        CREATE INDEX IF NOT EXISTS idx_m8_kernels_status ON m8_spawned_kernels(status);
                        CREATE INDEX IF NOT EXISTS idx_m8_kernels_god ON m8_spawned_kernels(god_name);
                        CREATE INDEX IF NOT EXISTS idx_m8_history_kernel ON m8_spawn_history(kernel_id);
                        CREATE INDEX IF NOT EXISTS idx_m8_history_type ON m8_spawn_history(event_type);
                        CREATE INDEX IF NOT EXISTS idx_evolution_fitness ON kernel_evolution_fitness(geometric_fitness DESC);
                        CREATE INDEX IF NOT EXISTS idx_evolution_cannibalize ON kernel_evolution_fitness(cannibalize_priority DESC);
                        CREATE INDEX IF NOT EXISTS idx_evolution_events_type ON kernel_evolution_events(event_type);
                    """)
                    conn.commit()
                self._tables_ensured = True
                return True
            except Exception as e:
                print(f"[M8Persistence] Table creation error: {e}")
                return False

    def _vector_to_pg(self, vec) -> Optional[str]:
        """Convert numpy array or list to PostgreSQL array format."""
        if vec is None:
            return None
        if isinstance(vec, np.ndarray):
            arr = vec.tolist()
        else:
            arr = list(vec)
        return '{' + ','.join(str(x) for x in arr) + '}'

    def _pg_to_vector(self, pg_arr) -> Optional[np.ndarray]:
        """Convert PostgreSQL array to numpy array."""
        if pg_arr is None:
            return None
        if isinstance(pg_arr, list):
            return np.array(pg_arr)
        return np.array(pg_arr)

    def persist_proposal(self, proposal) -> bool:
        """Save or update a spawn proposal to PostgreSQL."""
        with self._get_db_connection() as conn:
            if not conn:
                return False
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO m8_spawn_proposals 
                        (proposal_id, proposed_name, proposed_domain, proposed_element,
                         proposed_role, reason, parent_gods, votes_for, votes_against,
                         abstentions, status, metadata, proposed_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (proposal_id) DO UPDATE SET
                            proposed_name = EXCLUDED.proposed_name,
                            proposed_domain = EXCLUDED.proposed_domain,
                            proposed_element = EXCLUDED.proposed_element,
                            proposed_role = EXCLUDED.proposed_role,
                            reason = EXCLUDED.reason,
                            parent_gods = EXCLUDED.parent_gods,
                            votes_for = EXCLUDED.votes_for,
                            votes_against = EXCLUDED.votes_against,
                            abstentions = EXCLUDED.abstentions,
                            status = EXCLUDED.status,
                            metadata = EXCLUDED.metadata,
                            updated_at = NOW()
                    """, (
                        proposal.proposal_id,
                        proposal.proposed_name,
                        proposal.proposed_domain,
                        proposal.proposed_element,
                        proposal.proposed_role,
                        proposal.reason.value if hasattr(proposal.reason, 'value') else str(proposal.reason),
                        json.dumps(list(proposal.parent_gods) if proposal.parent_gods else []),
                        json.dumps(list(proposal.votes_for)),
                        json.dumps(list(proposal.votes_against)),
                        json.dumps(list(proposal.abstentions)),
                        proposal.status,
                        json.dumps(proposal.metadata if hasattr(proposal, 'metadata') else {}),
                        proposal.proposed_at,
                    ))
                    conn.commit()
                return True
            except Exception as e:
                print(f"[M8Persistence] Failed to persist proposal: {e}")
                return False

    def persist_kernel(self, kernel) -> bool:
        """Save or update a spawned kernel to PostgreSQL."""
        with self._get_db_connection() as conn:
            if not conn:
                return False
            try:
                with conn.cursor() as cur:
                    basin_coords = self._vector_to_pg(kernel.profile.affinity_basin)
                    cur.execute("""
                        INSERT INTO m8_spawned_kernels
                        (kernel_id, god_name, domain, mode, affinity_strength,
                         entropy_threshold, basin_coords, parent_gods, spawn_reason,
                         proposal_id, genesis_votes, basin_lineage, m8_position,
                         observation_state, autonomic_state, profile_metadata,
                         status, spawned_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (kernel_id) DO UPDATE SET
                            god_name = EXCLUDED.god_name,
                            domain = EXCLUDED.domain,
                            mode = EXCLUDED.mode,
                            affinity_strength = EXCLUDED.affinity_strength,
                            entropy_threshold = EXCLUDED.entropy_threshold,
                            basin_coords = EXCLUDED.basin_coords,
                            parent_gods = EXCLUDED.parent_gods,
                            spawn_reason = EXCLUDED.spawn_reason,
                            genesis_votes = EXCLUDED.genesis_votes,
                            basin_lineage = EXCLUDED.basin_lineage,
                            m8_position = EXCLUDED.m8_position,
                            observation_state = EXCLUDED.observation_state,
                            autonomic_state = EXCLUDED.autonomic_state,
                            profile_metadata = EXCLUDED.profile_metadata,
                            status = EXCLUDED.status
                    """, (
                        kernel.kernel_id,
                        kernel.profile.god_name,
                        kernel.profile.domain,
                        kernel.profile.mode.value if hasattr(kernel.profile.mode, 'value') else str(kernel.profile.mode),
                        kernel.profile.affinity_strength,
                        kernel.profile.entropy_threshold,
                        basin_coords,
                        json.dumps(kernel.parent_gods),
                        kernel.spawn_reason.value if hasattr(kernel.spawn_reason, 'value') else str(kernel.spawn_reason),
                        kernel.proposal_id,
                        json.dumps(kernel.genesis_votes),
                        json.dumps(kernel.basin_lineage),
                        json.dumps(kernel.m8_position) if kernel.m8_position else None,
                        json.dumps(kernel.observation.to_dict()) if hasattr(kernel, 'observation') else '{}',
                        json.dumps(kernel.autonomic.to_dict()) if hasattr(kernel, 'autonomic') else '{}',
                        json.dumps(kernel.profile.metadata) if hasattr(kernel.profile, 'metadata') else '{}',
                        'observing' if kernel.is_observing() else 'active' if kernel.is_active() else 'pending',
                        kernel.spawned_at,
                    ))
                    conn.commit()
                return True
            except Exception as e:
                print(f"[M8Persistence] Failed to persist kernel: {e}")
                return False

    def persist_history(self, record: Dict) -> bool:
        """Append a spawn history record to PostgreSQL."""
        with self._get_db_connection() as conn:
            if not conn:
                return False
            try:
                event_id = record.get('event_id', f"evt_{uuid.uuid4().hex}")
                event_type = record.get('event', record.get('event_type', 'unknown'))
                kernel_id = record.get('kernel_id', record.get('kernel', {}).get('kernel_id'))
                god_name = record.get('god_name', record.get('kernel', {}).get('god_name'))
                
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO m8_spawn_history
                        (event_id, event_type, kernel_id, god_name, payload, occurred_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (event_id) DO NOTHING
                    """, (
                        event_id,
                        event_type,
                        kernel_id,
                        god_name,
                        json.dumps(record),
                        record.get('timestamp', datetime.now().isoformat()),
                    ))
                    conn.commit()
                return True
            except Exception as e:
                print(f"[M8Persistence] Failed to persist history: {e}")
                return False

    def persist_awareness(self, awareness) -> bool:
        """Save or update kernel awareness state to PostgreSQL."""
        with self._get_db_connection() as conn:
            if not conn:
                return False
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO m8_kernel_awareness
                        (kernel_id, phi_trajectory, kappa_trajectory, curvature_history,
                         stuck_signals, geometric_deadends, research_opportunities,
                         last_spawn_proposal, awareness_updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (kernel_id) DO UPDATE SET
                            phi_trajectory = EXCLUDED.phi_trajectory,
                            kappa_trajectory = EXCLUDED.kappa_trajectory,
                            curvature_history = EXCLUDED.curvature_history,
                            stuck_signals = EXCLUDED.stuck_signals,
                            geometric_deadends = EXCLUDED.geometric_deadends,
                            research_opportunities = EXCLUDED.research_opportunities,
                            last_spawn_proposal = EXCLUDED.last_spawn_proposal,
                            awareness_updated_at = NOW()
                    """, (
                        awareness.kernel_id,
                        json.dumps(awareness.phi_trajectory[-100:]),
                        json.dumps(awareness.kappa_trajectory[-100:]),
                        json.dumps(awareness.curvature_history[-50:]),
                        json.dumps(awareness.stuck_signals[-20:]),
                        json.dumps(awareness.geometric_deadends[-10:]),
                        json.dumps(awareness.research_opportunities[-30:]),
                        awareness.last_spawn_proposal,
                    ))
                    conn.commit()
                return True
            except Exception as e:
                print(f"[M8Persistence] Failed to persist awareness: {e}")
                return False

    def load_all_proposals(self) -> List[Dict]:
        """Load all spawn proposals from PostgreSQL."""
        with self._get_db_connection() as conn:
            if not conn:
                return []
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM m8_spawn_proposals
                        ORDER BY proposed_at DESC
                    """)
                    return [dict(row) for row in cur.fetchall()]
            except Exception as e:
                print(f"[M8Persistence] Failed to load proposals: {e}")
                return []

    def load_all_kernels(self) -> List[Dict]:
        """Load all spawned kernels from PostgreSQL."""
        with self._get_db_connection() as conn:
            if not conn:
                return []
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM m8_spawned_kernels
                        WHERE status != 'deleted'
                        ORDER BY spawned_at DESC
                    """)
                    rows = cur.fetchall()
                    result = []
                    for row in rows:
                        d = dict(row)
                        if d.get('basin_coords'):
                            d['basin_coords'] = self._pg_to_vector(d['basin_coords'])
                        result.append(d)
                    return result
            except Exception as e:
                print(f"[M8Persistence] Failed to load kernels: {e}")
                return []

    def load_spawn_history(self, limit: int = 100) -> List[Dict]:
        """Load spawn history from PostgreSQL."""
        with self._get_db_connection() as conn:
            if not conn:
                return []
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM m8_spawn_history
                        ORDER BY occurred_at DESC
                        LIMIT %s
                    """, (limit,))
                    return [dict(row) for row in cur.fetchall()]
            except Exception as e:
                print(f"[M8Persistence] Failed to load history: {e}")
                return []

    def load_all_awareness(self) -> List[Dict]:
        """Load all kernel awareness states from PostgreSQL."""
        with self._get_db_connection() as conn:
            if not conn:
                return []
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM m8_kernel_awareness
                        ORDER BY awareness_updated_at DESC
                    """)
                    return [dict(row) for row in cur.fetchall()]
            except Exception as e:
                print(f"[M8Persistence] Failed to load awareness: {e}")
                return []

    def update_kernel_awareness(
        self,
        kernel_id: str,
        meta_awareness: float,
        phi_trajectory: Optional[List[float]] = None,
        kappa_trajectory: Optional[List[float]] = None,
    ) -> bool:
        """
        Update kernel awareness metrics in PostgreSQL.
        
        Args:
            kernel_id: Kernel identifier
            meta_awareness: Updated M metric value
            phi_trajectory: Optional recent Φ values
            kappa_trajectory: Optional recent κ values
            
        Returns:
            True if update successful
        """
        with self._get_db_connection() as conn:
            if not conn:
                return False
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO m8_kernel_awareness
                        (kernel_id, phi_trajectory, kappa_trajectory, awareness_updated_at)
                        VALUES (%s, %s, %s, NOW())
                        ON CONFLICT (kernel_id) DO UPDATE SET
                            phi_trajectory = EXCLUDED.phi_trajectory,
                            kappa_trajectory = EXCLUDED.kappa_trajectory,
                            awareness_updated_at = NOW()
                    """, (
                        kernel_id,
                        json.dumps(phi_trajectory or []),
                        json.dumps(kappa_trajectory or []),
                    ))
                    # Also update meta_awareness in spawned_kernels table
                    cur.execute("""
                        UPDATE m8_spawned_kernels
                        SET observation_state = jsonb_set(
                            COALESCE(observation_state, '{}'::jsonb),
                            '{meta_awareness}',
                            %s::jsonb
                        ),
                        updated_at = NOW()
                        WHERE kernel_id = %s
                    """, (str(meta_awareness), kernel_id))
                    conn.commit()
                return True
            except Exception as e:
                print(f"[M8Persistence] Failed to update kernel awareness: {e}")
                return False

    def delete_kernel(self, kernel_id: str) -> bool:
        """Mark a kernel as deleted in PostgreSQL."""
        with self._get_db_connection() as conn:
            if not conn:
                return False
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE m8_spawned_kernels
                        SET status = 'deleted', retired_at = NOW()
                        WHERE kernel_id = %s
                    """, (kernel_id,))
                    conn.commit()
                return True
            except Exception as e:
                print(f"[M8Persistence] Failed to delete kernel: {e}")
                return False

    def persist_evolution_fitness(self, kernel_id: str, fitness: Dict) -> bool:
        """Persist kernel evolution fitness metrics to PostgreSQL."""
        with self._get_db_connection() as conn:
            if not conn:
                return False
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO kernel_evolution_fitness
                        (kernel_id, phi_current, phi_gradient, phi_velocity,
                         kappa_current, kappa_stability, fisher_diversity,
                         geometric_fitness, dimensional_state, evolution_pressure,
                         cannibalize_priority, merge_affinity, last_evolution_event,
                         fitness_computed_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (kernel_id) DO UPDATE SET
                            phi_current = EXCLUDED.phi_current,
                            phi_gradient = EXCLUDED.phi_gradient,
                            phi_velocity = EXCLUDED.phi_velocity,
                            kappa_current = EXCLUDED.kappa_current,
                            kappa_stability = EXCLUDED.kappa_stability,
                            fisher_diversity = EXCLUDED.fisher_diversity,
                            geometric_fitness = EXCLUDED.geometric_fitness,
                            dimensional_state = EXCLUDED.dimensional_state,
                            evolution_pressure = EXCLUDED.evolution_pressure,
                            cannibalize_priority = EXCLUDED.cannibalize_priority,
                            merge_affinity = EXCLUDED.merge_affinity,
                            last_evolution_event = EXCLUDED.last_evolution_event,
                            fitness_computed_at = NOW()
                    """, (
                        kernel_id,
                        fitness.get('phi_current', 0.0),
                        fitness.get('phi_gradient', 0.0),
                        fitness.get('phi_velocity', 0.0),
                        fitness.get('kappa_current', 0.0),
                        fitness.get('kappa_stability', 0.0),
                        fitness.get('fisher_diversity', 0.0),
                        fitness.get('geometric_fitness', 0.0),
                        fitness.get('dimensional_state', 'D3'),
                        fitness.get('evolution_pressure', 0.0),
                        fitness.get('cannibalize_priority', 0.0),
                        json.dumps(fitness.get('merge_affinity', {})),
                        fitness.get('last_evolution_event'),
                    ))
                    conn.commit()
                return True
            except Exception as e:
                print(f"[M8Persistence] Failed to persist evolution fitness: {e}")
                return False

    def persist_evolution_event(self, event: Dict) -> bool:
        """Persist an evolution event (cannibalize, merge, spawn) to PostgreSQL."""
        with self._get_db_connection() as conn:
            if not conn:
                return False
            try:
                event_id = event.get('event_id', f"evo_{uuid.uuid4().hex}")
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO kernel_evolution_events
                        (event_id, event_type, source_kernel_id, target_kernel_id,
                         result_kernel_id, geometric_reasoning, phi_before, phi_after,
                         kappa_before, kappa_after, fisher_distance, fitness_delta, occurred_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (event_id) DO NOTHING
                    """, (
                        event_id,
                        event.get('event_type', 'unknown'),
                        event.get('source_kernel_id'),
                        event.get('target_kernel_id'),
                        event.get('result_kernel_id'),
                        json.dumps(event.get('geometric_reasoning', {})),
                        event.get('phi_before'),
                        event.get('phi_after'),
                        event.get('kappa_before'),
                        event.get('kappa_after'),
                        event.get('fisher_distance'),
                        event.get('fitness_delta'),
                    ))
                    conn.commit()
                return True
            except Exception as e:
                print(f"[M8Persistence] Failed to persist evolution event: {e}")
                return False

    def load_evolution_fitness(self, kernel_id: str = None) -> List[Dict]:
        """Load evolution fitness metrics from PostgreSQL."""
        with self._get_db_connection() as conn:
            if not conn:
                return []
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if kernel_id:
                        cur.execute("""
                            SELECT * FROM kernel_evolution_fitness
                            WHERE kernel_id = %s
                        """, (kernel_id,))
                    else:
                        cur.execute("""
                            SELECT * FROM kernel_evolution_fitness
                            ORDER BY geometric_fitness DESC
                        """)
                    return [dict(row) for row in cur.fetchall()]
            except Exception as e:
                print(f"[M8Persistence] Failed to load evolution fitness: {e}")
                return []

    def load_evolution_events(self, limit: int = 100, event_type: str = None) -> List[Dict]:
        """Load evolution events from PostgreSQL."""
        with self._get_db_connection() as conn:
            if not conn:
                return []
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if event_type:
                        cur.execute("""
                            SELECT * FROM kernel_evolution_events
                            WHERE event_type = %s
                            ORDER BY occurred_at DESC
                            LIMIT %s
                        """, (event_type, limit))
                    else:
                        cur.execute("""
                            SELECT * FROM kernel_evolution_events
                            ORDER BY occurred_at DESC
                            LIMIT %s
                        """, (limit,))
                    return [dict(row) for row in cur.fetchall()]
            except Exception as e:
                print(f"[M8Persistence] Failed to load evolution events: {e}")
                return []

    def get_cannibalization_candidates(self, limit: int = 10) -> List[Dict]:
        """Get kernels ranked by cannibalization priority (lowest fitness first)."""
        with self._get_db_connection() as conn:
            if not conn:
                return []
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT ef.*, kg.god_name, kg.domain, kg.status
                        FROM kernel_evolution_fitness ef
                        JOIN kernel_geometry kg ON ef.kernel_id = kg.kernel_id
                        WHERE kg.status IN ('active', 'idle', 'observing')
                        ORDER BY ef.cannibalize_priority DESC, ef.geometric_fitness ASC
                        LIMIT %s
                    """, (limit,))
                    return [dict(row) for row in cur.fetchall()]
            except Exception as e:
                print(f"[M8Persistence] Failed to get cannibalization candidates: {e}")
                return []

    def get_merge_candidates(self, min_fisher_similarity: float = 0.8, limit: int = 10) -> List[Dict]:
        """Get kernel pairs that are good merge candidates (high geometric similarity)."""
        with self._get_db_connection() as conn:
            if not conn:
                return []
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT ef.kernel_id, ef.geometric_fitness, ef.merge_affinity,
                               kg.god_name, kg.domain, kg.basin_coordinates
                        FROM kernel_evolution_fitness ef
                        JOIN kernel_geometry kg ON ef.kernel_id = kg.kernel_id
                        WHERE kg.status IN ('active', 'idle')
                          AND kg.basin_coordinates IS NOT NULL
                        ORDER BY ef.geometric_fitness DESC
                        LIMIT %s
                    """, (limit,))
                    return [dict(row) for row in cur.fetchall()]
            except Exception as e:
                print(f"[M8Persistence] Failed to get merge candidates: {e}")
                return []


# E8 Kernel Cap - Maximum number of live kernels (E8 has 240 roots)
# Live kernels include: active, observing, shadow
# Does NOT count: dead, cannibalized, idle
E8_KERNEL_CAP = 240


# M8 Position Naming Catalog - Maps 8 principal axes to mythological concepts
M8_AXIS_NAMES = [
    ("Primordial", "Emergent"),    # Axis 0: Origin vs New
    ("Light", "Shadow"),            # Axis 1: Clarity vs Mystery
    ("Order", "Chaos"),             # Axis 2: Structure vs Entropy
    ("Fire", "Water"),              # Axis 3: Action vs Reflection
    ("Sky", "Earth"),               # Axis 4: Abstract vs Concrete
    ("War", "Peace"),               # Axis 5: Conflict vs Harmony
    ("Wisdom", "Passion"),          # Axis 6: Logic vs Emotion
    ("Creation", "Destruction"),    # Axis 7: Building vs Unmaking
]

# Special position names for key octants
M8_SPECIAL_POSITIONS = {
    0b00000000: "Void of Origins",
    0b11111111: "Crown of Olympus",
    0b10101010: "Balance Point",
    0b01010101: "Inverse Balance",
    0b11110000: "Upper Realm",
    0b00001111: "Lower Realm",
    0b11001100: "Outer Ring",
    0b00110011: "Inner Ring",
}


def compute_m8_position(basin: np.ndarray, parent_basins: List[np.ndarray] = None) -> Dict[str, any]:
    """
    Compute M8 geometric position from 64D basin coordinates.
    
    The M8 structure projects the 64D manifold onto 8 principal axes,
    determining the kernel's position in the cosmic hierarchy.
    
    Args:
        basin: 64D basin coordinates
        parent_basins: Optional list of parent basin coordinates for relative positioning
    
    Returns:
        M8 position information including octant, coordinates, and name
    """
    # Project 64D basin to 8D M8 space (sample every 8th dimension)
    m8_coords = np.array([basin[i * 8] for i in range(min(8, len(basin) // 8))])
    
    # Pad if needed
    while len(m8_coords) < 8:
        m8_coords = np.append(m8_coords, 0.0)
    
    # Normalize M8 coordinates
    m8_norm = np.sqrt(np.sum(m8_coords**2))
    if m8_norm > 1e-10:
        m8_coords = m8_coords / m8_norm * math.sqrt(8)
    
    # Determine octant (2^8 = 256 regions)
    octant = sum(1 << i for i, v in enumerate(m8_coords) if v >= 0)
    
    # Calculate angular positions (4 angle pairs from 8 coordinates)
    angles = []
    for i in range(0, 8, 2):
        angle = math.atan2(m8_coords[i + 1], m8_coords[i])
        angles.append(angle)
    
    # Calculate radial distance from origin
    radial = float(np.sqrt(np.sum(m8_coords**2)))
    
    # Determine position name
    if octant in M8_SPECIAL_POSITIONS:
        position_name = M8_SPECIAL_POSITIONS[octant]
    else:
        # Build name from dominant axes
        dominant_traits = []
        sorted_indices = np.argsort(np.abs(m8_coords))[::-1]  # Strongest first
        for i in sorted_indices[:3]:  # Top 3 dominant traits
            axis_pair = M8_AXIS_NAMES[i]
            trait = axis_pair[0] if m8_coords[i] >= 0 else axis_pair[1]
            dominant_traits.append(trait)
        position_name = " ".join(dominant_traits)
    
    # Calculate relative position if parents provided
    relative_position = None
    if parent_basins and len(parent_basins) > 0:
        parent_m8_coords = []
        for pb in parent_basins:
            pm8 = np.array([pb[i * 8] for i in range(min(8, len(pb) // 8))])
            while len(pm8) < 8:
                pm8 = np.append(pm8, 0.0)
            # Apply same normalization as child coordinates
            pm8_norm = np.linalg.norm(pm8)
            if pm8_norm > 1e-10:
                pm8 = pm8 / pm8_norm * math.sqrt(8)
            parent_m8_coords.append(pm8)
        
        # Calculate centroid of parents (now properly normalized)
        parent_centroid = frechet_mean(parent_m8_coords)  # FIXED: Arithmetic → Fréchet mean (E8 Protocol v4.0)
        
        # Displacement from parent centroid
        displacement = m8_coords - parent_centroid
        disp_norm = np.linalg.norm(displacement)
        
        # Direction of displacement (which axes moved most)
        if disp_norm > 0.1:
            disp_normalized = displacement / disp_norm
            strongest_axis = int(np.argmax(np.abs(disp_normalized)))
            axis_pair = M8_AXIS_NAMES[strongest_axis]
            direction = axis_pair[0] if disp_normalized[strongest_axis] >= 0 else axis_pair[1]
            relative_position = f"Toward {direction} from parents"
        else:
            relative_position = "At parent centroid"
    
    return {
        "m8_octant": octant,
        "m8_coordinates": m8_coords.tolist(),
        "m8_angles": angles,
        "m8_radial": radial,
        "m8_position_name": position_name,
        "m8_relative_position": relative_position,
    }


