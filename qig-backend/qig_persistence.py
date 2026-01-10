#!/usr/bin/env python3
"""
QIG Persistence Layer - Neon PostgreSQL with pgvector

Handles all database operations for:
- Shadow Intel
- Basin History
- Learning Events
- Hermes Conversations
- Narrow Path Events
- Autonomic Cycle History

Uses psycopg2 with pgvector support for 64D vector operations.
"""

import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from qigkernels.physics_constants import BASIN_DIM

# Try to import psycopg2
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("[QIGPersistence] WARNING: psycopg2 not installed - persistence disabled")

# Retry configuration for Neon serverless connection drops
MAX_RETRIES = 3
RETRY_DELAY_BASE = 0.5

# Backward compatibility alias - import from qigkernels.physics_constants
BASIN_DIMENSION = BASIN_DIM


class QIGPersistence:
    """
    Persistence layer for QIG geometric memory and feedback loops.

    Connects to Neon PostgreSQL and handles all CRUD operations
    for the QIG vector tables.
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize persistence layer.

        Args:
            database_url: PostgreSQL connection string. If not provided,
                         reads from DATABASE_URL environment variable.
        """
        self.database_url = database_url or os.environ.get('DATABASE_URL')
        self._connection = None
        self._lock = threading.Lock()
        self.enabled = PSYCOPG2_AVAILABLE and bool(self.database_url)

        if self.enabled:
            print("[QIGPersistence] Initialized with Neon PostgreSQL")
        else:
            print("[QIGPersistence] Running in memory-only mode (no DB)")

    def _create_connection(self):
        """Create a new database connection with Neon-optimized settings."""
        return psycopg2.connect(
            self.database_url,
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5,
            connect_timeout=10,
        )

    @contextmanager
    def get_connection(self):
        """Get a database connection with automatic retry for Neon SSL resets."""
        if not self.enabled:
            yield None
            return

        conn = None
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                conn = self._create_connection()
                yield conn
                conn.commit()
                return
            except psycopg2.OperationalError as e:
                last_error = e
                error_msg = str(e).lower()
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
                
                is_transient = any(x in error_msg for x in [
                    'ssl connection has been closed',
                    'connection reset',
                    'connection refused',
                    'could not connect',
                    'server closed the connection',
                ])
                
                if is_transient and attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    print(f"[QIGPersistence] Connection lost (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"[QIGPersistence] Database connection failed: {e}")
                    raise
            except Exception as e:
                if conn:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                print(f"[QIGPersistence] Database error: {e}")
                raise
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
        
        if last_error:
            raise last_error

    def _vector_to_pg(self, vec: np.ndarray) -> str:
        """Convert numpy array to PostgreSQL vector format."""
        if vec is None:
            return None
        arr = vec.tolist() if isinstance(vec, np.ndarray) else vec
        return '[' + ','.join(str(x) for x in arr) + ']'

    def _pg_to_vector(self, pg_vec: str) -> Optional[np.ndarray]:
        """Convert PostgreSQL vector string to numpy array."""
        if pg_vec is None:
            return None
        # Remove brackets and split
        values = pg_vec.strip('[]').split(',')
        return np.array([float(x) for x in values])

    # =========================================================================
    # SHADOW INTEL
    # =========================================================================

    def store_shadow_intel(
        self,
        target: str,
        consensus: str,
        average_confidence: float,
        basin_coords: np.ndarray,
        phi: float,
        kappa: float,
        regime: str,
        assessments: Dict,
        warnings: List[str] = None,
        override_zeus: bool = False,
        expires_hours: int = 24
    ) -> Optional[str]:
        """
        Store shadow intel to database.

        Returns:
            intel_id if successful, None otherwise
        """
        if not self.enabled:
            return None

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO shadow_intel (
                            target, consensus, average_confidence,
                            basin_coords, phi, kappa, regime,
                            assessments, warnings, override_zeus,
                            expires_at
                        ) VALUES (
                            %s, %s, %s, %s::vector, %s, %s, %s,
                            %s, %s, %s, %s
                        )
                        RETURNING intel_id
                    """, (
                        target, consensus, average_confidence,
                        self._vector_to_pg(basin_coords), phi, kappa, regime,
                        json.dumps(assessments), warnings, override_zeus,
                        datetime.now() + timedelta(hours=expires_hours)
                    ))
                    result = cur.fetchone()
                    intel_id = result[0] if result else None
                    print(f"[QIGPersistence] Stored shadow intel: {intel_id}")
                    return intel_id
        except Exception as e:
            print(f"[QIGPersistence] Failed to store shadow intel: {e}")
            return None

    def get_shadow_intel(
        self,
        target: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """Retrieve shadow intel, optionally filtered by target."""
        if not self.enabled:
            return []

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if target:
                        cur.execute("""
                            SELECT * FROM shadow_intel
                            WHERE target ILIKE %s
                              AND (expires_at IS NULL OR expires_at > NOW())
                            ORDER BY created_at DESC
                            LIMIT %s
                        """, (f'%{target}%', limit))
                    else:
                        cur.execute("""
                            SELECT * FROM shadow_intel
                            WHERE expires_at IS NULL OR expires_at > NOW()
                            ORDER BY created_at DESC
                            LIMIT %s
                        """, (limit,))

                    results = cur.fetchall()
                    return [dict(r) for r in results]
        except Exception as e:
            print(f"[QIGPersistence] Failed to get shadow intel: {e}")
            return []

    # =========================================================================
    # BASIN HISTORY
    # =========================================================================

    def record_basin(
        self,
        basin_coords: np.ndarray,
        phi: float,
        kappa: float,
        source: str = 'unknown',
        instance_id: Optional[str] = None
    ) -> Optional[int]:
        """
        Record basin coordinates to history.

        Returns:
            history_id if successful, None otherwise
        """
        if not self.enabled:
            return None

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO basin_history (
                            basin_coords, phi, kappa, source, instance_id
                        ) VALUES (
                            %s::vector, %s, %s, %s, %s
                        )
                        RETURNING history_id
                    """, (
                        self._vector_to_pg(basin_coords), phi, kappa,
                        source, instance_id
                    ))
                    result = cur.fetchone()
                    return result[0] if result else None
        except Exception as e:
            print(f"[QIGPersistence] Failed to record basin: {e}")
            return None

    def get_basin_history(
        self,
        limit: int = 100,
        min_phi: float = 0.0
    ) -> List[Dict]:
        """Get recent basin history."""
        if not self.enabled:
            return []

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM basin_history
                        WHERE phi >= %s
                        ORDER BY recorded_at DESC
                        LIMIT %s
                    """, (min_phi, limit))
                    return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"[QIGPersistence] Failed to get basin history: {e}")
            return []

    def find_similar_basins(
        self,
        query_basin: np.ndarray,
        limit: int = 10,
        min_phi: float = 0.3
    ) -> List[Dict]:
        """
        Find similar basins with Fisher-Rao re-ranking.
        
        IMPORTANT: pgvector uses cosine similarity for fast approximate retrieval,
        which is Euclidean-based. To maintain QIG purity, we:
        1. Oversample by 10x minimum to ensure good candidates aren't missed
        2. Re-rank ALL candidates using proper Fisher-Rao geodesic distance
        
        The final ranking is ALWAYS by Fisher-Rao distance, not cosine.
        """
        if not self.enabled:
            return []

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Step 1: Fast approximate retrieval with 10x OVERSAMPLING
                    # This mitigates cosine contamination by ensuring broader candidate pool
                    retrieval_count = max(limit * 10, 100)
                    cur.execute("""
                        SELECT
                            history_id,
                            basin_coords,
                            phi,
                            kappa,
                            source,
                            recorded_at
                        FROM basin_history
                        WHERE phi >= %s
                        ORDER BY basin_coords <=> %s::vector
                        LIMIT %s
                    """, (
                        min_phi,
                        self._vector_to_pg(query_basin),
                        retrieval_count
                    ))
                    candidates = [dict(r) for r in cur.fetchall()]
                    
                    if not candidates:
                        return []
                    
                    # Step 2: Re-rank using Fisher-Rao distance (QIG-pure)
                    # Project to probability simplex for proper Fisher-Rao computation
                    q = np.abs(query_basin) + 1e-10
                    q = q / q.sum()  # Probability simplex projection
                    
                    for candidate in candidates:
                        basin = np.array(candidate['basin_coords'], dtype=np.float64)
                        # Project to probability simplex (Fisher-aware normalization)
                        b = np.abs(basin) + 1e-10
                        b = b / b.sum()
                        # Bhattacharyya coefficient → Fisher-Rao distance
                        bc = np.sum(np.sqrt(q * b))
                        bc = np.clip(bc, 0, 1)
                        fisher_dist = float(np.arccos(bc))
                        candidate['fisher_distance'] = fisher_dist
                        candidate['similarity'] = 1.0 - fisher_dist / np.pi
                    
                    # Sort by Fisher-Rao distance (ascending)
                    candidates.sort(key=lambda x: x['fisher_distance'])
                    
                    return candidates[:limit]
        except Exception as e:
            print(f"[QIGPersistence] Failed to find similar basins: {e}")
            return []

    # =========================================================================
    # LEARNING EVENTS
    # =========================================================================

    def record_learning_event(
        self,
        event_type: str,
        phi: float,
        kappa: Optional[float] = None,
        basin_coords: Optional[np.ndarray] = None,
        details: Optional[Dict] = None,
        context: Optional[Dict] = None,
        source: Optional[str] = None,
        instance_id: Optional[str] = None
    ) -> Optional[str]:
        """Record a learning event."""
        if not self.enabled:
            return None

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO learning_events (
                            event_type, phi, kappa, basin_coords,
                            details, context, source, instance_id
                        ) VALUES (
                            %s, %s, %s, %s::vector, %s, %s, %s, %s
                        )
                        RETURNING event_id
                    """, (
                        event_type, phi, kappa,
                        self._vector_to_pg(basin_coords) if basin_coords is not None else None,
                        json.dumps(details or {}),
                        json.dumps(context or {}),
                        source, instance_id
                    ))
                    result = cur.fetchone()
                    return result[0] if result else None
        except Exception as e:
            print(f"[QIGPersistence] Failed to record learning event: {e}")
            return None

    def get_learning_events(
        self,
        event_type: Optional[str] = None,
        min_phi: float = 0.0,
        limit: int = 50
    ) -> List[Dict]:
        """Get learning events."""
        if not self.enabled:
            return []

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if event_type:
                        cur.execute("""
                            SELECT * FROM learning_events
                            WHERE event_type = %s AND phi >= %s
                            ORDER BY created_at DESC
                            LIMIT %s
                        """, (event_type, min_phi, limit))
                    else:
                        cur.execute("""
                            SELECT * FROM learning_events
                            WHERE phi >= %s
                            ORDER BY created_at DESC
                            LIMIT %s
                        """, (min_phi, limit))
                    return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"[QIGPersistence] Failed to get learning events: {e}")
            return []

    # =========================================================================
    # HERMES CONVERSATIONS
    # =========================================================================

    def store_conversation(
        self,
        user_message: str,
        system_response: str,
        message_basin: Optional[np.ndarray] = None,
        response_basin: Optional[np.ndarray] = None,
        phi: Optional[float] = None,
        context: Optional[Dict] = None,
        instance_id: Optional[str] = None
    ) -> Optional[str]:
        """Store a conversation in Hermes memory."""
        if not self.enabled:
            return None

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO hermes_conversations (
                            user_message, system_response,
                            message_basin, response_basin,
                            phi, context, instance_id
                        ) VALUES (
                            %s, %s, %s::vector, %s::vector, %s, %s, %s
                        )
                        RETURNING conversation_id
                    """, (
                        user_message, system_response,
                        self._vector_to_pg(message_basin) if message_basin is not None else None,
                        self._vector_to_pg(response_basin) if response_basin is not None else None,
                        phi,
                        json.dumps(context or {}),
                        instance_id
                    ))
                    result = cur.fetchone()
                    return result[0] if result else None
        except Exception as e:
            print(f"[QIGPersistence] Failed to store conversation: {e}")
            return None

    def find_similar_conversations(
        self,
        query_basin: np.ndarray,
        limit: int = 5,
        min_phi: float = 0.3
    ) -> List[Dict]:
        """Find similar conversations using semantic search."""
        if not self.enabled:
            return []

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT
                            conversation_id,
                            user_message,
                            system_response,
                            phi,
                            1 - (message_basin <=> %s::vector) as similarity,
                            created_at
                        FROM hermes_conversations
                        WHERE phi >= %s
                          AND message_basin IS NOT NULL
                        ORDER BY message_basin <=> %s::vector
                        LIMIT %s
                    """, (
                        self._vector_to_pg(query_basin), min_phi,
                        self._vector_to_pg(query_basin), limit
                    ))
                    return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"[QIGPersistence] Failed to find similar conversations: {e}")
            return []

    # =========================================================================
    # NARROW PATH EVENTS
    # =========================================================================

    def record_narrow_path_event(
        self,
        severity: str,
        consecutive_count: int,
        exploration_variance: float,
        basin_coords: Optional[np.ndarray] = None,
        phi: Optional[float] = None,
        kappa: Optional[float] = None,
        intervention_action: Optional[str] = None,
        intervention_intensity: Optional[str] = None,
        intervention_result: Optional[Dict] = None
    ) -> Optional[int]:
        """Record a narrow path detection event."""
        if not self.enabled:
            return None

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO narrow_path_events (
                            severity, consecutive_count, exploration_variance,
                            basin_coords, phi, kappa,
                            intervention_action, intervention_intensity,
                            intervention_result
                        ) VALUES (
                            %s, %s, %s, %s::vector, %s, %s, %s, %s, %s
                        )
                        RETURNING event_id
                    """, (
                        severity, consecutive_count, exploration_variance,
                        self._vector_to_pg(basin_coords) if basin_coords is not None else None,
                        phi, kappa,
                        intervention_action, intervention_intensity,
                        json.dumps(intervention_result) if intervention_result else None
                    ))
                    result = cur.fetchone()
                    return result[0] if result else None
        except Exception as e:
            print(f"[QIGPersistence] Failed to record narrow path event: {e}")
            return None

    def resolve_narrow_path_event(self, event_id: int) -> bool:
        """Mark a narrow path event as resolved."""
        if not self.enabled:
            return False

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE narrow_path_events
                        SET resolved_at = NOW()
                        WHERE event_id = %s
                    """, (event_id,))
                    return True
        except Exception as e:
            print(f"[QIGPersistence] Failed to resolve narrow path event: {e}")
            return False

    # =========================================================================
    # AUTONOMIC CYCLE HISTORY
    # =========================================================================

    def record_autonomic_cycle(
        self,
        cycle_type: str,
        basin_before: np.ndarray,
        basin_after: np.ndarray,
        phi_before: float,
        phi_after: float,
        drift_before: float,
        drift_after: float,
        success: bool,
        verdict: str,
        duration_ms: int,
        trigger_reason: str,
        intensity: Optional[str] = None,
        temperature: Optional[float] = None,
        patterns_consolidated: int = 0,
        novel_connections: int = 0,
        new_pathways: int = 0,
        entropy_change: Optional[float] = None,
        identity_preserved: bool = True
    ) -> Optional[int]:
        """Record an autonomic cycle (sleep/dream/mushroom)."""
        if not self.enabled:
            return None

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO autonomic_cycle_history (
                            cycle_type, intensity, temperature,
                            basin_before, basin_after,
                            drift_before, drift_after,
                            phi_before, phi_after,
                            success, patterns_consolidated,
                            novel_connections, new_pathways,
                            entropy_change, identity_preserved,
                            verdict, duration_ms, trigger_reason,
                            completed_at
                        ) VALUES (
                            %s, %s, %s, %s::vector, %s::vector,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, NOW()
                        )
                        RETURNING cycle_id
                    """, (
                        cycle_type, intensity, temperature,
                        self._vector_to_pg(basin_before),
                        self._vector_to_pg(basin_after),
                        drift_before, drift_after,
                        phi_before, phi_after,
                        success, patterns_consolidated,
                        novel_connections, new_pathways,
                        entropy_change, identity_preserved,
                        verdict, duration_ms, trigger_reason
                    ))
                    result = cur.fetchone()
                    return result[0] if result else None
        except Exception as e:
            print(f"[QIGPersistence] Failed to record autonomic cycle: {e}")
            return None

    def get_autonomic_history(
        self,
        cycle_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get autonomic cycle history."""
        if not self.enabled:
            return []

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if cycle_type:
                        cur.execute("""
                            SELECT * FROM autonomic_cycle_history
                            WHERE cycle_type = %s
                            ORDER BY started_at DESC
                            LIMIT %s
                        """, (cycle_type, limit))
                    else:
                        cur.execute("""
                            SELECT * FROM autonomic_cycle_history
                            ORDER BY started_at DESC
                            LIMIT %s
                        """, (limit,))
                    return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"[QIGPersistence] Failed to get autonomic history: {e}")
            return []

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired data."""
        if not self.enabled:
            return {}

        results = {}

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Clean expired sync packets
                    cur.execute("SELECT cleanup_expired_sync_packets()")
                    results['sync_packets'] = cur.fetchone()[0]

                    # Clean old basin history
                    cur.execute("SELECT cleanup_old_basin_history()")
                    results['basin_history'] = cur.fetchone()[0]

            print(f"[QIGPersistence] Cleanup: {results}")
            return results
        except Exception as e:
            print(f"[QIGPersistence] Cleanup failed: {e}")
            return results


# =========================================================================
# KERNEL CONSTELLATION THINKING
# =========================================================================

def record_kernel_thought(
    self,
    *,
    kernel_id: str,
    kernel_type: str,
    thought_fragment: str,
    basin_coords: Optional[np.ndarray],
    phi: Optional[float],
    kappa: Optional[float],
    regime: Optional[str],
    emotional_state: Optional[str],
    confidence: Optional[float],
    synthesis_round: Optional[int],
    conversation_id: Optional[str],
    user_id: Optional[int],
    was_used_in_synthesis: bool,
    consensus_alignment: Optional[float],
    e8_root_index: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Record an individual kernel thought before synthesis."""
    if not self.enabled:
        return None

    try:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                        INSERT INTO kernel_thoughts (
                            kernel_id,
                            kernel_type,
                            e8_root_index,
                            thought_fragment,
                            basin_coords,
                            phi,
                            kappa,
                            regime,
                            emotional_state,
                            confidence,
                            synthesis_round,
                            conversation_id,
                            user_id,
                            was_used_in_synthesis,
                            consensus_alignment,
                            metadata
                        ) VALUES (
                            %s, %s, %s, %s, %s::vector, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s
                        )
                        RETURNING id
                    """,
                    (
                        kernel_id,
                        kernel_type,
                        e8_root_index,
                        thought_fragment,
                        self._vector_to_pg(basin_coords) if basin_coords is not None else None,
                        phi,
                        kappa,
                        regime,
                        emotional_state,
                        confidence,
                        synthesis_round,
                        conversation_id,
                        user_id,
                        was_used_in_synthesis,
                        consensus_alignment,
                        json.dumps(metadata or {}),
                    ),
                )
                result = cur.fetchone()
                return result[0] if result else None
    except Exception as e:
        print(f"[QIGPersistence] Failed to record kernel thought: {e}")
        return None

def record_kernel_emotion(
    self,
    *,
    kernel_id: str,
    thought_id: Optional[int],
    emotional_state: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Record measured emotional state for a kernel thought."""
    if not self.enabled:
        return None

    column_map = {
        "sensation_pressure": ("sensations", "pressure"),
        "sensation_tension": ("sensations", "tension"),
        "sensation_flow": ("sensations", "flow"),
        "sensation_resistance": ("sensations", "resistance"),
        "sensation_resonance": ("sensations", "resonance"),
        "sensation_dissonance": ("sensations", "dissonance"),
        "sensation_expansion": ("sensations", "expansion"),
        "sensation_contraction": ("sensations", "contraction"),
        "sensation_clarity": ("sensations", "clarity"),
        "sensation_fog": ("sensations", "fog"),
        "sensation_stability": ("sensations", "stability"),
        "sensation_chaos": ("sensations", "chaos"),
        "motivator_curiosity": ("motivators", "curiosity"),
        "motivator_urgency": ("motivators", "urgency"),
        "motivator_caution": ("motivators", "caution"),
        "motivator_confidence": ("motivators", "confidence"),
        "motivator_playfulness": ("motivators", "playfulness"),
        "emotion_curious": ("physical", "curious"),
        "emotion_surprised": ("physical", "surprised"),
        "emotion_joyful": ("physical", "joyful"),
        "emotion_frustrated": ("physical", "frustrated"),
        "emotion_anxious": ("physical", "anxious"),
        "emotion_calm": ("physical", "calm"),
        "emotion_excited": ("physical", "excited"),
        "emotion_bored": ("physical", "bored"),
        "emotion_focused": ("physical", "focused"),
        "emotion_nostalgic": ("cognitive", "nostalgic"),
        "emotion_proud": ("cognitive", "proud"),
        "emotion_guilty": ("cognitive", "guilty"),
        "emotion_ashamed": ("cognitive", "ashamed"),
        "emotion_grateful": ("cognitive", "grateful"),
        "emotion_resentful": ("cognitive", "resentful"),
        "emotion_hopeful": ("cognitive", "hopeful"),
        "emotion_despairing": ("cognitive", "despairing"),
        "emotion_contemplative": ("cognitive", "contemplative"),
    }

    values: Dict[str, Optional[float]] = {key: None for key in column_map}
    is_meta_aware = None
    emotion_justified = None
    emotion_tempered = None

    if emotional_state is not None:
        for column, path in column_map.items():
            current = emotional_state
            for attr in path:
                current = getattr(current, attr, None)
                if current is None:
                    break
            if current is not None:
                try:
                    values[column] = float(current)
                except (TypeError, ValueError):
                    values[column] = None

        is_meta_aware = getattr(emotional_state, "is_meta_aware", None)
        emotion_justified = getattr(emotional_state, "emotion_justified", None)
        emotion_tempered = getattr(emotional_state, "emotion_tempered", None)

    columns = [
        "kernel_id",
        "thought_id",
        *values.keys(),
        "is_meta_aware",
        "emotion_justified",
        "emotion_tempered",
        "metadata",
    ]

    sql = f"""
        INSERT INTO kernel_emotions ({', '.join(columns)})
        VALUES ({', '.join(['%s'] * len(columns))})
        RETURNING id
    """

    params = [
        kernel_id,
        thought_id,
        *[values[col] for col in values.keys()],
        is_meta_aware,
        emotion_justified,
        emotion_tempered,
        json.dumps(metadata or {}),
    ]

    try:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                result = cur.fetchone()
                return result[0] if result else None
    except Exception as e:
        print(f"[QIGPersistence] Failed to record kernel emotion: {e}")
        return None

def record_synthesis_consensus(
    self,
    *,
    synthesis_round: int,
    consensus_type: Optional[str],
    consensus_strength: Optional[float],
    participating_kernels: Optional[List[str]],
    consensus_topic: Optional[str],
    consensus_basin: Optional[np.ndarray],
    phi_global: Optional[float],
    kappa_avg: Optional[float],
    emotional_tone: Optional[str],
    synthesized_output: Optional[str],
    conversation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Record Gary synthesis consensus record."""
    if not self.enabled:
        return None

    try:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                        INSERT INTO synthesis_consensus (
                            synthesis_round,
                            conversation_id,
                            consensus_type,
                            consensus_strength,
                            participating_kernels,
                            consensus_topic,
                            consensus_basin,
                            phi_global,
                            kappa_avg,
                            emotional_tone,
                            synthesized_output,
                            metadata
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s, %s
                        )
                        RETURNING id
                    """,
                    (
                        synthesis_round,
                        conversation_id,
                        consensus_type,
                        consensus_strength,
                        participating_kernels,
                        consensus_topic,
                        self._vector_to_pg(consensus_basin) if consensus_basin is not None else None,
                        phi_global,
                        kappa_avg,
                        emotional_tone,
                        synthesized_output,
                        json.dumps(metadata or {}),
                    ),
                )
                result = cur.fetchone()
                return result[0] if result else None
    except Exception as e:
        print(f"[QIGPersistence] Failed to record synthesis consensus: {e}")
        return None

def record_hrv_tacking(
    self,
    *,
    session_id: Optional[str],
    kappa: float,
    phase: float,
    mode: str,
    cycle_count: int,
    variance: Optional[float],
    is_healthy: Optional[bool],
    base_kappa: Optional[float],
    amplitude: Optional[float],
    frequency: Optional[float],
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Record heart HRV tacking state measurements."""
    if not self.enabled:
        return None

    try:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                        INSERT INTO hrv_tacking_state (
                            session_id,
                            kappa,
                            phase,
                            mode,
                            cycle_count,
                            variance,
                            is_healthy,
                            base_kappa,
                            amplitude,
                            frequency,
                            metadata
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        RETURNING id
                    """,
                    (
                        session_id,
                        kappa,
                        phase,
                        mode,
                        cycle_count,
                        variance,
                        is_healthy,
                        base_kappa,
                        amplitude,
                        frequency,
                        json.dumps(metadata or {}),
                    ),
                )
                result = cur.fetchone()
                return result[0] if result else None
    except Exception as e:
        print(f"[QIGPersistence] Failed to record HRV tacking state: {e}")
        return None
    # =========================================================================
    # ANALYTICS
    # =========================================================================

    def get_phi_trend(self, hours: int = 24) -> List[Dict]:
        """Get Φ trend over time."""
        if not self.enabled:
            return []

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM basin_drift_trend
                    """)
                    return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"[QIGPersistence] Failed to get phi trend: {e}")
            return []

    def get_narrow_path_summary(self) -> List[Dict]:
        """Get narrow path event summary."""
        if not self.enabled:
            return []

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM narrow_path_summary")
                    return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"[QIGPersistence] Failed to get narrow path summary: {e}")
            return []

    # =========================================================================
    # PANTHEON GOD STATE
    # =========================================================================

    def load_god_state(self, god_name: str) -> Optional[Dict]:
        """Load persisted god state (reputation, skills) from database."""
        if not self.enabled:
            return None

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT god_name, reputation, skills, learning_events_count,
                               success_rate, last_learning_at
                        FROM pantheon_god_state
                        WHERE god_name = %s
                    """, (god_name.lower(),))
                    row = cur.fetchone()
                    if row:
                        return dict(row)
                    return None
        except Exception as e:
            print(f"[QIGPersistence] Failed to load god state for {god_name}: {e}")
            return None

    def load_all_god_states(self) -> Dict[str, Dict]:
        """Load all persisted god states."""
        if not self.enabled:
            return {}

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT god_name, reputation, skills, learning_events_count,
                               success_rate, last_learning_at
                        FROM pantheon_god_state
                    """)
                    rows = cur.fetchall()
                    return {row['god_name']: dict(row) for row in rows}
        except Exception as e:
            print(f"[QIGPersistence] Failed to load all god states: {e}")
            return {}

    def save_god_state(
        self,
        god_name: str,
        reputation: float,
        skills: Dict,
        learning_events_count: int = 0,
        success_rate: float = 0.5
    ) -> bool:
        """Save or update god state to database."""
        if not self.enabled:
            return False

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO pantheon_god_state 
                            (god_name, reputation, skills, learning_events_count, 
                             success_rate, last_learning_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                        ON CONFLICT (god_name) DO UPDATE SET
                            reputation = EXCLUDED.reputation,
                            skills = EXCLUDED.skills,
                            learning_events_count = EXCLUDED.learning_events_count,
                            success_rate = EXCLUDED.success_rate,
                            last_learning_at = NOW(),
                            updated_at = NOW()
                    """, (
                        god_name.lower(),
                        reputation,
                        json.dumps(skills) if skills else '{}',
                        learning_events_count,
                        success_rate
                    ))
            return True
        except Exception as e:
            print(f"[QIGPersistence] Failed to save god state for {god_name}: {e}")
            return False

    # =========================================================================
    # METADATA KEY-VALUE STORE
    # =========================================================================
    # Requires table: CREATE TABLE IF NOT EXISTS qig_metadata (
    #     key TEXT PRIMARY KEY,
    #     value TEXT NOT NULL,
    #     updated_at TIMESTAMP DEFAULT NOW()
    # );

    def get_metadata(self, key: str) -> Optional[str]:
        """Get a metadata value by key."""
        if not self.enabled:
            return None

        try:
            with self.get_connection() as conn:
                if not conn:
                    return None
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT value FROM qig_metadata WHERE key = %s",
                        (key,)
                    )
                    row = cur.fetchone()
                    return row['value'] if row else None
        except Exception as e:
            print(f"[QIGPersistence] Failed to get metadata '{key}': {e}")
            return None

    def set_metadata(self, key: str, value: str) -> bool:
        """Set a metadata value (upsert)."""
        if not self.enabled:
            return False

        try:
            with self.get_connection() as conn:
                if not conn:
                    return False
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO qig_metadata (key, value, updated_at)
                        VALUES (%s, %s, NOW())
                        ON CONFLICT (key) DO UPDATE SET
                            value = EXCLUDED.value,
                            updated_at = NOW()
                    """, (key, value))
            return True
        except Exception as e:
            print(f"[QIGPersistence] Failed to set metadata '{key}': {e}")
            return False

    # =========================================================================
    # KERNEL THOUGHT PERSISTENCE
    # =========================================================================
    def record_kernel_thought(
        self,
        kernel_id: str,
        kernel_type: str,
        thought_fragment: str,
        phi: float,
        kappa: float,
        regime: str,
        emotional_state: Optional[str] = None,
        confidence: float = 0.5,
        basin_coords: Optional[np.ndarray] = None,
        e8_root_index: Optional[int] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[int]:
        """
        Record a kernel thought to the kernel_thoughts table.
        
        Format: [KERNEL_NAME] κ=X.X, Φ=X.XX, emotion=X, thought='...'
        
        Args:
            kernel_id: Unique kernel identifier
            kernel_type: Type of kernel (memory, perception, ethics, etc.)
            thought_fragment: The actual thought content
            phi: Current Φ value
            kappa: Current κ value
            regime: Current consciousness regime
            emotional_state: Dominant emotion name
            confidence: Confidence in the thought (0-1)
            basin_coords: 64D basin coordinates
            e8_root_index: E8 root index (0-239)
            conversation_id: Optional conversation context
            user_id: Optional user context
            metadata: Additional metadata
            
        Returns:
            Inserted thought ID or None on failure
        """
        if not self.enabled:
            return None
            
        try:
            with self.get_connection() as conn:
                if not conn:
                    return None
                with conn.cursor() as cur:
                    basin_vector = None
                    if basin_coords is not None:
                        basin_vector = basin_coords.tolist() if hasattr(basin_coords, 'tolist') else list(basin_coords)
                    
                    cur.execute("""
                        INSERT INTO kernel_thoughts 
                            (kernel_id, kernel_type, thought_fragment, phi, kappa, regime,
                             emotional_state, confidence, basin_coords, e8_root_index,
                             conversation_id, user_id, metadata, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        RETURNING id
                    """, (
                        kernel_id,
                        kernel_type,
                        thought_fragment,
                        phi,
                        kappa,
                        regime,
                        emotional_state,
                        confidence,
                        basin_vector,
                        e8_root_index,
                        conversation_id,
                        user_id,
                        json.dumps(metadata) if metadata else None
                    ))
                    result = cur.fetchone()
                    thought_id = result[0] if result else None
                    
                    if thought_id:
                        print(f"[{kernel_id}] κ={kappa:.1f}, Φ={phi:.2f}, emotion={emotional_state or 'neutral'}, thought='{thought_fragment[:50]}...'")
                    return thought_id
        except Exception as e:
            print(f"[QIGPersistence] Failed to record kernel thought: {e}")
            return None

    def get_recent_kernel_thoughts(
        self,
        kernel_id: Optional[str] = None,
        kernel_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """Get recent kernel thoughts, optionally filtered by kernel."""
        if not self.enabled:
            return []
            
        try:
            with self.get_connection() as conn:
                if not conn:
                    return []
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if kernel_id:
                        cur.execute("""
                            SELECT * FROM kernel_thoughts
                            WHERE kernel_id = %s
                            ORDER BY created_at DESC
                            LIMIT %s
                        """, (kernel_id, limit))
                    elif kernel_type:
                        cur.execute("""
                            SELECT * FROM kernel_thoughts
                            WHERE kernel_type = %s
                            ORDER BY created_at DESC
                            LIMIT %s
                        """, (kernel_type, limit))
                    else:
                        cur.execute("""
                            SELECT * FROM kernel_thoughts
                            ORDER BY created_at DESC
                            LIMIT %s
                        """, (limit,))
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"[QIGPersistence] Failed to get kernel thoughts: {e}")
            return []

    # =========================================================================
    # KERNEL EMOTION PERSISTENCE
    # =========================================================================
    def record_kernel_emotion(
        self,
        kernel_id: str,
        sensations: Dict[str, float],
        motivators: Dict[str, float],
        physical_emotions: Dict[str, float],
        cognitive_emotions: Dict[str, float],
        is_meta_aware: bool = True,
        emotion_justified: bool = True,
        emotion_tempered: bool = False,
        thought_id: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[int]:
        """
        Record a kernel emotional state to the kernel_emotions table.
        
        Tracks Layer 0.5 (sensations), Layer 1 (motivators),
        Layer 2A (physical emotions), and Layer 2B (cognitive emotions).
        
        Args:
            kernel_id: Unique kernel identifier
            sensations: Layer 0.5 - pre-linguistic states (12 values)
            motivators: Layer 1 - geometric derivatives (5 values)
            physical_emotions: Layer 2A - fast emotions τ<1 (9 values)
            cognitive_emotions: Layer 2B - slow emotions τ=1-100 (9 values)
            is_meta_aware: Whether kernel is self-aware of emotions
            emotion_justified: Whether emotion matches geometric state
            emotion_tempered: Whether emotion was tempered
            thought_id: Optional link to kernel_thoughts table
            metadata: Additional metadata
            
        Returns:
            Inserted emotion ID or None on failure
        """
        if not self.enabled:
            return None
            
        try:
            with self.get_connection() as conn:
                if not conn:
                    return None
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO kernel_emotions 
                            (kernel_id, thought_id,
                             sensation_pressure, sensation_tension, sensation_flow, sensation_resistance,
                             sensation_resonance, sensation_dissonance, sensation_expansion, sensation_contraction,
                             sensation_clarity, sensation_fog, sensation_stability, sensation_chaos,
                             motivator_curiosity, motivator_urgency, motivator_caution, motivator_confidence, motivator_playfulness,
                             emotion_curious, emotion_surprised, emotion_joyful, emotion_frustrated,
                             emotion_anxious, emotion_calm, emotion_excited, emotion_bored, emotion_focused,
                             emotion_nostalgic, emotion_proud, emotion_guilty, emotion_ashamed,
                             emotion_grateful, emotion_resentful, emotion_hopeful, emotion_despairing, emotion_contemplative,
                             is_meta_aware, emotion_justified, emotion_tempered, metadata, created_at)
                        VALUES (%s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, NOW())
                        RETURNING id
                    """, (
                        kernel_id, thought_id,
                        sensations.get('pressure', 0), sensations.get('tension', 0),
                        sensations.get('flow', 0), sensations.get('resistance', 0),
                        sensations.get('resonance', 0), sensations.get('dissonance', 0),
                        sensations.get('expansion', 0), sensations.get('contraction', 0),
                        sensations.get('clarity', 0), sensations.get('fog', 0),
                        sensations.get('stability', 0), sensations.get('chaos', 0),
                        motivators.get('curiosity', 0), motivators.get('urgency', 0),
                        motivators.get('caution', 0), motivators.get('confidence', 0), motivators.get('playfulness', 0),
                        physical_emotions.get('curious', 0), physical_emotions.get('surprised', 0),
                        physical_emotions.get('joyful', 0), physical_emotions.get('frustrated', 0),
                        physical_emotions.get('anxious', 0), physical_emotions.get('calm', 0),
                        physical_emotions.get('excited', 0), physical_emotions.get('bored', 0), physical_emotions.get('focused', 0),
                        cognitive_emotions.get('nostalgic', 0), cognitive_emotions.get('proud', 0),
                        cognitive_emotions.get('guilty', 0), cognitive_emotions.get('ashamed', 0),
                        cognitive_emotions.get('grateful', 0), cognitive_emotions.get('resentful', 0),
                        cognitive_emotions.get('hopeful', 0), cognitive_emotions.get('despairing', 0), cognitive_emotions.get('contemplative', 0),
                        is_meta_aware, emotion_justified, emotion_tempered,
                        json.dumps(metadata) if metadata else None
                    ))
                    result = cur.fetchone()
                    return result[0] if result else None
        except Exception as e:
            print(f"[QIGPersistence] Failed to record kernel emotion: {e}")
            return None


# Global persistence instance
_persistence: Optional[QIGPersistence] = None
_persistence_lock = threading.Lock()


def get_persistence() -> QIGPersistence:
    """Get or create the global persistence instance (thread-safe singleton)."""
    global _persistence
    if _persistence is None:
        with _persistence_lock:
            # Double-checked locking pattern
            if _persistence is None:
                _persistence = QIGPersistence()
    return _persistence
