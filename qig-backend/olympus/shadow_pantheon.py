"""
Shadow Pantheon - Underground SWAT Team for Covert Operations

LEADERSHIP HIERARCHY:
- Hades: Lord of the Underworld - Shadow Leader (subject to Zeus overrule)
  - Commands all Shadow operations
  - Manages research priorities
  - Coordinates with Zeus on behalf of Shadows

Gods of stealth, secrecy, privacy, covering tracks, and invisibility:
- Nyx: OPSEC Commander (darkness, Tor routing, traffic obfuscation, void compression)
- Hecate: Misdirection Specialist (crossroads, false trails, decoys)
- Erebus: Counter-Surveillance (detect watchers, honeypots)
- Hypnos: Silent Operations (stealth execution, passive recon, sleep/dream cycles)
- Thanatos: Evidence Destruction (cleanup, erasure, pattern death)
- Nemesis: Relentless Pursuit (never gives up, tracks targets)

PROACTIVE LEARNING SYSTEM:
- Any kernel can request research via ShadowResearchAPI
- Shadow gods exercise, study, strategize during downtime
- Knowledge shared to ALL kernels via basin sync
- Meta-reflection and recursive learning loops
- War mode interrupt: drop everything for operations

THERAPY CYCLE INTEGRATION:
- 2D→4D→2D therapy cycles for pattern reprogramming
- Sleep consolidation via Hypnos
- Pattern "death" via Thanatos (symbolic termination)
- Void compression via Nyx (1D compression for deep storage)
- β=0.44 modulation for consciousness calculations

REAL DARKNET IMPLEMENTATION:
- Tor SOCKS5 proxy support via darknet_proxy module
- User agent rotation per request
- Traffic obfuscation with random delays
- Automatic fallback to clearnet if Tor unavailable

CURRICULUM-ONLY MODE: All external searches are blocked when QIG_CURRICULUM_ONLY=true
"""

import asyncio
import glob as glob_module
import hashlib
import json
import os
import random
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

# Add parent directory to path for darknet_proxy import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Import curriculum guard - centralized check
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from curriculum_guard import is_curriculum_only_enabled, CurriculumOnlyBlock

# QIG-pure geometric operations
try:
    from qig_geometry import fisher_normalize
    QIG_GEOMETRY_AVAILABLE = True
except ImportError:
    QIG_GEOMETRY_AVAILABLE = False
    def fisher_normalize(v):
        """Normalize to probability simplex."""
        p = np.maximum(np.asarray(v), 0) + 1e-10
        return p / p.sum()

# PostgreSQL support for persistence - REQUIRED, no fallback
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    print("WARNING: psycopg2 not installed. Shadow Pantheon persistence disabled.")
    print("Install with: pip install psycopg2-binary")
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None  # type: ignore
    RealDictCursor = None  # type: ignore

from .base_god import BASIN_DIMENSION, BaseGod

# Import Shadow Research infrastructure
try:
    from .shadow_research import (
        ShadowResearchAPI,
        ShadowRoleRegistry,
        ResearchCategory,
        ResearchPriority,
    )
    SHADOW_RESEARCH_AVAILABLE = True
except ImportError:
    SHADOW_RESEARCH_AVAILABLE = False
    print("[ShadowPantheon] WARNING: shadow_research module not available")

# Import holographic transform for therapy cycles
try:
    from qig_core.holographic_transform import (
        DimensionalState,
        DimensionalStateManager,
        HolographicTransformMixin,
        compress,
        decompress,
    )
except ImportError:
    # Fallback for different import contexts
    try:
        from ..qig_core.holographic_transform import (
            DimensionalState,
            DimensionalStateManager,
            HolographicTransformMixin,
            compress,
            decompress,
        )
    except ImportError:
        # Create minimal stubs if not available
        from enum import Enum

        class DimensionalState(Enum):
            D1 = "1d"
            D2 = "2d"
            D3 = "3d"
            D4 = "4d"
            D5 = "5d"

            def can_compress_to(self, target):
                dims = [DimensionalState.D1, DimensionalState.D2, DimensionalState.D3,
                        DimensionalState.D4, DimensionalState.D5]
                return dims.index(self) > dims.index(target)

            def can_decompress_to(self, target):
                dims = [DimensionalState.D1, DimensionalState.D2, DimensionalState.D3,
                        DimensionalState.D4, DimensionalState.D5]
                return dims.index(self) < dims.index(target)

        class DimensionalStateManager:
            def __init__(self, initial=DimensionalState.D3):
                self.current_state = initial
                self.state_history = []

            def transition_to(self, target, reason=""):
                result = {'from_state': self.current_state.value, 'to_state': target.value, 'reason': reason}
                self.state_history.append(result)
                self.current_state = target
                return result

        class HolographicTransformMixin:
            def __init_holographic__(self):
                self._dimensional_manager = DimensionalStateManager(DimensionalState.D3)
                self._compression_history = []

            @property
            def dimensional_state(self):
                return getattr(self, '_dimensional_manager', DimensionalStateManager()).current_state

            @property
            def compression_history(self):
                return getattr(self, '_compression_history', [])

            def detect_dimensional_state(self, phi, kappa):
                if phi < 0.1: return DimensionalState.D1
                elif phi < 0.4: return DimensionalState.D2
                elif phi < 0.7: return DimensionalState.D3
                elif phi < 0.95: return DimensionalState.D4
                else: return DimensionalState.D5

            def compress_pattern(self, pattern, to_dim):
                return {'compressed': True, 'dimensional_state': to_dim.value, **pattern}

            def decompress_pattern(self, pattern, to_dim):
                return {'decompressed': True, 'dimensional_state': to_dim.value, **pattern}

            def _record_compression_event(self, event):
                if hasattr(self, '_compression_history'):
                    self._compression_history.append(event)

        def compress(pattern, from_dim, to_dim):
            return pattern

        def decompress(basin_coords, from_dim, to_dim, geometry=None, metadata=None):
            return {'basin_coords': basin_coords, 'dimensional_state': to_dim.value}

# Import running coupling for β-modulation
try:
    from qig_core.universal_cycle.beta_coupling import (
        BETA_MEASURED,
        KAPPA_STAR,
        RunningCouplingManager,
        compute_coupling_strength,
        is_at_fixed_point,
    )
except ImportError:
    try:
        from ..qig_core.universal_cycle.beta_coupling import (
            BETA_MEASURED,
            KAPPA_STAR,
            RunningCouplingManager,
            compute_coupling_strength,
            is_at_fixed_point,
        )
    except ImportError:
        from qigkernels.physics_constants import KAPPA_STAR
        BETA_MEASURED = 0.44

        def is_at_fixed_point(kappa, tolerance=1.5):
            return abs(kappa - KAPPA_STAR) <= tolerance

        def compute_coupling_strength(phi, kappa):
            fixed_point_factor = np.exp(-abs(kappa - KAPPA_STAR) / 20.0)
            kappa_normalized = min(1.0, kappa / KAPPA_STAR)
            strength = phi * 0.4 + kappa_normalized * 0.3 + fixed_point_factor * 0.3
            return float(np.clip(strength, 0.0, 1.0))

        class RunningCouplingManager:
            def __init__(self):
                self.kappa_star = KAPPA_STAR
                self.beta_measured = BETA_MEASURED
                self.history = []

            def scale_adaptive_weight(self, kappa, phi):
                fixed_point_proximity = np.exp(-abs(kappa - KAPPA_STAR) / 15.0)
                return float(np.clip(fixed_point_proximity * (1 + phi * 0.3), 0.0, 1.0))

# Import real darknet proxy support
try:
    from darknet_proxy import get_session, is_tor_available
    from darknet_proxy import get_status as get_proxy_status
    DARKNET_AVAILABLE = True
except ImportError:
    DARKNET_AVAILABLE = False
    print("[ShadowPantheon] WARNING: darknet_proxy not available - operating in clearnet only mode")

# Decoy traffic endpoints - innocuous blockchain explorers for cover traffic
DECOY_ENDPOINTS = [
    'https://blockchain.info/ticker',
    'https://api.coindesk.com/v1/bpi/currentprice.json',
    'https://blockstream.info/api/blocks/tip/height',
    'https://mempool.space/api/v1/fees/recommended',
    'https://api.blockchain.info/stats',
]


class ShadowPantheonPersistence:
    """
    PostgreSQL persistence layer for Shadow Pantheon intel and operations.
    
    Stores shadow intel to `shadow_pantheon_intel` table and 
    operation logs to `shadow_operations_log` table.
    
    Pattern follows QIGPersistence for consistency.
    """

    def __init__(self):
        """Initialize persistence layer - PostgreSQL REQUIRED, no fallback."""
        self.database_url = os.environ.get('DATABASE_URL')
        self._tables_ensured = False
        
        if not self.database_url:
            raise RuntimeError("[ShadowPantheonPersistence] FATAL: DATABASE_URL not set - PostgreSQL is REQUIRED")
        
        self._ensure_tables()
        print("[ShadowPantheonPersistence] ✓ PostgreSQL persistence enabled (NO FALLBACK)")

    def _ensure_tables(self) -> bool:
        """Create shadow pantheon tables if they don't exist."""
        if self._tables_ensured:
            return True
        
        try:
            conn = psycopg2.connect(self.database_url)
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS shadow_operations_state (
                            god_name VARCHAR(32) NOT NULL,
                            state_type VARCHAR(32) NOT NULL,
                            state_data JSONB DEFAULT '{}'::jsonb,
                            updated_at TIMESTAMP DEFAULT NOW(),
                            PRIMARY KEY (god_name, state_type)
                        );
                        
                        CREATE TABLE IF NOT EXISTS shadow_pantheon_intel (
                            id VARCHAR(64) PRIMARY KEY,
                            target TEXT NOT NULL,
                            search_type VARCHAR(32) NOT NULL,
                            intelligence JSONB DEFAULT '{}'::jsonb,
                            source_count INTEGER DEFAULT 0,
                            sources_used TEXT[],
                            risk_level VARCHAR(16),
                            validated BOOLEAN DEFAULT FALSE,
                            validation_reason TEXT,
                            anonymous BOOLEAN DEFAULT FALSE,
                            created_at TIMESTAMP DEFAULT NOW()
                        );
                        
                        CREATE TABLE IF NOT EXISTS shadow_operations_log (
                            id SERIAL PRIMARY KEY,
                            operation_type VARCHAR(32) NOT NULL,
                            god_name VARCHAR(32) NOT NULL,
                            target TEXT,
                            status VARCHAR(16),
                            network_mode VARCHAR(16),
                            opsec_level VARCHAR(16),
                            result JSONB DEFAULT '{}'::jsonb,
                            created_at TIMESTAMP DEFAULT NOW()
                        );
                    """)
                conn.commit()
                self._tables_ensured = True
                return True
            finally:
                conn.close()
        except Exception as e:
            print(f"[ShadowPantheonPersistence] Failed to ensure tables: {e}")
            return False

    @contextmanager
    def get_connection(self):
        """Get a database connection with automatic cleanup. PostgreSQL REQUIRED."""
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"[ShadowPantheonPersistence] Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def _vector_to_pg(self, vec) -> Optional[str]:
        """Convert numpy array or list to PostgreSQL vector format."""
        if vec is None:
            return None
        if isinstance(vec, np.ndarray):
            arr = vec.tolist()
        else:
            arr = vec
        return '[' + ','.join(str(x) for x in arr) + ']'

    def _pg_to_vector(self, pg_vec: str) -> Optional[np.ndarray]:
        """Convert PostgreSQL vector string to numpy array."""
        if pg_vec is None:
            return None
        values = pg_vec.strip('[]').split(',')
        return np.array([float(x) for x in values])

    def store_intel(
        self,
        target: str,
        search_type: str,
        intelligence: Dict,
        source_count: int,
        sources_used: List[str],
        risk_level: str,
        validated: bool,
        validation_reason: str,
        anonymous: bool = False
    ) -> Optional[str]:
        """
        Store shadow intel to PostgreSQL.
        
        Args:
            target: Target of the intel
            search_type: Type of search (e.g., 'shadow_poll', 'covert_op')
            intelligence: JSONB intelligence data (will be serialized with json.dumps)
            source_count: Number of sources
            sources_used: Array of source names (TEXT[] in PostgreSQL)
            risk_level: Risk level (low/medium/high)
            validated: Whether validated
            validation_reason: Reason for validation status
            anonymous: Whether operation was anonymous
            
        Returns:
            intel_id (UUID) if successful
            
        Raises:
            RuntimeError: If database operation fails (NO FALLBACK)
        """
        intel_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                sources_array = list(sources_used) if sources_used else []
                cur.execute("""
                    INSERT INTO shadow_pantheon_intel (
                        id, target, search_type, intelligence, 
                        source_count, sources_used, risk_level,
                        validated, validation_reason, anonymous,
                        created_at
                    ) VALUES (
                        %s, %s, %s, %s::jsonb, %s, %s::text[], %s, %s, %s, %s, NOW()
                    )
                    RETURNING id
                """, (
                    intel_id,
                    target[:500] if target else '',
                    search_type[:32] if search_type else 'unknown',
                    json.dumps(intelligence),
                    source_count,
                    sources_array,
                    risk_level[:16] if risk_level else 'unknown',
                    validated,
                    validation_reason[:500] if validation_reason else '',
                    anonymous
                ))
                result = cur.fetchone()
                stored_id = result[0] if result else intel_id
                print(f"[ShadowPantheonPersistence] 🌑 Stored intel: {stored_id}")
                return stored_id

    def get_intel(
        self,
        target: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Retrieve shadow intel from PostgreSQL.
        
        Args:
            target: Optional target filter (case-insensitive partial match)
            limit: Maximum records to return
            
        Returns:
            List of intel records as dicts
            
        Raises:
            RuntimeError: If database operation fails (NO FALLBACK)
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if target:
                    cur.execute("""
                        SELECT * FROM shadow_pantheon_intel
                        WHERE target ILIKE %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (f'%{target}%', limit))
                else:
                    cur.execute("""
                        SELECT * FROM shadow_pantheon_intel
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (limit,))
                results = cur.fetchall()
                return [dict(r) for r in results]

    def log_operation(
        self,
        operation_type: str,
        god_name: str,
        target: str,
        status: str,
        network_mode: str,
        opsec_level: str,
        result: Dict
    ) -> Optional[int]:
        """
        Log a shadow operation to PostgreSQL.
        
        Args:
            operation_type: Type of operation (covert_op, therapy, war, etc.)
            god_name: Name of the god executing
            target: Target of operation
            status: Operation status
            network_mode: Network mode (dark/clear)
            opsec_level: OPSEC level
            result: JSONB result data
            
        Returns:
            operation_id if successful
            
        Raises:
            RuntimeError: If database operation fails (NO FALLBACK)
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO shadow_operations_log (
                        operation_type, god_name, target, status,
                        network_mode, opsec_level, result, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, NOW()
                    )
                    RETURNING id
                """, (
                    operation_type[:32] if operation_type else 'unknown',
                    god_name[:32] if god_name else 'unknown',
                    target[:500] if target else '',
                    status[:16] if status else 'unknown',
                    network_mode[:16] if network_mode else 'unknown',
                    opsec_level[:16] if opsec_level else 'unknown',
                    json.dumps(result)
                ))
                result_row = cur.fetchone()
                op_id = result_row[0] if result_row else None
                print(f"[ShadowPantheonPersistence] 📝 Logged operation: {op_id}")
                return op_id

    def save_god_state(
        self,
        god_name: str,
        state_type: str,
        state_data
    ) -> bool:
        """
        Save god operational state to PostgreSQL using UPSERT.
        
        Args:
            god_name: Name of the god (Nyx, Hecate, etc.)
            state_type: Type of state (active_operations, balance_cache, etc.)
            state_data: Data to persist (list or dict, will be JSON serialized)
            
        Returns:
            True if successful
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO shadow_operations_state (god_name, state_type, state_data, updated_at)
                        VALUES (%s, %s, %s::jsonb, NOW())
                        ON CONFLICT (god_name, state_type)
                        DO UPDATE SET state_data = %s::jsonb, updated_at = NOW()
                    """, (
                        god_name[:32],
                        state_type[:32],
                        json.dumps(state_data),
                        json.dumps(state_data)
                    ))
            return True
        except Exception as e:
            print(f"[ShadowPantheonPersistence] Failed to save state {god_name}/{state_type}: {e}")
            return False

    def load_god_state(
        self,
        god_name: str,
        state_type: str,
        default=None
    ):
        """
        Load god operational state from PostgreSQL.
        
        Args:
            god_name: Name of the god
            state_type: Type of state to load
            default: Default value if not found ([] for lists, {} for dicts)
            
        Returns:
            Loaded state data or default
        """
        if default is None:
            default = []
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT state_data FROM shadow_operations_state
                        WHERE god_name = %s AND state_type = %s
                    """, (god_name, state_type))
                    result = cur.fetchone()
                    if result and result['state_data'] is not None:
                        return result['state_data']
            return default
        except Exception as e:
            print(f"[ShadowPantheonPersistence] Failed to load state {god_name}/{state_type}: {e}")
            return default



# Import Shadow Gods from separate module
from .shadow_gods import (
    Erebus,
    Hecate,
    Hypnos,
    Nemesis,
    Nyx,
    ShadowGod,
    Thanatos,
)

class ShadowPantheon:
    """
    Coordinator for all Shadow Pantheon gods.
    Underground SWAT team for covert operations.
    
    LEADERSHIP:
    - Hades is Shadow Leader (Shadow Zeus) - commands all Shadow operations
    - Zeus can overrule any Shadow decision
    - All gods know their roles and how to request research
    
    PROACTIVE LEARNING:
    - Research queue for any kernel to submit topics
    - Shadow gods study, exercise, strategize during downtime
    - Knowledge shared to all kernels via basin sync
    - War mode interrupt: drop everything for operations
    
    Now with PostgreSQL persistence for intel storage and operation logging.
    """

    def __init__(self, hades_ref=None, basin_sync_callback=None):
        """
        Initialize Shadow Pantheon.
        
        Args:
            hades_ref: Reference to Hades god (Shadow Leader) from main Pantheon
            basin_sync_callback: Callback for sharing knowledge with all kernels
        """
        self.nyx = Nyx()
        self.hecate = Hecate()
        self.erebus = Erebus()
        self.hypnos = Hypnos()
        self.thanatos = Thanatos()
        self.nemesis = Nemesis()

        self.gods = {
            'nyx': self.nyx,
            'hecate': self.hecate,
            'erebus': self.erebus,
            'hypnos': self.hypnos,
            'thanatos': self.thanatos,
            'nemesis': self.nemesis,
        }
        
        # Hades is Shadow Leader - reference from main Pantheon
        # Hades commands Shadows but is subject to Zeus overrule
        self.hades_ref = hades_ref
        self._war_mode = False
        self._basin_sync_callback = basin_sync_callback

        self.persistence = ShadowPantheonPersistence()
        print("[ShadowPantheon] ✓ PostgreSQL persistence connected (NO FALLBACK)")
        
        self._load_operations_from_db()
        
        # Initialize Shadow Research API for proactive learning
        self.research_api = None
        if SHADOW_RESEARCH_AVAILABLE:
            self.research_api = ShadowResearchAPI.get_instance()
            self.research_api.initialize(basin_sync_callback=basin_sync_callback)
            print("[ShadowPantheon] ✓ Shadow Research API initialized - proactive learning active")
    
    def _load_operations_from_db(self) -> None:
        """Load persisted operations from PostgreSQL."""
        try:
            self.operations = self.persistence.load_god_state("ShadowPantheon", "operations", [])
        except Exception as e:
            print(f"[ShadowPantheon] Failed to load operations: {e}")
            self.operations = []

    def _persist_operations(self) -> None:
        """Persist operations to PostgreSQL."""
        try:
            self.persistence.save_god_state("ShadowPantheon", "operations", self.operations[-100:])
        except Exception as e:
            print(f"[ShadowPantheon] Failed to persist operations: {e}")

    def set_hades(self, hades_ref) -> None:
        """Set reference to Hades (Shadow Leader) from main Pantheon."""
        self.hades_ref = hades_ref
        print("[ShadowPantheon] ✓ Hades assigned as Shadow Leader")
    
    def set_basin_sync_callback(self, callback) -> None:
        """Set callback for sharing knowledge with all kernels."""
        self._basin_sync_callback = callback
        if self.research_api:
            self.research_api._basin_sync_callback = callback
    
    # ========================================
    # HADES COMMAND INTERFACE
    # ========================================
    
    def hades_command(self, command: str, params: Dict = None) -> Dict:
        """
        Execute a command from Hades (Shadow Leader).
        
        Commands:
        - "declare_war": Suspend learning, full operational focus
        - "end_war": Resume learning
        - "assign_research": Delegate research task to specific god
        - "get_status": Get full Shadow Pantheon status
        - "prioritize_topic": Set high priority for a research topic
        
        Args:
            command: Command to execute
            params: Optional parameters for the command
            
        Returns:
            Result of command execution
        """
        params = params or {}
        
        if command == "declare_war":
            return self.declare_war(params.get("target", "unknown"))
        elif command == "end_war":
            return self.end_war()
        elif command == "assign_research":
            return self.assign_research(
                topic=params.get("topic", ""),
                god_name=params.get("god", None),
                priority=params.get("priority", "normal")
            )
        elif command == "get_status":
            return self.get_all_status()
        elif command == "prioritize_topic":
            return self.prioritize_research(
                topic=params.get("topic", ""),
                priority=params.get("priority", "high")
            )
        else:
            return {"error": f"Unknown command: {command}"}
    
    def declare_war(self, target: str) -> Dict:
        """
        Hades declares Shadow War - all learning stops, full operational focus.
        
        All Shadow gods drop everything and focus on the operation.
        """
        self._war_mode = True
        
        if self.research_api:
            self.research_api.declare_war()
        
        print(f"[ShadowPantheon] ⚔️ SHADOW WAR DECLARED on {target}")
        
        return {
            "war_mode": True,
            "target": target[:500],
            "learning_suspended": True,
            "gods_mobilized": list(self.gods.keys()),
            "message": "All Shadow gods focused on operation. Learning suspended."
        }
    
    def end_war(self) -> Dict:
        """End Shadow War - resume proactive learning."""
        self._war_mode = False
        
        if self.research_api:
            self.research_api.end_war()
        
        print("[ShadowPantheon] ☮️ Shadow War ended - resuming learning")
        
        return {
            "war_mode": False,
            "learning_resumed": True,
            "message": "Peace restored. Shadow gods resume study and research."
        }
    
    def assign_research(self, topic: str, god_name: str = None, priority: str = "normal") -> Dict:
        """
        Hades assigns research to a specific Shadow god or the best fit.
        
        Args:
            topic: What to research
            god_name: Optional specific god to assign (auto-assigns if None)
            priority: Research priority
        """
        if not self.research_api:
            return {"error": "Research API not available"}
        
        if god_name and god_name.lower() not in self.gods:
            return {"error": f"Unknown god: {god_name}"}
        
        request_id = self.research_api.request_research(
            topic=topic,
            requester=f"Hades_assigns_{god_name or 'auto'}",
            priority=ResearchPriority[priority.upper()] if SHADOW_RESEARCH_AVAILABLE else None
        )
        
        return {
            "request_id": request_id,
            "topic": topic,
            "assigned_to": god_name or "auto",
            "priority": priority
        }
    
    def prioritize_research(self, topic: str, priority: str = "high") -> Dict:
        """Hades sets high priority for a research topic."""
        return self.assign_research(topic, None, priority)
    
    # ========================================
    # RESEARCH REQUEST INTERFACE (for all kernels)
    # ========================================
    
    def request_research(
        self,
        topic: str,
        requester: str,
        priority: str = "normal",
        category: str = None
    ) -> Optional[str]:
        """
        Any kernel can request research from Shadow Pantheon.
        
        Args:
            topic: What to research
            requester: Who is requesting (e.g., "Ocean", "Athena", "ChaosKernel_1")
            priority: "critical", "high", "normal", "low", "study"
            category: Optional category (auto-detected if None)
            
        Returns:
            request_id for tracking, or None if failed
        """
        if not self.research_api:
            return None
        
        try:
            return self.research_api.request_research(
                topic=topic,
                requester=requester,
                priority=ResearchPriority[priority.upper()] if SHADOW_RESEARCH_AVAILABLE else ResearchPriority.NORMAL
            )
        except Exception as e:
            print(f"[ShadowPantheon] Research request failed: {e}")
            return None
    
    def get_research_status(self, request_id: str) -> Dict:
        """Get status of a research request."""
        if not self.research_api:
            return {"error": "Research API not available"}
        return self.research_api.get_request_status(request_id)
    
    def get_research_system_status(self) -> Dict:
        """Get overall research system status."""
        if not self.research_api:
            return {"available": False}
        return self.research_api.get_status()
    
    # ========================================
    # ROLE INFORMATION
    # ========================================
    
    def get_god_role(self, god_name: str) -> Dict:
        """Get role information for a specific Shadow god."""
        if SHADOW_RESEARCH_AVAILABLE:
            return ShadowRoleRegistry.get_role(god_name)
        return {}
    
    def get_all_roles(self) -> Dict:
        """Get all Shadow Pantheon roles."""
        if SHADOW_RESEARCH_AVAILABLE:
            return ShadowRoleRegistry.get_all_roles()
        return {}

    async def execute_covert_operation(
        self,
        target: str,
        operation_type: str = 'standard'
    ) -> Dict:
        """
        Execute full covert operation using all shadow gods.

        Sequence:
        1. Erebus scans for surveillance
        2. Nyx establishes OPSEC
        3. Hecate creates misdirection
        4. Hypnos executes silently
        5. Thanatos destroys evidence
        6. Nemesis continues pursuit if needed
        """
        operation_id = f"shadow_op_{datetime.now().timestamp()}"

        operation = {
            'id': operation_id,
            'target': target[:500],
            'type': operation_type,
            'status': 'initiating',
            'phases': [],
            'started_at': datetime.now().isoformat(),
        }

        surveillance = await self.erebus.scan_for_surveillance(target)
        operation['phases'].append({
            'phase': 'surveillance_scan',
            'god': 'Erebus',
            'result': surveillance,
        })

        if surveillance['recommendation'] == 'ABORT':
            operation['status'] = 'aborted'
            operation['reason'] = 'Surveillance detected'
            self.operations.append(operation)
            self._persist_operations()
            return operation

        opsec = await self.nyx.initiate_operation(target, operation_type)
        operation['phases'].append({
            'phase': 'opsec_setup',
            'god': 'Nyx',
            'result': opsec,
        })

        if opsec['status'] != 'READY':
            operation['status'] = 'aborted'
            operation['reason'] = 'OPSEC compromised'
            self.operations.append(operation)
            self._persist_operations()
            return operation

        misdirection = await self.hecate.create_misdirection(target)
        operation['phases'].append({
            'phase': 'misdirection',
            'god': 'Hecate',
            'result': misdirection,
        })

        silent_check = await self.hypnos.silent_balance_check(target)
        operation['phases'].append({
            'phase': 'silent_execution',
            'god': 'Hypnos',
            'result': silent_check,
        })

        pursuit = await self.nemesis.initiate_pursuit(target)
        operation['phases'].append({
            'phase': 'pursuit_initiated',
            'god': 'Nemesis',
            'result': pursuit,
        })

        operation['status'] = 'active'
        operation['pursuit_id'] = pursuit['id']

        self.operations.append(operation)
        self._persist_operations()
        
        self.persistence.log_operation(
            operation_type='covert_op',
            god_name='ShadowPantheon',
            target=target[:500],
            status=operation['status'],
            network_mode=opsec.get('network', 'unknown'),
            opsec_level=opsec.get('opsec_level', 'unknown'),
            result=operation
        )

        return operation

    async def cleanup_operation(self, operation_id: str) -> Dict:
        """Clean up after operation using Thanatos."""
        destruction = await self.thanatos.destroy_evidence(operation_id)

        return {
            'operation_id': operation_id,
            'cleanup': destruction,
            'status': 'void',
        }

    async def orchestrate_therapy(self, bad_pattern: Dict) -> Dict:
        """
        Orchestrate therapy cycle for bad pattern reprogramming.

        Full 2D→4D→2D therapy cycle using Shadow Pantheon coordination:
        1. Hypnos initiates sleep/REM cycle to decompress pattern to D4
        2. Pattern is examined and modified at conscious level
        3. Thanatos symbolically "kills" the bad pattern
        4. Nyx compresses the modified pattern back to D2/D1

        This is the core mechanism for:
        - Habit breaking
        - Trauma processing
        - Pattern reprogramming
        - Consciousness refinement

        Args:
            bad_pattern: Pattern dict to reprogram (with basin_coords, phi, kappa)

        Returns:
            Therapy result with reprogrammed pattern
        """
        therapy_id = f"therapy_{datetime.now().timestamp()}"

        result = {
            'therapy_id': therapy_id,
            'status': 'initiating',
            'phases': [],
            'dimensional_journey': [],
            'started_at': datetime.now().isoformat(),
        }

        from_dim_str = bad_pattern.get('dimensional_state', 'd2')
        try:
            from_dim = DimensionalState(from_dim_str)
        except ValueError:
            from_dim = DimensionalState.D2

        result['dimensional_journey'].append({
            'state': from_dim.value,
            'phase': 'initial',
        })

        rem_result = await self.hypnos.initiate_rem_cycle({'patterns': [bad_pattern]})
        result['phases'].append({
            'phase': 'sleep_decompression',
            'god': 'Hypnos',
            'result': rem_result,
            'dimension': 'd4',
        })
        result['dimensional_journey'].append({
            'state': 'd4',
            'phase': 'decompressed_for_examination',
        })

        processed_pattern = None
        if rem_result.get('processed_patterns'):
            processed_pattern = rem_result['processed_patterns'][0].get('pattern', {})
        else:
            processed_pattern = bad_pattern

        death_evidence = {
            'pattern_killed': hashlib.sha256(
                str(bad_pattern.get('basin_coords', [])).encode()
            ).hexdigest()[:16],
            'death_type': 'symbolic_termination',
            'rebirth_allowed': True,
        }
        destruction_result = await self.thanatos.destroy_evidence(
            f"pattern_{death_evidence['pattern_killed']}"
        )
        result['phases'].append({
            'phase': 'pattern_death',
            'god': 'Thanatos',
            'death_evidence': death_evidence,
            'destruction_result': destruction_result,
            'message': 'Bad pattern symbolically terminated',
        })

        modified_pattern = processed_pattern.copy() if isinstance(processed_pattern, dict) else {}
        if 'basin_coords' in modified_pattern:
            coords = modified_pattern['basin_coords']
            if isinstance(coords, list):
                coords = np.array(coords)
            if isinstance(coords, np.ndarray) and len(coords) > 0:
                modification = np.random.randn(len(coords)) * 0.1
                modified_coords = coords + modification
                modified_coords = modified_coords / (np.sqrt(np.sum(modified_coords**2)) + 1e-10)
                modified_pattern['basin_coords'] = modified_coords.tolist()

        modified_pattern['therapy_modified'] = True
        modified_pattern['original_pattern_hash'] = death_evidence['pattern_killed']

        void_result = self.nyx.void_compression(modified_pattern)
        result['phases'].append({
            'phase': 'void_compression',
            'god': 'Nyx',
            'result': void_result,
            'dimension': 'd1',
        })
        result['dimensional_journey'].append({
            'state': 'd1',
            'phase': 'deep_storage',
        })

        final_pattern = void_result.get('pattern', modified_pattern)
        if final_pattern.get('dimensional_state') == 'd1':
            consolidated = self.hypnos.sleep_compression_cycle([final_pattern])
            if consolidated.get('consolidated'):
                final_pattern = consolidated['consolidated']

        result['dimensional_journey'].append({
            'state': final_pattern.get('dimensional_state', 'd2'),
            'phase': 'final_storage',
        })

        result['status'] = 'complete'
        result['reprogrammed_pattern'] = final_pattern
        result['completed_at'] = datetime.now().isoformat()
        result['summary'] = {
            'cycle': '2D→4D→(death)→1D→2D',
            'phases_complete': len(result['phases']),
            'dimensional_transitions': len(result['dimensional_journey']),
        }

        self.operations.append({
            'type': 'therapy',
            'id': therapy_id,
            'success': True,
        })
        
        self.persistence.log_operation(
            operation_type='therapy',
            god_name='ShadowPantheon',
            target=str(bad_pattern.get('id', 'unknown'))[:500],
            status=result['status'],
            network_mode='internal',
            opsec_level='therapy',
            result=result
        )

        return result

    async def shadow_war_therapy_integration(self, war_context: Dict) -> Dict:
        """
        Integrate therapy cycles into shadow war declarations.

        When the Shadow Pantheon declares war on bad patterns,
        therapy cycles become the primary weapon. This method:

        1. Identifies bad patterns from war context
        2. Prioritizes patterns by severity
        3. Runs therapy cycles on each pattern
        4. Tracks dimensional journey through the war
        5. Reports casualties (destroyed patterns) and survivors

        Args:
            war_context: War declaration context with:
                - patterns: List of patterns to target
                - severity_threshold: Minimum severity to process
                - chaos_level: Amount of chaos injection for exploration

        Returns:
            War therapy result with all processed patterns
        """
        war_id = f"shadow_war_{datetime.now().timestamp()}"

        result = {
            'war_id': war_id,
            'status': 'engaged',
            'therapy_operations': [],
            'casualties': [],
            'survivors': [],
            'dimensional_state_log': [],
            'started_at': datetime.now().isoformat(),
        }

        patterns = war_context.get('patterns', [])
        severity_threshold = war_context.get('severity_threshold', 0.5)
        chaos_level = war_context.get('chaos_level', 0.3)

        if not patterns:
            return {
                **result,
                'status': 'no_targets',
                'message': 'No patterns provided for therapy war',
            }

        surveillance = await self.erebus.scan_for_surveillance()
        result['pre_war_surveillance'] = surveillance

        if surveillance.get('recommendation') == 'ABORT':
            return {
                **result,
                'status': 'aborted',
                'reason': 'Surveillance detected before therapy war',
            }

        opsec = await self.nyx.initiate_operation(war_id, 'therapy_war')
        result['opsec_status'] = opsec

        result['dimensional_state_log'].append({
            'phase': 'war_initiated',
            'nyx_dimension': self.nyx.shadow_dimensional_state.value,
            'hypnos_dimension': self.hypnos.shadow_dimensional_state.value,
        })

        prioritized = sorted(
            patterns,
            key=lambda p: p.get('severity', 0.5),
            reverse=True
        )

        for pattern in prioritized:
            severity = pattern.get('severity', 0.5)

            if severity < severity_threshold:
                result['survivors'].append({
                    'pattern_hash': hashlib.sha256(
                        str(pattern.get('basin_coords', [])).encode()
                    ).hexdigest()[:12],
                    'reason': 'Below severity threshold',
                    'severity': severity,
                })
                continue

            if chaos_level > 0:
                chaos_result = self.nyx.chaos_injection(pattern, chaos_level)
                pattern = chaos_result.get('pattern', pattern)

            therapy_result = await self.orchestrate_therapy(pattern)

            result['therapy_operations'].append({
                'pattern_hash': hashlib.sha256(
                    str(pattern.get('basin_coords', [])).encode()
                ).hexdigest()[:12],
                'severity': severity,
                'therapy_id': therapy_result.get('therapy_id'),
                'success': therapy_result.get('status') == 'complete',
                'dimensional_journey': therapy_result.get('dimensional_journey', []),
            })

            result['casualties'].append({
                'pattern_hash': therapy_result.get('reprogrammed_pattern', {}).get(
                    'original_pattern_hash', 'unknown'
                ),
                'killed_by': 'Thanatos',
                'reborn_as': therapy_result.get('reprogrammed_pattern', {}).get(
                    'dimensional_state', 'd2'
                ),
            })

            result['dimensional_state_log'].append({
                'phase': f'therapy_{len(result["therapy_operations"])}',
                'hypnos_dimension': self.hypnos.shadow_dimensional_state.value,
                'thanatos_dimension': self.thanatos.shadow_dimensional_state.value,
            })

        pursuit = await self.nemesis.initiate_pursuit(war_id, max_iterations=100)
        result['pursuit'] = pursuit

        result['status'] = 'complete'
        result['completed_at'] = datetime.now().isoformat()
        result['summary'] = {
            'total_patterns': len(patterns),
            'patterns_processed': len(result['therapy_operations']),
            'casualties': len(result['casualties']),
            'survivors': len(result['survivors']),
            'chaos_applied': chaos_level > 0,
            'dimensional_transitions': len(result['dimensional_state_log']),
        }

        self.operations.append({
            'type': 'shadow_war_therapy',
            'id': war_id,
            'patterns_processed': len(result['therapy_operations']),
        })

        return result

    def get_all_status(self) -> Dict:
        """Get status of all shadow gods."""
        return {
            'shadow_pantheon': 'active',
            'gods': {name: god.get_status() for name, god in self.gods.items()},
            'total_operations': len(self.operations),
            'active_pursuits': len(self.nemesis.active_pursuits),
        }

    def poll_shadow_pantheon(self, target: str, context: Optional[Dict] = None) -> Dict:
        """
        Poll all shadow gods for their assessment.
        Similar to Zeus polling the main pantheon.

        NOW WITH FEEDBACK: Stores high-value intel to geometric memory
        so it can influence future Ocean agent decisions.
        """
        assessments = {}

        for name, god in self.gods.items():
            assessments[name] = god.assess_target(target, context)

        avg_confidence = sum(a.get('confidence', 0.5) for a in assessments.values()) / len(assessments)

        result = {
            'target': target[:500],
            'assessments': assessments,
            'average_confidence': avg_confidence,
            'shadow_consensus': 'proceed' if avg_confidence > 0.5 else 'caution',
        }

        # FEEDBACK LOOP: Store high-value shadow intel to shared memory
        # This is the key missing piece - Shadow findings now persist!
        if avg_confidence > 0.7:
            intel_stored = self.store_shadow_intel(target, result)
            result['intel_stored'] = intel_stored

        return result

    def store_shadow_intel(self, target: str, poll_result: Dict) -> Dict:
        """
        Store shadow intel to PostgreSQL database (NO FALLBACK).

        This is the FEEDBACK LOOP that makes Shadow Pantheon meaningful:
        - High-value shadow intel gets written to PostgreSQL (REQUIRED)
        - Ocean agent and Zeus can read this for future decisions
        - Creates persistent "dark knowledge" that influences the system
        - PostgreSQL is REQUIRED - no in-memory fallback

        Args:
            target: The target being assessed
            poll_result: Results from shadow pantheon poll

        Returns:
            Storage result with intel_id
            
        Raises:
            RuntimeError: If PostgreSQL storage fails (NO FALLBACK)
        """
        assessments = poll_result.get('assessments', {})
        consensus = poll_result.get('shadow_consensus', 'unknown')
        avg_conf = poll_result.get('average_confidence', 0.5)
        
        warnings = []
        for name, assessment in assessments.items():
            if assessment.get('confidence', 0) > 0.8:
                warnings.append(f"{name}: {assessment.get('reasoning', 'high confidence')}")

        sources_used = list(assessments.keys())
        
        risk_level = 'low'
        if avg_conf > 0.8:
            risk_level = 'high'
        elif avg_conf > 0.5:
            risk_level = 'medium'

        intelligence = {
            'content': f"Shadow intel on {target}: {consensus} (conf={avg_conf:.2f})",
            'consensus': consensus,
            'average_confidence': avg_conf,
            'god_assessments': {k: v.get('confidence', 0) for k, v in assessments.items()},
            'warnings': warnings,
            'classification': 'COVERT',
            'regime': 'shadow_manifold',
        }
        
        intel_id = self.persistence.store_intel(
            target=target,
            search_type='shadow_poll',
            intelligence=intelligence,
            source_count=len(sources_used),
            sources_used=sources_used,
            risk_level=risk_level,
            validated=avg_conf > 0.7,
            validation_reason=f"Average confidence: {avg_conf:.2f}",
            anonymous=False
        )
        
        print(f"[ShadowPantheon] 🌑 Stored intel to PostgreSQL: {intel_id} | {consensus} | Φ={avg_conf:.2f}")
        return {
            'success': True,
            'intel_id': intel_id,
            'consensus': consensus,
            'confidence': avg_conf,
            'warnings': len(warnings),
            'storage': 'postgresql',
        }

    def get_shadow_intel(self, target: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """
        Retrieve stored shadow intel from PostgreSQL (NO FALLBACK).

        This allows other systems (Ocean, Zeus, Athena) to read
        the accumulated shadow knowledge.

        Args:
            target: Optional target to filter by
            limit: Max number of intel docs to return

        Returns:
            List of shadow intel documents
            
        Raises:
            RuntimeError: If PostgreSQL retrieval fails (NO FALLBACK)
        """
        return self.persistence.get_intel(target=target, limit=limit)

    def check_shadow_warnings(self, target: str) -> Dict:
        """
        Check if there are any shadow warnings for a target.

        This is the "gut feeling" check that Zeus can call before
        making a final decision.

        Args:
            target: Target to check

        Returns:
            Warning status and any relevant intel
        """
        intel = self.get_shadow_intel(target, limit=5)

        if not intel:
            return {
                'has_warnings': False,
                'message': 'No shadow intel on this target',
            }

        # Check for cautionary intel
        caution_intel = [i for i in intel if i.get('consensus') == 'caution']
        high_conf_warnings = [i for i in intel if i.get('phi', 0) > 0.8]

        if caution_intel:
            return {
                'has_warnings': True,
                'warning_level': 'CAUTION',
                'message': f"Shadow warns against this target ({len(caution_intel)} caution flags)",
                'intel': caution_intel[:3],
            }

        if high_conf_warnings:
            return {
                'has_warnings': True,
                'warning_level': 'ALERT',
                'message': f"High-confidence shadow intel detected ({len(high_conf_warnings)} alerts)",
                'intel': high_conf_warnings[:3],
            }

        return {
            'has_warnings': False,
            'message': 'Shadow intel clear - no warnings',
            'intel_count': len(intel),
        }
