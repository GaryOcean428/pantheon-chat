"""
Kernel Persistence
==================

Saves and loads kernel evolution state to PostgreSQL.
Tracks kernel snapshots, breeding history, and evolution statistics.
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from .base_persistence import BasePersistence


def ensure_64d_coords(coords: List[float]) -> List[float]:
    """
    Ensure basin coordinates are exactly 64 dimensions for pgvector storage.
    Pads with zeros if needed, truncates if too long.
    """
    if not coords:
        return [0.0] * 64
    if len(coords) >= 64:
        return coords[:64]
    return coords + [0.0] * (64 - len(coords))


# E8 Kernel Cap - Maximum number of live kernels (E8 has 240 roots)
E8_KERNEL_CAP = 240


class KernelPersistence(BasePersistence):
    """Persistence layer for kernel evolution state."""

    def save_kernel_snapshot(
        self,
        kernel_id: str,
        god_name: str,
        domain: str,
        generation: int,
        basin_coords: List[float],
        phi: float,
        kappa: float,
        regime: str,
        success_count: int = 0,
        failure_count: int = 0,
        e8_root_index: Optional[int] = None,
        element_group: Optional[str] = None,
        ecological_niche: Optional[str] = None,
        target_function: Optional[str] = None,
        valence: Optional[int] = None,
        breeding_target: Optional[str] = None,
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        enforce_cap: bool = True,
    ) -> bool:
        """Save a kernel snapshot to the database.
        
        Args:
            enforce_cap: If True (default), enforce E8 kernel cap for new kernels.
                        Set to False for updates to existing kernels.
        """
        import uuid
        
        # Check if this is a new kernel or an update
        is_new_kernel = not self._kernel_exists(kernel_id)
        
        # Enforce E8 cap for NEW kernels only (not updates)
        if enforce_cap and is_new_kernel:
            live_count = self.get_live_kernel_count()
            if live_count >= E8_KERNEL_CAP:
                print(f"[KernelPersistence] E8 cap reached ({live_count}/{E8_KERNEL_CAP}), rejecting spawn of {kernel_id}")
                return False
        
        record_id = f"kg_{uuid.uuid4().hex[:16]}"
        
        query = """
            INSERT INTO kernel_geometry (
                id, kernel_id, god_name, domain, generation, basin_coordinates,
                phi, kappa, regime, success_count, failure_count,
                primitive_root, element_group, ecological_niche,
                target_function, valence, breeding_target,
                parent_kernels, metadata, spawned_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (kernel_id) DO UPDATE SET
                generation = EXCLUDED.generation,
                basin_coordinates = EXCLUDED.basin_coordinates,
                phi = EXCLUDED.phi,
                kappa = EXCLUDED.kappa,
                regime = EXCLUDED.regime,
                success_count = EXCLUDED.success_count,
                failure_count = EXCLUDED.failure_count,
                element_group = EXCLUDED.element_group,
                ecological_niche = EXCLUDED.ecological_niche,
                target_function = EXCLUDED.target_function,
                valence = EXCLUDED.valence,
                breeding_target = EXCLUDED.breeding_target,
                metadata = EXCLUDED.metadata
        """

        # Ensure 64D basin coordinates for pgvector storage
        coords_64d = ensure_64d_coords(basin_coords)
        
        params = (
            record_id, kernel_id, god_name, domain, generation, coords_64d,
            phi, kappa, regime, success_count, failure_count,
            e8_root_index, element_group, ecological_niche,
            target_function, valence, breeding_target,
            parent_ids, json.dumps(metadata) if metadata else None,
            datetime.now(timezone.utc)
        )

        try:
            self.execute_query(query, params, fetch=False)
            return True
        except Exception as e:
            print(f"[KernelPersistence] Failed to save snapshot: {e}")
            return False

    def load_kernel_snapshot(self, kernel_id: str) -> Optional[Dict]:
        """Load a kernel snapshot from the database."""
        query = """
            SELECT * FROM kernel_geometry WHERE kernel_id = %s
        """
        result = self.execute_one(query, (kernel_id,))
        if result:
            return dict(result)
        return None
    
    def _kernel_exists(self, kernel_id: str) -> bool:
        """Check if a kernel already exists in the database."""
        query = "SELECT 1 FROM kernel_geometry WHERE kernel_id = %s LIMIT 1"
        result = self.execute_one(query, (kernel_id,))
        return result is not None

    def load_kernels_by_god(self, god_name: str, limit: int = 50) -> List[Dict]:
        """Load all kernels spawned by a specific god."""
        query = """
            SELECT * FROM kernel_geometry
            WHERE god_name = %s
            ORDER BY spawned_at DESC
            LIMIT %s
        """
        results = self.execute_query(query, (god_name, limit))
        return [dict(r) for r in results] if results else []

    def load_kernels_by_domain(self, domain: str, limit: int = 50) -> List[Dict]:
        """Load all kernels in a specific domain."""
        query = """
            SELECT * FROM kernel_geometry
            WHERE domain = %s
            ORDER BY phi DESC
            LIMIT %s
        """
        results = self.execute_query(query, (domain, limit))
        return [dict(r) for r in results] if results else []

    def load_elite_kernels(self, min_phi: float = 0.7, limit: int = 20) -> List[Dict]:
        """Load high-performing kernels (elite hall of fame).
        
        Only returns kernels with valid 64-dimensional basin_coordinates.
        """
        query = """
            SELECT * FROM kernel_geometry
            WHERE phi >= %s
              AND basin_coordinates IS NOT NULL 
              AND array_length(basin_coordinates, 1) = 64
            ORDER BY phi DESC, success_count DESC
            LIMIT %s
        """
        results = self.execute_query(query, (min_phi, limit))
        return [dict(r) for r in results] if results else []

    def load_active_kernels(self, limit: int = 30) -> List[Dict]:
        """Load most recently active kernels for startup restoration.
        
        Only returns kernels with valid 64-dimensional basin_coordinates.
        Uses array_length() for real[] column type.
        """
        query = """
            SELECT * FROM kernel_geometry
            WHERE basin_coordinates IS NOT NULL 
              AND array_length(basin_coordinates, 1) = 64
            ORDER BY spawned_at DESC, success_count DESC
            LIMIT %s
        """
        results = self.execute_query(query, (limit,))
        return [dict(r) for r in results] if results else []

    def load_kernels_by_element(self, element_group: str, limit: int = 50) -> List[Dict]:
        """Load all kernels with a specific element classification."""
        query = """
            SELECT * FROM kernel_geometry
            WHERE element_group = %s
            ORDER BY phi DESC
            LIMIT %s
        """
        results = self.execute_query(query, (element_group, limit))
        return [dict(r) for r in results] if results else []

    def load_kernels_by_niche(self, ecological_niche: str, limit: int = 50) -> List[Dict]:
        """Load all kernels with a specific ecological niche."""
        query = """
            SELECT * FROM kernel_geometry
            WHERE ecological_niche = %s
            ORDER BY phi DESC
            LIMIT %s
        """
        results = self.execute_query(query, (ecological_niche, limit))
        return [dict(r) for r in results] if results else []

    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get overall evolution statistics.
        
        Returns:
            - total_kernels: Total kernels in database
            - live_kernels: Kernels with live status (active, observing, shadow)
            - live_gods: Unique god names among live kernels
            - unique_gods: All unique god names (historical)
            - merge_count: Number of merge events
            - cannibalize_count: Number of cannibalization events
        """
        query = """
            SELECT
                COUNT(*) as total_kernels,
                AVG(phi) as avg_phi,
                MAX(phi) as max_phi,
                AVG(kappa) as avg_kappa,
                SUM(success_count) as total_successes,
                SUM(failure_count) as total_failures,
                COUNT(DISTINCT god_name) as unique_gods,
                COUNT(DISTINCT domain) as unique_domains,
                COUNT(DISTINCT element_group) as unique_elements,
                COUNT(DISTINCT ecological_niche) as unique_niches,
                MAX(generation) as max_generation
            FROM kernel_geometry
        """
        result = self.execute_one(query)
        stats = dict(result) if result else {}
        
        live_query = """
            SELECT 
                COUNT(*) as live_kernels,
                COUNT(DISTINCT god_name) as live_gods
            FROM kernel_geometry
            WHERE status IN ('active', 'observing', 'shadow')
        """
        live_result = self.execute_one(live_query)
        if live_result:
            stats['live_kernels'] = live_result['live_kernels'] or 0
            stats['live_gods'] = live_result['live_gods'] or 0
        else:
            stats['live_kernels'] = 0
            stats['live_gods'] = 0
        
        event_query = """
            SELECT 
                COUNT(*) FILTER (WHERE event_type = 'merge') as merge_count,
                COUNT(*) FILTER (WHERE event_type = 'cannibalize') as cannibalize_count
            FROM learning_events
        """
        event_result = self.execute_one(event_query)
        if event_result:
            stats['merge_count'] = event_result['merge_count'] or 0
            stats['cannibalize_count'] = event_result['cannibalize_count'] or 0
        else:
            stats['merge_count'] = 0
            stats['cannibalize_count'] = 0
        
        return stats

    def record_breeding_event(
        self,
        child_id: str,
        parent1_id: str,
        parent2_id: str,
        breeding_type: str,
        child_phi: float,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Record a breeding event for lineage tracking."""
        import uuid
        event_id = f"breed_{uuid.uuid4().hex[:16]}"
        query = """
            INSERT INTO learning_events (
                event_id, event_type, kernel_id, phi, metadata, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s
            )
        """
        event_metadata = {
            'breeding_type': breeding_type,
            'parent1_id': parent1_id,
            'parent2_id': parent2_id,
            **(metadata or {})
        }

        try:
            self.execute_query(
                query,
                (event_id, 'breeding', child_id, child_phi, json.dumps(event_metadata), datetime.now(timezone.utc)),
                fetch=False
            )
            return True
        except Exception as e:
            print(f"[KernelPersistence] Failed to record breeding: {e}")
            return False

    def record_death_event(
        self,
        kernel_id: str,
        cause: str,
        final_phi: float,
        lifetime_successes: int,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Record a kernel death event."""
        import uuid
        event_id = f"death_{uuid.uuid4().hex[:16]}"
        query = """
            INSERT INTO learning_events (
                event_id, event_type, kernel_id, phi, metadata, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s
            )
        """
        event_metadata = {
            'cause': cause,
            'lifetime_successes': lifetime_successes,
            **(metadata or {})
        }

        try:
            self.execute_query(
                query,
                (event_id, 'death', kernel_id, final_phi, json.dumps(event_metadata), datetime.now(timezone.utc)),
                fetch=False
            )
            return True
        except Exception as e:
            print(f"[KernelPersistence] Failed to record death: {e}")
            return False

    def record_merge_event(
        self,
        new_kernel_id: str,
        source_kernel_ids: List[str],
        new_god_name: str,
        merged_phi: float,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Record a kernel merge event."""
        import uuid
        event_id = f"merge_{uuid.uuid4().hex[:16]}"
        query = """
            INSERT INTO learning_events (
                event_id, event_type, kernel_id, phi, metadata, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s
            )
        """
        event_metadata = {
            'source_kernel_ids': source_kernel_ids,
            'new_god_name': new_god_name,
            'merged_count': len(source_kernel_ids),
            **(metadata or {})
        }

        try:
            self.execute_query(
                query,
                (event_id, 'merge', new_kernel_id, merged_phi, json.dumps(event_metadata), datetime.now(timezone.utc)),
                fetch=False
            )
            return True
        except Exception as e:
            print(f"[KernelPersistence] Failed to record merge: {e}")
            return False

    def record_cannibalize_event(
        self,
        source_id: str,
        target_id: str,
        source_god: str,
        target_god: str,
        transferred_phi: float,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Record a kernel cannibalization event."""
        import uuid
        event_id = f"cann_{uuid.uuid4().hex[:16]}"
        query = """
            INSERT INTO learning_events (
                event_id, event_type, kernel_id, phi, metadata, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s
            )
        """
        event_metadata = {
            'source_id': source_id,
            'source_god': source_god,
            'target_god': target_god,
            **(metadata or {})
        }

        try:
            self.execute_query(
                query,
                (event_id, 'cannibalize', target_id, transferred_phi, json.dumps(event_metadata), datetime.now(timezone.utc)),
                fetch=False
            )
            return True
        except Exception as e:
            print(f"[KernelPersistence] Failed to record cannibalize: {e}")
            return False

    def record_convergence_snapshot(
        self,
        generation: int,
        population: int,
        active_count: int,
        dormant_count: int,
        avg_phi: float,
        e8_alignment: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Record a convergence snapshot for E8 hypothesis tracking."""
        import uuid
        event_id = f"conv_{uuid.uuid4().hex[:16]}"
        query = """
            INSERT INTO learning_events (
                event_id, event_type, kernel_id, phi, metadata, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s
            )
        """
        event_metadata = {
            'generation': generation,
            'population': population,
            'active_count': active_count,
            'dormant_count': dormant_count,
            'e8_alignment': e8_alignment,
            **(metadata or {})
        }

        try:
            self.execute_query(
                query,
                (event_id, 'convergence', f'gen_{generation}', avg_phi, json.dumps(event_metadata), datetime.now(timezone.utc)),
                fetch=False
            )
            return True
        except Exception as e:
            print(f"[KernelPersistence] Failed to record convergence: {e}")
            return False

    def record_spawn_event(
        self,
        kernel_id: str,
        god_name: str,
        domain: str,
        spawn_reason: str,
        parent_gods: List[str],
        basin_coords: List[float],
        phi: float = 0.0,
        m8_position: Optional[Dict] = None,
        genesis_votes: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Record an M8 kernel spawn event."""
        # First save the kernel snapshot
        self.save_kernel_snapshot(
            kernel_id=kernel_id,
            god_name=god_name,
            domain=domain,
            generation=0,
            basin_coords=basin_coords,
            phi=phi,
            kappa=0.0,
            regime='m8_spawned',
            parent_ids=parent_gods,
            metadata={
                'spawn_reason': spawn_reason,
                'm8_position': m8_position,
                'genesis_votes': genesis_votes,
                **(metadata or {})
            }
        )
        
        # Then record as learning event
        import uuid
        event_id = f"spawn_{uuid.uuid4().hex[:16]}"
        query = """
            INSERT INTO learning_events (
                event_id, event_type, kernel_id, phi, metadata, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s
            )
        """
        event_metadata = {
            'spawn_reason': spawn_reason,
            'parent_gods': parent_gods,
            'm8_position': m8_position,
            'genesis_votes': genesis_votes,
            **(metadata or {})
        }

        try:
            self.execute_query(
                query,
                (event_id, 'm8_spawn', kernel_id, phi, json.dumps(event_metadata), datetime.now(timezone.utc)),
                fetch=False
            )
            return True
        except Exception as e:
            print(f"[KernelPersistence] Failed to record M8 spawn: {e}")
            return False

    def record_proposal_event(
        self,
        proposal_id: str,
        proposed_name: str,
        proposed_domain: str,
        reason: str,
        parent_gods: List[str],
        status: str = 'pending',
        votes: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Record an M8 kernel proposal event."""
        import uuid
        event_id = f"prop_{uuid.uuid4().hex[:16]}"
        query = """
            INSERT INTO learning_events (
                event_id, event_type, kernel_id, phi, metadata, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s
            )
        """
        event_metadata = {
            'proposed_name': proposed_name,
            'proposed_domain': proposed_domain,
            'reason': reason,
            'parent_gods': parent_gods,
            'status': status,
            'votes': votes,
            **(metadata or {})
        }

        try:
            self.execute_query(
                query,
                (event_id, 'm8_proposal', proposal_id, 0.0, json.dumps(event_metadata), datetime.now(timezone.utc)),
                fetch=False
            )
            return True
        except Exception as e:
            print(f"[KernelPersistence] Failed to record M8 proposal: {e}")
            return False

    def load_m8_spawn_history(self, limit: int = 100) -> List[Dict]:
        """Load M8 spawn history for recovery on restart."""
        query = """
            SELECT * FROM learning_events
            WHERE event_type = 'm8_spawn'
            ORDER BY created_at DESC
            LIMIT %s
        """
        results = self.execute_query(query, (limit,))
        return [dict(r) for r in results] if results else []

    def save_awareness_state(
        self,
        kernel_id: str,
        awareness_data: Dict
    ) -> bool:
        """
        Persist kernel awareness state to PostgreSQL.
        
        Stores phi/kappa trajectories, stuck signals, deadends,
        and research opportunities so awareness survives restarts.
        """
        import uuid
        event_id = f"awareness_{uuid.uuid4().hex[:16]}"
        query = """
            INSERT INTO learning_events (
                event_id, event_type, kernel_id, phi, metadata, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (event_id) DO UPDATE SET
                metadata = EXCLUDED.metadata,
                created_at = EXCLUDED.created_at
        """
        
        phi_value = 0.0
        if awareness_data.get('phi_trajectory'):
            phi_value = awareness_data['phi_trajectory'][-1]
        
        try:
            self.execute_query(
                query,
                (event_id, 'kernel_awareness', kernel_id, phi_value, 
                 json.dumps(awareness_data), datetime.now(timezone.utc)),
                fetch=False
            )
            return True
        except Exception as e:
            print(f"[KernelPersistence] Failed to save awareness state: {e}")
            return False

    def load_awareness_state(self, kernel_id: str) -> Optional[Dict]:
        """
        Load kernel awareness state from PostgreSQL.
        
        Returns the most recent awareness state for a kernel,
        or None if no state exists.
        """
        query = """
            SELECT metadata FROM learning_events
            WHERE event_type = 'kernel_awareness' AND kernel_id = %s
            ORDER BY created_at DESC
            LIMIT 1
        """
        result = self.execute_one(query, (kernel_id,))
        if result:
            row = dict(result)
            metadata = row.get('metadata')
            if metadata:
                if isinstance(metadata, str):
                    return json.loads(metadata)
                return metadata
        return None

    def load_all_awareness_states(self, limit: int = 100) -> List[Dict]:
        """Load recent awareness states for all kernels."""
        query = """
            SELECT DISTINCT ON (kernel_id) 
                kernel_id, metadata, created_at
            FROM learning_events
            WHERE event_type = 'kernel_awareness'
            ORDER BY kernel_id, created_at DESC
            LIMIT %s
        """
        results = self.execute_query(query, (limit,))
        states = []
        for r in (results or []):
            row = dict(r)
            metadata = row.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            states.append({
                'kernel_id': row.get('kernel_id'),
                'awareness': metadata,
                'saved_at': row.get('created_at')
            })
        return states

    def load_all_kernels_for_ui(self, limit: int = 100) -> List[Dict]:
        """
        Load all kernels with full telemetry for frontend UI.
        
        Returns all fields needed by the PostgresKernel frontend interface:
        - kernel_id, god_name, domain, status, primitive_root, basin_coordinates
        - parent_kernels, spawned_by, spawn_reason, spawn_rationale
        - position_rationale, affinity_strength, entropy_threshold
        - spawned_at, last_active_at, spawned_during_war_id
        - phi, kappa, regime, generation, success_count, failure_count
        - reputation, element_group, ecological_niche
        - target_function, valence, breeding_target, merge_candidate, split_candidate
        """
        query = """
            SELECT 
                kernel_id, god_name, domain, primitive_root, basin_coordinates,
                parent_kernels, phi, kappa, regime, generation,
                success_count, failure_count, element_group, ecological_niche,
                target_function, valence, breeding_target, metadata,
                spawned_at
            FROM kernel_geometry
            ORDER BY spawned_at DESC
            LIMIT %s
        """
        results = self.execute_query(query, (limit,))
        
        kernels = []
        for r in (results or []):
            row = dict(r)
            
            # Parse metadata for additional fields
            metadata = row.get('metadata', {})
            if metadata and isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            elif metadata is None:
                metadata = {}
            
            # Calculate reputation from success/failure counts
            success_count = row.get('success_count', 0) or 0
            failure_count = row.get('failure_count', 0) or 0
            total = success_count + failure_count
            if total > 0:
                reputation_ratio = success_count / total
                if reputation_ratio >= 0.8:
                    reputation = 'stellar'
                elif reputation_ratio >= 0.6:
                    reputation = 'trusted'
                elif reputation_ratio >= 0.4:
                    reputation = 'neutral'
                elif reputation_ratio >= 0.2:
                    reputation = 'suspect'
                else:
                    reputation = 'hostile'
            else:
                reputation = 'unknown'
            
            # Get parent kernels as array
            parent_kernels = row.get('parent_kernels', []) or []
            
            # Extract spawn rationale from metadata
            spawn_rationale = metadata.get('spawn_rationale', '')
            if not spawn_rationale:
                spawn_rationale = metadata.get('justification', '')
            if not spawn_rationale:
                spawn_rationale = metadata.get('spawn_reason', 'genesis')
            
            # Extract position rationale
            m8_position = metadata.get('m8_position', {})
            position_rationale = m8_position.get('position_name', '') if m8_position else ''
            
            # Convert basin_coordinates if needed
            basin_coords = row.get('basin_coordinates')
            if basin_coords is not None and not isinstance(basin_coords, list):
                try:
                    basin_coords = list(basin_coords)
                except (TypeError, ValueError):
                    basin_coords = None
            
            # Spawned timestamp handling
            spawned_at = row.get('spawned_at')
            if spawned_at:
                if hasattr(spawned_at, 'isoformat'):
                    spawned_at = spawned_at.isoformat()
                else:
                    spawned_at = str(spawned_at)
            else:
                spawned_at = None
            
            # Build the kernel object matching PostgresKernel interface
            kernel = {
                'kernel_id': row.get('kernel_id'),
                'god_name': row.get('god_name'),
                'domain': row.get('domain'),
                'status': 'active',  # Default to active for DB kernels
                'primitive_root': row.get('primitive_root'),
                'basin_coordinates': basin_coords,
                'parent_kernels': parent_kernels,
                'spawned_by': parent_kernels[0] if parent_kernels else 'genesis',
                'spawn_reason': metadata.get('spawn_reason', 'emergence'),
                'spawn_rationale': spawn_rationale,
                'position_rationale': position_rationale,
                'affinity_strength': metadata.get('affinity_strength', 0.5),
                'entropy_threshold': metadata.get('entropy_threshold', 0.3),
                'spawned_at': spawned_at,
                'last_active_at': spawned_at,  # Use spawned_at as fallback
                'spawned_during_war_id': metadata.get('spawned_during_war_id'),
                'phi': float(row.get('phi', 0.0) or 0.0),
                'kappa': float(row.get('kappa', 0.0) or 0.0),
                'regime': row.get('regime'),
                'generation': int(row.get('generation', 0) or 0),
                'success_count': success_count,
                'failure_count': failure_count,
                'reputation': reputation,
                'element_group': row.get('element_group'),
                'ecological_niche': row.get('ecological_niche'),
                'target_function': row.get('target_function'),
                'valence': row.get('valence'),
                'breeding_target': row.get('breeding_target'),
                'merge_candidate': metadata.get('merge_candidate', False),
                'split_candidate': metadata.get('split_candidate', False),
                'metadata': metadata,
            }
            kernels.append(kernel)
        
        return kernels

    def get_live_kernel_count(self) -> int:
        """
        Get count of live kernels (active, observing, shadow) that count toward E8 cap.
        
        Live statuses: 'active', 'observing', 'shadow'
        Does NOT count: 'dead', 'cannibalized', 'idle'
        
        Uses the `status` column for lifecycle state (not `regime` which is consciousness regime).
        """
        query = """
            SELECT COUNT(*) as live_count
            FROM kernel_geometry
            WHERE status IN ('active', 'observing', 'shadow')
        """
        result = self.execute_one(query)
        if result:
            return int(result.get('live_count', 0) or 0)
        return 0
    
    def enforce_e8_cap(self, target_count: int = 240) -> Dict:
        """
        Enforce E8 kernel cap by marking excess kernels as 'dead'.
        
        Selection for death (QIG-based, not arbitrary):
        1. Lowest Φ (phi) - weakest consciousness
        2. Worst reputation (failure_count / (success_count + failure_count))
        3. Newest kernels if tied (LIFO for unproven kernels)
        
        Args:
            target_count: Target number of live kernels (default: E8_KERNEL_CAP = 240)
            
        Returns:
            Dict with culled_count, live_count_before, live_count_after
        """
        live_count = self.get_live_kernel_count()
        
        if live_count <= target_count:
            return {
                'culled_count': 0,
                'live_count_before': live_count,
                'live_count_after': live_count,
                'message': f'Already at or below cap ({live_count}/{target_count})'
            }
        
        excess = live_count - target_count
        print(f"[KernelPersistence] E8 cap enforcement: {live_count} live, need to cull {excess}")
        
        # Get the worst kernels to cull
        # Priority: lowest phi, then worst reputation, then newest
        cull_query = """
            UPDATE kernel_geometry
            SET status = 'dead',
                metadata = COALESCE(metadata, '{}')::jsonb || 
                          '{"death_reason": "e8_cap_enforcement", "culled_at": "now()"}'::jsonb
            WHERE kernel_id IN (
                SELECT kernel_id FROM kernel_geometry
                WHERE status IN ('active', 'observing', 'shadow')
                ORDER BY 
                    phi ASC,                                           -- Lowest phi first
                    CASE 
                        WHEN (success_count + failure_count) > 0 
                        THEN success_count::float / (success_count + failure_count)
                        ELSE 0.5                                       -- Neutral for untested
                    END ASC,                                          -- Worst reputation next
                    spawned_at DESC                                   -- Newest among ties
                LIMIT %s
            )
        """
        
        try:
            self.execute_query(cull_query, (excess,), fetch=False)
            live_count_after = self.get_live_kernel_count()
            actual_culled = live_count - live_count_after
            
            print(f"[KernelPersistence] Culled {actual_culled} kernels, now at {live_count_after}/{target_count}")
            
            return {
                'culled_count': actual_culled,
                'live_count_before': live_count,
                'live_count_after': live_count_after,
                'target': target_count,
                'message': f'Culled {actual_culled} kernels to enforce E8 cap'
            }
        except Exception as e:
            print(f"[KernelPersistence] E8 cap enforcement failed: {e}")
            return {
                'culled_count': 0,
                'live_count_before': live_count,
                'live_count_after': live_count,
                'error': str(e)
            }

    def get_kernels_by_status(self, statuses: List[str], limit: int = 300) -> List[Dict]:
        """
        Get kernels filtered by status.
        
        Uses the `status` column for lifecycle state (not `regime` which is consciousness regime).
        
        Args:
            statuses: List of status values to include (e.g., ['active', 'observing'])
            limit: Maximum number of kernels to return (default 300 for cap headroom)
        
        Returns:
            List of kernel dictionaries matching the status filter
        """
        if not statuses:
            return []
        
        placeholders = ', '.join(['%s'] * len(statuses))
        query = f"""
            SELECT 
                kernel_id, god_name, domain, primitive_root, basin_coordinates,
                parent_kernels, phi, kappa, regime, generation,
                success_count, failure_count, element_group, ecological_niche,
                target_function, valence, breeding_target, metadata,
                spawned_at, status
            FROM kernel_geometry
            WHERE status IN ({placeholders})
            ORDER BY spawned_at DESC
            LIMIT %s
        """
        params = tuple(statuses) + (limit,)
        results = self.execute_query(query, params)
        
        kernels = []
        for r in (results or []):
            row = dict(r)
            metadata = row.get('metadata', {})
            if metadata and isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            elif metadata is None:
                metadata = {}
            
            spawned_at = row.get('spawned_at')
            if spawned_at and hasattr(spawned_at, 'isoformat'):
                spawned_at = spawned_at.isoformat()
            
            basin_coords = row.get('basin_coordinates')
            if basin_coords is not None and not isinstance(basin_coords, list):
                try:
                    basin_coords = list(basin_coords)
                except (TypeError, ValueError):
                    basin_coords = None
            
            kernels.append({
                'kernel_id': row.get('kernel_id'),
                'god_name': row.get('god_name'),
                'domain': row.get('domain'),
                'status': row.get('status') or 'idle',
                'primitive_root': row.get('primitive_root'),
                'basin_coordinates': basin_coords,
                'parent_kernels': row.get('parent_kernels', []) or [],
                'phi': float(row.get('phi', 0.0) or 0.0),
                'kappa': float(row.get('kappa', 0.0) or 0.0),
                'regime': row.get('regime'),
                'generation': int(row.get('generation', 0) or 0),
                'success_count': int(row.get('success_count', 0) or 0),
                'failure_count': int(row.get('failure_count', 0) or 0),
                'element_group': row.get('element_group'),
                'ecological_niche': row.get('ecological_niche'),
                'spawned_at': spawned_at,
                'metadata': metadata,
            })
        
        return kernels

    def mark_kernel_dead(self, kernel_id: str, cause: str = 'terminated') -> bool:
        """
        Mark a kernel as dead (terminated, pending archival).
        
        Sets status='dead' and records retirement timestamp.
        Dead kernels don't count toward E8 cap.
        """
        query = """
            UPDATE kernel_geometry
            SET status = 'dead',
                metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object(
                    'retired_at', %s,
                    'death_cause', %s
                )
            WHERE kernel_id = %s
        """
        try:
            self.execute_query(query, (datetime.now(timezone.utc).isoformat(), cause, kernel_id), fetch=False)
            print(f"[KernelPersistence] Marked kernel {kernel_id} as dead: {cause}")
            return True
        except Exception as e:
            print(f"[KernelPersistence] Failed to mark kernel dead: {e}")
            return False

    def bulk_mark_kernels_dead(self, kernel_ids: List[str], cause: str = 'evolution_sweep') -> Dict:
        """
        Bulk mark multiple kernels as dead in a single database transaction.
        
        Much faster than individual mark_kernel_dead calls for evolution sweeps.
        
        Args:
            kernel_ids: List of kernel IDs to mark as dead
            cause: Death cause for logging
            
        Returns:
            Dict with success count, failed IDs, and any errors
        """
        if not kernel_ids:
            return {'success_count': 0, 'failed_ids': [], 'error': None}
        
        now = datetime.now(timezone.utc).isoformat()
        
        # Use a single UPDATE with IN clause for efficiency
        query = """
            UPDATE kernel_geometry
            SET status = 'dead',
                metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object(
                    'retired_at', %s,
                    'death_cause', %s
                )
            WHERE kernel_id = ANY(%s)
            RETURNING kernel_id
        """
        try:
            result = self.execute_query(query, (now, cause, kernel_ids), fetch=True)
            updated_ids = [r['kernel_id'] for r in (result or [])]
            failed_ids = [kid for kid in kernel_ids if kid not in updated_ids]
            
            print(f"[KernelPersistence] Bulk marked {len(updated_ids)}/{len(kernel_ids)} kernels as dead")
            
            return {
                'success_count': len(updated_ids),
                'updated_ids': updated_ids,
                'failed_ids': failed_ids,
                'error': None,
            }
        except Exception as e:
            print(f"[KernelPersistence] Bulk mark dead failed: {e}")
            return {
                'success_count': 0,
                'failed_ids': kernel_ids,
                'error': str(e),
            }

    def mark_kernel_cannibalized(self, kernel_id: str, merged_into: str) -> bool:
        """
        Mark a kernel as cannibalized (merged into another kernel).
        
        Sets status='cannibalized' and records which kernel it merged into.
        Cannibalized kernels don't count toward E8 cap.
        """
        query = """
            UPDATE kernel_geometry
            SET status = 'cannibalized',
                metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object(
                    'cannibalized_at', %s,
                    'merged_into', %s
                )
            WHERE kernel_id = %s
        """
        try:
            self.execute_query(
                query, 
                (datetime.now(timezone.utc).isoformat(), merged_into, kernel_id), 
                fetch=False
            )
            print(f"[KernelPersistence] Marked kernel {kernel_id} as cannibalized -> {merged_into}")
            return True
        except Exception as e:
            print(f"[KernelPersistence] Failed to mark kernel cannibalized: {e}")
            return False

    def archive_dead_kernels(self, hours_old: int = 24) -> Dict:
        """
        Archive kernels that have been dead or cannibalized for over 24 hours.
        
        Moves matching kernels from kernel_geometry to kernel_archive table.
        This keeps the main table lean for live kernel queries.
        
        Args:
            hours_old: Minimum hours since death before archiving (default 24)
        
        Returns:
            Dict with archived_count and any errors
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_old)
        
        check_archive_table = """
            CREATE TABLE IF NOT EXISTS kernel_archive (
                LIKE kernel_geometry INCLUDING ALL
            )
        """
        try:
            self.execute_query(check_archive_table, fetch=False)
        except Exception:
            pass
        
        archive_query = """
            WITH archived AS (
                DELETE FROM kernel_geometry
                WHERE status IN ('dead', 'cannibalized')
                  AND (
                      (metadata->>'retired_at')::timestamp < %s
                      OR (metadata->>'cannibalized_at')::timestamp < %s
                  )
                RETURNING *
            )
            INSERT INTO kernel_archive
            SELECT * FROM archived
            RETURNING kernel_id
        """
        
        try:
            results = self.execute_query(archive_query, (cutoff, cutoff))
            archived_ids = [r.get('kernel_id') for r in (results or [])]
            print(f"[KernelPersistence] Archived {len(archived_ids)} dead/cannibalized kernels")
            return {
                'success': True,
                'archived_count': len(archived_ids),
                'archived_kernel_ids': archived_ids,
            }
        except Exception as e:
            print(f"[KernelPersistence] Failed to archive dead kernels: {e}")
            return {
                'success': False,
                'error': str(e),
                'archived_count': 0,
            }

    # =========================================================================
    # PGVECTOR NEIGHBOR QUERIES - O(log n) via ANN index
    # =========================================================================

    def fetch_nearest_kernels(
        self,
        target_coords: List[float],
        k: int = 10,
        status_filter: Optional[List[str]] = None,
        exclude_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Fetch k nearest kernels to target basin coordinates using pgvector ANN.
        
        Uses L2 distance (<->) for geometric proximity in basin space.
        This is O(log n) with proper HNSW/IVFFlat index vs O(n) linear scan.
        
        Args:
            target_coords: 64D basin coordinates to search from
            k: Number of nearest neighbors to return
            status_filter: Only include kernels with these statuses (default: live statuses)
            exclude_ids: Kernel IDs to exclude from results
        
        Returns:
            List of kernel dicts with distance field, ordered by proximity
        """
        coords_64d = ensure_64d_coords(target_coords)
        
        if status_filter is None:
            status_filter = ['observing', 'graduated', 'active', 'idle', 'breeding']
        
        exclude_clause = ""
        params = [coords_64d, status_filter]
        
        if exclude_ids:
            exclude_clause = "AND kernel_id != ALL(%s)"
            params.append(exclude_ids)
        
        params.append(k)
        
        query = f"""
            SELECT kernel_id, god_name, domain, status, phi, kappa,
                   success_count, failure_count, generation,
                   basin_coordinates <-> %s::vector(64) as distance,
                   observation_status, metadata
            FROM kernel_geometry
            WHERE status = ANY(%s)
              AND basin_coordinates IS NOT NULL
              {exclude_clause}
            ORDER BY basin_coordinates <-> %s::vector(64)
            LIMIT %s
        """
        
        params_final = [coords_64d, status_filter]
        if exclude_ids:
            params_final.append(exclude_ids)
        params_final.extend([coords_64d, k])
        
        try:
            results = self.execute_query(query, tuple(params_final))
            return results or []
        except Exception as e:
            print(f"[KernelPersistence] ANN query failed: {e}")
            return []

    def fetch_weak_kernels_for_culling(
        self,
        phi_threshold: float = 0.5,
        min_age_seconds: float = 10.0,
        limit: int = 50
    ) -> List[Dict]:
        """
        Fetch kernels that are candidates for phi-selection culling.
        
        Returns kernels with phi below threshold, excluding those in observation
        or younger than the grace period.
        
        Args:
            phi_threshold: Kernels with phi below this are candidates
            min_age_seconds: Minimum age before kernel can be culled (grace period)
            limit: Maximum kernels to return
        
        Returns:
            List of weak kernel dicts ordered by phi ascending (weakest first)
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=min_age_seconds)
        
        query = """
            SELECT kernel_id, god_name, domain, status, phi, kappa,
                   success_count, failure_count, generation,
                   observation_status, spawned_at
            FROM kernel_geometry
            WHERE status IN ('active', 'graduated', 'idle')
              AND observation_status != 'observing'
              AND phi < %s
              AND spawned_at < %s
            ORDER BY phi ASC
            LIMIT %s
        """
        
        try:
            results = self.execute_query(query, (phi_threshold, cutoff, limit))
            return results or []
        except Exception as e:
            print(f"[KernelPersistence] Weak kernel query failed: {e}")
            return []

    def fetch_kernels_with_pending_proposals(
        self,
        proposal_types: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch kernels that have pending governance proposals.
        
        Proposals are stored in metadata->>'pending_proposal_type'.
        
        Args:
            proposal_types: Filter by proposal types (spawn, death)
            limit: Maximum kernels to return
        
        Returns:
            List of kernels with pending proposals
        """
        if proposal_types:
            query = """
                SELECT kernel_id, god_name, domain, status, phi,
                       success_count, failure_count,
                       metadata->>'pending_proposal_type' as proposal_type,
                       metadata
                FROM kernel_geometry
                WHERE metadata->>'pending_proposal_type' = ANY(%s)
                  AND status NOT IN ('dead', 'cannibalized')
                LIMIT %s
            """
            params = (proposal_types, limit)
        else:
            query = """
                SELECT kernel_id, god_name, domain, status, phi,
                       success_count, failure_count,
                       metadata->>'pending_proposal_type' as proposal_type,
                       metadata
                FROM kernel_geometry
                WHERE metadata->>'pending_proposal_type' IS NOT NULL
                  AND status NOT IN ('dead', 'cannibalized')
                LIMIT %s
            """
            params = (limit,)
        
        try:
            results = self.execute_query(query, params)
            return results or []
        except Exception as e:
            print(f"[KernelPersistence] Pending proposals query failed: {e}")
            return []

    def update_kernel_proposal(
        self,
        kernel_id: str,
        proposal_type: Optional[str],
        proposal_data: Optional[Dict] = None
    ) -> bool:
        """
        Update kernel's pending proposal in metadata.
        
        Args:
            kernel_id: Kernel to update
            proposal_type: 'spawn', 'death', or None to clear
            proposal_data: Optional additional proposal details
        
        Returns:
            True if update succeeded
        """
        if proposal_type is None:
            query = """
                UPDATE kernel_geometry
                SET metadata = metadata - 'pending_proposal_type' - 'pending_proposal_data'
                WHERE kernel_id = %s
            """
            params = (kernel_id,)
        else:
            query = """
                UPDATE kernel_geometry
                SET metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object(
                    'pending_proposal_type', %s,
                    'pending_proposal_data', %s,
                    'proposal_created_at', %s
                )
                WHERE kernel_id = %s
            """
            params = (
                proposal_type,
                json.dumps(proposal_data or {}),
                datetime.now(timezone.utc).isoformat(),
                kernel_id
            )
        
        try:
            self.execute_query(query, params, fetch=False)
            return True
        except Exception as e:
            print(f"[KernelPersistence] Failed to update proposal: {e}")
            return False

    def fetch_strongest_kernel_near(
        self,
        target_coords: List[float],
        radius: float = 2.0,
        exclude_id: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Find the strongest kernel within geometric radius of target.
        
        Useful for finding cannibalization/merge targets.
        
        Args:
            target_coords: 64D basin coordinates
            radius: Maximum L2 distance to consider
            exclude_id: Kernel to exclude (the dying kernel itself)
        
        Returns:
            Strongest kernel dict or None
        """
        coords_64d = ensure_64d_coords(target_coords)
        
        exclude_clause = ""
        params = [coords_64d, radius]
        
        if exclude_id:
            exclude_clause = "AND kernel_id != %s"
            params.append(exclude_id)
        
        query = f"""
            SELECT kernel_id, god_name, domain, status, phi, kappa,
                   success_count, failure_count, generation,
                   basin_coordinates <-> %s::vector(64) as distance
            FROM kernel_geometry
            WHERE status IN ('active', 'graduated', 'idle')
              AND basin_coordinates IS NOT NULL
              AND basin_coordinates <-> %s::vector(64) < %s
              {exclude_clause}
            ORDER BY phi DESC, success_count DESC
            LIMIT 1
        """
        
        params_final = [coords_64d, coords_64d, radius]
        if exclude_id:
            params_final.append(exclude_id)
        
        try:
            results = self.execute_query(query, tuple(params_final))
            return results[0] if results else None
        except Exception as e:
            print(f"[KernelPersistence] Strongest kernel query failed: {e}")
            return None

    def ensure_basin_index(self) -> bool:
        """
        Ensure pgvector HNSW index exists on basin_coordinates.
        
        HNSW provides O(log n) approximate nearest neighbor queries.
        Should be called once during startup.
        
        Note: Uses non-CONCURRENT index creation to work within transactions.
        For large tables in production, run manually with CONCURRENTLY outside transactions.
        
        Returns:
            True if index exists or was created
        """
        check_query = """
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'kernel_geometry' 
              AND indexname = 'idx_kernel_geometry_basin_ann'
        """
        
        try:
            results = self.execute_query(check_query)
            if results:
                print("[KernelPersistence] HNSW index already exists on basin_coordinates")
                return True
            
            # Use non-CONCURRENT for compatibility with transactions
            # Production deployments with large tables should create index manually with CONCURRENTLY
            create_query = """
                CREATE INDEX IF NOT EXISTS idx_kernel_geometry_basin_ann
                ON kernel_geometry
                USING hnsw (basin_coordinates vector_l2_ops)
                WITH (m = 16, ef_construction = 64)
            """
            self.execute_query(create_query, fetch=False)
            print("[KernelPersistence] Created HNSW index on basin_coordinates")
            return True
        except Exception as e:
            # Index creation may fail if pgvector extension not available
            # or if basin_coordinates column doesn't exist yet
            print(f"[KernelPersistence] Failed to ensure basin index (non-fatal): {e}")
            return False
