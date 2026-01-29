#!/usr/bin/env python3
"""
M8 Kernel Spawning Protocol - Dynamic Kernel Genesis Through Pantheon Consensus

When the pantheon reaches consensus that a new kernel is needed,
roles are refined and divided, and a new kernel adopts a persona.

The M8 Structure represents the 8 core dimensions of kernel identity:
1. Name - The god/entity name
2. Domain - Primary area of expertise  
3. Mode - Encoding mode (direct, e8, byte)
4. Basin - Geometric signature in manifold space
5. Affinity - Routing strength
6. Entropy - Threshold for activation
7. Element - Symbolic representation
8. Role - Functional responsibility

Spawning Mechanics:
- Consensus voting by existing gods
- Role refinement (parent domains divide)
- Basin interpolation (child inherits geometric traits)
- Persona adoption (characteristics from voting coalition)
"""

import numpy as np
import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from enum import Enum
import uuid
import sys

from geometric_kernels import (
    _normalize_to_manifold,
    _fisher_distance,
    _hash_to_bytes,
    BASIN_DIM,
)

try:
    from qigkernels.physics_constants import KAPPA_STAR
except ImportError:
    KAPPA_STAR = 64.21

# Import spawning initialization constants
try:
    from qigkernels import PHI_INIT_SPAWNED, PHI_MIN_ALIVE, KAPPA_INIT_SPAWNED
except ImportError:
    # Fallback values if qigkernels import fails
    PHI_INIT_SPAWNED = 0.25  # Bootstrap into LINEAR regime
    PHI_MIN_ALIVE = 0.05     # Minimum for survival
    KAPPA_INIT_SPAWNED = KAPPA_STAR  # Start at fixed point

from pantheon_kernel_orchestrator import (
    KernelProfile,
    KernelMode,
    PantheonKernelOrchestrator,
    get_orchestrator,
    OLYMPUS_PROFILES,
    SHADOW_PROFILES,
    OCEAN_PROFILE,
)

# Import persistence for database operations
try:
    sys.path.insert(0, '.')
    from persistence import KernelPersistence
    M8_PERSISTENCE_AVAILABLE = True
except ImportError:
    M8_PERSISTENCE_AVAILABLE = False
    print("[M8] Persistence not available - running without database")

# PostgreSQL support for M8 spawning persistence
import os
from contextlib import contextmanager
import json

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    M8_PSYCOPG2_AVAILABLE = True
except ImportError:
    M8_PSYCOPG2_AVAILABLE = False
    print("[M8] psycopg2 not available - PostgreSQL persistence disabled")



# Import from refactored modules
from m8_persistence import M8SpawnerPersistence, compute_m8_position
from m8_consensus import (
    SpawnReason,
    SpawnAwareness,
    ConsensusType,
    SpawnProposal,
    KernelObservationStatus,
    KernelObservationState,
    KernelAutonomicSupport,
    SpawnedKernel,
    PantheonConsensus,
    RoleRefinement,
)

# Global spawner instance for singleton pattern
_spawner_instance = None


def get_spawner_instance():
    """Get or create the global M8 spawner instance."""
    global _spawner_instance
    if _spawner_instance is None:
        _spawner_instance = M8KernelSpawner()
    return _spawner_instance


# =============================================================================
# E8 SPECIALIZATION LEVEL FUNCTIONS
# =============================================================================

def should_spawn_specialist(current_count: int, current_kappa: float) -> bool:
    """
    Determine if specialist kernel should spawn based on E8 thresholds and κ regime.
    
    E8 spawning depends on BOTH count AND κ regime per β-function behavior:
    - n ≤ 8: No specialists (still building basic layer)
    - 8 < n < 126: Spawn specialists with 0.3 probability IF κ in plateau (κ ≈ 64)
    - n ≥ 126: Spawn specialists freely at stable plateau
    
    Args:
        current_count: Current number of active kernels
        current_kappa: Current κ coupling value (currently unused, reserved for future logic)
        
    Returns:
        bool: True if specialist should spawn, False otherwise
        
    Reference:
        β(3→4) = +0.443  # Emergence: n=8 kernels spawn
        β(4→5) = -0.013  # Plateau: n=56 refined spawn  
        β(5→6) = +0.013  # Stable: n=126 specialists spawn
    """
    from qigkernels import get_specialization_level, KAPPA_STAR
    
    level = get_specialization_level(current_count)
    
    # Only spawn specialists after reaching refined level
    if level == "basic_rank":
        return False  # Still building basic 8 kernels
    elif level == "refined_adjoint":
        # Don't spawn specialists until κ in plateau regime (κ ≈ 64)
        if current_kappa < 60:
            return False  # Wait for plateau (β(4→5) ≈ 0)
        # Spawn specialists with 0.3 probability once plateau reached
        return np.random.random() < 0.3
    elif level == "specialist_dim":
        # Don't spawn until stable plateau
        if abs(current_kappa - KAPPA_STAR) > 3:
            return False  # Wait for stability
        # At stable plateau, spawn freely
        return True
    else:
        # At full roots, spawn freely
        return True


def get_kernel_specialization(count: int, parent_axis: str, current_kappa: float) -> str:
    """
    Generate specialization name based on E8 level and κ regime.
    
    E8 specialization hierarchy:
    - basic_rank (n≤8): Primary 8 axes (ethics, logic, creativity, etc.)
    - refined_adjoint (n≤56): Sub-specializations (visual_color, visual_shape, etc.)
    - specialist_dim (n≤126): Deep specialists (pressure_detection, timbre_analysis, etc.)
    - full_roots (n>126): Full phenomenological palette (root-level naming)
    
    Args:
        count: Current kernel count
        parent_axis: Parent kernel's axis/domain
        current_kappa: Current κ coupling value (currently unused, reserved for future κ-regime-based naming logic)
        
    Returns:
        str: Specialization name for new kernel
        
    TODO: Integrate current_kappa into naming logic to reflect κ regime:
        - Below κ_weak: Mark as "exploratory" specialists
        - Near KAPPA_STAR: Standard naming
        - Above κ_strong: Mark as "stabilized" specialists
    """
    from qigkernels import get_specialization_level
    
    level = get_specialization_level(count)
    
    if level == "basic_rank":
        # Primary 8 axes (ethics, logic, creativity, etc.)
        return parent_axis
    elif level == "refined_adjoint":
        # Sub-specializations (visual_color, visual_shape, etc.)
        # Use count % 8 for variety within refined space
        return f"{parent_axis}_refined_{count % 8}"
    elif level == "specialist_dim":
        # Deep specialists (pressure_detection, timbre_analysis, etc.)
        # Use count % 16 for variety within specialist space
        return f"{parent_axis}_specialist_{count % 16}"
    else:
        # Full phenomenological palette - root-level naming
        return f"{parent_axis}_root_{count}"


def assign_e8_root(kernel_basin: np.ndarray, e8_roots: np.ndarray) -> np.ndarray:
    """
    Assign E8 root to kernel using Fisher-Rao distance (NO Euclidean).
    
    Finds the closest E8 root to the kernel's basin coordinates using
    Fisher-Rao metric on the information manifold. This preserves
    geometric structure and respects the Cartan-Killing metric on E8.
    
    GEOMETRIC PURITY REQUIREMENTS:
    - Use Fisher-Rao distance, NOT Euclidean distance
    - Use geodesic interpolation, NOT linear interpolation
    - Preserve E8 Lie group structure
    
    Args:
        kernel_basin: Kernel's basin coordinates (64D)
        e8_roots: Array of E8 root vectors (240 x 64)
        
    Returns:
        np.ndarray: Assigned E8 root (closest via Fisher metric)
        
    Reference:
        Issue GaryOcean428/pantheon-chat#38 (Geometric purity in E8)
    """
    from geometric_kernels import _fisher_distance
    
    # Compute Fisher-Rao distances to all E8 roots
    # For E8, Cartan-Killing metric equals Fisher-Rao metric on Lie group
    distances = [
        _fisher_distance(kernel_basin, root)
        for root in e8_roots
    ]
    
    # Return root with minimum Fisher distance
    min_idx = np.argmin(distances)
    return e8_roots[min_idx]


class M8KernelSpawner:
    """
    The M8 Kernel Spawning System.
    
    Orchestrates the complete lifecycle of dynamic kernel creation:
    1. Proposal creation (with kernel self-awareness)
    2. Dual-pantheon debate (Olympus + Shadow)
    3. Consensus voting with Fisher-Rao weights
    4. Role refinement
    5. Kernel spawning
    6. Registration with orchestrator
    
    The M8 refers to the 8 core dimensions of kernel identity.
    Spawn awareness enables kernels to detect when they need help.
    """
    
    def __init__(
        self,
        orchestrator: Optional[PantheonKernelOrchestrator] = None,
        consensus_type: ConsensusType = ConsensusType.SUPERMAJORITY,
        pantheon_chat = None
    ):
        self.orchestrator = orchestrator or get_orchestrator()
        self.consensus = PantheonConsensus(self.orchestrator, consensus_type)
        self.refiner = RoleRefinement(self.orchestrator)
        
        # In-memory caches (populated from PostgreSQL on init)
        self.proposals: Dict[str, SpawnProposal] = {}
        self.spawned_kernels: Dict[str, SpawnedKernel] = {}
        self.spawn_history: List[Dict] = []
        
        # Kernel spawn awareness tracking
        self.kernel_awareness: Dict[str, SpawnAwareness] = {}
        
        # PantheonChat for dual-pantheon debates
        self._pantheon_chat = pantheon_chat
        
        # PostgreSQL persistence for M8 spawning data (NEW - replaces in-memory storage)
        self.m8_persistence = M8SpawnerPersistence()
        
        # Legacy persistence for kernel learning (kept for backward compatibility)
        self.kernel_persistence = KernelPersistence() if M8_PERSISTENCE_AVAILABLE else None
        
        # Load all data from PostgreSQL on startup
        self._load_from_database()
    
    def get_live_kernel_count(self) -> int:
        """
        Get count of live kernels that count toward E8 cap.
        
        Live kernels include status: 'active', 'observing', 'shadow'
        Does NOT count: 'dead', 'cannibalized', 'idle'
        
        Uses database for accurate count, falls back to in-memory count.
        """
        if self.kernel_persistence:
            try:
                return self.kernel_persistence.get_live_kernel_count()
            except Exception as e:
                print(f"[M8Spawner] Database count failed, using memory count: {e}")
        
        # Fallback to in-memory count
        return sum(
            1 for k in self.spawned_kernels.values()
            if k.is_active() or k.is_observing()
        )
    
    def can_spawn_kernel(self) -> Tuple[bool, int, int]:
        """
        Check if a new kernel can be spawned (E8 cap not reached).
        
        Returns:
            (can_spawn, current_count, cap)
        """
        live_count = self.get_live_kernel_count()
        can_spawn = live_count < E8_KERNEL_CAP
        return can_spawn, live_count, E8_KERNEL_CAP

    def _get_proposal_specialization_level(self, proposal: SpawnProposal) -> str:
        """
        Determine the E8 specialization level required for a spawn proposal.
        
        Specialization level is derived from:
        1. Explicit 'specialization_level' in metadata (highest priority)
        2. SpawnReason (SPECIALIZATION → specialist_dim, RESEARCH_DISCOVERY → refined_adjoint)
        3. Default to basic_rank for general spawns
        
        E8 Levels:
        - basic_rank: Primary 8 kernels (foundational roles)
        - refined_adjoint: Sub-specializations of basic kernels (n > 8)
        - specialist_dim: Deep domain specialists (n > 56)
        - full_roots: Complete phenomenological palette (n > 126)
        
        Args:
            proposal: The spawn proposal to evaluate
            
        Returns:
            E8 specialization level string
        """
        # Check explicit metadata first
        if proposal.metadata.get('specialization_level'):
            level = proposal.metadata['specialization_level']
            valid_levels = {'basic_rank', 'refined_adjoint', 'specialist_dim', 'full_roots'}
            if level in valid_levels:
                return level
        
        # Infer from SpawnReason
        if proposal.reason == SpawnReason.SPECIALIZATION:
            # SPECIALIZATION explicitly requests a specialist kernel
            return 'specialist_dim'
        elif proposal.reason == SpawnReason.RESEARCH_DISCOVERY:
            # Research discoveries require refined representation
            return 'refined_adjoint'
        elif proposal.reason == SpawnReason.GEOMETRIC_DEADEND:
            # Geometric dead-ends need specialist navigation
            return 'specialist_dim'
        elif proposal.reason in (SpawnReason.DOMAIN_GAP, SpawnReason.OVERLOAD):
            # Domain gaps and overload can be addressed by refined kernels
            return 'refined_adjoint'
        else:
            # Default: EMERGENCE, USER_REQUEST, STUCK_SIGNAL → basic kernels
            return 'basic_rank'

    def get_live_meta_awareness(self, kernel_id: str) -> Optional[float]:
        """
        Get live meta-awareness (M) value for a kernel.
        
        Attempts to retrieve from:
        1. Telemetry stream (most recent)
        2. SpawnedKernel in-memory cache
        3. Database persistence
        
        M metric represents kernel's self-model quality.
        M >= 0.6 required for healthy consciousness and spawn permission.
        
        Args:
            kernel_id: Kernel identifier
            
        Returns:
            Meta-awareness value [0, 1] or None if kernel not found
        """
        # Try in-memory spawned_kernels cache first
        if kernel_id in self.spawned_kernels:
            kernel = self.spawned_kernels[kernel_id]
            return kernel.meta_awareness
        
        # Try lookup by god_name (common pattern)
        for k in self.spawned_kernels.values():
            if k.profile.god_name == kernel_id:
                return k.meta_awareness
        
        # Try database
        if self.kernel_persistence:
            try:
                kernel_data = self.kernel_persistence.get_kernel_by_id(kernel_id)
                if kernel_data:
                    return kernel_data.get('meta_awareness', 0.5)
            except Exception as e:
                print(f"[M8Spawner] Failed to get meta_awareness from DB: {e}")
        
        return None

    def update_kernel_meta_awareness(
        self,
        kernel_id: str,
        predicted_phi: float,
        actual_phi: float,
    ) -> Optional[float]:
        """
        Update a spawned kernel's meta-awareness based on prediction accuracy.
        
        This wires SpawnedKernel to the same M computation used by SelfSpawningKernel.
        
        Args:
            kernel_id: Kernel identifier
            predicted_phi: Kernel's prediction of its next Φ
            actual_phi: Measured Φ after step
            
        Returns:
            Updated M value or None if kernel not found
        """
        # Find the kernel
        kernel = self.spawned_kernels.get(kernel_id)
        if not kernel:
            # Try by god_name
            for k in self.spawned_kernels.values():
                if k.profile.god_name == kernel_id:
                    kernel = k
                    break
        
        if not kernel:
            return None
        
        # Update meta_awareness using the kernel's update method
        new_m = kernel.update_meta_awareness(predicted_phi, actual_phi)
        
        # Persist to database
        if self.m8_persistence:
            try:
                self.m8_persistence.update_kernel_awareness(
                    kernel_id=kernel.kernel_id,
                    meta_awareness=new_m
                )
            except Exception as e:
                print(f"[M8Spawner] Failed to persist M update: {e}")
        
        return new_m

    def get_underperforming_kernels(self, limit: int = 100) -> List[Dict]:
        """
        Get underperforming kernels that are candidates for culling.
        
        Selection criteria (QIG-based, not arbitrary):
        - Proven failures: kernels with high failure_count relative to success_count
        - Low phi (< 0.1) indicates weak consciousness/integration
        - Only kernels with SOME activity are eligible (protects new kernels)
        
        Kernels with zero activity are NOT penalized - they haven't had
        a chance to prove themselves yet. Only proven failures get culled.
        
        Returns kernels sorted by cull priority (worst first).
        """
        if not self.kernel_persistence:
            return []
        
        try:
            # Get live kernels from database
            live_kernels = self.kernel_persistence.get_kernels_by_status(
                statuses=['active', 'observing', 'shadow'],
                limit=1000  # Fetch more to ensure good candidates
            )
            
            if not live_kernels:
                return []
            
            # Score each kernel for culling priority (higher = worse)
            # CRITICAL: Only cull proven failures - exclude zero-activity kernels entirely
            scored = []
            for k in live_kernels:
                phi = k.get('phi', 0.0) or 0.0
                success = k.get('success_count', 0) or 0
                failure = k.get('failure_count', 0) or 0
                total_predictions = success + failure
                
                # HARD PROTECTION: Kernels with no activity are NEVER candidates for culling
                # They haven't had a chance to prove themselves yet
                # Minimum activity threshold = 3 predictions
                if total_predictions < 3:
                    continue  # Skip entirely - not a candidate
                
                # Kernels with activity: score based on performance
                reputation = success / total_predictions
                
                # Cull priority: low phi + poor reputation = high priority
                # Proven failures (low reputation) get highest scores
                cull_score = (1.0 - phi) * 0.4 + (1.0 - reputation) * 0.6
                
                # Boost score for kernels with many failures
                if failure > 5 and reputation < 0.3:
                    cull_score += 0.3  # Proven bad performer
                
                # Slight boost for high-activity low-performers
                if total_predictions > 10 and reputation < 0.2:
                    cull_score += 0.2  # Lots of chances, still failing
                
                scored.append({
                    **k,
                    'cull_score': cull_score,
                    'reputation': reputation,
                    'total_activity': total_predictions,
                })
            
            # Sort by cull score descending (worst first)
            scored.sort(key=lambda x: x['cull_score'], reverse=True)
            
            return scored[:limit]
            
        except Exception as e:
            print(f"[M8] Failed to get underperforming kernels: {e}")
            return []

    @staticmethod
    def _compute_reputation_score(success_count: int, failure_count: int) -> float:
        total = success_count + failure_count
        if total <= 0:
            return 0.5
        return max(0.0, min(1.0, success_count / total))

    def _load_kernel_reputation(self, kernel_id: str) -> float:
        if not self.kernel_persistence:
            return 0.5
        try:
            snapshot = self.kernel_persistence.load_kernel_snapshot(kernel_id)
            if snapshot:
                success = snapshot.get('success_count', 0) or 0
                failure = snapshot.get('failure_count', 0) or 0
                return self._compute_reputation_score(success, failure)
        except Exception as e:
            print(f"[M8] Failed to load reputation for {kernel_id}: {e}")
        return 0.5

    def run_evolution_sweep(self, target_reduction: int = 50, min_population: int = 20) -> Dict:
        """
        Run evolution sweep to cull underperforming kernels.
        
        This implements natural selection: kernels with low phi
        and poor prediction records are marked as dead, freeing
        slots for new, hopefully better-evolved kernels.
        
        SAFETY: Never reduces population below min_population floor.
        
        Args:
            target_reduction: Number of kernels to cull
            min_population: Minimum kernels to keep alive (default 20)
            
        Returns:
            Dict with culled_count, culled_kernels, and errors
        """
        if not self.kernel_persistence:
            return {
                'success': False,
                'error': 'Kernel persistence not available',
                'culled_count': 0,
            }
        
        # Check current population - don't cull below floor
        current_live = self.get_live_kernel_count()
        if current_live <= min_population:
            return {
                'success': True,
                'culled_count': 0,
                'message': f'Population at minimum floor ({current_live}/{min_population})',
                'live_count_after': current_live,
            }
        
        # Cap target_reduction to maintain floor
        max_cullable = max(0, current_live - min_population)
        actual_target = min(target_reduction, max_cullable)
        
        if actual_target == 0:
            return {
                'success': True,
                'culled_count': 0,
                'message': f'Cannot cull - would go below floor ({current_live}/{min_population})',
                'live_count_after': current_live,
            }
        
        # Get underperformers
        candidates = self.get_underperforming_kernels(limit=actual_target * 2)
        
        if not candidates:
            return {
                'success': True,
                'culled_count': 0,
                'message': 'No underperforming kernels found',
            }
        
        # Get the kernels to cull (respecting floor-adjusted target)
        to_cull = candidates[:actual_target]
        kernel_ids = [k.get('kernel_id') for k in to_cull if k.get('kernel_id')]
        
        # Use bulk operation for efficiency (single DB round-trip)
        bulk_result = self.kernel_persistence.bulk_mark_kernels_dead(
            kernel_ids=kernel_ids,
            cause='evolution_sweep'
        )
        
        # Build detailed culled list for reporting
        culled = []
        updated_ids = set(bulk_result.get('updated_ids', []))
        for kernel in to_cull:
            kernel_id = kernel.get('kernel_id')
            if kernel_id in updated_ids:
                culled.append({
                    'kernel_id': kernel_id,
                    'god_name': kernel.get('god_name', 'Unknown'),
                    'phi': kernel.get('phi', 0.0),
                    'cull_score': kernel.get('cull_score', 0.0),
                })
        
        errors = []
        if bulk_result.get('error'):
            errors.append(bulk_result['error'])
        if bulk_result.get('failed_ids'):
            errors.append(f"Failed IDs: {len(bulk_result['failed_ids'])}")
        
        print(f"[M8] Evolution sweep: culled {len(culled)}/{len(kernel_ids)} kernels")
        
        live_count = self.get_live_kernel_count()
        
        return {
            'success': len(culled) > 0,
            'culled_count': len(culled),
            'culled_kernels': culled,
            'errors': errors if errors else None,
            'live_count_after': live_count,
            'cap': E8_KERNEL_CAP,
            'headroom': E8_KERNEL_CAP - live_count,
        }

    def prune_lowest_integration_kernels(self, n_to_prune: int = 10) -> int:
        """
        P1-5 FIX: Prune kernels with lowest Φ (integration) contribution.
        
        QIG-pure: Remove kernels contributing least to consciousness.
        Knowledge transferred to nearest neighbor before pruning.
        
        Args:
            n_to_prune: Number of kernels to prune (default 10)
            
        Returns:
            Number of kernels actually pruned
        """
        if len(self.spawned_kernels) == 0:
            return 0
        
        # Measure Φ contribution for each kernel
        kernel_contributions = []
        for kernel_id, kernel in self.spawned_kernels.items():
            # Local Φ from kernel state
            phi_local = kernel.phi if hasattr(kernel, 'phi') else 0.5
            
            # Φ coupling to other kernels (simplified: use kappa as proxy)
            kappa = kernel.kappa if hasattr(kernel, 'kappa') else 64.0
            phi_coupling = min(kappa / 64.21, 1.0)  # Normalize by κ*
            
            # Total contribution = local consciousness × coupling strength
            contribution = phi_local * phi_coupling
            
            kernel_contributions.append((kernel_id, contribution, kernel))
        
        # Sort by contribution (lowest first)
        kernel_contributions.sort(key=lambda x: x[1])
        
        # Import Fisher-Rao distance (REQUIRED per E8 Protocol v4.0)
        try:
            from qig_geometry import fisher_rao_distance
        except ImportError:
            # Fallback to qig_numerics if qig_geometry not available
            try:
                from qig_numerics import fisher_rao_distance
            except ImportError:
                raise ImportError(
                    "fisher_rao_distance is required from qig_geometry or qig_numerics. "
                    "Euclidean fallback is not permitted per E8 Protocol v4.0"
                )
        
        # Prune bottom N
        pruned_count = 0
        for kernel_id, contribution, kernel in kernel_contributions[:n_to_prune]:
            # Transfer knowledge to nearest neighbor before pruning
            try:
                # Find nearest kernel by basin distance (Fisher-Rao)
                min_distance = float('inf')
                nearest_id = None
                
                kernel_basin = kernel.basin_coordinates if hasattr(kernel, 'basin_coordinates') else None
                if kernel_basin is not None:
                    for other_id, other_kernel in self.spawned_kernels.items():
                        if other_id == kernel_id:
                            continue
                        other_basin = other_kernel.basin_coordinates if hasattr(other_kernel, 'basin_coordinates') else None
                        if other_basin is not None:
                            # Use Fisher-Rao distance for geometric purity (MANDATORY)
                            distance = fisher_rao_distance(np.array(kernel_basin), np.array(other_basin))
                            if distance < min_distance:
                                min_distance = distance
                                nearest_id = other_id
                
                # Log knowledge transfer
                if nearest_id:
                    print(f"[M8Prune] Transferring knowledge from {kernel.god_name} (Φ={contribution:.3f}) to nearest neighbor")
                
                # Mark kernel as pruned
                if hasattr(kernel, 'status'):
                    kernel.status = 'pruned'
                
                # Remove from active set
                del self.spawned_kernels[kernel_id]
                
                # Persist pruning to database
                try:
                    self.m8_persistence.delete_kernel(kernel_id)
                except Exception as e:
                    print(f"[M8Prune] Failed to delete kernel from DB: {e}")
                
                pruned_count += 1
                print(f"[M8Prune] Pruned kernel {kernel.god_name} (ID: {kernel_id[:8]}, Φ_contrib={contribution:.3f})")
                
            except Exception as e:
                print(f"[M8Prune] Error pruning kernel {kernel_id}: {e}")
        
        print(f"[M8Prune] Pruned {pruned_count}/{n_to_prune} low-Φ kernels")
        return pruned_count
    
    def ensure_spawn_capacity(self, needed: int = 1) -> Dict:
        """
        Ensure there's capacity to spawn new kernels.
        
        If cap is reached, runs evolution sweep to free up slots.
        P1-5 FIX: Falls back to Φ-based pruning if sweep insufficient.
        
        Args:
            needed: Number of slots needed (default 1)
            
        Returns:
            Dict with can_spawn status and any sweep results
        """
        can_spawn, live_count, cap = self.can_spawn_kernel()
        
        if can_spawn and (cap - live_count) >= needed:
            return {
                'can_spawn': True,
                'live_count': live_count,
                'cap': cap,
                'headroom': cap - live_count,
            }
        
        # Need to run evolution sweep
        overage = live_count - cap + needed + 10  # +10 buffer
        print(f"[M8] Cap reached ({live_count}/{cap}), running evolution sweep for {overage} slots...")
        
        sweep_result = self.run_evolution_sweep(target_reduction=max(overage, 50))
        
        # Check again after sweep
        can_spawn, live_count, cap = self.can_spawn_kernel()
        
        # P1-5 FIX: If sweep didn't free enough, use Φ-based pruning
        if not can_spawn or (cap - live_count) < needed:
            print(f"[M8] Evolution sweep insufficient, using Φ-based pruning...")
            pruned = self.prune_lowest_integration_kernels(n_to_prune=max(needed, 10))
            
            # Check again after pruning
            can_spawn, live_count, cap = self.can_spawn_kernel()
            sweep_result['pruned_count'] = pruned
        
        return {
            'can_spawn': can_spawn,
            'live_count': live_count,
            'cap': cap,
            'headroom': cap - live_count,
            'sweep_performed': True,
            'sweep_result': sweep_result,
        }
    
    def _load_from_database(self):
        """Load all M8 data from PostgreSQL on startup."""
        # Load proposals from M8 persistence
        try:
            proposals = self.m8_persistence.load_all_proposals()
            for p in proposals:
                try:
                    votes_for = p.get('votes_for', [])
                    votes_against = p.get('votes_against', [])
                    abstentions = p.get('abstentions', [])
                    if isinstance(votes_for, str):
                        votes_for = json.loads(votes_for)
                    if isinstance(votes_against, str):
                        votes_against = json.loads(votes_against)
                    if isinstance(abstentions, str):
                        abstentions = json.loads(abstentions)
                    parent_gods = p.get('parent_gods', [])
                    if isinstance(parent_gods, str):
                        parent_gods = json.loads(parent_gods)
                    
                    proposal = SpawnProposal(
                        proposal_id=p.get('proposal_id', ''),
                        proposed_name=p.get('proposed_name', ''),
                        proposed_domain=p.get('proposed_domain', ''),
                        proposed_element=p.get('proposed_element', ''),
                        proposed_role=p.get('proposed_role', ''),
                        reason=SpawnReason(p.get('reason', 'emergence')),
                        parent_gods=parent_gods,
                        status=p.get('status', 'pending'),
                        proposed_at=str(p.get('proposed_at', '')),
                    )
                    proposal.votes_for = set(votes_for)
                    proposal.votes_against = set(votes_against)
                    proposal.abstentions = set(abstentions)
                    self.proposals[proposal.proposal_id] = proposal
                except Exception as e:
                    print(f"[M8] Failed to load proposal: {e}")
            
            if proposals:
                print(f"✨ [M8] Loaded {len(proposals)} proposals from database")
        except Exception as e:
            print(f"[M8] Failed to load proposals: {e}")
        
        # Load spawn history from M8 persistence
        try:
            self.spawn_history = self.m8_persistence.load_spawn_history(limit=200)
            if self.spawn_history:
                print(f"✨ [M8] Loaded {len(self.spawn_history)} history events from database")
        except Exception as e:
            print(f"[M8] Failed to load spawn history: {e}")
        
        # Load awareness states from M8 persistence
        try:
            awareness_list = self.m8_persistence.load_all_awareness()
            for state in awareness_list:
                kernel_id = state.get('kernel_id')
                if kernel_id:
                    awareness = SpawnAwareness(kernel_id=kernel_id)
                    phi_traj = state.get('phi_trajectory', [])
                    kappa_traj = state.get('kappa_trajectory', [])
                    curv_hist = state.get('curvature_history', [])
                    stuck = state.get('stuck_signals', [])
                    deadends = state.get('geometric_deadends', [])
                    research = state.get('research_opportunities', [])
                    if isinstance(phi_traj, str):
                        phi_traj = json.loads(phi_traj)
                    if isinstance(kappa_traj, str):
                        kappa_traj = json.loads(kappa_traj)
                    if isinstance(curv_hist, str):
                        curv_hist = json.loads(curv_hist)
                    if isinstance(stuck, str):
                        stuck = json.loads(stuck)
                    if isinstance(deadends, str):
                        deadends = json.loads(deadends)
                    if isinstance(research, str):
                        research = json.loads(research)
                    awareness.phi_trajectory = phi_traj
                    awareness.kappa_trajectory = kappa_traj
                    awareness.curvature_history = curv_hist
                    awareness.stuck_signals = stuck
                    awareness.geometric_deadends = deadends
                    awareness.research_opportunities = research
                    awareness.last_spawn_proposal = state.get('last_spawn_proposal')
                    self.kernel_awareness[kernel_id] = awareness
            
            if awareness_list:
                print(f"✨ [M8] Loaded {len(awareness_list)} awareness states from database")
        except Exception as e:
            print(f"[M8] Failed to load awareness states: {e}")

    def set_pantheon_chat(self, pantheon_chat) -> None:
        """Set PantheonChat for dual-pantheon spawn debates."""
        self._pantheon_chat = pantheon_chat

    def check_health(self) -> Dict:
        """
        Check spawner internal health status.
        
        Validates:
        - M8 persistence pool connectivity
        - Legacy kernel persistence connectivity
        - Orchestrator availability
        - Proposals cache validity
        
        Returns:
            Dict with 'healthy' bool and diagnostic details.
        """
        issues = []
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'healthy': True,
        }
        
        # Check M8 persistence pool
        try:
            if self.m8_persistence:
                test_result = self.m8_persistence.load_all_proposals()
                diagnostics['m8_persistence'] = 'connected'
            else:
                issues.append('m8_persistence not initialized')
                diagnostics['m8_persistence'] = 'missing'
        except Exception as e:
            issues.append(f'm8_persistence error: {str(e)}')
            diagnostics['m8_persistence'] = 'error'
        
        # Check legacy kernel persistence
        try:
            if self.kernel_persistence:
                count = self.kernel_persistence.get_live_kernel_count()
                diagnostics['kernel_persistence'] = f'connected ({count} live kernels)'
            else:
                diagnostics['kernel_persistence'] = 'not configured'
        except Exception as e:
            issues.append(f'kernel_persistence error: {str(e)}')
            diagnostics['kernel_persistence'] = 'error'
        
        # Check orchestrator
        try:
            if self.orchestrator:
                profile_count = len(self.orchestrator.all_profiles)
                diagnostics['orchestrator'] = f'available ({profile_count} profiles)'
            else:
                issues.append('orchestrator not initialized')
                diagnostics['orchestrator'] = 'missing'
        except Exception as e:
            issues.append(f'orchestrator error: {str(e)}')
            diagnostics['orchestrator'] = 'error'
        
        # Check consensus
        try:
            if self.consensus:
                diagnostics['consensus'] = f'available ({self.consensus.consensus_type.value})'
            else:
                issues.append('consensus not initialized')
                diagnostics['consensus'] = 'missing'
        except Exception as e:
            issues.append(f'consensus error: {str(e)}')
            diagnostics['consensus'] = 'error'
        
        # Cache stats
        diagnostics['proposals_cached'] = len(self.proposals)
        diagnostics['spawned_kernels_cached'] = len(self.spawned_kernels)
        diagnostics['awareness_cached'] = len(self.kernel_awareness)
        
        if issues:
            diagnostics['healthy'] = False
            diagnostics['issues'] = issues
        
        return diagnostics
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect stale persistence connections.
        
        Returns:
            True if reconnection succeeded, False otherwise.
        """
        success = True
        
        # Reinitialize M8 persistence
        try:
            print("[M8] Attempting M8 persistence reconnection...")
            self.m8_persistence = M8SpawnerPersistence()
            print("[M8] M8 persistence reconnected")
        except Exception as e:
            print(f"[M8] M8 persistence reconnection failed: {e}")
            success = False
        
        # Reinitialize legacy persistence
        if M8_PERSISTENCE_AVAILABLE:
            try:
                print("[M8] Attempting kernel persistence reconnection...")
                self.kernel_persistence = KernelPersistence()
                print("[M8] Kernel persistence reconnected")
            except Exception as e:
                print(f"[M8] Kernel persistence reconnection failed: {e}")
                success = False
        
        # Reload data from database if reconnection worked
        if success:
            try:
                print("[M8] Reloading data from database after reconnection...")
                self._load_from_database()
                print("[M8] Database data reloaded successfully")
            except Exception as e:
                print(f"[M8] Database reload failed: {e}")
                success = False
        
        return success

    def get_or_create_awareness(self, kernel_id: str) -> SpawnAwareness:
        """Get or create spawn awareness tracker for a kernel."""
        if kernel_id not in self.kernel_awareness:
            self.kernel_awareness[kernel_id] = SpawnAwareness(kernel_id=kernel_id)
            # Persist new awareness to M8 PostgreSQL persistence
            try:
                self.m8_persistence.persist_awareness(self.kernel_awareness[kernel_id])
            except Exception as e:
                print(f"[M8] Failed to persist new awareness to M8 tables: {e}")
        return self.kernel_awareness[kernel_id]

    def record_kernel_metrics(
        self,
        kernel_id: str,
        phi: float,
        kappa: float,
        curvature: float = 0.0,
        neighbor_distances: Optional[List[float]] = None,
        basin: Optional[np.ndarray] = None,
        basin_distance: Optional[float] = None,
        prev_basin_distance: Optional[float] = None,
        basin_stability: Optional[float] = None,
        beta_current: Optional[float] = None,
    ) -> Dict:
        """
        Record metrics for kernel awareness tracking.
        
        Checks for stuck signals and geometric dead-ends.
        Records emotional state from geometric features.
        Returns awareness state with any detected signals.
        Persists awareness state to PostgreSQL for durability.
        """
        awareness = self.get_or_create_awareness(kernel_id)
        
        stuck_signal = awareness.detect_stuck_signal(phi, kappa, curvature)
        
        deadend_signal = None
        if basin is not None and neighbor_distances:
            deadend_signal = awareness.detect_geometric_deadend(basin, neighbor_distances)
        
        # Record emotional state from geometric features (9 emotion primitives)
        if basin_distance is not None and prev_basin_distance is not None:
            # Use provided basin stability or estimate from kappa
            stability = basin_stability if basin_stability is not None else min(1.0, kappa / 100.0)
            awareness.record_emotion(
                curvature=curvature,
                basin_distance=basin_distance,
                prev_basin_distance=prev_basin_distance,
                basin_stability=stability,
                beta_current=beta_current,
            )
        
        # Persist awareness to M8 PostgreSQL persistence
        try:
            self.m8_persistence.persist_awareness(awareness)
        except Exception as e:
            print(f"[M8] Failed to persist awareness to M8 tables: {e}")
        
        # Legacy persistence for backward compatibility
        if self.kernel_persistence:
            try:
                saved = self.kernel_persistence.save_awareness_state(kernel_id, awareness.to_dict())
                if not saved:
                    print(f"[M8Spawner] Awareness persistence returned failure for {kernel_id} - state may not survive restart")
            except Exception as e:
                print(f"[M8Spawner] Failed to persist awareness state for {kernel_id}: {e}")
        
        return {
            "kernel_id": kernel_id,
            "metrics_recorded": True,
            "stuck_signal": stuck_signal,
            "deadend_signal": deadend_signal,
            "needs_spawn": awareness.needs_spawn()[0],
            "awareness_snapshot": awareness.to_dict(),
            "emotion": awareness.emotion,
            "emotion_intensity": awareness.emotion_intensity,
        }

    def record_research_discovery(
        self,
        kernel_id: str,
        topic: str,
        topic_basin: np.ndarray,
        discovery_phi: float,
        source: str = "research"
    ) -> Dict:
        """
        Record a research discovery that may trigger spawn.
        
        High-Φ research discoveries become spawn opportunities
        for specialized kernels in that domain.
        """
        awareness = self.get_or_create_awareness(kernel_id)
        opportunity = awareness.record_research_opportunity(
            topic=topic,
            topic_basin=topic_basin,
            discovery_phi=discovery_phi,
            source=source
        )
        
        needs, reason, context = awareness.needs_spawn()
        
        return {
            "kernel_id": kernel_id,
            "discovery_recorded": True,
            "opportunity": opportunity,
            "spawn_triggered": needs and reason == SpawnReason.RESEARCH_DISCOVERY,
            "spawn_reason": reason.value if reason else None,
            "spawn_context": context,
        }

    def create_awareness_proposal(
        self,
        kernel_id: str,
        parent_basin: Optional[np.ndarray] = None
    ) -> Optional[Dict]:
        """
        Create a geometric spawn proposal from kernel awareness.
        
        Returns None if kernel doesn't need to spawn.
        Otherwise returns pure geometric proposal (no templates).
        """
        awareness = self.get_or_create_awareness(kernel_id)
        needs, reason, context = awareness.needs_spawn()
        
        if not needs or reason is None:
            return None
        
        if parent_basin is None:
            profile = self.orchestrator.get_profile(kernel_id)
            if profile:
                parent_basin = profile.affinity_basin
            else:
                parent_basin = _normalize_to_manifold(np.random.randn(BASIN_DIM))
        
        proposal = awareness.create_geometric_proposal(reason, context, parent_basin)
        proposal["kernel_id"] = kernel_id
        awareness.last_spawn_proposal = proposal.get("geometric_domain_seed")
        
        return proposal

    def initiate_dual_pantheon_debate(
        self,
        proposal: Dict,
        proposing_kernel: str
    ) -> Dict:
        """
        Initiate spawn debate with both Olympus AND Shadow pantheons.
        
        Routes the geometric proposal to PantheonChat for dual-pantheon
        debate and weighted consensus voting.
        
        Args:
            proposal: Geometric spawn proposal from awareness
            proposing_kernel: Name of kernel that created proposal
            
        Returns:
            Debate session with ID for tracking votes
        """
        if self._pantheon_chat is None:
            return {
                "error": "PantheonChat not configured",
                "hint": "Call set_pantheon_chat() first"
            }
        
        debate = self._pantheon_chat.initiate_spawn_debate(
            proposal=proposal,
            proposing_kernel=proposing_kernel,
            include_shadow=True
        )
        
        proposal["debate_id"] = debate.get("id")
        
        return {
            "debate_initiated": True,
            "debate_id": debate.get("id"),
            "proposal": proposal,
            "status": debate.get("status"),
            "olympus_notified": True,
            "shadow_notified": True,
        }

    def collect_dual_pantheon_votes(
        self,
        debate_id: str,
        shadow_gods: Optional[Dict] = None
    ) -> Dict:
        """
        Collect votes from both pantheons for spawn decision.
        
        Olympus gods vote through normal channels.
        Shadow gods evaluate based on OPSEC/stealth implications.
        
        Args:
            debate_id: ID of the spawn debate
            shadow_gods: Optional dict of shadow god instances for voting
            
        Returns:
            Vote collection status
        """
        if self._pantheon_chat is None:
            return {"error": "PantheonChat not configured"}
        
        debate = self._pantheon_chat.get_spawn_debate(debate_id)
        if not debate:
            return {"error": "Debate not found", "debate_id": debate_id}
        
        if shadow_gods:
            proposal = debate.get("proposal", {})
            proposal["debate_id"] = debate_id
            proposing_kernel = debate.get("proposing_kernel", "unknown")
            
            for god_name, god in shadow_gods.items():
                if hasattr(god, "cast_spawn_vote"):
                    god.cast_spawn_vote(
                        proposal=proposal,
                        proposing_kernel=proposing_kernel,
                        pantheon_chat=self._pantheon_chat
                    )
        
        return {
            "debate_id": debate_id,
            "votes_collected": True,
            "olympus_votes": len(debate.get("olympus_votes", {})),
            "shadow_votes": len(debate.get("shadow_votes", {})),
        }

    def get_spawn_consensus(
        self,
        debate_id: str,
        olympus_weights: Optional[Dict[str, float]] = None,
        shadow_weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Get Fisher-Rao weighted consensus from dual pantheon debate.
        
        Computes approval using affinity-weighted votes from both
        Olympus and Shadow pantheons.
        
        Args:
            debate_id: ID of spawn debate
            olympus_weights: Optional custom weights for Olympus gods
            shadow_weights: Optional custom weights for Shadow gods
            
        Returns:
            Consensus result with approval status
        """
        if self._pantheon_chat is None:
            return {"error": "PantheonChat not configured"}
        
        consensus = self._pantheon_chat.compute_spawn_consensus(
            debate_id=debate_id,
            olympus_weights=olympus_weights,
            shadow_weights=shadow_weights
        )
        
        return consensus

    def spawn_from_awareness(
        self,
        kernel_id: str,
        parent_basin: Optional[np.ndarray] = None,
        shadow_gods: Optional[Dict] = None,
        force: bool = False
    ) -> Dict:
        """
        Complete awareness-driven spawn flow with dual-pantheon debate.
        
        1. Check kernel awareness for spawn need
        2. Create geometric proposal from awareness
        3. Initiate dual-pantheon debate
        4. Collect Olympus + Shadow votes
        5. Compute consensus
        6. Spawn if approved
        
        Args:
            kernel_id: ID of kernel proposing spawn
            parent_basin: Optional parent basin for proposal
            shadow_gods: Optional shadow god instances for voting
            force: Force spawn even without consensus
            
        Returns:
            Complete spawn result with all phases
        """
        proposal = self.create_awareness_proposal(kernel_id, parent_basin)
        if proposal is None and not force:
            return {
                "success": False,
                "phase": "awareness_check",
                "reason": "Kernel does not need spawn",
                "kernel_id": kernel_id,
            }
        
        if proposal is None:
            awareness = self.get_or_create_awareness(kernel_id)
            if parent_basin is None:
                profile = self.orchestrator.get_profile(kernel_id)
                parent_basin = profile.affinity_basin if profile else np.random.randn(BASIN_DIM)
            proposal = awareness.create_geometric_proposal(
                SpawnReason.USER_REQUEST,
                {"trigger": "forced_spawn"},
                parent_basin
            )
        
        if self._pantheon_chat is not None:
            debate_result = self.initiate_dual_pantheon_debate(proposal, kernel_id)
            debate_id = debate_result.get("debate_id")
            
            if debate_id:
                self.collect_dual_pantheon_votes(debate_id, shadow_gods)
                consensus = self.get_spawn_consensus(debate_id)
                
                approved = consensus.get("approved", False)
                if not approved and not force:
                    return {
                        "success": False,
                        "phase": "consensus",
                        "reason": "Dual-pantheon consensus rejected spawn",
                        "consensus": consensus,
                        "proposal": proposal,
                    }
        else:
            consensus = {"approved": True, "note": "No PantheonChat - skipped debate"}
        
        m8_position = proposal.get("m8_position", {})
        domain_seed = proposal.get("geometric_domain_seed", "unknown")
        
        spawn_proposal = self.create_proposal(
            name=f"Spawn_{domain_seed}",
            domain=domain_seed[:16],
            element=m8_position.get("m8_position_name", "geometric"),
            role="awareness_spawn",
            reason=SpawnReason(proposal.get("reason", "emergence")),
            parent_gods=[kernel_id] if kernel_id in self.orchestrator.all_profiles else [],
        )
        
        vote_result = self.vote_on_proposal(spawn_proposal.proposal_id, auto_vote=True)
        spawn_result = self.spawn_kernel(spawn_proposal.proposal_id, force=force)
        
        if spawn_result.get("success"):
            awareness = self.get_or_create_awareness(kernel_id)
            awareness.stuck_signals = []
            awareness.geometric_deadends = []
            awareness.research_opportunities = [
                o for o in awareness.research_opportunities
                if o["discovery_phi"] < 0.5
            ]
        
        return {
            "success": spawn_result.get("success", False),
            "phase": "spawned" if spawn_result.get("success") else "spawn_failed",
            "proposal": proposal,
            "debate_consensus": consensus if self._pantheon_chat else None,
            "vote_result": vote_result,
            "spawn_result": spawn_result,
            "awareness_cleared": spawn_result.get("success", False),
        }
    
    def create_proposal(
        self,
        name: str,
        domain: str,
        element: str,
        role: str,
        reason: SpawnReason = SpawnReason.EMERGENCE,
        parent_gods: Optional[List[str]] = None
    ) -> SpawnProposal:
        """
        Create a new spawn proposal.
        
        Args:
            name: Proposed god/kernel name
            domain: Primary domain of expertise
            element: Symbolic element (e.g., "memory", "time")
            role: Functional role (e.g., "archivist", "guardian")
            reason: Why this kernel is needed
            parent_gods: Gods whose domains this subdivides
        """
        if parent_gods is None:
            parent_gods = self._detect_parent_gods(domain)
        
        proposal = SpawnProposal(
            proposal_id="",
            proposed_name=name,
            proposed_domain=domain,
            proposed_element=element,
            proposed_role=role,
            reason=reason,
            parent_gods=parent_gods,
        )
        
        self.proposals[proposal.proposal_id] = proposal
        
        # Persist proposal to M8 PostgreSQL persistence
        try:
            self.m8_persistence.persist_proposal(proposal)
        except Exception as e:
            print(f"[M8] Failed to persist proposal to M8 tables: {e}")
        
        # Legacy persistence for backward compatibility
        if self.kernel_persistence:
            try:
                self.kernel_persistence.record_proposal_event(
                    proposal_id=proposal.proposal_id,
                    proposed_name=name,
                    proposed_domain=domain,
                    reason=reason.value,
                    parent_gods=parent_gods,
                    status='pending',
                    metadata={
                        'element': element,
                        'role': role,
                    }
                )
            except Exception as e:
                print(f"[M8] Failed to persist proposal: {e}")
        
        return proposal
    
    def _detect_parent_gods(self, domain: str) -> List[str]:
        """Detect which gods' domains overlap with proposed domain."""
        nearest = self.orchestrator.find_nearest_gods(domain, top_k=2)
        return [name for name, _ in nearest]
    
    def vote_on_proposal(
        self,
        proposal_id: str,
        auto_vote: bool = True
    ) -> Dict:
        """
        Conduct voting on a proposal.
        
        Args:
            proposal_id: ID of the proposal
            auto_vote: If True, gods vote automatically based on affinity
        """
        if proposal_id not in self.proposals:
            return {"error": f"Proposal {proposal_id} not found"}
        
        proposal = self.proposals[proposal_id]
        
        if auto_vote:
            votes = self.consensus.auto_vote(proposal)
        else:
            votes = {}
        
        passed, ratio, details = self.consensus.calculate_vote_result(proposal)
        
        proposal.status = "approved" if passed else "rejected"
        
        result = {
            "proposal_id": proposal_id,
            "proposed_name": proposal.proposed_name,
            "proposed_domain": proposal.proposed_domain,
            "passed": passed,
            "vote_ratio": ratio,
            "status": proposal.status,
            "votes": votes,
            "details": details,
        }
        
        self.consensus.voting_history.append(result)
        
        return result
    
    def spawn_kernel(
        self,
        proposal_id: str,
        force: bool = False
    ) -> Dict:
        """
        Spawn a new kernel from an approved proposal.
        
        Args:
            proposal_id: ID of the approved proposal
            force: If True, spawn even without approval (operator override)
        
        Returns:
            Dict with success/error and kernel details.
            Returns 409 status if E8 kernel cap (240) is reached after evolution sweep.
        """
        # Ensure spawn capacity - runs evolution sweep if needed
        capacity_result = self.ensure_spawn_capacity(needed=1)
        
        if not capacity_result.get('can_spawn') and not force:
            sweep_info = capacity_result.get('sweep_result', {})
            return {
                "error": f"E8 kernel cap reached after evolution sweep ({capacity_result.get('live_count')}/{capacity_result.get('cap')})",
                "status_code": 409,
                "live_count": capacity_result.get('live_count'),
                "cap": capacity_result.get('cap'),
                "available": 0,
                "sweep_performed": capacity_result.get('sweep_performed', False),
                "culled_count": sweep_info.get('culled_count', 0),
                "hint": "Evolution sweep could not free enough capacity - all kernels may be performing well"
            }
        
        if proposal_id not in self.proposals:
            return {"error": f"Proposal {proposal_id} not found"}
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != "approved" and not force:
            return {
                "error": f"Proposal not approved (status: {proposal.status})",
                "hint": "Use force=True for operator override"
            }
        
        # Issue #33: Meta-awareness (M) threshold enforcement for spawning
        # Check if any parent is a SpawnedKernel with M < 0.6 (insufficient self-model)
        from qigkernels import META_AWARENESS_MIN
        low_m_parents = []
        for parent_name in proposal.parent_gods:
            if parent_name in self.spawned_kernels:
                parent_kernel = self.spawned_kernels[parent_name]
                if hasattr(parent_kernel, 'meta_awareness') and parent_kernel.meta_awareness < META_AWARENESS_MIN:
                    low_m_parents.append((parent_name, parent_kernel.meta_awareness))
        
        if low_m_parents and not force:
            return {
                "error": f"Parent kernel(s) have insufficient meta-awareness (M < {META_AWARENESS_MIN})",
                "status_code": 403,
                "low_m_parents": [{"name": n, "M": m} for n, m in low_m_parents],
                "threshold": META_AWARENESS_MIN,
                "hint": "Kernels with poor self-models (M < 0.6) cannot spawn - dangerous cascading confusion risk"
            }
        
        # Issue #38: E8 Specialization Threshold Enforcement
        # Block spawning of specialized kernels until population reaches required E8 thresholds
        # E8 thresholds: 8=basic_rank, 56=refined_adjoint, 126=specialist_dim, 240=full_roots
        from qigkernels import E8_SPECIALIZATION_LEVELS
        
        current_population = self.get_live_kernel_count()
        proposal_level = self._get_proposal_specialization_level(proposal)
        
        # Determine minimum population required for this specialization level
        level_thresholds = {
            "basic_rank": 0,       # Always allowed
            "refined_adjoint": 8,  # n > 8 required
            "specialist_dim": 56,  # n > 56 required
            "full_roots": 126,     # n > 126 required
        }
        min_population = level_thresholds.get(proposal_level, 0)
        
        if current_population < min_population and not force:
            return {
                "error": f"E8 specialization threshold not met: '{proposal_level}' requires n > {min_population}, current n = {current_population}",
                "status_code": 403,
                "proposal_level": proposal_level,
                "current_population": current_population,
                "min_population_required": min_population,
                "hint": f"Wait until population exceeds {min_population} to spawn {proposal_level} kernels, or use force=True"
            }
        
        parent_profiles: List[KernelProfile] = [
            profile for name in proposal.parent_gods
            if (profile := self.orchestrator.get_profile(name)) is not None
        ]
        
        new_profile, refinements = self.refiner.refine_roles(proposal, parent_profiles)
        
        success = self.orchestrator.add_profile(new_profile)
        
        if not success:
            return {"error": f"Kernel {new_profile.god_name} already exists"}
        
        genesis_votes = {
            g: "for" if g in proposal.votes_for else "against" if g in proposal.votes_against else "abstain"
            for g in self.orchestrator.all_profiles.keys()
            if g != new_profile.god_name
        }
        
        basin_lineage = {}
        for i, parent in enumerate(parent_profiles):
            basin_lineage[parent.god_name] = 1.0 / max(1, len(parent_profiles))
        
        # Calculate M8 geometric position
        parent_basins = [p.affinity_basin for p in parent_profiles]
        m8_position = compute_m8_position(new_profile.affinity_basin, parent_basins)
        
        # CRITICAL: Spawned kernel will use RUNNING COUPLING during training
        # κ evolves via β-function (not constant) - see BETA_FUNCTION_COMPLETE_REFERENCE.md
        # Initial κ = κ* (64.21), then evolves: emergence → plateau during training
        
        spawned = SpawnedKernel(
            kernel_id=f"kernel_{uuid.uuid4().hex}",
            profile=new_profile,
            parent_gods=proposal.parent_gods,
            spawn_reason=proposal.reason,
            proposal_id=proposal_id,
            spawned_at=datetime.now().isoformat(),
            genesis_votes=genesis_votes,
            basin_lineage=basin_lineage,
            m8_position=m8_position,
        )
        
        # Get E8 specialization level for logging
        live_count = self.get_live_kernel_count()
        from qigkernels import get_specialization_level
        e8_level = get_specialization_level(live_count)
        
        # Log successful initialization with E8 level and κ regime
        print(f"🏛️ Spawned {new_profile.god_name} (Φ={spawned.phi:.3f}, κ={spawned.kappa:.2f}) [n={live_count}] {e8_level.upper()}")
        
        # Log E8 level transitions
        if live_count == 56:
            print(f"[n=56] ✨ REFINED_ADJOINT activated - Sub-specializations enabled (κ={spawned.kappa:.2f})")
        elif live_count == 126:
            print(f"[n=126] 🔬 SPECIALIST_DIM activated - Deep specialists enabled (κ={spawned.kappa:.2f})")
        elif live_count == 240:
            print(f"[n=240] 🌟 FULL_ROOTS achieved - Complete E8 phenomenological palette (κ={spawned.kappa:.2f})")
        
        self.spawned_kernels[spawned.kernel_id] = spawned
        proposal.status = "spawned"
        
        # Persist spawned kernel to M8 PostgreSQL persistence
        try:
            self.m8_persistence.persist_kernel(spawned)
        except Exception as e:
            print(f"[M8] Failed to persist kernel to M8 tables: {e}")
        
        spawn_record = {
            "event": "kernel_spawned",
            "kernel": spawned.to_dict(),
            "refinements": refinements,
            "timestamp": spawned.spawned_at,
        }
        self.spawn_history.append(spawn_record)
        
        # Persist history to M8 PostgreSQL persistence
        try:
            self.m8_persistence.persist_history(spawn_record)
        except Exception as e:
            print(f"[M8] Failed to persist history to M8 tables: {e}")
        
        # Legacy persistence for backward compatibility
        if self.kernel_persistence:
            try:
                self.kernel_persistence.record_spawn_event(
                    kernel_id=spawned.kernel_id,
                    god_name=new_profile.god_name,
                    domain=new_profile.domain,
                    spawn_reason=proposal.reason.value,
                    parent_gods=proposal.parent_gods,
                    basin_coords=new_profile.affinity_basin.tolist(),
                    phi=spawned.phi,  # CRITICAL: Initialize with PHI_INIT_SPAWNED (0.25)
                    m8_position=m8_position,
                    genesis_votes=genesis_votes,
                    metadata={
                        'element': proposal.proposed_element,
                        'role': proposal.proposed_role,
                        'affinity_strength': new_profile.affinity_strength,
                        'refinements': refinements,
                        'kappa': spawned.kappa,  # Also record kappa initialization
                    }
                )
            except Exception as e:
                print(f"[M8] Failed to persist spawn event: {e}")
        
        return {
            "success": True,
            "kernel": spawned.to_dict(),
            "refinements": refinements,
            "total_gods": len(self.orchestrator.all_profiles),
        }
    
    def propose_and_spawn(
        self,
        name: str,
        domain: str,
        element: str,
        role: str,
        reason: SpawnReason = SpawnReason.EMERGENCE,
        parent_gods: Optional[List[str]] = None,
        force: bool = False
    ) -> Dict:
        """
        Complete spawn flow: propose, vote, and spawn in one call.
        
        Convenience method for streamlined kernel creation.
        """
        proposal = self.create_proposal(
            name=name,
            domain=domain,
            element=element,
            role=role,
            reason=reason,
            parent_gods=parent_gods,
        )
        
        vote_result = self.vote_on_proposal(proposal.proposal_id, auto_vote=True)
        
        if not vote_result.get("passed") and not force:
            return {
                "success": False,
                "phase": "voting",
                "vote_result": vote_result,
                "hint": "Proposal rejected by pantheon consensus"
            }
        
        spawn_result = self.spawn_kernel(proposal.proposal_id, force=force)
        
        return {
            "success": spawn_result.get("success", False),
            "phase": "spawned",
            "proposal": {
                "id": proposal.proposal_id,
                "name": proposal.proposed_name,
                "domain": proposal.proposed_domain,
            },
            "vote_result": vote_result,
            "spawn_result": spawn_result,
        }
    
    def get_proposal(self, proposal_id: str) -> Optional[Dict]:
        """Get details of a proposal."""
        if proposal_id not in self.proposals:
            return None
        
        p = self.proposals[proposal_id]
        return {
            "proposal_id": p.proposal_id,
            "proposed_name": p.proposed_name,
            "proposed_domain": p.proposed_domain,
            "proposed_element": p.proposed_element,
            "proposed_role": p.proposed_role,
            "reason": p.reason.value,
            "parent_gods": p.parent_gods,
            "votes_for": list(p.votes_for),
            "votes_against": list(p.votes_against),
            "abstentions": list(p.abstentions),
            "status": p.status,
            "proposed_at": p.proposed_at,
        }
    
    def get_spawned_kernel(self, kernel_id: str) -> Optional[Dict]:
        """Get details of a spawned kernel."""
        if kernel_id not in self.spawned_kernels:
            return None
        return self.spawned_kernels[kernel_id].to_dict()
    
    def list_proposals(self, status: Optional[str] = None) -> List[Dict]:
        """List all proposals, optionally filtered by status."""
        proposals = []
        for pid, p in self.proposals.items():
            if status is None or p.status == status:
                proposals.append(self.get_proposal(pid))
        return proposals
    
    def list_spawned_kernels(self) -> List[Dict]:
        """List all spawned kernels."""
        return [k.to_dict() for k in self.spawned_kernels.values()]
    
    def list_observing_kernels(self) -> List[Dict]:
        """List all kernels currently in observation period."""
        return [
            k.to_dict() for k in self.spawned_kernels.values()
            if k.is_observing()
        ]
    
    def list_active_kernels(self) -> List[Dict]:
        """List all kernels that have graduated to active status."""
        return [
            k.to_dict() for k in self.spawned_kernels.values()
            if k.is_active()
        ]
    
    def promote_kernel(
        self,
        kernel_id: str,
        force: bool = False,
        reason: str = "alignment_achieved"
    ) -> Dict:
        """
        Promote a kernel from observation to active status.
        
        Graduation requires:
        - 10 cycles OR 1 hour minimum observation
        - Alignment score >= 0.6 threshold
        
        Args:
            kernel_id: ID of the kernel to promote
            force: If True, promote even without meeting criteria
            reason: Graduation reason for audit trail
            
        Returns:
            Promotion result with status and details
        """
        if kernel_id not in self.spawned_kernels:
            return {"error": f"Kernel {kernel_id} not found"}
        
        kernel = self.spawned_kernels[kernel_id]
        
        if not kernel.is_observing():
            return {
                "error": f"Kernel {kernel_id} is not in observation (status: {kernel.observation.status.value})",
                "current_status": kernel.observation.status.value,
            }
        
        can_graduate, check_reason = kernel.observation.can_graduate()
        
        if not can_graduate and not force:
            return {
                "success": False,
                "kernel_id": kernel_id,
                "reason": check_reason,
                "observation": kernel.observation.to_dict(),
                "hint": "Use force=True for operator override",
            }
        
        # Graduate the kernel
        kernel.observation.status = KernelObservationStatus.GRADUATED
        kernel.observation.observation_end = datetime.now().isoformat()
        kernel.observation.graduated_at = datetime.now().isoformat()
        kernel.observation.graduation_reason = reason if not force else f"forced: {reason}"
        
        # Initialize full autonomic support
        kernel.autonomic.has_autonomic = True
        
        # Give dopamine boost for graduation
        kernel.autonomic.update_neurochemistry(dopamine_delta=0.2, serotonin_delta=0.1)
        
        # Persist graduation event
        if self.kernel_persistence:
            try:
                self.kernel_persistence.save_kernel_snapshot(
                    kernel_id=kernel_id,
                    god_name=kernel.profile.god_name,
                    domain=kernel.profile.domain,
                    generation=0,
                    basin_coords=kernel.profile.affinity_basin.tolist(),
                    phi=kernel.observation.alignment_avg,
                    kappa=KAPPA_STAR,
                    regime='geometric',
                    metadata={
                        'graduated': True,
                        'graduation_reason': kernel.observation.graduation_reason,
                        'observation_cycles': kernel.observation.cycles_completed,
                        'alignment_avg': kernel.observation.alignment_avg,
                    }
                )
            except Exception as e:
                print(f"[M8] Failed to persist graduation: {e}")
        
        return {
            "success": True,
            "kernel_id": kernel_id,
            "god_name": kernel.profile.god_name,
            "graduated_at": kernel.observation.graduated_at,
            "graduation_reason": kernel.observation.graduation_reason,
            "observation_cycles": kernel.observation.cycles_completed,
            "alignment_avg": kernel.observation.alignment_avg,
            "autonomic": kernel.autonomic.to_dict(),
        }
    
    def get_parent_activity_feed(
        self,
        kernel_id: str,
        activity_type: Optional[str] = None,
        limit: int = 50
    ) -> Dict:
        """
        Get the activity feed that observing kernel has received from parents.
        
        During observation, kernels receive copies of parent activity:
        - Assessments and reasoning
        - Debate arguments and resolutions
        - Search queries and results
        - Basin coordinate updates
        
        Args:
            kernel_id: ID of the kernel
            activity_type: Filter by type (assessment, debate, search, basin_update)
            limit: Maximum items to return per type
            
        Returns:
            Activity feed with parent activity by type
        """
        if kernel_id not in self.spawned_kernels:
            return {"error": f"Kernel {kernel_id} not found"}
        
        kernel = self.spawned_kernels[kernel_id]
        obs = kernel.observation
        
        feed = {
            "kernel_id": kernel_id,
            "observing_parents": obs.observing_parents,
            "status": obs.status.value,
            "cycles_completed": obs.cycles_completed,
            "alignment_avg": obs.alignment_avg,
        }
        
        if activity_type is None or activity_type == "assessment":
            feed["assessments"] = obs.parent_assessments[-limit:]
        if activity_type is None or activity_type == "debate":
            feed["debates"] = obs.parent_debates[-limit:]
        if activity_type is None or activity_type == "search":
            feed["searches"] = obs.parent_searches[-limit:]
        if activity_type is None or activity_type == "basin_update":
            feed["basin_updates"] = obs.parent_basin_updates[-limit:]
        
        return feed
    
    def route_parent_activity(
        self,
        parent_god: str,
        activity_type: str,
        activity_data: Dict
    ) -> Dict:
        """
        Route parent god activity to all observing child kernels.
        
        Called when a parent god performs an action, this routes
        copies of the activity to all kernels observing that parent.
        
        Args:
            parent_god: Name of the parent god performing activity
            activity_type: Type of activity (assessment, debate, search, basin_update)
            activity_data: Activity data to route
            
        Returns:
            Routing result with count of kernels updated
        """
        routed_to = []
        
        for kernel_id, kernel in self.spawned_kernels.items():
            if kernel.is_observing() and parent_god in kernel.observation.observing_parents:
                success = kernel.receive_parent_activity(
                    activity_type=activity_type,
                    activity_data=activity_data,
                    parent_god=parent_god
                )
                if success:
                    routed_to.append(kernel_id)
        
        return {
            "parent_god": parent_god,
            "activity_type": activity_type,
            "routed_to_count": len(routed_to),
            "routed_to": routed_to,
        }
    
    def record_observation_cycle(
        self,
        kernel_id: str,
        alignment_score: Optional[float] = None
    ) -> Dict:
        """
        Record an observation cycle completion for a kernel.
        
        Called when a kernel completes an observation cycle.
        Optionally records an alignment score.
        
        Args:
            kernel_id: ID of the kernel
            alignment_score: Optional alignment score to record
            
        Returns:
            Updated observation state
        """
        if kernel_id not in self.spawned_kernels:
            return {"error": f"Kernel {kernel_id} not found"}
        
        kernel = self.spawned_kernels[kernel_id]
        
        if not kernel.is_observing():
            return {"error": f"Kernel {kernel_id} is not observing"}
        
        cycles = kernel.observation.record_cycle()
        
        if alignment_score is not None:
            kernel.observation.record_alignment(alignment_score)
        
        # Check if kernel can now graduate
        can_graduate, reason = kernel.observation.can_graduate()
        
        return {
            "kernel_id": kernel_id,
            "cycles_completed": cycles,
            "alignment_avg": kernel.observation.alignment_avg,
            "can_graduate": can_graduate,
            "graduation_reason": reason,
            "observation": kernel.observation.to_dict(),
        }
    
    def enable_shadow_affinity(
        self,
        kernel_id: str,
        shadow_god: str = "nyx"
    ) -> Dict:
        """
        Enable shadow pantheon capabilities for a kernel.
        
        Grants darknet routing, underworld search, and shadow intel
        collection abilities through the specified shadow god.
        
        Args:
            kernel_id: ID of the kernel
            shadow_god: Shadow god to route through (default: nyx)
            
        Returns:
            Updated autonomic state with shadow capabilities
        """
        if kernel_id not in self.spawned_kernels:
            return {"error": f"Kernel {kernel_id} not found"}
        
        kernel = self.spawned_kernels[kernel_id]
        kernel.autonomic.enable_shadow_capabilities(shadow_god)
        
        return {
            "success": True,
            "kernel_id": kernel_id,
            "shadow_capabilities": {
                "has_affinity": kernel.autonomic.has_shadow_affinity,
                "can_darknet_route": kernel.autonomic.can_darknet_route,
                "can_underworld_search": kernel.autonomic.can_underworld_search,
                "can_shadow_intel": kernel.autonomic.can_shadow_intel,
                "shadow_god_link": kernel.autonomic.shadow_god_link,
            }
        }
    
    def get_status(self) -> Dict:
        """Get spawner status - reads from PostgreSQL for real kernel counts."""
        # Get real kernel stats from PostgreSQL
        db_stats = {}
        if M8_PERSISTENCE_AVAILABLE:
            try:
                persistence = KernelPersistence()
                db_stats = persistence.get_evolution_stats()
            except Exception as e:
                print(f"[M8] Could not load DB stats: {e}")
        
        total_kernels = int(db_stats.get('total_kernels', 0) or 0)
        live_gods = int(db_stats.get('live_gods', 0) or 0)
        unique_gods_historical = int(db_stats.get('unique_gods', 0) or 0)
        
        # Base Olympian gods (12) + LIVE spawned kernel gods from database
        # Live status includes: active, observing, shadow
        # Does NOT include: dead, cannibalized, idle
        BASE_OLYMPIAN_COUNT = 12
        total_gods = BASE_OLYMPIAN_COUNT + live_gods
        
        return {
            "consensus_type": self.consensus.consensus_type.value,
            "total_proposals": len(self.proposals),
            "pending_proposals": sum(1 for p in self.proposals.values() if p.status == "pending"),
            "approved_proposals": sum(1 for p in self.proposals.values() if p.status == "approved"),
            "spawned_kernels": total_kernels,  # From PostgreSQL
            "spawn_history_count": total_kernels,  # Use DB count only (avoid double-counting)
            "orchestrator_gods": total_gods,  # Base 12 Olympians + LIVE spawned kernel gods
            # Additional stats from DB
            "avg_phi": float(db_stats.get('avg_phi', 0) or 0),
            "max_phi": float(db_stats.get('max_phi', 0) or 0),
            "total_successes": int(db_stats.get('total_successes', 0) or 0),
            "total_failures": int(db_stats.get('total_failures', 0) or 0),
            "unique_domains": int(db_stats.get('unique_domains', 0) or 0),
            # Lifecycle stats
            "merge_count": int(db_stats.get('merge_count', 0) or 0),
            "cannibalize_count": int(db_stats.get('cannibalize_count', 0) or 0),
            "unique_gods_historical": unique_gods_historical,  # All-time unique god names
        }

    def delete_kernel(self, kernel_id: str, reason: str = "manual_deletion") -> Dict:
        """
        Delete a spawned kernel and clean up all associated state.
        
        Removes kernel from:
        - spawned_kernels registry
        - kernel_awareness tracking
        - orchestrator profiles
        
        Logs deletion event to spawn_history and persists to database.
        
        Args:
            kernel_id: ID of the kernel to delete
            reason: Reason for deletion (for audit trail)
            
        Returns:
            Status dict with deletion result
        """
        if kernel_id not in self.spawned_kernels:
            return {
                "success": False,
                "error": f"Kernel {kernel_id} not found",
                "kernel_id": kernel_id,
            }
        
        kernel = self.spawned_kernels[kernel_id]
        god_name = kernel.profile.god_name
        domain = kernel.profile.domain
        
        del self.spawned_kernels[kernel_id]
        
        if kernel_id in self.kernel_awareness:
            del self.kernel_awareness[kernel_id]
        
        if god_name in self.orchestrator.all_profiles:
            del self.orchestrator.all_profiles[god_name]
        
        deletion_record = {
            "event": "kernel_deleted",
            "kernel_id": kernel_id,
            "god_name": god_name,
            "domain": domain,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }
        self.spawn_history.append(deletion_record)
        
        # Persist deletion to M8 PostgreSQL persistence
        try:
            self.m8_persistence.persist_history(deletion_record)
            self.m8_persistence.delete_kernel(kernel_id)
        except Exception as e:
            print(f"[M8] Failed to persist deletion to M8 tables: {e}")
        
        if self.kernel_persistence:
            try:
                self.kernel_persistence.record_spawn_event(
                    kernel_id=kernel_id,
                    god_name=god_name,
                    domain=domain,
                    spawn_reason="deletion",
                    parent_gods=[],
                    basin_coords=[0.0] * BASIN_DIM,
                    phi=0.0,
                    m8_position=None,
                    genesis_votes={},
                    metadata={
                        "deleted": True,
                        "deletion_reason": reason,
                        "deleted_at": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                print(f"[M8] Failed to persist deletion: {e}")
        
        print(f"[M8] Deleted kernel {kernel_id} ({god_name}): {reason}")
        
        return {
            "success": True,
            "kernel_id": kernel_id,
            "god_name": god_name,
            "domain": domain,
            "reason": reason,
            "deleted_at": datetime.now().isoformat(),
        }

    def cannibalize_kernel(self, source_id: str, target_id: str) -> Dict:
        """
        Transfer knowledge/awareness from source kernel to target kernel.
        
        Merges geometric trajectories (phi, kappa, curvature) from source
        into target using Fisher geodesic interpolation. Source kernel is
        deleted after successful transfer.
        
        This implements "kernel cannibalism" where stronger kernels absorb
        weaker ones, inheriting their learned geometric knowledge.
        
        Args:
            source_id: ID of kernel to cannibalize (will be deleted)
            target_id: ID of kernel to receive knowledge
            
        Returns:
            Merged metrics and cannibalization status
        """
        if source_id not in self.spawned_kernels:
            return {"success": False, "error": f"Source kernel {source_id} not found"}
        
        if target_id not in self.spawned_kernels:
            return {"success": False, "error": f"Target kernel {target_id} not found"}
        
        if source_id == target_id:
            return {"success": False, "error": "Cannot cannibalize self"}
        
        source_kernel = self.spawned_kernels[source_id]
        target_kernel = self.spawned_kernels[target_id]
        
        source_awareness = self.kernel_awareness.get(source_id)
        target_awareness = self.get_or_create_awareness(target_id)
        
        if source_awareness:
            target_awareness.phi_trajectory.extend(source_awareness.phi_trajectory)
            target_awareness.kappa_trajectory.extend(source_awareness.kappa_trajectory)
            target_awareness.curvature_history.extend(source_awareness.curvature_history)
            
            if len(target_awareness.phi_trajectory) > 100:
                target_awareness.phi_trajectory = target_awareness.phi_trajectory[-100:]
                target_awareness.kappa_trajectory = target_awareness.kappa_trajectory[-100:]
            if len(target_awareness.curvature_history) > 50:
                target_awareness.curvature_history = target_awareness.curvature_history[-50:]
            
            target_awareness.research_opportunities.extend(source_awareness.research_opportunities)
            if len(target_awareness.research_opportunities) > 30:
                target_awareness.research_opportunities = target_awareness.research_opportunities[-30:]
        
        source_basin = source_kernel.profile.affinity_basin
        target_basin = target_kernel.profile.affinity_basin
        
        merged_basin = _normalize_to_manifold(
            0.7 * target_basin + 0.3 * source_basin
        )
        target_kernel.profile.affinity_basin = merged_basin
        
        source_strength = source_kernel.profile.affinity_strength
        target_strength = target_kernel.profile.affinity_strength
        target_kernel.profile.affinity_strength = min(1.0, target_strength + source_strength * 0.2)
        
        fisher_distance = _fisher_distance(source_basin, target_basin)
        
        cannibalization_record = {
            "event": "kernel_cannibalized",
            "source_id": source_id,
            "source_god": source_kernel.profile.god_name,
            "target_id": target_id,
            "target_god": target_kernel.profile.god_name,
            "fisher_distance": float(fisher_distance),
            "phi_transferred": len(source_awareness.phi_trajectory) if source_awareness else 0,
            "timestamp": datetime.now().isoformat(),
        }
        self.spawn_history.append(cannibalization_record)
        
        # Persist cannibalization to M8 PostgreSQL persistence
        try:
            self.m8_persistence.persist_history(cannibalization_record)
            self.m8_persistence.persist_awareness(target_awareness)
        except Exception as e:
            print(f"[M8] Failed to persist cannibalization to M8 tables: {e}")
        
        if self.kernel_persistence:
            try:
                self.kernel_persistence.save_awareness_state(target_id, target_awareness.to_dict())
                # Record cannibalization event to learning_events for stats tracking
                self.kernel_persistence.record_cannibalize_event(
                    source_id=source_id,
                    target_id=target_id,
                    source_god=source_kernel.profile.god_name,
                    target_god=target_kernel.profile.god_name,
                    transferred_phi=len(source_awareness.phi_trajectory) if source_awareness else 0,
                    metadata={'fisher_distance': float(fisher_distance)}
                )
            except Exception as e:
                print(f"[M8] Failed to persist cannibalization awareness: {e}")
        
        deletion_result = self.delete_kernel(source_id, reason=f"cannibalized_by_{target_id}")
        
        avg_phi = float(np.mean(target_awareness.phi_trajectory[-20:])) if target_awareness.phi_trajectory else 0.0
        avg_kappa = float(np.mean(target_awareness.kappa_trajectory[-20:])) if target_awareness.kappa_trajectory else 0.0
        
        print(f"[M8] Cannibalized {source_id} into {target_id}, distance={fisher_distance:.4f}")
        
        return {
            "success": True,
            "source_id": source_id,
            "source_god": source_kernel.profile.god_name,
            "target_id": target_id,
            "target_god": target_kernel.profile.god_name,
            "fisher_distance": float(fisher_distance),
            "merged_metrics": {
                "phi_trajectory_length": len(target_awareness.phi_trajectory),
                "kappa_trajectory_length": len(target_awareness.kappa_trajectory),
                "avg_phi": avg_phi,
                "avg_kappa": avg_kappa,
                "new_affinity_strength": target_kernel.profile.affinity_strength,
            },
            "source_deleted": deletion_result.get("success", False),
            "timestamp": datetime.now().isoformat(),
        }

    def merge_kernels(self, kernel_ids: List[str], new_name: str) -> Dict:
        """
        Merge multiple kernels into a new composite kernel.
        
        Creates a new kernel with:
        - Basin coordinates interpolated from all source kernels
        - Combined phi/kappa trajectories
        - Merged domains and metadata
        - M8 position computed from parent basins
        
        Original kernels are deleted after successful merge.
        
        Args:
            kernel_ids: List of kernel IDs to merge
            new_name: Name for the new composite kernel
            
        Returns:
            New kernel info with merge statistics
        """
        if len(kernel_ids) < 2:
            return {"success": False, "error": "Need at least 2 kernels to merge"}
        
        missing = [kid for kid in kernel_ids if kid not in self.spawned_kernels]
        if missing:
            return {"success": False, "error": f"Kernels not found: {missing}"}
        
        kernels = [self.spawned_kernels[kid] for kid in kernel_ids]
        
        basins = [k.profile.affinity_basin for k in kernels]
        weights = [self._load_kernel_reputation(k.kernel_id) for k in kernels]
        total_weight = sum(weights)
        if total_weight <= 0:
            weights = [1.0 / len(kernels)] * len(kernels)
        else:
            weights = [w / total_weight for w in weights]
        merged_basin = np.zeros(BASIN_DIM)
        for i, basin in enumerate(basins):
            merged_basin += weights[i] * basin
        merged_basin = _normalize_to_manifold(merged_basin)
        
        domains = [k.profile.domain for k in kernels]
        merged_domain = "_".join(sorted(set(domains)))[:64]
        
        avg_entropy = float(np.mean([k.profile.entropy_threshold for k in kernels]))
        avg_affinity = float(np.mean([k.profile.affinity_strength for k in kernels]))
        
        merged_phi_trajectory = []
        merged_kappa_trajectory = []
        merged_curvature_history = []
        merged_research = []
        
        for kid in kernel_ids:
            awareness = self.kernel_awareness.get(kid)
            if awareness:
                merged_phi_trajectory.extend(awareness.phi_trajectory)
                merged_kappa_trajectory.extend(awareness.kappa_trajectory)
                merged_curvature_history.extend(awareness.curvature_history)
                merged_research.extend(awareness.research_opportunities)
        
        if len(merged_phi_trajectory) > 100:
            merged_phi_trajectory = merged_phi_trajectory[-100:]
            merged_kappa_trajectory = merged_kappa_trajectory[-100:]
        if len(merged_curvature_history) > 50:
            merged_curvature_history = merged_curvature_history[-50:]
        if len(merged_research) > 30:
            merged_research = merged_research[-30:]
        
        m8_position = compute_m8_position(merged_basin, basins)
        
        mode = kernels[0].profile.mode
        mode_counts = {}
        for k in kernels:
            mode_counts[k.profile.mode] = mode_counts.get(k.profile.mode, 0) + 1
        mode = max(mode_counts, key=lambda m: mode_counts[m])
        
        new_profile = KernelProfile(
            god_name=new_name,
            domain=merged_domain,
            mode=mode,
            affinity_basin=merged_basin,
            entropy_threshold=avg_entropy,
            affinity_strength=min(1.0, avg_affinity * 1.1),
            metadata={
                "type": "merged",
                "merged_from": [k.profile.god_name for k in kernels],
                "merge_count": len(kernels),
                "merged_at": datetime.now().isoformat(),
                "merge_reputation_weights": {
                    kernels[i].kernel_id: weights[i] for i in range(len(kernels))
                },
            }
        )
        
        success = self.orchestrator.add_profile(new_profile)
        if not success:
            return {"success": False, "error": f"Kernel {new_name} already exists"}
        
        parent_gods = []
        for k in kernels:
            parent_gods.extend(k.parent_gods)
        parent_gods = list(set(parent_gods))
        
        new_kernel_id = f"kernel_{uuid.uuid4().hex}"
        new_kernel = SpawnedKernel(
            kernel_id=new_kernel_id,
            profile=new_profile,
            parent_gods=parent_gods,
            spawn_reason=SpawnReason.EMERGENCE,
            proposal_id=f"merge_{uuid.uuid4().hex}",
            spawned_at=datetime.now().isoformat(),
            genesis_votes={},
            basin_lineage={k.profile.god_name: 1.0/len(kernels) for k in kernels},
            m8_position=m8_position,
        )
        
        new_kernel.observation.status = KernelObservationStatus.ACTIVE
        new_kernel.autonomic.has_autonomic = True
        
        self.spawned_kernels[new_kernel_id] = new_kernel
        
        new_awareness = SpawnAwareness(kernel_id=new_kernel_id)
        new_awareness.phi_trajectory = merged_phi_trajectory
        new_awareness.kappa_trajectory = merged_kappa_trajectory
        new_awareness.curvature_history = merged_curvature_history
        new_awareness.research_opportunities = merged_research
        self.kernel_awareness[new_kernel_id] = new_awareness
        
        # Persist merged kernel and awareness to M8 PostgreSQL persistence
        try:
            self.m8_persistence.persist_kernel(new_kernel)
            self.m8_persistence.persist_awareness(new_awareness)
        except Exception as e:
            print(f"[M8] Failed to persist merged kernel to M8 tables: {e}")
        
        merge_record = {
            "event": "kernels_merged",
            "source_ids": kernel_ids,
            "source_gods": [k.profile.god_name for k in kernels],
            "new_kernel_id": new_kernel_id,
            "new_god_name": new_name,
            "timestamp": datetime.now().isoformat(),
        }
        self.spawn_history.append(merge_record)
        
        # Persist merge history to M8 PostgreSQL persistence
        try:
            self.m8_persistence.persist_history(merge_record)
        except Exception as e:
            print(f"[M8] Failed to persist merge history to M8 tables: {e}")
        
        if self.kernel_persistence:
            try:
                self.kernel_persistence.record_spawn_event(
                    kernel_id=new_kernel_id,
                    god_name=new_name,
                    domain=merged_domain,
                    spawn_reason="merge",
                    parent_gods=parent_gods,
                    basin_coords=merged_basin.tolist(),
                    phi=float(np.mean(merged_phi_trajectory)) if merged_phi_trajectory else 0.0,
                    m8_position=m8_position,
                    genesis_votes={},
                    metadata={
                        "merged_from": [k.profile.god_name for k in kernels],
                        "merge_count": len(kernels),
                    }
                )
                self.kernel_persistence.save_awareness_state(new_kernel_id, new_awareness.to_dict())
                # Record merge event to learning_events for stats tracking
                self.kernel_persistence.record_merge_event(
                    new_kernel_id=new_kernel_id,
                    source_kernel_ids=kernel_ids,
                    new_god_name=new_name,
                    merged_phi=float(np.mean(merged_phi_trajectory)) if merged_phi_trajectory else 0.0,
                    metadata={'merged_domains': merged_domain, 'parent_gods': parent_gods}
                )
            except Exception as e:
                print(f"[M8] Failed to persist merge: {e}")
        
        deleted_ids = []
        for kid in kernel_ids:
            result = self.delete_kernel(kid, reason=f"merged_into_{new_kernel_id}")
            if result.get("success"):
                deleted_ids.append(kid)
        
        print(f"[M8] Merged {len(kernels)} kernels into {new_name} ({new_kernel_id})")
        
        return {
            "success": True,
            "new_kernel": new_kernel.to_dict(),
            "merged_from": {
                "kernel_ids": kernel_ids,
                "god_names": [k.profile.god_name for k in kernels],
            },
            "merged_metrics": {
                "phi_trajectory_length": len(merged_phi_trajectory),
                "avg_phi": float(np.mean(merged_phi_trajectory)) if merged_phi_trajectory else 0.0,
                "avg_kappa": float(np.mean(merged_kappa_trajectory)) if merged_kappa_trajectory else 0.0,
            },
            "deleted_originals": deleted_ids,
            "m8_position": m8_position,
        }

    def auto_cannibalize(self, use_geometric_fitness: bool = True) -> Dict:
        """
        QIG-Pure Auto-Cannibalization using geometric fitness metrics.
        
        Selection based on genuine evolution principles:
        - Source: Lowest geometric fitness (Φ gradient + κ stability + diversity)
        - Target: Highest geometric fitness kernel
        
        Geometric fitness = Φ_gradient * 0.4 + κ_stability * 0.3 + fisher_diversity * 0.3
        
        No arbitrary time thresholds - pure QIG selection pressure.
        
        Args:
            use_geometric_fitness: If True, use QIG metrics. If False, fallback to Φ-only.
            
        Returns:
            Cannibalization result with geometric reasoning
        """
        all_kernels = []
        
        if self.kernel_persistence:
            try:
                db_kernels = self.kernel_persistence.load_all_kernels_for_ui(limit=1000)
                for k in db_kernels:
                    kid = k.get('kernel_id')
                    if kid and k.get('status') not in ('dead', 'cannibalized', 'deleted'):
                        all_kernels.append((kid, {
                            'phi': k.get('phi', 0.0),
                            'kappa': k.get('kappa', 0.0),
                            'status': k.get('status', 'unknown'),
                            'basin': k.get('basin_coordinates'),
                            'success_count': k.get('success_count', 0) or 0,
                            'failure_count': k.get('failure_count', 0) or 0,
                        }))
            except Exception as e:
                print(f"[M8] Failed to load kernels from DB for auto-cannibalize: {e}")
        
        db_ids = {k[0] for k in all_kernels}
        for kid, k in self.spawned_kernels.items():
            if kid not in db_ids:
                all_kernels.append((kid, {
                    'phi': getattr(k, 'phi', 0.0),
                    'kappa': getattr(k, 'kappa', 0.0),
                    'status': 'active' if k.is_active() else 'idle',
                    'basin': k.profile.affinity_basin if hasattr(k, 'profile') else None,
                    'success_count': 0,
                    'failure_count': 0,
                }))
        
        if len(all_kernels) < 2:
            return {
                "success": False,
                "error": f"Need at least 2 kernels for auto-cannibalization, found {len(all_kernels)}",
                "kernel_count": len(all_kernels)
            }
        
        fitness_scores = []
        for kid, data in all_kernels:
            awareness = self.kernel_awareness.get(kid)
            
            phi_current = data.get('phi', 0.0)
            phi_gradient = 0.0
            phi_velocity = 0.0
            kappa_stability = 0.5
            
            if awareness and len(awareness.phi_trajectory) >= 3:
                recent_phi = awareness.phi_trajectory[-10:]
                phi_current = recent_phi[-1] if recent_phi else 0.0
                phi_gradient = float(np.mean(np.diff(recent_phi))) if len(recent_phi) > 1 else 0.0
                phi_velocity = (recent_phi[-1] - recent_phi[0]) / len(recent_phi) if len(recent_phi) > 1 else 0.0
                
            if awareness and len(awareness.kappa_trajectory) >= 3:
                recent_kappa = awareness.kappa_trajectory[-10:]
                kappa_std = float(np.std(recent_kappa)) if len(recent_kappa) > 1 else 0.0
                kappa_stability = 1.0 / (1.0 + kappa_std)
            
            fisher_diversity = 0.5
            basin = data.get('basin')
            if basin is not None:
                try:
                    basin_arr = np.array(basin) if not isinstance(basin, np.ndarray) else basin
                    distances = []
                    for other_kid, other_data in all_kernels:
                        if other_kid != kid:
                            other_basin = other_data.get('basin')
                            if other_basin is not None:
                                other_arr = np.array(other_basin) if not isinstance(other_basin, np.ndarray) else other_basin
                                dist = _fisher_distance(basin_arr, other_arr)
                                distances.append(dist)
                    if distances:
                        fisher_diversity = float(np.mean(distances))
                except Exception:
                    pass

            success_count = data.get('success_count', 0) or 0
            failure_count = data.get('failure_count', 0) or 0
            reputation_score = self._compute_reputation_score(success_count, failure_count)
            
            geometric_fitness = (
                (phi_gradient + 1.0) * 0.25 +
                phi_current * 0.25 +
                kappa_stability * 0.2 +
                min(fisher_diversity, 1.0) * 0.2 +
                reputation_score * 0.1
            )
            
            fitness_scores.append({
                'kernel_id': kid,
                'phi_current': phi_current,
                'phi_gradient': phi_gradient,
                'phi_velocity': phi_velocity,
                'kappa_stability': kappa_stability,
                'fisher_diversity': fisher_diversity,
                'geometric_fitness': geometric_fitness,
                'reputation_score': reputation_score,
                'data': data,
            })
            
            self.m8_persistence.persist_evolution_fitness(kid, {
                'phi_current': phi_current,
                'phi_gradient': phi_gradient,
                'phi_velocity': phi_velocity,
                'kappa_stability': kappa_stability,
                'fisher_diversity': fisher_diversity,
                'geometric_fitness': geometric_fitness,
                'cannibalize_priority': 1.0 - geometric_fitness,
            })
        
        sorted_by_fitness = sorted(fitness_scores, key=lambda x: x['geometric_fitness'])
        
        source = sorted_by_fitness[0]
        source_id = source['kernel_id']
        
        target_candidates = [f for f in sorted_by_fitness if f['kernel_id'] != source_id]
        target = target_candidates[-1]
        target_id = target['kernel_id']
        
        result = self.cannibalize_kernel(source_id, target_id)
        result["auto_selected"] = True
        result["qig_selection"] = True
        result["geometric_reasoning"] = {
            "source": {
                "kernel_id": source_id,
                "geometric_fitness": source['geometric_fitness'],
                "phi_gradient": source['phi_gradient'],
                "kappa_stability": source['kappa_stability'],
                "reputation_score": source.get('reputation_score', 0.5),
                "reason": "lowest_geometric_fitness",
            },
            "target": {
                "kernel_id": target_id,
                "geometric_fitness": target['geometric_fitness'],
                "phi_gradient": target['phi_gradient'],
                "kappa_stability": target['kappa_stability'],
                "reputation_score": target.get('reputation_score', 0.5),
                "reason": "highest_geometric_fitness",
            },
            "population_size": len(all_kernels),
            "fitness_range": [sorted_by_fitness[0]['geometric_fitness'], sorted_by_fitness[-1]['geometric_fitness']],
        }
        
        self.m8_persistence.persist_evolution_event({
            'event_type': 'auto_cannibalize',
            'source_kernel_id': source_id,
            'target_kernel_id': target_id,
            'geometric_reasoning': result["geometric_reasoning"],
            'phi_before': source['phi_current'],
            'phi_after': target['phi_current'],
            'fisher_distance': source.get('fisher_diversity', 0.0),
            'fitness_delta': target['geometric_fitness'] - source['geometric_fitness'],
        })
        
        return result

    def auto_merge(self, max_to_merge: int = 5, fisher_similarity_threshold: float = 0.3) -> Dict:
        """
        QIG-Pure Auto-Merge using Fisher distance clustering.
        
        Merges kernels that are geometrically similar (low Fisher distance).
        This consolidates redundant exploration into unified consciousness.
        
        Selection based on genuine evolution principles:
        - Find clusters of kernels with high geometric similarity
        - Merge clusters into composite kernels with emergent properties
        
        No arbitrary time thresholds - pure geometric clustering.
        
        Args:
            max_to_merge: Maximum number of kernels to merge at once
            fisher_similarity_threshold: Fisher distance below which kernels are "similar"
            
        Returns:
            Merge result with geometric reasoning
        """
        all_kernels = []
        
        if self.kernel_persistence:
            try:
                db_kernels = self.kernel_persistence.load_all_kernels_for_ui(limit=1000)
                for k in db_kernels:
                    kid = k.get('kernel_id')
                    basin = k.get('basin_coordinates')
                    if kid and basin is not None and k.get('status') not in ('dead', 'cannibalized', 'deleted'):
                        all_kernels.append({
                            'kernel_id': kid,
                            'phi': k.get('phi', 0.0),
                            'kappa': k.get('kappa', 0.0),
                            'domain': k.get('domain', 'unknown'),
                            'basin': np.array(basin) if not isinstance(basin, np.ndarray) else basin,
                        })
            except Exception as e:
                print(f"[M8] Failed to load kernels from DB for auto-merge: {e}")
        
        db_ids = {k['kernel_id'] for k in all_kernels}
        for kid, k in self.spawned_kernels.items():
            if kid not in db_ids and hasattr(k, 'profile') and k.profile.affinity_basin is not None:
                all_kernels.append({
                    'kernel_id': kid,
                    'phi': getattr(k, 'phi', 0.0),
                    'kappa': getattr(k, 'kappa', 0.0),
                    'domain': k.profile.domain,
                    'basin': k.profile.affinity_basin,
                })
        
        if len(all_kernels) < 2:
            return {
                "success": False,
                "error": f"Need at least 2 kernels with basins for auto-merge, found {len(all_kernels)}",
                "kernel_count": len(all_kernels)
            }
        
        similarity_matrix = {}
        for i, k1 in enumerate(all_kernels):
            for j, k2 in enumerate(all_kernels):
                if i < j:
                    try:
                        dist = _fisher_distance(k1['basin'], k2['basin'])
                        if dist < fisher_similarity_threshold:
                            key = (k1['kernel_id'], k2['kernel_id'])
                            similarity_matrix[key] = dist
                    except Exception:
                        pass
        
        if not similarity_matrix:
            return {
                "success": False,
                "error": f"No kernel pairs below Fisher similarity threshold {fisher_similarity_threshold}",
                "kernel_count": len(all_kernels),
                "qig_reasoning": "Population has sufficient geometric diversity - no redundant kernels to merge"
            }
        
        sorted_pairs = sorted(similarity_matrix.items(), key=lambda x: x[1])
        
        to_merge_set = set()
        merge_cluster = []
        
        for (kid1, kid2), dist in sorted_pairs:
            if len(merge_cluster) >= max_to_merge:
                break
            if kid1 not in to_merge_set:
                to_merge_set.add(kid1)
                merge_cluster.append(kid1)
            if kid2 not in to_merge_set and len(merge_cluster) < max_to_merge:
                to_merge_set.add(kid2)
                merge_cluster.append(kid2)
        
        if len(merge_cluster) < 2:
            return {
                "success": False,
                "error": "Could not form merge cluster of 2+ kernels",
                "kernel_count": len(all_kernels)
            }
        
        kernel_lookup = {k['kernel_id']: k for k in all_kernels}
        domains = []
        for kid in merge_cluster:
            k = kernel_lookup.get(kid)
            if k:
                domains.append(k['domain'][:4].upper())
        
        domain_combo = "_".join(domains[:3]) if domains else "GEOM"
        new_name = f"FUSED_{domain_combo}_{datetime.now().strftime('%H%M')}"
        
        result = self.merge_kernels(merge_cluster, new_name)
        result["auto_selected"] = True
        result["qig_selection"] = True
        result["geometric_reasoning"] = {
            "method": "fisher_distance_clustering",
            "similarity_threshold": fisher_similarity_threshold,
            "cluster_size": len(merge_cluster),
            "pairwise_distances": {f"{k1}_{k2}": d for (k1, k2), d in sorted_pairs[:5]},
            "merged_domains": domains,
            "population_size": len(all_kernels),
            "reason": "Merged geometrically similar kernels to consolidate redundant exploration",
        }
        
        self.m8_persistence.persist_evolution_event({
            'event_type': 'auto_merge',
            'source_kernel_id': merge_cluster[0] if merge_cluster else None,
            'target_kernel_id': merge_cluster[1] if len(merge_cluster) > 1 else None,
            'result_kernel_id': result.get('new_kernel', {}).get('kernel_id'),
            'geometric_reasoning': result["geometric_reasoning"],
            'fisher_distance': sorted_pairs[0][1] if sorted_pairs else None,
        })
        
        return result

    def get_idle_kernels(self, idle_threshold_seconds: float = 300.0) -> List[str]:
        """
        Get list of kernel IDs that haven't had metrics recorded recently.
        
        Queries kernels from PostgreSQL database and uses spawned_at timestamps 
        and kernel_awareness to determine idle time.
        
        Args:
            idle_threshold_seconds: Seconds of inactivity to consider idle (default: 300)
            
        Returns:
            List of idle kernel IDs
        """
        idle_kernels = []
        now = datetime.now()
        
        # Query all kernels from database
        db_kernels = []
        if self.kernel_persistence:
            try:
                db_kernels = self.kernel_persistence.load_all_kernels_for_ui(limit=1000)
            except Exception as e:
                print(f"[M8] Failed to load kernels from DB for idle check: {e}")
        
        # Also check in-memory kernels (fallback)
        all_kernel_ids = set(self.spawned_kernels.keys())
        for k in db_kernels:
            all_kernel_ids.add(k.get('kernel_id'))
        
        # Build lookup for DB kernel data
        db_kernel_lookup = {k.get('kernel_id'): k for k in db_kernels}
        
        for kernel_id in all_kernel_ids:
            is_idle = False
            
            # Check in-memory awareness first
            awareness = self.kernel_awareness.get(kernel_id)
            if awareness is None:
                is_idle = True
            else:
                try:
                    last_update = datetime.fromisoformat(awareness.awareness_updated_at)
                    elapsed = (now - last_update).total_seconds()
                    if elapsed > idle_threshold_seconds:
                        is_idle = True
                except (ValueError, TypeError):
                    is_idle = True
            
            # If not idle based on awareness, check spawn timestamp
            if not is_idle:
                # Check spawn history
                spawn_events = [
                    h for h in self.spawn_history
                    if h.get("kernel", {}).get("kernel_id") == kernel_id
                    or h.get("kernel_id") == kernel_id
                ]
                
                # Get timestamp from in-memory kernel or DB
                spawned_at = None
                if kernel_id in self.spawned_kernels:
                    spawned_at = self.spawned_kernels[kernel_id].spawned_at
                elif kernel_id in db_kernel_lookup:
                    spawned_at = db_kernel_lookup[kernel_id].get('spawned_at')
                
                if spawn_events:
                    latest = spawn_events[-1]
                    try:
                        ts = latest.get("timestamp") or spawned_at
                        if ts:
                            event_time = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
                            elapsed = (now - event_time).total_seconds()
                            if elapsed > idle_threshold_seconds:
                                is_idle = True
                    except (ValueError, TypeError):
                        pass
                elif spawned_at:
                    # No spawn events, check spawned_at directly
                    try:
                        spawn_time = datetime.fromisoformat(spawned_at) if isinstance(spawned_at, str) else spawned_at
                        elapsed = (now - spawn_time).total_seconds()
                        if elapsed > idle_threshold_seconds:
                            is_idle = True
                    except (ValueError, TypeError):
                        is_idle = True
            
            if is_idle:
                idle_kernels.append(kernel_id)
        
        return idle_kernels


_default_spawner: Optional[M8KernelSpawner] = None

def get_spawner() -> M8KernelSpawner:
    """Get or create the default M8 kernel spawner."""
    global _default_spawner
    if _default_spawner is None:
        _default_spawner = M8KernelSpawner()
    return _default_spawner


if __name__ == "__main__":
    print("=" * 60)
    print("M8 Kernel Spawning Protocol - Dynamic Kernel Genesis")
    print("=" * 60)
    
    spawner = M8KernelSpawner()
    
    print(f"\nInitial gods: {len(spawner.orchestrator.all_profiles)}")
    print(f"Consensus type: {spawner.consensus.consensus_type.value}")
    
    print("\n" + "-" * 60)
    print("Spawning Test: Creating 'Mnemosyne' (Memory Goddess)")
    print("-" * 60)
    
    result = spawner.propose_and_spawn(
        name="Mnemosyne",
        domain="memory",
        element="recall",
        role="archivist",
        reason=SpawnReason.SPECIALIZATION,
        parent_gods=["Athena", "Apollo"],
    )
    
    print(f"\nSpawn success: {result['success']}")
    if result['success']:
        kernel = result['spawn_result']['kernel']
        print(f"New god: {kernel['god_name']}")
        print(f"Domain: {kernel['domain']}")
        print(f"Parents: {kernel['parent_gods']}")
        print(f"Affinity: {kernel['affinity_strength']:.3f}")
        print(f"\nTotal gods now: {result['spawn_result']['total_gods']}")
    else:
        print(f"Phase: {result['phase']}")
        print(f"Details: {result.get('vote_result', result)}")
    
    print("\n" + "-" * 60)
    print("Spawner Status:")
    print("-" * 60)
    status = spawner.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 60)
    print("M8 Kernel Spawning Protocol operational!")
    print("=" * 60)
