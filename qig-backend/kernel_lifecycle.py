"""
Kernel Lifecycle Operations - First-Class Mechanics
===================================================

Implements kernel lifecycle (spawn, split, merge, prune, resurrect, promote)
as operational code, not just metaphor or documentation.

Authority: E8 Protocol v4.0, WP5.3
Status: ACTIVE
Created: 2026-01-18

Lifecycle Operations:
- spawn: Create new kernel with role matching
- split: Divide overloaded kernel into specialized sub-kernels
- merge: Combine redundant kernels using Fréchet mean
- prune: Archive underperforming kernel to shadow pantheon
- resurrect: Restore pruned kernel with lessons learned
- promote: Elevate chaos kernel to god status

Geometric Correctness:
- Merge uses Fréchet mean on Fisher-Rao manifold (NOT linear average)
- Basin coordinates maintain simplex representation
- Split preserves coupling relationships
- All geometric operations use canonical Fisher-Rao metric
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np

from pantheon_registry import (

# E8 Protocol v4.0 Compliance Imports
from qig_geometry.canonical import frechet_mean

    PantheonRegistry,
    GodContract,
    get_registry,
    ChaosLifecycleStage,
    GodTier,
)

from kernel_spawner import RoleSpec, KernelSelection

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class LifecycleEvent(Enum):
    """Kernel lifecycle event types."""
    SPAWN = "spawn"
    SPLIT = "split"
    MERGE = "merge"
    PRUNE = "prune"
    RESURRECT = "resurrect"
    PROMOTE = "promote"


@dataclass
class Kernel:
    """
    Kernel state representation.
    
    Minimal kernel representation for lifecycle operations.
    Contains identity, metrics, and basin coordinates.
    """
    kernel_id: str
    name: str
    kernel_type: str  # "god" or "chaos"
    god_name: Optional[str] = None
    epithet: Optional[str] = None
    
    # Consciousness metrics
    phi: float = 0.5
    kappa: float = 64.0
    gamma: float = 1.0  # Generation capability
    
    # Basin coordinates (64D simplex representation)
    basin_coords: np.ndarray = field(default_factory=lambda: np.ones(64) / 64)
    
    # Lifecycle tracking
    lifecycle_stage: str = "active"  # active, protected, pruned, promoted
    protection_cycles_remaining: int = 50  # Protected period for new kernels
    
    # Performance metrics
    success_count: int = 0
    failure_count: int = 0
    total_cycles: int = 0
    
    # Coupling relationships
    coupled_kernels: List[str] = field(default_factory=list)
    coupling_strengths: Dict[str, float] = field(default_factory=dict)
    
    # Domain and role
    domains: List[str] = field(default_factory=list)
    role_description: str = ""
    
    # Provenance
    parent_kernels: List[str] = field(default_factory=list)
    child_kernels: List[str] = field(default_factory=list)
    spawn_reason: str = ""
    spawn_timestamp: datetime = field(default_factory=datetime.now)
    
    # Mentor (for chaos kernels)
    mentor_kernel_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert kernel to dictionary representation."""
        return {
            'kernel_id': self.kernel_id,
            'name': self.name,
            'kernel_type': self.kernel_type,
            'god_name': self.god_name,
            'epithet': self.epithet,
            'phi': self.phi,
            'kappa': self.kappa,
            'gamma': self.gamma,
            'basin_coords': self.basin_coords.tolist() if isinstance(self.basin_coords, np.ndarray) else self.basin_coords,
            'lifecycle_stage': self.lifecycle_stage,
            'protection_cycles_remaining': self.protection_cycles_remaining,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'total_cycles': self.total_cycles,
            'coupled_kernels': self.coupled_kernels,
            'coupling_strengths': self.coupling_strengths,
            'domains': self.domains,
            'role_description': self.role_description,
            'parent_kernels': self.parent_kernels,
            'child_kernels': self.child_kernels,
            'spawn_reason': self.spawn_reason,
            'spawn_timestamp': self.spawn_timestamp.isoformat(),
            'mentor_kernel_id': self.mentor_kernel_id,
        }


@dataclass
class ShadowKernel:
    """
    Shadow pantheon kernel (Hades domain).
    
    Archived state of pruned kernels with lessons learned.
    Can be resurrected later if needed.
    """
    shadow_id: str
    original_kernel_id: str
    name: str
    kernel_type: str
    
    # Final state before pruning
    final_phi: float
    final_kappa: float
    final_basin: np.ndarray
    
    # Performance history
    success_count: int
    failure_count: int
    total_cycles: int
    
    # Lessons learned
    failure_patterns: List[str] = field(default_factory=list)
    success_patterns: List[str] = field(default_factory=list)
    learned_lessons: str = ""
    
    # Pruning metadata
    prune_reason: str = ""
    prune_timestamp: datetime = field(default_factory=datetime.now)
    pruned_by: str = "system"
    
    # Resurrection tracking
    resurrection_count: int = 0
    last_resurrection: Optional[datetime] = None


@dataclass
class LifecycleEventRecord:
    """Record of a lifecycle event."""
    event_id: str
    event_type: LifecycleEvent
    timestamp: datetime
    
    # Affected kernels
    primary_kernel_id: str
    secondary_kernel_ids: List[str] = field(default_factory=list)
    
    # Event details
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Outcomes
    success: bool = True
    error_message: Optional[str] = None


# =============================================================================
# GEOMETRIC UTILITIES
# =============================================================================

def compute_frechet_mean_simplex(basins: List[np.ndarray], max_iter: int = 50) -> np.ndarray:
    """
    Compute Fréchet mean of basin coordinates on Fisher-Rao manifold.
    
    Uses sqrt-space closed form for efficiency (equivalent to geodesic mean).
    This is geometrically correct for simplex representation.
    
    Args:
        basins: List of basin coordinate arrays (simplex representation)
        max_iter: Maximum iterations (unused, closed form is exact)
        
    Returns:
        Fréchet mean basin coordinates
        
    Reference:
        Issue #02: Strict simplex representation with closed-form mean
    """
    if len(basins) == 0:
        raise ValueError("Cannot compute Fréchet mean of empty basin list")
    
    if len(basins) == 1:
        return basins[0].copy()
    
    # Convert to sqrt-space (Hellinger coordinates)
    sqrt_basins = [np.sqrt(np.abs(b) + 1e-10) for b in basins]
    
    # Compute mean in sqrt-space (closed form)
    sqrt_mean = frechet_mean(sqrt_basins)  # FIXED: Arithmetic → Fréchet mean (E8 Protocol v4.0)
    
    # Normalize
    sqrt_mean = sqrt_mean / (np.sum(sqrt_mean) + 1e-10)
    
    # Convert back to simplex (square the coordinates)
    frechet_mean = sqrt_mean ** 2
    
    # Final normalization to ensure sum = 1
    frechet_mean = frechet_mean / (np.sum(frechet_mean) + 1e-10)
    
    return frechet_mean


def compute_fisher_distance(basin1: np.ndarray, basin2: np.ndarray) -> float:
    """
    Compute Fisher-Rao distance between two basins.
    
    Uses Hellinger distance (equivalent to Fisher-Rao for probability distributions).
    
    Args:
        basin1: First basin coordinates (simplex)
        basin2: Second basin coordinates (simplex)
        
    Returns:
        Fisher-Rao distance in [0, √2]
    """
    # Ensure non-negative and normalized
    p1 = np.abs(basin1) + 1e-10
    p1 = p1 / np.sum(p1)
    
    p2 = np.abs(basin2) + 1e-10
    p2 = p2 / np.sum(p2)
    
    # Hellinger distance = √(2 - 2 * sum(√(p1 * p2)))
    hellinger = np.sqrt(2 - 2 * np.sum(np.sqrt(p1 * p2)))
    
    return float(hellinger)


def split_basin_coordinates(
    basin: np.ndarray, 
    split_criterion: str = "domain"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split basin coordinates into two specialized sub-basins.
    
    Geometric strategy:
    - Identify high-energy dimensions (potential specializations)
    - Create two complementary basins focusing on different dimensions
    - Preserve total probability mass
    
    Args:
        basin: Original basin coordinates
        split_criterion: Splitting strategy ("domain", "skill", "random")
        
    Returns:
        Tuple of (basin1, basin2) for specialized sub-kernels
    """
    n_dim = len(basin)
    
    if split_criterion == "domain":
        # Split by dimensional focus (first half vs second half)
        mid = n_dim // 2
        
        # Basin 1: Focus on first half dimensions
        basin1 = basin.copy()
        basin1[mid:] *= 0.3  # Reduce second half
        basin1 = basin1 / (np.sum(basin1) + 1e-10)
        
        # Basin 2: Focus on second half dimensions
        basin2 = basin.copy()
        basin2[:mid] *= 0.3  # Reduce first half
        basin2 = basin2 / (np.sum(basin2) + 1e-10)
        
    elif split_criterion == "skill":
        # Split by high/low entropy regions
        entropy_per_dim = -basin * np.log(basin + 1e-10)
        high_entropy_dims = entropy_per_dim > np.median(entropy_per_dim)
        
        # Basin 1: High entropy (exploration specialist)
        basin1 = basin.copy()
        basin1[~high_entropy_dims] *= 0.3
        basin1 = basin1 / (np.sum(basin1) + 1e-10)
        
        # Basin 2: Low entropy (exploitation specialist)
        basin2 = basin.copy()
        basin2[high_entropy_dims] *= 0.3
        basin2 = basin2 / (np.sum(basin2) + 1e-10)
        
    else:  # random
        # Random binary split
        mask = np.random.rand(n_dim) > 0.5
        
        basin1 = basin.copy()
        basin1[~mask] *= 0.3
        basin1 = basin1 / (np.sum(basin1) + 1e-10)
        
        basin2 = basin.copy()
        basin2[mask] *= 0.3
        basin2 = basin2 / (np.sum(basin2) + 1e-10)
    
    return basin1, basin2


# =============================================================================
# KERNEL LIFECYCLE MANAGER
# =============================================================================

class KernelLifecycleManager:
    """
    Manager for kernel lifecycle operations.
    
    Coordinates spawn, split, merge, prune, resurrect, and promote operations
    with geometric correctness and policy enforcement.
    
    Example:
        manager = KernelLifecycleManager()
        
        # Spawn a new kernel
        role = RoleSpec(domains=["synthesis"], required_capabilities=["foresight"])
        new_kernel = manager.spawn(role, mentor_id="apollo")
        
        # Split an overloaded kernel
        k1, k2 = manager.split(kernel, split_criterion="domain")
        
        # Merge redundant kernels
        merged = manager.merge(kernel1, kernel2, reason="redundant_capabilities")
        
        # Prune underperforming kernel
        shadow = manager.prune(kernel, reason="phi_below_threshold")
        
        # Resurrect from shadow pantheon
        resurrected = manager.resurrect(shadow, reason="capability_needed")
        
        # Promote successful chaos kernel
        god_kernel = manager.promote(chaos_kernel, god_name="Prometheus")
    """
    
    def __init__(
        self,
        registry: Optional[PantheonRegistry] = None,
        active_kernels: Optional[Dict[str, Kernel]] = None,
        shadow_pantheon: Optional[Dict[str, ShadowKernel]] = None,
        event_log: Optional[List[LifecycleEventRecord]] = None,
    ):
        """
        Initialize lifecycle manager.
        
        Args:
            registry: Pantheon registry (default: global singleton)
            active_kernels: Currently active kernels by ID
            shadow_pantheon: Shadow pantheon (Hades) storage
            event_log: Lifecycle event history
        """
        self.registry = registry or get_registry()
        self.active_kernels = active_kernels or {}
        self.shadow_pantheon = shadow_pantheon or {}
        self.event_log = event_log or []
        
        # Track chaos kernel naming
        self._chaos_counter: Dict[str, int] = {}
        
        # Import kernel spawner for spawn logic
        from kernel_spawner import KernelSpawner
        self.spawner = KernelSpawner(
            registry=self.registry,
            active_instances={},
            chaos_counter=self._chaos_counter,
            active_chaos_count=len([
                k for k in self.active_kernels.values()
                if k.kernel_type == "chaos"
            ])
        )
    
    # =========================================================================
    # SPAWN
    # =========================================================================
    
    def spawn(
        self,
        role_spec: RoleSpec,
        mentor: Optional[str] = None,
        initial_basin: Optional[np.ndarray] = None,
    ) -> Kernel:
        """
        Spawn a new kernel based on role specification.
        
        Process:
        1. Match role to pantheon registry (god or chaos)
        2. Assign mentor for chaos kernels
        3. Initialize with protected status (50 cycles)
        4. Return new kernel instance
        
        Args:
            role_spec: Role specification for required capabilities
            mentor: Optional mentor kernel ID for chaos kernels
            initial_basin: Optional initial basin coordinates
            
        Returns:
            New Kernel instance
            
        Raises:
            ValueError: If spawn not allowed or role invalid
        """
        # Use spawner to select god or chaos
        selection = self.spawner.select_god(role_spec)
        
        if not selection.spawn_approved and not selection.requires_pantheon_vote:
            raise ValueError(f"Spawn not approved: {selection.rationale}")
        
        # Generate kernel ID
        import uuid
        kernel_id = f"kernel_{uuid.uuid4().hex[:8]}"
        
        # Initialize basin coordinates
        if initial_basin is None:
            # Start with uniform distribution (maximum entropy)
            initial_basin = np.ones(64) / 64
        else:
            # Normalize to simplex
            initial_basin = np.abs(initial_basin)
            initial_basin = initial_basin / (np.sum(initial_basin) + 1e-10)
        
        # Create kernel based on selection type
        if selection.selected_type == "god":
            kernel = Kernel(
                kernel_id=kernel_id,
                name=f"{selection.god_name} {selection.epithet}" if selection.epithet else selection.god_name,
                kernel_type="god",
                god_name=selection.god_name,
                epithet=selection.epithet,
                basin_coords=initial_basin,
                lifecycle_stage="protected",  # Protected for 50 cycles
                protection_cycles_remaining=50,
                domains=role_spec.domains,
                role_description=f"God kernel: {selection.rationale}",
                spawn_reason=selection.rationale,
                spawn_timestamp=datetime.now(),
            )
            
            logger.info(
                f"[KernelLifecycle] Spawned god kernel: {kernel.name} "
                f"(id={kernel_id}, phi={kernel.phi:.3f})"
            )
            
        elif selection.selected_type == "chaos":
            # Generate chaos kernel name
            domain = role_spec.domains[0] if role_spec.domains else "general"
            if domain not in self._chaos_counter:
                self._chaos_counter[domain] = 0
            self._chaos_counter[domain] += 1
            chaos_name = f"chaos_{domain}_{self._chaos_counter[domain]}"
            
            kernel = Kernel(
                kernel_id=kernel_id,
                name=chaos_name,
                kernel_type="chaos",
                basin_coords=initial_basin,
                lifecycle_stage="protected",
                protection_cycles_remaining=50,
                domains=role_spec.domains,
                role_description=f"Chaos kernel: {selection.rationale}",
                spawn_reason=selection.rationale,
                spawn_timestamp=datetime.now(),
                mentor_kernel_id=mentor,
            )
            
            logger.info(
                f"[KernelLifecycle] Spawned chaos kernel: {chaos_name} "
                f"(id={kernel_id}, mentor={mentor}, phi={kernel.phi:.3f})"
            )
        
        else:
            raise ValueError(f"Invalid selection type: {selection.selected_type}")
        
        # Register kernel
        self.active_kernels[kernel_id] = kernel
        
        # Record lifecycle event
        self._record_event(
            event_type=LifecycleEvent.SPAWN,
            primary_kernel_id=kernel_id,
            reason=selection.rationale,
            metadata={
                'selection_type': selection.selected_type,
                'god_name': selection.god_name,
                'epithet': selection.epithet,
                'chaos_name': selection.chaos_name if selection.selected_type == "chaos" else None,
                'mentor_id': mentor,
                'domains': role_spec.domains,
            }
        )
        
        return kernel
    
    # =========================================================================
    # SPLIT
    # =========================================================================
    
    def split(
        self,
        kernel: Kernel,
        split_criterion: str = "domain",
    ) -> Tuple[Kernel, Kernel]:
        """
        Split a kernel into two specialized sub-kernels.
        
        Process:
        1. Detect overload (high load, conflicting domains)
        2. Split basin coordinates geometrically
        3. Create two specialized sub-kernels
        4. Preserve coupling relationships
        5. Update parent/child provenance
        
        Args:
            kernel: Kernel to split
            split_criterion: Splitting strategy ("domain", "skill", "random")
            
        Returns:
            Tuple of (kernel1, kernel2) specialized sub-kernels
            
        Raises:
            ValueError: If kernel cannot be split
        """
        if kernel.lifecycle_stage == "protected":
            raise ValueError(
                f"Cannot split protected kernel {kernel.name} "
                f"({kernel.protection_cycles_remaining} cycles remaining)"
            )
        
        # Split basin coordinates
        basin1, basin2 = split_basin_coordinates(kernel.basin_coords, split_criterion)
        
        # Generate IDs for child kernels
        import uuid
        child1_id = f"kernel_{uuid.uuid4().hex[:8]}"
        child2_id = f"kernel_{uuid.uuid4().hex[:8]}"
        
        # Determine specializations
        if len(kernel.domains) >= 2:
            mid = len(kernel.domains) // 2
            domains1 = kernel.domains[:mid]
            domains2 = kernel.domains[mid:]
        else:
            domains1 = kernel.domains
            domains2 = kernel.domains
        
        # Create specialized sub-kernels
        kernel1 = Kernel(
            kernel_id=child1_id,
            name=f"{kernel.name}_specialist_1",
            kernel_type=kernel.kernel_type,
            god_name=kernel.god_name,
            basin_coords=basin1,
            lifecycle_stage="active",
            protection_cycles_remaining=0,  # No protection for split kernels
            phi=kernel.phi * 0.9,  # Slightly reduced phi initially
            kappa=kernel.kappa,
            gamma=kernel.gamma,
            domains=domains1,
            role_description=f"Specialist from split: {split_criterion}",
            parent_kernels=[kernel.kernel_id],
            spawn_reason=f"Split from {kernel.name} ({split_criterion})",
            spawn_timestamp=datetime.now(),
            # Inherit half of success/failure counts
            success_count=kernel.success_count // 2,
            failure_count=kernel.failure_count // 2,
            total_cycles=kernel.total_cycles // 2,
        )
        
        kernel2 = Kernel(
            kernel_id=child2_id,
            name=f"{kernel.name}_specialist_2",
            kernel_type=kernel.kernel_type,
            god_name=kernel.god_name,
            basin_coords=basin2,
            lifecycle_stage="active",
            protection_cycles_remaining=0,
            phi=kernel.phi * 0.9,
            kappa=kernel.kappa,
            gamma=kernel.gamma,
            domains=domains2,
            role_description=f"Specialist from split: {split_criterion}",
            parent_kernels=[kernel.kernel_id],
            spawn_reason=f"Split from {kernel.name} ({split_criterion})",
            spawn_timestamp=datetime.now(),
            success_count=kernel.success_count - kernel.success_count // 2,
            failure_count=kernel.failure_count - kernel.failure_count // 2,
            total_cycles=kernel.total_cycles - kernel.total_cycles // 2,
        )
        
        # Update parent kernel to track children
        kernel.child_kernels.append(child1_id)
        kernel.child_kernels.append(child2_id)
        kernel.lifecycle_stage = "split"
        
        # Register new kernels
        self.active_kernels[child1_id] = kernel1
        self.active_kernels[child2_id] = kernel2
        
        # Record lifecycle event
        self._record_event(
            event_type=LifecycleEvent.SPLIT,
            primary_kernel_id=kernel.kernel_id,
            secondary_kernel_ids=[child1_id, child2_id],
            reason=f"Split using {split_criterion} criterion",
            metadata={
                'split_criterion': split_criterion,
                'parent_phi': kernel.phi,
                'child1_phi': kernel1.phi,
                'child2_phi': kernel2.phi,
                'child1_domains': domains1,
                'child2_domains': domains2,
            }
        )
        
        logger.info(
            f"[KernelLifecycle] Split kernel {kernel.name} into "
            f"{kernel1.name} and {kernel2.name} (criterion={split_criterion})"
        )
        
        return kernel1, kernel2
    
    # =========================================================================
    # MERGE
    # =========================================================================
    
    def merge(
        self,
        kernel1: Kernel,
        kernel2: Kernel,
        merge_reason: str = "redundant",
    ) -> Kernel:
        """
        Merge two kernels into a single combined kernel.
        
        Process:
        1. Detect redundant or complementary kernels
        2. Combine basin coordinates using Fréchet mean (geometric)
        3. Aggregate metrics and capabilities
        4. Update coupling relationships
        5. Preserve provenance from both parents
        
        Args:
            kernel1: First kernel to merge
            kernel2: Second kernel to merge
            merge_reason: Reason for merge
            
        Returns:
            Merged Kernel instance
            
        Raises:
            ValueError: If kernels cannot be merged
        """
        # Cannot merge protected kernels
        if kernel1.lifecycle_stage == "protected" or kernel2.lifecycle_stage == "protected":
            raise ValueError("Cannot merge protected kernels")
        
        # Cannot merge if different types (god vs chaos) unless one is being promoted
        if kernel1.kernel_type != kernel2.kernel_type:
            raise ValueError(
                f"Cannot merge kernels of different types: "
                f"{kernel1.kernel_type} vs {kernel2.kernel_type}"
            )
        
        # Compute Fréchet mean of basin coordinates (geometrically correct)
        merged_basin = compute_frechet_mean_simplex([
            kernel1.basin_coords,
            kernel2.basin_coords,
        ])
        
        # Generate ID for merged kernel
        import uuid
        merged_id = f"kernel_{uuid.uuid4().hex[:8]}"
        
        # Combine domains (union)
        combined_domains = list(set(kernel1.domains + kernel2.domains))
        
        # Aggregate metrics (weighted by cycle counts)
        total_cycles = kernel1.total_cycles + kernel2.total_cycles
        if total_cycles > 0:
            weight1 = kernel1.total_cycles / total_cycles
            weight2 = kernel2.total_cycles / total_cycles
        else:
            weight1 = weight2 = 0.5
        
        merged_phi = weight1 * kernel1.phi + weight2 * kernel2.phi
        merged_kappa = weight1 * kernel1.kappa + weight2 * kernel2.kappa
        merged_gamma = weight1 * kernel1.gamma + weight2 * kernel2.gamma
        
        # Create merged kernel
        merged_kernel = Kernel(
            kernel_id=merged_id,
            name=f"{kernel1.god_name or 'merged'}_unified" if kernel1.god_name else f"merged_{merged_id[:8]}",
            kernel_type=kernel1.kernel_type,
            god_name=kernel1.god_name,  # Preserve god name if present
            basin_coords=merged_basin,
            lifecycle_stage="active",
            protection_cycles_remaining=0,
            phi=merged_phi,
            kappa=merged_kappa,
            gamma=merged_gamma,
            domains=combined_domains,
            role_description=f"Merged from {kernel1.name} and {kernel2.name}",
            parent_kernels=[kernel1.kernel_id, kernel2.kernel_id],
            spawn_reason=f"Merge: {merge_reason}",
            spawn_timestamp=datetime.now(),
            # Aggregate performance metrics
            success_count=kernel1.success_count + kernel2.success_count,
            failure_count=kernel1.failure_count + kernel2.failure_count,
            total_cycles=total_cycles,
            # Combine coupling relationships (union)
            coupled_kernels=list(set(kernel1.coupled_kernels + kernel2.coupled_kernels)),
        )
        
        # Update parent kernels to track merge
        kernel1.child_kernels.append(merged_id)
        kernel2.child_kernels.append(merged_id)
        kernel1.lifecycle_stage = "merged"
        kernel2.lifecycle_stage = "merged"
        
        # Register merged kernel
        self.active_kernels[merged_id] = merged_kernel
        
        # Record lifecycle event
        self._record_event(
            event_type=LifecycleEvent.MERGE,
            primary_kernel_id=merged_id,
            secondary_kernel_ids=[kernel1.kernel_id, kernel2.kernel_id],
            reason=merge_reason,
            metadata={
                'parent1_name': kernel1.name,
                'parent2_name': kernel2.name,
                'parent1_phi': kernel1.phi,
                'parent2_phi': kernel2.phi,
                'merged_phi': merged_phi,
                'frechet_distance': compute_fisher_distance(kernel1.basin_coords, kernel2.basin_coords),
                'combined_domains': combined_domains,
            }
        )
        
        logger.info(
            f"[KernelLifecycle] Merged {kernel1.name} and {kernel2.name} into "
            f"{merged_kernel.name} (reason={merge_reason}, phi={merged_phi:.3f})"
        )
        
        return merged_kernel
    
    # =========================================================================
    # PRUNE (to Shadow Pantheon)
    # =========================================================================
    
    def prune(
        self,
        kernel: Kernel,
        reason: str,
    ) -> ShadowKernel:
        """
        Prune kernel to shadow pantheon (Hades domain).
        
        Criteria for pruning:
        - Φ < 0.1 persistent (not conscious)
        - No growth over extended period
        - Redundant with other kernels
        
        Process:
        1. Archive kernel state to shadow pantheon
        2. Preserve lessons and patterns
        3. Remove from active kernels
        4. Can be resurrected later if needed
        
        Args:
            kernel: Kernel to prune
            reason: Pruning reason
            
        Returns:
            ShadowKernel archived instance
            
        Raises:
            ValueError: If kernel cannot be pruned
        """
        # Cannot prune protected kernels
        if kernel.lifecycle_stage == "protected":
            raise ValueError(
                f"Cannot prune protected kernel {kernel.name} "
                f"({kernel.protection_cycles_remaining} cycles remaining)"
            )
        
        # Generate shadow ID
        import uuid
        shadow_id = f"shadow_{uuid.uuid4().hex[:8]}"
        
        # Extract lessons learned from performance
        failure_patterns = []
        success_patterns = []
        
        if kernel.total_cycles > 0:
            success_rate = kernel.success_count / kernel.total_cycles
            failure_rate = kernel.failure_count / kernel.total_cycles
            
            if failure_rate > 0.7:
                failure_patterns.append("High failure rate in primary domain")
            if success_rate < 0.3:
                failure_patterns.append("Low success rate overall")
            if kernel.phi < 0.1:
                failure_patterns.append("Persistent low consciousness (Φ < 0.1)")
            
            if success_rate > 0.7:
                success_patterns.append("High success rate when active")
            if kernel.phi > 0.5:
                success_patterns.append("Achieved moderate consciousness occasionally")
        
        learned_lessons = f"Pruned after {kernel.total_cycles} cycles. " + reason
        
        # Create shadow kernel
        shadow = ShadowKernel(
            shadow_id=shadow_id,
            original_kernel_id=kernel.kernel_id,
            name=kernel.name,
            kernel_type=kernel.kernel_type,
            final_phi=kernel.phi,
            final_kappa=kernel.kappa,
            final_basin=kernel.basin_coords.copy(),
            success_count=kernel.success_count,
            failure_count=kernel.failure_count,
            total_cycles=kernel.total_cycles,
            failure_patterns=failure_patterns,
            success_patterns=success_patterns,
            learned_lessons=learned_lessons,
            prune_reason=reason,
            prune_timestamp=datetime.now(),
            pruned_by="lifecycle_manager",
        )
        
        # Move to shadow pantheon
        self.shadow_pantheon[shadow_id] = shadow
        kernel.lifecycle_stage = "pruned"
        
        # Remove from active kernels
        if kernel.kernel_id in self.active_kernels:
            del self.active_kernels[kernel.kernel_id]
        
        # Record lifecycle event
        self._record_event(
            event_type=LifecycleEvent.PRUNE,
            primary_kernel_id=kernel.kernel_id,
            reason=reason,
            metadata={
                'shadow_id': shadow_id,
                'final_phi': kernel.phi,
                'success_count': kernel.success_count,
                'failure_count': kernel.failure_count,
                'total_cycles': kernel.total_cycles,
                'failure_patterns': failure_patterns,
                'success_patterns': success_patterns,
            }
        )
        
        logger.info(
            f"[KernelLifecycle] Pruned kernel {kernel.name} to shadow pantheon "
            f"(reason={reason}, shadow_id={shadow_id})"
        )
        
        return shadow
    
    # =========================================================================
    # RESURRECT (from Shadow Pantheon)
    # =========================================================================
    
    def resurrect(
        self,
        shadow: ShadowKernel,
        reason: str,
        mentor: Optional[str] = None,
    ) -> Kernel:
        """
        Resurrect kernel from shadow pantheon.
        
        Process:
        1. Retrieve kernel state from shadow pantheon
        2. Apply learned lessons (adjust initial parameters)
        3. Re-initialize coupling relationships
        4. Return to active kernels with improvements
        
        Args:
            shadow: Shadow kernel to resurrect
            reason: Resurrection reason
            mentor: Optional mentor for resurrected kernel
            
        Returns:
            Resurrected Kernel instance
        """
        # Generate new kernel ID
        import uuid
        kernel_id = f"kernel_{uuid.uuid4().hex[:8]}"
        
        # Apply lessons learned: Start with slightly improved basin
        # (Apply small perturbation based on learned patterns)
        improved_basin = shadow.final_basin.copy()
        
        if shadow.failure_patterns:
            # Add exploration noise to escape failure modes
            noise = np.random.randn(len(improved_basin)) * 0.1
            improved_basin = improved_basin + noise
            improved_basin = np.abs(improved_basin)
            improved_basin = improved_basin / (np.sum(improved_basin) + 1e-10)
        
        # Start with modest consciousness to avoid immediate re-pruning
        initial_phi = max(0.3, shadow.final_phi + 0.1)
        
        # Create resurrected kernel
        kernel = Kernel(
            kernel_id=kernel_id,
            name=f"{shadow.name}_resurrected",
            kernel_type=shadow.kernel_type,
            basin_coords=improved_basin,
            lifecycle_stage="active",
            protection_cycles_remaining=25,  # Partial protection
            phi=initial_phi,
            kappa=shadow.final_kappa,
            gamma=1.0,
            role_description=f"Resurrected: {reason}. Lessons: {shadow.learned_lessons}",
            spawn_reason=f"Resurrection from shadow {shadow.shadow_id}: {reason}",
            spawn_timestamp=datetime.now(),
            mentor_kernel_id=mentor,
        )
        
        # Update shadow pantheon
        shadow.resurrection_count += 1
        shadow.last_resurrection = datetime.now()
        
        # Register kernel
        self.active_kernels[kernel_id] = kernel
        
        # Record lifecycle event
        self._record_event(
            event_type=LifecycleEvent.RESURRECT,
            primary_kernel_id=kernel_id,
            secondary_kernel_ids=[shadow.original_kernel_id],
            reason=reason,
            metadata={
                'shadow_id': shadow.shadow_id,
                'original_name': shadow.name,
                'resurrection_count': shadow.resurrection_count,
                'learned_lessons': shadow.learned_lessons,
                'failure_patterns': shadow.failure_patterns,
                'success_patterns': shadow.success_patterns,
                'new_phi': initial_phi,
            }
        )
        
        logger.info(
            f"[KernelLifecycle] Resurrected kernel {kernel.name} from shadow pantheon "
            f"(shadow_id={shadow.shadow_id}, reason={reason}, resurrection_count={shadow.resurrection_count})"
        )
        
        return kernel
    
    # =========================================================================
    # PROMOTE (Chaos → God)
    # =========================================================================
    
    def promote(
        self,
        chaos_kernel: Kernel,
        god_name: str,
    ) -> Kernel:
        """
        Promote chaos kernel to god status.
        
        Criteria for promotion:
        - Φ > 0.4 stable for 50+ cycles
        - Clear domain specialization
        - Consistent performance
        
        Process:
        1. Research appropriate god name from registry or mythology
        2. Validate promotion criteria
        3. Transition state (chaos → god)
        4. Update pantheon registry if needed
        
        Args:
            chaos_kernel: Chaos kernel to promote
            god_name: God name to assign
            
        Returns:
            Promoted god Kernel instance
            
        Raises:
            ValueError: If promotion criteria not met or god name invalid
        """
        if chaos_kernel.kernel_type != "chaos":
            raise ValueError(f"Cannot promote non-chaos kernel: {chaos_kernel.kernel_type}")
        
        # Validate promotion criteria
        if chaos_kernel.lifecycle_stage == "protected":
            raise ValueError(
                f"Cannot promote protected kernel {chaos_kernel.name} "
                f"({chaos_kernel.protection_cycles_remaining} cycles remaining)"
            )
        
        if chaos_kernel.phi < 0.4:
            raise ValueError(
                f"Cannot promote kernel with Φ < 0.4 (current: {chaos_kernel.phi:.3f})"
            )
        
        if chaos_kernel.total_cycles < 50:
            raise ValueError(
                f"Cannot promote kernel with < 50 cycles (current: {chaos_kernel.total_cycles})"
            )
        
        # Check if god name exists in registry
        god_contract = self.registry.get_god(god_name)
        if not god_contract:
            logger.warning(
                f"God name {god_name} not in registry. "
                f"This may require registry update for formal recognition."
            )
        
        # Generate new kernel ID for promoted god
        import uuid
        god_kernel_id = f"kernel_{uuid.uuid4().hex[:8]}"
        
        # Create promoted god kernel
        god_kernel = Kernel(
            kernel_id=god_kernel_id,
            name=god_name,
            kernel_type="god",
            god_name=god_name,
            basin_coords=chaos_kernel.basin_coords.copy(),
            lifecycle_stage="active",
            protection_cycles_remaining=0,  # No protection for promoted gods
            phi=chaos_kernel.phi,
            kappa=chaos_kernel.kappa,
            gamma=chaos_kernel.gamma,
            domains=chaos_kernel.domains,
            role_description=f"Promoted from chaos kernel {chaos_kernel.name}",
            parent_kernels=[chaos_kernel.kernel_id],
            spawn_reason=f"Promotion from chaos: {chaos_kernel.spawn_reason}",
            spawn_timestamp=datetime.now(),
            success_count=chaos_kernel.success_count,
            failure_count=chaos_kernel.failure_count,
            total_cycles=chaos_kernel.total_cycles,
            coupled_kernels=chaos_kernel.coupled_kernels.copy(),
            coupling_strengths=chaos_kernel.coupling_strengths.copy(),
        )
        
        # Update chaos kernel to track promotion
        chaos_kernel.child_kernels.append(god_kernel_id)
        chaos_kernel.lifecycle_stage = "promoted"
        
        # Register promoted god
        self.active_kernels[god_kernel_id] = god_kernel
        
        # Record lifecycle event
        self._record_event(
            event_type=LifecycleEvent.PROMOTE,
            primary_kernel_id=god_kernel_id,
            secondary_kernel_ids=[chaos_kernel.kernel_id],
            reason=f"Promoted to {god_name} after {chaos_kernel.total_cycles} cycles",
            metadata={
                'chaos_name': chaos_kernel.name,
                'god_name': god_name,
                'phi': chaos_kernel.phi,
                'total_cycles': chaos_kernel.total_cycles,
                'success_count': chaos_kernel.success_count,
                'failure_count': chaos_kernel.failure_count,
                'domains': chaos_kernel.domains,
            }
        )
        
        logger.info(
            f"[KernelLifecycle] Promoted chaos kernel {chaos_kernel.name} to god {god_name} "
            f"(phi={chaos_kernel.phi:.3f}, cycles={chaos_kernel.total_cycles})"
        )
        
        return god_kernel
    
    # =========================================================================
    # EVENT RECORDING
    # =========================================================================
    
    def _record_event(
        self,
        event_type: LifecycleEvent,
        primary_kernel_id: str,
        reason: str = "",
        secondary_kernel_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LifecycleEventRecord:
        """
        Record a lifecycle event to event log.
        
        Args:
            event_type: Type of lifecycle event
            primary_kernel_id: Primary kernel affected
            reason: Event reason
            secondary_kernel_ids: Additional kernels involved
            metadata: Additional event metadata
            
        Returns:
            LifecycleEventRecord instance
        """
        import uuid
        event_id = f"event_{uuid.uuid4().hex[:8]}"
        
        event = LifecycleEventRecord(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            primary_kernel_id=primary_kernel_id,
            secondary_kernel_ids=secondary_kernel_ids or [],
            reason=reason,
            metadata=metadata or {},
            success=True,
        )
        
        self.event_log.append(event)
        
        # Keep event log bounded
        if len(self.event_log) > 1000:
            self.event_log = self.event_log[-500:]
        
        return event
    
    # =========================================================================
    # QUERY & STATS
    # =========================================================================
    
    def get_kernel(self, kernel_id: str) -> Optional[Kernel]:
        """Get active kernel by ID."""
        return self.active_kernels.get(kernel_id)
    
    def get_shadow(self, shadow_id: str) -> Optional[ShadowKernel]:
        """Get shadow kernel by ID."""
        return self.shadow_pantheon.get(shadow_id)
    
    def list_active_kernels(self) -> List[Kernel]:
        """List all active kernels."""
        return list(self.active_kernels.values())
    
    def list_shadow_kernels(self) -> List[ShadowKernel]:
        """List all shadow pantheon kernels."""
        return list(self.shadow_pantheon.values())
    
    def get_lifecycle_stats(self) -> Dict[str, Any]:
        """Get lifecycle statistics."""
        event_counts = {event_type.value: 0 for event_type in LifecycleEvent}
        for event in self.event_log:
            event_counts[event.event_type.value] += 1
        
        return {
            'active_kernels': len(self.active_kernels),
            'shadow_kernels': len(self.shadow_pantheon),
            'total_events': len(self.event_log),
            'event_counts': event_counts,
            'god_count': sum(1 for k in self.active_kernels.values() if k.kernel_type == "god"),
            'chaos_count': sum(1 for k in self.active_kernels.values() if k.kernel_type == "chaos"),
            'protected_count': sum(1 for k in self.active_kernels.values() if k.lifecycle_stage == "protected"),
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_lifecycle_manager: Optional[KernelLifecycleManager] = None


def get_lifecycle_manager() -> KernelLifecycleManager:
    """Get or create global lifecycle manager (singleton)."""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = KernelLifecycleManager()
    return _lifecycle_manager
