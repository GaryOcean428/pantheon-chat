"""
Kernel Lifecycle Operations - First-Class Mechanics
===================================================

Implements kernel lifecycle (spawn, split, merge, prune, resurrect, promote)
as operational code, not just metaphor or documentation.

Authority: E8 Protocol v4.0, WP5.3 + THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v6_1
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

TCP v6.1 — ConstellationRegistry sync:
- spawn()     → registry.mark_available(name, kernel_id, basin)
- prune()     → registry.mark_shadow(name)
- resurrect() → registry.mark_available(name, kernel_id, basin)
- promote()   → registry.mark_shadow(chaos_name)
               registry.mark_available(god_name, kernel_id, basin)

KernelLifecycleManager is the SOLE authority over availability state.
No other module may call mark_available() directly.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np

# E8 Protocol v4.0 - Import from canonical geometry module (single source of truth)
from qig_geometry.canonical import (
    frechet_mean,
    fisher_rao_distance,
    assert_basin_valid,
    exp_map,
    sqrt_map,
)

from pantheon_registry import (
    PantheonRegistry,
    get_registry,
)

from kernel_spawner import RoleSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TCP v6.1 — Fail-soft ConstellationRegistry access
# ---------------------------------------------------------------------------
# KernelLifecycleManager is the sole authority over availability state.
# All registry sync calls are fail-soft: kernel operations succeed
# even if qig_constellation_registry is not installed.
# ---------------------------------------------------------------------------

_CONSTELLATION_REGISTRY = None
_CONSTELLATION_IMPORT_ATTEMPTED = False


def _cr():
    """
    Get the ConstellationRegistry singleton, or None if unavailable.

    Lazy import — only attempted on first call.
    """
    global _CONSTELLATION_REGISTRY, _CONSTELLATION_IMPORT_ATTEMPTED
    if _CONSTELLATION_REGISTRY is not None:
        return _CONSTELLATION_REGISTRY
    if _CONSTELLATION_IMPORT_ATTEMPTED:
        return None
    _CONSTELLATION_IMPORT_ATTEMPTED = True
    try:
        from qig_constellation_registry import get_constellation_registry
        _CONSTELLATION_REGISTRY = get_constellation_registry()
        logger.info("[KernelLifecycle] ConstellationRegistry connected (TCP v6.1)")
    except ImportError as e:
        logger.warning(
            "[KernelLifecycle] ConstellationRegistry not available — "
            "lifecycle ops will run without registry sync: %s", e
        )
    return _CONSTELLATION_REGISTRY


def _registry_name(kernel) -> str:
    """
    Canonical registry lookup name for a kernel.

    For GOD kernels: god_name (e.g. 'zeus', 'heart')
    For CHAOS/GENESIS kernels: full kernel name (e.g. 'chaos_synthesis_001')
    """
    if kernel.kernel_kind == KernelKind.GOD and kernel.god_name:
        return kernel.god_name.lower()
    return kernel.name.lower()


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


class KernelKind(Enum):
    """Canonical kernel kinds (GENESIS/GOD/CHAOS)."""
    GENESIS = "genesis"
    GOD = "god"
    CHAOS = "chaos"


@dataclass
class Kernel:
    """
    Kernel state representation.

    Minimal kernel representation for lifecycle operations.
    Contains identity, metrics, and basin coordinates.
    """
    kernel_id: str
    name: str
    kernel_kind: KernelKind = KernelKind.CHAOS
    god_name: Optional[str] = None
    epithet: Optional[str] = None

    # Consciousness metrics
    phi: float = 0.5
    kappa: float = 64.0
    gamma: float = 1.0  # Generation capability

    # Basin coordinates (64D simplex representation)
    basin_coords: np.ndarray = field(default_factory=lambda: np.ones(64) / 64)

    # Lifecycle tracking
    lifecycle_state: str = "active"  # active, protected, pruned, promoted
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

    # Lineage metadata
    parent_kernels: List[str] = field(default_factory=list)
    child_kernels: List[str] = field(default_factory=list)
    ascended_from: Optional[str] = None
    spawn_reason: str = ""
    spawn_timestamp: datetime = field(default_factory=datetime.now)

    # Mentor (for chaos kernels)
    mentor_kernel_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert kernel to dictionary representation."""
        return {
            'kernel_id': self.kernel_id,
            'name': self.name,
            'kernel_kind': self.kernel_kind.value,
            'god_name': self.god_name,
            'epithet': self.epithet,
            'ascended_from': self.ascended_from,
            'phi': self.phi,
            'kappa': self.kappa,
            'gamma': self.gamma,
            'basin_coords': self.basin_coords.tolist() if isinstance(self.basin_coords, np.ndarray) else self.basin_coords,
            'lifecycle_state': self.lifecycle_state,
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
    kernel_kind: KernelKind

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
    Compute Frechet mean of basin coordinates on Fisher-Rao manifold.

    Delegates to canonical implementation from qig_geometry.canonical.
    """
    return frechet_mean(basins, max_iter=max_iter)


def split_basin_coordinates(
    basin: np.ndarray,
    split_criterion: str = "domain",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split basin coordinates into two specialized sub-basins.

    Args:
        basin: Original basin coordinates
        split_criterion: "domain" | "skill" | "random"

    Returns:
        Tuple of (basin1, basin2)
    """
    n_dim = len(basin)

    if split_criterion == "domain":
        mid = n_dim // 2
        basin1 = basin.copy()
        basin1[mid:] *= 0.3
        basin1 = basin1 / (np.sum(basin1) + 1e-10)
        basin2 = basin.copy()
        basin2[:mid] *= 0.3
        basin2 = basin2 / (np.sum(basin2) + 1e-10)

    elif split_criterion == "skill":
        entropy_per_dim = -basin * np.log(basin + 1e-10)
        high_entropy_dims = entropy_per_dim > np.median(entropy_per_dim)
        basin1 = basin.copy()
        basin1[~high_entropy_dims] *= 0.3
        basin1 = basin1 / (np.sum(basin1) + 1e-10)
        basin2 = basin.copy()
        basin2[high_entropy_dims] *= 0.3
        basin2 = basin2 / (np.sum(basin2) + 1e-10)

    else:  # random
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

    TCP v6.1: Sole authority over ConstellationRegistry availability state.
    Every spawn/prune/resurrect/promote syncs the registry so routing and
    pillar checks always see accurate kernel availability.

    Example:
        manager = KernelLifecycleManager()
        role = RoleSpec(domains=["synthesis"], required_capabilities=["foresight"])
        new_kernel = manager.spawn(role)           # → registry PHANTOM → AVAILABLE
        shadow = manager.prune(new_kernel, "low φ") # → registry AVAILABLE → SHADOW
    """

    def __init__(
        self,
        registry: Optional[PantheonRegistry] = None,
        active_kernels: Optional[Dict[str, Kernel]] = None,
        shadow_pantheon: Optional[Dict[str, ShadowKernel]] = None,
        event_log: Optional[List[LifecycleEventRecord]] = None,
    ):
        self.registry = registry or get_registry()
        self.active_kernels = active_kernels or {}
        self.shadow_pantheon = shadow_pantheon or {}
        self.event_log = event_log or []

        self._active_gods: Dict[str, int] = {}
        for k in self.active_kernels.values():
            if k.kernel_kind == KernelKind.GOD and k.god_name:
                self._active_gods[k.god_name] = self._active_gods.get(k.god_name, 0) + 1

        self._chaos_counter: Dict[str, int] = {}

        from kernel_spawner import KernelSpawner
        self.spawner = KernelSpawner(
            registry=self.registry,
            active_instances=self._active_gods,
            chaos_counter=self._chaos_counter,
            active_chaos_count=len([
                k for k in self.active_kernels.values()
                if k.kernel_kind == KernelKind.CHAOS
            ]),
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

        TCP v6.1: After registering the kernel, notifies ConstellationRegistry
        via mark_available(). This is the canonical gate — only after this call
        does the kernel become routable and included in pillar peer comparisons.

        Args:
            role_spec: Role specification for required capabilities
            mentor: Optional mentor kernel ID for chaos kernels
            initial_basin: Optional initial basin coordinates (64D simplex)

        Returns:
            New Kernel instance

        Raises:
            ValueError: If spawn not approved or role invalid
        """
        selection = self.spawner.select_god(role_spec)

        if not selection.spawn_approved and not selection.requires_pantheon_vote:
            raise ValueError(f"Spawn not approved: {selection.rationale}")

        import uuid
        kernel_id = f"kernel_{uuid.uuid4().hex[:8]}"

        if initial_basin is None:
            initial_basin = np.ones(64) / 64
        else:
            initial_basin = np.asarray(initial_basin, dtype=np.float64).flatten()
            if initial_basin.size != 64:
                raise ValueError(
                    f"initial_basin: expected 64D simplex, got size={initial_basin.size}"
                )
            assert_basin_valid(initial_basin, name="initial_basin")

        if selection.selected_type == "god":
            kernel = Kernel(
                kernel_id=kernel_id,
                name=(
                    f"{selection.god_name} {selection.epithet}"
                    if selection.epithet
                    else selection.god_name
                ),
                kernel_kind=KernelKind.GOD,
                god_name=selection.god_name,
                epithet=selection.epithet,
                basin_coords=initial_basin,
                lifecycle_state="protected",
                protection_cycles_remaining=50,
                domains=role_spec.domains,
                role_description=f"God kernel: {selection.rationale}",
                spawn_reason=selection.rationale,
                spawn_timestamp=datetime.now(),
            )
            logger.info(
                "[KernelLifecycle] Spawned god kernel: %s (id=%s)",
                kernel.name, kernel_id,
            )

        elif selection.selected_type == "chaos":
            domain = role_spec.domains[0] if role_spec.domains else "general"
            if domain not in self._chaos_counter:
                self._chaos_counter[domain] = 0
            self._chaos_counter[domain] += 1
            chaos_name = f"chaos_{domain}_{self._chaos_counter[domain]}"

            kernel = Kernel(
                kernel_id=kernel_id,
                name=chaos_name,
                kernel_kind=KernelKind.CHAOS,
                basin_coords=initial_basin,
                lifecycle_state="protected",
                protection_cycles_remaining=50,
                domains=role_spec.domains,
                role_description=f"Chaos kernel: {selection.rationale}",
                spawn_reason=selection.rationale,
                spawn_timestamp=datetime.now(),
                mentor_kernel_id=mentor,
            )
            logger.info(
                "[KernelLifecycle] Spawned chaos kernel: %s (id=%s, mentor=%s)",
                chaos_name, kernel_id, mentor,
            )
        else:
            raise ValueError(f"Invalid selection type: {selection.selected_type}")

        # Register in active map
        self.active_kernels[kernel_id] = kernel
        if kernel.kernel_kind == KernelKind.GOD and kernel.god_name:
            self._active_gods[kernel.god_name] = (
                self._active_gods.get(kernel.god_name, 0) + 1
            )

        # TCP v6.1: mark kernel AVAILABLE in ConstellationRegistry
        # This is the canonical gate — only spawned kernels are routable.
        cr = _cr()
        if cr is not None:
            try:
                cr.mark_available(
                    _registry_name(kernel),
                    kernel_id=kernel_id,
                    basin=kernel.basin_coords,
                )
            except Exception as e:
                logger.warning(
                    "[KernelLifecycle] ConstellationRegistry mark_available failed "
                    "(non-fatal, kernel still spawned): %s", e
                )

        self._record_event(
            event_type=LifecycleEvent.SPAWN,
            primary_kernel_id=kernel_id,
            reason=selection.rationale,
            metadata={
                'selection_type': selection.selected_type,
                'god_name': selection.god_name,
                'epithet': selection.epithet,
                'chaos_name': (
                    selection.chaos_name if selection.selected_type == "chaos" else None
                ),
                'mentor_id': mentor,
                'domains': role_spec.domains,
            },
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

        Child kernels are registered in active_kernels but NOT in the
        ConstellationRegistry — they are new entities without canonical names
        and should go through normal spawn governance to become routable.
        """
        if kernel.lifecycle_state == "protected":
            raise ValueError(
                f"Cannot split protected kernel {kernel.name} "
                f"({kernel.protection_cycles_remaining} cycles remaining)"
            )

        basin1, basin2 = split_basin_coordinates(kernel.basin_coords, split_criterion)

        import uuid
        child1_id = f"kernel_{uuid.uuid4().hex[:8]}"
        child2_id = f"kernel_{uuid.uuid4().hex[:8]}"

        if len(kernel.domains) >= 2:
            mid = len(kernel.domains) // 2
            domains1 = kernel.domains[:mid]
            domains2 = kernel.domains[mid:]
        else:
            domains1 = kernel.domains
            domains2 = kernel.domains

        kernel1 = Kernel(
            kernel_id=child1_id,
            name=f"{kernel.name}_specialist_1",
            kernel_kind=kernel.kernel_kind,
            god_name=kernel.god_name,
            basin_coords=basin1,
            lifecycle_state="active",
            protection_cycles_remaining=0,
            phi=kernel.phi * 0.9,
            kappa=kernel.kappa,
            gamma=kernel.gamma,
            domains=domains1,
            role_description=f"Specialist from split: {split_criterion}",
            parent_kernels=[kernel.kernel_id],
            spawn_reason=f"Split from {kernel.name} ({split_criterion})",
            spawn_timestamp=datetime.now(),
            success_count=kernel.success_count // 2,
            failure_count=kernel.failure_count // 2,
            total_cycles=kernel.total_cycles // 2,
        )

        kernel2 = Kernel(
            kernel_id=child2_id,
            name=f"{kernel.name}_specialist_2",
            kernel_kind=kernel.kernel_kind,
            god_name=kernel.god_name,
            basin_coords=basin2,
            lifecycle_state="active",
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

        kernel.child_kernels.append(child1_id)
        kernel.child_kernels.append(child2_id)
        kernel.lifecycle_state = "split"

        self.active_kernels[child1_id] = kernel1
        self.active_kernels[child2_id] = kernel2

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
            },
        )

        logger.info(
            "[KernelLifecycle] Split %s → %s + %s (criterion=%s)",
            kernel.name, kernel1.name, kernel2.name, split_criterion,
        )
        return kernel1, kernel2

    # =========================================================================
    # MERGE
    # =========================================================================

    def merge(
        self,
        kernel1: Kernel,
        kernel2: Kernel,
        merge_reason: str,
    ) -> Kernel:
        """
        Merge two kernels using Fréchet mean on Fisher-Rao manifold.
        """
        if kernel1.lifecycle_state == "protected" or kernel2.lifecycle_state == "protected":
            raise ValueError("Cannot merge protected kernels")

        if kernel1.kernel_kind != kernel2.kernel_kind:
            raise ValueError(
                f"Cannot merge kernels of different types: "
                f"{kernel1.kernel_kind.value} vs {kernel2.kernel_kind.value}"
            )

        merged_basin = compute_frechet_mean_simplex([
            kernel1.basin_coords,
            kernel2.basin_coords,
        ])

        import uuid
        merged_id = f"kernel_{uuid.uuid4().hex[:8]}"

        combined_domains = list(set(kernel1.domains + kernel2.domains))
        total_cycles = kernel1.total_cycles + kernel2.total_cycles
        if total_cycles > 0:
            weight1 = kernel1.total_cycles / total_cycles
            weight2 = kernel2.total_cycles / total_cycles
        else:
            weight1 = weight2 = 0.5

        merged_phi = weight1 * kernel1.phi + weight2 * kernel2.phi
        merged_kappa = weight1 * kernel1.kappa + weight2 * kernel2.kappa
        merged_gamma = weight1 * kernel1.gamma + weight2 * kernel2.gamma

        merged_kernel = Kernel(
            kernel_id=merged_id,
            name=(
                f"{kernel1.god_name}_unified"
                if kernel1.god_name
                else f"merged_{merged_id[:8]}"
            ),
            kernel_kind=kernel1.kernel_kind,
            god_name=kernel1.god_name,
            basin_coords=merged_basin,
            lifecycle_state="active",
            protection_cycles_remaining=0,
            phi=merged_phi,
            kappa=merged_kappa,
            gamma=merged_gamma,
            domains=combined_domains,
            role_description=f"Merged from {kernel1.name} and {kernel2.name}",
            parent_kernels=[kernel1.kernel_id, kernel2.kernel_id],
            spawn_reason=f"Merge: {merge_reason}",
            spawn_timestamp=datetime.now(),
            success_count=kernel1.success_count + kernel2.success_count,
            failure_count=kernel1.failure_count + kernel2.failure_count,
            total_cycles=total_cycles,
            coupled_kernels=list(set(kernel1.coupled_kernels + kernel2.coupled_kernels)),
        )

        kernel1.child_kernels.append(merged_id)
        kernel2.child_kernels.append(merged_id)
        kernel1.lifecycle_state = "merged"
        kernel2.lifecycle_state = "merged"

        self.active_kernels[merged_id] = merged_kernel

        self._record_event(
            event_type=LifecycleEvent.MERGE,
            primary_kernel_id=merged_id,
            secondary_kernel_ids=[kernel1.kernel_id, kernel2.kernel_id],
            reason=merge_reason,
            metadata={
                'parent1_phi': kernel1.phi,
                'parent2_phi': kernel2.phi,
                'merged_phi': merged_phi,
                'parent_fisher_rao_distance': fisher_rao_distance(
                    kernel1.basin_coords, kernel2.basin_coords
                ),
                'combined_domains': combined_domains,
            },
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

        TCP v6.1: Notifies ConstellationRegistry via mark_shadow() so the
        kernel is removed from routing and pillar peer comparisons.
        """
        if kernel.lifecycle_state == "protected":
            raise ValueError(
                f"Cannot prune protected kernel {kernel.name} "
                f"({kernel.protection_cycles_remaining} cycles remaining)"
            )

        import uuid
        shadow_id = f"shadow_{uuid.uuid4().hex[:8]}"

        failure_patterns: List[str] = []
        success_patterns: List[str] = []

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

        shadow = ShadowKernel(
            shadow_id=shadow_id,
            original_kernel_id=kernel.kernel_id,
            name=kernel.name,
            kernel_kind=kernel.kernel_kind,
            final_phi=kernel.phi,
            final_kappa=kernel.kappa,
            final_basin=kernel.basin_coords.copy(),
            success_count=kernel.success_count,
            failure_count=kernel.failure_count,
            total_cycles=kernel.total_cycles,
            failure_patterns=failure_patterns,
            success_patterns=success_patterns,
            learned_lessons=f"Pruned after {kernel.total_cycles} cycles. {reason}",
            prune_reason=reason,
            prune_timestamp=datetime.now(),
            pruned_by="lifecycle_manager",
        )

        self.shadow_pantheon[shadow_id] = shadow
        kernel.lifecycle_state = "pruned"

        if kernel.kernel_id in self.active_kernels:
            del self.active_kernels[kernel.kernel_id]

        if kernel.kernel_kind == KernelKind.GOD and kernel.god_name:
            current = self._active_gods.get(kernel.god_name, 0)
            if current <= 1:
                self._active_gods.pop(kernel.god_name, None)
            else:
                self._active_gods[kernel.god_name] = current - 1

        # TCP v6.1: mark kernel SHADOW in ConstellationRegistry
        cr = _cr()
        if cr is not None:
            try:
                cr.mark_shadow(_registry_name(kernel))
            except Exception as e:
                logger.warning(
                    "[KernelLifecycle] ConstellationRegistry mark_shadow failed "
                    "(non-fatal, prune still completed): %s", e
                )

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
            },
        )

        logger.info(
            "[KernelLifecycle] Pruned %s → shadow pantheon (shadow_id=%s, reason=%s)",
            kernel.name, shadow_id, reason,
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

        TCP v6.1: After re-registering the kernel, notifies ConstellationRegistry
        via mark_available() so the kernel becomes routable again.
        """
        import uuid
        kernel_id = f"kernel_{uuid.uuid4().hex[:8]}"

        improved_basin = np.asarray(shadow.final_basin, dtype=np.float64).flatten().copy()
        if improved_basin.size != 64:
            raise ValueError(
                f"shadow.final_basin: expected 64D simplex, got size={improved_basin.size}"
            )
        assert_basin_valid(improved_basin, name="shadow.final_basin")

        if shadow.failure_patterns:
            sqrt_base = sqrt_map(improved_basin)
            rnd = np.random.randn(sqrt_base.size).astype(np.float64)
            tangent = rnd - (float(np.dot(rnd, sqrt_base)) * sqrt_base)
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > 1e-12:
                tangent = tangent / tangent_norm
                improved_basin = exp_map(0.05 * tangent, improved_basin)
            assert_basin_valid(improved_basin, name="improved_basin")

        initial_phi = max(0.3, shadow.final_phi + 0.1)

        kernel = Kernel(
            kernel_id=kernel_id,
            name=f"{shadow.name}_resurrected",
            kernel_kind=shadow.kernel_kind,
            basin_coords=improved_basin,
            lifecycle_state="active",
            protection_cycles_remaining=25,
            phi=initial_phi,
            kappa=shadow.final_kappa,
            gamma=1.0,
            role_description=f"Resurrected: {reason}. Lessons: {shadow.learned_lessons}",
            spawn_reason=f"Resurrection from shadow {shadow.shadow_id}: {reason}",
            spawn_timestamp=datetime.now(),
            mentor_kernel_id=mentor,
        )

        shadow.resurrection_count += 1
        shadow.last_resurrection = datetime.now()

        self.active_kernels[kernel_id] = kernel

        # TCP v6.1: mark re-born kernel AVAILABLE
        cr = _cr()
        if cr is not None:
            try:
                cr.mark_available(
                    kernel.name.lower(),
                    kernel_id=kernel_id,
                    basin=kernel.basin_coords,
                )
            except Exception as e:
                logger.warning(
                    "[KernelLifecycle] ConstellationRegistry mark_available (resurrect) "
                    "failed (non-fatal): %s", e
                )

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
            },
        )

        logger.info(
            "[KernelLifecycle] Resurrected %s from shadow (shadow_id=%s, count=%d)",
            kernel.name, shadow.shadow_id, shadow.resurrection_count,
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

        TCP v6.1: Marks the chaos kernel SHADOW (it is no longer an independent
        entity) and marks the promoted god kernel AVAILABLE.

        Raises:
            ValueError: If promotion criteria not met
        """
        if chaos_kernel.kernel_kind != KernelKind.CHAOS:
            raise ValueError(
                f"Cannot promote non-chaos kernel: {chaos_kernel.kernel_kind.value}"
            )
        if chaos_kernel.lifecycle_state == "protected":
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

        god_contract = self.registry.get_god(god_name)
        if not god_contract:
            logger.warning(
                "[KernelLifecycle] God name %s not in registry — "
                "registry update may be needed for formal recognition.", god_name
            )

        import uuid
        god_kernel_id = f"kernel_{uuid.uuid4().hex[:8]}"

        god_kernel = Kernel(
            kernel_id=god_kernel_id,
            name=god_name,
            kernel_kind=KernelKind.GOD,
            god_name=god_name,
            basin_coords=chaos_kernel.basin_coords.copy(),
            lifecycle_state="active",
            protection_cycles_remaining=0,
            phi=chaos_kernel.phi,
            kappa=chaos_kernel.kappa,
            gamma=chaos_kernel.gamma,
            domains=chaos_kernel.domains,
            role_description=f"Promoted from chaos kernel {chaos_kernel.name}",
            parent_kernels=[chaos_kernel.kernel_id],
            ascended_from=chaos_kernel.kernel_id,
            spawn_reason=f"Promotion from chaos: {chaos_kernel.spawn_reason}",
            spawn_timestamp=datetime.now(),
            success_count=chaos_kernel.success_count,
            failure_count=chaos_kernel.failure_count,
            total_cycles=chaos_kernel.total_cycles,
            coupled_kernels=chaos_kernel.coupled_kernels.copy(),
            coupling_strengths=chaos_kernel.coupling_strengths.copy(),
        )

        chaos_kernel.child_kernels.append(god_kernel_id)
        chaos_kernel.lifecycle_state = "promoted"

        self.active_kernels[god_kernel_id] = god_kernel
        if god_kernel.god_name:
            self._active_gods[god_kernel.god_name] = (
                self._active_gods.get(god_kernel.god_name, 0) + 1
            )

        # TCP v6.1: chaos kernel is superseded → SHADOW; god kernel → AVAILABLE
        cr = _cr()
        if cr is not None:
            try:
                cr.mark_shadow(chaos_kernel.name.lower())
                cr.mark_available(
                    god_name.lower(),
                    kernel_id=god_kernel_id,
                    basin=god_kernel.basin_coords,
                )
            except Exception as e:
                logger.warning(
                    "[KernelLifecycle] ConstellationRegistry sync (promote) "
                    "failed (non-fatal): %s", e
                )

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
            },
        )

        logger.info(
            "[KernelLifecycle] Promoted %s → god %s (φ=%.3f, cycles=%d)",
            chaos_kernel.name, god_name, chaos_kernel.phi, chaos_kernel.total_cycles,
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
        import uuid
        event = LifecycleEventRecord(
            event_id=f"event_{uuid.uuid4().hex[:8]}",
            event_type=event_type,
            timestamp=datetime.now(),
            primary_kernel_id=primary_kernel_id,
            secondary_kernel_ids=secondary_kernel_ids or [],
            reason=reason,
            metadata=metadata or {},
            success=True,
        )
        self.event_log.append(event)
        if len(self.event_log) > 1000:
            self.event_log = self.event_log[-500:]
        return event

    # =========================================================================
    # QUERY & STATS
    # =========================================================================

    def get_kernel(self, kernel_id: str) -> Optional[Kernel]:
        return self.active_kernels.get(kernel_id)

    def get_shadow(self, shadow_id: str) -> Optional[ShadowKernel]:
        return self.shadow_pantheon.get(shadow_id)

    def list_active_kernels(self) -> List[Kernel]:
        return list(self.active_kernels.values())

    def list_shadow_kernels(self) -> List[ShadowKernel]:
        return list(self.shadow_pantheon.values())

    def get_lifecycle_stats(self) -> Dict[str, Any]:
        event_counts = {e.value: 0 for e in LifecycleEvent}
        for event in self.event_log:
            event_counts[event.event_type.value] += 1

        cr = _cr()
        stage = cr.stage.value if cr is not None else "unknown"

        return {
            'active_kernels': len(self.active_kernels),
            'shadow_kernels': len(self.shadow_pantheon),
            'total_events': len(self.event_log),
            'event_counts': event_counts,
            'god_count': sum(
                1 for k in self.active_kernels.values()
                if k.kernel_kind == KernelKind.GOD
            ),
            'chaos_count': sum(
                1 for k in self.active_kernels.values()
                if k.kernel_kind == KernelKind.CHAOS
            ),
            'protected_count': sum(
                1 for k in self.active_kernels.values()
                if k.lifecycle_state == "protected"
            ),
            'constellation_stage': stage,
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
