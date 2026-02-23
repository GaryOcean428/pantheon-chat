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

# TCP v6.1 — VoterRegistry: live φ/κ for governance vote weighting
_VOTER_REGISTRY = None
_VOTER_REGISTRY_ATTEMPTED = False

def _vr():
    global _VOTER_REGISTRY, _VOTER_REGISTRY_ATTEMPTED
    if _VOTER_REGISTRY is not None:
        return _VOTER_REGISTRY
    if _VOTER_REGISTRY_ATTEMPTED:
        return None
    _VOTER_REGISTRY_ATTEMPTED = True
    try:
        from olympus.voter_registry import get_voter_registry
        _VOTER_REGISTRY = get_voter_registry()
    except ImportError:
        pass
    return _VOTER_REGISTRY

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
    Compute the canonical registry name for a kernel.

    For GOD kernels: god_name (e.g. 'zeus', 'heart')
    For CHAOS kernels: chaos kernel name
    For GENESIS: 'genesis'
    """
    if kernel.kernel_kind == KernelKind.GOD and kernel.god_name:
        return kernel.god_name.lower()
    return kernel.name.lower()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class KernelKind(Enum):
    """Kind classification for kernel instances."""
    GENESIS = "genesis"
    GOD     = "god"
    CHAOS   = "chaos"
    SHADOW  = "shadow"


@dataclass
class Kernel:
    """
    A kernel instance in the constellation.

    All fields are mutable (not frozen) — lifecycle manager updates them
    as the kernel progresses through stages.
    """
    kernel_id: str
    name: str
    kernel_kind: KernelKind = KernelKind.CHAOS
    god_name: Optional[str] = None
    epithet: Optional[str] = None
    basin_coords: Optional[np.ndarray] = None
    lifecycle_state: str = "protected"
    protection_cycles_remaining: int = 50
    phi: float = 0.5
    kappa: float = 64.0
    gamma: float = 1.0  # Generation capability
    domains: List[str] = field(default_factory=list)
    role_description: str = ""
    spawn_reason: str = ""
    spawn_timestamp: Optional[datetime] = None
    mentor_kernel_id: Optional[str] = None
    parent_kernels: List[str] = field(default_factory=list)
    child_kernels: List[str] = field(default_factory=list)
    coupled_kernels: List[str] = field(default_factory=list)
    coupling_strengths: Dict[str, float] = field(default_factory=dict)
    ascended_from: Optional[str] = None
    success_count: int = 0
    failure_count: int = 0
    total_cycles: int = 0
    # TCP v6.1 — optional governance charter (attached by GovernedLifecycleManager)
    capability_charter: Optional[Any] = None

    def to_dict(self) -> dict:
        return {
            'kernel_id': self.kernel_id,
            'name': self.name,
            'kernel_kind': self.kernel_kind.value,
            'god_name': self.god_name,
            'lifecycle_state': self.lifecycle_state,
            'phi': self.phi,
            'kappa': self.kappa,
            'domains': self.domains,
        }

    def decrement_protection(self) -> None:
        if self.protection_cycles_remaining > 0:
            self.protection_cycles_remaining -= 1
        if self.protection_cycles_remaining == 0 and self.lifecycle_state == "protected":
            self.lifecycle_state = "active"


@dataclass
class LifecycleEvent:
    """Record of a lifecycle operation."""
    event_type: str
    primary_kernel_id: str
    secondary_kernel_id: Optional[str]
    reason: str
    timestamp: datetime
    final_phi: float
    final_kappa: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class LifecycleEvent(str, Enum):  # noqa: F811
    """Lifecycle event types."""
    SPAWN      = "spawn"
    SPLIT      = "split"
    MERGE      = "merge"
    PRUNE      = "prune"
    RESURRECT  = "resurrect"
    PROMOTE    = "promote"
    CANNIBALIZE = "cannibalize"


# ---------------------------------------------------------------------------
# KernelLifecycleManager
# ---------------------------------------------------------------------------

class KernelLifecycleManager:
    """
    Manages kernel lifecycle: spawn, split, merge, prune, resurrect, promote.

    This is the SOLE authority over kernel availability state.
    All ConstellationRegistry mark_available() calls go through here.
    """

    def __init__(
        self,
        registry: Optional[PantheonRegistry] = None,
        active_kernels: Optional[Dict[str, Kernel]] = None,
    ):
        self.registry = registry or get_registry()
        self.active_kernels = active_kernels or {}
        self.spawner = None
        self._chaos_counter: Dict[str, int] = {}
        self._active_gods: Dict[str, int] = {}
        self._event_log: List[dict] = []

        # Rebuild god count from existing kernels
        for k in self.active_kernels.values():
            if k.kernel_kind == KernelKind.GOD and k.god_name:
                self._active_gods[k.god_name] = self._active_gods.get(k.god_name, 0) + 1

        # Lazy spawner init
        try:
            from kernel_spawner import KernelSpawner
            self.spawner = KernelSpawner(registry=self.registry)
        except ImportError:
            logger.warning("[KernelLifecycle] KernelSpawner not available")

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
            # TCP v6.1: register with VoterRegistry for live φ/κ weighting
            _vr_inst = _vr()
            if _vr_inst is not None:
                try:
                    _vr_inst.register(
                        god_name=kernel.god_name.capitalize(),
                        kernel_id=kernel_id,
                        phi=kernel.phi,
                        kappa=kernel.kappa,
                    )
                except Exception as _vr_err:
                    logger.debug("[KernelLifecycle] VoterRegistry register failed: %s", _vr_err)

        # TCP v6.1: mark kernel AVAILABLE in ConstellationRegistry
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

        import uuid
        id_a = f"kernel_{uuid.uuid4().hex[:8]}"
        id_b = f"kernel_{uuid.uuid4().hex[:8]}"

        # Split domains
        domains = kernel.domains or ["general"]
        mid = max(1, len(domains) // 2)
        domains_a = domains[:mid]
        domains_b = domains[mid:] or domains

        # Split basin via geodesic perturbation (Fisher-Rao manifold preserving)
        noise_a = np.random.dirichlet(np.ones(64) * 10)
        noise_b = np.random.dirichlet(np.ones(64) * 10)
        basin_a = sqrt_map(exp_map(kernel.basin_coords, 0.1 * (noise_a - noise_a.mean())))
        basin_b = sqrt_map(exp_map(kernel.basin_coords, 0.1 * (noise_b - noise_b.mean())))

        # Normalise to simplex
        basin_a = np.abs(basin_a); basin_a /= (basin_a.sum() + 1e-12)
        basin_b = np.abs(basin_b); basin_b /= (basin_b.sum() + 1e-12)

        child_a = Kernel(
            kernel_id=id_a,
            name=f"{kernel.name}_A",
            kernel_kind=kernel.kernel_kind,
            god_name=kernel.god_name,
            basin_coords=basin_a,
            lifecycle_state="active",
            protection_cycles_remaining=0,
            phi=kernel.phi * 0.8,
            kappa=kernel.kappa,
            domains=domains_a,
            role_description=f"Split from {kernel.name}: {split_criterion} (A)",
            spawn_reason=f"split:{kernel.kernel_id}",
            spawn_timestamp=datetime.now(),
            parent_kernels=[kernel.kernel_id],
            coupled_kernels=[id_b],
            coupling_strengths={id_b: 0.8},
        )
        child_b = Kernel(
            kernel_id=id_b,
            name=f"{kernel.name}_B",
            kernel_kind=kernel.kernel_kind,
            god_name=kernel.god_name,
            basin_coords=basin_b,
            lifecycle_state="active",
            protection_cycles_remaining=0,
            phi=kernel.phi * 0.8,
            kappa=kernel.kappa,
            domains=domains_b,
            role_description=f"Split from {kernel.name}: {split_criterion} (B)",
            spawn_reason=f"split:{kernel.kernel_id}",
            spawn_timestamp=datetime.now(),
            parent_kernels=[kernel.kernel_id],
            coupled_kernels=[id_a],
            coupling_strengths={id_a: 0.8},
        )

        kernel.child_kernels.extend([id_a, id_b])
        kernel.lifecycle_state = "retired"

        self.active_kernels[id_a] = child_a
        self.active_kernels[id_b] = child_b
        self.active_kernels.pop(kernel.kernel_id, None)

        self._record_event(
            event_type=LifecycleEvent.SPLIT,
            primary_kernel_id=kernel.kernel_id,
            reason=f"split_criterion={split_criterion}",
            metadata={'child_a': id_a, 'child_b': id_b},
        )
        logger.info("[KernelLifecycle] Split %s → (%s, %s)", kernel.name, id_a, id_b)
        return child_a, child_b

    # =========================================================================
    # MERGE
    # =========================================================================

    def merge(
        self,
        primary: Kernel,
        secondary: Kernel,
    ) -> Kernel:
        """
        Merge two kernels via Fréchet mean on the Fisher-Rao manifold.

        GEOMETRIC CORRECTNESS: uses frechet_mean(), NOT arithmetic average.
        The merged kernel inherits primary's god_name if it is a GOD kernel.
        Secondary is retired from the constellation.
        """
        if primary.lifecycle_state == "protected" or secondary.lifecycle_state == "protected":
            raise ValueError("Cannot merge protected kernels")

        import uuid
        merged_id = f"kernel_{uuid.uuid4().hex[:8]}"

        # Fréchet mean (NOT linear average — preserves simplex geometry)
        merged_basin = frechet_mean([primary.basin_coords, secondary.basin_coords])
        merged_phi = (primary.phi + secondary.phi) / 2.0
        merged_kappa = (primary.kappa + secondary.kappa) / 2.0
        merged_domains = list(set(primary.domains + secondary.domains))

        merged = Kernel(
            kernel_id=merged_id,
            name=f"{primary.name}+{secondary.name}",
            kernel_kind=primary.kernel_kind,
            god_name=primary.god_name,
            basin_coords=merged_basin,
            lifecycle_state="active",
            phi=merged_phi,
            kappa=merged_kappa,
            domains=merged_domains,
            role_description=f"Merged: {primary.name} + {secondary.name}",
            spawn_reason=f"merge:{primary.kernel_id}+{secondary.kernel_id}",
            spawn_timestamp=datetime.now(),
            parent_kernels=[primary.kernel_id, secondary.kernel_id],
            coupled_kernels=list(set(primary.coupled_kernels + secondary.coupled_kernels)),
        )

        primary.lifecycle_state = "retired"
        secondary.lifecycle_state = "retired"
        self.active_kernels.pop(primary.kernel_id, None)
        self.active_kernels.pop(secondary.kernel_id, None)
        self.active_kernels[merged_id] = merged

        # TCP v6.1: merged kernel takes primary's registry slot
        cr = _cr()
        if cr is not None:
            try:
                cr.mark_available(_registry_name(merged), kernel_id=merged_id, basin=merged_basin)
                cr.mark_shadow(_registry_name(secondary))
            except Exception as e:
                logger.warning("[KernelLifecycle] ConstellationRegistry merge sync failed: %s", e)

        self._record_event(
            event_type=LifecycleEvent.MERGE,
            primary_kernel_id=primary.kernel_id,
            secondary_kernel_id=secondary.kernel_id,
            reason="merge",
            metadata={'merged_id': merged_id},
        )
        logger.info("[KernelLifecycle] Merged (%s + %s) → %s", primary.name, secondary.name, merged.name)
        return merged

    # =========================================================================
    # PRUNE
    # =========================================================================

    def prune(
        self,
        kernel: Kernel,
        reason: str = "underperforming",
    ) -> None:
        """
        Archive kernel to shadow pantheon.
        TCP v6.1: marks shadow in ConstellationRegistry.
        """
        if kernel.lifecycle_state == "protected":
            raise ValueError(f"Cannot prune protected kernel {kernel.name}")

        kernel.lifecycle_state = "shadow"
        self.active_kernels.pop(kernel.kernel_id, None)

        cr = _cr()
        if cr is not None:
            try:
                cr.mark_shadow(_registry_name(kernel))
            except Exception as e:
                logger.warning("[KernelLifecycle] ConstellationRegistry prune sync failed: %s", e)

        self._record_event(
            event_type=LifecycleEvent.PRUNE,
            primary_kernel_id=kernel.kernel_id,
            reason=reason,
            metadata={},
        )
        logger.info("[KernelLifecycle] Pruned: %s (%s)", kernel.name, reason)

    # =========================================================================
    # RESURRECT
    # =========================================================================

    def resurrect(
        self,
        kernel: Kernel,
        reason: str = "needed",
    ) -> Kernel:
        """Restore a shadow kernel to active status."""
        if kernel.lifecycle_state != "shadow":
            raise ValueError(f"Cannot resurrect non-shadow kernel: {kernel.lifecycle_state}")

        kernel.lifecycle_state = "active"
        self.active_kernels[kernel.kernel_id] = kernel

        cr = _cr()
        if cr is not None:
            try:
                cr.mark_available(_registry_name(kernel), kernel_id=kernel.kernel_id, basin=kernel.basin_coords)
            except Exception as e:
                logger.warning("[KernelLifecycle] ConstellationRegistry resurrect sync failed: %s", e)

        self._record_event(
            event_type=LifecycleEvent.RESURRECT,
            primary_kernel_id=kernel.kernel_id,
            reason=reason,
            metadata={},
        )
        logger.info("[KernelLifecycle] Resurrected: %s", kernel.name)
        return kernel

    # =========================================================================
    # PROMOTE
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

        # TCP v6.1: register promoted god with VoterRegistry
        _vr_inst = _vr()
        if _vr_inst is not None:
            try:
                _vr_inst.register(
                    god_name=god_name.capitalize(),
                    kernel_id=god_kernel_id,
                    phi=god_kernel.phi,
                    kappa=god_kernel.kappa,
                )
            except Exception as _vr_err:
                logger.debug("[KernelLifecycle] VoterRegistry register (promote) failed: %s", _vr_err)

        # TCP v6.1: chaos kernel is superseded → SHADOW; god kernel → AVAILABLE
        cr = _cr()
        if cr is not None:
            try:
                cr.mark_shadow(_registry_name(chaos_kernel))
                cr.mark_available(
                    god_name.lower(),
                    kernel_id=god_kernel_id,
                    basin=god_kernel.basin_coords,
                )
            except Exception as e:
                logger.warning(
                    "[KernelLifecycle] ConstellationRegistry promote sync failed: %s", e
                )

        self._record_event(
            event_type=LifecycleEvent.PROMOTE,
            primary_kernel_id=chaos_kernel.kernel_id,
            secondary_kernel_id=god_kernel_id,
            reason=f"chaos_ascension:{chaos_kernel.name}→{god_name}",
            metadata={'god_kernel_id': god_kernel_id, 'god_name': god_name},
        )
        logger.info(
            "[KernelLifecycle] Promoted %s → %s (id=%s)",
            chaos_kernel.name, god_name, god_kernel_id,
        )
        return god_kernel

    # =========================================================================
    # CANNIBALIZE
    # =========================================================================

    def cannibalize(
        self,
        absorber: Kernel,
        victim: Kernel,
        reason: str = "consolidation",
    ) -> Kernel:
        """
        Absorb victim's basin into absorber via Fréchet mean, then retire victim.

        More aggressive than merge: absorber retains its identity,
        victim is permanently retired (not just shadowed).
        Requires UNANIMOUS Pantheon vote (enforced by GovernedLifecycleManager).
        """
        if absorber.lifecycle_state == "protected" or victim.lifecycle_state == "protected":
            raise ValueError("Cannot cannibalize protected kernels")

        # Fisher-Rao Fréchet mean — absorber dominates (weight 0.7 vs 0.3)
        # Approximated via repeated-element trick (no Euclidean averaging)
        absorbed_basin = frechet_mean(
            [absorber.basin_coords, absorber.basin_coords, victim.basin_coords]
        )
        absorber.basin_coords = absorbed_basin
        absorber.phi = max(absorber.phi, victim.phi * 0.5)
        absorber.domains = list(set(absorber.domains + victim.domains))

        victim.lifecycle_state = "retired"
        self.active_kernels.pop(victim.kernel_id, None)

        cr = _cr()
        if cr is not None:
            try:
                cr.mark_shadow(_registry_name(victim))
                cr.mark_available(_registry_name(absorber), kernel_id=absorber.kernel_id, basin=absorbed_basin)
            except Exception as e:
                logger.warning("[KernelLifecycle] Cannibalize registry sync failed: %s", e)

        self._record_event(
            event_type=LifecycleEvent.CANNIBALIZE,
            primary_kernel_id=absorber.kernel_id,
            secondary_kernel_id=victim.kernel_id,
            reason=reason,
            metadata={},
        )
        logger.info("[KernelLifecycle] Cannibalized %s → absorbed by %s", victim.name, absorber.name)
        return absorber

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_active_gods(self) -> Dict[str, int]:
        """Return god name → active instance count."""
        return dict(self._active_gods)

    def get_chaos_kernels(self) -> List[Kernel]:
        """Return all active chaos kernels."""
        return [k for k in self.active_kernels.values() if k.kernel_kind == KernelKind.CHAOS]

    def get_god_kernels(self) -> List[Kernel]:
        """Return all active god kernels."""
        return [k for k in self.active_kernels.values() if k.kernel_kind == KernelKind.GOD]

    def update_kernel_metrics(self, kernel_id: str, phi: float, kappa: float) -> None:
        """
        Update Φ/κ for a kernel after a generation cycle.
        Also pushes to VoterRegistry if it's a GOD kernel.
        """
        kernel = self.active_kernels.get(kernel_id)
        if kernel is None:
            return
        kernel.phi = phi
        kernel.kappa = kappa
        kernel.total_cycles += 1

        if kernel.kernel_kind == KernelKind.GOD and kernel.god_name:
            _vr_inst = _vr()
            if _vr_inst is not None:
                try:
                    _vr_inst.update(kernel.god_name.capitalize(), phi, kappa)
                except Exception:
                    pass

    def _record_event(
        self,
        event_type: LifecycleEvent,
        primary_kernel_id: str,
        reason: str,
        metadata: Dict[str, Any],
        secondary_kernel_id: Optional[str] = None,
    ) -> None:
        self._event_log.append({
            'event_type': event_type.value,
            'primary_kernel_id': primary_kernel_id,
            'secondary_kernel_id': secondary_kernel_id,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata,
        })

    def get_event_log(self) -> List[dict]:
        return list(self._event_log)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_lifecycle_singleton: Optional[KernelLifecycleManager] = None


def get_lifecycle_manager() -> KernelLifecycleManager:
    global _lifecycle_singleton
    if _lifecycle_singleton is None:
        _lifecycle_singleton = KernelLifecycleManager()
    return _lifecycle_singleton
