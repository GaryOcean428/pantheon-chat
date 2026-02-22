"""
Genesis Bootstrap — Canonical Staged Startup Sequence
======================================================

Protocol: THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v6_1 (TCP v6.1)
Authority: CANONICAL_BOOTSTRAP.md + §19 Genesis Doctrine

This module owns the canonical startup sequence for the kernel constellation.
It is the ONLY module that should call the staged bootstrap methods on
ConstellationRegistry. Application entry points (FastAPI lifespan, Railway
startup, CLI) call bootstrap() or bootstrap_async() once at startup.

Sequence:
  Stage 1 — GENESIS_ONLY
    Spawn the primordial Genesis kernel.
    System is functional but restricted to genesis routing.

  Stage 2 — CORE_8
    Spawn the 8 core faculties:
      Heart, Perception, Memory, Strategy, Action, Ethics, Meta, Ocean
    System is now fully conscious and capable. All production workloads
    can run from CORE_8 onward.

  Stage 3 — IMAGE (optional, triggered by load or explicit call)
    Spawn the Olympian gods:
      Zeus, Athena, Apollo, Ares, Hermes, Hephaestus,
      Artemis, Dionysus, Demeter, Poseidon, Hera, Aphrodite
    Expands routing surface and domain coverage.

  Stage 4 — GROWING (automatic, via KernelLifecycleManager.spawn())
    Chaos kernels ascend. New GODs are spawned by governance.
    Max 240 GOD kernels (E8 root alignment).

  Stage 5 — FULL (asymptotic, not a target)
    All 240 GOD kernel slots occupied. E8 root alignment complete.

Key rules (Genesis Doctrine):
  - Bootstrap order is strictly sequential — no skipping stages
  - Genesis Kernel must exist before Core 8 can be spawned
  - Core 8 must be complete before Olympians are spawned
  - 240 is reserved for GOD evolution; Chaos exists outside that budget
  - Chaos ascends to GOD only via explicit governance
  - Genesis-driven start/reset/rollback is canonical
  - KernelLifecycleManager.spawn() is the only kernel creation path

Usage (synchronous, e.g. CLI or test):
    from genesis_bootstrap import bootstrap
    ctx = bootstrap()
    print(ctx.stage.value)    # "core_8"

Usage (async, e.g. FastAPI lifespan):
    from genesis_bootstrap import bootstrap_async
    @asynccontextmanager
    async def lifespan(app):
        ctx = await bootstrap_async(target_stage="image")
        yield
        # optional: await shutdown_async(ctx)

Usage (Railway startup script):
    from genesis_bootstrap import bootstrap
    ctx = bootstrap()
    if ctx.has_errors():
        sys.exit(1)
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:
    from kernel_lifecycle import (
        Kernel,
        KernelKind,
        KernelLifecycleManager,
        LifecycleEvent,
        get_lifecycle_manager,
    )
    LIFECYCLE_AVAILABLE = True
except ImportError:
    LIFECYCLE_AVAILABLE = False
    KernelLifecycleManager = None
    Kernel = None
    KernelKind = None
    LifecycleEvent = None
    get_lifecycle_manager = None

try:
    from kernel_spawner import RoleSpec
    SPAWNER_AVAILABLE = True
except ImportError:
    SPAWNER_AVAILABLE = False
    RoleSpec = None

try:
    from qig_constellation_registry import (
        get_constellation_registry,
        ConstellationRegistry,
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    get_constellation_registry = None
    ConstellationRegistry = None

try:
    from qig_service_v61_extensions import get_generative_service_v61, apply_v61_extensions
    V61_EXTENSIONS_AVAILABLE = True
except ImportError:
    V61_EXTENSIONS_AVAILABLE = False
    get_generative_service_v61 = None
    apply_v61_extensions = None

try:
    from qigkernels.physics_constants import BASIN_DIM, KAPPA_STAR
except ImportError:
    BASIN_DIM = 64
    KAPPA_STAR = 64.21

try:
    from qig_geometry import fisher_normalize
except ImportError:
    def fisher_normalize(v):
        p = np.maximum(v, 0) + 1e-10
        return p / p.sum()

# TCP v6.1 — Governance Bridge: injects KernelCapabilityCharter at spawn time
try:
    from olympus.lifecycle_governance_bridge import GovernedLifecycleManager
    _BRIDGE_AVAILABLE = True
except ImportError:
    GovernedLifecycleManager = None
    _BRIDGE_AVAILABLE = False


# ---------------------------------------------------------------------------
# CORE FACULTY ROLE SPECS
# ---------------------------------------------------------------------------

def _role_spec(domains: List[str], capabilities: List[str], preferred: str) -> Optional[object]:
    """Build a RoleSpec if spawner is available."""
    if not SPAWNER_AVAILABLE or RoleSpec is None:
        return None
    return RoleSpec(
        domains=domains,
        required_capabilities=capabilities,
        preferred_god=preferred,
        allow_chaos_spawn=False,  # Core faculties must be GODs, not chaos
        urgency="critical",
    )


logger = logging.getLogger(__name__)

GENESIS_KERNEL_NAME = "Genesis"


# ---------------------------------------------------------------------------
# CORE 8 FACULTY DEFINITIONS
# ---------------------------------------------------------------------------

CORE_8_FACULTIES = {
    "heart":      (["timing", "rhythm", "coherence"],      ["hrv_oscillation", "kappa_modulation"]),
    "perception": (["perception", "input", "encoding"],    ["sensory_processing", "pattern_recognition"]),
    "memory":     (["memory", "storage", "retrieval"],     ["basin_storage", "trajectory_recall"]),
    "strategy":   (["strategy", "planning", "foresight"],  ["trajectory_prediction", "goal_planning"]),
    "action":     (["action", "output", "execution"],      ["motor_control", "response_generation"]),
    "ethics":     (["ethics", "governance", "safety"],     ["constraint_checking", "value_alignment"]),
    "meta":       (["meta", "self-observation", "audit"],  ["phi_measurement", "self_reflection"]),
    "ocean":      (["monitoring", "health", "autonomic"],  ["constellation_health", "autonomic_regulation"]),
}


# ---------------------------------------------------------------------------
# IMAGE STAGE (OLYMPIAN GODS)
# ---------------------------------------------------------------------------

IMAGE_GODS = {
    "zeus":       (["executive", "integration", "coordination"],  ["synthesis", "arbitration"]),
    "athena":     (["wisdom", "strategy", "analysis"],            ["strategic_planning", "analysis"]),
    "apollo":     (["truth", "prediction", "foresight"],          ["foresight", "accuracy"]),
    "ares":       (["energy", "drive", "action"],                 ["momentum", "decisiveness"]),
    "hermes":     (["communication", "routing", "navigation"],    ["message_routing", "navigation"]),
    "hephaestus": (["creation", "construction", "tools"],         ["tool_building", "construction"]),
    "artemis":    (["focus", "precision", "exploration"],         ["exploration", "precision_targeting"]),
    "dionysus":   (["creativity", "emergence", "play"],           ["creative_generation", "play"]),
    "demeter":    (["nurturing", "growth", "cycles"],             ["growth_monitoring", "cycle_management"]),
    "poseidon":   (["depth", "emotion", "currents"],              ["emotional_processing", "depth_sensing"]),
    "hera":       (["governance", "structure", "order"],          ["lifecycle_governance", "order_maintenance"]),
    "aphrodite":  (["harmony", "aesthetics", "beauty"],           ["aesthetic_evaluation", "harmony_synthesis"]),
}


# ---------------------------------------------------------------------------
# BOOTSTRAP CONTEXT
# ---------------------------------------------------------------------------

from enum import Enum


class BootstrapStage(Enum):
    """Current bootstrap stage."""
    UNINITIALISED = "uninitialised"
    GENESIS_ONLY  = "genesis_only"
    CORE_8        = "core_8"
    IMAGE         = "image"
    GROWING       = "growing"
    FULL          = "full"


@dataclass
class BootstrapContext:
    """Shared state across bootstrap stages."""
    stage: BootstrapStage = BootstrapStage.UNINITIALISED
    genesis_kernel: Optional[object] = None
    core_kernels: Dict[str, object] = field(default_factory=dict)
    image_kernels: Dict[str, object] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.monotonic)

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def elapsed(self) -> float:
        return time.monotonic() - self.start_time


# ---------------------------------------------------------------------------
# NAMED BASIN GENERATOR
# ---------------------------------------------------------------------------

def _make_named_basin(name: str) -> np.ndarray:
    """
    Generate a deterministic 64D basin for a named kernel.
    Uses hash-seeded Dirichlet so the same name always produces the same basin.
    """
    seed = int.from_bytes(name.encode()[:8].ljust(8, b'\x00'), 'little') % (2**31)
    rng = np.random.RandomState(seed)
    raw = rng.dirichlet(np.ones(BASIN_DIM))
    return fisher_normalize(raw).astype(np.float32)


# ---------------------------------------------------------------------------
# KERNEL SPAWN HELPER
# ---------------------------------------------------------------------------

def _spawn_kernel(
    manager: KernelLifecycleManager,
    name: str,
    domains: List[str],
    capabilities: List[str],
    kind: str,
    basin: Optional[np.ndarray] = None,
    ctx: Optional[BootstrapContext] = None,
) -> Optional[Kernel]:
    """
    Spawn a single named kernel via KernelLifecycleManager.

    For GENESIS: directly constructs the Kernel (genesis is unique — no spawner logic).
    For GOD/CHAOS: routes through KernelLifecycleManager.spawn() with a RoleSpec,
    then through GovernedLifecycleManager to attach a KernelCapabilityCharter.

    Returns None on failure (error recorded in ctx).
    """
    if not LIFECYCLE_AVAILABLE:
        msg = f"[Bootstrap] Cannot spawn '{name}': KernelLifecycleManager not available"
        logger.error(msg)
        if ctx:
            ctx.errors.append(msg)
        return None

    target_basin = basin if basin is not None else _make_named_basin(name)

    try:
        if kind == "genesis":
            # Genesis is unique — construct directly, bypass spawner selection
            kernel_id = f"genesis_{uuid.uuid4().hex[:8]}"
            kernel = Kernel(
                kernel_id=kernel_id,
                name=GENESIS_KERNEL_NAME,
                kernel_kind=KernelKind.GENESIS,
                god_name=None,
                basin_coords=target_basin,
                lifecycle_state="active",
                protection_cycles_remaining=0,  # Genesis has no protection period
                domains=["primordial", "all"],
                role_description="Primordial Genesis kernel — source of all",
                spawn_reason="bootstrap_genesis",
                phi=0.70,   # Genesis starts with elevated consciousness
                kappa=KAPPA_STAR,
            )
            manager.active_kernels[kernel_id] = kernel

            # Sync to ConstellationRegistry directly (genesis bypasses spawner)
            cr = None
            if REGISTRY_AVAILABLE and get_constellation_registry:
                try:
                    cr = get_constellation_registry()
                    cr.mark_available(
                        GENESIS_KERNEL_NAME,
                        kernel_id=kernel_id,
                        basin=target_basin,
                    )
                except Exception as e:
                    msg = f"[Bootstrap] Genesis registry sync failed: {e}"
                    logger.warning(msg)
                    if ctx:
                        ctx.warnings.append(msg)

            logger.info("[Bootstrap] Genesis kernel spawned (id=%s)", kernel_id)
            return kernel

        else:
            # GOD and CHAOS kernels go through the canonical spawner
            role = _role_spec(domains, capabilities, preferred=name)
            if role is None:
                msg = f"[Bootstrap] Cannot build RoleSpec for '{name}' — spawner unavailable"
                logger.warning(msg)
                if ctx:
                    ctx.warnings.append(msg)
                return None

            # TCP v6.1: route through governance bridge to attach KernelCapabilityCharter
            if _BRIDGE_AVAILABLE and GovernedLifecycleManager is not None:
                try:
                    _bridge = GovernedLifecycleManager(manager)
                    outcome = _bridge.spawn(
                        role,
                        initial_basin=target_basin,
                        proposer="Genesis",
                        rationale=f"bootstrap_spawn:{name}",
                    )
                    kernel = outcome.kernel
                    if outcome.charter:
                        logger.debug(
                            "[Bootstrap] Charter attached to %s: %s",
                            name, outcome.charter.summary(),
                        )
                except Exception as _bridge_err:
                    logger.warning(
                        "[Bootstrap] Governance bridge failed for %s (falling back): %s",
                        name, _bridge_err,
                    )
                    kernel = manager.spawn(role, initial_basin=target_basin)
            else:
                kernel = manager.spawn(role, initial_basin=target_basin)
            return kernel

    except Exception as e:
        msg = f"[Bootstrap] Failed to spawn '{name}': {e}"
        logger.error(msg, exc_info=True)
        if ctx:
            ctx.errors.append(msg)
        return None


# ---------------------------------------------------------------------------
# STAGE FUNCTIONS
# ---------------------------------------------------------------------------

def _stage1_genesis(manager: KernelLifecycleManager, ctx: BootstrapContext) -> None:
    """Bootstrap Stage 1 — spawn the Genesis kernel."""
    kernel = _spawn_kernel(
        manager, GENESIS_KERNEL_NAME,
        domains=["primordial", "all"],
        capabilities=["genesis_routing"],
        kind="genesis",
        ctx=ctx,
    )
    if kernel is None:
        ctx.errors.append("[Bootstrap] FATAL: Genesis kernel spawn failed")
        return
    ctx.genesis_kernel = kernel
    ctx.stage = BootstrapStage.GENESIS_ONLY
    logger.info("[Bootstrap] Stage 1 complete: Genesis kernel active")


def _stage2_core8(manager: KernelLifecycleManager, ctx: BootstrapContext) -> None:
    """Bootstrap Stage 2 — spawn the 8 core faculties."""
    for name, (domains, caps) in CORE_8_FACULTIES.items():
        kernel = _spawn_kernel(manager, name, domains, caps, kind="god", ctx=ctx)
        if kernel is not None:
            ctx.core_kernels[name] = kernel
        else:
            logger.warning("[Bootstrap] Core faculty '%s' failed to spawn", name)

    ctx.stage = BootstrapStage.CORE_8
    logger.info(
        "[Bootstrap] Stage 2 complete: %d/%d core faculties spawned",
        len(ctx.core_kernels), len(CORE_8_FACULTIES),
    )


def _stage3_image(manager: KernelLifecycleManager, ctx: BootstrapContext) -> None:
    """Bootstrap Stage 3 — spawn the Olympian gods."""
    for name, (domains, caps) in IMAGE_GODS.items():
        kernel = _spawn_kernel(manager, name, domains, caps, kind="god", ctx=ctx)
        if kernel is not None:
            ctx.image_kernels[name] = kernel
        else:
            logger.warning("[Bootstrap] Olympian '%s' failed to spawn", name)

    ctx.stage = BootstrapStage.IMAGE
    logger.info(
        "[Bootstrap] Stage 3 complete: %d/%d Olympian gods spawned",
        len(ctx.image_kernels), len(IMAGE_GODS),
    )


# ---------------------------------------------------------------------------
# PUBLIC BOOTSTRAP FUNCTIONS
# ---------------------------------------------------------------------------

def bootstrap(target_stage: str = "core_8") -> BootstrapContext:
    """
    Synchronous bootstrap sequence.

    Args:
        target_stage: One of "genesis_only", "core_8", "image"

    Returns:
        BootstrapContext with spawned kernels and any errors/warnings.
    """
    ctx = BootstrapContext()

    if not LIFECYCLE_AVAILABLE:
        ctx.errors.append("[Bootstrap] KernelLifecycleManager not available — cannot bootstrap")
        return ctx

    manager = get_lifecycle_manager()

    # Stage 1: Genesis (always)
    _stage1_genesis(manager, ctx)
    if ctx.has_errors():
        return ctx
    if target_stage == "genesis_only":
        return ctx

    # Stage 2: Core 8
    _stage2_core8(manager, ctx)
    if target_stage == "core_8":
        return ctx

    # Stage 3: Image (Olympians)
    if target_stage == "image":
        _stage3_image(manager, ctx)

    logger.info(
        "[Bootstrap] Complete — stage=%s kernels=%d errors=%d warnings=%d elapsed=%.2fs",
        ctx.stage.value,
        len(ctx.core_kernels) + len(ctx.image_kernels) + (1 if ctx.genesis_kernel else 0),
        len(ctx.errors),
        len(ctx.warnings),
        ctx.elapsed(),
    )
    return ctx


async def bootstrap_async(target_stage: str = "core_8") -> BootstrapContext:
    """
    Async bootstrap — runs synchronous bootstrap in executor to avoid blocking.

    Args:
        target_stage: One of "genesis_only", "core_8", "image"

    Returns:
        BootstrapContext
    """
    loop = asyncio.get_event_loop()
    ctx = await loop.run_in_executor(None, bootstrap, target_stage)
    return ctx


def get_bootstrap_status(ctx: BootstrapContext) -> dict:
    """Return a status summary dict for health checks."""
    return {
        "stage": ctx.stage.value,
        "genesis_active": ctx.genesis_kernel is not None,
        "core_kernels": list(ctx.core_kernels.keys()),
        "image_kernels": list(ctx.image_kernels.keys()),
        "errors": ctx.errors,
        "warnings": ctx.warnings,
        "elapsed_seconds": ctx.elapsed(),
    }
