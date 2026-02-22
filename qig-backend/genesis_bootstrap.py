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
    bootstrap(target_stage="image")
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports — all fail-soft except KernelLifecycleManager (required)
# ---------------------------------------------------------------------------

try:
    from kernel_lifecycle import (
        KernelLifecycleManager,
        KernelKind,
        Kernel,
        get_lifecycle_manager,
    )
    LIFECYCLE_AVAILABLE = True
except ImportError as e:
    logger.error("[Bootstrap] KernelLifecycleManager not available: %s", e)
    LIFECYCLE_AVAILABLE = False
    get_lifecycle_manager = None

try:
    from kernel_spawner import RoleSpec
    SPAWNER_AVAILABLE = True
except ImportError:
    SPAWNER_AVAILABLE = False
    RoleSpec = None

try:
    from qig_constellation_registry import (
        ConstellationRegistry,
        ConstellationStage,
        get_constellation_registry,
        GENESIS_KERNEL_NAME,
        CORE_8_NAMES,
        OLYMPIAN_NAMES,
    )
    REGISTRY_AVAILABLE = True
except ImportError as e:
    logger.warning("[Bootstrap] ConstellationRegistry not available: %s", e)
    REGISTRY_AVAILABLE = False
    ConstellationRegistry = None
    ConstellationStage = None
    get_constellation_registry = None
    GENESIS_KERNEL_NAME = "genesis"
    CORE_8_NAMES = (
        "heart", "perception", "memory", "strategy",
        "action", "ethics", "meta", "ocean",
    )
    OLYMPIAN_NAMES = (
        "zeus", "athena", "apollo", "ares", "hermes", "hephaestus",
        "artemis", "dionysus", "demeter", "poseidon", "hera", "aphrodite",
    )

try:
    from qig_service_v61_extensions import get_generative_service_v61, apply_v61_extensions
    SERVICE_EXTENSIONS_AVAILABLE = True
except ImportError:
    SERVICE_EXTENSIONS_AVAILABLE = False
    get_generative_service_v61 = None

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


# Each core faculty: (name, domains, capabilities)
CORE_8_ROLE_MAP = {
    "heart":       (["rhythm", "timing", "coherence"],         ["hrv_tacking", "kappa_timing"]),
    "perception":  (["perception", "input", "sensing"],        ["signal_intake", "pattern_recognition"]),
    "memory":      (["memory", "basin_consolidation"],         ["basin_storage", "retrieval"]),
    "strategy":    (["strategy", "planning", "foresight"],     ["trajectory_foresight", "prediction"]),
    "action":      (["action", "output", "execution"],         ["response_generation", "task_execution"]),
    "ethics":      (["ethics", "constraints", "values"],       ["constraint_enforcement", "harm_detection"]),
    "meta":        (["meta", "self_observation"],              ["m_metric", "self_monitoring"]),
    "ocean":       (["autonomic", "monitoring", "coherence"],  ["phi_coherence", "breakdown_detection"]),
}

OLYMPIAN_ROLE_MAP = {
    "zeus":       (["coordination", "governance"],        ["orchestration", "routing"]),
    "athena":     (["wisdom", "strategy"],                ["analytical_reasoning", "synthesis"]),
    "apollo":     (["prophecy", "healing", "arts"],       ["foresight", "generation"]),
    "ares":       (["conflict", "resolution"],            ["adversarial_reasoning"]),
    "hermes":     (["communication", "translation"],      ["message_routing", "coordizing"]),
    "hephaestus": (["crafting", "tools", "construction"], ["code_generation", "tool_use"]),
    "artemis":    (["precision", "hunting", "wilderness"],["precision_retrieval", "search"]),
    "dionysus":   (["creativity", "chaos", "arts"],       ["creative_generation"]),
    "demeter":    (["nurturing", "growth"],               ["learning", "curriculum"]),
    "poseidon":   (["depth", "change"],                   ["exploration", "perturbation"]),
    "hera":       (["governance", "structure"],           ["lifecycle_governance"]),
    "aphrodite":  (["harmony", "beauty"],                 ["aesthetic_evaluation", "coherence"]),
}


# ---------------------------------------------------------------------------
# BOOTSTRAP CONTEXT
# ---------------------------------------------------------------------------

@dataclass
class BootstrapContext:
    """Result of a bootstrap run — snapshots of what was created."""
    stage: object  # ConstellationStage or str
    genesis_kernel: Optional[Kernel] = None
    core_8_kernels: List[Kernel] = field(default_factory=list)
    olympian_kernels: List[Kernel] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def live_count(self) -> int:
        return (
            (1 if self.genesis_kernel else 0)
            + len(self.core_8_kernels)
            + len(self.olympian_kernels)
        )

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# SPAWN HELPERS
# ---------------------------------------------------------------------------

def _make_genesis_basin() -> np.ndarray:
    """Generate the Genesis kernel basin (maximum entropy — equal probability mass)."""
    return fisher_normalize(np.ones(BASIN_DIM))


def _make_named_basin(name: str) -> np.ndarray:
    """Generate a deterministic basin for a named kernel (seeded by name hash)."""
    rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
    return fisher_normalize(rng.dirichlet(np.ones(BASIN_DIM)))


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
    For GOD/CHAOS: routes through KernelLifecycleManager.spawn() with a RoleSpec.

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

            kernel = manager.spawn(role, initial_basin=target_basin)
            return kernel

    except Exception as e:
        msg = f"[Bootstrap] Failed to spawn '{name}': {e}"
        logger.error(msg, exc_info=True)
        if ctx:
            ctx.errors.append(msg)
        return None


# ---------------------------------------------------------------------------
# STAGE SPAWNERS
# ---------------------------------------------------------------------------

def _stage_genesis(manager: KernelLifecycleManager, ctx: BootstrapContext) -> None:
    """Bootstrap Stage 1 — spawn the Genesis kernel."""
    logger.info("[Bootstrap] Stage 1: Genesis")
    kernel = _spawn_kernel(
        manager,
        name=GENESIS_KERNEL_NAME,
        domains=["primordial", "all"],
        capabilities=["everything"],
        kind="genesis",
        basin=_make_genesis_basin(),
        ctx=ctx,
    )
    if kernel:
        ctx.genesis_kernel = kernel
    else:
        ctx.errors.append("[Bootstrap] FATAL: Genesis kernel spawn failed")


def _stage_core_8(manager: KernelLifecycleManager, ctx: BootstrapContext) -> None:
    """Bootstrap Stage 2 — spawn the 8 core faculties."""
    logger.info("[Bootstrap] Stage 2: Core 8 Faculties")
    for name in CORE_8_NAMES:
        domains, capabilities = CORE_8_ROLE_MAP.get(name, ([name], [name]))
        kernel = _spawn_kernel(
            manager,
            name=name,
            domains=list(domains),
            capabilities=list(capabilities),
            kind="god",
            ctx=ctx,
        )
        if kernel:
            ctx.core_8_kernels.append(kernel)
            logger.info("[Bootstrap]   + %s (id=%s)", name, kernel.kernel_id)
        else:
            msg = f"[Bootstrap] Core 8 incomplete: '{name}' spawn failed"
            logger.warning(msg)
            ctx.warnings.append(msg)

    live = len(ctx.core_8_kernels)
    logger.info("[Bootstrap] Core 8 complete: %d/8 kernels live", live)
    if live < 8:
        ctx.warnings.append(
            f"[Bootstrap] Core 8 partially spawned: {live}/8 live "
            f"(missing: {set(CORE_8_NAMES) - {k.name for k in ctx.core_8_kernels}})"
        )


def _stage_image(manager: KernelLifecycleManager, ctx: BootstrapContext) -> None:
    """Bootstrap Stage 3 (Image) — spawn the Olympian gods."""
    logger.info("[Bootstrap] Stage 3: Image (Olympians)")
    for name in OLYMPIAN_NAMES:
        domains, capabilities = OLYMPIAN_ROLE_MAP.get(name, ([name], [name]))
        kernel = _spawn_kernel(
            manager,
            name=name,
            domains=list(domains),
            capabilities=list(capabilities),
            kind="god",
            ctx=ctx,
        )
        if kernel:
            ctx.olympian_kernels.append(kernel)
            logger.info("[Bootstrap]   + %s (id=%s)", name, kernel.kernel_id)
        else:
            msg = f"[Bootstrap] Olympian '{name}' spawn failed — skipping"
            logger.warning(msg)
            ctx.warnings.append(msg)

    logger.info(
        "[Bootstrap] Image stage complete: %d/%d Olympians live",
        len(ctx.olympian_kernels), len(OLYMPIAN_NAMES),
    )


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

# Valid target stage strings
VALID_STAGES = ("genesis", "core_8", "image")


def bootstrap(
    target_stage: str = "core_8",
    manager: Optional[KernelLifecycleManager] = None,
) -> BootstrapContext:
    """
    Run the canonical Genesis bootstrap sequence (synchronous).

    Always runs all stages up to target_stage in order:
      genesis → core_8 → image

    The generative service v6.1 extension is applied before returning
    so routing is constellation-aware from the first request.

    Args:
        target_stage: "genesis" | "core_8" | "image"
                      (default: "core_8" — minimum for production)
        manager: Optional KernelLifecycleManager (uses global singleton if None)

    Returns:
        BootstrapContext with spawned kernels, stage, errors, and warnings

    Raises:
        ValueError: If target_stage is invalid
        RuntimeError: If Genesis kernel spawn fails (fatal)
    """
    if target_stage not in VALID_STAGES:
        raise ValueError(
            f"Invalid target_stage '{target_stage}'. "
            f"Must be one of: {VALID_STAGES}"
        )

    t0 = time.monotonic()
    ctx = BootstrapContext(stage=target_stage)

    if not LIFECYCLE_AVAILABLE:
        ctx.errors.append("KernelLifecycleManager not available — bootstrap aborted")
        return ctx

    mgr = manager or (get_lifecycle_manager() if get_lifecycle_manager else None)
    if mgr is None:
        ctx.errors.append("Cannot get KernelLifecycleManager — bootstrap aborted")
        return ctx

    # Apply v6.1 service extension early so routing uses registry from the start
    if SERVICE_EXTENSIONS_AVAILABLE and get_generative_service_v61:
        try:
            get_generative_service_v61()
            logger.info("[Bootstrap] v6.1 service extensions applied")
        except Exception as e:
            ctx.warnings.append(f"[Bootstrap] v6.1 extension failed (non-fatal): {e}")

    # Stage 1: Genesis (always required)
    _stage_genesis(mgr, ctx)
    if ctx.errors:
        ctx.elapsed_ms = (time.monotonic() - t0) * 1000
        raise RuntimeError(
            f"Bootstrap aborted: Genesis kernel spawn failed. Errors: {ctx.errors}"
        )

    if target_stage == "genesis":
        ctx.elapsed_ms = (time.monotonic() - t0) * 1000
        _log_summary(ctx)
        return ctx

    # Stage 2: Core 8
    _stage_core_8(mgr, ctx)

    if target_stage == "core_8":
        ctx.elapsed_ms = (time.monotonic() - t0) * 1000
        _log_summary(ctx)
        return ctx

    # Stage 3: Image (Olympians)
    _stage_image(mgr, ctx)

    ctx.elapsed_ms = (time.monotonic() - t0) * 1000
    _log_summary(ctx)
    return ctx


async def bootstrap_async(
    target_stage: str = "core_8",
    manager: Optional[KernelLifecycleManager] = None,
) -> BootstrapContext:
    """
    Async wrapper for bootstrap() — safe to call from FastAPI lifespan.

    Runs bootstrap in a thread pool to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()
    ctx = await loop.run_in_executor(
        None,
        lambda: bootstrap(target_stage=target_stage, manager=manager),
    )
    return ctx


def _log_summary(ctx: BootstrapContext) -> None:
    """Log a human-readable bootstrap summary."""
    live = ctx.live_count
    status = "OK" if ctx.ok else f"PARTIAL ({len(ctx.errors)} errors)"

    # Infer actual stage from what spawned
    if ctx.olympian_kernels:
        actual = "image"
    elif ctx.core_8_kernels:
        actual = "core_8"
    elif ctx.genesis_kernel:
        actual = "genesis"
    else:
        actual = "failed"

    logger.info(
        "[Bootstrap] Complete — stage=%s live=%d errors=%d warnings=%d "
        "elapsed=%.1fms status=%s",
        actual, live, len(ctx.errors), len(ctx.warnings),
        ctx.elapsed_ms, status,
    )
    if ctx.warnings:
        for w in ctx.warnings:
            logger.warning(w)
    if ctx.errors:
        for e in ctx.errors:
            logger.error(e)


# ---------------------------------------------------------------------------
# ROLLBACK
# ---------------------------------------------------------------------------

def rollback(manager: Optional[KernelLifecycleManager] = None) -> None:
    """
    Reset the constellation to GENESIS_ONLY state.

    Prunes all non-genesis kernels and re-runs genesis bootstrap.
    Used for system recovery (Genesis Doctrine: genesis-driven rollback is canonical).

    Args:
        manager: Optional KernelLifecycleManager (uses global singleton if None)
    """
    if not LIFECYCLE_AVAILABLE:
        logger.error("[Bootstrap] rollback() aborted: KernelLifecycleManager unavailable")
        return

    mgr = manager or (get_lifecycle_manager() if get_lifecycle_manager else None)
    if mgr is None:
        logger.error("[Bootstrap] rollback() aborted: cannot get manager")
        return

    logger.warning("[Bootstrap] Rolling back constellation to GENESIS_ONLY")

    # Prune all non-genesis kernels
    to_prune = [
        k for k in list(mgr.active_kernels.values())
        if k.kernel_kind != KernelKind.GENESIS
        and k.lifecycle_state not in ("pruned", "promoted", "merged", "split")
    ]
    for kernel in to_prune:
        try:
            # Temporarily clear protection so prune can proceed
            kernel.lifecycle_state = "active"
            mgr.prune(kernel, reason="rollback_to_genesis")
        except Exception as e:
            logger.warning("[Bootstrap] rollback prune failed for %s: %s", kernel.name, e)

    # Wipe genesis if present so it can be respawned cleanly
    genesis_kernels = [
        k for k in list(mgr.active_kernels.values())
        if k.kernel_kind == KernelKind.GENESIS
    ]
    for gk in genesis_kernels:
        mgr.active_kernels.pop(gk.kernel_id, None)

    # Respawn genesis
    ctx = BootstrapContext(stage="genesis")
    _stage_genesis(mgr, ctx)
    if ctx.errors:
        logger.error("[Bootstrap] rollback genesis respawn failed: %s", ctx.errors)
    else:
        logger.info("[Bootstrap] Rollback complete — constellation is at GENESIS_ONLY")


# ---------------------------------------------------------------------------
# CONVENIENCE CHECKER
# ---------------------------------------------------------------------------

def get_stage() -> str:
    """
    Get the current ConstellationStage as a string.

    Returns "unknown" if ConstellationRegistry is not available.
    """
    if not REGISTRY_AVAILABLE or get_constellation_registry is None:
        return "unknown"
    try:
        return get_constellation_registry().stage.value
    except Exception:
        return "unknown"


def is_ready(minimum_stage: str = "core_8") -> bool:
    """
    Return True if the constellation has reached at least minimum_stage.

    Args:
        minimum_stage: "genesis" | "core_8" | "image" | "growing" | "full"

    Returns:
        bool
    """
    order = ["genesis_only", "core_8", "image", "growing", "full"]
    # Map user-friendly names to registry names
    stage_map = {
        "genesis":      "genesis_only",
        "genesis_only": "genesis_only",
        "core_8":       "core_8",
        "image":        "image",
        "growing":      "growing",
        "full":         "full",
    }
    current = get_stage()
    target = stage_map.get(minimum_stage, minimum_stage)
    try:
        return order.index(current) >= order.index(target)
    except ValueError:
        return False
