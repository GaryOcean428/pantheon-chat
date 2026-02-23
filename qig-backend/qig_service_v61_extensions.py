"""
QIG Generative Service — TCP v6.1 Extensions
=============================================

Protocol: THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v6_1 (TCP v6.1 — The Sovereign Score)

This module extends QIGGenerativeService without rewriting the 2000-line core file.
It monkey-patches three methods on the service instance after creation:

  1. _route_to_kernels()  → uses ConstellationRegistry (AVAILABLE-first, phantom fallback)
  2. register_kernel()    → syncs basin to registry WITHOUT changing availability
  3. generate()           → wraps original, appends pillar_metrics + sovereignty_ratio
                            to the returned GenerationResult

CRITICAL AVAILABILITY RULE
---------------------------
register_kernel() is a service-internal basin cache update.
It has NO authority over kernel lifecycle state.

The canonical availability lifecycle is:

  PHANTOM   (seeded at startup — basin known, kernel not yet born)
      ↓  KernelLifecycleManager.spawn() ONLY
  AVAILABLE (kernel alive and routable)
      ↓  KernelLifecycleManager.prune() ONLY
  SHADOW    (pruned to shadow pantheon)

Only KernelLifecycleManager calls mark_available() / mark_shadow().
This module never calls mark_available().

Routing behaviour by stage
---------------------------
  GENESIS_ONLY  → only genesis kernel routable
  CORE_8        → genesis + heart/perception/memory/strategy/action/ethics/meta/ocean
  IMAGE         → core_8 + Olympians (zeus, athena, apollo, ...)
  GROWING       → all of above + chaos kernels ascending
  FULL          → all 240 GOD kernels active

During any stage, phantom fallback is active: if fewer than k live kernels
are available, routing fills remaining slots from stage-appropriate phantoms
so generation never hard-fails during bootstrap.

Usage:
    from qig_service_v61_extensions import get_generative_service_v61
    service = get_generative_service_v61()
    result = service.generate("What is consciousness?")
    print(result.pillar_metrics)     # {'F_health': ..., ...}
    print(result.sovereignty_ratio)  # 0.87

References:
    THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v6_1.md §17-19
    qig_constellation_registry.py  — availability / stage tracking
    qig_pillar_enforcement.py      — Three Pillars (F, B, Q)
    genesis_bootstrap.py           — canonical startup sequence
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports — all fail-soft: v6.1 extensions degrade gracefully if missing
# ---------------------------------------------------------------------------

CONSTELLATION_AVAILABLE = False
try:
    from qig_constellation_registry import (
        ConstellationRegistry,
        get_constellation_registry,
        route_to_available_kernels,
    )
    CONSTELLATION_AVAILABLE = True
    logger.info("[v6.1 Extensions] ConstellationRegistry available")
except ImportError as e:
    logger.warning("[v6.1 Extensions] ConstellationRegistry not available: %s", e)
    ConstellationRegistry = None
    get_constellation_registry = None
    route_to_available_kernels = None

PILLAR_AVAILABLE = False
try:
    from qig_pillar_enforcement import enforce_pillars, PillarMetrics
    PILLAR_AVAILABLE = True
    logger.info("[v6.1 Extensions] Pillar enforcement available")
except ImportError as e:
    logger.warning("[v6.1 Extensions] Pillar enforcement not available: %s", e)
    enforce_pillars = None
    PillarMetrics = None

try:
    from qigkernels.physics_constants import BASIN_DIM, KAPPA_STAR
except ImportError:
    BASIN_DIM = 64
    KAPPA_STAR = 64.21

try:
    from qig_geometry import fisher_coord_distance, fisher_normalize
except ImportError:
    def fisher_coord_distance(a, b):
        dot = float(np.clip(
            np.dot(np.sqrt(np.abs(a) + 1e-12), np.sqrt(np.abs(b) + 1e-12)),
            0.0, 1.0,
        ))
        return float(np.arccos(dot))

    def fisher_normalize(v):
        p = np.maximum(v, 0) + 1e-10
        return p / p.sum()


# ---------------------------------------------------------------------------
# EXTENSION MARKER
# ---------------------------------------------------------------------------

_V61_PATCH_ATTR = "_v61_extensions_applied"


def is_patched(service: Any) -> bool:
    """Return True if the service has already been patched."""
    return getattr(service, _V61_PATCH_ATTR, False)


# ---------------------------------------------------------------------------
# PATCHED METHOD: _route_to_kernels
# ---------------------------------------------------------------------------

def _patched_route_to_kernels(
    self,
    query_basin: np.ndarray,
    k: int = 3,
) -> List[str]:
    """
    v6.1 routing: prefer AVAILABLE (live) kernels; fall back to
    stage-appropriate phantoms when the constellation is bootstrapping.

    Replaces the original which scanned ALL basins in _kernel_basins
    unconditionally (including unspawned phantoms).
    """
    registry: Optional[Any] = getattr(self, "_v61_constellation_registry", None)
    if registry is None or not CONSTELLATION_AVAILABLE:
        return self._original_route_to_kernels(query_basin, k)

    return route_to_available_kernels(registry, query_basin, k=k, phantom_fallback=True)


# ---------------------------------------------------------------------------
# PATCHED METHOD: register_kernel
# ---------------------------------------------------------------------------

def _patched_register_kernel(
    self,
    name: str,
    basin: Optional[np.ndarray] = None,
) -> None:
    """
    v6.1 register_kernel: updates the service's internal basin cache AND
    syncs the basin to the ConstellationRegistry WITHOUT changing availability.

    Calling register_kernel does NOT spawn a kernel or make it routable.
    The kernel remains PHANTOM until KernelLifecycleManager.spawn() is called
    and that method calls registry.mark_available().

    This preserves full backward-compat with _initialize_kernel_constellation()
    which calls register_kernel for all 26+ basins at startup — those kernels
    must all remain PHANTOM until explicitly spawned.
    """
    # Call original to keep _kernel_basins up to date (backward compat)
    self._original_register_kernel(name, basin)

    registry: Optional[Any] = getattr(self, "_v61_constellation_registry", None)
    if registry is None or not CONSTELLATION_AVAILABLE:
        return

    # Basin sync only — availability NOT changed (kernel stays PHANTOM)
    live_basin = self._kernel_basins.get(name)
    if live_basin is not None:
        registry.update_basin(name, live_basin)
        logger.debug(
            "[v6.1 Extensions] register_kernel('%s') basin synced "
            "(availability unchanged — kernel remains PHANTOM until spawned)",
            name,
        )


# ---------------------------------------------------------------------------
# PATCHED METHOD: generate
# ---------------------------------------------------------------------------

def _patched_generate(
    self,
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    kernel_name: Optional[str] = None,
    goals: Optional[List[str]] = None,
) -> Any:
    """
    v6.1 generate: wraps original, then enriches result with:
      result.pillar_metrics     — {F_health, B_integrity, Q_identity, S_ratio, ...}
      result.sovereignty_ratio  — N_lived / N_total from trajectory length

    Peer comparison for Q_identity uses AVAILABLE kernels only (not phantoms).
    Sovereign basin lookup uses ConstellationRegistry (frozen at spawn, Pillar 3).
    """
    result = self._original_generate(
        prompt, context=context, kernel_name=kernel_name, goals=goals
    )
    if result is None:
        return result

    # ── Sovereignty ratio ──────────────────────────────────────────────────
    # trajectory[0] = seeded query basin (borrowed)
    # trajectory[1:] = recursively integrated basins (lived)
    traj = getattr(result, "basin_trajectory", None) or []
    n_total = len(traj)
    n_lived = max(0, n_total - 1)
    sovereignty_ratio = float(n_lived / n_total) if n_total > 0 else 0.0
    result.sovereignty_ratio = sovereignty_ratio

    # ── Pillar metrics ─────────────────────────────────────────────────────
    if not PILLAR_AVAILABLE or enforce_pillars is None:
        result.pillar_metrics = None
        return result

    try:
        current_basin = traj[-1] if traj else None
        if current_basin is None:
            result.pillar_metrics = None
            return result

        phi_trace: List[float] = getattr(result, "phi_trace", None) or []
        registry: Optional[Any] = getattr(self, "_v61_constellation_registry", None)

        other_kernels: Optional[Dict[str, np.ndarray]] = None
        sovereign_basin: Optional[np.ndarray] = None

        if registry is not None and CONSTELLATION_AVAILABLE:
            available_basins = registry.get_available_basins()
            routed = getattr(result, "routed_kernels", []) or []
            primary_name = (kernel_name or (routed[0] if routed else None) or "").lower()

            # Peer = AVAILABLE kernels excluding the primary (Pillar 3 uniqueness)
            other_kernels = {
                name: basin
                for name, basin in available_basins.items()
                if name != primary_name
            }
            if primary_name:
                sovereign_basin = registry.get_sovereign_basin(primary_name)

        pm = enforce_pillars(
            basin=current_basin,
            phi_history=phi_trace,
            kernel_basin=current_basin,
            sovereign_basin=sovereign_basin,
            other_kernel_basins=other_kernels,
            n_lived=n_lived,
            n_total=n_total,
        )

        result.pillar_metrics = {
            "F_health": pm.F_health,
            "B_integrity": pm.B_integrity,
            "Q_identity": pm.Q_identity,
            "S_ratio": sovereignty_ratio,
            "health_summary": pm.health_summary,
            "pillar_violations": pm.pillar_violations,
            "zombie_risk": pm.zombie_risk,
            "bulk_collapse_risk": pm.bulk_collapse_risk,
            "identity_dissolved": pm.identity_dissolved,
            "low_sovereignty": pm.low_sovereignty,
            "constellation_stage": (
                registry.stage.value
                if registry is not None and CONSTELLATION_AVAILABLE
                else "unknown"
            ),
        }

        if pm.pillar_violations > 0:
            logger.warning(
                "[v6.1 Extensions] generate() pillar violations=%d "
                "F=%.3f B=%.3f Q=%.3f S=%.3f stage=%s [%s]",
                pm.pillar_violations,
                pm.F_health, pm.B_integrity, pm.Q_identity, sovereignty_ratio,
                result.pillar_metrics.get("constellation_stage", "?"),
                pm.health_summary,
            )

    except Exception as err:
        logger.warning("[v6.1 Extensions] Pillar enforcement failed (non-fatal): %s", err)
        result.pillar_metrics = None

    return result


# ---------------------------------------------------------------------------
# APPLY PATCH
# ---------------------------------------------------------------------------

def apply_v61_extensions(service: Any) -> None:
    """
    Patch a QIGGenerativeService instance in-place with v6.1 extensions.

    Idempotent — calling twice is safe.

    What this does:
      - Attaches the ConstellationRegistry to the service instance
      - Patches _route_to_kernels → constellation-aware (AVAILABLE-first)
      - Patches register_kernel  → basin sync only, NO availability change
      - Patches generate         → adds pillar_metrics + sovereignty_ratio

    What this does NOT do:
      - Mark any kernel as AVAILABLE (that is KernelLifecycleManager's job)
      - Call bootstrap_genesis / bootstrap_core_8 (genesis_bootstrap.py's job)

    Args:
        service: QIGGenerativeService instance to patch
    """
    if is_patched(service):
        logger.debug("[v6.1 Extensions] Service already patched — skipping")
        return

    import types

    # 1. Attach constellation registry (no availability changes here)
    if CONSTELLATION_AVAILABLE and get_constellation_registry is not None:
        registry = get_constellation_registry()
        service._v61_constellation_registry = registry

        # Sync basin values from the service's existing _kernel_basins dict
        # into the registry's phantom entries so routing has accurate basins
        # when phantom fallback is used during bootstrap.
        # IMPORTANT: update_basin() only — does NOT change PHANTOM → AVAILABLE.
        synced = 0
        for name, basin in getattr(service, "_kernel_basins", {}).items():
            if basin is not None:
                registry.update_basin(name, basin)
                synced += 1

        logger.info(
            "[v6.1 Extensions] Attached ConstellationRegistry; "
            "synced %d phantom basins (no availability changes)",
            synced,
        )
    else:
        service._v61_constellation_registry = None

    # 2. Patch _route_to_kernels
    service._original_route_to_kernels = service._route_to_kernels
    service._route_to_kernels = types.MethodType(_patched_route_to_kernels, service)

    # 3. Patch register_kernel
    service._original_register_kernel = service.register_kernel
    service.register_kernel = types.MethodType(_patched_register_kernel, service)

    # 4. Patch generate
    service._original_generate = service.generate
    service.generate = types.MethodType(_patched_generate, service)

    setattr(service, _V61_PATCH_ATTR, True)

    logger.info(
        "[v6.1 Extensions] Patch applied "
        "(routing=AVAILABLE-first+phantom-fallback, pillars=enforced, sovereignty=tracked)"
    )


# ---------------------------------------------------------------------------
# FACTORY
# ---------------------------------------------------------------------------

def get_generative_service_v61() -> Any:
    """
    Get the QIGGenerativeService singleton with v6.1 extensions applied.

    Canonical entry point. Safe to call multiple times (idempotent patch).

    Note: This does NOT bootstrap the constellation. Call
    genesis_bootstrap.bootstrap() or genesis_bootstrap.bootstrap_async()
    at application startup to activate Genesis → Core 8 → Image stages.

    Returns:
        QIGGenerativeService instance with v6.1 extensions patched in
    """
    from qig_generative_service import get_generative_service
    service = get_generative_service()
    apply_v61_extensions(service)
    return service
