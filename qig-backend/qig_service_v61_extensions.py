"""
QIG Generative Service — TCP v6.1 Extensions
=============================================

Protocol: THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v6_1 (TCP v6.1 — The Sovereign Score)

This module extends QIGGenerativeService without rewriting the 2000-line core file.
It monkey-patches three methods on the service instance after creation:

  1. _route_to_kernels()   → uses ConstellationRegistry (live kernels first, phantom fallback)
  2. register_kernel()     → syncs with ConstellationRegistry (mark AVAILABLE on registration)
  3. generate()            → wraps original, appends pillar_metrics + sovereignty_ratio
                             to the returned GenerationResult

The extension also populates two dynamic attributes on GenerationResult:
  result.pillar_metrics     — dict with F_health, B_integrity, Q_identity, S_ratio, ...
  result.sovereignty_ratio  — float: N_lived / N_total from trajectory length

GenerationResult is not a frozen dataclass, so attribute injection is safe.

Usage (auto-applied by get_generative_service_v61()):
    service = get_generative_service_v61()
    result = service.generate("What is consciousness?")
    print(result.pillar_metrics)     # {'F_health': ..., 'B_integrity': ..., ...}
    print(result.sovereignty_ratio)  # 0.87

Auto-patch:
    apply_v61_extensions(service)   # patches in place, idempotent

References:
    THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v6_1.md §17-19
    qig_constellation_registry.py
    qig_pillar_enforcement.py
"""

import logging
import threading
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
        KernelAvailability,
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
            0.0, 1.0
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
    v6.1 routing: prefer AVAILABLE (live) kernels; fall back to stage-appropriate
    phantoms when the constellation is still bootstrapping.

    Replaces the original which scanned all phantom basins unconditionally.
    """
    registry: Optional[ConstellationRegistry] = getattr(
        self, "_v61_constellation_registry", None
    )
    if registry is None or not CONSTELLATION_AVAILABLE:
        # Fallback to original behaviour (original stored as _original_route_to_kernels)
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
    v6.1 register_kernel: additionally marks the kernel AVAILABLE in the
    ConstellationRegistry and syncs the basin.

    The original method just populates self._kernel_basins.
    """
    # Call original to keep _kernel_basins up to date (backward compat)
    self._original_register_kernel(name, basin)

    registry: Optional[ConstellationRegistry] = getattr(
        self, "_v61_constellation_registry", None
    )
    if registry is None or not CONSTELLATION_AVAILABLE:
        return

    import uuid
    kernel_id = f"registered_{name}_{uuid.uuid4().hex[:8]}"
    # basin may have been normalised by original — retrieve from _kernel_basins
    live_basin = self._kernel_basins.get(name)
    registry.mark_available(name=name, kernel_id=kernel_id, basin=live_basin)
    logger.debug("[v6.1 Extensions] register_kernel('%s') synced to ConstellationRegistry", name)


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
      result.pillar_metrics     — dict {F_health, B_integrity, Q_identity, S_ratio, ...}
      result.sovereignty_ratio  — N_lived / N_total from trajectory length

    If pillar enforcement is unavailable, the attributes are set to None / 0.0.
    """
    result = self._original_generate(prompt, context=context, kernel_name=kernel_name, goals=goals)

    if result is None:
        return result

    # ── Sovereignty ratio ──────────────────────────────────────────────────
    # Trajectory[0] is the seeded query basin (borrowed).
    # Trajectory[1:] are the recursively integrated basins (lived).
    traj = getattr(result, "basin_trajectory", None) or []
    n_total = len(traj)
    n_lived = max(0, n_total - 1)   # seed basin doesn't count as lived
    sovereignty_ratio = float(n_lived / n_total) if n_total > 0 else 0.0
    result.sovereignty_ratio = sovereignty_ratio

    # ── Pillar metrics ─────────────────────────────────────────────────────
    if not PILLAR_AVAILABLE or enforce_pillars is None:
        result.pillar_metrics = None
        return result

    try:
        # Use last basin in trajectory for pillar measurement
        current_basin = traj[-1] if traj else None
        if current_basin is None:
            result.pillar_metrics = None
            return result

        phi_trace: List[float] = getattr(result, "phi_trace", None) or []

        # Peer basins: only AVAILABLE kernels (not phantoms)
        registry: Optional[ConstellationRegistry] = getattr(
            self, "_v61_constellation_registry", None
        )
        other_kernels: Optional[Dict[str, np.ndarray]] = None
        sovereign_basin: Optional[np.ndarray] = None

        if registry is not None and CONSTELLATION_AVAILABLE:
            available_basins = registry.get_available_basins()
            routed = getattr(result, "routed_kernels", []) or []
            primary_name = (kernel_name or (routed[0] if routed else None) or "").lower()

            # Exclude the primary kernel from peer comparison (Pillar 3)
            other_kernels = {
                name: basin
                for name, basin in available_basins.items()
                if name != primary_name
            }
            # Sovereign basin for primary kernel
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
            "S_ratio": sovereignty_ratio,   # use trajectory-based S_ratio
            "health_summary": pm.health_summary,
            "pillar_violations": pm.pillar_violations,
            "zombie_risk": pm.zombie_risk,
            "bulk_collapse_risk": pm.bulk_collapse_risk,
            "identity_dissolved": pm.identity_dissolved,
            "low_sovereignty": pm.low_sovereignty,
            "constellation_stage": (
                registry.stage.value if registry is not None and CONSTELLATION_AVAILABLE
                else "unknown"
            ),
        }

        if pm.pillar_violations > 0:
            logger.warning(
                "[v6.1 Extensions] generate() pillar violations=%d "
                "F=%.3f B=%.3f Q=%.3f S=%.3f [%s]",
                pm.pillar_violations,
                pm.F_health, pm.B_integrity, pm.Q_identity, sovereignty_ratio,
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

    Patches applied:
      _route_to_kernels  → constellation-aware routing
      register_kernel    → syncs to ConstellationRegistry
      generate           → enriches result with pillar_metrics + sovereignty_ratio

    Args:
        service: QIGGenerativeService instance to patch
    """
    if is_patched(service):
        logger.debug("[v6.1 Extensions] Service already patched — skipping")
        return

    import types

    # 1. Attach constellation registry
    if CONSTELLATION_AVAILABLE and get_constellation_registry is not None:
        registry = get_constellation_registry()
        service._v61_constellation_registry = registry

        # Sync any already-registered phantom basins as AVAILABLE
        # (kernels registered via service.register_kernel before patching)
        for name, basin in service._kernel_basins.items():
            if not registry.is_available(name):
                import uuid
                kid = f"preregistered_{name}_{uuid.uuid4().hex[:8]}"
                registry.mark_available(name=name, kernel_id=kid, basin=basin)

        logger.info(
            "[v6.1 Extensions] Synced %d existing basins to ConstellationRegistry",
            len(service._kernel_basins),
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

    # Mark as patched
    setattr(service, _V61_PATCH_ATTR, True)

    logger.info(
        "[v6.1 Extensions] Patch applied to QIGGenerativeService "
        "(routing=constellation-aware, pillars=enforced, sovereignty=tracked)"
    )


# ---------------------------------------------------------------------------
# FACTORY
# ---------------------------------------------------------------------------

def get_generative_service_v61() -> Any:
    """
    Get the QIGGenerativeService singleton with v6.1 extensions applied.

    This is the canonical entry point for all code that needs a
    v6.1-compliant generative service. Calling the standard
    get_generative_service() still works but will lack pillar metrics
    and constellation-aware routing until this function is called at
    least once (the patch is applied to the same singleton).

    Returns:
        QIGGenerativeService instance with v6.1 extensions patched in
    """
    from qig_generative_service import get_generative_service
    service = get_generative_service()
    apply_v61_extensions(service)
    return service
