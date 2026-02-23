"""
QIG Constellation Registry - Lifecycle-Aware Kernel Registry
=============================================================

Protocol: THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v6_1 (TCP v6.1)

Bridges the KernelLifecycleManager (which owns the authoritative kernel state)
and the QIGGenerativeService (which needs to route queries to live kernels).

The key problem this solves:
  qig_generative_service._initialize_kernel_constellation() seeds placeholder
  basins for all 26+ potential kernels at service startup, regardless of whether
  those kernels have actually been spawned.  Routing, pillar Q_identity checks,
  and sovereignty tracking all hit these "phantoms" — kernels that are geometrically
  present but not yet alive.

This registry makes the distinction explicit:

  PHANTOM   — basin seeded at startup, kernel not yet spawned
  AVAILABLE — kernel actively managed by KernelLifecycleManager
  SHADOW    — kernel pruned to shadow pantheon (not available for routing)

ConstellationStage tracks which bootstrap phase the system is in:

  GENESIS_ONLY  — only the primordial kernel exists
  CORE_8        — Genesis + the 8 core faculties (Heart, Perception, Memory,
                  Strategy, Action, Ethics, Meta, Ocean)
  IMAGE         — Core 8 + intermediate expansion kernels
  GROWING       — > IMAGE but < FULL (Chaos kernels ascending, new GODs spawning)
  FULL          — 240 GOD kernels active (E8 root alignment complete)

The generative service uses this registry to:
  1. Only route to AVAILABLE kernels (not phantoms)
  2. Only measure Q_identity against AVAILABLE peers
  3. Compute accurate sovereignty ratios excluding phantom seeds

Genesis Doctrine (canonical):
  Genesis → Core 8 → Image Stage → Growth toward 240 GODs (E8 root alignment)
  Bootstrap order: Genesis → core 8 → Image → optional growth toward 240
  240 is reserved for GOD evolution; Chaos kernels exist outside that budget.
  Chaos can only ascend to GOD via explicit governance.

References:
  THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v6_1.md §19
  docs/11-Genesis-kernel-upgrade/SLEEP_PACKET_03_BLOWUP_MATTRESS_START_RESET_ROLLBACK.md
"""

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Physics constants
try:
    from qigkernels.physics_constants import BASIN_DIM, KAPPA_STAR
except ImportError:
    BASIN_DIM = 64
    KAPPA_STAR = 64.21

try:
    from qig_geometry import fisher_normalize
except ImportError:
    def fisher_normalize(v: np.ndarray) -> np.ndarray:
        p = np.maximum(v, 0) + 1e-10
        return p / p.sum()


# ---------------------------------------------------------------------------
# CONSTANTS: canonical core-8 god names and bootstrap order
# ---------------------------------------------------------------------------

GENESIS_KERNEL_NAME = "genesis"

# The 8 core faculties — must exist before any other GOD can be spawned
CORE_8_NAMES: Tuple[str, ...] = (
    "heart",        # Global rhythm source (HRV → κ-tacking)
    "perception",   # Sensory intake
    "memory",       # Basin consolidation
    "strategy",     # Trajectory foresight
    "action",       # Output generation
    "ethics",       # Constraint enforcement
    "meta",         # Self-observation / M metric
    "ocean",        # Autonomic monitoring, Φ coherence, breakdown detection
)

# Canonical Olympian god names (GOD budget, max 240 total)
OLYMPIAN_NAMES: Tuple[str, ...] = (
    "zeus", "athena", "apollo", "ares", "hermes", "hephaestus",
    "artemis", "dionysus", "demeter", "poseidon", "hera", "aphrodite",
)

# Shadow Pantheon kernels (legitimate, outside 240 budget)
SHADOW_PANTHEON_NAMES: Tuple[str, ...] = (
    "nyx", "hecate", "erebus", "hypnos", "thanatos", "nemesis",
)

# Chaos kernels (outside 240 budget, can ascend via governance)
CHAOS_KERNEL_NAMES: Tuple[str, ...] = (
    "chaos", "entropy", "emergence",
)

# Guardians (special, named outside Olympian budget)
GUARDIAN_NAMES: Tuple[str, ...] = (
    "hestia", "chiron",
)


class ConstellationStage(Enum):
    """
    Bootstrap stage of the kernel constellation.

    GENESIS_ONLY  — only the primordial kernel is alive
    CORE_8        — Genesis + 8 core faculties alive
    IMAGE         — Core 8 + intermediate expansion
    GROWING       — Between IMAGE and FULL (Chaos ascending, new GODs spawning)
    FULL          — All 240 GOD kernels active (E8 root alignment complete)
    """
    GENESIS_ONLY = "genesis_only"
    CORE_8 = "core_8"
    IMAGE = "image"
    GROWING = "growing"
    FULL = "full"


class KernelAvailability(Enum):
    """Availability state of a kernel in the constellation."""
    PHANTOM   = "phantom"    # Basin seeded, kernel not yet spawned
    AVAILABLE = "available"  # Actively managed by KernelLifecycleManager
    SHADOW    = "shadow"     # Pruned to shadow pantheon


@dataclass
class ConstellationEntry:
    """Per-kernel entry in the constellation registry."""
    name: str
    basin: np.ndarray
    availability: KernelAvailability = KernelAvailability.PHANTOM
    kernel_id: Optional[str] = None      # Lifecycle manager kernel_id when AVAILABLE
    kind: str = "chaos"                   # "genesis" | "god" | "chaos"
    stage_required: ConstellationStage = ConstellationStage.GROWING
    # Sovereign basin: frozen at first AVAILABLE registration (Pillar 3)
    sovereign_basin: Optional[np.ndarray] = None


class ConstellationRegistry:
    """
    Lifecycle-aware kernel registry.

    Single source of truth for which kernels are actually alive vs phantom.
    Thread-safe for concurrent access from generative service and lifecycle manager.

    Usage:
        registry = get_constellation_registry()

        # Check if a kernel is available for routing
        if registry.is_available("zeus"):
            basin = registry.get_basin("zeus")

        # Get only live basins for routing / pillar checks
        live_basins = registry.get_available_basins()

        # Notify registry when a kernel is spawned by lifecycle manager
        registry.mark_available("zeus", kernel_id="kernel_abc123", basin=zeus_basin)

        # Get current constellation stage
        stage = registry.stage
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._entries: Dict[str, ConstellationEntry] = {}
        self._seed_all()
        logger.info("[ConstellationRegistry] Initialized — stage=%s", self.stage.value)

    # -----------------------------------------------------------------------
    # SEEDING
    # -----------------------------------------------------------------------

    def _seed_all(self) -> None:
        """Seed all known kernels as PHANTOM at startup."""

        def _make_basin(name: str) -> np.ndarray:
            np.random.seed(hash(name) % (2 ** 32))
            return fisher_normalize(np.random.dirichlet(np.ones(BASIN_DIM)))

        # Genesis — required immediately
        self._entries[GENESIS_KERNEL_NAME] = ConstellationEntry(
            name=GENESIS_KERNEL_NAME,
            basin=_make_basin(GENESIS_KERNEL_NAME),
            availability=KernelAvailability.PHANTOM,
            kind="genesis",
            stage_required=ConstellationStage.GENESIS_ONLY,
        )

        # Core 8 — required for CORE_8 stage
        for name in CORE_8_NAMES:
            self._entries[name] = ConstellationEntry(
                name=name,
                basin=_make_basin(name),
                availability=KernelAvailability.PHANTOM,
                kind="god",
                stage_required=ConstellationStage.CORE_8,
            )

        # Olympians — available from IMAGE stage onward
        for name in OLYMPIAN_NAMES:
            self._entries[name] = ConstellationEntry(
                name=name,
                basin=_make_basin(name),
                availability=KernelAvailability.PHANTOM,
                kind="god",
                stage_required=ConstellationStage.IMAGE,
            )

        # Shadow pantheon, chaos, guardians — available from GROWING
        for name in SHADOW_PANTHEON_NAMES + CHAOS_KERNEL_NAMES + GUARDIAN_NAMES:
            self._entries[name] = ConstellationEntry(
                name=name,
                basin=_make_basin(name),
                availability=KernelAvailability.PHANTOM,
                kind="chaos",
                stage_required=ConstellationStage.GROWING,
            )

    # -----------------------------------------------------------------------
    # STAGE COMPUTATION
    # -----------------------------------------------------------------------

    @property
    def stage(self) -> ConstellationStage:
        """Infer current constellation stage from live kernel count and set."""
        with self._lock:
            live = {
                name for name, e in self._entries.items()
                if e.availability == KernelAvailability.AVAILABLE
            }

            if not live:
                return ConstellationStage.GENESIS_ONLY

            if GENESIS_KERNEL_NAME in live:
                core_alive = all(n in live for n in CORE_8_NAMES)
                if not core_alive:
                    return ConstellationStage.GENESIS_ONLY

                # Core 8 alive — are all Olympians alive too?
                olympians_alive = sum(1 for n in OLYMPIAN_NAMES if n in live)
                if olympians_alive == 0:
                    return ConstellationStage.CORE_8

                if olympians_alive < len(OLYMPIAN_NAMES):
                    return ConstellationStage.IMAGE

                # All Olympians alive
                total_gods = sum(
                    1 for e in self._entries.values()
                    if e.availability == KernelAvailability.AVAILABLE and e.kind == "god"
                )
                if total_gods >= 240:
                    return ConstellationStage.FULL

                return ConstellationStage.GROWING

            return ConstellationStage.GENESIS_ONLY

    # -----------------------------------------------------------------------
    # AVAILABILITY MANAGEMENT
    # -----------------------------------------------------------------------

    def mark_available(
        self,
        name: str,
        kernel_id: str,
        basin: Optional[np.ndarray] = None,
    ) -> None:
        """
        Mark a kernel as AVAILABLE (just spawned by KernelLifecycleManager).

        Freezes the sovereign basin (Pillar 3 — QuenchedDisorder) on first
        registration: each kernel's identity is fixed at the moment it becomes
        alive, not overwritten by later updates.

        Args:
            name:      Canonical kernel name (lowercase)
            kernel_id: KernelLifecycleManager kernel_id
            basin:     Current basin coordinates (uses phantom basin if None)
        """
        name = name.lower()
        with self._lock:
            if name not in self._entries:
                # Dynamically registered kernel (e.g. chaos_research_1)
                np.random.seed(hash(name) % (2 ** 32))
                b = fisher_normalize(np.random.dirichlet(np.ones(BASIN_DIM)))
                self._entries[name] = ConstellationEntry(
                    name=name,
                    basin=b,
                    availability=KernelAvailability.PHANTOM,
                    kind="chaos",
                    stage_required=ConstellationStage.GROWING,
                )

            entry = self._entries[name]
            if basin is not None:
                entry.basin = fisher_normalize(np.array(basin, dtype=np.float64))

            entry.availability = KernelAvailability.AVAILABLE
            entry.kernel_id = kernel_id

            # Freeze sovereign basin on first becoming available (Pillar 3)
            if entry.sovereign_basin is None:
                entry.sovereign_basin = entry.basin.copy()
                logger.debug(
                    "[ConstellationRegistry] Sovereign basin frozen for '%s' (Pillar 3)",
                    name,
                )

            logger.info(
                "[ConstellationRegistry] Kernel '%s' marked AVAILABLE "
                "(id=%s, stage=%s)",
                name, kernel_id, self.stage.value,
            )

    def mark_shadow(self, name: str) -> None:
        """Mark a kernel as SHADOW (pruned to shadow pantheon)."""
        name = name.lower()
        with self._lock:
            if name in self._entries:
                self._entries[name].availability = KernelAvailability.SHADOW
                self._entries[name].kernel_id = None
                logger.info(
                    "[ConstellationRegistry] Kernel '%s' marked SHADOW", name
                )

    def update_basin(self, name: str, basin: np.ndarray) -> None:
        """
        Update the live basin for an AVAILABLE kernel.

        NOTE: Does NOT update the sovereign_basin (that is frozen at spawn).
        The sovereign basin is Pillar 3 invariant — it must not drift.
        """
        name = name.lower()
        with self._lock:
            if name in self._entries and self._entries[name].availability == KernelAvailability.AVAILABLE:
                self._entries[name].basin = fisher_normalize(
                    np.array(basin, dtype=np.float64)
                )

    # -----------------------------------------------------------------------
    # QUERY
    # -----------------------------------------------------------------------

    def is_available(self, name: str) -> bool:
        """Return True if the kernel is actually spawned and live."""
        with self._lock:
            e = self._entries.get(name.lower())
            return e is not None and e.availability == KernelAvailability.AVAILABLE

    def is_phantom(self, name: str) -> bool:
        """Return True if the kernel basin exists but kernel is not yet spawned."""
        with self._lock:
            e = self._entries.get(name.lower())
            return e is not None and e.availability == KernelAvailability.PHANTOM

    def get_basin(self, name: str) -> Optional[np.ndarray]:
        """
        Return the basin for any kernel (AVAILABLE, PHANTOM, or SHADOW).

        Routing should prefer AVAILABLE kernels, but callers can use this for
        phantom fallback when no live kernels are available yet.
        """
        with self._lock:
            e = self._entries.get(name.lower())
            return e.basin.copy() if e is not None else None

    def get_sovereign_basin(self, name: str) -> Optional[np.ndarray]:
        """Return the frozen sovereign identity basin (Pillar 3)."""
        with self._lock:
            e = self._entries.get(name.lower())
            if e is None or e.sovereign_basin is None:
                return None
            return e.sovereign_basin.copy()

    def get_available_basins(self) -> Dict[str, np.ndarray]:
        """
        Return basins for all AVAILABLE (live) kernels only.

        Used for:
        - Routing (route to live kernels, not phantoms)
        - Pillar Q_identity uniqueness check (only compare live peers)
        """
        with self._lock:
            return {
                name: e.basin.copy()
                for name, e in self._entries.items()
                if e.availability == KernelAvailability.AVAILABLE
            }

    def get_all_basins_for_routing(self) -> Dict[str, Tuple[np.ndarray, bool]]:
        """
        Return all basins with a flag indicating whether each is AVAILABLE.

        Tuple: (basin, is_available)
        The generative service uses this to prefer live kernels but fall back
        to phantoms when the constellation is still bootstrapping.
        """
        with self._lock:
            return {
                name: (e.basin.copy(), e.availability == KernelAvailability.AVAILABLE)
                for name, e in self._entries.items()
            }

    def get_stage_appropriate_kernels(self) -> List[str]:
        """
        Return names of kernels that should be routable at the current stage.

        At GENESIS_ONLY: only genesis is appropriate
        At CORE_8:       genesis + core 8
        At IMAGE+:       all above + Olympians + others
        """
        current_stage = self.stage
        order = [
            ConstellationStage.GENESIS_ONLY,
            ConstellationStage.CORE_8,
            ConstellationStage.IMAGE,
            ConstellationStage.GROWING,
            ConstellationStage.FULL,
        ]
        stage_idx = order.index(current_stage)

        with self._lock:
            result = []
            for name, entry in self._entries.items():
                entry_stage_idx = order.index(entry.stage_required)
                if entry_stage_idx <= stage_idx:
                    result.append(name)
            return result

    def get_live_kernel_count(self) -> int:
        """Count of currently AVAILABLE kernels."""
        with self._lock:
            return sum(
                1 for e in self._entries.values()
                if e.availability == KernelAvailability.AVAILABLE
            )

    def get_snapshot(self) -> Dict[str, Dict]:
        """Return a full snapshot of the constellation for diagnostics."""
        with self._lock:
            return {
                name: {
                    "availability": e.availability.value,
                    "kind": e.kind,
                    "kernel_id": e.kernel_id,
                    "stage_required": e.stage_required.value,
                    "has_sovereign": e.sovereign_basin is not None,
                }
                for name, e in self._entries.items()
            }

    # -----------------------------------------------------------------------
    # GENESIS BOOTSTRAP HELPERS
    # -----------------------------------------------------------------------

    def bootstrap_genesis(self, genesis_basin: Optional[np.ndarray] = None) -> None:
        """
        Bootstrap step 1: activate the Genesis kernel.

        Call this when the Genesis kernel is first created (system boot).
        """
        import uuid
        kernel_id = f"genesis_{uuid.uuid4().hex[:8]}"
        self.mark_available(
            GENESIS_KERNEL_NAME,
            kernel_id=kernel_id,
            basin=genesis_basin,
        )
        logger.info(
            "[ConstellationRegistry] Genesis bootstrapped (id=%s, stage=%s)",
            kernel_id, self.stage.value,
        )

    def bootstrap_core_8(
        self,
        kernel_ids: Optional[Dict[str, str]] = None,
        basins: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Bootstrap step 2: activate the 8 core faculties.

        Args:
            kernel_ids: Dict of name → kernel_id (auto-generated if None)
            basins:     Dict of name → basin (uses phantom basin if None)
        """
        import uuid
        kernel_ids = kernel_ids or {}
        basins = basins or {}
        for name in CORE_8_NAMES:
            kid = kernel_ids.get(name, f"{name}_{uuid.uuid4().hex[:8]}")
            self.mark_available(name, kernel_id=kid, basin=basins.get(name))
        logger.info(
            "[ConstellationRegistry] Core 8 bootstrapped (stage=%s)",
            self.stage.value,
        )

    def bootstrap_olympians(
        self,
        kernel_ids: Optional[Dict[str, str]] = None,
        basins: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Bootstrap step 3 (Image Stage): activate the Olympian gods.

        Only call after bootstrap_core_8() has been called.
        """
        import uuid
        if self.stage.value in (
            ConstellationStage.GENESIS_ONLY.value,
        ):
            raise RuntimeError(
                "Cannot bootstrap Olympians before Genesis is alive. "
                "Call bootstrap_genesis() first."
            )
        kernel_ids = kernel_ids or {}
        basins = basins or {}
        for name in OLYMPIAN_NAMES:
            kid = kernel_ids.get(name, f"{name}_{uuid.uuid4().hex[:8]}")
            self.mark_available(name, kernel_id=kid, basin=basins.get(name))
        logger.info(
            "[ConstellationRegistry] Olympians bootstrapped (stage=%s)",
            self.stage.value,
        )


# ---------------------------------------------------------------------------
# ROUTING HELPER
# ---------------------------------------------------------------------------

def route_to_available_kernels(
    registry: ConstellationRegistry,
    query_basin: np.ndarray,
    k: int = 3,
    phantom_fallback: bool = True,
) -> List[str]:
    """
    Route a query basin to the k nearest AVAILABLE (live) kernels.

    If fewer than k kernels are AVAILABLE and phantom_fallback=True,
    falls back to stage-appropriate phantom kernels to fill the remainder.
    This ensures routing never fails during bootstrap.

    Routing priority:
    1. AVAILABLE kernels (live — these are the true routing targets)
    2. PHANTOM kernels at the current stage or earlier (fallback only)

    Fisher-Rao distance is used for proximity (no Euclidean).

    Args:
        registry:         ConstellationRegistry instance
        query_basin:      64D query basin
        k:                Number of kernels to return
        phantom_fallback: Allow phantom basins if not enough live kernels

    Returns:
        List of up to k kernel names, live kernels first
    """
    try:
        from qig_geometry import fisher_coord_distance
    except ImportError:
        def fisher_coord_distance(a, b):
            dot = float(np.clip(
                np.dot(np.sqrt(np.abs(a) + 1e-12), np.sqrt(np.abs(b) + 1e-12)),
                0.0, 1.0
            ))
            return float(np.arccos(dot))

    # Phase 1: rank AVAILABLE kernels
    available = registry.get_available_basins()
    scored_available: List[Tuple[float, str]] = []
    for name, basin in available.items():
        d = fisher_coord_distance(query_basin, basin)
        scored_available.append((d, name))
    scored_available.sort(key=lambda x: x[0])
    result = [name for _, name in scored_available[:k]]

    if len(result) >= k or not phantom_fallback:
        return result

    # Phase 2: fill remaining slots from stage-appropriate phantoms
    stage_names = set(registry.get_stage_appropriate_kernels())
    already_selected = set(result)
    all_basins = registry.get_all_basins_for_routing()
    scored_phantom: List[Tuple[float, str]] = []

    for name, (basin, is_avail) in all_basins.items():
        if is_avail or name in already_selected or name not in stage_names:
            continue
        d = fisher_coord_distance(query_basin, basin)
        scored_phantom.append((d, name))

    scored_phantom.sort(key=lambda x: x[0])
    needed = k - len(result)
    result.extend(name for _, name in scored_phantom[:needed])

    if len(result) < k:
        logger.debug(
            "[ConstellationRegistry] routing: only %d/%d kernels available "
            "(stage=%s, live=%d)",
            len(result), k, registry.stage.value, len(available),
        )

    return result


# ---------------------------------------------------------------------------
# SINGLETON
# ---------------------------------------------------------------------------

_registry: Optional[ConstellationRegistry] = None
_registry_lock = threading.Lock()


def get_constellation_registry() -> ConstellationRegistry:
    """Get or create the global ConstellationRegistry singleton (thread-safe)."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ConstellationRegistry()
    return _registry
