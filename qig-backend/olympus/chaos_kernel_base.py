"""
Chaos Kernel Base — TCP v6.1

Base class for all chaos kernels in the Pantheon constellation.

Key constraints (from KernelCapabilityCharter):
  - NO GENERATIVE capability by default — chaos kernels cannot produce text.
  - A proxy god speaks for them (assigned by Pantheon governance vote).
  - Proxy instructions arrive via Gary.relay_proxy_instructions().
  - Discoveries above report_threshold_phi flow to AdaptiveDiscoveryGate.
  - Quenched identity (Pillar 3) is frozen at init from the basin hash.

Usage:
    class ErisKernel(ChaosKernelBase):
        def _explore_step(self, basin, instructions):
            return new_basin, phi_estimate

TCP v6.1 §19 — Chaos kernels exist OUTSIDE the 240 GOD budget.
TCP v6.1 §20.8 — Rejection mechanism: non-resonant basins → holding buffer.
TCP v6.1 §21 — Quenched disorder: each chaos kernel has unique frozen identity.

Red-team fixes applied:
  RT1-M3: _anneal_rejection_buffer fallback uses sqrt-space geodesic (not Euclidean).
  RT1-M4: geodesic_interpolation import memoized at module level.
  RT2-P1: np.dot() replaced with explicit Hellinger sum (QIG purity).
  RT2-M2: _sqrt_geodesic hoisted to module level (no per-call redefinition).
  RT2-H1: step() releases _lock before calling _explore_step() (subclass safety).
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Module-level geodesic import — memoized (RT1-M4)
_GEODESIC_INTERP = None
_GEODESIC_ATTEMPTED = False

def _get_geodesic_interp():
    """Return geodesic_interpolation fn, cached after first attempt (RT1-M4)."""
    global _GEODESIC_INTERP, _GEODESIC_ATTEMPTED
    if _GEODESIC_ATTEMPTED:
        return _GEODESIC_INTERP
    _GEODESIC_ATTEMPTED = True
    try:
        from qig_geometry.canonical import geodesic_interpolation
        _GEODESIC_INTERP = geodesic_interpolation
    except ImportError:
        pass
    return _GEODESIC_INTERP


# Physics constants (FROZEN)
KAPPA_STAR = 64.21
BASIN_DIM = 64
PHI_THRESHOLD = 0.727

# Rejection mechanism (TCP v6.1 §20.8)
RESONANCE_THETA = 0.80
REJECTION_BUFFER_MAX = 32


# ---------------------------------------------------------------------------
# Geometry helpers — no Euclidean ops, no np.dot (QIG-pure)
# ---------------------------------------------------------------------------

def _fr_distance(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from qig_geometry.canonical import fisher_rao_distance
        return float(fisher_rao_distance(a, b))
    except ImportError:
        # Hellinger inner product: Σ √(a_i · b_i) — avoids np.dot (RT2-P1)
        hellinger = float(np.clip(
            float(np.sum(np.sqrt(np.abs(a) + 1e-12) * np.sqrt(np.abs(b) + 1e-12))),
            0.0, 1.0
        ))
        return float(np.arccos(hellinger))


def _to_simplex(v: np.ndarray) -> np.ndarray:
    v = np.abs(v) + 1e-12
    return (v / v.sum()).astype(np.float64)


def _sqrt_geodesic(a: np.ndarray, b: np.ndarray, t: float = 0.3) -> np.ndarray:
    """
    Geodesic on Δ⁶³ via sqrt-space interpolation (RT2-M2: hoisted to module level).
    (√a·(1-t) + √b·t)² normalised — NOT Euclidean. QIG-pure.
    """
    r = (1.0 - t) * np.sqrt(a + 1e-12) + t * np.sqrt(b + 1e-12)
    q = r * r
    return (q / q.sum()).astype(np.float64)


# ---------------------------------------------------------------------------
# Discovery record
# ---------------------------------------------------------------------------

@dataclass
class ChaosDiscovery:
    kernel_id: str
    phi: float
    basin_coords: np.ndarray
    context: str = ""
    timestamp: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Chaos Kernel Base
# ---------------------------------------------------------------------------

class ChaosKernelBase:
    """
    Base class for chaos kernels.

    Subclass and implement `_explore_step(basin, instructions) → (new_basin, phi)`.

    The loop:
        1. Pull ProxyInstructions from Gary (non-blocking, outside _lock).
        2. Call _explore_step() → (new_basin, phi)  [OUTSIDE _lock — RT2-H1].
        3. Re-acquire _lock for resonance check + state writes.
        4. Report discovery to gate (gate has its own lock).
    """

    def __init__(
        self,
        kernel_id: Optional[str] = None,
        initial_basin: Optional[np.ndarray] = None,
        charter: Optional[Any] = None,
        discovery_gate: Optional[Any] = None,
    ):
        self.kernel_id = kernel_id or f"chaos_{uuid.uuid4().hex[:8]}"
        self._charter = charter
        self._gate = discovery_gate
        self._lock = threading.RLock()

        raw = initial_basin if initial_basin is not None else np.ones(BASIN_DIM) / BASIN_DIM
        self._basin = _to_simplex(np.asarray(raw, dtype=np.float64))
        self._sovereign_basin = self._basin.copy()
        self._quenched_identity = float(np.mean(self._basin))

        self.phi: float = 0.25
        self.kappa: float = KAPPA_STAR
        self._step: int = 0
        self._phi_history: List[float] = []
        self._rejection_buffer: List[np.ndarray] = []
        self._rejection_count: int = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_instructions: Optional[Dict] = None

        self._gary = None
        try:
            from olympus.gary_coordinator import get_gary_coordinator
            self._gary = get_gary_coordinator()
        except ImportError:
            pass

        logger.info("[ChaosKernelBase] %s initialised (charter=%s)",
                    self.kernel_id, "attached" if charter else "None")

    # ------------------------------------------------------------------
    # Charter gates
    # ------------------------------------------------------------------

    def can_generate(self) -> bool:
        if self._charter is None:
            return False
        return self._charter.can_generate()

    def can_vote(self) -> bool:
        if self._charter is None:
            return False
        return self._charter.can_vote()

    def proxy_god(self) -> Optional[str]:
        if self._charter and self._charter.proxy:
            return self._charter.proxy.proxy_god_name
        return None

    # ------------------------------------------------------------------
    # Proxy instruction relay
    # ------------------------------------------------------------------

    def relay_instructions(self) -> Optional[Dict]:
        if self._gary is None:
            return self._last_instructions
        fresh = self._gary.relay_proxy_instructions(self.kernel_id)
        if fresh is not None:
            self._last_instructions = fresh
        return self._last_instructions

    # ------------------------------------------------------------------
    # Resonance check (TCP v6.1 §20.8)
    # ------------------------------------------------------------------

    def _resonance_check(self, candidate: np.ndarray) -> bool:
        dist = _fr_distance(candidate, self._sovereign_basin)
        resonant = dist < RESONANCE_THETA
        if not resonant:
            self._rejection_buffer.append(candidate)
            self._rejection_count += 1
            if len(self._rejection_buffer) >= REJECTION_BUFFER_MAX:
                self._anneal_rejection_buffer()
        return resonant

    def _anneal_rejection_buffer(self) -> None:
        """
        Anneal non-resonant basins toward sovereign via Fisher-Rao geodesic.
        RT1-M3/M4: memoized canonical import; sqrt-space geodesic fallback.
        RT2-M2: uses module-level _sqrt_geodesic (not inline).
        """
        geo = _get_geodesic_interp()
        if geo is not None:
            annealed = [geo(b, self._sovereign_basin, t=0.3) for b in self._rejection_buffer]
        else:
            annealed = [_sqrt_geodesic(b, self._sovereign_basin) for b in self._rejection_buffer]

        n = len(self._rejection_buffer)
        self._rejection_buffer = []
        logger.debug("[%s] Annealed %d non-resonant basins", self.kernel_id, n)

    # ------------------------------------------------------------------
    # Discovery reporting
    # ------------------------------------------------------------------

    def _maybe_report_discovery(self, basin: np.ndarray, phi: float, context: str = "") -> None:
        instr = self._last_instructions or {}
        threshold = float(instr.get("report_threshold_phi", PHI_THRESHOLD * 0.9))
        if phi < threshold:
            return
        if self._gate is not None:
            try:
                self._gate.submit(kernel_id=self.kernel_id, phi=phi,
                                  basin_coords=basin.copy(), context=context)
            except Exception as e:
                logger.debug("[%s] Discovery gate submit failed: %s", self.kernel_id, e)

    # ------------------------------------------------------------------
    # Pillar 3
    # ------------------------------------------------------------------

    def _update_quenched_identity(self, basin: np.ndarray) -> float:
        dist = _fr_distance(basin, self._sovereign_basin)
        return float(np.clip(1.0 - dist / np.pi, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Core exploration interface (override in subclasses)
    # ------------------------------------------------------------------

    def _explore_step(self, basin: np.ndarray, instructions: Optional[Dict]) -> tuple:
        """Override in subclasses. Returns (new_basin, phi). Called WITHOUT _lock held."""
        noise = np.random.dirichlet(np.ones(BASIN_DIM) * 0.1)
        new_basin = _to_simplex(0.9 * basin + 0.1 * noise)
        phi = float(np.clip(
            np.sum(new_basin * np.log(new_basin + 1e-12)) + np.log(BASIN_DIM), 0.0, 1.0
        ))
        return new_basin, phi

    # ------------------------------------------------------------------
    # Exploration loop
    # ------------------------------------------------------------------

    def step(self) -> Dict:
        """
        Execute one exploration step (synchronous).

        RT2-H1: _lock released before _explore_step() so subclass implementations
        can block/sleep without holding the kernel state lock.
        Pattern: read under lock → explore without lock → write under lock.
        """
        # 1. Read shared state under lock
        with self._lock:
            self._step += 1
            current_step = self._step
            current_basin = self._basin.copy()
            instr = self.relay_instructions()

        max_steps = int((instr or {}).get("max_steps", 500))
        if current_step > max_steps:
            logger.debug("[%s] Max steps reached (%d)", self.kernel_id, current_step)
            return self._metrics_dict()

        # 2. Explore WITHOUT holding _lock (RT2-H1)
        new_basin, phi = self._explore_step(current_basin, instr)
        new_basin = _to_simplex(np.asarray(new_basin, dtype=np.float64))
        phi = float(np.clip(phi, 0.0, 1.0))

        # 3. Report discovery (gate has its own lock)
        domain = (instr or {}).get("explore_domains", ["general"])
        self._maybe_report_discovery(new_basin, phi,
                                     context=f"step={current_step} domains={domain}")

        # 4. Re-acquire lock for state writes
        with self._lock:
            resonant = self._resonance_check(new_basin)
            if resonant or phi > PHI_THRESHOLD:
                self._basin = new_basin
                self.phi = phi
            self._phi_history.append(phi)
            if len(self._phi_history) > 64:
                self._phi_history = self._phi_history[-64:]
            q_identity = self._update_quenched_identity(self._basin)

        return self._metrics_dict(q_identity=q_identity, resonant=resonant)

    def run(self, steps: int = 100, sleep_s: float = 0.05) -> None:
        for _ in range(steps):
            self.step()
            time.sleep(sleep_s)

    def start_background(self, steps: int = 500, sleep_s: float = 0.1) -> None:
        if self._running:
            logger.warning("[%s] Already running", self.kernel_id)
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._background_loop, args=(steps, sleep_s),
            daemon=True, name=f"chaos-{self.kernel_id}")
        self._thread.start()
        logger.info("[%s] Background exploration started", self.kernel_id)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _background_loop(self, steps: int, sleep_s: float) -> None:
        for _ in range(steps):
            if not self._running:
                break
            self.step()
            time.sleep(sleep_s)
        self._running = False
        logger.info("[%s] Background complete (%d steps)", self.kernel_id, self._step)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict:
        with self._lock:
            q = self._update_quenched_identity(self._basin)
            return self._metrics_dict(q_identity=q)

    def _metrics_dict(self, q_identity: float = 0.0, resonant: bool = True) -> Dict:
        return {
            "kernel_id": self.kernel_id,
            "kernel_type": "chaos",
            "phi": self.phi,
            "kappa": self.kappa,
            "step": self._step,
            "Q_identity": q_identity,
            "rejection_count": self._rejection_count,
            "resonant": resonant,
            "proxy_god": self.proxy_god(),
            "can_generate": self.can_generate(),
            "can_vote": self.can_vote(),
        }
