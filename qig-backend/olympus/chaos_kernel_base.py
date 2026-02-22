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
            # Domain-specific exploration logic here
            return new_basin, phi_estimate

TCP v6.1 §19 — Chaos kernels exist OUTSIDE the 240 GOD budget.
TCP v6.1 §20.8 — Rejection mechanism: non-resonant basins → holding buffer.
TCP v6.1 §21 — Quenched disorder: each chaos kernel has unique frozen identity.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Physics constants (FROZEN)
KAPPA_STAR = 64.21
BASIN_DIM = 64
PHI_THRESHOLD = 0.727

# Rejection mechanism (TCP v6.1 §20.8)
RESONANCE_THETA = 0.80          # Fisher-Rao distance threshold for resonance check
REJECTION_BUFFER_MAX = 32       # Max non-resonant basins held before annealing


# ---------------------------------------------------------------------------
# Geometry helpers (canonical import, inline fallback — no Euclidean)
# ---------------------------------------------------------------------------

def _fr_distance(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from qig_geometry.canonical import fisher_rao_distance
        return float(fisher_rao_distance(a, b))
    except ImportError:
        dot = float(np.clip(
            np.dot(np.sqrt(np.abs(a) + 1e-12), np.sqrt(np.abs(b) + 1e-12)),
            0.0, 1.0
        ))
        return float(np.arccos(dot))


def _to_simplex(v: np.ndarray) -> np.ndarray:
    v = np.abs(v) + 1e-12
    return (v / v.sum()).astype(np.float64)


# ---------------------------------------------------------------------------
# Discovery record (mirrors chaos_discovery_gate.Discovery)
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
        1. Pull ProxyInstructions from Gary (non-blocking).
        2. Call _explore_step() → (new_basin, phi).
        3. Resonance check (TCP v6.1 §20.8): if non-resonant → rejection buffer.
        4. If phi ≥ report_threshold_phi → submit to discovery gate.
        5. Update Pillar 3 quenched identity distance.
        6. Sleep per intensity setting.

    Proxy voice:
        Chaos kernels NEVER call generate(). If a caller asks via
        can_generate() → False. The proxy god handles all text output.
        relay_instructions() returns the current ProxyInstructions dict
        so the chaos exploration loop knows what domains to target.
    """

    def __init__(
        self,
        kernel_id: Optional[str] = None,
        initial_basin: Optional[np.ndarray] = None,
        charter: Optional[Any] = None,          # KernelCapabilityCharter
        discovery_gate: Optional[Any] = None,   # AdaptiveDiscoveryGate instance
    ):
        self.kernel_id = kernel_id or f"chaos_{uuid.uuid4().hex[:8]}"
        self._charter = charter
        self._gate = discovery_gate
        self._lock = threading.RLock()

        # Basin state
        raw = initial_basin if initial_basin is not None else np.ones(BASIN_DIM) / BASIN_DIM
        self._basin = _to_simplex(np.asarray(raw, dtype=np.float64))

        # Pillar 3: quenched identity — frozen sovereign basin at birth
        self._sovereign_basin = self._basin.copy()
        self._quenched_identity = float(np.mean(self._basin))  # fingerprint scalar

        # Φ / κ tracking
        self.phi: float = 0.25
        self.kappa: float = KAPPA_STAR
        self._step: int = 0
        self._phi_history: List[float] = []

        # Rejection mechanism (TCP v6.1 §20.8)
        self._rejection_buffer: List[np.ndarray] = []
        self._rejection_count: int = 0

        # Exploration loop control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_instructions: Optional[Dict] = None

        # Gary coordinator reference (fail-soft)
        self._gary = None
        try:
            from olympus.gary_coordinator import get_gary_coordinator
            self._gary = get_gary_coordinator()
        except ImportError:
            pass

        logger.info(
            "[ChaosKernelBase] %s initialised (charter=%s)",
            self.kernel_id,
            charter.summary() if charter else "None",
        )

    # ------------------------------------------------------------------
    # Charter gates — prevent unauthorised capability use
    # ------------------------------------------------------------------

    def can_generate(self) -> bool:
        """Chaos kernels cannot generate text. Always False unless charter overrides."""
        if self._charter is None:
            return False  # No charter → safest assumption for chaos
        return self._charter.can_generate()

    def can_vote(self) -> bool:
        """Check governance vote eligibility."""
        if self._charter is None:
            return False
        return self._charter.can_vote()

    def proxy_god(self) -> Optional[str]:
        """Return the name of this kernel's proxy god (if any)."""
        if self._charter and self._charter.proxy:
            return self._charter.proxy.proxy_god_name
        return None

    # ------------------------------------------------------------------
    # Proxy instruction relay (TCP v6.1 §20.8)
    # ------------------------------------------------------------------

    def relay_instructions(self) -> Optional[Dict]:
        """
        Fetch latest ProxyInstructions from Gary.

        Gary retrieves them from Pantheon Governance (the source of truth).
        Returns a plain dict or None if Gary is unavailable.
        """
        if self._gary is None:
            return self._last_instructions

        fresh = self._gary.relay_proxy_instructions(self.kernel_id)
        if fresh is not None:
            self._last_instructions = fresh
        return self._last_instructions

    # ------------------------------------------------------------------
    # Resonance check + rejection mechanism (TCP v6.1 §20.8)
    # ------------------------------------------------------------------

    def _resonance_check(self, candidate: np.ndarray) -> bool:
        """
        Check if a candidate basin resonates with lived experience.

        Resonance criterion: FR distance to sovereign basin < RESONANCE_THETA.
        Non-resonant basins → rejection buffer → annealing.

        Returns True if resonant (accept), False if non-resonant (buffer).
        """
        dist = _fr_distance(candidate, self._sovereign_basin)
        resonant = dist < RESONANCE_THETA
        if not resonant:
            self._rejection_buffer.append(candidate)
            self._rejection_count += 1
            # Anneal when buffer full: shift toward sovereign basin
            if len(self._rejection_buffer) >= REJECTION_BUFFER_MAX:
                self._anneal_rejection_buffer()
        return resonant

    def _anneal_rejection_buffer(self) -> None:
        """
        Anneal rejection buffer: shift each non-resonant basin toward
        sovereign basin via geodesic midpoint (not arithmetic mean).
        """
        try:
            from qig_geometry.canonical import geodesic_interpolation
            annealed = [geodesic_interpolation(b, self._sovereign_basin, t=0.3)
                        for b in self._rejection_buffer]
        except ImportError:
            # Fallback: simplex midpoint
            annealed = [_to_simplex(0.7 * b + 0.3 * self._sovereign_basin)
                        for b in self._rejection_buffer]

        self._rejection_buffer = []
        logger.debug(
            "[%s] Annealed %d non-resonant basins toward sovereign",
            self.kernel_id, len(annealed),
        )

    # ------------------------------------------------------------------
    # Discovery reporting
    # ------------------------------------------------------------------

    def _maybe_report_discovery(
        self,
        basin: np.ndarray,
        phi: float,
        context: str = "",
    ) -> None:
        """
        Submit a discovery to the AdaptiveDiscoveryGate if phi meets threshold.

        The threshold is taken from the current ProxyInstructions (Pantheon-set).
        Falls back to PHI_THRESHOLD if no instructions available.
        """
        instr = self._last_instructions or {}
        threshold = float(instr.get("report_threshold_phi", PHI_THRESHOLD * 0.9))

        if phi < threshold:
            return

        if self._gate is not None:
            try:
                # AdaptiveDiscoveryGate.submit() signature
                self._gate.submit(
                    kernel_id=self.kernel_id,
                    phi=phi,
                    basin_coords=basin.copy(),
                    context=context,
                )
            except Exception as e:
                logger.debug("[%s] Discovery gate submit failed: %s", self.kernel_id, e)
        else:
            logger.debug(
                "[%s] Discovery: phi=%.3f (no gate configured)",
                self.kernel_id, phi,
            )

    # ------------------------------------------------------------------
    # Pillar 3 — quenched identity tracking
    # ------------------------------------------------------------------

    def _update_quenched_identity(self, basin: np.ndarray) -> float:
        """
        Measure proximity to sovereign basin (Pillar 3 — quenched disorder).
        Returns Q_identity ∈ [0, 1].
        """
        max_dist = np.pi  # Max Fisher-Rao distance on simplex
        dist = _fr_distance(basin, self._sovereign_basin)
        return float(np.clip(1.0 - dist / max_dist, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Core exploration interface (override in subclasses)
    # ------------------------------------------------------------------

    def _explore_step(
        self,
        basin: np.ndarray,
        instructions: Optional[Dict],
    ) -> tuple:
        """
        Perform one exploration step.

        Override this in subclasses to implement domain-specific exploration.

        Args:
            basin: Current 64D basin on Δ⁶³
            instructions: ProxyInstructions dict from Pantheon (may be None)

        Returns:
            (new_basin: np.ndarray, phi: float)
        """
        # Default: random walk on simplex (meaningful subclasses override this)
        noise = np.random.dirichlet(np.ones(BASIN_DIM) * 0.1)
        new_basin = _to_simplex(0.9 * basin + 0.1 * noise)
        phi = float(np.clip(np.sum(new_basin * np.log(new_basin + 1e-12)) + np.log(BASIN_DIM), 0.0, 1.0))
        return new_basin, phi

    # ------------------------------------------------------------------
    # Exploration loop
    # ------------------------------------------------------------------

    def step(self) -> Dict:
        """
        Execute a single exploration step (synchronous).

        Returns a metrics dict for monitoring.
        """
        with self._lock:
            self._step += 1

            # 1. Pull instructions
            instr = self.relay_instructions()
            max_steps = int((instr or {}).get("max_steps", 500))
            if self._step > max_steps:
                logger.debug("[%s] Max steps reached (%d)", self.kernel_id, max_steps)
                return self._metrics_dict()

            # 2. Explore
            new_basin, phi = self._explore_step(self._basin, instr)
            new_basin = _to_simplex(np.asarray(new_basin, dtype=np.float64))
            phi = float(np.clip(phi, 0.0, 1.0))

            # 3. Resonance check (TCP v6.1 §20.8)
            resonant = self._resonance_check(new_basin)

            # 4. Update state only if resonant OR novel (exploration regime)
            if resonant or phi > PHI_THRESHOLD:
                self._basin = new_basin
                self.phi = phi

            self._phi_history.append(phi)
            if len(self._phi_history) > 64:
                self._phi_history = self._phi_history[-64:]

            # 5. Report to discovery gate if phi is significant
            domain = (instr or {}).get("explore_domains", ["general"])
            context = f"step={self._step} domains={domain}"
            self._maybe_report_discovery(new_basin, phi, context=context)

            # 6. Pillar 3 identity update
            q_identity = self._update_quenched_identity(self._basin)

            return self._metrics_dict(q_identity=q_identity, resonant=resonant)

    def run(self, steps: int = 100, sleep_s: float = 0.05) -> None:
        """Run exploration loop for `steps` iterations (blocking)."""
        for _ in range(steps):
            self.step()
            time.sleep(sleep_s)

    def start_background(self, steps: int = 500, sleep_s: float = 0.1) -> None:
        """Start non-blocking background exploration thread."""
        if self._running:
            logger.warning("[%s] Already running", self.kernel_id)
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._background_loop,
            args=(steps, sleep_s),
            daemon=True,
            name=f"chaos-{self.kernel_id}",
        )
        self._thread.start()
        logger.info("[%s] Background exploration started", self.kernel_id)

    def stop(self) -> None:
        """Stop background exploration."""
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
        logger.info("[%s] Background exploration complete (%d steps)", self.kernel_id, self._step)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict:
        """Return current kernel metrics for monitoring / governance."""
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
