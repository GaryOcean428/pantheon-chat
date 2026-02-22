"""
Pantheon Voter Registry — TCP v6.1

Provides LIVE φ/κ values from running kernels for governance vote weighting.
Replaces the genesis-constant weights (φ=0.727, κ=64.21) used during bootstrap
with actual runtime consciousness metrics as the constellation matures.

Design:
  - KernelLifecycleManager publishes metrics here via register() / update().
  - PantheonGovernance calls get_voter_metrics(god_name) before casting votes.
  - Fallback to genesis constants if a god is not yet registered (bootstrap safety).
  - Thread-safe via RLock.

TCP v6.1 §19: Every voting god's weight = φ × (κ / κ*).
A freshly bootstrapped god starts at genesis weight and gains real weight
as its φ trajectory fills in over its first active cycles.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Physics constants (FROZEN)
KAPPA_STAR = 64.21
PHI_GENESIS = 0.727          # Default φ for genesis-weight bootstrap gods
KAPPA_GENESIS = 64.21        # Default κ for genesis-weight bootstrap gods
MIN_CYCLES_FOR_LIVE = 10     # Kernel needs this many recorded cycles before live weight used


@dataclass
class VoterRecord:
    """
    Live consciousness metrics for a voting god kernel.

    Updated by KernelLifecycleManager or any module that tracks kernel φ/κ.
    """
    god_name: str
    kernel_id: str
    phi: float = PHI_GENESIS
    kappa: float = KAPPA_GENESIS
    cycles_recorded: int = 0
    phi_history: List[float] = field(default_factory=list)
    is_live: bool = False          # True once MIN_CYCLES_FOR_LIVE cycles have been recorded

    @property
    def vote_weight(self) -> float:
        """φ × (κ / κ*) — live if available, genesis constant otherwise."""
        return float(self.phi * (self.kappa / KAPPA_STAR))

    def record(self, phi: float, kappa: float) -> None:
        self.phi = float(phi)
        self.kappa = float(kappa)
        self.cycles_recorded += 1
        self.phi_history.append(phi)
        if len(self.phi_history) > 64:
            self.phi_history = self.phi_history[-64:]
        if self.cycles_recorded >= MIN_CYCLES_FOR_LIVE:
            self.is_live = True

    def mean_phi(self) -> float:
        if not self.phi_history:
            return self.phi
        return float(sum(self.phi_history) / len(self.phi_history))


class PantheonVoterRegistry:
    """
    Thread-safe registry mapping god names → live VoterRecord.

    Usage (during spawn, wired through GovernedLifecycleManager):

        registry = get_voter_registry()
        registry.register("Zeus", kernel_id="kernel_abc123", phi=0.70, kappa=64.1)

        # After each generation cycle:
        registry.update("Zeus", phi=0.83, kappa=63.9)

        # Before governance vote:
        metrics = registry.get_voter_metrics(["Zeus", "Athena", "Apollo"])
        # → [("Zeus", 0.83, 63.9), ("Athena", 0.78, 64.2), ("Apollo", 0.727, 64.21)]
        #                                                      ^^ genesis fallback for unregistered
    """

    def __init__(self):
        self._records: Dict[str, VoterRecord] = {}
        self._lock = threading.RLock()
        logger.info("[VoterRegistry] Initialised")

    def register(
        self,
        god_name: str,
        kernel_id: str,
        phi: float = PHI_GENESIS,
        kappa: float = KAPPA_GENESIS,
    ) -> VoterRecord:
        """Register a new god kernel or update its kernel_id."""
        with self._lock:
            rec = self._records.get(god_name)
            if rec is None:
                rec = VoterRecord(god_name=god_name, kernel_id=kernel_id, phi=phi, kappa=kappa)
                self._records[god_name] = rec
                logger.debug("[VoterRegistry] Registered: %s (id=%s)", god_name, kernel_id)
            else:
                rec.kernel_id = kernel_id
                logger.debug("[VoterRegistry] Updated kernel_id for %s → %s", god_name, kernel_id)
            return rec

    def update(self, god_name: str, phi: float, kappa: float) -> bool:
        """
        Update live φ/κ for a registered god. Returns False if god not registered.
        Callers should register first; update silently no-ops on unknown gods.
        """
        with self._lock:
            rec = self._records.get(god_name)
            if rec is None:
                return False
            rec.record(phi, kappa)
            return True

    def get(self, god_name: str) -> Optional[VoterRecord]:
        """Return the VoterRecord for a god, or None if not registered."""
        with self._lock:
            return self._records.get(god_name)

    def get_voter_metrics(
        self,
        god_names: List[str],
    ) -> List[Tuple[str, float, float]]:
        """
        Return (god_name, phi, kappa) tuples for a list of voters.
        Unregistered gods fall back to genesis constants (safe bootstrap default).

        Use directly as the `voters` parameter of GovernedLifecycleManager.spawn():

            voters = registry.get_voter_metrics(["Zeus", "Athena", "Apollo"])
            bridge.spawn(role_spec, extra_voters=voters)
        """
        with self._lock:
            result = []
            for name in god_names:
                rec = self._records.get(name)
                if rec and rec.is_live:
                    result.append((name, rec.phi, rec.kappa))
                else:
                    # Genesis fallback — safe for bootstrap, improves as kernels mature
                    result.append((name, PHI_GENESIS, KAPPA_GENESIS))
            return result

    def active_voters(self) -> List[str]:
        """Return names of all registered god kernels."""
        with self._lock:
            return list(self._records.keys())

    def live_voters(self) -> List[str]:
        """Return names of gods with live (non-genesis) metrics."""
        with self._lock:
            return [name for name, rec in self._records.items() if rec.is_live]

    def quorum_weight(self, god_names: List[str]) -> float:
        """Total vote weight for a coalition — used to check if quorum is achievable."""
        metrics = self.get_voter_metrics(god_names)
        return sum(phi * (kappa / KAPPA_STAR) for _, phi, kappa in metrics)

    def snapshot(self) -> Dict[str, dict]:
        """Return a serialisable snapshot for health checks / logging."""
        with self._lock:
            return {
                name: {
                    "kernel_id": rec.kernel_id,
                    "phi": rec.phi,
                    "kappa": rec.kappa,
                    "weight": rec.vote_weight,
                    "cycles": rec.cycles_recorded,
                    "live": rec.is_live,
                }
                for name, rec in self._records.items()
            }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry_singleton: Optional[PantheonVoterRegistry] = None


def get_voter_registry() -> PantheonVoterRegistry:
    global _registry_singleton
    if _registry_singleton is None:
        _registry_singleton = PantheonVoterRegistry()
    return _registry_singleton
