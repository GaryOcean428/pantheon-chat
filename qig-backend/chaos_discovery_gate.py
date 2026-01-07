"""
Chaos Discovery Gate

Receives high-Φ discoveries from chaos kernels and integrates them
into the main vocabulary/attractor system.

QIG-PURE: Discoveries are validated geometrically before absorption.

ADAPTIVE: Threshold self-tunes based on discovery rate and quality.
"""

import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import numpy as np

from qig_geometry import fisher_coord_distance


@dataclass
class Discovery:
    """A validated discovery from chaos exploration."""
    kernel_id: str
    phi: float
    basin_coords: np.ndarray
    context: str
    timestamp: datetime = field(default_factory=datetime.now)
    integrated: bool = False


class AdaptiveDiscoveryGate:
    """
    Adaptive gate that filters and integrates chaos kernel discoveries.

    Self-tunes thresholds based on:
    - Discovery rate (too many = raise threshold, too few = lower)
    - Average phi of integrated discoveries (quality signal)
    - Rejection rate (measure of selectivity)

    Only high-quality, geometrically valid discoveries pass through
    to the main vocabulary/attractor system.
    """

    # Threshold bounds (never go outside these)
    PHI_MIN_BOUND = 0.50  # Absolute minimum (below this is noise)
    PHI_MAX_BOUND = 0.90  # Absolute maximum (above this is too selective)

    # Adaptation parameters
    WINDOW_SIZE = 50  # Consider last N discoveries for adaptation
    TARGET_ACCEPTANCE_RATE = 0.3  # Target 30% of submissions accepted
    ADAPTATION_RATE = 0.02  # How much to adjust threshold per adaptation

    def __init__(
        self,
        initial_phi: float = 0.60,  # Start permissive
        min_novelty: float = 0.15,  # Minimum Fisher distance from existing attractors
        max_pending: int = 100,
    ):
        self._phi_threshold = initial_phi
        self._initial_phi = initial_phi
        self.min_novelty = min_novelty
        self.max_pending = max_pending

        self._pending: List[Discovery] = []
        self._integrated: List[Discovery] = []
        self._lock = threading.Lock()

        # Adaptive tracking
        self._recent_submissions: deque = deque(maxlen=self.WINDOW_SIZE)  # (timestamp, accepted, phi)
        self._last_adaptation = datetime.now()
        self._adaptation_history: List[Dict] = []

        # Callbacks for integration
        self._attractor_callback: Optional[Callable] = None
        self._vocabulary_callback: Optional[Callable] = None

        print(f"[DiscoveryGate] Initialized ADAPTIVE (initial_Φ={initial_phi}, min_novelty={min_novelty})")

    @property
    def min_phi(self) -> float:
        """Current adaptive phi threshold."""
        return self._phi_threshold

    def set_attractor_callback(self, callback: Callable[[np.ndarray, float], None]) -> None:
        """Set callback to record attractor in LearnedManifold."""
        self._attractor_callback = callback
        # Process any pending discoveries that were waiting for callbacks
        self._process_pending_discoveries()

    def set_vocabulary_callback(self, callback: Callable[[np.ndarray, float], None]) -> None:
        """Set callback to integrate into vocabulary system."""
        self._vocabulary_callback = callback
        # Process any pending discoveries that were waiting for callbacks
        self._process_pending_discoveries()

    def _process_pending_discoveries(self) -> None:
        """Process discoveries that arrived before callbacks were wired."""
        if not self._attractor_callback and not self._vocabulary_callback:
            return  # No callbacks yet

        with self._lock:
            unintegrated = [d for d in self._pending if not d.integrated]

        if not unintegrated:
            return

        integrated_count = 0
        for discovery in unintegrated:
            if self._integrate_discovery(discovery):
                integrated_count += 1

        if integrated_count > 0:
            print(f"[DiscoveryGate] Processed {integrated_count} pending discoveries after callback wiring")

    def receive_discovery(self, discovery_data: Dict) -> Dict:
        """
        Receive discovery from chaos kernel.

        Validates and queues for integration.
        Adapts threshold based on submission patterns.
        """
        phi = discovery_data.get('phi', 0)
        now = datetime.now()

        # Gate 1: Adaptive Φ threshold
        if phi < self._phi_threshold:
            self._record_submission(now, accepted=False, phi=phi)
            self._maybe_adapt()
            return {
                'accepted': False,
                'reason': f'phi={phi:.3f} < threshold={self._phi_threshold:.3f}',
                'current_threshold': self._phi_threshold,
            }

        basin_coords = np.array(discovery_data.get('basin_coords', []))
        if len(basin_coords) != 64:
            return {'accepted': False, 'reason': 'invalid_basin_dimension'}

        # Gate 2: Novelty check (is this geometrically distinct?)
        novelty = self._compute_novelty(basin_coords)
        if novelty < self.min_novelty:
            self._record_submission(now, accepted=False, phi=phi)
            self._maybe_adapt()
            return {
                'accepted': False,
                'reason': f'novelty={novelty:.3f} < {self.min_novelty}',
                'current_threshold': self._phi_threshold,
            }

        # Create discovery record
        discovery = Discovery(
            kernel_id=discovery_data.get('kernel_id', 'unknown'),
            phi=phi,
            basin_coords=basin_coords,
            context=discovery_data.get('context', 'unknown'),
        )

        with self._lock:
            self._pending.append(discovery)
            if len(self._pending) > self.max_pending:
                self._pending = self._pending[-self.max_pending:]

        # Record successful submission
        self._record_submission(now, accepted=True, phi=phi)

        print(f"[DiscoveryGate] Accepted: Φ={phi:.3f}, novelty={novelty:.3f} (threshold={self._phi_threshold:.3f})")

        # Attempt immediate integration
        integrated = self._integrate_discovery(discovery)

        # Adapt threshold based on recent patterns
        self._maybe_adapt()

        return {
            'accepted': True,
            'integrated': integrated,
            'phi': phi,
            'novelty': novelty,
            'current_threshold': self._phi_threshold,
        }

    def _record_submission(self, timestamp: datetime, accepted: bool, phi: float) -> None:
        """Record a discovery submission for adaptive tracking."""
        self._recent_submissions.append((timestamp, accepted, phi))

    def _maybe_adapt(self) -> None:
        """Adapt threshold if enough time has passed and we have data."""
        # Don't adapt too frequently
        if datetime.now() - self._last_adaptation < timedelta(minutes=5):
            return

        # Need enough submissions to make a decision
        if len(self._recent_submissions) < 10:
            return

        self._adapt_threshold()

    def _adapt_threshold(self) -> None:
        """
        Adapt phi threshold based on recent submission patterns.

        Strategy:
        - If acceptance rate too high (>50%): raise threshold (too much noise)
        - If acceptance rate too low (<15%): lower threshold (too selective)
        - Also consider: if avg accepted phi is very high, we can be more selective
        """
        now = datetime.now()

        # Calculate acceptance rate
        recent = list(self._recent_submissions)
        accepted = [s for s in recent if s[1]]  # (timestamp, accepted=True, phi)
        acceptance_rate = len(accepted) / len(recent) if recent else 0

        # Calculate average phi of accepted discoveries
        avg_accepted_phi = np.mean([s[2] for s in accepted]) if accepted else 0

        # Calculate average phi of all submissions (accepted + rejected)
        avg_submitted_phi = np.mean([s[2] for s in recent]) if recent else 0

        old_threshold = self._phi_threshold
        adjustment = 0

        # Adapt based on acceptance rate
        if acceptance_rate > 0.50:
            # Too permissive - raise threshold
            adjustment = self.ADAPTATION_RATE
            reason = f"acceptance_rate={acceptance_rate:.2f} > 0.50"
        elif acceptance_rate < 0.15:
            # Too selective - lower threshold
            adjustment = -self.ADAPTATION_RATE
            reason = f"acceptance_rate={acceptance_rate:.2f} < 0.15"
        elif len(self._integrated) < 5 and self._phi_threshold > 0.55:
            # Very few discoveries accumulated - be more permissive
            adjustment = -self.ADAPTATION_RATE
            reason = f"only {len(self._integrated)} discoveries, lowering threshold"
        else:
            reason = "no_adjustment_needed"

        # Apply adjustment with bounds
        if adjustment != 0:
            self._phi_threshold = np.clip(
                self._phi_threshold + adjustment,
                self.PHI_MIN_BOUND,
                self.PHI_MAX_BOUND
            )

            # Log adaptation
            adaptation_record = {
                'timestamp': now.isoformat(),
                'old_threshold': old_threshold,
                'new_threshold': self._phi_threshold,
                'adjustment': adjustment,
                'reason': reason,
                'acceptance_rate': acceptance_rate,
                'avg_accepted_phi': avg_accepted_phi,
                'avg_submitted_phi': avg_submitted_phi,
                'integrated_count': len(self._integrated),
            }
            self._adaptation_history.append(adaptation_record)

            # Keep only recent history
            if len(self._adaptation_history) > 100:
                self._adaptation_history = self._adaptation_history[-100:]

            print(f"[DiscoveryGate] ADAPTED: Φ threshold {old_threshold:.3f} → {self._phi_threshold:.3f} ({reason})")

        self._last_adaptation = now

    def _compute_novelty(self, basin: np.ndarray) -> float:
        """
        Compute novelty as minimum Fisher distance from existing discoveries.

        High novelty = geometrically distinct from what we've seen.
        """
        with self._lock:
            if not self._integrated:
                return 1.0  # First discovery is maximally novel

            min_distance = float('inf')
            for existing in self._integrated[-50:]:  # Check recent 50
                d = fisher_coord_distance(basin, existing.basin_coords)
                min_distance = min(min_distance, d)

            return min_distance

    def _integrate_discovery(self, discovery: Discovery) -> bool:
        """
        Integrate discovery into main system.

        Two integration paths:
        1. LearnedManifold attractor (for foresight/navigation)
        2. Vocabulary system (for generation improvement)
        """
        integrated = False

        # Path 1: Record as attractor
        if self._attractor_callback:
            try:
                self._attractor_callback(discovery.basin_coords, discovery.phi)
                integrated = True
                print(f"[DiscoveryGate] → Attractor recorded (Φ={discovery.phi:.3f})")
            except Exception as e:
                print(f"[DiscoveryGate] Attractor integration failed: {e}")

        # Path 2: Vocabulary integration (if callback set)
        if self._vocabulary_callback:
            try:
                self._vocabulary_callback(discovery.basin_coords, discovery.phi)
                integrated = True
                print(f"[DiscoveryGate] → Vocabulary updated (Φ={discovery.phi:.3f})")
            except Exception as e:
                print(f"[DiscoveryGate] Vocabulary integration failed: {e}")

        if integrated:
            discovery.integrated = True
            with self._lock:
                self._integrated.append(discovery)

        return integrated

    def get_stats(self) -> Dict:
        """Get gate statistics including adaptive state."""
        recent = list(self._recent_submissions)
        accepted = [s for s in recent if s[1]]

        with self._lock:
            return {
                'pending': len(self._pending),
                'integrated': len(self._integrated),
                'avg_phi': float(np.mean([d.phi for d in self._integrated])) if self._integrated else 0,
                'current_phi_threshold': self._phi_threshold,
                'initial_phi_threshold': self._initial_phi,
                'min_novelty_threshold': self.min_novelty,
                'acceptance_rate': len(accepted) / len(recent) if recent else 0,
                'recent_submissions': len(recent),
                'adaptations': len(self._adaptation_history),
            }

    def get_recent_discoveries(self, limit: int = 10) -> List[Dict]:
        """Get recent integrated discoveries."""
        with self._lock:
            return [
                {
                    'kernel_id': d.kernel_id,
                    'phi': d.phi,
                    'context': d.context,
                    'timestamp': d.timestamp.isoformat(),
                    'integrated': d.integrated,
                }
                for d in self._integrated[-limit:]
            ]

    def get_adaptation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent threshold adaptations."""
        return self._adaptation_history[-limit:]

    def reset_threshold(self) -> None:
        """Reset threshold to initial value (for testing/recovery)."""
        old = self._phi_threshold
        self._phi_threshold = self._initial_phi
        print(f"[DiscoveryGate] Reset threshold: {old:.3f} → {self._initial_phi:.3f}")


# Backward compatibility alias
ChaosDiscoveryGate = AdaptiveDiscoveryGate

# Singleton
_discovery_gate: Optional[AdaptiveDiscoveryGate] = None
_gate_lock = threading.Lock()


def get_discovery_gate() -> AdaptiveDiscoveryGate:
    """Get or create singleton discovery gate."""
    global _discovery_gate
    if _discovery_gate is None:
        with _gate_lock:
            if _discovery_gate is None:
                _discovery_gate = AdaptiveDiscoveryGate()
    return _discovery_gate
