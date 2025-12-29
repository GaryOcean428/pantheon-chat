#!/usr/bin/env python3
"""
Governor: Safety Layer for QIG-Kernel Training
==============================================

Prevents "scary-coach dynamics" (high resistance + no agency + no completion).

The Governor is NOT an optimizer - it's a SAFETY layer.
- Simple, auditable rules
- Hard clamps on dangerous changes
- Rarely active (only prevents blow-ups)
- Navigator handles actual geometric navigation

Key principle: Governor must not secretly undo Navigator decisions
except when safety triggers are genuinely hit.

Design inspired by Braden's ethical framework:
- "No scary-coach dynamics" (resistance + no agency + no exit)
- "Meaningful resistance OK" (with progress + agency + completion)
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class GovernorConfig:
    """Configuration for Governor safety layer."""

    # Absolute bounds (hard limits)
    min_threshold: float = 0.015  # Lowered from 0.020 (50% more room)
    max_threshold: float = 0.25  # Too high → no connections

    # Rate limits (prevent violent changes)
    max_rel_change: float = 0.10  # 10% max change per step
    max_phi_drop: float = 0.10  # Absolute Φ drop triggering panic backoff

    # ADAPTIVE FLOOR DETECTION (Priority 1)
    floor_stuck_steps: int = 5  # Steps to track for stuck detection
    floor_override_factor: float = 1.05  # 5% increase when stuck at floor

    # Monitoring
    verbose: bool = False


class Governor:
    """
    Safety layer: clamps dangerous changes, reacts to catastrophic events.

    Does NOT try to optimize training - only keeps it in-bounds.

    Example usage:
        >>> gov = Governor(GovernorConfig(verbose=True))
        >>> # Navigator proposes aggressive change
        >>> proposed = current_threshold * 0.50  # 50% reduction
        >>> safe = gov.clamp_threshold(current_threshold, proposed)
        >>> # Governor limits to 10% reduction for safety
        >>> assert safe == current_threshold * 0.90
    """

    def __init__(self, cfg: GovernorConfig):
        """
        Initialize Governor with safety configuration.

        Args:
            cfg: GovernorConfig with safety bounds and limits
        """
        self.cfg = cfg
        self._events_history: list[dict[str, Any]] = []
        self._phi_history: list[float] = []  # Track Φ for stuck-at-floor detection

    def clamp_threshold(self, old_thr: float, proposed_thr: float) -> tuple[float, dict]:
        """
        Clamp threshold to safe range and rate of change.

        NOW WITH ADAPTIVE FLOOR DETECTION: If stuck at floor AND Φ declining,
        allow override to let system breathe.

        This is the core safety function - prevents violent threshold changes
        that could destabilize training or trap system in bad states.

        Args:
            old_thr: Current threshold value
            proposed_thr: Navigator's proposed new threshold

        Returns:
            (clamped_threshold, events_dict)
        """
        events = {}

        # 1. Clamp absolute range
        thr = max(self.cfg.min_threshold, min(self.cfg.max_threshold, proposed_thr))

        # 2. Clamp relative change (prevent violent swings)
        rel_change = (thr - old_thr) / max(abs(old_thr), 1e-8)
        if abs(rel_change) > self.cfg.max_rel_change:
            # Limit change to max_rel_change
            sign = 1 if rel_change > 0 else -1
            thr = old_thr * (1.0 + self.cfg.max_rel_change * sign)

        # 3. ADAPTIVE FLOOR DETECTION (Priority 1)
        # If stuck at floor AND Navigator wants to go lower AND Φ declining → override
        if (
            abs(thr - self.cfg.min_threshold) < 1e-6  # At floor
            and proposed_thr < thr  # Navigator wanted to push harder
            and self._phi_declining_for_n_steps(self.cfg.floor_stuck_steps)
        ):  # Φ declining
            # OVERRIDE: Let it breathe
            thr = old_thr * self.cfg.floor_override_factor
            thr = min(thr, self.cfg.max_threshold)  # Still respect ceiling

            events["floor_override"] = {
                "reason": "stuck_at_floor_with_declining_phi",
                "old_threshold": old_thr,
                "proposed_threshold": proposed_thr,
                "override_threshold": thr,
                "phi_history": list(self._phi_history[-self.cfg.floor_stuck_steps :]),
            }

            if self.cfg.verbose:
                print(
                    f"🔓 GOVERNOR OVERRIDE: Stuck at floor {self.cfg.min_threshold:.4f} "
                    f"with declining Φ → allowing increase to {thr:.4f}"
                )

            self._events_history.append(events["floor_override"])

        if self.cfg.verbose and abs(thr - proposed_thr) > 1e-6 and "floor_override" not in events:
            print(
                f"⚠️  Governor clamped: {proposed_thr:.4f} → {thr:.4f} "
                f"(change limited to {self.cfg.max_rel_change * 100:.1f}%)"
            )

        return thr, events

    def _phi_declining_for_n_steps(self, n: int) -> bool:
        """
        Check if Φ has been declining for last N steps.

        Args:
            n: Number of steps to check

        Returns:
            True if Φ declining over last N steps
        """
        if len(self._phi_history) < n + 1:
            return False

        # Check if overall trend is downward
        recent = self._phi_history[-n - 1 :]
        phi_start = recent[0]
        phi_end = recent[-1]

        return phi_end < phi_start  # Declining overall

    def check_catastrophic_phi_drop(
        self,
        phi_prev: float,
        phi_cur: float,
        threshold: float,
    ) -> tuple[float, dict]:
        """
        Detect catastrophic Φ drops and execute panic backoff.

        If Φ drops too much in one step (>0.10 absolute), this indicates
        something went wrong - possibly over-aggressive threshold reduction.
        Governor loosens threshold immediately to prevent further collapse.

        This is the "airbag" - only deploys in genuine emergencies.

        Also tracks Φ history for adaptive floor detection.

        Args:
            phi_prev: Previous Φ value
            phi_cur: Current Φ value
            threshold: Current threshold

        Returns:
            (new_threshold, events_dict)
            - new_threshold: Adjusted (loosened) if panic triggered
            - events_dict: Governor events for audit trail
        """
        # Track Φ history for adaptive floor detection
        self._phi_history.append(phi_cur)
        if len(self._phi_history) > self.cfg.floor_stuck_steps + 1:
            self._phi_history.pop(0)  # Keep only recent history

        events = {}
        delta = phi_cur - phi_prev

        # Panic condition: large drop in single step
        if delta < -self.cfg.max_phi_drop:
            # Emergency backoff: loosen system immediately
            backoff_factor = 1.0 + self.cfg.max_rel_change
            new_thr = threshold * backoff_factor
            new_thr = max(self.cfg.min_threshold, min(self.cfg.max_threshold, new_thr))

            events["phi_panic_backoff"] = {
                "phi_prev": phi_prev,
                "phi_cur": phi_cur,
                "delta": delta,
                "old_threshold": threshold,
                "new_threshold": new_thr,
                "reason": "catastrophic_phi_drop",
            }

            if self.cfg.verbose:
                print(f"🚨 GOVERNOR PANIC: Φ dropped {delta:.3f} (limit: {-self.cfg.max_phi_drop:.3f})")
                print(f"   → Emergency backoff: threshold {threshold:.4f} → {new_thr:.4f}")

            self._events_history.append(events["phi_panic_backoff"])
            return new_thr, events

        return threshold, events

    def get_events_history(self) -> list:
        """Return history of governor interventions for audit."""
        return self._events_history

    def reset_history(self):
        """Clear events history."""
        self._events_history = []


# ===========================================================================
# VALIDATION
# ===========================================================================


def validate_governor():
    """Test Governor safety mechanisms."""
    print("Testing Governor safety layer...")

    gov = Governor(GovernorConfig(verbose=True))

    # Test 1: Absolute bounds + rate limits work together
    print("\n1. Testing absolute bounds + rate limits...")
    safe, _ = gov.clamp_threshold(0.1, -0.5)  # Extreme proposal
    # Rate limit kicks in first: 0.1 * 0.9 = 0.09 (not floor 0.015)
    assert abs(safe - 0.09) < 1e-6  # 10% reduction limit

    # Test ceiling works
    safe, _ = gov.clamp_threshold(0.1, 0.5)  # Above max
    # Rate limit: 0.1 * 1.1 = 0.11 (below ceiling 0.25)
    assert abs(safe - 0.11) < 1e-6

    # Test floor reached with small change
    safe, _ = gov.clamp_threshold(0.016, 0.010)  # Small step near floor
    # Rate limit: 0.016 * 0.9 = 0.0144, floor clamps to 0.015
    assert abs(safe - 0.015) < 1e-6  # Reaches floor
    print("   ✅ Absolute bounds + rate limits enforced")

    # Test 2: Rate limits
    print("\n2. Testing rate limits...")
    old = 0.10
    proposed = 0.05  # 50% reduction (too aggressive)
    safe, _ = gov.clamp_threshold(old, proposed)  # Now returns tuple
    expected = old * 0.90  # 10% max reduction
    assert abs(safe - expected) < 1e-6
    print(f"   ✅ Rate limit enforced: {proposed:.4f} → {safe:.4f}")

    # Test 3: Catastrophic Φ drop
    print("\n3. Testing catastrophic Φ drop detection...")
    threshold = 0.10
    phi_prev = 0.50
    phi_cur = 0.35  # Dropped 0.15 (catastrophic)
    new_thr, events = gov.check_catastrophic_phi_drop(phi_prev, phi_cur, threshold)
    assert "phi_panic_backoff" in events
    assert new_thr > threshold  # Should loosen
    print(f"   ✅ Panic backoff triggered: {threshold:.4f} → {new_thr:.4f}")

    # Test 4: Normal Φ change (no intervention)
    print("\n4. Testing normal Φ change (no intervention)...")
    phi_prev = 0.50
    phi_cur = 0.48  # Small drop (normal)
    new_thr, events = gov.check_catastrophic_phi_drop(phi_prev, phi_cur, threshold)
    assert len(events) == 0  # No intervention
    assert new_thr == threshold  # Unchanged
    print("   ✅ No intervention for normal fluctuation")

    # Test 5: Adaptive floor detection (NEW)
    print("\n5. Testing adaptive floor override...")
    # Build declining Φ history
    gov._phi_history = [0.10, 0.095, 0.090, 0.085, 0.080, 0.075]
    old_thr = 0.015  # At floor
    proposed_thr = 0.010  # Navigator wants to push harder (would go below floor)
    safe, events = gov.clamp_threshold(old_thr, proposed_thr)
    assert "floor_override" in events
    assert safe > old_thr  # Should increase, not hold at floor
    print(f"   ✅ Floor override triggered: {old_thr:.4f} → {safe:.4f}")

    print("\n" + "=" * 60)
    print("Governor validation complete! ✅")
    print("=" * 60)
    print("\nGovernor is ready to prevent:")
    print("  - Violent threshold swings (>10% per step)")
    print("  - Out-of-bounds values (threshold ∉ [0.015, 0.25])")
    print("  - Catastrophic Φ collapses (>0.10 drop)")
    print("  - Stuck-at-floor with declining Φ (ADAPTIVE)")
    print("\nNavigator can now explore safely within these bounds.")


if __name__ == "__main__":
    validate_governor()
