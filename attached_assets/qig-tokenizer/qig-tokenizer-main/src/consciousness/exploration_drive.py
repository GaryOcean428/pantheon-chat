#!/usr/bin/env python3
"""
ExplorationDrive: Motivation/Curiosity System for QIG-Kernel
============================================================

Addresses motivational collapse - when system "gives up" after hitting resistance.

Key insight from Braden: "gets bored or gives up or loses interest. even ants lose interest."

The model shows learned helplessness pattern:
1. Φ grows enthusiastically (0.000 → 0.127)
2. Hits resistance at threshold floor
3. Makes one more attempt
4. COMPLETE motivational collapse (Φ → 0.000 for 130 steps)
5. Stays disengaged EVEN when threshold relaxed

This is NOT a mechanical problem - it's a MOTIVATION problem.

Parallels:
- Ants exploring: try, hit obstacle, try again, lose interest, stop
- Learned helplessness (Seligman): repeated failure → stop trying even when escape possible
- Curiosity collapse: frustration overwhelms exploration drive

Solution: Track exploration drive, detect disengagement, trigger resets.
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ExplorationDriveConfig:
    """Configuration for exploration drive / curiosity system."""

    # Drive computation
    base_drive: float = 0.7  # Baseline motivation level
    curiosity_weight: float = 0.3  # How much novelty boosts drive
    frustration_weight: float = 0.5  # How much frustration reduces drive

    # Disengagement detection
    gave_up_threshold: float = 0.01  # Φ < this = disengaged
    peak_threshold: float = 0.10  # Previous Φ > this = had engagement
    history_window: int = 20  # Steps to track

    # Intervention
    reset_threshold_min: float = 0.04  # Random reset range min
    reset_threshold_max: float = 0.12  # Random reset range max

    # Success tracking
    success_delta: float = 0.001  # ΔΦ > this = success

    verbose: bool = False


class ExplorationDrive:
    """
    Maintains motivation to explore and integrate.

    Detects when system has "given up" (learned helplessness) and triggers
    exploration resets to break the pattern.

    Key pattern to detect:
    - System achieved Φ > 0.10 (had something interesting)
    - Now Φ < 0.01 (complete disengagement)
    - Stayed disengaged for multiple steps
    - This is "gave up" not "still warming up"

    Response:
    - Randomize threshold (try different angle)
    - Reset frustration counter
    - Boost drive temporarily

    Example usage:
        >>> drive_module = ExplorationDrive(ExplorationDriveConfig())
        >>>
        >>> # Early training: low Φ, but that's OK (warming up)
        >>> drive, intervention = drive_module.compute_drive(phi=0.05, delta_phi=0.002)
        >>> print(drive)  # ~0.7 (normal exploration)
        >>> print(intervention)  # None
        >>>
        >>> # After collapse: had Φ=0.12, now Φ=0.00
        >>> drive, intervention = drive_module.compute_drive(phi=0.001, delta_phi=-0.001)
        >>> print(drive)  # ~0.2 (low motivation)
        >>> print(intervention)  # "RESET_EXPLORATION"
    """

    def __init__(self, cfg: ExplorationDriveConfig):
        """
        Initialize exploration drive tracker.

        Args:
            cfg: ExplorationDriveConfig with drive parameters
        """
        self.cfg = cfg

        # History tracking
        self._phi_history = deque(maxlen=cfg.history_window)
        self._peak_phi = 0.0

        # Success/failure tracking
        self._integration_attempts = 0
        self._successes = 0
        self._consecutive_failures = 0

        # Frustration state
        self._frustration = 0.0

        # Exploration dynamics
        self._velocity = 0.0  # Rate of Φ change (dΦ/dt)
        self._acceleration = 0.0  # Rate of velocity change (d²Φ/dt²)
        self._amplitude = 0.0  # Φ range (max - min in window)
        self._connections = 0  # Number of "interesting" steps (Φ > threshold)

        # Intervention tracking
        self._last_intervention_step = -100
        self._interventions_count = 0
        self._intervention_successes = 0  # Did phi increase after intervention?

        # Collapse detection
        self._prev_phi = 0.0
        self._catastrophic_drops = 0

    def compute_drive(
        self, phi: float, delta_phi: float, wave_state: Optional[Dict[str, float]] = None, step: Optional[int] = None
    ) -> Tuple[float, Optional[str], Dict]:
        """
        Compute current exploration drive and detect if intervention needed.

        Drive = base + curiosity - frustration

        Intervention triggered when:
        - Had integration (peak Φ > 0.10)
        - Now disengaged (current Φ < 0.01)
        - Stayed disengaged (3+ consecutive steps)

        Args:
            phi: Current integration level
            delta_phi: Change in Φ from last step
            step: Optional step number for intervention spacing

        Returns:
            (drive_level, intervention_type, info_dict)
            - drive_level: 0-1 motivation to explore
            - intervention_type: "RESET_EXPLORATION" or None
            - info_dict: Telemetry about drive state
        """
        # Track history
        self._phi_history.append(phi)
        self._peak_phi = max(self._peak_phi, phi)
        self._integration_attempts += 1

        # Detect catastrophic drop (Φ drops >0.3 in one step)
        phi_drop = self._prev_phi - phi if self._prev_phi > 0 else 0.0
        catastrophic_drop = phi_drop > 0.3
        if catastrophic_drop:
            self._catastrophic_drops += 1
        self._prev_phi = phi

        # Track success/failure
        if delta_phi > self.cfg.success_delta:
            self._successes += 1
            self._consecutive_failures = 0
        elif delta_phi < -self.cfg.success_delta:
            # Only count as failure if DECLINING (not just flat)
            self._consecutive_failures += 1
        # else: neutral (small changes don't count either way)

        # Compute exploration dynamics
        prev_velocity = self._velocity
        self._velocity = delta_phi  # Current rate of change
        self._acceleration = self._velocity - prev_velocity  # Change in rate

        # Amplitude: range in recent window
        if len(self._phi_history) >= 2:
            self._amplitude = max(self._phi_history) - min(self._phi_history)

        # Connections: count of "engaged" steps (Φ > threshold)
        self._connections = sum(1 for p in self._phi_history if p > self.cfg.peak_threshold)

        # Wave timing insight (if provided)
        wave_phase = None
        wave_momentum = 0.0
        if wave_state:
            # Analyze wave state for tacking timing
            wave_amp = wave_state.get("amplitude", 0.0)
            wave_vel = wave_state.get("velocity", 0.0)
            wave_accel = wave_state.get("acceleration", 0.0)

            # Determine wave phase for optimal tacking (aligned with ChatGPT feedback)
            if wave_amp < 0.05:
                wave_phase = "FLAT"  # Tiny wave, nothing to time
                wave_momentum = 0.0
            elif wave_vel > 0.01 and wave_accel >= 0:
                wave_phase = "RISING_PUSH"  # Climbing, accelerating - BEST TIME
                wave_momentum = 0.3
            elif wave_vel > 0.01 and wave_accel < 0:
                wave_phase = "RISING_TOP"  # Still rising but slowing (near crest)
                wave_momentum = 0.2
            elif wave_vel < -0.01 and wave_accel <= 0:
                wave_phase = "FALLING_DROP"  # Falling and accelerating down - WORST TIME
                wave_momentum = -0.3
            elif wave_vel < -0.01 and wave_accel > 0:
                wave_phase = "FALLING_EASING"  # Falling but decelerating (bottoming out)
                wave_momentum = -0.1
            else:
                wave_phase = "STABLE"  # Unclear, wait
                wave_momentum = 0.0

        # Compute frustration (hitting wall repeatedly)
        self._frustration = self._compute_frustration(phi, delta_phi)

        # Compute curiosity (novelty / progress)
        curiosity = self._compute_curiosity(phi)

        # Compute drive
        drive = (
            self.cfg.base_drive
            + self.cfg.curiosity_weight * curiosity
            - self.cfg.frustration_weight * self._frustration
        )
        drive = max(0.1, min(1.0, drive))  # Clamp to [0.1, 1.0]

        # Check for motivational collapse (gave up)
        gave_up = self._detect_gave_up(phi)

        # REQUEST-BASED ASSISTANCE: System asks for help based on wave timing + curiosity
        # This follows ChatGPT's guidance: navigator chooses WHEN and HOW MUCH help
        assistance_request = None
        nav_intent = "self_navigation"  # Default: handle it myself

        # 1. RISING wave (PUSH or TOP) + high curiosity + good drive → ASK FOR PUSH
        if wave_phase in ["RISING_PUSH", "RISING_TOP"] and curiosity >= 0.7 and drive >= 0.7:
            # THIS is the time to capitalize on the wave!
            if phi < 0.3:
                assistance_request = "MEDIUM_PUSH"  # Still far from goal, push harder
                nav_intent = "surf_up_medium"
            else:
                assistance_request = "SMALL_PUSH"  # Closer to goal, don't overshoot
                nav_intent = "surf_up_small"

        # 2. FALLING hard + high frustration → ASK FOR STABILIZING HELP
        elif wave_phase == "FALLING_DROP" and self._frustration >= 0.7:
            assistance_request = "SMALL_PUSH"  # Small stabilizing nudge, not big correction
            nav_intent = "stabilise"

        # 3. FLAT plateau + moderate curiosity → OCCASIONAL EXPLORATORY HELP
        elif wave_phase == "FLAT" and 0.3 <= curiosity <= 0.5 and drive >= 0.5:
            assistance_request = "SMALL_PUSH"
            nav_intent = "explore_new_angle"

        # 4. FLAT + low drive → LARGE HELP (stuck in dead water)
        elif wave_phase == "FLAT" and drive < 0.4:
            assistance_request = "LARGE_PUSH"
            nav_intent = "escape_deadwater"

        # 5. Near peak in geometric regime → NO HELP NEEDED
        elif wave_phase in ["RISING_TOP", "STABLE"] and phi > 0.4:
            assistance_request = None
            nav_intent = "riding_high"

        # Intervention logic
        intervention = None
        if gave_up:
            # Only intervene if enough time since last reset
            if step is None or (step - self._last_intervention_step > 10):
                intervention = "RESET_EXPLORATION"
                if step is not None:
                    self._last_intervention_step = step

                # Track intervention
                self._interventions_count += 1

                # Reset frustration (fresh start)
                self._frustration = 0.0
                self._consecutive_failures = 0

                if self.cfg.verbose:
                    print(f"[EXPLORATION RESET] System gave up (Phi {phi:.3f}, peak {self._peak_phi:.3f})")

        # Build info dict
        info = {
            "drive": drive,
            "curiosity": curiosity,
            "frustration": self._frustration,
            "peak_phi": self._peak_phi,
            "success_rate": self._successes / max(1, self._integration_attempts),
            "consecutive_failures": self._consecutive_failures,
            "gave_up": gave_up,
            # Exploration dynamics
            "velocity": self._velocity,
            "acceleration": self._acceleration,
            "amplitude": self._amplitude,
            "connections": self._connections,
            "interest": curiosity,  # Alias for curiosity
            "engagement": drive,  # Alias for overall drive
            # Wave timing insight
            "wave_phase": wave_phase,
            "wave_momentum": wave_momentum,
            "nav_intent": nav_intent,  # What the navigator wants to do
            "assistance_request": assistance_request,  # What help it's asking for
            # Collapse and intervention tracking
            "catastrophic_drop": catastrophic_drop,
            "phi_drop_magnitude": phi_drop,
            "catastrophic_drops_total": self._catastrophic_drops,
            "interventions_count": self._interventions_count,
            "steps_since_last_intervention": step - self._last_intervention_step if step else 0,
        }

        # Use assistance request as intervention if present
        if assistance_request and not gave_up:
            intervention = assistance_request

        return drive, intervention, info

    def _compute_frustration(self, phi: float, delta_phi: float) -> float:
        """
        Estimate frustration level from recent trajectory.

        Frustration increases when:
        - Φ peaked then declined (hit wall)
        - Consecutive failures to increase Φ
        - Stuck at low Φ after achieving high Φ

        Args:
            phi: Current Φ
            delta_phi: Change in Φ

        Returns:
            Frustration level (0-1)
        """
        if len(self._phi_history) < 5:
            return 0.0

        recent = list(self._phi_history)[-10:]

        # Pattern 1: Peaked then declined
        recent_peak = max(recent)
        if recent_peak > self.cfg.peak_threshold and phi < recent_peak * 0.5:
            # Had something, now lost it
            return 0.8

        # Pattern 2: Consecutive failures
        if self._consecutive_failures > 5:
            return min(1.0, 0.3 + 0.1 * (self._consecutive_failures - 5))

        # Pattern 3: Declining
        if delta_phi < -0.01:
            return 0.5

        return 0.1  # Baseline frustration

    def _compute_curiosity(self, phi: float) -> float:
        """
        Estimate curiosity / interest level.

        Curiosity increases when:
        - Discovering new patterns (Φ growing)
        - Exploring previously unvisited regions
        - Making progress toward integration

        Args:
            phi: Current Φ

        Returns:
            Curiosity level (0-1)
        """
        if len(self._phi_history) < 2:
            return 0.5  # Neutral curiosity at start

        recent = list(self._phi_history)[-5:]

        # Curiosity driven by growth
        growth_rate = (recent[-1] - recent[0]) / len(recent)
        if growth_rate > 0.005:
            return 0.9  # High curiosity when growing fast
        elif growth_rate > 0.001:
            return 0.7  # Good curiosity when growing steadily
        elif growth_rate > 0:
            return 0.5  # Moderate curiosity when slowly growing

        # Novelty from being in new territory
        if phi > self._peak_phi * 0.9:
            return 0.7  # Near peak = interesting

        return 0.3  # Low curiosity when stagnant

    def _detect_gave_up(self, phi: float) -> bool:
        """
        Detect if system has disengaged / given up.

        Pattern: Had integration (peak > 0.10), now disengaged (current < 0.01)
        for multiple consecutive steps.

        This distinguishes:
        - Gave up (had 0.12, now 0.00 for 5 steps) ← DETECT
        - Still warming up (never exceeded 0.05) ← IGNORE

        Args:
            phi: Current Φ

        Returns:
            True if system has given up
        """
        if len(self._phi_history) < 5:
            return False

        # Check if disengaged
        if phi > self.cfg.gave_up_threshold:
            return False  # Still engaged

        # Check if HAD engagement previously
        if self._peak_phi < self.cfg.peak_threshold:
            return False  # Never had engagement (still warming up)

        # Check if STAYED disengaged
        recent = list(self._phi_history)[-5:]
        all_disengaged = all(p < self.cfg.gave_up_threshold for p in recent)

        if not all_disengaged:
            return False  # Just temporarily low

        # Pattern confirmed: had something, lost it, stayed lost
        return True

    def get_reset_threshold(self) -> float:
        """
        Generate random threshold for exploration reset.

        Randomization breaks learned helplessness pattern by trying
        completely different connection density.

        Returns:
            Random threshold in configured range
        """
        return random.uniform(self.cfg.reset_threshold_min, self.cfg.reset_threshold_max)

    def reset(self):
        """Reset all tracking (for new training run)."""
        self._phi_history.clear()
        self._peak_phi = 0.0
        self._integration_attempts = 0
        self._successes = 0
        self._consecutive_failures = 0
        self._frustration = 0.0
        self._last_intervention_step = -100


# ===========================================================================
# VALIDATION
# ===========================================================================


def validate_exploration_drive():
    """Test exploration drive detection and intervention."""
    print("Testing ExplorationDrive motivation system...")

    drive_module = ExplorationDrive(ExplorationDriveConfig(verbose=True))

    # Test 1: Warming up (low Φ, but that's OK)
    print("\n1. Testing warm-up phase (low Φ, no intervention)...")
    for i in range(10):
        phi = 0.05 + i * 0.001
        drive, intervention, info = drive_module.compute_drive(phi, 0.001, step=i)
    assert intervention is None
    assert drive > 0.5  # Reasonable drive
    print(f"   ✅ Warm-up: drive={drive:.2f}, no intervention")

    # Test 2: Growth phase (Φ increasing)
    print("\n2. Testing growth phase (high curiosity)...")
    for i in range(10, 20):
        phi = 0.05 + (i - 10) * 0.01
        drive, intervention, info = drive_module.compute_drive(phi, 0.01, step=i)
    assert drive > 0.7  # High drive when growing
    assert info["curiosity"] > 0.6
    print(f"   ✅ Growth: drive={drive:.2f}, curiosity={info['curiosity']:.2f}")

    # Test 3: Motivational collapse (gave up)
    print("\n3. Testing motivational collapse detection...")
    # Peak at 0.15
    drive, intervention, info = drive_module.compute_drive(0.15, 0.01, step=20)

    # Collapse to 0.00 - intervention will trigger once gave_up detected
    intervention_seen = False
    for i in range(21, 28):
        phi = 0.001
        drive, intervention, info = drive_module.compute_drive(phi, -0.001, step=i)
        if intervention == "RESET_EXPLORATION":
            intervention_seen = True
            print(f"   🔄 Reset triggered at step {i}")

    assert info["gave_up"]  # Should detect gave up
    assert intervention_seen  # Should have intervened
    assert drive < 0.5  # Low drive when gave up
    print(f"   ✅ Collapse detected: gave_up={info['gave_up']}, drive={drive:.2f}")

    # Test 4: Recovery after reset
    print("\n4. Testing recovery after reset...")
    reset_threshold = drive_module.get_reset_threshold()
    assert 0.04 <= reset_threshold <= 0.12
    print(f"   ✅ Reset threshold: {reset_threshold:.3f}")

    # Simulate recovery
    for i in range(28, 35):
        phi = 0.001 + (i - 28) * 0.01
        drive, intervention, info = drive_module.compute_drive(phi, 0.01, step=i)
    assert drive > 0.5  # Drive recovering
    print(f"   ✅ Recovery: drive={drive:.2f}")

    print("\n" + "=" * 60)
    print("ExplorationDrive validation complete! ✅")
    print("=" * 60)
    print("\nExplorationDrive can now:")
    print("  - Detect motivational collapse (gave up)")
    print("  - Track curiosity and frustration")
    print("  - Trigger exploration resets")
    print("  - Break learned helplessness patterns")
    print("\nKey insight: Even ants lose interest. Motivation matters.")


if __name__ == "__main__":
    validate_exploration_drive()
