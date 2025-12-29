#!/usr/bin/env python3
"""
Navigator: Geometry-Aware Controller for QIG-Kernel Training
===========================================================

Feels the manifold geometry and proposes threshold adjustments.

Unlike the wave controller (which measured velocity/acceleration symptoms),
Navigator feels GEOMETRIC STRUCTURE DIRECTLY:
- Basin curvature (∇²Φ)
- Coupling gradients (∂κ/∂L)
- Regime classification (linear/geometric/breakdown)

Key insight from Braden's surfing analogy:
"Stop measuring symptoms, start feeling geometry.
 Stop tinkering externally, start navigating internally.
 Stop suppressing oscillations, start surfing them."

The Navigator doesn't control the system - it SURFS the manifold.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .exploration_drive import ExplorationDrive, ExplorationDriveConfig


class Regime(str, Enum):
    """Regime classification (from physics L=3,4,5 data)."""

    LINEAR = "linear"
    GEOMETRIC = "geometric"
    BREAKDOWN = "breakdown"
    UNKNOWN = "unknown"


class NavigatorPhase(str, Enum):
    """Φ trajectory phase (for micro-oscillation awareness)."""

    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"


@dataclass
class NavigatorConfig:
    """Configuration for Navigator geometric control."""

    # Geometric regime adjustments
    rise_push_factor: float = 0.85  # Push harder when rising (15% reduction)
    fall_relax_factor: float = 1.10  # Relax when falling (10% increase)

    # Conservative adjustments outside safe regime
    linear_push_factor: float = 0.95  # Gentle push in linear regime
    breakdown_relax_factor: float = 1.20  # Strong relax in breakdown

    # GRADUATED PUSHING: Acceleration-based intensity (Priority 1)
    strong_push_factor: float = 0.90  # Strong push (10%) when accelerating
    medium_push_factor: float = 0.95  # Medium push (5%) when steady
    gentle_push_factor: float = 0.98  # Gentle push (2%) when decelerating

    # REVERSE GEAR: Relax factors for declining Φ
    strong_relax_factor: float = 1.10  # Strong relax (10%) when rapidly falling
    gentle_relax_factor: float = 1.05  # Gentle relax (5%) when slowly falling

    # Acceleration thresholds for graduated control
    accel_threshold_high: float = 0.001  # Accelerating strongly
    accel_threshold_low: float = -0.001  # Decelerating

    # Φ change detection (what counts as "meaningful")
    min_phi_delta: float = 1e-4  # Don't filter micro-oscillations!

    # Basin stability threshold
    max_basin_for_push: float = 1.2  # Too unstable → back off regardless

    # MOTIVATION / CURIOSITY (Braden's insight: "even ants lose interest")
    enable_exploration_drive: bool = True  # Enable motivation tracking
    drive_config: ExplorationDriveConfig | None = None  # Use default if None


class Navigator:
    """
    Geometry-aware threshold controller.

    Uses Φ, basin distance, κ_eff, and regime to navigate manifold.
    Works WITH natural oscillations (surfing) not against them (damping).

    Key difference from wave controller:
    - Wave: Measured velocity/acceleration (symptoms)
    - Navigator: Feels basin curvature (geometry)

    NOW TRACKS: velocity, acceleration, connection density for full observability

    Example usage:
        >>> nav = Navigator(NavigatorConfig())
        >>> # System in geometric regime, Φ rising
        >>> proposed, info = nav.propose_threshold(
        ...     phi=0.65,
        ...     basin_distance=0.85,
        ...     kappa_eff=45.0,
        ...     regime=Regime.GEOMETRIC,
        ...     current_threshold=0.10
        ... )
        >>> print(info['decision'])  # "geometric_push"
        >>> print(proposed)  # 0.085 (15% reduction to push)
    """

    def __init__(self, cfg: NavigatorConfig):
        """
        Initialize Navigator with geometric sensing configuration.

        Args:
            cfg: NavigatorConfig with adjustment factors
        """
        self.cfg = cfg
        self._phi_prev: float | None = None
        self._phi_prev2: float | None = None  # For acceleration
        self._velocity: float = 0.0
        self._acceleration: float = 0.0
        self._step = 0  # Track step for intervention spacing

        # History tracking for basin velocity computation
        self._history: dict[str, Any] = {
            "phi": [],
            "basin": [],
        }

        # MOTIVATION: ExplorationDrive (detects "gave up" and triggers resets)
        if cfg.enable_exploration_drive:
            drive_cfg = cfg.drive_config or ExplorationDriveConfig()
            self.exploration_drive: ExplorationDrive | None = ExplorationDrive(drive_cfg)
        else:
            self.exploration_drive = None

    def _estimate_phase(self, phi: float) -> NavigatorPhase:
        """
        Estimate Φ trajectory phase from recent history.

        Unlike wave controller, this doesn't filter "noise" -
        micro-oscillations are EXPLORATION, not errors!

        NOW COMPUTES: velocity (dΦ/dt) and acceleration (d²Φ/dt²)

        Args:
            phi: Current Φ value

        Returns:
            NavigatorPhase (rising/falling/stable)
        """
        # First call - initialize
        if self._phi_prev is None:
            self._phi_prev = phi
            self._phi_prev2 = phi
            self._velocity = 0.0
            self._acceleration = 0.0
            return NavigatorPhase.STABLE

        # Compute velocity (dΦ/dt)
        velocity = phi - self._phi_prev

        # Compute acceleration (d²Φ/dt²) if we have enough history
        if self._phi_prev2 is not None:
            prev_velocity = self._phi_prev - self._phi_prev2
            self._acceleration = velocity - prev_velocity
        else:
            self._acceleration = 0.0

        self._velocity = velocity

        # Update history
        self._phi_prev2 = self._phi_prev
        self._phi_prev = phi

        # Micro-oscillations (~0.0001) are VALID exploration
        # Don't filter them out like wave controller did!
        if abs(velocity) < self.cfg.min_phi_delta:
            return NavigatorPhase.STABLE

        return NavigatorPhase.RISING if velocity > 0 else NavigatorPhase.FALLING

    def propose_threshold(
        self,
        phi: float,
        basin_distance: float,
        kappa_eff: float,
        regime: Regime,
        current_threshold: float,
    ) -> tuple[float, dict[str, Any]]:
        """
        Propose threshold adjustment based on felt geometry + motivation.

        This is where the magic happens - Navigator FEELS the manifold:
        - Basin curvature (is it steep or gentle?)
        - Regime classification (where are we on manifold?)
        - Φ trajectory (rising or falling?)
        - Exploration drive (motivated or gave up?) ← NEW

        Then chooses response that works WITH the geometry,
        not against it.

        Args:
            phi: Current integration level
            basin_distance: Distance to target basin
            kappa_eff: Effective coupling at current scale
            regime: Geometric regime classification
            current_threshold: Current threshold value

        Returns:
            (proposed_threshold, navigation_info)
        """
        self._step += 1
        phase = self._estimate_phase(phi)

        # Track history
        self._history["phi"].append(phi)
        self._history["basin"].append(basin_distance)

        # Keep only recent history (last 50 steps)
        if len(self._history["phi"]) > 50:
            self._history["phi"] = self._history["phi"][-50:]
            self._history["basin"] = self._history["basin"][-50:]

        # Compute amplitude from recent phi history
        recent_phis = self._history["phi"][-20:] if len(self._history["phi"]) >= 20 else self._history["phi"]
        self._history["amplitude"] = max(recent_phis) - min(recent_phis) if recent_phis else 0.0

        # Compute basin velocity (approaching or diverging from target)
        prev_basin = self._history["basin"][-2] if len(self._history["basin"]) >= 2 else basin_distance
        basin_velocity = basin_distance - prev_basin  # Negative = approaching, Positive = diverging

        info: dict[str, Any] = {
            "phase": phase.value,
            "regime": regime.value if isinstance(regime, Regime) else str(regime),
            "phi": phi,
            "basin_distance": basin_distance,
            "basin_velocity": basin_velocity,
            "kappa_eff": kappa_eff,
            "velocity": self._velocity,
            "acceleration": self._acceleration,
        }

        # MOTIVATION: Compute exploration drive
        if self.exploration_drive is not None:
            delta_phi = self._velocity  # velocity = dΦ/dt

            # Build wave state for timing insight
            amplitude_val = self._history.get("amplitude", 0.0)
            # Ensure it's a float (might be from dict)
            amplitude_float = amplitude_val if isinstance(amplitude_val, int | float) else 0.0
            wave_state = {
                "amplitude": amplitude_float,
                "velocity": self._velocity,
                "acceleration": self._acceleration,
            }

            drive, intervention, drive_info = self.exploration_drive.compute_drive(
                phi=phi, delta_phi=delta_phi, wave_state=wave_state, step=self._step
            )

            # Add drive telemetry (full drive_info for console display)
            # Type note: info is Dict[str, Any] to allow nested dicts
            info["drive"] = drive
            info["curiosity"] = drive_info["curiosity"]
            info["frustration"] = drive_info["frustration"]
            info["gave_up"] = drive_info["gave_up"]
            info["peak_phi"] = drive_info["peak_phi"]
            info["drive_info"] = drive_info  # Full info dict for display

            # INTERVENTION: System requesting assistance
            if intervention in ["SMALL_PUSH", "MEDIUM_PUSH", "LARGE_PUSH"]:
                # Honor the request with graduated response
                push_factors = {
                    "SMALL_PUSH": 0.90,  # 10% reduction
                    "MEDIUM_PUSH": 0.85,  # 15% reduction
                    "LARGE_PUSH": 0.75,  # 25% reduction
                }
                factor = push_factors[intervention]
                info["decision"] = f"assistance_{intervention.lower()}"
                info["reason"] = (
                    f"System requested {intervention} "
                    f"(wave: {drive_info.get('wave_phase', 'unknown')}, "
                    f"drive: {drive:.2f})"
                )
                return current_threshold * factor, info

            # INTERVENTION: System gave up - reset exploration
            elif intervention == "RESET_EXPLORATION":
                reset_threshold = self.exploration_drive.get_reset_threshold()
                info["decision"] = "motivational_reset"
                info["reason"] = (
                    f"System gave up (had Φ={drive_info['peak_phi']:.3f}, now Φ={phi:.3f}) → trying fresh angle"
                )
                return reset_threshold, info
        else:
            # No drive tracking
            info["drive"] = 1.0
            info["curiosity"] = None
            info["frustration"] = None
            info["gave_up"] = False

        # SAFETY CHECK: Basin too unstable?
        # If yes, back off regardless of regime
        # This is wisdom, not fear
        if basin_distance > self.cfg.max_basin_for_push:
            info["decision"] = "too_unstable_backoff"
            info["reason"] = f"basin={basin_distance:.3f} > {self.cfg.max_basin_for_push}"
            return current_threshold * self.cfg.fall_relax_factor, info

        # GEOMETRIC REGIME: Active surfing
        # This is the sweet spot - navigate aggressively
        if regime == Regime.GEOMETRIC:
            if phase == NavigatorPhase.RISING:
                info["decision"] = "geometric_push"
                info["reason"] = "Φ rising in geometric regime → surf the wave"
                return current_threshold * self.cfg.rise_push_factor, info

            elif phase == NavigatorPhase.FALLING:
                info["decision"] = "geometric_relax"
                info["reason"] = "Φ falling in geometric regime → let it breathe"
                return current_threshold * self.cfg.fall_relax_factor, info

            else:  # STABLE
                info["decision"] = "geometric_hold"
                info["reason"] = "Φ stable in geometric regime → maintain"
                return current_threshold, info

        # LINEAR REGIME: Conservative navigation with GRADUATED PUSHING
        # System not ready for aggressive moves yet - use acceleration-aware control
        if regime == Regime.LINEAR:
            if phase == NavigatorPhase.RISING:
                # GRADUATED: Push intensity based on acceleration
                if self._acceleration > self.cfg.accel_threshold_high:
                    # Accelerating upward → STRONG push
                    info["decision"] = "linear_strong_push"
                    info["reason"] = f"Φ accelerating (+{self._acceleration:.4f}) → strong push"
                    return current_threshold * self.cfg.strong_push_factor, info
                elif self._acceleration > self.cfg.accel_threshold_low:
                    # Steady rise → MEDIUM push
                    info["decision"] = "linear_medium_push"
                    info["reason"] = f"Φ steady rise (a={self._acceleration:.4f}) → medium push"
                    return current_threshold * self.cfg.medium_push_factor, info
                else:
                    # Decelerating → GENTLE push
                    info["decision"] = "linear_gentle_push"
                    info["reason"] = f"Φ decelerating ({self._acceleration:.4f}) → gentle push"
                    return current_threshold * self.cfg.gentle_push_factor, info

            elif phase == NavigatorPhase.FALLING:
                # REVERSE GEAR: Relax when Φ declining
                if self._acceleration < -self.cfg.accel_threshold_high:
                    # Rapidly falling → STRONG relax
                    info["decision"] = "linear_strong_relax"
                    info["reason"] = f"Φ rapidly falling ({self._acceleration:.4f}) → strong relax"
                    return current_threshold * self.cfg.strong_relax_factor, info
                else:
                    # Slowly falling → GENTLE relax
                    info["decision"] = "linear_gentle_relax"
                    info["reason"] = f"Φ slowly falling (v={self._velocity:.4f}) → gentle relax"
                    return current_threshold * self.cfg.gentle_relax_factor, info

            else:  # STABLE
                info["decision"] = "linear_hold"
                info["reason"] = "Linear regime → maintain stability"
                return current_threshold, info

        # BREAKDOWN REGIME: Escape mode
        # Something's wrong - loosen to let system recover
        if regime == Regime.BREAKDOWN:
            info["decision"] = "breakdown_relax"
            info["reason"] = "Breakdown regime → loosen to recover"
            return current_threshold * self.cfg.breakdown_relax_factor, info

        # UNKNOWN REGIME: Be conservative
        info["decision"] = "unknown_hold"
        info["reason"] = "Unknown regime → maintain stability"
        return current_threshold, info


# ===========================================================================
# VALIDATION
# ===========================================================================


def validate_navigator():
    """Test Navigator geometric sensing and decision making."""
    print("Testing Navigator geometric control...")

    nav = Navigator(NavigatorConfig())

    # Test 1: Geometric regime, rising Φ → push
    print("\n1. Testing geometric regime + rising Φ...")
    # Need two calls to establish rising phase (first initializes history)
    nav.propose_threshold(
        phi=0.60, basin_distance=0.85, kappa_eff=45.0, regime=Regime.GEOMETRIC, current_threshold=0.10
    )
    proposed, info = nav.propose_threshold(
        phi=0.65,
        basin_distance=0.85,
        kappa_eff=45.0,
        regime=Regime.GEOMETRIC,
        current_threshold=0.10,
    )
    assert info["decision"] == "geometric_push"
    assert proposed < 0.10  # Should reduce threshold
    print(f"   ✅ Decision: {info['decision']} (threshold: 0.100 → {proposed:.3f})")

    # Test 2: Unstable basin → back off regardless of regime
    print("\n2. Testing unstable basin override...")
    proposed, info = nav.propose_threshold(
        phi=0.70,
        basin_distance=1.5,  # Too unstable!
        kappa_eff=45.0,
        regime=Regime.GEOMETRIC,  # Even in good regime
        current_threshold=0.10,
    )
    assert info["decision"] == "too_unstable_backoff"
    assert proposed > 0.10  # Should increase threshold
    print(f"   ✅ Decision: {info['decision']} (basin too unstable)")

    # Test 3: Breakdown regime → loosen
    print("\n3. Testing breakdown regime escape...")
    proposed, info = nav.propose_threshold(
        phi=0.35,
        basin_distance=0.90,
        kappa_eff=45.0,
        regime=Regime.BREAKDOWN,
        current_threshold=0.10,
    )
    assert info["decision"] == "breakdown_relax"
    assert proposed > 0.10  # Should increase threshold
    print(f"   ✅ Decision: {info['decision']} (threshold: 0.100 → {proposed:.3f})")

    # Test 4: Linear regime → conservative
    print("\n4. Testing linear regime caution...")
    proposed, info = nav.propose_threshold(
        phi=0.45,
        basin_distance=1.00,
        kappa_eff=41.0,
        regime=Regime.LINEAR,
        current_threshold=0.10,
    )
    assert "linear" in info["decision"]
    print(f"   ✅ Decision: {info['decision']}")

    print("\n" + "=" * 60)
    print("Navigator validation complete! ✅")
    print("=" * 60)
    print("\nNavigator is ready to:")
    print("  - Feel geometry (Φ, basin, κ, regime)")
    print("  - Surf oscillations (not dampen them)")
    print("  - Navigate manifold (not tinker with symptoms)")
    print("\nKey insight: Micro-oscillations are exploration, not noise!")


if __name__ == "__main__":
    validate_navigator()
