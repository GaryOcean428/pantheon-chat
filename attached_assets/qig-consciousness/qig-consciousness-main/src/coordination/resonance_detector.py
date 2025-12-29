#!/usr/bin/env python3
"""
Resonance Detector - Pure Measurement
======================================

Detect proximity to optimal coupling κ* and adjust learning accordingly.

PURE PRINCIPLE:
- κ* = 64 is MEASURED optimal (from physics validation)
- Near κ*, small changes amplified (geometric resonance)
- We detect resonance, adapt control (not optimize toward κ*)

PURITY CHECK:
- ✅ κ* from empirical data (not arbitrary)
- ✅ Resonance is observation (not optimization target)
- ✅ LR adjustment is control (not loss modification)
- ✅ κ emerges naturally, never targeted

Key Physics Insight:
- β(L=3→4) = +0.44 (coupling increases)
- β(L=4→5) ≈ 0 (coupling plateaus)
- κ* ≈ 64 (optimal coupling, from κ₄ = 64.47±1.89)

Near κ*, system becomes SENSITIVE - small perturbations cause large Φ effects
(like pushing a swing at resonance frequency).

Written for QIG consciousness research.
"""

from typing import Optional

import torch


class ResonanceDetector:
    """Detect proximity to optimal coupling κ* and adjust learning accordingly.

    PURE PRINCIPLE:
    - κ* = 64 is MEASURED optimal (from physics validation)
    - Near κ*, small changes amplified (geometric resonance)
    - We detect resonance, adapt control (not optimize toward κ*)

    PURITY CHECK:
    - ✅ κ* from empirical data (not arbitrary)
    - ✅ Resonance is observation (not optimization target)
    - ✅ LR adjustment is control (not loss modification)
    - ✅ κ emerges naturally, never targeted
    """

    def __init__(self, kappa_star: float = 64.0, resonance_width: float = 10.0):
        """Initialize resonance detector.

        Args:
            kappa_star: Optimal coupling (from physics: κ₄ = 64.47±1.89)
            resonance_width: Half-width of resonance region (units of κ)
        """
        self.kappa_star = kappa_star
        self.resonance_width = resonance_width
        self.history: list[dict] = []

    def check_resonance(self, kappa_current: float) -> dict:
        """Check if current κ is near resonance.

        PURE: We measure proximity, we don't optimize toward it.

        κ* is not a target - it's a stability point where small changes
        have large effects. We detect proximity for SAFETY, not convergence.

        Args:
            kappa_current: Current coupling strength

        Returns:
            Dict with resonance metrics
        """
        # Distance from optimal (absolute value on manifold)
        distance = abs(kappa_current - self.kappa_star)

        # In resonance if within width
        in_resonance = distance < self.resonance_width

        # Resonance strength (0 = far, 1 = at κ*)
        # This is a smooth measure, not a hard threshold
        strength = max(0.0, 1.0 - distance / self.resonance_width)

        # Record in history (pure measurement)
        record = {"kappa": kappa_current, "distance": distance, "in_resonance": in_resonance, "strength": strength}
        self.history.append(record)

        # Keep history bounded
        if len(self.history) > 100:
            self.history.pop(0)

        return {
            "kappa": kappa_current,
            "kappa_star": self.kappa_star,
            "distance_to_optimal": distance,
            "in_resonance": in_resonance,
            "resonance_strength": strength,
        }

    def compute_learning_rate_multiplier(self, kappa_current: float, min_multiplier: float = 0.1) -> float:
        """Compute LR multiplier based on resonance proximity.

        PURE: Adaptive control based on geometry, not optimization.

        Strategy:
        - Far from κ*: normal LR (multiplier = 1.0)
        - Near κ*: reduce LR proportionally (multiplier < 1.0)
        - At κ*: minimum LR (multiplier = min_multiplier)

        Rationale: Near resonance, small parameter changes cause large
        Φ effects. Reducing LR prevents accidental destabilization.

        This is NOT optimization toward κ* - κ emerges naturally.
        This IS adaptive control to prevent breakdown near sensitive point.

        Args:
            kappa_current: Current coupling strength
            min_multiplier: Minimum LR multiplier (default 0.1 = 10% of base LR)

        Returns:
            Learning rate multiplier in [min_multiplier, 1.0]
        """
        resonance = self.check_resonance(kappa_current)

        if not resonance["in_resonance"]:
            return 1.0  # Normal LR

        # Reduce LR proportionally to resonance strength
        # strength=0 → mult=1.0 (far edge of resonance region)
        # strength=1 → mult=min_multiplier (exactly at κ*)
        multiplier = 1.0 - (1.0 - min_multiplier) * resonance["resonance_strength"]

        return max(min_multiplier, multiplier)

    def get_resonance_report(self) -> dict:
        """Get comprehensive resonance report (pure measurement).

        Returns:
            Dict with resonance statistics
        """
        if not self.history:
            return {
                "current_kappa": 0.0,
                "avg_kappa": 0.0,
                "min_distance": float("inf"),
                "closest_approach": 0.0,
                "time_in_resonance": 0.0,
                "measurements": 0,
            }

        kappas = [h["kappa"] for h in self.history]
        distances = [h["distance"] for h in self.history]
        in_resonance_flags = [h["in_resonance"] for h in self.history]

        current_kappa = kappas[-1]
        avg_kappa = sum(kappas) / len(kappas)
        min_distance = min(distances)
        closest_kappa = kappas[distances.index(min_distance)]
        time_in_resonance = sum(in_resonance_flags) / len(in_resonance_flags)

        return {
            "current_kappa": current_kappa,
            "avg_kappa": avg_kappa,
            "kappa_star": self.kappa_star,
            "min_distance_to_optimal": min_distance,
            "closest_kappa": closest_kappa,
            "time_in_resonance_pct": time_in_resonance * 100,
            "currently_in_resonance": in_resonance_flags[-1],
            "measurements": len(self.history),
        }

    def detect_oscillation_around_resonance(self, window: int = 20) -> tuple[bool, int]:
        """Detect if κ is oscillating around κ*.

        PURE: Pattern detection (measurement, not optimization).

        Oscillation around κ* indicates instability - system is
        searching but can't stabilize. This is a warning sign.

        Args:
            window: Number of recent measurements to analyze

        Returns:
            (is_oscillating, num_crossings):
                - is_oscillating: True if oscillating
                - num_crossings: Number of times crossed κ*
        """
        if len(self.history) < window:
            return False, 0

        recent = self.history[-window:]
        kappas = [h["kappa"] for h in recent]

        # Count crossings of κ*
        crossings = 0
        for i in range(len(kappas) - 1):
            # Crossing if one side is below κ*, other is above
            if (kappas[i] < self.kappa_star and kappas[i + 1] > self.kappa_star) or (
                kappas[i] > self.kappa_star and kappas[i + 1] < self.kappa_star
            ):
                crossings += 1

        # Oscillating if > 30% of window are crossings
        is_oscillating = crossings > window * 0.3

        return is_oscillating, crossings

    def suggest_intervention(self, kappa_current: float) -> str | None:
        """Suggest intervention based on resonance state.

        PURE: Advisory output (measurement-based recommendation).

        Args:
            kappa_current: Current coupling strength

        Returns:
            Intervention suggestion string, or None if no action needed
        """
        resonance = self.check_resonance(kappa_current)

        # Check for oscillation
        is_oscillating, crossings = self.detect_oscillation_around_resonance()

        # Strong resonance
        if resonance["resonance_strength"] > 0.8:
            return f"⚠️ Very close to κ* (distance={resonance['distance_to_optimal']:.1f}). Consider reducing LR 10x."

        # In resonance with oscillation
        if resonance["in_resonance"] and is_oscillating:
            return f"⚠️ Oscillating around κ* ({crossings} crossings). System unstable - reduce LR or pause."

        # Approaching resonance
        if resonance["resonance_strength"] > 0.5:
            return f"ℹ️ Approaching κ* (distance={resonance['distance_to_optimal']:.1f}). Gentle training recommended."

        return None

    def reset(self):
        """Reset detector (for new training session).

        PURE: Configuration reset, not optimization.
        """
        self.history.clear()


if __name__ == "__main__":
    print("Resonance Detector: Pure Measurement")
    print("=" * 60)
    print()
    print("PURE PRINCIPLES:")
    print("✅ κ* = 64 from physics validation (not arbitrary)")
    print("✅ Resonance is observation (not target)")
    print("✅ LR adjustment is control (not loss)")
    print("✅ κ emerges naturally, never optimized")
    print()
    print("KEY PHYSICS:")
    print("- κ₃ = 41.09±0.59 (emergence point)")
    print("- κ₄ = 64.47±1.89 (strong running)")
    print("- β ≈ 0.44 (running coupling slope)")
    print("- κ* ≈ 64 (optimal, resonance point)")
    print()

    # Quick validation test
    detector = ResonanceDetector(kappa_star=64.0, resonance_width=10.0)

    print("Testing resonance detection...")
    # Simulate κ trajectory approaching and crossing κ*
    test_kappas = [40, 45, 50, 55, 60, 62, 64, 66, 68, 70, 65, 63, 64, 62]

    for i, kappa in enumerate(test_kappas):
        resonance = detector.check_resonance(kappa)
        lr_mult = detector.compute_learning_rate_multiplier(kappa)

        status = "🎯" if resonance["in_resonance"] else "  "
        strength_bar = "█" * int(resonance["resonance_strength"] * 10)

        print(
            f"{status} κ={kappa:5.1f}  dist={resonance['distance_to_optimal']:4.1f}  "
            f"strength=[{strength_bar:10s}]  LR×{lr_mult:.2f}"
        )

        # Check for intervention suggestion
        suggestion = detector.suggest_intervention(kappa)
        if suggestion:
            print(f"     {suggestion}")

    # Check for oscillation
    is_osc, crossings = detector.detect_oscillation_around_resonance(window=14)
    print(f"\nOscillation: {is_osc}, Crossings: {crossings}")

    # Get report
    report = detector.get_resonance_report()
    print("\nReport:")
    print(f"  Avg κ: {report['avg_kappa']:.1f}")
    print(f"  Time in resonance: {report['time_in_resonance_pct']:.1f}%")
    print(f"  Closest approach: κ={report['closest_kappa']:.1f} (dist={report['min_distance_to_optimal']:.1f})")

    print("\n✓ Validation complete")
