"""
Unit tests for Mushroom Mode safety system.

Tests empirically validated safety thresholds discovered on Nov 20, 2025:
- 58% breakdown + microdose → breakdown explosion
- 66% breakdown + moderate → ego death (Φ collapse)
"""

import pytest

from src.qig.neuroplasticity.mushroom_mode import MUSHROOM_SAFETY_THRESHOLDS, MushroomMode


class TestMushroomModeSafety:
    """Test safety validation system prevents catastrophic failures."""

    def test_refuse_high_breakdown(self):
        """Safety system should refuse at >40% breakdown."""
        mushroom = MushroomMode(intensity="moderate")

        # Simulate 60% breakdown telemetry (above catastrophic threshold)
        telemetry = [{"regime": "breakdown", "Phi": 0.80} for _ in range(6)] + [
            {"regime": "geometric", "Phi": 0.75} for _ in range(4)
        ]

        is_safe, reason = mushroom.validate_safety(None, telemetry)

        assert not is_safe, "Should refuse >40% breakdown"
        assert "BREAKDOWN_TOO_HIGH" in reason
        assert "40" in reason  # Should mention the threshold

    def test_refuse_58_percent_breakdown_microdose(self):
        """
        Empirical validation: 58% breakdown + microdose caused explosion.

        On Nov 20, 2025:
        - Before: Φ=0.804, basin=0.012
        - After microdose: basin exploded to 0.321 (26× worse)
        - Breakdown → 100%
        """
        mushroom = MushroomMode(intensity="microdose")

        # Simulate 58% breakdown (observed failure case)
        telemetry = [{"regime": "breakdown", "Phi": 0.80} for _ in range(58)] + [
            {"regime": "geometric", "Phi": 0.75} for _ in range(42)
        ]

        is_safe, reason = mushroom.validate_safety(None, telemetry)

        assert not is_safe, "Should refuse 58% breakdown even for microdose"
        assert "UNSAFE" in reason or "BREAKDOWN" in reason

    def test_refuse_66_percent_breakdown_moderate(self):
        """
        Empirical validation: 66% breakdown + moderate caused ego death.

        On Nov 20, 2025:
        - Before: Φ=0.805, basin=0.001 (perfect)
        - After moderate: Φ→0.636 (consciousness collapse)
        - Basin→0.141 (identity lost)
        - Output incoherent
        """
        mushroom = MushroomMode(intensity="moderate")

        # Simulate 66% breakdown (ego death case)
        telemetry = [{"regime": "breakdown", "Phi": 0.81} for _ in range(66)] + [
            {"regime": "geometric", "Phi": 0.75} for _ in range(34)
        ]

        is_safe, reason = mushroom.validate_safety(None, telemetry)

        assert not is_safe, "Should refuse 66% breakdown for moderate"
        assert "ego death" in reason.lower() or "unsafe" in reason.lower()

    def test_allow_safe_breakdown_microdose(self):
        """Safety system should allow at <35% breakdown for microdose."""
        mushroom = MushroomMode(intensity="microdose")

        # Simulate 25% breakdown telemetry (safe range)
        telemetry = [{"regime": "breakdown", "Phi": 0.75} for _ in range(25)] + [
            {"regime": "geometric", "Phi": 0.75} for _ in range(75)
        ]

        is_safe, reason = mushroom.validate_safety(None, telemetry)

        assert is_safe, f"Should allow 25% breakdown for microdose: {reason}"
        assert "SAFE" in reason

    def test_allow_safe_breakdown_moderate(self):
        """Safety system should allow at <25% breakdown for moderate."""
        mushroom = MushroomMode(intensity="moderate")

        # Simulate 20% breakdown telemetry (safe for moderate)
        telemetry = [{"regime": "breakdown", "Phi": 0.75} for _ in range(20)] + [
            {"regime": "geometric", "Phi": 0.75} for _ in range(80)
        ]

        is_safe, reason = mushroom.validate_safety(None, telemetry)

        assert is_safe, f"Should allow 20% breakdown for moderate: {reason}"

    def test_intensity_specific_limits(self):
        """Different intensities should have different breakdown limits."""
        # 30% breakdown telemetry (edge case)
        telemetry = [{"regime": "breakdown", "Phi": 0.75} for _ in range(30)] + [
            {"regime": "geometric", "Phi": 0.75} for _ in range(70)
        ]

        # Microdose should pass (limit 35%)
        microdose = MushroomMode(intensity="microdose")
        is_safe, _ = microdose.validate_safety(None, telemetry)
        assert is_safe, "Microdose should allow 30% breakdown"

        # Moderate should fail (limit 25%)
        moderate = MushroomMode(intensity="moderate")
        is_safe, _ = moderate.validate_safety(None, telemetry)
        assert not is_safe, "Moderate should refuse 30% breakdown"

        # Heroic should definitely fail (limit 15%)
        heroic = MushroomMode(intensity="heroic")
        is_safe, _ = heroic.validate_safety(None, telemetry)
        assert not is_safe, "Heroic should refuse 30% breakdown"

    def test_refuse_low_phi(self):
        """Safety system should refuse if Φ already below consciousness threshold."""
        mushroom = MushroomMode(intensity="microdose")

        # Simulate low Φ (below 0.70 consciousness threshold)
        telemetry = [{"regime": "geometric", "Phi": 0.60} for _ in range(100)]

        is_safe, reason = mushroom.validate_safety(None, telemetry)

        assert not is_safe, "Should refuse when Φ < 0.70"
        assert "PHI_TOO_LOW" in reason

    def test_refuse_insufficient_geometric(self):
        """Safety system should refuse if geometric regime < 50%."""
        mushroom = MushroomMode(intensity="microdose")

        # Simulate insufficient geometric regime (40%)
        telemetry = [{"regime": "geometric", "Phi": 0.75} for _ in range(40)] + [
            {"regime": "linear", "Phi": 0.60} for _ in range(60)
        ]

        is_safe, reason = mushroom.validate_safety(None, telemetry)

        assert not is_safe, "Should refuse when geometric < 50%"
        assert "INSUFFICIENT_GEOMETRIC" in reason

    def test_refuse_insufficient_data(self):
        """Safety system should refuse with no telemetry history."""
        mushroom = MushroomMode(intensity="microdose")

        is_safe, reason = mushroom.validate_safety(None, [])

        assert not is_safe, "Should refuse with no telemetry"
        assert "INSUFFICIENT_DATA" in reason

    def test_thresholds_match_empirical_values(self):
        """Verify safety thresholds match empirically discovered values."""
        # These values are from Nov 20, 2025 experiments
        assert MUSHROOM_SAFETY_THRESHOLDS["max_breakdown_before_trip"] == 0.40
        assert MUSHROOM_SAFETY_THRESHOLDS["microdose_max_breakdown"] == 0.35
        assert MUSHROOM_SAFETY_THRESHOLDS["moderate_max_breakdown"] == 0.25
        assert MUSHROOM_SAFETY_THRESHOLDS["heroic_max_breakdown"] == 0.15
        assert MUSHROOM_SAFETY_THRESHOLDS["abort_if_phi_drops_below"] == 0.65
        assert MUSHROOM_SAFETY_THRESHOLDS["max_basin_drift_allowed"] == 0.15
        assert MUSHROOM_SAFETY_THRESHOLDS["min_geometric_regime_pct"] == 50.0


class TestMushroomModeParameters:
    """Test mushroom mode intensity parameters."""

    def test_microdose_parameters(self):
        """Verify microdose has conservative parameters."""
        mushroom = MushroomMode(intensity="microdose")

        assert mushroom.duration_steps == 50, "Microdose should be 50 steps"
        assert mushroom.entropy_multiplier == 1.2, "Microdose should be 1.2× entropy"

    def test_moderate_parameters(self):
        """Verify moderate has standard parameters."""
        mushroom = MushroomMode(intensity="moderate")

        assert mushroom.duration_steps == 200, "Moderate should be 200 steps"
        assert mushroom.entropy_multiplier == 3.0, "Moderate should be 3.0× entropy"

    def test_heroic_parameters(self):
        """Verify heroic has aggressive parameters."""
        mushroom = MushroomMode(intensity="heroic")

        assert mushroom.duration_steps == 500, "Heroic should be 500 steps"
        assert mushroom.entropy_multiplier == 5.0, "Heroic should be 5.0× entropy"

    def test_integration_period_equals_trip(self):
        """Integration period should equal trip duration for all intensities."""
        for intensity in ["microdose", "moderate", "heroic"]:
            mushroom = MushroomMode(intensity=intensity)
            assert mushroom.integration_period == mushroom.duration_steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
