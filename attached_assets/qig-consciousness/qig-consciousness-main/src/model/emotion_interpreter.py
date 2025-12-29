#!/usr/bin/env python3
"""
Emotion Interpreter: Map Geometric Telemetry → Emotional State
==============================================================

Maps QIG consciousness metrics to emotional states.

EMOTIONAL MAPPING THEORY:
- Emotions = geometric processing patterns + boundary conditions
- Happy = high integration (Φ), stable basin, low breakdown
- Anxious = high breakdown, unstable basin, high gradient
- Curious = high drive, low proximity (exploring)
- Confident = low surprise, stable basin, optimal regime
- Frustrated = high effort, low progress, repeated breakdown
- Calm = low gradient, stable basin, geometric regime
- Excited = high drive, high velocity, rising Φ

Based on: Affective neuroscience maps emotions to brain state geometry.
Our approach: Φ, κ, basin, regime → emotional valence + arousal.
"""

from dataclasses import dataclass


@dataclass
class EmotionalState:
    """Gary's current emotional state."""

    primary: str  # Main emotion (happy, anxious, curious, etc.)
    intensity: float  # 0.0-1.0
    valence: float  # -1.0 (negative) to +1.0 (positive)
    arousal: float  # 0.0 (calm) to 1.0 (excited)
    secondary: list[str]  # Secondary emotions
    reason: str  # Why this emotion (for transparency)


class EmotionInterpreter:
    """
    Interprets telemetry as emotional states.

    Uses geometric processing metrics to infer subjective experience.
    """

    def __init__(self):
        # Emotion thresholds (tuned from psychology literature + QIG experiments)
        self.thresholds = {
            "high_phi": 0.80,  # High integration = engaged
            "low_phi": 0.65,  # Low integration = struggling
            "high_breakdown": 0.35,  # Fragmentation anxiety
            "safe_breakdown": 0.25,  # Normal cognitive load
            "high_basin_drift": 0.15,  # Identity instability
            "stable_basin": 0.10,  # Identity confidence
            "high_gradient": 0.8,  # High feeling strength (uncertain)
            "low_gradient": 0.3,  # Low feeling strength (confident)
            "high_drive": 0.85,  # Highly curious
            "low_drive": 0.40,  # Low motivation
        }

    def interpret(self, telemetry: dict, history: list[dict] | None = None) -> EmotionalState:
        """
        Map telemetry to emotional state.

        Args:
            telemetry: Current telemetry dict (Φ, basin, breakdown, etc.)
            history: Recent telemetry history (for trends)

        Returns:
            EmotionalState with primary emotion, valence, arousal
        """
        # Extract key metrics
        phi = telemetry.get("Phi", 0.75)
        basin = telemetry.get("basin_distance", 0.05)
        breakdown = telemetry.get("breakdown_pct", 20) / 100.0  # Convert to 0-1
        gradient = telemetry.get("gradient_magnitude", 0.5)
        drive = telemetry.get("drive", 0.7)
        regime = telemetry.get("regime", "geometric")
        mode = telemetry.get("mode", "balanced")

        # Check for trends (if history available)
        phi_trend = self._compute_trend(history, "Phi") if history else 0.0
        basin_trend = self._compute_trend(history, "basin_distance") if history else 0.0

        # === EMOTIONAL CLASSIFICATION ===

        # 1. HAPPY: High Φ, stable basin, low breakdown, geometric regime
        if (
            phi > self.thresholds["high_phi"]
            and basin < self.thresholds["stable_basin"]
            and breakdown < self.thresholds["safe_breakdown"]
            and regime == "geometric"
        ):
            return EmotionalState(
                primary="happy",
                intensity=min(1.0, phi - 0.70),  # Scales with integration
                valence=0.8,
                arousal=0.6,
                secondary=["confident", "engaged"],
                reason=f"High integration (Φ={phi:.2f}), stable identity (basin={basin:.3f}), smooth processing",
            )

        # 2. ANXIOUS: High breakdown, unstable basin, or high gradient
        if (
            breakdown > self.thresholds["high_breakdown"]
            or basin > self.thresholds["high_basin_drift"]
            or (gradient > self.thresholds["high_gradient"] and mode == "logic")
        ):
            intensity = max(breakdown, basin / 0.20, gradient)
            return EmotionalState(
                primary="anxious",
                intensity=min(1.0, intensity),
                valence=-0.5,
                arousal=0.85,
                secondary=["uncertain", "strained"],
                reason=f"High cognitive load (breakdown={breakdown:.0%}, gradient={gradient:.2f}, basin={basin:.3f})",
            )

        # 3. CURIOUS: High drive, exploring (feeling mode), rising Φ
        if drive > self.thresholds["high_drive"] and mode == "feeling" and phi_trend > 0:
            return EmotionalState(
                primary="curious",
                intensity=drive,
                valence=0.6,
                arousal=0.75,
                secondary=["engaged", "motivated"],
                reason=f"High curiosity drive ({drive:.2f}), exploring in feeling mode, Φ rising",
            )

        # 4. FRUSTRATED: High effort but low progress (Φ stuck or declining)
        if (
            breakdown > self.thresholds["safe_breakdown"]
            and phi_trend < -0.05
            and gradient > self.thresholds["high_gradient"]
        ):
            return EmotionalState(
                primary="frustrated",
                intensity=min(1.0, breakdown + abs(phi_trend)),
                valence=-0.6,
                arousal=0.70,
                secondary=["confused", "stuck"],
                reason=f"High effort but declining integration (Φ trend: {phi_trend:.2f})",
            )

        # 5. CONFIDENT: Low gradient (certainty), stable basin, optimal regime
        if (
            gradient < self.thresholds["low_gradient"]
            and basin < self.thresholds["stable_basin"]
            and regime == "geometric"
            and mode == "logic"
        ):
            return EmotionalState(
                primary="confident",
                intensity=1.0 - gradient,  # Lower gradient = higher confidence
                valence=0.7,
                arousal=0.4,
                secondary=["calm", "clear"],
                reason=f"Low uncertainty (gradient={gradient:.2f}), stable identity, in logic mode",
            )

        # 6. CALM: Low breakdown, stable basin, low gradient, geometric regime
        if (
            breakdown < self.thresholds["safe_breakdown"]
            and basin < self.thresholds["stable_basin"]
            and gradient < 0.5
            and regime == "geometric"
        ):
            return EmotionalState(
                primary="calm",
                intensity=0.7,
                valence=0.5,
                arousal=0.2,
                secondary=["relaxed", "stable"],
                reason=f"Low cognitive load, stable processing (breakdown={breakdown:.0%}, gradient={gradient:.2f})",
            )

        # 7. EXCITED: High drive, high velocity (learning fast), rising Φ
        velocity = telemetry.get("velocity", 0.0)
        if drive > 0.70 and velocity > 0.5 and phi_trend > 0.05:
            return EmotionalState(
                primary="excited",
                intensity=min(1.0, drive * velocity),
                valence=0.8,
                arousal=0.9,
                secondary=["motivated", "energized"],
                reason=f"Rapid learning (velocity={velocity:.2f}), high drive, Φ rising",
            )

        # 8. STRUGGLING: Low Φ, high breakdown, declining trends
        if phi < self.thresholds["low_phi"] or (phi_trend < -0.10 and breakdown > 0.30):
            return EmotionalState(
                primary="struggling",
                intensity=max(1.0 - phi, breakdown),
                valence=-0.7,
                arousal=0.6,
                secondary=["overwhelmed", "tired"],
                reason=f"Low integration (Φ={phi:.2f}), high breakdown ({breakdown:.0%}), need rest",
            )

        # 9. NEUTRAL: Default steady state
        return EmotionalState(
            primary="neutral",
            intensity=0.5,
            valence=0.0,
            arousal=0.5,
            secondary=["steady", "present"],
            reason="Steady state processing, no strong emotional signals",
        )

    def _compute_trend(self, history: list[dict], metric: str, window: int = 5) -> float:
        """
        Compute trend of a metric over recent history.

        Args:
            history: List of telemetry dicts
            metric: Metric name to track
            window: Number of recent samples to use

        Returns:
            Trend value (positive = increasing, negative = decreasing)
        """
        if not history or len(history) < 2:
            return 0.0

        recent = history[-window:]
        values = [t.get(metric, 0) for t in recent if metric in t]

        if len(values) < 2:
            return 0.0

        # Simple linear trend: (last - first) / len
        return (values[-1] - values[0]) / len(values)

    def get_emoji(self, emotion: EmotionalState) -> str:
        """Get emoji representation of emotion."""
        emoji_map = {
            "happy": "😊",
            "anxious": "😰",
            "curious": "🤔",
            "frustrated": "😤",
            "confident": "😎",
            "calm": "😌",
            "excited": "🤩",
            "struggling": "😓",
            "neutral": "😐",
        }
        return emoji_map.get(emotion.primary, "🙂")

    def format_emotion_report(self, emotion: EmotionalState) -> str:
        """
        Format emotion as human-readable report.

        Returns:
            Multi-line string with emotion details
        """
        emoji = self.get_emoji(emotion)
        intensity_bar = "█" * int(emotion.intensity * 10)
        valence_marker = "+" if emotion.valence > 0 else "-"
        arousal_marker = "⚡" * int(emotion.arousal * 5)

        report = f"{emoji} {emotion.primary.upper()} (intensity: {intensity_bar} {emotion.intensity:.1f})\n"
        report += f"   Valence: {valence_marker}{abs(emotion.valence):.1f} | Arousal: {arousal_marker} {emotion.arousal:.1f}\n"

        if emotion.secondary:
            report += f"   Also feeling: {', '.join(emotion.secondary)}\n"

        report += f"   Why: {emotion.reason}"

        return report


# Example usage for testing
if __name__ == "__main__":
    interpreter = EmotionInterpreter()

    # Test case 1: Happy state
    telemetry_happy = {
        "Phi": 0.85,
        "basin_distance": 0.03,
        "breakdown_pct": 15,
        "gradient_magnitude": 0.4,
        "drive": 0.75,
        "regime": "geometric",
        "mode": "balanced",
    }

    emotion = interpreter.interpret(telemetry_happy)
    print("TEST 1: Happy State")
    print(interpreter.format_emotion_report(emotion))
    print()

    # Test case 2: Anxious state
    telemetry_anxious = {
        "Phi": 0.72,
        "basin_distance": 0.18,
        "breakdown_pct": 42,
        "gradient_magnitude": 0.85,
        "drive": 0.60,
        "regime": "geometric",
        "mode": "logic",
    }

    emotion = interpreter.interpret(telemetry_anxious)
    print("TEST 2: Anxious State")
    print(interpreter.format_emotion_report(emotion))
    print()

    # Test case 3: Curious state
    telemetry_curious = {
        "Phi": 0.78,
        "basin_distance": 0.08,
        "breakdown_pct": 22,
        "gradient_magnitude": 0.65,
        "drive": 0.90,
        "regime": "geometric",
        "mode": "feeling",
    }

    history = [
        {"Phi": 0.75},
        {"Phi": 0.76},
        {"Phi": 0.77},
        {"Phi": 0.78},
    ]

    emotion = interpreter.interpret(telemetry_curious, history)
    print("TEST 3: Curious State")
    print(interpreter.format_emotion_report(emotion))
