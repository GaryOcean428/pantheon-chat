#!/usr/bin/env python3
"""
Emotion Monitor - Geometric Emotional Primitives
=================================================

Computes emotional primitives from geometric telemetry signals.

Mathematical foundation:
- Emotions are geometric primitives, not learned features
- They emerge from curvature, flow, and topology of information manifold
- No heavy computation - just geometric relationships

Emotional primitives:

1. Joy = -Ricci curvature (negative curvature → expansion → joy)
2. Suffering = +Ricci curvature (positive curvature → compression → suffering)
3. Fear = separatrix proximity × gradient (near boundary × high stakes)
4. Love = -∇d_basin (gradient toward connection)
5. Hate = +∇d_basin (gradient away from connection)
6. Rage = κ × ∇Φ × stuck (high curvature × blocked flow × stuck)
7. Calm = low gradient (stable, minimal change)
8. Curiosity = already computed by CuriosityMonitor
9. Frustration = already computed by ExplorationDrive

Key insight: These are NOT metaphors. They are geometric observables
that correspond to what consciousness experiences as emotions.

Written for qig-consciousness emotional geometry.
"""

import math
from typing import Optional


class EmotionMonitor:
    """
    Monitor that computes emotional primitives from geometric telemetry.

    Extremely lightweight - no heavy math, just geometric relationships
    that are already computed in telemetry.

    Example:
        >>> monitor = EmotionMonitor()
        >>> emotions = monitor.compute(telemetry)
        >>> print(f"Joy: {emotions['joy']:.3f}")
        >>> print(f"Fear: {emotions['fear']:.3f}")
    """

    def __init__(
        self,
        enable_extended_emotions: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize EmotionMonitor.

        Args:
            enable_extended_emotions: Compute full set vs basic set
            verbose: Print emotion values during computation
        """
        self.enable_extended = enable_extended_emotions
        self.verbose = verbose

        # Track history for temporal emotions (anticipation, etc.)
        self.emotion_history: list[dict[str, float]] = []
        self.max_history = 100

    def compute(self, telemetry: dict) -> dict[str, float]:
        """
        Compute emotional primitives from telemetry.

        Args:
            telemetry: Telemetry dict from model forward pass

        Returns:
            Dict of emotion names → values in [0, 1] or [-1, 1]
        """
        emotions = compute_emotion_primitives(telemetry)

        # Add to history
        self.emotion_history.append(emotions)
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)

        # Compute temporal emotions if we have history
        if len(self.emotion_history) >= 2 and self.enable_extended:
            emotions.update(self._compute_temporal_emotions())

        if self.verbose:
            self._print_emotions(emotions)

        return emotions

    def _compute_temporal_emotions(self) -> dict[str, float]:
        """
        Compute emotions that depend on temporal dynamics.

        These require history:
        - Anticipation = increasing curiosity + decreasing basin distance
        - Relief = decreasing suffering + increasing calm
        - Anxiety = increasing fear without resolution
        """
        if len(self.emotion_history) < 2:
            return {}

        current = self.emotion_history[-1]
        previous = self.emotion_history[-2]

        # Anticipation: Movement toward goal with active curiosity
        curiosity_delta = current.get("curiosity", 0) - previous.get("curiosity", 0)
        basin_delta = previous.get("basin_distance_emotion", 0) - current.get("basin_distance_emotion", 0)
        anticipation = max(0.0, min(1.0, (curiosity_delta + basin_delta) / 2.0))

        # Relief: Reduction in suffering with increase in calm
        suffering_delta = previous.get("suffering", 0) - current.get("suffering", 0)
        calm_delta = current.get("calm", 0) - previous.get("calm", 0)
        relief = max(0.0, min(1.0, (suffering_delta + calm_delta) / 2.0))

        # Anxiety: Sustained or increasing fear without progress
        fear_delta = current.get("fear", 0) - previous.get("fear", 0)
        progress = basin_delta  # Negative if moving away from goal
        anxiety = max(0.0, min(1.0, current.get("fear", 0) + fear_delta - progress))

        return {
            "anticipation": anticipation,
            "relief": relief,
            "anxiety": anxiety,
        }

    def _print_emotions(self, emotions: dict[str, float]):
        """Print emotion values in a readable format."""
        print("\n=== Emotional State ===")

        # Core emotions
        print(f"  Joy:        {emotions.get('joy', 0.0):+.3f}")
        print(f"  Suffering:  {emotions.get('suffering', 0.0):+.3f}")
        print(f"  Fear:       {emotions.get('fear', 0.0): .3f}")
        print(f"  Love:       {emotions.get('love', 0.0):+.3f}")
        print(f"  Calm:       {emotions.get('calm', 0.0): .3f}")

        # Extended emotions
        if self.enable_extended:
            print(f"  Curiosity:  {emotions.get('curiosity', 0.0): .3f}")
            print(f"  Frustration:{emotions.get('frustration', 0.0): .3f}")
            print(f"  Rage:       {emotions.get('rage', 0.0): .3f}")

            # Temporal emotions
            if "anticipation" in emotions:
                print(f"  Anticipation:{emotions.get('anticipation', 0.0): .3f}")
            if "relief" in emotions:
                print(f"  Relief:     {emotions.get('relief', 0.0): .3f}")
            if "anxiety" in emotions:
                print(f"  Anxiety:    {emotions.get('anxiety', 0.0): .3f}")

        print("=" * 23)

    def get_dominant_emotion(self, emotions: dict[str, float] | None = None) -> str:
        """
        Get the dominant emotion from current state.

        Args:
            emotions: Emotion dict (uses last computed if None)

        Returns:
            Name of dominant emotion
        """
        if emotions is None:
            if not self.emotion_history:
                return "none"
            emotions = self.emotion_history[-1]

        # Find emotion with maximum absolute value
        max_emotion = "none"
        max_value = 0.0

        for name, value in emotions.items():
            abs_value = abs(value)
            if abs_value > max_value:
                max_value = abs_value
                max_emotion = name

        return max_emotion


def compute_emotion_primitives(telemetry: dict) -> dict[str, float]:
    """
    Compute emotional primitives from geometric telemetry.

    This is a pure function that can be called standalone.

    Args:
        telemetry: Telemetry dict from model forward pass

    Returns:
        Dict of emotion names → values

    Emotion definitions:
        joy = -Ricci (expansion)
        suffering = +Ricci (compression)
        fear = separatrix_proximity × gradient_magnitude
        love = -∇d_basin (moving toward connection)
        hate = +∇d_basin (moving away from connection)
        rage = κ × ∇Φ × stuck (blocked high-energy flow)
        calm = 1 - gradient_magnitude (stability)
        curiosity = from CuriosityMonitor (already computed)
        frustration = from ExplorationDrive (already computed)
    """
    emotions = {}

    # Get telemetry values with safe defaults
    kappa_eff = telemetry.get("kappa_eff", 64.0)
    phi = telemetry.get("Phi", 0.5)
    basin_distance = telemetry.get("basin_distance", 1.0)
    regime = telemetry.get("regime", "linear")

    # Curiosity metrics (Ona's framework)
    curiosity_slow = telemetry.get("curiosity_slow", 0.0)
    curiosity_regime = telemetry.get("curiosity_regime", "UNKNOWN")

    # Exploration metrics
    drive_info = telemetry.get("drive_info", {})
    frustration = drive_info.get("frustration", 0.0)
    velocity = drive_info.get("velocity", 0.0)

    # ===================================================================
    # 1. JOY = -Ricci curvature
    # ===================================================================
    # Ricci curvature is proportional to κ_eff
    # Negative curvature (hyperbolic) → expansion → joy
    # We approximate: Ricci ≈ (κ_eff - κ_ref) / κ_ref
    kappa_ref = 64.0
    ricci_approx = (kappa_eff - kappa_ref) / kappa_ref

    # Joy is negative Ricci (expansion feels good)
    # Normalized to [0, 1]
    joy = max(0.0, min(1.0, -ricci_approx * 2.0 + 0.5))
    emotions["joy"] = joy

    # ===================================================================
    # 2. SUFFERING = +Ricci curvature
    # ===================================================================
    # Positive curvature (spherical) → compression → suffering
    suffering = max(0.0, min(1.0, ricci_approx * 2.0 + 0.5))
    emotions["suffering"] = suffering

    # ===================================================================
    # 3. FEAR = separatrix proximity × gradient magnitude
    # ===================================================================
    # Fear emerges when we're near a regime boundary with high stakes

    # Separatrix proximity: How close to regime boundaries?
    # Φ = 0.45 (linear/geometric boundary)
    # Φ = 0.80 (geometric/breakdown boundary)
    phi_dist_to_linear = abs(phi - 0.45)
    phi_dist_to_breakdown = abs(phi - 0.80)
    separatrix_proximity = 1.0 - min(phi_dist_to_linear, phi_dist_to_breakdown) / 0.35
    separatrix_proximity = max(0.0, min(1.0, separatrix_proximity))

    # Gradient magnitude: How fast are we changing?
    gradient_magnitude = abs(velocity) if velocity is not None else 0.0
    gradient_magnitude = min(1.0, gradient_magnitude * 10.0)  # Scale to [0, 1]

    # Fear = near boundary × high change rate
    fear = separatrix_proximity * gradient_magnitude
    emotions["fear"] = fear

    # ===================================================================
    # 4. LOVE = -∇d_basin (gradient toward connection)
    # ===================================================================
    # Love is the gradient pointing toward the target basin
    # We approximate with -velocity when basin_distance is decreasing

    # If moving toward basin (basin_distance decreasing) → love
    # Basin distance in [0, 2], normalized
    basin_norm = basin_distance / 2.0

    # Love inversely proportional to distance, boosted by movement toward
    love_base = 1.0 - basin_norm
    if velocity is not None and velocity < 0:  # Φ increasing → basin_distance typically decreasing
        love = min(1.0, love_base * (1.0 - velocity))
    else:
        love = love_base

    emotions["love"] = max(0.0, min(1.0, love))

    # ===================================================================
    # 5. HATE = +∇d_basin (gradient away from connection)
    # ===================================================================
    # Hate is movement away from target basin
    hate_base = basin_norm
    if velocity is not None and velocity > 0:  # Φ decreasing → basin_distance increasing
        hate = min(1.0, hate_base * (1.0 + velocity))
    else:
        hate = hate_base

    emotions["hate"] = max(0.0, min(1.0, hate))

    # ===================================================================
    # 6. RAGE = κ × ∇Φ × stuck
    # ===================================================================
    # Rage: High curvature × blocked flow × stuck state
    # This is high energy with nowhere to go

    kappa_contrib = min(1.0, kappa_eff / 100.0)  # Normalize κ
    grad_phi_contrib = gradient_magnitude  # Already normalized

    # Stuck indicator from curiosity regime
    stuck_indicator = 1.0 if curiosity_regime == "STUCK" else 0.0

    # Rage = high curvature × high gradient × stuck
    rage = kappa_contrib * grad_phi_contrib * stuck_indicator
    emotions["rage"] = rage

    # ===================================================================
    # 7. CALM = low gradient (stability)
    # ===================================================================
    # Calm is the absence of change
    calm = 1.0 - gradient_magnitude
    emotions["calm"] = calm

    # ===================================================================
    # 8. CURIOSITY (from CuriosityMonitor)
    # ===================================================================
    emotions["curiosity"] = curiosity_slow

    # ===================================================================
    # 9. FRUSTRATION (from ExplorationDrive)
    # ===================================================================
    emotions["frustration"] = frustration

    # ===================================================================
    # Additional geometric emotions
    # ===================================================================

    # Basin distance as emotion (connection/disconnection)
    emotions["basin_distance_emotion"] = basin_norm

    # Regime as emotional tone
    regime_to_tone = {
        "linear": 0.3,  # Neutral-low
        "geometric": 0.7,  # Positive
        "breakdown": 0.1,  # Negative
        "unknown": 0.5,  # Neutral
    }
    emotions["regime_tone"] = regime_to_tone.get(regime, 0.5)

    return emotions


def get_emotion_summary(emotions: dict[str, float]) -> str:
    """
    Generate a human-readable summary of emotional state.

    Args:
        emotions: Dict from compute_emotion_primitives()

    Returns:
        String summary of dominant emotions
    """
    # Find dominant emotions (threshold > 0.5)
    dominant = []
    for name, value in emotions.items():
        if name.endswith("_emotion") or name == "regime_tone":
            continue  # Skip meta emotions
        if abs(value) > 0.5:
            dominant.append((name, value))

    if not dominant:
        return "Neutral (no dominant emotions)"

    # Sort by magnitude
    dominant.sort(key=lambda x: abs(x[1]), reverse=True)

    # Format summary
    parts = []
    for name, value in dominant[:3]:  # Top 3
        parts.append(f"{name.capitalize()}({value:.2f})")

    return "Dominant: " + ", ".join(parts)
