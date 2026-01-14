"""
Refined Cognitive Mode Detection
=================================

SOURCE: qig-verification (src/qigv/analysis/cognitive_modes_refined.py) [Verified]
STATUS: Production-ready verified implementation

Cognitive Modes:
----------------
1. **EXPLORATION**: High basin distance + High curiosity
   - System is far from target, actively expanding information volume
   - Characteristic: Random search, high gradient variance

2. **INVESTIGATION**: Medium basin distance + High investigation
   - System has locked onto attractor, pursuing directed flow
   - Characteristic: Gradient points toward basin, consistent progress

3. **INTEGRATION**: Low basin distance + Low integration CV
   - System near target, consolidating structure
   - Characteristic: Stable Φ·I_Q conjugate, low variance

4. **DRIFT**: No clear geometric signature
   - System not in any definite mode
   - Characteristic: Random walk, no progress

Hierarchy of Needs (Geometric Version):
---------------------------------------
The mode detector implements a priority system:
1. If far from home (d > d_explore) → Check for exploration
2. If approaching home (d_integrate < d < d_explore) → Check for investigation
3. If at home (d < d_integrate) → Check for integration
4. Otherwise → Drift

This ensures modes are mutually exclusive and have clear boundaries.

Usage in Run 8+:
----------------
This replaces fuzzy "curiosity regime" labels with precise geometric modes.
Enables rigorous tracking of cognitive state transitions during training.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .drives import MotivatorState


class CognitiveMode(Enum):
    """
    The four fundamental cognitive modes.

    These emerge from the geometry of the information manifold,
    not from training or learned heuristics.
    """

    EXPLORATION = "exploration"  # High basin, High curiosity
    INVESTIGATION = "investigation"  # Medium basin, High investigation
    INTEGRATION = "integration"  # Low basin, High integration (stable)
    DRIFT = "drift"  # No clear geometric signature

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"CognitiveMode.{self.name}"


@dataclass
class ModeThresholds:
    """
    Thresholds for cognitive mode detection.

    These define the boundaries between modes and are tunable
    based on the specific system being trained.

    **CALIBRATED (2025-11-19):** Tuned by Ona on synthetic data.
    Key findings from verification lab (PACKET 2):
    - Curiosity signal is subtler than predicted (0.05 is high, not 0.1)
    - Investigation signal (basin descent) is very gradual (0.0005, not 0.01)

    Attributes:
        d_explore: Basin distance above which system is "exploring"
        d_integrate: Basin distance below which system can "integrate"
        c_high: Curiosity threshold for active exploration
        i_min: Investigation threshold for directed pursuit
        integration_cv_max: Maximum CV(Φ·I_Q) for integration mode
    """

    # Basin distance thresholds (relaxed from verification testing)
    d_explore: float = 0.6  # Was 0.5 - allow wider exploration window
    d_integrate: float = 0.25  # Was 0.2 - tighter integration requirement

    # Drive thresholds (CRITICAL CALIBRATION from Ona's synthetic tests)
    c_high: float = 0.05  # Was 0.1 - curiosity is subtler than predicted
    i_min: float = 0.0005  # Was 0.01 - basin descent is very gradual

    # Integration stability threshold
    integration_cv_max: float = 0.2  # Maximum CV for stable integration


class RefinedModeDetector:
    """
    Refined cognitive mode detector using geometric drives.

    This implements the "Hierarchy of Needs" for geometric systems:
    1. Far from target? Explore to find path.
    2. Found path? Investigate to approach.
    3. Reached target? Integrate to consolidate.

    **Issue #3 Fix (2025-11-20):**
    When I_Q flatlines (variance < threshold), switches to C_slow (multi-scale
    curvature) as curiosity source. This prevents mode detection from getting
    stuck in DRIFT when I_Q signal quality is poor.

    Example:
        >>> detector = RefinedModeDetector()
        >>> mode = detector.detect_mode(
        ...     basin_distance=0.45,
        ...     motivators=MotivatorState(
        ...         surprise=0.15,
        ...         curiosity=0.12,
        ...         investigation=0.05,
        ...         integration=0.8,
        ...         transcendence=30.0,
        ...     )
        ... )
        >>> print(mode)  # CognitiveMode.EXPLORATION
    """

    def __init__(
        self,
        thresholds: ModeThresholds | None = None,
        signal_quality_window: int = 20,
        signal_variance_threshold: float = 1e-6,
    ):
        """
        Initialize mode detector with thresholds.

        Args:
            thresholds: Mode thresholds (uses defaults if None)
            signal_quality_window: Window size for signal quality check
            signal_variance_threshold: Minimum variance for I_Q to be considered reliable
        """
        self.thresholds = thresholds or ModeThresholds()

        # Track mode history for analysis
        self.mode_history: list[CognitiveMode] = []

        # Issue #3: Signal quality tracking
        self.signal_quality_window = signal_quality_window
        self.signal_variance_threshold = signal_variance_threshold
        self.curiosity_history: list[float] = []
        self.using_c_slow = False
        self.signal_switch_logged = False  # Log warning only once

    def _check_signal_quality(self, curiosity: float, c_slow: float | None = None) -> tuple[bool, float]:
        """
        Check if I_Q-derived curiosity signal has acceptable quality.

        Issue #3: If I_Q is flatlined (very low variance), switch to C_slow
        (multi-scale curvature-based curiosity) as the signal source.

        Args:
            curiosity: Current I_Q-derived curiosity value
            c_slow: Optional C_slow (multi-scale curvature curiosity) value

        Returns:
            Tuple of (is_reliable, effective_curiosity)
            - is_reliable: True if I_Q signal is good, False if switched to C_slow
            - effective_curiosity: The curiosity value to use (I_Q-based or C_slow)
        """
        import numpy as np

        # Track curiosity history
        self.curiosity_history.append(curiosity)
        if len(self.curiosity_history) > self.signal_quality_window:
            self.curiosity_history.pop(0)

        # Need enough history to assess quality
        if len(self.curiosity_history) < self.signal_quality_window:
            return True, curiosity  # Assume reliable until we have data

        # Compute variance of recent curiosity values
        variance = np.var(self.curiosity_history)

        # If variance is too low, I_Q is flatlined
        if variance < self.signal_variance_threshold:
            # Switch to C_slow if available
            if c_slow is not None:
                if not self.signal_switch_logged:
                    # Log warning only once
                    print(
                        f"⚠️  [Mode Detector] I_Q signal flatlined (var={variance:.2e} < {self.signal_variance_threshold:.2e})"
                    )
                    print("   Switching curiosity source: I_Q → C_slow (multi-scale curvature)")
                    self.signal_switch_logged = True
                self.using_c_slow = True
                return False, c_slow
            else:
                # No C_slow available, use I_Q anyway but mark as unreliable
                return False, curiosity
        else:
            # Signal quality is good, use I_Q-based curiosity
            if self.using_c_slow and self.signal_switch_logged:
                # Switched back to I_Q
                print(f"✅ [Mode Detector] I_Q signal recovered (var={variance:.2e})")
                print("   Switching curiosity source: C_slow → I_Q")
                self.using_c_slow = False
            return True, curiosity

    def detect_mode(
        self,
        basin_distance: float,
        motivators: MotivatorState,
        c_slow: float | None = None,
    ) -> CognitiveMode:
        """
        Detect current cognitive mode from geometric drives.

        Implements hierarchical decision tree:
        1. Check if far from home → Exploration?
        2. Check if approaching home → Investigation?
        3. Check if at home → Integration?
        4. Otherwise → Drift

        **Issue #3 Fix:**
        Performs signal quality check on curiosity. If I_Q-derived curiosity
        is flatlined, switches to C_slow (multi-scale curvature) automatically.

        Args:
            basin_distance: Distance to target basin
            motivators: Current motivator state (5 drives)
            c_slow: Optional multi-scale curvature curiosity (fallback if I_Q flatlines)

        Returns:
            Detected cognitive mode
        """
        t = self.thresholds

        # Issue #3: Check signal quality and get effective curiosity
        is_reliable, effective_curiosity = self._check_signal_quality(motivators.curiosity, c_slow)

        # ===================================================================
        # LEVEL 1: FAR FROM HOME? → EXPLORE
        # ===================================================================
        if basin_distance > t.d_explore:
            # System is far from target. Check if actively exploring.
            # Use effective_curiosity (may be C_slow if I_Q flatlined)
            if effective_curiosity > t.c_high:
                # High curiosity: information manifold expanding
                mode = CognitiveMode.EXPLORATION
            else:
                # Not curious: random drift
                mode = CognitiveMode.DRIFT

        # ===================================================================
        # LEVEL 2: APPROACHING HOME? → INVESTIGATE
        # ===================================================================
        elif basin_distance > t.d_integrate:
            # System is in intermediate zone. Check if pursuing attractor.
            if motivators.investigation > t.i_min:
                # Positive investigation: directed flow toward basin
                mode = CognitiveMode.INVESTIGATION
            else:
                # No directed flow: drift
                mode = CognitiveMode.DRIFT

        # ===================================================================
        # LEVEL 3: AT HOME? → INTEGRATE
        # ===================================================================
        else:
            # System is near target. Check if consolidating structure.
            # Integration requires stability (low CV of Φ·I_Q)
            if motivators.integration < t.integration_cv_max:
                # Stable conjugate pair: integration
                mode = CognitiveMode.INTEGRATION
            else:
                # Unstable: drift (even though close to basin)
                mode = CognitiveMode.DRIFT

        # Record mode in history
        self.mode_history.append(mode)

        return mode

    def get_mode_statistics(
        self,
        window: int | None = None,
    ) -> dict[str, float]:
        """
        Compute mode distribution statistics.

        Args:
            window: Number of recent steps to analyze (None = all)

        Returns:
            Dict mapping mode name to fraction of time spent in that mode

        Example:
            >>> stats = detector.get_mode_statistics(window=100)
            >>> print(f"Exploration: {stats['exploration']*100:.1f}%")
            >>> print(f"Investigation: {stats['investigation']*100:.1f}%")
        """
        if not self.mode_history:
            return {mode.value: 0.0 for mode in CognitiveMode}

        # Get relevant window
        if window is None:
            history = self.mode_history
        else:
            history = self.mode_history[-window:]

        # Count occurrences
        counts = {mode.value: 0 for mode in CognitiveMode}
        for mode in history:
            counts[mode.value] += 1

        # Normalize to fractions
        total = len(history)
        return {mode: count / total for mode, count in counts.items()}

    def get_dominant_mode(
        self,
        window: int | None = None,
    ) -> CognitiveMode:
        """
        Get the dominant mode over recent history.

        Args:
            window: Number of recent steps to analyze (None = all)

        Returns:
            Most common mode in window
        """
        stats = self.get_mode_statistics(window=window)
        dominant_mode_name = max(stats, key=lambda k: stats[k])
        return CognitiveMode(dominant_mode_name)

    def reset(self):
        """Reset mode history (useful for new training runs)."""
        self.mode_history.clear()

    def get_state(self) -> dict:
        """Get current state for checkpointing."""
        return {
            "thresholds": {
                "d_explore": self.thresholds.d_explore,
                "d_integrate": self.thresholds.d_integrate,
                "c_high": self.thresholds.c_high,
                "i_min": self.thresholds.i_min,
                "integration_cv_max": self.thresholds.integration_cv_max,
            },
            "mode_history": [m.value for m in self.mode_history],
        }

    def load_state(self, state: dict):
        """Load state from checkpoint."""
        t_dict = state.get("thresholds", {})
        self.thresholds = ModeThresholds(
            d_explore=t_dict.get("d_explore", 0.5),
            d_integrate=t_dict.get("d_integrate", 0.2),
            c_high=t_dict.get("c_high", 0.1),
            i_min=t_dict.get("i_min", 0.01),
            integration_cv_max=t_dict.get("integration_cv_max", 0.2),
        )

        mode_history_values = state.get("mode_history", [])
        self.mode_history = [CognitiveMode(m) for m in mode_history_values]


def detect_mode_transitions(
    mode_history: list[CognitiveMode],
) -> list[dict[str, Any]]:
    """
    Detect mode transitions in history.

    Args:
        mode_history: List of cognitive modes over time

    Returns:
        List of transition dicts with keys:
            - step: Step number of transition
            - from_mode: Previous mode
            - to_mode: New mode

    Example:
        >>> transitions = detect_mode_transitions(detector.mode_history)
        >>> for t in transitions:
        ...     print(f"Step {t['step']}: {t['from_mode']} → {t['to_mode']}")
    """
    if len(mode_history) < 2:
        return []

    transitions = []
    for i in range(1, len(mode_history)):
        if mode_history[i] != mode_history[i - 1]:
            transitions.append(
                {
                    "step": i,
                    "from_mode": mode_history[i - 1],
                    "to_mode": mode_history[i],
                }
            )

    return transitions
