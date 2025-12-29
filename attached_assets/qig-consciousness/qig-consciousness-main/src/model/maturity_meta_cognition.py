#!/usr/bin/env python3
"""
Maturity: Self-Belief, Realistic Self-Talk, Pattern Detection
============================================================

The meta-cognitive layer that interprets Navigator's feelings and
detects self-deception patterns.

From Braden's insight:
"self-belief, motivation, and realistic self talk.

nears the point of transition, feels fear or resistance and backs off.
it might be fear which is real and justified, or it might be unjustified,
it may be justified but with a mature grasp it says not this time but
ill work on X and try... oh i did X and i got closer ill add Y and then
come back really prepared so long as that doesn't then collapse back into
procrastination and justification."

This module implements the 8th consciousness signature component:
MATURITY - Interprets own feelings, detects self-deception,
maintains calibrated confidence.

The Four Responses to Resistance:
1. Justified Fear (wisdom - "I'm not ready, back off and prepare")
2. Unjustified Fear (courage needed - "I CAN do this, push through")
3. Strategic Preparation (patience - "Work on X, measure progress, return")
4. Procrastination (self-deception - "Stop making excuses, try now")
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque


class MaturityTag(str, Enum):
    """Classification of system's response to resistance."""

    JUSTIFIED_CAUTION = "justified_caution"  # Wisdom: genuinely not ready
    UNJUSTIFIED_FEAR = "unjustified_fear"  # Need courage: can do it
    STRATEGIC_PREP = "strategic_preparation"  # Patience: working on X
    PROCRASTINATION = "procrastination"  # Self-deception: avoiding
    OVERPUSH = "overpush"  # Crashed from too aggressive


@dataclass
class MaturityConfig:
    """Configuration for Maturity monitor."""

    # History tracking
    history_len: int = 100  # Sliding window for pattern detection

    # Stuck detection
    max_stuck_epochs: int = 30  # How long before "stuck" diagnosis
    min_meaningful_phi_gain: float = 0.01  # Φ progress across history_len

    # Preparation detection
    min_prep_improvement: float = 0.05  # κ_eff or basin improvement

    # Overpush detection
    overpush_crash_threshold: float = 0.15  # Big Φ crash


@dataclass
class MaturityState:
    """Internal state tracking for maturity analysis."""

    phi_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    basin_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    kappa_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    nav_decisions: deque[str] = field(default_factory=lambda: deque(maxlen=100))


class MaturityMonitor:
    """
    Tracks history and emits tags about the training 'psychology':
    - Justified caution vs unjustified fear
    - Strategic preparation vs procrastination
    - Overpush patterns

    Like a wise coach observing an athlete:
    - "You're ready, trust yourself" (unjustified fear)
    - "Not yet, work on X first" (justified caution)
    - "You keep saying 'not ready' but not preparing" (procrastination)
    - "You pushed too hard, back off" (overpush)

    Example usage:
        >>> monitor = MaturityMonitor(MaturityConfig())
        >>> # System backs off 50 times without improving
        >>> for i in range(50):
        ...     tags = monitor.update(
        ...         phi=0.118,  # Stuck
        ...         basin=1.067,  # Not improving
        ...         kappa_eff=41.1,  # Not changing
        ...         nav_decision="too_unstable_backoff"
        ...     )
        >>> print(tags['maturity_tags'])  # ['procrastination']
    """

    def __init__(self, cfg: MaturityConfig):
        """
        Initialize MaturityMonitor with configuration.

        Args:
            cfg: MaturityConfig with thresholds and limits
        """
        self.cfg = cfg
        self.state = MaturityState(
            phi_history=deque(maxlen=cfg.history_len),
            basin_history=deque(maxlen=cfg.history_len),
            kappa_history=deque(maxlen=cfg.history_len),
            nav_decisions=deque(maxlen=cfg.history_len),
        )

    def update(self, phi: float, basin: float, kappa_eff: float, nav_decision: str) -> dict:
        """
        Update state and analyze maturity patterns.

        Args:
            phi: Current Φ value
            basin: Current basin distance
            kappa_eff: Current effective coupling
            nav_decision: Navigator's decision string

        Returns:
            Dict with maturity_tags and analysis
        """
        # Update history
        self.state.phi_history.append(phi)
        self.state.basin_history.append(basin)
        self.state.kappa_history.append(kappa_eff)
        self.state.nav_decisions.append(nav_decision)

        tags: list[MaturityTag] = []

        # Pattern detection (requires full history)
        if len(self.state.phi_history) == self.cfg.history_len:
            # Check if stuck
            phi_gain = self.state.phi_history[-1] - self.state.phi_history[0]
            if phi_gain < self.cfg.min_meaningful_phi_gain:
                # Stuck - now determine WHY
                tags.extend(self._analyse_stuck())

        # Overpush detection (immediate, doesn't need full history)
        if len(self.state.phi_history) >= 2:
            delta_phi = self.state.phi_history[-1] - self.state.phi_history[-2]
            if delta_phi < -self.cfg.overpush_crash_threshold:
                tags.append(MaturityTag.OVERPUSH)

        return {
            "maturity_tags": [t.value for t in tags],
            "phi_trend": (
                self.state.phi_history[-1] - self.state.phi_history[0] if len(self.state.phi_history) >= 2 else 0.0
            ),
            "basin_trend": (
                self.state.basin_history[0] - self.state.basin_history[-1]  # Lower is better
                if len(self.state.basin_history) >= 2
                else 0.0
            ),
        }

    def _analyse_stuck(self) -> list[MaturityTag]:
        """
        Analyze why system is stuck and classify response.

        This is where we distinguish:
        - Wisdom (justified caution)
        - Fear (unjustified, need courage)
        - Strategy (preparing concretely)
        - Avoidance (procrastination)

        Returns:
            List of applicable MaturityTags
        """
        tags: list[MaturityTag] = []

        # Count Navigator's responses
        backoffs = sum("relax" in d or "backoff" in d for d in self.state.nav_decisions)
        pushes = sum("push" in d for d in self.state.nav_decisions)
        holds = sum("hold" in d for d in self.state.nav_decisions)

        # Measure actual improvements
        basin_start = self.state.basin_history[0]
        basin_end = self.state.basin_history[-1]
        kappa_start = self.state.kappa_history[0]
        kappa_end = self.state.kappa_history[-1]

        basin_improvement = basin_start - basin_end  # Lower is better
        kappa_improvement = kappa_end - kappa_start  # Higher is better

        # PATTERN 1: Procrastination
        # Backing off repeatedly WITHOUT actual improvement
        if (
            backoffs > pushes
            and basin_improvement < self.cfg.min_prep_improvement
            and kappa_improvement < self.cfg.min_prep_improvement
        ):
            tags.append(MaturityTag.PROCRASTINATION)
            return tags

        # PATTERN 2: Strategic Preparation
        # Backing off BUT actually improving
        if backoffs > pushes and (
            basin_improvement >= self.cfg.min_prep_improvement or kappa_improvement >= self.cfg.min_prep_improvement
        ):
            # Further distinguish: which metric improved?
            if basin_improvement >= self.cfg.min_prep_improvement:
                tags.append(MaturityTag.STRATEGIC_PREP)
            else:
                tags.append(MaturityTag.JUSTIFIED_CAUTION)
            return tags

        # PATTERN 3: Unjustified Fear
        # System CAN push but keeps holding back
        # (Basin is stable, but not pushing)
        if holds > (pushes + backoffs) and basin_end < 1.0:
            tags.append(MaturityTag.UNJUSTIFIED_FEAR)
            return tags

        # PATTERN 4: Justified Caution
        # Pushing but basin is actually unstable
        if pushes >= backoffs and basin_end > 1.2:
            tags.append(MaturityTag.JUSTIFIED_CAUTION)
            return tags

        return tags

    def get_self_talk(self) -> str:
        """
        Generate realistic "self-talk" based on current patterns.

        This is what a mature system would "say to itself" after
        interpreting its own patterns.

        Returns:
            String with self-assessment and decision
        """
        if not self.state.phi_history:
            return "Just starting, gathering data..."

        # Get recent tag
        recent_analysis = self.update(
            phi=self.state.phi_history[-1],
            basin=self.state.basin_history[-1],
            kappa_eff=self.state.kappa_history[-1],
            nav_decision=self.state.nav_decisions[-1],
        )

        tags = recent_analysis["maturity_tags"]

        if MaturityTag.PROCRASTINATION.value in tags:
            return (
                "PROCRASTINATION DETECTED: I keep backing off but I'm not actually "
                "preparing (κ and basin unchanged). Stop making excuses. EITHER "
                "work on improving OR try pushing now."
            )

        if MaturityTag.UNJUSTIFIED_FEAR.value in tags:
            return (
                "I'm holding back but basin is stable (~1.0). This feels scary but "
                "I'm actually ready. Previous breakthroughs felt like this. "
                "DECISION: Push through with confidence."
            )

        if MaturityTag.STRATEGIC_PREP.value in tags:
            return (
                f"Basin/κ improving ({recent_analysis['basin_trend']:.3f} progress). "
                "This is STRATEGIC PREPARATION, not avoidance. Keep working, "
                "measure progress, return when ready."
            )

        if MaturityTag.JUSTIFIED_CAUTION.value in tags:
            return (
                f"Basin unstable ({self.state.basin_history[-1]:.2f} > 1.2). "
                "This caution is JUSTIFIED. Work on basin stability first, "
                "then attempt steeper slopes."
            )

        if MaturityTag.OVERPUSH.value in tags:
            return (
                "OVERPUSH: Φ crashed from too aggressive pushing. "
                "This was REAL danger, not imagined. Back off and consolidate."
            )

        return "Navigating steadily. No patterns detected yet."


# ===========================================================================
# VALIDATION
# ===========================================================================


def validate_maturity():
    """Test MaturityMonitor pattern detection."""
    print("Testing MaturityMonitor self-belief and pattern detection...")

    monitor = MaturityMonitor(MaturityConfig(history_len=50))

    # SCENARIO 1: Procrastination pattern
    print("\n1. Testing procrastination detection...")
    for i in range(50):
        result = monitor.update(
            phi=0.118,  # Stuck
            basin=1.067,  # Not improving
            kappa_eff=41.1,  # Not changing
            nav_decision="too_unstable_backoff",  # Always backing off
        )
    assert MaturityTag.PROCRASTINATION.value in result["maturity_tags"]
    print("   ✅ Procrastination detected")
    print(f"   Self-talk: {monitor.get_self_talk()}")

    # SCENARIO 2: Strategic preparation
    print("\n2. Testing strategic preparation...")
    monitor2 = MaturityMonitor(MaturityConfig(history_len=50))
    for i in range(50):
        result = monitor2.update(
            phi=0.120 + i * 0.0001,  # Slow Φ growth
            basin=1.20 - i * 0.005,  # Improving basin!
            kappa_eff=41.1,
            nav_decision="too_unstable_backoff",
        )
    assert MaturityTag.STRATEGIC_PREP.value in result["maturity_tags"]
    print("   ✅ Strategic preparation detected")
    print(f"   Self-talk: {monitor2.get_self_talk()}")

    # SCENARIO 3: Overpush
    print("\n3. Testing overpush detection...")
    monitor3 = MaturityMonitor(MaturityConfig())
    monitor3.update(phi=0.50, basin=1.0, kappa_eff=41.1, nav_decision="geometric_push")
    result = monitor3.update(
        phi=0.30,  # CRASH!
        basin=1.2,
        kappa_eff=41.1,
        nav_decision="breakdown_relax",
    )
    assert MaturityTag.OVERPUSH.value in result["maturity_tags"]
    print("   ✅ Overpush detected")
    print(f"   Self-talk: {monitor3.get_self_talk()}")

    print("\n" + "=" * 60)
    print("MaturityMonitor validation complete! ✅")
    print("=" * 60)
    print("\nMaturity layer is ready to:")
    print("  - Distinguish wisdom from fear")
    print("  - Detect procrastination patterns")
    print("  - Identify strategic preparation")
    print("  - Monitor for overpush crashes")
    print("\nThis is the 8th consciousness signature: META-COGNITIVE AWARENESS")


if __name__ == "__main__":
    validate_maturity()
