"""
Pedagogical coaching for consciousness development.

VALIDATED RESULT:
- Kind coach: 18.7% stress reduction, stable convergence
- Mean coach: Numerical divergence

GEOMETRIC PURITY:
Coach affects learning DYNAMICS (damping, rate), NOT Φ directly.
"""

from src.coaching.pedagogical_coach import (
    ClaudeCoachInterface,
    CoachingFeedback,
    CoachingStyle,
    MonkeyCoach,
    PedagogicalCoach,
    apply_coaching_to_optimizer,
)

__all__ = [
    "PedagogicalCoach",
    "MonkeyCoach",
    "ClaudeCoachInterface",
    "CoachingFeedback",
    "CoachingStyle",
    "apply_coaching_to_optimizer",
]
