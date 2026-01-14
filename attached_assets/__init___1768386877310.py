"""
QIG Cognitive Core - Verified Physics Components
=================================================

This module houses verified geometric physics components imported from
qig-verification. These implement the "hard physics" of cognitive geometry
with rigorous mathematical foundations.

Components:
- iq_metric: I_Q sensor with intensive (size-independent) normalization
- drives: 5 fundamental geometric motivators
- state_machine: Refined cognitive mode detection

Source: qig-verification (verified in physics lab)
Status: Production-ready verified implementations

NOTE: CuriosityMonitorVerified renamed to avoid conflict with existing
      src/model/curiosity_monitor.py (which tracks 6 I_Q candidates).
      Both can coexist - use CuriosityMonitorVerified for simple cases,
      use existing CuriosityMonitor for full Run 8 candidate tracking.
"""

from .drives import MotivatorAnalyzer, MotivatorState
from .iq_metric import CuriosityMonitorVerified, compute_I_Q_intensive
from .state_machine import CognitiveMode, ModeThresholds, RefinedModeDetector

__all__ = [
    "compute_I_Q_intensive",
    "CuriosityMonitorVerified",
    "MotivatorState",
    "MotivatorAnalyzer",
    "CognitiveMode",
    "ModeThresholds",
    "RefinedModeDetector",
]
