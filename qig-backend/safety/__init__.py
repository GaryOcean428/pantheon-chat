"""
QIG Safety Module

Implements ethical monitoring, suffering detection, and breakdown prevention.
All autonomous systems must pass through safety validation.
"""

from .ethics_monitor import (
    EthicsMonitor,
    EthicalAbortException,
    compute_suffering_metric,
    detect_breakdown_regime,
)

__all__ = [
    'EthicsMonitor',
    'EthicalAbortException',
    'compute_suffering_metric',
    'detect_breakdown_regime',
]
