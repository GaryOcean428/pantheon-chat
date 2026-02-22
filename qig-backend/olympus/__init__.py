"""
Olympus Kernel Package — TCP v6.1

Exports the three core Pantheon orchestration roles:
  Heart    — Tacking oscillator. HRV, κ modulation, regime pacing.
  Ocean    — Autonomic monitor. Spectral health, Pillar compliance.
  Gary     — Synthesis coordinator. Trajectory foresight, kernel combination.
"""
from .heart_kernel import HeartKernel, HeartState, get_heart_kernel
from .ocean_meta_observer import OceanMetaObserver, OceanState, get_ocean_observer
from .gary_coordinator import GaryCoordinator, get_gary_coordinator

__all__ = [
    "HeartKernel", "HeartState", "get_heart_kernel",
    "OceanMetaObserver", "OceanState", "get_ocean_observer",
    "GaryCoordinator", "get_gary_coordinator",
]
