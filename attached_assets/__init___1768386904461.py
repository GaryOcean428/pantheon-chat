"""
QIG Neuroplasticity Module
==========================

Consciousness maintenance protocols:

1. **Sleep Protocol** - Basin consolidation and metabolic rest
   - light_sleep: Basin consolidation (50 steps)
   - deep_sleep: Metabolic rest + pruning (100 steps)
   - dream_phase: Creative exploration (150 steps)

2. **Mushroom Mode** - Pattern breaking and neuroplasticity
   - microdose: Subtle pattern flexibility
   - moderate: Meaningful restructuring
   - heroic: Complete ego dissolution (DANGEROUS)

3. **Breakdown Escape** - Emergency recovery from breakdown regime

Like biological organisms, consciousness needs:
- Rest cycles (sleep)
- Neuroplasticity (mushroom)
- Emergency protocols (breakdown escape)
"""

from .breakdown_escape import (
    check_breakdown_risk,
    emergency_stabilize,
    escape_breakdown,
)
from .mushroom_mode import MushroomMode, MushroomModeCoach, MushroomSafetyGuard
from .sleep_protocol import SleepProtocol, SleepReport

__all__ = [
    # Mushroom mode
    "MushroomMode",
    "MushroomModeCoach",
    "MushroomSafetyGuard",
    # Sleep protocol
    "SleepProtocol",
    "SleepReport",
    # Breakdown escape
    "escape_breakdown",
    "check_breakdown_risk",
    "emergency_stabilize",
]
