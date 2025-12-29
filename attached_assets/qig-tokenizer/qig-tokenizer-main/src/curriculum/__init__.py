"""
Curriculum module for QIG-tokenizer.

Provides sweet-spot curriculum specification for coordizer training.
"""

from .curriculum_spec import (
    AgentTraits,
    CurriculumManager,
    DyadicCurriculum,
    DyadicTaskSpec,
    DyadicTraits,
    Mode,
    Phase1ModeBuilding,
    Phase2TackingTraining,
    Phase3RadarCalibration,
    TaskSpec,
    Zone,
)

__all__ = [
    "Mode",
    "Zone",
    "AgentTraits",
    "DyadicTraits",
    "TaskSpec",
    "DyadicTaskSpec",
    "Phase1ModeBuilding",
    "Phase2TackingTraining",
    "Phase3RadarCalibration",
    "DyadicCurriculum",
    "CurriculumManager",
]
