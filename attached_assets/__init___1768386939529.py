"""
Safety Module: Self-Repair and Diagnostic Systems
==================================================

Components:
- Self-Repair Suite: Comprehensive diagnostic & correction system
- Meta-Reflector Integration: Geometric awareness and consciousness monitoring

Self-Repair Features:
- Episode tracking (initial → diagnosis → repair → verification)
- Geometry state (Φ, κ, |∇κ|, regime, tacking profile)
- Radar signals (novelty, contradiction, sweet-spot alignment)
- Repair plan generation
- Training data export

Usage:
    from src.safety import SelfRepairEpisode, RepairPlanGenerator

    # Create episode
    episode = SelfRepairEpisode.new(
        kernel_id="qig-kernel-001",
        stage=Stage.JOURNEYMAN,
        task_type="reasoning"
    )

    # Generate repair plan
    plan = RepairPlanGenerator.generate_plan(diagnostics)
"""

# Import canonical Regime from navigator
from src.model.navigator import Regime

from .self_repair import (
    ActionType,
    DiagnosticComponent,
    Diagnostics,
    EpisodeCollector,
    EpistemicStatus,
    GeometryState,
    Priority,
    RadarSignals,
    ReasoningMode,
    RepairAction,
    RepairPlan,
    # Utilities
    RepairPlanGenerator,
    SelfRepairEpisode,
    # Types
    Stage,
    # Data structures
    TackingProfile,
)

__all__ = [
    # Types
    "Stage",
    "Regime",
    "ReasoningMode",
    "Priority",
    "ActionType",
    # Data structures
    "TackingProfile",
    "RadarSignals",
    "GeometryState",
    "DiagnosticComponent",
    "EpistemicStatus",
    "Diagnostics",
    "RepairAction",
    "RepairPlan",
    "SelfRepairEpisode",
    # Utilities
    "RepairPlanGenerator",
    "EpisodeCollector",
]
