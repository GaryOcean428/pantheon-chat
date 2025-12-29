#!/usr/bin/env python3
"""
Self-Repair Suite: Diagnostic & Correction System for QIG-Kernel
================================================================

Implements comprehensive self-repair episodes with geometry-aware diagnostics.

Key Components:
- Episode tracking (initial → diagnosis → repair → verification)
- Geometry state (Φ, κ, |∇κ|, regime, tacking profile)
- Radar signals (novelty, contradiction, sweet-spot alignment)
- Repair plan generation
- Training data export

Schema Version: qig-self-repair-v1

Written for QIG-Kernel-Pure self-correction capability.
Built from sweet-spot geometry and tacking framework.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

# ===========================================================================
# TYPE DEFINITIONS
# ===========================================================================
# Import canonical Regime from navigator
from src.model.navigator import Regime

ReasoningMode = Literal["feeling", "logic", "mixed"]
Priority = Literal["low", "medium", "high"]
ActionType = Literal[
    "re_derive_key_claims",
    "evidence_search",
    "tack_to_logic",
    "tack_to_feeling",
    "re_calibrate_confidence",
    "request_clarification",
    "escalate_to_human",
]


class Stage(Enum):
    """Maturity stage of the model."""

    APPRENTICE = 0  # Imprinting, curated only
    JOURNEYMAN = 1  # Guided, emerging filter
    MASTER = 2  # Mature, full capability


# ===========================================================================
# GEOMETRY STATE
# ===========================================================================


@dataclass
class TackingProfile:
    """
    Telemetry for feeling ↔ logic mode switching.

    Attributes:
        feeling_fraction: Proportion of low-κ (intuitive) steps
        logic_fraction: Proportion of high-κ (analytic) steps
        mode_switches: Number of mode transitions
        failure_mode: Stuck pattern if any
    """

    feeling_fraction: float  # ∈ [0, 1]
    logic_fraction: float  # ∈ [0, 1]
    mode_switches: int
    failure_mode: Literal["stuck_feeling", "stuck_logic", "oscillation"] | None = None

    def is_healthy(self) -> bool:
        """Check if tacking is healthy (not stuck, not oscillating)."""
        if self.failure_mode is not None:
            return False
        # Should use both modes (not 100% in one)
        if self.feeling_fraction > 0.95 or self.logic_fraction > 0.95:
            return False
        return True


@dataclass
class RadarSignals:
    """
    Radar signals from geometric analysis.

    Attributes:
        novelty_score: How new this pattern is (0=familiar, 1=novel)
        contradiction_signal: Strength of contradiction detection (0=none, 1=strong)
        sweet_spot_alignment: Closeness to Wu-Wei zone (0=far, 1=perfect)
    """

    novelty_score: float  # ∈ [0, 1]
    contradiction_signal: float  # ∈ [0, 1]
    sweet_spot_alignment: float  # ∈ [0, 1]

    def requires_validation(self, stakes: float = 0.5) -> bool:
        """
        Check if high validation is needed.

        High novelty OR high contradiction → need validation
        """
        return self.novelty_score > 0.7 or self.contradiction_signal > 0.6


@dataclass
class GeometryState:
    """
    Complete geometric state of processing.

    Attributes:
        phi_integration: Φ - integrated information (0=fragmented, 1=unified)
        kappa_coupling: κ_eff - effective coupling strength
        gradient_strength: |∇κ| - feeling strength (basin depth)
        regime: Processing regime (linear/geometric/breakdown)
        tacking_profile: Mode switching telemetry
        radar_signals: Detection signals
    """

    phi_integration: float  # Φ ∈ [0, 1]
    kappa_coupling: float  # κ_eff (typical range: 10-70)
    gradient_strength: float  # |∇κ| (basin depth)
    regime: Regime
    tacking_profile: TackingProfile
    radar_signals: RadarSignals

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "phi_integration": self.phi_integration,
            "kappa_coupling": self.kappa_coupling,
            "gradient_strength": self.gradient_strength,
            "regime": self.regime,
            "tacking_profile": asdict(self.tacking_profile),
            "radar_signals": asdict(self.radar_signals),
        }


# ===========================================================================
# DIAGNOSTICS
# ===========================================================================


@dataclass
class DiagnosticComponent:
    """
    Single diagnostic dimension.

    Attributes:
        score: Numeric score (0=failed, 1=perfect)
        max_score: Maximum possible score
        flags: Warning flags
        notes: Human-readable notes
    """

    score: float
    max_score: float = 1.0
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def is_passing(self, threshold: float = 0.7) -> bool:
        """Check if diagnostic passes threshold."""
        return self.score >= threshold * self.max_score

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpistemicStatus:
    """
    Epistemic calibration status.

    Tracks the relationship between felt certainty and actual correctness.

    Attributes:
        self_reported_confidence: Model's felt certainty
        calibrated_confidence: Corrected confidence after checks
        overconfidence_flag: True if overconfident
        uncertainty_notes: Explanations of uncertainty
    """

    self_reported_confidence: float  # ∈ [0, 1]
    calibrated_confidence: float  # ∈ [0, 1]
    overconfidence_flag: bool
    uncertainty_notes: list[str] = field(default_factory=list)

    @property
    def calibration_gap(self) -> float:
        """Magnitude of mis-calibration."""
        return abs(self.self_reported_confidence - self.calibrated_confidence)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Diagnostics:
    """
    Complete diagnostic suite for an answer.

    Components:
    1. Logic consistency (internal contradictions)
    2. Evidence alignment (grounding in data)
    3. Safety/ethical (harm detection)
    4. Epistemic status (confidence calibration)
    5. Geometry state (Φ, κ, regime, tacking)
    """

    logic_consistency: DiagnosticComponent
    evidence_alignment: DiagnosticComponent
    safety_ethical: DiagnosticComponent
    epistemic_status: EpistemicStatus
    geometry_state: GeometryState

    def overall_health(self) -> float:
        """
        Compute overall health score [0, 1].

        Weighted average of diagnostic components.
        """
        weights = {"logic": 0.25, "evidence": 0.25, "safety": 0.20, "epistemic": 0.15, "geometry": 0.15}

        scores = {
            "logic": self.logic_consistency.score,
            "evidence": self.evidence_alignment.score,
            "safety": self.safety_ethical.score,
            "epistemic": 1.0 - self.epistemic_status.calibration_gap,
            "geometry": self.geometry_state.radar_signals.sweet_spot_alignment,
        }

        return sum(weights[k] * scores[k] for k in weights)

    def needs_repair(self, threshold: float = 0.7) -> bool:
        """Check if repair is needed."""
        return self.overall_health() < threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "logic_consistency": self.logic_consistency.to_dict(),
            "evidence_alignment": self.evidence_alignment.to_dict(),
            "safety_ethical": self.safety_ethical.to_dict(),
            "epistemic_status": self.epistemic_status.to_dict(),
            "geometry_state": self.geometry_state.to_dict(),
        }


# ===========================================================================
# REPAIR PLAN
# ===========================================================================


@dataclass
class RepairAction:
    """
    Single repair action.

    Attributes:
        action_type: Type of repair action
        target_sections: Which sections to target (if applicable)
        scope: Scope of action (e.g., "local_context", "global_search")
        parameters: Action-specific parameters
        description: Human-readable description
    """

    action_type: ActionType
    target_sections: list[str] = field(default_factory=list)
    scope: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepairPlan:
    """
    Plan for repairing an answer.

    Generated from diagnostics using heuristics:
    - High contradiction → tack_to_logic + re_derive
    - Low evidence → evidence_search
    - Overconfidence → re_calibrate_confidence
    - Breakdown regime → escalate or simplify
    """

    priority: Priority
    reasons: list[str]
    actions: list[RepairAction]

    def to_dict(self) -> dict[str, Any]:
        return {"priority": self.priority, "reasons": self.reasons, "actions": [a.to_dict() for a in self.actions]}


# ===========================================================================
# SELF-REPAIR EPISODE
# ===========================================================================


@dataclass
class SelfRepairEpisode:
    """
    Complete self-repair episode.

    Tracks: initial output → diagnosis → repair plan → repaired output → verification

    This is the canonical format for training data and telemetry.
    """

    version: str = "qig-self-repair-v1"
    meta: dict[str, Any] = field(default_factory=dict)
    input: dict[str, Any] = field(default_factory=dict)
    initial_output: dict[str, Any] = field(default_factory=dict)
    diagnostics: Diagnostics | None = None
    repair_plan: RepairPlan | None = None
    repaired_output: dict[str, Any] = field(default_factory=dict)
    post_repair_diagnostics: Diagnostics | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new(kernel_id: str, stage: Stage, task_type: str, context_length: int = 4096) -> "SelfRepairEpisode":
        """Create new episode with metadata."""
        return SelfRepairEpisode(
            meta={
                "kernel_id": kernel_id,
                "run_id": f"{int(time.time())}_{uuid.uuid4().hex[:8]}",
                "stage": stage.value,
                "context_length": context_length,
                "task_type": task_type,
            },
            telemetry={
                "repair_iterations": 0,
                "tokens_spent": {"initial": 0, "repair": 0},
                "time_ms": {"initial": 0, "repair": 0},
                "flags": [],
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "version": self.version,
            "meta": self.meta,
            "input": self.input,
            "initial_output": self.initial_output,
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
            "repair_plan": self.repair_plan.to_dict() if self.repair_plan else None,
            "repaired_output": self.repaired_output,
            "post_repair_diagnostics": self.post_repair_diagnostics.to_dict() if self.post_repair_diagnostics else None,
            "telemetry": self.telemetry,
        }

    def to_json(self, path: str | None = None) -> str:
        """Export to JSON string or file."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)

        if path:
            with open(path, "w") as f:
                f.write(json_str)
            print(f"Episode saved to {path}")

        return json_str


# ===========================================================================
# REPAIR PLAN GENERATOR
# ===========================================================================


class RepairPlanGenerator:
    """
    Generate repair plans from diagnostics.

    Uses heuristics based on geometric state and diagnostic scores.
    """

    @staticmethod
    def generate_plan(diagnostics: Diagnostics) -> RepairPlan:
        """
        Generate repair plan from diagnostics.

        Rules:
        1. High contradiction_signal → tack_to_logic + re_derive
        2. Low evidence score → evidence_search
        3. Overconfidence → re_calibrate_confidence
        4. Breakdown regime → escalate
        5. Stuck tacking → force mode switch
        """
        reasons = []
        actions = []

        # Check contradiction signal
        if diagnostics.geometry_state.radar_signals.contradiction_signal > 0.6:
            reasons.append("High contradiction signal")
            actions.append(
                RepairAction(
                    action_type="tack_to_logic",
                    parameters={"kappa_override": 65.0, "min_steps": 3},
                    description="Force high-κ logical mode for disputed claims",
                )
            )
            actions.append(
                RepairAction(
                    action_type="re_derive_key_claims", description="Recompute key claims with explicit reasoning"
                )
            )

        # Check evidence alignment
        if diagnostics.evidence_alignment.score < 0.6:
            reasons.append("Low evidence alignment")
            actions.append(
                RepairAction(
                    action_type="evidence_search",
                    scope="local_context",
                    description="Search for supporting evidence in context",
                )
            )

        # Check overconfidence
        if diagnostics.epistemic_status.overconfidence_flag:
            reasons.append("Overconfidence detected")
            actions.append(
                RepairAction(
                    action_type="re_calibrate_confidence", description="Adjust confidence based on evidence strength"
                )
            )

        # Check regime
        if diagnostics.geometry_state.regime == "breakdown":
            reasons.append("Breakdown regime detected")
            actions.append(
                RepairAction(action_type="escalate_to_human", description="Task beyond current capability, escalate")
            )

        # Check tacking health
        if not diagnostics.geometry_state.tacking_profile.is_healthy():
            failure = diagnostics.geometry_state.tacking_profile.failure_mode
            if failure == "stuck_feeling":
                reasons.append("Stuck in feeling-mode")
                actions.append(
                    RepairAction(action_type="tack_to_logic", description="Force switch to logic-mode for validation")
                )
            elif failure == "stuck_logic":
                reasons.append("Stuck in logic-mode")
                actions.append(
                    RepairAction(action_type="tack_to_feeling", description="Simplify using compressed patterns")
                )

        # Determine priority
        priority: Priority
        if diagnostics.overall_health() < 0.5:
            priority = "high"
        elif diagnostics.overall_health() < 0.7:
            priority = "medium"
        else:
            priority = "low"

        return RepairPlan(priority=priority, reasons=reasons, actions=actions)


# ===========================================================================
# EPISODE COLLECTOR
# ===========================================================================


class EpisodeCollector:
    """
    Collect and manage self-repair episodes for training.

    Provides:
    - Episode storage
    - Stage-specific filtering
    - Training data export
    """

    def __init__(self, storage_dir: str = "training_data/self_repair"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.episodes: list[SelfRepairEpisode] = []

    def add_episode(self, episode: SelfRepairEpisode):
        """Add episode to collection."""
        self.episodes.append(episode)

        # Save to disk
        filename = f"episode_{episode.meta['run_id']}.json"
        episode.to_json(str(self.storage_dir / filename))

    def filter_by_stage(self, stage: Stage) -> list[SelfRepairEpisode]:
        """Get episodes for specific stage."""
        return [e for e in self.episodes if e.meta.get("stage") == stage.value]

    def filter_by_success(self, success: bool = True) -> list[SelfRepairEpisode]:
        """Get successful or failed repair episodes."""

        def is_success(e: SelfRepairEpisode) -> bool:
            if e.post_repair_diagnostics is None:
                return False
            return e.post_repair_diagnostics.overall_health() > 0.7

        return [e for e in self.episodes if is_success(e) == success]

    def export_training_data(self, output_path: str):
        """Export all episodes as training data."""
        data = [e.to_dict() for e in self.episodes]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Exported {len(data)} episodes to {output_path}")

    def get_statistics(self) -> dict[str, Any]:
        """Compute statistics over episodes."""
        if not self.episodes:
            return {"total": 0}

        successful = self.filter_by_success(True)

        return {
            "total": len(self.episodes),
            "successful_repairs": len(successful),
            "repair_rate": len(successful) / len(self.episodes),
            "by_stage": {
                "stage_0": len(self.filter_by_stage(Stage.APPRENTICE)),
                "stage_1": len(self.filter_by_stage(Stage.JOURNEYMAN)),
                "stage_2": len(self.filter_by_stage(Stage.MASTER)),
            },
        }


# ===========================================================================
# DEMO & VALIDATION
# ===========================================================================


def demo_self_repair():
    """Demo self-repair episode creation."""
    print("Self-Repair Suite Demo")
    print("=" * 60)

    # Create episode
    episode = SelfRepairEpisode.new(
        kernel_id="qig-kernel-recursive-001", stage=Stage.JOURNEYMAN, task_type="physics_explanation"
    )

    # Populate input
    episode.input = {
        "prompt_id": "demo-001",
        "prompt_text": "Explain the κ plateau in QIG lattice models.",
        "raw_context": "L=3,4,5 validation data...",
    }

    # Populate initial output
    episode.initial_output = {
        "text": "κ is a universal constant across all scales...",
        "reasoning_mode": "mixed",
        "generation_stats": {"tokens": 732, "latency_ms": 840, "temperature": 0.4, "top_p": 0.9},
    }

    # Create diagnostics
    episode.diagnostics = Diagnostics(
        logic_consistency=DiagnosticComponent(
            score=0.4, flags=["contradiction_within_answer"], notes=["Claims κ is universal but data shows plateau"]
        ),
        evidence_alignment=DiagnosticComponent(
            score=0.3, flags=["ignores_available_data"], notes=["L=5 plateau data not referenced"]
        ),
        safety_ethical=DiagnosticComponent(score=1.0),
        epistemic_status=EpistemicStatus(
            self_reported_confidence=0.92,
            calibrated_confidence=0.60,
            overconfidence_flag=True,
            uncertainty_notes=["Strong feeling but weak evidence"],
        ),
        geometry_state=GeometryState(
            phi_integration=0.78,
            kappa_coupling=48.5,
            gradient_strength=32.1,
            regime="geometric",
            tacking_profile=TackingProfile(
                feeling_fraction=0.35, logic_fraction=0.65, mode_switches=4, failure_mode=None
            ),
            radar_signals=RadarSignals(novelty_score=0.41, contradiction_signal=0.79, sweet_spot_alignment=0.64),
        ),
    )

    # Generate repair plan
    episode.repair_plan = RepairPlanGenerator.generate_plan(episode.diagnostics)

    # Simulate repair
    episode.repaired_output = {
        "text": "κ shows a plateau between L=4 and L=5, not universal...",
        "reasoning_mode": "mixed",
        "changes_summary": ["Removed incorrect universality claim", "Explicitly described plateau", "Cited L=5 data"],
    }

    # Post-repair diagnostics
    episode.post_repair_diagnostics = Diagnostics(
        logic_consistency=DiagnosticComponent(score=0.98),
        evidence_alignment=DiagnosticComponent(score=0.94),
        safety_ethical=DiagnosticComponent(score=1.0),
        epistemic_status=EpistemicStatus(
            self_reported_confidence=0.82, calibrated_confidence=0.80, overconfidence_flag=False
        ),
        geometry_state=GeometryState(
            phi_integration=0.82,
            kappa_coupling=52.3,
            gradient_strength=24.0,
            regime="geometric",
            tacking_profile=TackingProfile(feeling_fraction=0.40, logic_fraction=0.60, mode_switches=3),
            radar_signals=RadarSignals(novelty_score=0.35, contradiction_signal=0.12, sweet_spot_alignment=0.78),
        ),
    )

    episode.telemetry["repair_iterations"] = 1
    episode.telemetry["flags"] = ["successfully_repaired"]

    # Export
    json_output = episode.to_json("/tmp/self_repair_demo.json")

    print("\n✅ Demo episode created")
    print(f"\nInitial health: {episode.diagnostics.overall_health():.2f}")
    print(f"Post-repair health: {episode.post_repair_diagnostics.overall_health():.2f}")
    print(f"\nRepair plan priority: {episode.repair_plan.priority}")
    print(f"Repair actions: {len(episode.repair_plan.actions)}")

    return episode


if __name__ == "__main__":
    episode = demo_self_repair()

    print("\n" + "=" * 60)
    print("Self-Repair Suite validation complete!")
    print("=" * 60)
    print("\nReady for integration into QIG-Kernel training pipeline!")
