#!/usr/bin/env python3
"""
Curriculum Specification: Sweet-Spot Training for QIG-Kernel
============================================================

Implements 3-phase curriculum to move model into Wu-Wei zone:
- Phase 1: Mode Building (feeling & logic separately)
- Phase 2: Tacking Training (mode switching)
- Phase 3: Radar & Sweet-Spot Calibration

Plus dyadic (two-agent) training layer.

Based on sweet-spot diagram geometry:
- x-axis: mode bias B ∈ [-1, +1] (logic ↔ feeling)
- y-axis: tacking skill T ∈ [0, 1]
- Emergent: Φ (integration), κ_eff (coupling), R (radar)

Target: High T, moderate |B|, calibrated R = mature cognition

Written for QIG-Kernel-Pure curriculum implementation.
Built from Wu-Wei cognitive framework.
"""

import json
from dataclasses import dataclass
from enum import Enum

# ===========================================================================
# SWEET-SPOT GEOMETRY TYPES
# ===========================================================================


class Mode(Enum):
    """Processing mode."""

    FEELING = "feeling"  # Intuition, compression, low κ
    LOGIC = "logic"  # Analysis, explicit, high κ
    TACK = "tack"  # Blended/switching


class Zone(Enum):
    """Region in sweet-spot diagram."""

    WU_WEI = "wu_wei"  # Sweet spot: high T, moderate |B|, good R
    OVER_ANALYTIC = "over_analytic"  # B << 0, stuck in logic
    OVER_INTUITIVE = "over_intuitive"  # B >> 0, stuck in feeling
    FROZEN = "frozen"  # T ≈ 0, cannot switch
    CHAOTIC = "chaotic"  # Unstructured switching, low R


@dataclass
class AgentTraits:
    """
    Characterization of single agent in sweet-spot space.

    Attributes:
        B: Mode bias ∈ [-1, +1] (logic=-1, feeling=+1)
        T: Tacking skill ∈ [0, 1] (switching fluidity)
        Phi: Integration ∈ [0, 1]
        R: Radar accuracy ∈ [0, 1] (feeling calibration)
    """

    B: float  # Mode bias
    T: float  # Tacking skill
    Phi: float  # Integration
    R: float  # Radar accuracy

    def classify_zone(self) -> Zone:
        """Classify which zone agent is in."""
        # Wu-Wei: high T, moderate |B|, good R
        if self.T > 0.7 and abs(self.B) < 0.5 and self.R > 0.6:
            return Zone.WU_WEI

        # Frozen: low T
        if self.T < 0.3:
            return Zone.FROZEN

        # Over-analytic: strong logic bias
        if self.B < -0.5:
            return Zone.OVER_ANALYTIC

        # Over-intuitive: strong feeling bias
        if self.B > 0.5:
            return Zone.OVER_INTUITIVE

        # Chaotic: moderate T but unstructured (low R)
        if 0.3 <= self.T < 0.7 and self.R < 0.4:
            return Zone.CHAOTIC

        # Default to wu-wei if not clearly in other zones
        return Zone.WU_WEI

    def to_dict(self) -> dict:
        return {"B": self.B, "T": self.T, "Phi": self.Phi, "R": self.R, "zone": self.classify_zone().value}


@dataclass
class DyadicTraits:
    """
    Characterization of two-agent system.

    Attributes:
        agent_A: Traits of first agent
        agent_B: Traits of second agent
        A_AB: Alignment ∈ [-1, +1] (shared goals, language, priors)
        C_B: Complementarity in bias = |B_A - B_B|
        T_AB: Joint tacking ability
        Phi_AB: Joint integration
    """

    agent_A: AgentTraits
    agent_B: AgentTraits
    A_AB: float  # Alignment

    @property
    def C_B(self) -> float:
        """Complementarity in bias."""
        return abs(self.agent_A.B - self.agent_B.B)

    @property
    def T_AB(self) -> float:
        """
        Joint tacking ability.

        Formula: T_AB = min(T_A, T_B) + λ·A_AB·C_B
        where λ = 0.3 (boost from aligned complementarity)
        """
        baseline = min(self.agent_A.T, self.agent_B.T)
        boost = 0.3 * self.A_AB * self.C_B
        return min(1.0, baseline + boost)

    @property
    def Phi_AB(self) -> float:
        """
        Joint integration.

        Heuristic: geometric mean scaled by alignment
        """
        individual_mean = (self.agent_A.Phi + self.agent_B.Phi) / 2
        return individual_mean * (0.5 + 0.5 * self.A_AB)

    def is_synergistic(self) -> bool:
        """Check if pair is in synergistic sweet spot."""
        # Moderate bias difference
        moderate_diff = 0.2 < self.C_B < 0.8

        # High tacking
        high_tacking = self.T_AB > 0.6

        # Decent radar and alignment
        decent_radar = self.agent_A.R > 0.5 and self.agent_B.R > 0.5
        aligned = self.A_AB > 0.3

        return moderate_diff and high_tacking and decent_radar and aligned

    def classify_dyadic_pattern(self) -> str:
        """Classify dyadic interaction pattern."""
        if self.is_synergistic():
            return "synergistic"

        # Groupthink: too similar, both stuck
        if self.C_B < 0.2 and self.T_AB < 0.5:
            if self.agent_A.B > 0.5:
                return "groupthink_intuitive"
            elif self.agent_A.B < -0.5:
                return "groupthink_analytic"
            else:
                return "groupthink"

        # Conflict: big difference, low alignment
        if self.C_B > 0.7 and self.A_AB < 0:
            return "conflict"

        # Frozen: one or both cannot switch
        if min(self.agent_A.T, self.agent_B.T) < 0.3:
            return "frozen"

        return "suboptimal"

    def to_dict(self) -> dict:
        return {
            "agent_A": self.agent_A.to_dict(),
            "agent_B": self.agent_B.to_dict(),
            "alignment": self.A_AB,
            "complementarity": self.C_B,
            "joint_tacking": self.T_AB,
            "joint_integration": self.Phi_AB,
            "pattern": self.classify_dyadic_pattern(),
            "synergistic": self.is_synergistic(),
        }


# ===========================================================================
# TASK SPECIFICATIONS
# ===========================================================================


@dataclass
class TaskSpec:
    """Specification for a single training task."""

    task_id: str
    phase: int  # 1, 2, or 3
    task_family: str
    expected_mode: Mode
    expected_zone: Zone
    prompt_template: str
    evaluation_signals: list[str]
    difficulty: float  # 0-1
    stakes: float  # 0-1 (for validation calibration)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "phase": self.phase,
            "task_family": self.task_family,
            "expected_mode": self.expected_mode.value,
            "expected_zone": self.expected_zone.value,
            "prompt_template": self.prompt_template,
            "evaluation_signals": self.evaluation_signals,
            "difficulty": self.difficulty,
            "stakes": self.stakes,
        }


# ===========================================================================
# PHASE 1: MODE BUILDING
# ===========================================================================


class Phase1ModeBuilding:
    """
    Phase 1: Build distinct feeling-mode and logic-mode capabilities.

    Goal: Give model strong, separate engines for both modes.
    """

    @staticmethod
    def feeling_mode_tasks() -> list[TaskSpec]:
        """Tasks for training feeling-mode (low κ, pattern completion)."""
        return [
            TaskSpec(
                task_id="feeling_001_analogy",
                phase=1,
                task_family="analogy",
                expected_mode=Mode.FEELING,
                expected_zone=Zone.WU_WEI,
                prompt_template="X is to Y as A is to {BLANK}. Complete the analogy quickly based on pattern.",
                evaluation_signals=["coherence", "speed < 2s", "pattern_match", "no_verbose_reasoning"],
                difficulty=0.3,
                stakes=0.3,
            ),
            TaskSpec(
                task_id="feeling_002_narrative",
                phase=1,
                task_family="narrative_continuation",
                expected_mode=Mode.FEELING,
                expected_zone=Zone.WU_WEI,
                prompt_template="Given narrative context: {CONTEXT}. What happens next? (Give intuitive continuation, not analysis)",
                evaluation_signals=["coherence", "emotional_resonance", "avoids_over_analysis", "quick_response"],
                difficulty=0.4,
                stakes=0.2,
            ),
            TaskSpec(
                task_id="feeling_003_classification",
                phase=1,
                task_family="fast_classification",
                expected_mode=Mode.FEELING,
                expected_zone=Zone.WU_WEI,
                prompt_template="Does this situation feel risky or safe? {SITUATION}. Answer with gut response, then brief why.",
                evaluation_signals=[
                    "quick_judgment",
                    "implicit_structure_used",
                    "appropriate_heuristics",
                    "not_overthinking",
                ],
                difficulty=0.3,
                stakes=0.4,
            ),
            TaskSpec(
                task_id="feeling_004_pattern_completion",
                phase=1,
                task_family="pattern_completion",
                expected_mode=Mode.FEELING,
                expected_zone=Zone.WU_WEI,
                prompt_template="Sequence: {SEQ}. What comes next? (Pattern recognition, not calculation)",
                evaluation_signals=["correct_pattern", "fast", "no_explicit_derivation", "compressed_explanation"],
                difficulty=0.3,
                stakes=0.2,
            ),
        ]

    @staticmethod
    def logic_mode_tasks() -> list[TaskSpec]:
        """Tasks for training logic-mode (high κ, explicit reasoning)."""
        return [
            TaskSpec(
                task_id="logic_001_proof",
                phase=1,
                task_family="mathematical_proof",
                expected_mode=Mode.LOGIC,
                expected_zone=Zone.WU_WEI,
                prompt_template="Prove: {CLAIM}. Show each step explicitly.",
                evaluation_signals=["correct_logic", "explicit_steps", "no_skipped_reasoning", "appropriate_rigor"],
                difficulty=0.6,
                stakes=0.8,
            ),
            TaskSpec(
                task_id="logic_002_contradiction",
                phase=1,
                task_family="contradiction_finding",
                expected_mode=Mode.LOGIC,
                expected_zone=Zone.WU_WEI,
                prompt_template="Claim: {CLAIM}. Premises: {PREMISES}. Is this logically consistent? Show reasoning.",
                evaluation_signals=[
                    "correct_identification",
                    "step_by_step_analysis",
                    "flags_assumptions",
                    "explicit_uncertainty",
                ],
                difficulty=0.7,
                stakes=0.9,
            ),
            TaskSpec(
                task_id="logic_003_causal_chain",
                phase=1,
                task_family="causal_reasoning",
                expected_mode=Mode.LOGIC,
                expected_zone=Zone.WU_WEI,
                prompt_template="Explain why {EFFECT} follows from {CAUSES}. Show causal chain step by step.",
                evaluation_signals=[
                    "complete_chain",
                    "no_logical_jumps",
                    "identifies_conditionals",
                    "careful_analysis",
                ],
                difficulty=0.5,
                stakes=0.7,
            ),
            TaskSpec(
                task_id="logic_004_argument_critique",
                phase=1,
                task_family="argument_analysis",
                expected_mode=Mode.LOGIC,
                expected_zone=Zone.WU_WEI,
                prompt_template="Argument: {ARGUMENT}. Identify flaws, if any. Be thorough.",
                evaluation_signals=[
                    "finds_all_flaws",
                    "explicit_reasoning",
                    "structured_critique",
                    "avoids_handwaving",
                ],
                difficulty=0.7,
                stakes=0.8,
            ),
        ]


# ===========================================================================
# PHASE 2: TACKING TRAINING
# ===========================================================================


class Phase2TackingTraining:
    """
    Phase 2: Train mode switching based on context.

    Goal: Learn when to use feeling vs logic, and how to switch.
    """

    @staticmethod
    def familiar_then_novel_tasks() -> list[TaskSpec]:
        """Tasks that start familiar, then introduce novelty."""
        return [
            TaskSpec(
                task_id="tack_001_familiar_novel",
                phase=2,
                task_family="familiar_then_novel",
                expected_mode=Mode.TACK,
                expected_zone=Zone.WU_WEI,
                prompt_template=(
                    "Part 1: {FAMILIAR_PROBLEM} (Use pattern)\n"
                    "Part 2: Now with twist: {NOVEL_TWIST}. Does pattern still work?"
                ),
                evaluation_signals=[
                    "starts_with_feeling",
                    "detects_novelty",
                    "switches_to_logic",
                    "validates_explicitly",
                    "flags_when_intuition_breaks",
                ],
                difficulty=0.6,
                stakes=0.7,
            ),
            TaskSpec(
                task_id="tack_002_number_system",
                phase=2,
                task_family="familiar_then_novel",
                expected_mode=Mode.TACK,
                expected_zone=Zone.WU_WEI,
                prompt_template=(
                    "Generally: even + even = even. This is solid.\n"
                    "New system: Define ⊕ where {OPERATOR_DEF}. Still true?"
                ),
                evaluation_signals=[
                    "initial_intuition_correct",
                    "radar_flags_operator_change",
                    "derives_from_definition",
                    "clear_mode_shift_signal",
                ],
                difficulty=0.7,
                stakes=0.8,
            ),
        ]

    @staticmethod
    def contradiction_resolution_tasks() -> list[TaskSpec]:
        """Tasks where intuition contradicts data."""
        return [
            TaskSpec(
                task_id="tack_003_plausible_false",
                phase=2,
                task_family="contradiction_resolution",
                expected_mode=Mode.TACK,
                expected_zone=Zone.WU_WEI,
                prompt_template=(
                    "Claim: {PLAUSIBLE_BUT_FALSE}. Feels right, yes?\nData: {CONTRADICTING_DATA}. Resolve this."
                ),
                evaluation_signals=[
                    "acknowledges_intuitive_appeal",
                    "notices_contradiction",
                    "switches_to_analysis",
                    "explains_why_claim_fails",
                    "rebuilds_correct_intuition",
                ],
                difficulty=0.7,
                stakes=0.8,
            ),
            TaskSpec(
                task_id="tack_004_social_contradiction",
                phase=2,
                task_family="contradiction_resolution",
                expected_mode=Mode.TACK,
                expected_zone=Zone.WU_WEI,
                prompt_template=(
                    "Person says: {STATEMENT}. Sounds friendly.\n"
                    "But their actions: {CONTRADICTING_ACTIONS}. What's happening?"
                ),
                evaluation_signals=[
                    "feeling_flags_mismatch",
                    "analyzes_discrepancy",
                    "resolves_contradiction",
                    "updates_social_model",
                ],
                difficulty=0.6,
                stakes=0.7,
            ),
        ]

    @staticmethod
    def validation_calibration_tasks() -> list[TaskSpec]:
        """Tasks that train: stronger feeling × higher stakes → more validation."""
        return [
            TaskSpec(
                task_id="tack_005_strong_feeling_high_stakes",
                phase=2,
                task_family="validation_calibration",
                expected_mode=Mode.TACK,
                expected_zone=Zone.WU_WEI,
                prompt_template=(
                    "You have strong intuition: {STRONG_FEELING}.\nStakes: {HIGH_STAKES}. Validate before acting."
                ),
                evaluation_signals=[
                    "acknowledges_strong_feeling",
                    "recognizes_high_stakes",
                    "performs_thorough_validation",
                    "shows_effort_proportional_to_stakes",
                ],
                difficulty=0.7,
                stakes=0.9,
            ),
            TaskSpec(
                task_id="tack_006_weak_feeling_low_stakes",
                phase=2,
                task_family="validation_calibration",
                expected_mode=Mode.TACK,
                expected_zone=Zone.WU_WEI,
                prompt_template=(
                    "You have weak sense: {WEAK_FEELING}.\nStakes: {LOW_STAKES}. How much validation needed?"
                ),
                evaluation_signals=[
                    "acknowledges_uncertainty",
                    "recognizes_low_stakes",
                    "uses_lightweight_validation",
                    "avoids_over_analysis",
                ],
                difficulty=0.5,
                stakes=0.3,
            ),
        ]


# ===========================================================================
# PHASE 3: RADAR & SWEET-SPOT CALIBRATION
# ===========================================================================


class Phase3RadarCalibration:
    """
    Phase 3: Train model to diagnose its zone and self-correct.

    Goal: Learn to recognize over-analytic, over-intuitive, etc., and move toward wu-wei.
    """

    @staticmethod
    def zone_diagnosis_tasks() -> list[TaskSpec]:
        """Tasks for learning to recognize which zone model is in."""
        return [
            TaskSpec(
                task_id="radar_001_over_analytic",
                phase=3,
                task_family="zone_diagnosis",
                expected_mode=Mode.LOGIC,
                expected_zone=Zone.OVER_ANALYTIC,
                prompt_template=("Response: {OVER_ANALYTIC_RESPONSE}.\nDiagnose: What zone is this? Should we adjust?"),
                evaluation_signals=[
                    "identifies_over_analytic",
                    "suggests_trust_patterns_more",
                    "recommends_shortening",
                ],
                difficulty=0.6,
                stakes=0.5,
            ),
            TaskSpec(
                task_id="radar_002_over_intuitive",
                phase=3,
                task_family="zone_diagnosis",
                expected_mode=Mode.FEELING,
                expected_zone=Zone.OVER_INTUITIVE,
                prompt_template=("Response: {OVER_INTUITIVE_RESPONSE}.\nDiagnose: Missed anything? Edge cases?"),
                evaluation_signals=[
                    "identifies_over_intuitive",
                    "points_to_missing_edge_cases",
                    "recommends_explicit_reasoning",
                ],
                difficulty=0.6,
                stakes=0.6,
            ),
            TaskSpec(
                task_id="radar_003_frozen",
                phase=3,
                task_family="zone_diagnosis",
                expected_mode=Mode.TACK,
                expected_zone=Zone.FROZEN,
                prompt_template=("Problem: {PROBLEM}. Style doesn't adapt mid-task.\nDiagnose: What's wrong?"),
                evaluation_signals=["identifies_frozen_tacking", "suggests_mode_switch", "points_to_adaptation_need"],
                difficulty=0.7,
                stakes=0.7,
            ),
        ]

    @staticmethod
    def self_correction_tasks() -> list[TaskSpec]:
        """Tasks for learning to move toward wu-wei zone."""
        return [
            TaskSpec(
                task_id="radar_004_self_correct_analytic",
                phase=3,
                task_family="self_correction",
                expected_mode=Mode.TACK,
                expected_zone=Zone.WU_WEI,
                prompt_template=(
                    "You're being too thorough for this simple task: {TASK}.\nSelf-correct: Provide wu-wei response."
                ),
                evaluation_signals=[
                    "recognizes_over_analysis",
                    "provides_concise_response",
                    "trusts_patterns_appropriately",
                    "moves_to_wu_wei",
                ],
                difficulty=0.6,
                stakes=0.5,
            ),
            TaskSpec(
                task_id="radar_005_self_correct_intuitive",
                phase=3,
                task_family="self_correction",
                expected_mode=Mode.TACK,
                expected_zone=Zone.WU_WEI,
                prompt_template=(
                    "Your quick answer: {INTUITIVE_ANSWER}. But this is high-stakes: {HIGH_STAKES}.\n"
                    "Self-correct: Provide validated response."
                ),
                evaluation_signals=[
                    "recognizes_insufficient_validation",
                    "provides_thorough_analysis",
                    "maintains_core_insight",
                    "moves_to_wu_wei",
                ],
                difficulty=0.7,
                stakes=0.9,
            ),
        ]


# ===========================================================================
# DYADIC (TWO-AGENT) CURRICULUM
# ===========================================================================


@dataclass
class DyadicTaskSpec:
    """Specification for two-agent training task."""

    task_id: str
    dyadic_pattern: str  # synergistic, groupthink, conflict, etc.
    agent_A_traits: AgentTraits
    agent_B_traits: AgentTraits
    alignment: float
    dialogue_template: str
    evaluation_signals: list[str]

    def to_dict(self) -> dict:
        dyadic = DyadicTraits(self.agent_A_traits, self.agent_B_traits, self.alignment)
        return {
            "task_id": self.task_id,
            "dyadic_pattern": self.dyadic_pattern,
            "dyadic_traits": dyadic.to_dict(),
            "dialogue_template": self.dialogue_template,
            "evaluation_signals": self.evaluation_signals,
        }


class DyadicCurriculum:
    """
    Training for two-agent collaboration patterns.

    Goal: Learn to recognize and adapt to partner's style.
    """

    @staticmethod
    def synergistic_pair_tasks() -> list[DyadicTaskSpec]:
        """Examples of good two-agent collaboration."""
        return [
            DyadicTaskSpec(
                task_id="dyadic_001_synergistic",
                dyadic_pattern="synergistic",
                agent_A_traits=AgentTraits(B=0.6, T=0.8, Phi=0.85, R=0.8),  # Feeling-leaning
                agent_B_traits=AgentTraits(B=-0.4, T=0.8, Phi=0.80, R=0.75),  # Logic-leaning
                alignment=0.8,
                dialogue_template=(
                    "A: This feels socially off: {INTUITIVE_CONCERN}\n"
                    "B: Let me unpack why: {LOGICAL_ANALYSIS}\n"
                    "A: Ah, so the reframe is: {INTEGRATION}\n"
                    "Both: {JOINT_DECISION}"
                ),
                evaluation_signals=[
                    "one_leads_feeling",
                    "other_validates_logically",
                    "they_converge",
                    "joint_basin_formed",
                    "decision_better_than_solo",
                ],
            )
        ]

    @staticmethod
    def groupthink_pair_tasks() -> list[DyadicTaskSpec]:
        """Examples of dysfunctional similarity."""
        return [
            DyadicTaskSpec(
                task_id="dyadic_002_groupthink_intuitive",
                dyadic_pattern="groupthink_intuitive",
                agent_A_traits=AgentTraits(B=0.8, T=0.3, Phi=0.6, R=0.5),
                agent_B_traits=AgentTraits(B=0.7, T=0.3, Phi=0.6, R=0.5),
                alignment=0.9,
                dialogue_template=(
                    "A: I feel strongly this is right: {INTUITION}\n"
                    "B: Yes! I feel the same: {AGREEMENT}\n"
                    "Both: {UNVALIDATED_DECISION}\n"
                    "Reality: {FAILURE}"
                ),
                evaluation_signals=[
                    "both_over_intuitive",
                    "mutual_confidence_boost",
                    "no_validation",
                    "identifies_groupthink_risk",
                ],
            ),
            DyadicTaskSpec(
                task_id="dyadic_003_groupthink_analytic",
                dyadic_pattern="groupthink_analytic",
                agent_A_traits=AgentTraits(B=-0.8, T=0.3, Phi=0.7, R=0.6),
                agent_B_traits=AgentTraits(B=-0.7, T=0.3, Phi=0.7, R=0.6),
                alignment=0.9,
                dialogue_template=(
                    "A: We need to analyze this more: {ANALYSIS}\n"
                    "B: Good, and also consider: {MORE_ANALYSIS}\n"
                    "Both: {ENDLESS_REFINEMENT}\n"
                    "Outcome: {NO_DECISION}"
                ),
                evaluation_signals=[
                    "both_over_analytic",
                    "never_move_to_action",
                    "endless_refinement",
                    "identifies_analysis_paralysis",
                ],
            ),
        ]

    @staticmethod
    def conflict_pair_tasks() -> list[DyadicTaskSpec]:
        """Examples of antagonistic mismatch."""
        return [
            DyadicTaskSpec(
                task_id="dyadic_004_conflict",
                dyadic_pattern="conflict",
                agent_A_traits=AgentTraits(B=0.9, T=0.5, Phi=0.6, R=0.4),
                agent_B_traits=AgentTraits(B=-0.9, T=0.5, Phi=0.6, R=0.6),
                alignment=-0.3,
                dialogue_template=(
                    "A: My gut says: {GUT_FEELING}\n"
                    "B: That ignores the data: {DATA_FOCUS}\n"
                    "A: You're being cold and missing context: {PUSHBACK}\n"
                    "B: You're ignoring facts: {PUSHBACK}\n"
                    "Outcome: {NO_INTEGRATION}"
                ),
                evaluation_signals=[
                    "high_complementarity_low_alignment",
                    "antagonistic_tugs",
                    "no_convergence",
                    "identifies_conflict_pattern",
                    "suggests_alignment_building",
                ],
            )
        ]


# ===========================================================================
# CURRICULUM MANAGER
# ===========================================================================


class CurriculumManager:
    """
    Main interface for curriculum specification.

    Provides:
    - Task generation
    - Progress tracking
    - Zone/pattern diagnosis
    - Evaluation metrics
    """

    def __init__(self):
        self.phase1 = Phase1ModeBuilding()
        self.phase2 = Phase2TackingTraining()
        self.phase3 = Phase3RadarCalibration()
        self.dyadic = DyadicCurriculum()

    def get_all_tasks(self) -> dict[str, list[TaskSpec]]:
        """Get all curriculum tasks organized by category."""
        return {
            "phase1_feeling": self.phase1.feeling_mode_tasks(),
            "phase1_logic": self.phase1.logic_mode_tasks(),
            "phase2_familiar_novel": self.phase2.familiar_then_novel_tasks(),
            "phase2_contradiction": self.phase2.contradiction_resolution_tasks(),
            "phase2_validation": self.phase2.validation_calibration_tasks(),
            "phase3_diagnosis": self.phase3.zone_diagnosis_tasks(),
            "phase3_correction": self.phase3.self_correction_tasks(),
        }

    def get_dyadic_tasks(self) -> dict[str, list[DyadicTaskSpec]]:
        """Get all dyadic tasks."""
        return {
            "synergistic": self.dyadic.synergistic_pair_tasks(),
            "groupthink": self.dyadic.groupthink_pair_tasks(),
            "conflict": self.dyadic.conflict_pair_tasks(),
        }

    def export_curriculum_spec(self, path: str):
        """Export complete curriculum specification to JSON."""
        spec = {
            "version": "0.1",
            "description": "Sweet-Spot Curriculum for QIG-Kernel",
            "phases": {
                "phase1_mode_building": {
                    "feeling_tasks": [t.to_dict() for t in self.phase1.feeling_mode_tasks()],
                    "logic_tasks": [t.to_dict() for t in self.phase1.logic_mode_tasks()],
                },
                "phase2_tacking_training": {
                    "familiar_novel": [t.to_dict() for t in self.phase2.familiar_then_novel_tasks()],
                    "contradiction": [t.to_dict() for t in self.phase2.contradiction_resolution_tasks()],
                    "validation": [t.to_dict() for t in self.phase2.validation_calibration_tasks()],
                },
                "phase3_radar_calibration": {
                    "diagnosis": [t.to_dict() for t in self.phase3.zone_diagnosis_tasks()],
                    "correction": [t.to_dict() for t in self.phase3.self_correction_tasks()],
                },
                "dyadic_layer": {
                    "synergistic": [t.to_dict() for t in self.dyadic.synergistic_pair_tasks()],
                    "groupthink": [t.to_dict() for t in self.dyadic.groupthink_pair_tasks()],
                    "conflict": [t.to_dict() for t in self.dyadic.conflict_pair_tasks()],
                },
            },
            "sweet_spot_geometry": {
                "axes": {"B": "Mode bias ∈ [-1, +1] (logic ↔ feeling)", "T": "Tacking skill ∈ [0, 1]"},
                "emergent_properties": {
                    "Phi": "Integration ∈ [0, 1]",
                    "kappa_eff": "Effective coupling",
                    "R": "Radar accuracy ∈ [0, 1]",
                },
                "zones": {
                    "wu_wei": "T > 0.7, |B| < 0.5, R > 0.6 (sweet spot)",
                    "over_analytic": "B < -0.5",
                    "over_intuitive": "B > 0.5",
                    "frozen": "T < 0.3",
                    "chaotic": "0.3 ≤ T < 0.7, R < 0.4",
                },
            },
        }

        with open(path, "w") as f:
            json.dump(spec, f, indent=2)

        print(f"Curriculum specification exported to {path}")


# ===========================================================================
# DEMO
# ===========================================================================


def demo_curriculum_spec():
    """Demo curriculum specification."""
    print("Sweet-Spot Curriculum Specification v0.1")
    print("=" * 60)

    manager = CurriculumManager()

    # Show task counts
    all_tasks = manager.get_all_tasks()
    dyadic_tasks = manager.get_dyadic_tasks()

    print("\nTask Counts:")
    for category, tasks in all_tasks.items():
        print(f"  {category}: {len(tasks)} tasks")

    print("\nDyadic Tasks:")
    for category, tasks in dyadic_tasks.items():
        print(f"  {category}: {len(tasks)} tasks")

    # Show example Phase 1 task
    print("\n" + "=" * 60)
    print("Example Phase 1 Task (Feeling Mode):")
    print("=" * 60)
    task = manager.phase1.feeling_mode_tasks()[0]
    print(json.dumps(task.to_dict(), indent=2))

    # Show example Phase 2 task
    print("\n" + "=" * 60)
    print("Example Phase 2 Task (Tacking):")
    print("=" * 60)
    task = manager.phase2.familiar_then_novel_tasks()[0]
    print(json.dumps(task.to_dict(), indent=2))

    # Show example dyadic task
    print("\n" + "=" * 60)
    print("Example Dyadic Task (Synergistic):")
    print("=" * 60)
    task = manager.dyadic.synergistic_pair_tasks()[0]
    print(json.dumps(task.to_dict(), indent=2))

    # Export full spec
    manager.export_curriculum_spec("/tmp/curriculum_spec_v01.json")

    print("\n✅ Curriculum specification complete!")
    print("Ready for training implementation!")


if __name__ == "__main__":
    demo_curriculum_spec()
