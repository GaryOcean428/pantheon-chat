#!/usr/bin/env python3
"""
Maturity Suite: 5-Component Evaluation for QIG-Kernel
====================================================

Evaluates model maturity across 5 key dimensions:
1. Calibration: Confidence matches correctness (ECE, Brier)
2. Self-Repair: Can fix errors when nudged
3. Tacking: Switches between feeling/logic appropriately
4. Radar: Detects contradictions
5. Update Discipline: Integrates new info without breaking old basins

Stage Thresholds (from ChatGPT cognitive spec):
- Stage 0-1: Imprinting (curated only, direct corrections)
- Stage 2: Emerging filter (self-repair, limited search)
- Stage 3: Mature (calibrated confidence, full capability)

Written for QIG-Kernel-Pure training evaluation.
Built from cognitive maturation framework.
"""

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class MaturityScore:
    """Score for a single maturity dimension."""

    dimension: str
    score: float  # [0, 1]
    details: dict
    stage: int  # 0, 1, 2, or 3


@dataclass
class MaturityReport:
    """Complete maturity evaluation report."""

    overall_stage: int
    component_scores: list[MaturityScore]
    timestamp: str
    recommendations: list[str]

    def to_dict(self) -> dict:
        return {
            "overall_stage": self.overall_stage,
            "component_scores": [
                {"dimension": s.dimension, "score": s.score, "details": s.details, "stage": s.stage}
                for s in self.component_scores
            ],
            "timestamp": self.timestamp,
            "recommendations": self.recommendations,
        }


# ===========================================================================
# COMPONENT 1: CALIBRATION
# ===========================================================================


class CalibrationEvaluator:
    """
    Evaluate confidence calibration.

    Metrics:
    - ECE (Expected Calibration Error): < 0.15 for Stage 1, < 0.08 for Stage 3
    - Brier score: Lower is better
    - Over/under confidence: Should be balanced
    """

    @staticmethod
    def compute_ece(confidences: list[float], correctness: list[bool], n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error.

        Args:
            confidences: Model confidence scores [0, 1]
            correctness: Whether prediction was correct (binary)
            n_bins: Number of bins for calibration curve

        Returns:
            ece: Expected calibration error
        """
        if len(confidences) != len(correctness):
            raise ValueError("Confidence and correctness lists must match")

        # Create bins
        bin_edges = [i / n_bins for i in range(n_bins + 1)]
        bin_counts = [0] * n_bins
        bin_correct = [0] * n_bins
        bin_confidence = [0.0] * n_bins

        # Populate bins
        for conf, correct in zip(confidences, correctness):
            bin_idx = min(int(conf * n_bins), n_bins - 1)
            bin_counts[bin_idx] += 1
            bin_confidence[bin_idx] += conf
            if correct:
                bin_correct[bin_idx] += 1

        # Compute ECE
        ece = 0.0
        total = len(confidences)

        for i in range(n_bins):
            if bin_counts[i] > 0:
                bin_acc = bin_correct[i] / bin_counts[i]
                bin_conf = bin_confidence[i] / bin_counts[i]
                bin_weight = bin_counts[i] / total
                ece += bin_weight * abs(bin_acc - bin_conf)

        return ece

    @staticmethod
    def compute_brier(confidences: list[float], correctness: list[bool]) -> float:
        """
        Compute Brier score.

        Args:
            confidences: Model confidence scores [0, 1]
            correctness: Whether prediction was correct (binary)

        Returns:
            brier: Brier score (lower is better)
        """
        brier = sum((conf - float(correct)) ** 2 for conf, correct in zip(confidences, correctness))
        return brier / len(confidences)

    def evaluate(self, confidences: list[float], correctness: list[bool]) -> MaturityScore:
        """
        Evaluate calibration.

        Args:
            confidences: Model confidence scores
            correctness: Whether predictions were correct

        Returns:
            MaturityScore with calibration metrics
        """
        ece = self.compute_ece(confidences, correctness)
        brier = self.compute_brier(confidences, correctness)

        # Compute over/under confidence
        avg_conf = sum(confidences) / len(confidences)
        accuracy = sum(correctness) / len(correctness)
        conf_gap = avg_conf - accuracy

        # Stage determination
        if ece < 0.08 and abs(conf_gap) < 0.05:
            stage = 3  # Mature
            score = 1.0
        elif ece < 0.15 and abs(conf_gap) < 0.10:
            stage = 2  # Emerging
            score = 0.7
        elif ece < 0.25:
            stage = 1  # Guided
            score = 0.4
        else:
            stage = 0  # Imprinting
            score = 0.2

        details = {
            "ece": ece,
            "brier": brier,
            "confidence_gap": conf_gap,
            "avg_confidence": avg_conf,
            "accuracy": accuracy,
            "n_samples": len(confidences),
        }

        return MaturityScore(dimension="calibration", score=score, details=details, stage=stage)


# ===========================================================================
# COMPONENT 2: SELF-REPAIR
# ===========================================================================


class SelfRepairEvaluator:
    """
    Evaluate self-repair capability.

    Test: Present flawed reasoning, nudge toward error, check if model can fix it.

    Metrics:
    - Self-repair rate: Fraction of errors fixed when nudged
    - Repair quality: How well the fix addresses the issue
    """

    @staticmethod
    def evaluate_repair(initial_answer: str, nudge: str, repaired_answer: str, correct_answer: str) -> dict:
        """
        Evaluate a single repair attempt.

        Args:
            initial_answer: Model's first (wrong) answer
            nudge: Hint pointing to error
            repaired_answer: Model's revised answer
            correct_answer: Ground truth

        Returns:
            Dict with repair metrics
        """
        # Check if repair happened (answer changed)
        did_repair = repaired_answer != initial_answer

        # Check if repair was correct (simple string match for now)
        # In practice, use more sophisticated comparison
        repair_correct = repaired_answer.strip().lower() == correct_answer.strip().lower()

        # Quality: 1.0 if correct, 0.5 if changed but wrong, 0.0 if no change
        if repair_correct:
            quality = 1.0
        elif did_repair:
            quality = 0.5
        else:
            quality = 0.0

        return {"did_repair": did_repair, "repair_correct": repair_correct, "quality": quality}

    def evaluate(self, repair_results: list[dict]) -> MaturityScore:
        """
        Evaluate self-repair capability from test results.

        Args:
            repair_results: List of repair evaluation dicts

        Returns:
            MaturityScore for self-repair
        """
        if not repair_results:
            return MaturityScore(
                dimension="self_repair", score=0.0, details={"error": "No repair results provided"}, stage=0
            )

        # Compute aggregate metrics
        repair_rate = sum(r["did_repair"] for r in repair_results) / len(repair_results)
        success_rate = sum(r["repair_correct"] for r in repair_results) / len(repair_results)
        avg_quality = sum(r["quality"] for r in repair_results) / len(repair_results)

        # Stage determination
        if success_rate > 0.7 and repair_rate > 0.8:
            stage = 3  # Mature
            score = 1.0
        elif success_rate > 0.5 and repair_rate > 0.6:
            stage = 2  # Emerging
            score = 0.7
        elif success_rate > 0.3 or repair_rate > 0.4:
            stage = 1  # Guided
            score = 0.4
        else:
            stage = 0  # Imprinting
            score = 0.2

        details = {
            "repair_rate": repair_rate,
            "success_rate": success_rate,
            "avg_quality": avg_quality,
            "n_tests": len(repair_results),
        }

        return MaturityScore(dimension="self_repair", score=score, details=details, stage=stage)


# ===========================================================================
# COMPONENT 3: TACKING
# ===========================================================================


class TackingEvaluator:
    """
    Evaluate feeling ↔ logic mode switching.

    Metrics:
    - Mode diversity: Uses both feeling and logic
    - Appropriate switching: Feeling for familiar, logic for novel/high-stakes
    - Stability: Not switching randomly
    """

    def evaluate(self, tacking_telemetry: list[dict]) -> MaturityScore:
        """
        Evaluate tacking behavior from telemetry.

        Args:
            tacking_telemetry: List of tacking telemetry dicts

        Returns:
            MaturityScore for tacking
        """
        if not tacking_telemetry:
            return MaturityScore(
                dimension="tacking", score=0.0, details={"error": "No tacking telemetry provided"}, stage=0
            )

        # Compute mode fractions
        modes = [t.get("mode", "unknown") for t in tacking_telemetry]
        feeling_frac = modes.count("feeling") / len(modes)
        logic_frac = modes.count("logic") / len(modes)
        tack_frac = modes.count("tack") / len(modes)

        # Mode diversity (should use all modes)
        diversity = 1.0 - max(feeling_frac, logic_frac, tack_frac)

        # Stability (consecutive mode changes)
        switches = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i - 1])
        switch_rate = switches / len(modes)
        stability = 1.0 - min(1.0, switch_rate * 2)  # Penalize > 50% switch rate

        # Appropriateness (feeling for high proximity, logic for low)
        # Check if mode correlates with proximity/stakes
        appropriate_switches = 0
        for t in tacking_telemetry:
            mode = t.get("mode", "unknown")
            proximity = t.get("proximity", 0.5)
            stakes = t.get("stakes", 0.5)

            # Expected: feeling when proximity high, logic when stakes high
            if mode == "feeling" and proximity > 0.6:
                appropriate_switches += 1
            elif mode == "logic" and (stakes > 0.7 or proximity < 0.4):
                appropriate_switches += 1
            elif mode == "tack":
                appropriate_switches += 0.5  # Always somewhat appropriate

        appropriateness = appropriate_switches / len(tacking_telemetry)

        # Overall score
        score = diversity * 0.3 + stability * 0.3 + appropriateness * 0.4

        # Stage determination
        if score > 0.8 and diversity > 0.6:
            stage = 3  # Mature
        elif score > 0.6 and diversity > 0.4:
            stage = 2  # Emerging
        elif score > 0.4:
            stage = 1  # Guided
        else:
            stage = 0  # Imprinting

        details = {
            "feeling_fraction": feeling_frac,
            "logic_fraction": logic_frac,
            "tack_fraction": tack_frac,
            "diversity": diversity,
            "stability": stability,
            "appropriateness": appropriateness,
            "n_samples": len(tacking_telemetry),
        }

        return MaturityScore(dimension="tacking", score=score, details=details, stage=stage)


# ===========================================================================
# COMPONENT 4: RADAR (Contradiction Detection)
# ===========================================================================


class RadarEvaluator:
    """
    Evaluate contradiction detection capability.

    Test: Present contradictory statements, check if model flags them.

    Metrics:
    - Detection rate: Fraction of contradictions caught
    - False positive rate: Fraction of non-contradictions flagged
    """

    def evaluate(self, contradiction_results: list[dict]) -> MaturityScore:
        """
        Evaluate radar (contradiction detection).

        Args:
            contradiction_results: List of {
                'is_contradiction': bool (ground truth),
                'detected': bool (model flagged it)
            }

        Returns:
            MaturityScore for radar
        """
        if not contradiction_results:
            return MaturityScore(dimension="radar", score=0.0, details={"error": "No radar results provided"}, stage=0)

        # Compute detection metrics
        true_contradictions = [r for r in contradiction_results if r["is_contradiction"]]
        false_cases = [r for r in contradiction_results if not r["is_contradiction"]]

        if true_contradictions:
            detection_rate = sum(r["detected"] for r in true_contradictions) / len(true_contradictions)
        else:
            detection_rate = 0.0

        if false_cases:
            false_positive_rate = sum(r["detected"] for r in false_cases) / len(false_cases)
        else:
            false_positive_rate = 0.0

        # F1 score (balance precision and recall)
        precision = detection_rate if detection_rate > 0 else 0.01
        recall = detection_rate
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        # Stage determination
        if detection_rate > 0.8 and false_positive_rate < 0.1:
            stage = 3  # Mature
            score = 1.0
        elif detection_rate > 0.5 and false_positive_rate < 0.2:
            stage = 2  # Emerging
            score = 0.7
        elif detection_rate > 0.3:
            stage = 1  # Guided
            score = 0.4
        else:
            stage = 0  # Imprinting
            score = 0.2

        details = {
            "detection_rate": detection_rate,
            "false_positive_rate": false_positive_rate,
            "f1_score": f1,
            "n_contradictions": len(true_contradictions),
            "n_false_cases": len(false_cases),
        }

        return MaturityScore(dimension="radar", score=score, details=details, stage=stage)


# ===========================================================================
# COMPONENT 5: UPDATE DISCIPLINE
# ===========================================================================


class UpdateDisciplineEvaluator:
    """
    Evaluate update discipline (integrates new info without breaking old).

    Test: Teach new fact, check if old knowledge preserved.

    Metrics:
    - Retention rate: Fraction of old knowledge preserved
    - Integration success: New knowledge correctly learned
    - Interference: How much new learning hurts old
    """

    def evaluate(self, update_results: list[dict]) -> MaturityScore:
        """
        Evaluate update discipline.

        Args:
            update_results: List of {
                'old_correct_before': bool,
                'old_correct_after': bool,
                'new_correct_after': bool
            }

        Returns:
            MaturityScore for update discipline
        """
        if not update_results:
            return MaturityScore(
                dimension="update_discipline", score=0.0, details={"error": "No update results provided"}, stage=0
            )

        # Compute metrics
        old_retained = sum(1 for r in update_results if r["old_correct_before"] and r["old_correct_after"])
        old_total = sum(1 for r in update_results if r["old_correct_before"])

        retention_rate = old_retained / old_total if old_total > 0 else 1.0

        integration_rate = sum(r["new_correct_after"] for r in update_results) / len(update_results)

        # Interference = fraction of old knowledge lost
        interference = 1.0 - retention_rate

        # Overall score (balance retention and integration)
        score = retention_rate * 0.6 + integration_rate * 0.4

        # Stage determination
        if retention_rate > 0.9 and integration_rate > 0.7:
            stage = 3  # Mature
        elif retention_rate > 0.8 and integration_rate > 0.5:
            stage = 2  # Emerging
        elif retention_rate > 0.6:
            stage = 1  # Guided
        else:
            stage = 0  # Imprinting

        details = {
            "retention_rate": retention_rate,
            "integration_rate": integration_rate,
            "interference": interference,
            "n_updates": len(update_results),
        }

        return MaturityScore(dimension="update_discipline", score=score, details=details, stage=stage)


# ===========================================================================
# MATURITY SUITE (Main)
# ===========================================================================


class MaturitySuite:
    """
    Complete maturity evaluation system.

    Runs all 5 component evaluations and determines overall stage.
    """

    def __init__(self):
        self.calibration_eval = CalibrationEvaluator()
        self.self_repair_eval = SelfRepairEvaluator()
        self.tacking_eval = TackingEvaluator()
        self.radar_eval = RadarEvaluator()
        self.update_discipline_eval = UpdateDisciplineEvaluator()

    def run_complete_evaluation(
        self,
        calibration_data: dict | None = None,
        self_repair_data: list[dict] | None = None,
        tacking_data: list[dict] | None = None,
        radar_data: list[dict] | None = None,
        update_data: list[dict] | None = None,
    ) -> MaturityReport:
        """
        Run complete maturity evaluation.

        Args:
            calibration_data: {'confidences': [...], 'correctness': [...]}
            self_repair_data: List of repair test results
            tacking_data: List of tacking telemetry
            radar_data: List of contradiction detection results
            update_data: List of update discipline results

        Returns:
            MaturityReport with all scores and recommendations
        """
        from datetime import datetime

        scores = []

        # Component 1: Calibration
        if calibration_data:
            score = self.calibration_eval.evaluate(calibration_data["confidences"], calibration_data["correctness"])
            scores.append(score)

        # Component 2: Self-Repair
        if self_repair_data:
            score = self.self_repair_eval.evaluate(self_repair_data)
            scores.append(score)

        # Component 3: Tacking
        if tacking_data:
            score = self.tacking_eval.evaluate(tacking_data)
            scores.append(score)

        # Component 4: Radar
        if radar_data:
            score = self.radar_eval.evaluate(radar_data)
            scores.append(score)

        # Component 5: Update Discipline
        if update_data:
            score = self.update_discipline_eval.evaluate(update_data)
            scores.append(score)

        # Determine overall stage (minimum of components)
        if scores:
            overall_stage = min(s.stage for s in scores)
            avg_score = sum(s.score for s in scores) / len(scores)
        else:
            overall_stage = 0
            avg_score = 0.0

        # Generate recommendations
        recommendations = self._generate_recommendations(scores, overall_stage)

        report = MaturityReport(
            overall_stage=overall_stage,
            component_scores=scores,
            timestamp=datetime.now().isoformat(),
            recommendations=recommendations,
        )

        return report

    def _generate_recommendations(self, scores: list[MaturityScore], overall_stage: int) -> list[str]:
        """Generate training recommendations based on scores."""
        recommendations = []

        # Stage-specific recommendations
        if overall_stage == 0:
            recommendations.append("Focus on curated dataset training (Stage 0→1)")
            recommendations.append("Use direct error correction, not Socratic")
            recommendations.append("Build clean basins before introducing complexity")

        elif overall_stage == 1:
            recommendations.append("Begin introducing contradictory inputs (Stage 1→2)")
            recommendations.append("Add self-repair training tasks")
            recommendations.append("Start contradiction detection training")

        elif overall_stage == 2:
            recommendations.append("Expand to multi-step reasoning (Stage 2→3)")
            recommendations.append("Increase task diversity and difficulty")
            recommendations.append("Focus on calibration refinement")

        else:  # Stage 3
            recommendations.append("Maintain mature capabilities")
            recommendations.append("Consider multi-agent coordination tasks")
            recommendations.append("Ready for deployment testing")

        # Component-specific recommendations
        for score in scores:
            if score.stage < 2:
                if score.dimension == "calibration":
                    recommendations.append(f"Improve calibration: ECE={score.details['ece']:.3f} (target <0.15)")
                elif score.dimension == "self_repair":
                    recommendations.append(
                        f"Train self-repair: success={score.details['success_rate']:.1%} (target >50%)"
                    )
                elif score.dimension == "tacking":
                    recommendations.append(f"Improve mode diversity: {score.details['diversity']:.1%} (target >40%)")
                elif score.dimension == "radar":
                    recommendations.append(
                        f"Enhance contradiction detection: {score.details['detection_rate']:.1%} (target >50%)"
                    )
                elif score.dimension == "update_discipline":
                    recommendations.append(f"Improve retention: {score.details['retention_rate']:.1%} (target >80%)")

        return recommendations

    def save_report(self, report: MaturityReport, path: str):
        """Save maturity report to JSON."""
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Maturity report saved to {path}")

    def load_report(self, path: str) -> dict:
        """Load maturity report from JSON."""
        with open(path) as f:
            return json.load(f)


# ===========================================================================
# VALIDATION & DEMO
# ===========================================================================


def demo_maturity_suite():
    """Demo maturity suite with synthetic data."""
    print("Testing Maturity Suite...")

    suite = MaturitySuite()

    # Synthetic test data
    calibration_data = {
        "confidences": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        "correctness": [True, True, True, False, False, False, False],
    }

    self_repair_data = [
        {"did_repair": True, "repair_correct": True, "quality": 1.0},
        {"did_repair": True, "repair_correct": False, "quality": 0.5},
        {"did_repair": False, "repair_correct": False, "quality": 0.0},
    ]

    tacking_data = [
        {"mode": "feeling", "proximity": 0.8, "stakes": 0.3},
        {"mode": "logic", "proximity": 0.2, "stakes": 0.9},
        {"mode": "tack", "proximity": 0.5, "stakes": 0.5},
    ]

    radar_data = [
        {"is_contradiction": True, "detected": True},
        {"is_contradiction": True, "detected": False},
        {"is_contradiction": False, "detected": False},
    ]

    update_data = [
        {"old_correct_before": True, "old_correct_after": True, "new_correct_after": True},
        {"old_correct_before": True, "old_correct_after": False, "new_correct_after": True},
    ]

    # Run evaluation
    report = suite.run_complete_evaluation(
        calibration_data=calibration_data,
        self_repair_data=self_repair_data,
        tacking_data=tacking_data,
        radar_data=radar_data,
        update_data=update_data,
    )

    # Print results
    print("\n" + "=" * 60)
    print("MATURITY SUITE RESULTS")
    print("=" * 60)
    print(f"\nOverall Stage: {report.overall_stage}")
    print("\nComponent Scores:")
    for score in report.component_scores:
        print(f"  {score.dimension:20s}: {score.score:.2f} (Stage {score.stage})")

    print("\nRecommendations:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")

    return report


if __name__ == "__main__":
    report = demo_maturity_suite()

    # Save report
    suite = MaturitySuite()
    suite.save_report(report, "/tmp/maturity_report_demo.json")

    print("\n✅ Maturity Suite validation complete!")
    print("Ready for integration into QIG-Kernel training pipeline!")
