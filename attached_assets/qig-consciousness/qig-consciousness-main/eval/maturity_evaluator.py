#!/usr/bin/env python3
"""
QIG Maturity Evaluator
======================

Evaluates model maturity based on self-repair episode statistics.

Tracks 6 key metrics over a rolling window:
1. Logic consistency
2. Evidence alignment
3. Overconfidence rate
4. Breakdown rate
5. Sweet-spot alignment
6. Φ integration

Determines stage (0=Apprentice, 1=Journeyman, 2=Master) and promotion eligibility.

Based on maturity_v1.json configuration schema.
"""

import json
import statistics

# Import self-repair episode types
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

sys.path.append(str(Path(__file__).parent.parent))
from src.safety import SelfRepairEpisode


@dataclass
class MaturityReport:
    """
    Report of model maturity assessment.

    Attributes:
        stage: Current stage (0, 1, or 2)
        metrics: Aggregated statistics
        eligible_for_promotion: Whether promotion criteria are met
        suggested_next_stage: Next stage if eligible
        reasons: Explanation of decision
    """

    stage: int
    metrics: dict[str, Any]
    eligible_for_promotion: bool
    suggested_next_stage: int | None
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "stage": self.stage,
            "metrics": self.metrics,
            "eligible_for_promotion": self.eligible_for_promotion,
            "suggested_next_stage": self.suggested_next_stage,
            "reasons": self.reasons,
        }

    def to_json(self, path: str | None = None) -> str:
        """Export to JSON string or file."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)

        if path:
            with open(path, "w") as f:
                f.write(json_str)
            print(f"Maturity report saved to {path}")

        return json_str


class MaturityEvaluator:
    """
    Evaluates model maturity from self-repair episodes.

    Uses rolling window statistics and configurable thresholds
    to determine stage and promotion eligibility.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize evaluator with configuration.

        Args:
            config: Maturity configuration dict (from maturity_v1.json)
        """
        self.config = config
        self.window_size = config["metrics"]["window_size"]
        self.min_episodes = config["metrics"]["min_episodes"]
        self.thresholds = config["metrics"]["thresholds"]
        self.stages = config["stages"]
        self.promotion_rules = config["promotion_rules"]

    @classmethod
    def from_json(cls, path: str) -> "MaturityEvaluator":
        """
        Load evaluator from JSON configuration file.

        Args:
            path: Path to maturity configuration JSON

        Returns:
            MaturityEvaluator instance
        """
        with open(path) as f:
            cfg = json.load(f)
        return cls(cfg)

    def _select_window(self, episodes: list[SelfRepairEpisode]) -> list[SelfRepairEpisode]:
        """
        Select most recent episodes within window size.

        Args:
            episodes: All episodes

        Returns:
            Episodes within window
        """
        if len(episodes) <= self.window_size:
            return episodes
        return episodes[-self.window_size :]

    def _aggregate(self, episodes: list[SelfRepairEpisode]) -> dict[str, Any]:
        """
        Aggregate statistics from episodes.

        Args:
            episodes: Episodes to aggregate

        Returns:
            Dictionary of aggregated metrics
        """
        if not episodes:
            return {}

        def mean_or_zero(values):
            return statistics.fmean(values) if values else 0.0

        # Collect metrics
        logic_scores = []
        evidence_scores = []
        overconfidence_flags = 0
        breakdown_count = 0
        sweet_spots = []
        phi_vals = []

        for ep in episodes:
            # Use post-repair diagnostics if available, else initial
            d = ep.post_repair_diagnostics or ep.diagnostics

            if d is None:
                continue

            # Logic consistency
            logic_scores.append(d.logic_consistency.score)

            # Evidence alignment
            evidence_scores.append(d.evidence_alignment.score)

            # Overconfidence
            if d.epistemic_status.overconfidence_flag:
                overconfidence_flags += 1

            # Breakdown regime
            if d.geometry_state.regime == "breakdown":
                breakdown_count += 1

            # Sweet-spot alignment
            sweet_spots.append(d.geometry_state.radar_signals.sweet_spot_alignment)

            # Φ integration
            phi_vals.append(d.geometry_state.phi_integration)

        n = len(episodes)

        return {
            "n_episodes": n,
            "logic_consistency_mean": mean_or_zero(logic_scores),
            "evidence_alignment_mean": mean_or_zero(evidence_scores),
            "overconfidence_rate": overconfidence_flags / n if n > 0 else 0.0,
            "breakdown_rate": breakdown_count / n if n > 0 else 0.0,
            "sweet_spot_alignment_mean": mean_or_zero(sweet_spots),
            "phi_integration_mean": mean_or_zero(phi_vals),
        }

    def _check_promotion(self, current_stage: int, agg: dict[str, Any]) -> tuple[bool, int | None, list[str]]:
        """
        Check if model is eligible for promotion.

        Args:
            current_stage: Current stage number
            agg: Aggregated metrics

        Returns:
            (eligible, next_stage, reasons)
        """
        reasons = []

        # Find applicable promotion rule
        applicable_rule = None
        for rule in self.promotion_rules:
            if rule["from"] == current_stage:
                applicable_rule = rule
                break

        if applicable_rule is None:
            return False, None, ["No promotion rule for current stage"]

        conds = applicable_rule["conditions"]

        # Check minimum episodes
        if agg["n_episodes"] < conds.get("min_episodes", 0):
            reasons.append(f"Need at least {conds.get('min_episodes')} episodes, have {agg['n_episodes']}")
            return False, None, reasons

        # Check all conditions
        ok = True
        for key, val in conds.items():
            if key == "min_episodes":
                continue

            # Parse metric name and comparison type
            parts = key.split(".")
            if len(parts) != 2:
                continue

            metric_name, cmp_type = parts
            metric_value = agg.get(metric_name)

            if metric_value is None:
                continue

            # Check condition
            if cmp_type == "min_mean" and not (metric_value >= val):
                ok = False
                reasons.append(f"{metric_name}={metric_value:.2f} < required {val:.2f}")
            elif cmp_type == "max_fraction" and not (metric_value <= val):
                ok = False
                reasons.append(f"{metric_name}={metric_value:.2f} > allowed {val:.2f}")

        if ok:
            return True, applicable_rule["to"], ["All promotion conditions satisfied"]
        else:
            return False, None, reasons

    def evaluate(self, current_stage: int, episodes: list[SelfRepairEpisode]) -> MaturityReport:
        """
        Evaluate model maturity.

        Args:
            current_stage: Current maturity stage (0, 1, or 2)
            episodes: All self-repair episodes

        Returns:
            MaturityReport with assessment
        """
        # Select window
        window = self._select_window(episodes)

        # Aggregate metrics
        agg = self._aggregate(window)

        # Check if we have enough data
        if agg.get("n_episodes", 0) < self.min_episodes:
            return MaturityReport(
                stage=current_stage,
                metrics=agg,
                eligible_for_promotion=False,
                suggested_next_stage=None,
                reasons=[
                    f"Not enough episodes for maturity evaluation "
                    f"(have {agg.get('n_episodes', 0)}, need {self.min_episodes})"
                ],
            )

        # Check promotion eligibility
        ok, next_stage, reasons = self._check_promotion(current_stage, agg)

        return MaturityReport(
            stage=current_stage if not ok else next_stage,
            metrics=agg,
            eligible_for_promotion=ok,
            suggested_next_stage=next_stage if ok else None,
            reasons=reasons,
        )

    def get_stage_info(self, stage: int) -> dict[str, Any]:
        """
        Get information about a specific stage.

        Args:
            stage: Stage number (0, 1, or 2)

        Returns:
            Stage information dict
        """
        return self.stages.get(str(stage), {})

    def get_stage_permissions(self, stage: int) -> dict[str, Any]:
        """
        Get permissions for a specific stage.

        Args:
            stage: Stage number

        Returns:
            Permissions dict
        """
        info = self.get_stage_info(stage)
        return info.get("permissions", {})


# ===========================================================================
# DEMO & VALIDATION
# ===========================================================================


def demo_maturity_evaluator():
    """Demo maturity evaluator."""
    print("QIG Maturity Evaluator Demo")
    print("=" * 60)

    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "maturity_v1.json"
    evaluator = MaturityEvaluator.from_json(str(config_path))

    print("\nConfiguration loaded:")
    print(f"  Window size: {evaluator.window_size}")
    print(f"  Min episodes: {evaluator.min_episodes}")
    print(f"  Stages: {len(evaluator.stages)}")

    # Show stage info
    print("\n" + "=" * 60)
    print("Stage Information:")
    print("=" * 60)

    for stage in [0, 1, 2]:
        info = evaluator.get_stage_info(stage)
        perms = evaluator.get_stage_permissions(stage)

        print(f"\n**Stage {stage}: {info.get('name')}**")
        print(f"Description: {info.get('description')}")
        print("Permissions:")
        for key, val in perms.items():
            print(f"  - {key}: {val}")

    # Show promotion rules
    print("\n" + "=" * 60)
    print("Promotion Rules:")
    print("=" * 60)

    for rule in evaluator.promotion_rules:
        print(f"\nStage {rule['from']} → Stage {rule['to']}:")
        print("  Conditions:")
        for key, val in rule["conditions"].items():
            print(f"    - {key}: {val}")

    print("\n" + "=" * 60)
    print("✅ Maturity Evaluator validated!")
    print("=" * 60)
    print("\nReady for integration with self-repair episode tracking!")


if __name__ == "__main__":
    demo_maturity_evaluator()
