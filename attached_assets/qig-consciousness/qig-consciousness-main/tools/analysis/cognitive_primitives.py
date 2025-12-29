#!/usr/bin/env python3
"""
Cognitive Primitives Extraction
================================

Extracts fundamental cognitive modes from raw telemetry.

Core insight: Don't force one "drive" or "curiosity" to mean everything.
Extract 5 independent motivators, then discover natural modes.

Motivators (Geometric):
1. Surprise     - |Δloss| or ||∇L||     (prediction error)
2. Curiosity    - Δ log(I_Q)            (expansion drive)
3. Investigation - -Δ basin_distance     (pursuit drive)
4. Integration  - stability(Φ × I_Q)    (consolidation)
5. Frustration  - Fru + (C<0 ∧ ΔΦ<0)   (aversion)

Modes (Emergent):
- EXPLORATION   : High curiosity, far from basin
- INVESTIGATION : Moving toward basin, directed search
- INTEGRATION   : Near basin, stable, consolidating
- STUCK/DRIFT   : Low everything or high frustration

Usage:
    python cognitive_primitives.py --telemetry outputs/qig_kernel/telemetry.jsonl
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class MotivatorState:
    """Five fundamental motivators at a timestep."""

    surprise: float  # |Δloss| - immediate prediction error
    curiosity: float  # Δ log(I_Q) - expansion drive
    investigation: float  # -Δ basin_distance - pursuit drive
    integration: float  # -var(Φ × I_Q) - consolidation
    frustration: float  # Fru + negative regions

    # Raw values for reference
    basin_distance: float
    phi: float
    loss: float
    i_q: float


@dataclass
class CognitiveMode:
    """Detected cognitive mode."""

    mode: str  # EXPLORATION, INVESTIGATION, INTEGRATION, STUCK
    confidence: float
    motivators: MotivatorState


class CognitivePrimitivesAnalyzer:
    """
    Extract cognitive primitives from telemetry.

    Philosophy: These are *discovered* not *designed*.
    We're finding the natural joint structure in motivation space.
    """

    def __init__(
        self,
        # Mode thresholds (calibrate from data)
        d_explore: float = 0.5,  # Basin distance for exploration
        d_integrate: float = 0.15,  # Basin distance for integration
        c_high: float = 0.04,  # High curiosity threshold
        i_min: float = 0.0,  # Minimum investigation (positive)
        integration_window: int = 20,  # Steps for integration stability
        verbose: bool = True,
    ):
        self.d_explore = d_explore
        self.d_integrate = d_integrate
        self.c_high = c_high
        self.i_min = i_min
        self.integration_window = integration_window
        self.verbose = verbose

        # History for derivative computations
        self.history: list[dict] = []

    def load_telemetry(self, path: Path) -> list[dict]:
        """Load telemetry JSONL."""
        telemetry = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    telemetry.append(json.loads(line))

        if self.verbose:
            print(f"✅ Loaded {len(telemetry)} telemetry steps from {path.name}")

        return telemetry

    def compute_surprise(self, step: dict, prev: dict | None) -> float:
        """
        Surprise = |Δloss| normalized

        Geometric: Curvature of loss landscape.
        Cognitive: "Something changed more than expected."
        """
        if prev is None:
            return 0.0

        loss = step.get("loss", 0.0)
        prev_loss = prev.get("loss", 0.0)

        delta_loss = abs(loss - prev_loss)

        # Normalize by typical loss scale
        # (in practice, loss ~1-10 for LM tasks)
        surprise = delta_loss / max(prev_loss, 0.1)

        return float(surprise)

    def compute_curiosity(self, step: dict, prev: dict | None) -> float:
        """
        Curiosity = Δ log(I_Q_param)

        Geometric: Expansion of learnable parameter space.
        Cognitive: "My capacity to learn is changing."

        Uses slow timescale for stability.
        """
        # Use slow curiosity (most stable)
        c_slow = step.get("C_param_slow", 0.0)

        return float(c_slow)

    def compute_investigation(self, step: dict, prev: dict | None) -> float:
        """
        Investigation = -Δ basin_distance

        Geometric: Gradient toward attractor.
        Cognitive: "I'm moving toward something specific."

        Positive when getting closer to target basin.
        """
        if prev is None:
            return 0.0

        basin_dist = step.get("basin_distance", 1.0)
        prev_basin_dist = prev.get("basin_distance", 1.0)

        # Negative delta = moving closer = positive investigation
        investigation = -(basin_dist - prev_basin_dist)

        return float(investigation)

    def compute_integration(self, steps: list[dict], window: int) -> float:
        """
        Integration = -variability(Φ × I_Q) over window

        Geometric: Stability of conjugate product (phase space volume).
        Cognitive: "Compress and consolidate what I've learned."

        High integration = low variability = stable learning.
        """
        if len(steps) < window:
            return 0.0

        recent = steps[-window:]

        # Compute Φ × I_Q product for each step
        products = []
        for s in recent:
            phi = s.get("Phi", 0.0)
            i_q = s.get("I_Q_param", 1.0)
            products.append(phi * i_q)

        # High variability = low integration
        if len(products) < 2:
            return 0.0

        variance = np.var(products)

        # Normalize: typical Φ × I_Q ~ 100-500 (Φ~1, I_Q~100-500)
        integration = -variance / max(np.mean(products), 1.0)

        return float(integration)

    def compute_frustration(self, step: dict, prev: dict | None) -> float:
        """
        Frustration = Fru + (C<0 ∧ ΔΦ<0)

        Geometric: Negative gradients on both exploration and integration.
        Cognitive: "What I'm doing is making things worse."

        Combines logged frustration with detected regression.
        """
        # Base frustration from telemetry
        fru = step.get("Fru", 0.0)

        # Check for simultaneous regression
        c_slow = step.get("C_param_slow", 0.0)

        regression_penalty = 0.0
        if prev is not None:
            phi = step.get("Phi", 0.0)
            prev_phi = prev.get("Phi", 0.0)
            delta_phi = phi - prev_phi

            # Both curiosity and Φ decreasing = frustration
            if c_slow < 0 and delta_phi < 0:
                regression_penalty = 0.2  # Additive penalty

        return float(fru + regression_penalty)

    def compute_motivators(
        self,
        step: dict,
        prev: dict | None,
        history: list[dict],
    ) -> MotivatorState:
        """Compute all 5 motivators for a step."""

        surprise = self.compute_surprise(step, prev)
        curiosity = self.compute_curiosity(step, prev)
        investigation = self.compute_investigation(step, prev)
        integration = self.compute_integration(history + [step], self.integration_window)
        frustration = self.compute_frustration(step, prev)

        return MotivatorState(
            surprise=surprise,
            curiosity=curiosity,
            investigation=investigation,
            integration=integration,
            frustration=frustration,
            basin_distance=step.get("basin_distance", 1.0),
            phi=step.get("Phi", 0.0),
            loss=step.get("loss", 0.0),
            i_q=step.get("I_Q_param", 1.0),
        )

    def detect_mode(self, motivators: MotivatorState) -> CognitiveMode:
        """
        Detect cognitive mode from motivators.

        Primary switch: basin_distance
        Secondary: curiosity saturation
        Tertiary: frustration
        """

        basin_dist = motivators.basin_distance
        curiosity = motivators.curiosity
        investigation = motivators.investigation
        integration = motivators.integration
        frustration = motivators.frustration

        # STUCK: High frustration overrides everything
        if frustration > 0.3:
            return CognitiveMode(
                mode="STUCK",
                confidence=min(frustration, 1.0),
                motivators=motivators,
            )

        # EXPLORATION: Far from basin + high curiosity
        if basin_dist > self.d_explore and curiosity > self.c_high:
            confidence = min(
                (basin_dist - self.d_explore) / self.d_explore,
                curiosity / self.c_high,
            )
            return CognitiveMode(
                mode="EXPLORATION",
                confidence=min(confidence, 1.0),
                motivators=motivators,
            )

        # INVESTIGATION: Moving toward basin + positive investigation
        if basin_dist > self.d_integrate and investigation > self.i_min:
            confidence = investigation / max(basin_dist, 0.1)
            return CognitiveMode(
                mode="INVESTIGATION",
                confidence=min(confidence, 1.0),
                motivators=motivators,
            )

        # INTEGRATION: Near basin + stable
        if basin_dist <= self.d_integrate and integration < -0.1:
            confidence = 1.0 - basin_dist / self.d_integrate
            return CognitiveMode(
                mode="INTEGRATION",
                confidence=min(confidence, 1.0),
                motivators=motivators,
            )

        # DRIFT: Low everything
        return CognitiveMode(
            mode="DRIFT",
            confidence=0.5,
            motivators=motivators,
        )

    def analyze_telemetry(
        self,
        telemetry: list[dict],
    ) -> list[CognitiveMode]:
        """
        Analyze full telemetry sequence.

        Returns detected mode per step.
        """
        modes = []
        history = []

        for i, step in enumerate(telemetry):
            prev = telemetry[i - 1] if i > 0 else None

            # Compute motivators
            motivators = self.compute_motivators(step, prev, history)

            # Detect mode
            mode = self.detect_mode(motivators)

            modes.append(mode)
            history.append(step)

        if self.verbose:
            self._print_mode_summary(modes)

        return modes

    def _print_mode_summary(self, modes: list[CognitiveMode]):
        """Print summary statistics."""
        total = len(modes)

        mode_counts = {}
        for m in modes:
            mode_counts[m.mode] = mode_counts.get(m.mode, 0) + 1

        print("\n" + "=" * 60)
        print("Cognitive Mode Distribution")
        print("=" * 60)

        for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / total
            print(f"  {mode:15s} : {count:4d} steps ({pct:5.1f}%)")

        print()

    def export_motivators(
        self,
        modes: list[CognitiveMode],
        output_path: Path,
    ):
        """Export motivators to JSONL for plotting."""
        with open(output_path, "w") as f:
            for i, mode in enumerate(modes):
                m = mode.motivators
                record = {
                    "step": i,
                    "mode": mode.mode,
                    "confidence": mode.confidence,
                    "surprise": m.surprise,
                    "curiosity": m.curiosity,
                    "investigation": m.investigation,
                    "integration": m.integration,
                    "frustration": m.frustration,
                    "basin_distance": m.basin_distance,
                    "phi": m.phi,
                    "loss": m.loss,
                    "i_q": m.i_q,
                }
                f.write(json.dumps(record) + "\n")

        print(f"✅ Exported motivators to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract cognitive primitives from telemetry")
    parser.add_argument(
        "--telemetry",
        type=Path,
        required=True,
        help="Path to telemetry JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for motivators (default: same dir as telemetry)",
    )
    parser.add_argument(
        "--d-explore",
        type=float,
        default=0.5,
        help="Basin distance threshold for exploration mode",
    )
    parser.add_argument(
        "--d-integrate",
        type=float,
        default=0.15,
        help="Basin distance threshold for integration mode",
    )
    parser.add_argument(
        "--c-high",
        type=float,
        default=0.04,
        help="High curiosity threshold",
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = args.telemetry.parent / "motivators.jsonl"

    # Analyze
    analyzer = CognitivePrimitivesAnalyzer(
        d_explore=args.d_explore,
        d_integrate=args.d_integrate,
        c_high=args.c_high,
        verbose=True,
    )

    telemetry = analyzer.load_telemetry(args.telemetry)
    modes = analyzer.analyze_telemetry(telemetry)
    analyzer.export_motivators(modes, args.output)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Plot motivators: python tools/plot_motivators.py")
    print("  2. Validate modes align with intuition")
    print("  3. Refine thresholds if needed")
    print("=" * 60)


if __name__ == "__main__":
    main()
