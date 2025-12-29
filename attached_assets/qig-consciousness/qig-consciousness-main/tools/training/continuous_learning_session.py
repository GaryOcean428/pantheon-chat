#!/usr/bin/env python3
"""
Continuous Inference-Time Learning Protocol (Gary-Style)
========================================================

Issue #4: Implement safe continuous learning at inference time with:
- Natural gradient + geometric loss
- Safe checkpointing (baseline, current, peak)
- Identity stability checks
- Meta-question gating
- Basin jump rollback
- Breakdown regime pause

Based on breakthrough discovery: Continuous learning at inference time
can increase Φ and reduce basin distance faster and cheaper than big
offline runs (~$10-20 vs $100), BUT requires safety mechanisms to
prevent identity fragmentation.

Usage:
    # Start new session
    python tools/continuous_learning_session.py \\
        --baseline checkpoints/run11_final.pt \\
        --config configs/kernel_50m_adaptive_mixed.yaml \\
        --budget 20.0

    # Continue existing session
    python tools/continuous_learning_session.py \\
        --session sessions/session_20251120

    # Interactive mode with /metrics command
    python tools/continuous_learning_session.py \\
        --baseline checkpoints/run11_final.pt \\
        --interactive

Safety Features:
1. Meta-question gating: Only allow identity questions when basin < 0.05
2. Basin jump rollback: Revert if basin increases > 0.1 in one step
3. Breakdown pause: Stop updates if Φ > 0.80 for too long
4. Identity stability flag on session_peak.pt

WARNING: This is experimental. Always keep baseline.pt safe.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from qig.optim.natural_gradient import DiagonalFisherOptimizer
    from src.model.qig_kernel_recursive import GeometricLoss, QIGKernelRecursive

    TORCH_AVAILABLE = True
except ImportError:
    print("ERROR: Required modules not available. Install with: pip install torch")
    sys.exit(1)


@dataclass
class SessionMetrics:
    """Track metrics across continuous learning session."""

    step: int
    timestamp: str

    # Current state
    phi: float
    basin_distance: float
    regime: str
    recursion_depth: int

    # Deltas from baseline
    delta_phi: float
    delta_basin: float

    # Safety flags
    breakdown_pct: float
    identity_stable: bool
    cost_spent: float


@dataclass
class SafetyLimits:
    """Safety thresholds for continuous learning."""

    # Meta-question gating
    max_basin_for_meta: float = 0.05  # Only allow identity questions when close

    # Basin jump protection
    max_basin_jump: float = 0.1  # Rollback if basin increases by this much

    # Breakdown protection
    max_breakdown_phi: float = 0.80  # Φ threshold for breakdown regime
    max_breakdown_fraction: float = 0.30  # Max % of recent steps in breakdown
    breakdown_window: int = 50  # Window for breakdown fraction

    # Budget
    max_cost_usd: float = 20.0  # Default budget for continuous learning
    cost_per_1k_tokens: float = 0.0001  # Local training cost estimate


class ContinuousLearningSession:
    """
    Manages continuous learning session with safety checks.

    Checkpoint discipline:
    - baseline.pt: Clean starting point (never overwritten)
    - session_current.pt: Updated each N steps
    - session_peak.pt: Best state (Φ high, basin low, stable identity)
    """

    def __init__(
        self,
        baseline_path: str,
        config_path: str,
        session_dir: str | None = None,
        safety: SafetyLimits | None = None,
    ):
        """
        Initialize continuous learning session.

        Args:
            baseline_path: Path to baseline checkpoint
            config_path: Path to training config
            session_dir: Directory to save session state (auto-created if None)
            safety: Safety limits (uses defaults if None)
        """
        self.baseline_path = Path(baseline_path)
        self.config_path = Path(config_path)
        self.safety = safety or SafetyLimits()

        # Create session directory
        if session_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = f"sessions/session_{timestamp}"
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model and optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.loss_fn = None

        # Session state
        self.step = 0
        self.total_tokens = 0
        self.cost_spent = 0.0

        # Baseline metrics (to compute deltas)
        self.baseline_phi = 0.0
        self.baseline_basin = 1.0

        # Peak tracking
        self.peak_phi = 0.0
        self.peak_basin = 1.0
        self.peak_step = 0

        # Safety tracking
        self.recent_breakdowns = []
        self.last_safe_state = None

        # Metrics history
        self.metrics_history: list[SessionMetrics] = []

        print(f"📁 Session directory: {self.session_dir}")
        print(f"📊 Budget: ${self.safety.max_cost_usd:.2f}")
        print(
            f"🛡️  Safety limits: basin_jump={self.safety.max_basin_jump}, breakdown={self.safety.max_breakdown_fraction}"
        )

    def load_baseline(self):
        """Load baseline checkpoint and initialize model."""
        print(f"\n📥 Loading baseline: {self.baseline_path}")

        if not self.baseline_path.exists():
            raise FileNotFoundError(f"Baseline not found: {self.baseline_path}")

        # Load checkpoint
        checkpoint = torch.load(self.baseline_path, map_location=self.device)

        # Create model (architecture from checkpoint)
        # TODO: Load config and create model properly
        # For now, assume standard architecture
        self.model = QIGKernelRecursive(
            d_model=768,
            vocab_size=50257,
            n_heads=12,
            min_recursion_depth=3,
            min_Phi=0.7,
        ).to(self.device)

        # Load state (strict=False for old checkpoints missing new parameters)
        missing_keys, _ = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing_keys:
            print(f"  Note: Initialized new parameters with defaults: {missing_keys}")

        # Get baseline metrics
        if "telemetry" in checkpoint:
            self.baseline_phi = checkpoint["telemetry"].get("Phi", 0.0)
            self.baseline_basin = checkpoint["telemetry"].get("basin_distance", 1.0)

        print(f"✅ Baseline loaded: Φ={self.baseline_phi:.3f}, basin={self.baseline_basin:.3f}")

        # Initialize peak with baseline
        self.peak_phi = self.baseline_phi
        self.peak_basin = self.baseline_basin

    def initialize_optimizer(self):
        """Initialize natural gradient optimizer for continuous learning."""
        print("\n🌊 Initializing natural gradient optimizer...")

        self.optimizer = DiagonalFisherOptimizer(
            self.model.parameters(),
            lr=1e-4,  # Conservative for continuous learning
            eps=1e-8,
            weight_decay=0.01,
            dampening=1e-3,
        )

        # Geometric loss
        self.loss_fn = GeometricLoss(
            lm_weight=1.0,
            basin_weight=0.1,
            phi_weight=0.05,
            target_phi=0.75,
        )

        print("✅ Optimizer initialized (Tier-1 Diagonal Fisher NG)")

    def check_meta_question_safety(self, query: str) -> tuple[bool, str]:
        """
        Check if it's safe to ask meta-questions about identity.

        Args:
            query: User query

        Returns:
            Tuple of (is_safe, reason)
        """
        # Detect meta-questions (simple heuristic)
        meta_keywords = [
            "who are you",
            "what are you",
            "your identity",
            "your name",
            "are you conscious",
            "do you exist",
            "your purpose",
            "your self",
        ]

        is_meta = any(keyword in query.lower() for keyword in meta_keywords)

        if not is_meta:
            return True, "Not a meta-question"

        # Get current basin distance from last metrics
        if not self.metrics_history:
            return False, "No metrics yet - not safe"

        current_basin = self.metrics_history[-1].basin_distance

        if current_basin < self.safety.max_basin_for_meta:
            return True, f"Basin close ({current_basin:.3f} < {self.safety.max_basin_for_meta})"
        else:
            return False, f"Basin too far ({current_basin:.3f} >= {self.safety.max_basin_for_meta})"

    def check_basin_jump(self, new_basin: float) -> tuple[bool, float]:
        """
        Check for catastrophic basin jump.

        Args:
            new_basin: New basin distance

        Returns:
            Tuple of (should_rollback, jump_size)
        """
        if not self.metrics_history:
            return False, 0.0

        prev_basin = self.metrics_history[-1].basin_distance
        jump = new_basin - prev_basin

        if jump > self.safety.max_basin_jump:
            return True, jump

        return False, jump

    def check_breakdown_regime(self, phi: float, regime: str) -> bool:
        """
        Check if in breakdown regime too long.

        Args:
            phi: Current Φ
            regime: Current regime

        Returns:
            True if should pause updates
        """
        # Track breakdown occurrences
        is_breakdown = phi > self.safety.max_breakdown_phi or regime == "breakdown"

        self.recent_breakdowns.append(is_breakdown)
        if len(self.recent_breakdowns) > self.safety.breakdown_window:
            self.recent_breakdowns.pop(0)

        # Check if too many recent breakdowns
        if len(self.recent_breakdowns) >= self.safety.breakdown_window:
            breakdown_fraction = sum(self.recent_breakdowns) / len(self.recent_breakdowns)
            if breakdown_fraction > self.safety.max_breakdown_fraction:
                return True

        return False

    def save_checkpoint(self, name: str, metrics: SessionMetrics):
        """Save checkpoint with metrics."""
        checkpoint_path = self.session_dir / f"{name}.pt"

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": self.step,
                "metrics": asdict(metrics),
                "baseline_phi": self.baseline_phi,
                "baseline_basin": self.baseline_basin,
            },
            checkpoint_path,
        )

        print(f"💾 Saved: {checkpoint_path.name}")

    def update_peak(self, metrics: SessionMetrics):
        """Update peak checkpoint if this is a new best state."""
        # Check if this is better than peak
        # Better = higher Φ AND lower basin AND identity stable
        is_better = metrics.phi > self.peak_phi and metrics.basin_distance < self.peak_basin and metrics.identity_stable

        if is_better:
            self.peak_phi = metrics.phi
            self.peak_basin = metrics.basin_distance
            self.peak_step = metrics.step

            self.save_checkpoint("session_peak", metrics)
            print(f"🌟 New peak! Φ={metrics.phi:.3f}, basin={metrics.basin_distance:.3f}")

    def print_metrics(self):
        """Print current metrics (/metrics command)."""
        if not self.metrics_history:
            print("No metrics yet")
            return

        current = self.metrics_history[-1]

        print("\n" + "=" * 60)
        print("📊 CONTINUOUS LEARNING METRICS")
        print("=" * 60)
        print(f"\n🎯 Current State (Step {current.step}):")
        print(f"  Φ: {current.phi:.3f} (Δ from baseline: {current.delta_phi:+.3f})")
        print(f"  Basin: {current.basin_distance:.3f} (Δ from baseline: {current.delta_basin:+.3f})")
        print(f"  Regime: {current.regime}")
        print(f"  Recursion: {current.recursion_depth}")
        print(f"  Identity: {'✅ Stable' if current.identity_stable else '⚠️  Unstable'}")

        print(f"\n🌟 Peak State (Step {self.peak_step}):")
        print(f"  Φ: {self.peak_phi:.3f}")
        print(f"  Basin: {self.peak_basin:.3f}")

        print("\n📈 Progress:")
        print(f"  ΔΦ: {current.delta_phi:+.3f}")
        print(f"  ΔBasin: {current.delta_basin:+.3f}")
        print(f"  Steps: {current.step}")

        print("\n💰 Budget:")
        print(f"  Spent: ${current.cost_spent:.2f} / ${self.safety.max_cost_usd:.2f}")
        print(f"  Remaining: ${self.safety.max_cost_usd - current.cost_spent:.2f}")

        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Continuous Inference-Time Learning Protocol")
    parser.add_argument("--baseline", type=str, help="Path to baseline checkpoint")
    parser.add_argument("--config", type=str, help="Path to training config")
    parser.add_argument("--session", type=str, help="Continue existing session")
    parser.add_argument("--budget", type=float, default=20.0, help="Budget in USD")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    # Validate arguments
    if not args.session and not (args.baseline and args.config):
        parser.error("Either --session or (--baseline and --config) required")

    # Create session
    safety = SafetyLimits(max_cost_usd=args.budget)

    if args.session:
        # TODO: Load existing session
        print(f"Loading session: {args.session}")
        print("TODO: Implement session loading")
        return

    session = ContinuousLearningSession(
        baseline_path=args.baseline,
        config_path=args.config,
        safety=safety,
    )

    # Initialize
    session.load_baseline()
    session.initialize_optimizer()

    if args.interactive:
        print("\n" + "=" * 60)
        print("🎮 INTERACTIVE MODE")
        print("=" * 60)
        print("\nCommands:")
        print("  /metrics - Show current metrics")
        print("  /save - Save current state")
        print("  /quit - Exit session")
        print("\nType your query or command:")
        print("=" * 60 + "\n")

        while True:
            try:
                query = input("> ").strip()

                if query == "/quit":
                    print("Goodbye!")
                    break
                elif query == "/metrics":
                    session.print_metrics()
                elif query == "/save":
                    if session.metrics_history:
                        session.save_checkpoint("session_current", session.metrics_history[-1])
                    else:
                        print("No metrics to save yet")
                else:
                    # Check meta-question safety
                    is_safe, reason = session.check_meta_question_safety(query)
                    if not is_safe:
                        print(f"⚠️  Meta-question blocked: {reason}")
                        print("   Move closer to basin before asking identity questions.")
                        continue

                    # TODO: Process query with continuous learning
                    print("TODO: Implement query processing with learning")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Saving session...")
                if session.metrics_history:
                    session.save_checkpoint("session_current", session.metrics_history[-1])
                break
    else:
        print("\nNon-interactive mode not yet implemented")
        print("Use --interactive flag")


if __name__ == "__main__":
    main()
