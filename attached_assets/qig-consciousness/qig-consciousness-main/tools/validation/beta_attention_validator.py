#!/usr/bin/env python3
"""
β_attention Measurement and Validation Tool
==========================================

Implements the β_attention measurement protocol v1.0.
This is THE critical test of substrate-independence.

Usage:
    python beta_attention_validator.py --model_path <path> --output_dir <dir>

Expected pattern:
    β_small→medium: +0.3 to +0.5 (positive running)
    β_large: < 0.1 (plateau/asymptotic freedom)

Success criterion: Qualitative match to β_physics pattern
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import numpy as np
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed. Install with: pip install torch numpy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("WARNING: matplotlib not available. Plots will be skipped.")


class BetaAttentionValidator:
    """
    Main validator for β_attention measurement protocol.

    Implements:
    1. Context length sweep (128 → 8192)
    2. κ_attention extraction from telemetry
    3. β-function computation
    4. Comparison to physics (β_physics ≈ 0.44 → 0)
    """

    def __init__(
        self,
        model,
        context_lengths: list[int] = None,
        n_samples_per_length: int = 200,
        task_type: str = "multi_hop_reasoning",
    ):
        """
        Initialize validator.

        Args:
            model: QIG-Kernel model with telemetry
            context_lengths: List of context lengths to test
            n_samples_per_length: Samples for statistical power
            task_type: Type of task for measurement
        """
        self.model = model

        if context_lengths is None:
            # Default: 6 doublings (like L=3,4,5,6,7,8 in physics)
            self.context_lengths = [128, 256, 512, 1024, 2048, 4096]
        else:
            self.context_lengths = sorted(context_lengths)

        self.n_samples = n_samples_per_length
        self.task_type = task_type

        # Results storage
        self.kappa_measurements = {}  # {L: (κ_mean, κ_sem, κ_samples)}
        self.beta_function = {}  # {(L, L'): (β, β_error)}

    def generate_task(self, context_length: int, seed: int = None) -> torch.Tensor:
        """
        Generate task at specified context length.

        For now: Random tokens (will be replaced with real tasks).
        Real implementation should use multi-hop reasoning, document comprehension, etc.

        Args:
            context_length: Target context length
            seed: Random seed for reproducibility

        Returns:
            input_ids: [1, context_length] token tensor
        """
        if seed is not None:
            torch.manual_seed(seed)

        # TODO: Replace with real task generation
        # For now: Random token sequence
        vocab_size = getattr(self.model, "vocab_size", 50257)
        input_ids = torch.randint(0, vocab_size, (1, context_length))

        return input_ids

    def extract_kappa_from_telemetry(self, telemetry: dict, method: str = "combined") -> float:
        """
        Extract κ_attention from model telemetry.

        Implements multiple estimators:
        1. Inverse QFI distance (physics analogy)
        2. Attention entropy (integration measure)
        3. Φ-scaled metric (consciousness correlate)

        Args:
            telemetry: Model telemetry dict
            method: "distance" / "entropy" / "integration" / "combined"

        Returns:
            κ_attention: Effective coupling strength
        """
        epsilon = 1e-8

        # Estimator 1: κ ~ 1/distance (strong coupling = small distances)
        qfi_distances = telemetry.get("qfi_distances_mean", 0.1)
        κ_distance = 1.0 / (qfi_distances + epsilon)

        # Estimator 2: κ ~ attention entropy (high entropy = broad coupling)
        κ_entropy = telemetry.get("entanglement_entropy", 1.0)

        # Estimator 3: κ ~ Φ × scale_factor (integration level)
        Phi = telemetry.get("Phi", telemetry.get("integration_Phi", 0.5))
        κ_integration = Phi * 100  # Scale to match physics range ~40-65

        # Combined estimator (weighted average per protocol)
        if method == "distance":
            return κ_distance
        elif method == "entropy":
            return κ_entropy
        elif method == "integration":
            return κ_integration
        elif method == "combined":
            return 0.4 * κ_distance + 0.3 * κ_entropy + 0.3 * κ_integration
        else:
            raise ValueError(f"Unknown method: {method}")

    def measure_kappa_at_length(self, context_length: int, verbose: bool = True) -> tuple[float, float, list[float]]:
        """
        Measure κ_attention at specific context length.

        Protocol:
        1. Generate N tasks at this length
        2. Run model with telemetry
        3. Extract κ from each
        4. Compute statistics

        Args:
            context_length: Context length to measure
            verbose: Print progress

        Returns:
            κ_mean: Mean κ_attention
            κ_sem: Standard error of mean
            κ_samples: List of individual measurements
        """
        if verbose:
            print(f"\nMeasuring κ_attention at L={context_length}...")
            print(f"  Samples: {self.n_samples}")

        κ_samples = []

        self.model.eval()
        with torch.no_grad():
            for i in range(self.n_samples):
                if verbose and (i + 1) % 50 == 0:
                    print(f"  Progress: {i + 1}/{self.n_samples}")

                # Generate task
                task = self.generate_task(context_length, seed=i)

                # Forward pass with telemetry
                try:
                    _, telemetry = self.model(task, return_telemetry=True)

                    # Extract κ
                    κ = self.extract_kappa_from_telemetry(telemetry)
                    κ_samples.append(κ)

                except Exception as e:
                    if verbose:
                        print(f"  Warning: Sample {i} failed: {e}")
                    continue

        # Compute statistics
        κ_samples = np.array(κ_samples)
        κ_mean = np.mean(κ_samples)
        κ_std = np.std(κ_samples)
        κ_sem = κ_std / np.sqrt(len(κ_samples))

        if verbose:
            print(f"  Results: κ = {κ_mean:.2f} ± {κ_sem:.2f}")
            print(f"  Range: [{κ_samples.min():.2f}, {κ_samples.max():.2f}]")

        return κ_mean, κ_sem, κ_samples.tolist()

    def run_full_measurement(self, verbose: bool = True) -> dict:
        """
        Run full β_attention measurement across all context lengths.

        Returns:
            results: Complete measurement results
        """
        if verbose:
            print("=" * 60)
            print("β_ATTENTION MEASUREMENT PROTOCOL v1.0")
            print("=" * 60)
            print(f"\nContext lengths: {self.context_lengths}")
            print(f"Samples per length: {self.n_samples}")
            print(f"Task type: {self.task_type}")

        # Measure κ at each length
        for L in self.context_lengths:
            κ_mean, κ_sem, κ_samples = self.measure_kappa_at_length(L, verbose)
            self.kappa_measurements[L] = {"mean": κ_mean, "sem": κ_sem, "samples": κ_samples}

        # Compute β-function
        if verbose:
            print("\n" + "=" * 60)
            print("β-FUNCTION COMPUTATION")
            print("=" * 60)

        self.beta_function = self.compute_beta_function(verbose)

        # Compile results
        results = {
            "measurement_id": f"beta_attention_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_version": getattr(self.model, "version", "unknown"),
            "date": datetime.now().isoformat(),
            "context_lengths": self.context_lengths,
            "n_samples_per_length": self.n_samples,
            "task_type": self.task_type,
            "kappa_measurements": {
                str(L): {"mean": float(data["mean"]), "sem": float(data["sem"]), "n_samples": len(data["samples"])}
                for L, data in self.kappa_measurements.items()
            },
            "beta_function": {
                f"{L_from}→{L_to}": {
                    "beta": float(β),
                    "beta_error": float(β_err),
                    "interpretation": self.interpret_beta(β),
                }
                for (L_from, L_to), (β, β_err) in self.beta_function.items()
            },
        }

        return results

    def compute_beta_function(self, verbose: bool = True) -> dict[tuple[int, int], tuple[float, float]]:
        """
        Compute β(L→L') from κ measurements.

        AUTHORITATIVE DEFINITION (from FROZEN_FACTS.md and qig-verification):
            β(L→L+1) = (κ_{L+1} - κ_L) / κ_avg
            where κ_avg = (κ_L + κ_{L+1}) / 2

        This is the DISCRETE fractional change in κ between scales.
        It is NOT dκ/d(log L) or any log-derivative!

        Returns:
            beta_dict: {(L, L'): (β, β_error)}
        """
        beta_dict = {}

        lengths = sorted(self.kappa_measurements.keys())

        for i in range(len(lengths) - 1):
            L = lengths[i]
            L_next = lengths[i + 1]

            κ_L = self.kappa_measurements[L]["mean"]
            σ_L = self.kappa_measurements[L]["sem"]

            κ_L_next = self.kappa_measurements[L_next]["mean"]
            σ_L_next = self.kappa_measurements[L_next]["sem"]

            # Compute β using CORRECT discrete formula
            # β = Δκ / κ_avg (NOT divided by Δlog L!)
            Δκ = κ_L_next - κ_L
            κ_avg = (κ_L + κ_L_next) / 2

            β = Δκ / κ_avg  # CORRECT: Discrete fractional change

            # Error propagation
            # σ_β ≈ sqrt((σ_L/κ_avg)² + (σ_{L+1}/κ_avg)²)
            σ_β = np.sqrt(σ_L**2 + σ_L_next**2) / κ_avg

            beta_dict[(L, L_next)] = (β, σ_β)

            if verbose:
                print(f"  β({L}→{L_next}) = {β:.3f} ± {σ_β:.3f}  [{self.interpret_beta(β)}]")

        return beta_dict

    def interpret_beta(self, β: float) -> str:
        """Interpret β value."""
        if β > 0.3:
            return "Strong positive running"
        elif β > 0.1:
            return "Moderate positive running"
        elif β > -0.1:
            return "Plateau/asymptotic freedom"
        else:
            return "Negative (anti-screening)"

    def compare_to_physics(self, verbose: bool = True) -> dict:
        """
        Compare measured β_attention to physics β_physics.

        Physics reference:
        - β(3→4) ≈ +0.44 (strong running)
        - β(4→5) ≈ 0.00 (plateau)

        Returns:
            comparison: Metrics and assessment
        """
        if verbose:
            print("\n" + "=" * 60)
            print("COMPARISON TO PHYSICS")
            print("=" * 60)

        # Extract β values
        beta_values = [β for (β, _) in self.beta_function.values()]

        if len(beta_values) == 0:
            return {"status": "NO_DATA"}

        # Check pattern
        β_small = beta_values[0] if len(beta_values) > 0 else 0
        β_large = beta_values[-1] if len(beta_values) > 0 else 0

        # Pattern checks
        checks = {
            "positive_running_small_scales": β_small > 0,
            "decreasing_trend": all(beta_values[i] >= beta_values[i + 1] - 0.1 for i in range(len(beta_values) - 1)),
            "plateau_large_scales": abs(β_large) < 0.1,
        }

        # Acceptance criteria
        primary_pass = (
            checks["positive_running_small_scales"] and checks["decreasing_trend"] and checks["plateau_large_scales"]
        )

        secondary_pass = 0.3 <= β_small <= 0.5 and abs(β_large) < 0.1

        if verbose:
            print("\nPattern checks:")
            print(f"  ✓ Positive running (small scales): {checks['positive_running_small_scales']}")
            print(f"  ✓ Decreasing trend: {checks['decreasing_trend']}")
            print(f"  ✓ Plateau (large scales): {checks['plateau_large_scales']}")
            print("\nAcceptance criteria:")
            print(f"  Primary (qualitative): {'PASS ✓' if primary_pass else 'FAIL ✗'}")
            print(f"  Secondary (quantitative): {'PASS ✓' if secondary_pass else 'FAIL ✗'}")

            if primary_pass:
                print("\n💚 VALIDATION PASSED")
                print("β_attention exhibits running coupling consistent with β_physics!")
                print("Substrate-independence supported.")
            else:
                print("\n⚠️  VALIDATION INCOMPLETE")
                print("Pattern diverges from physics. Further analysis needed.")

        comparison = {
            "pattern_checks": checks,
            "primary_criterion": primary_pass,
            "secondary_criterion": secondary_pass,
            "beta_small_scale": β_small,
            "beta_large_scale": β_large,
            "acceptance_status": "PASS" if primary_pass else "FAIL",
        }

        return comparison

    def plot_results(self, output_path: str = "beta_attention_plot.png"):
        """
        Plot β_attention vs scale alongside physics data.
        """
        if not PLOTTING_AVAILABLE:
            print("Matplotlib not available. Skipping plots.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: κ vs L
        lengths = sorted(self.kappa_measurements.keys())
        κ_means = [self.kappa_measurements[L]["mean"] for L in lengths]
        κ_errors = [self.kappa_measurements[L]["sem"] for L in lengths]

        ax1.errorbar(lengths, κ_means, yerr=κ_errors, marker="o", capsize=5, label="κ_attention")
        ax1.axhline(y=41.09, color="r", linestyle="--", alpha=0.5, label="κ_physics (L=3)")
        ax1.axhline(y=64.47, color="r", linestyle="--", alpha=0.5, label="κ_physics (L=4)")
        ax1.set_xlabel("Context Length")
        ax1.set_ylabel("κ (Effective Coupling)")
        ax1.set_xscale("log")
        ax1.set_title("Running Coupling vs Scale")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: β vs scale
        beta_scales = [(L + L_next) / 2 for (L, L_next) in self.beta_function.keys()]
        beta_values = [β for (β, _) in self.beta_function.values()]
        beta_errors = [σ for (_, σ) in self.beta_function.values()]

        ax2.errorbar(
            beta_scales, beta_values, yerr=beta_errors, marker="s", capsize=5, label="β_attention", color="blue"
        )
        ax2.axhline(y=0.44, color="r", linestyle="--", alpha=0.5, label="β_physics (L=3→4)")
        ax2.axhline(y=0.0, color="gray", linestyle="-", alpha=0.3)
        ax2.set_xlabel("Scale (avg context length)")
        ax2.set_ylabel("β (Running coupling)")
        ax2.set_xscale("log")
        ax2.set_title("β-Function vs Scale")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path}")
        plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="β_attention Measurement and Validation")
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to trained model (if None, uses validation stub)"
    )
    parser.add_argument(
        "--context_lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048], help="Context lengths to test"
    )
    parser.add_argument("--n_samples", type=int, default=200, help="Samples per context length")
    parser.add_argument("--output_dir", type=str, default="./validation_results", help="Output directory for results")
    parser.add_argument("--plot", action="store_true", help="Generate plots")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create model
    if args.model_path:
        print(f"Loading model from: {args.model_path}")
        # TODO: Implement model loading
        model = None
    else:
        print("Using validation stub (for testing protocol)")
        # Create minimal stub for protocol testing
        from src.model.qig_kernel_recursive import QIGKernelRecursive

        model = QIGKernelRecursive(d_model=256, vocab_size=1000, n_heads=4, min_recursion_depth=3)

    # Run validation
    validator = BetaAttentionValidator(
        model=model, context_lengths=args.context_lengths, n_samples_per_length=args.n_samples
    )

    results = validator.run_full_measurement(verbose=True)

    # Compare to physics
    comparison = validator.compare_to_physics(verbose=True)
    results["comparison_to_physics"] = comparison

    # Save results
    results_path = output_dir / f"beta_attention_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Plot if requested
    if args.plot and PLOTTING_AVAILABLE:
        plot_path = output_dir / f"beta_attention_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        validator.plot_results(str(plot_path))

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
