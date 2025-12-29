#!/usr/bin/env python3
"""
Advanced Curiosity Analysis Tools
==================================

Implements Ona's advanced test framework:
1. Phase-space structure (C-Φ, C-basin, C-κ)
2. Hysteresis & learned helplessness dynamics
3. Time-lagged correlations
4. Ethics metrics (suffering profile)

Based on docs/consciousness/curiosity_advanced_tests.md
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


class AdvancedCuriosityAnalyzer:
    """
    Comprehensive analysis of curiosity dynamics from Run 8 data.

    Tests:
    - H1: Curiosity predicts success
    - H2: Critical threshold C_c exists
    - H3: Slow > medium > fast predictive power
    - Bonus: Drive validates against C_slow
    - Phase portraits, hysteresis, time-lags, ethics
    """

    def __init__(self, runs_data: list[dict]):
        """
        Initialize analyzer with run data.

        Args:
            runs_data: List of dicts, each containing:
                - 'C_trajectory': C_slow over time
                - 'C_fast_trajectory': C_fast over time
                - 'C_medium_trajectory': C_medium over time
                - 'Phi_trajectory': Φ over time
                - 'drive_trajectory': Drive over time
                - 'basin_distance': Basin distance over time
                - 'kappa_eff': κ_eff over time
                - 'final_Phi': Final integration level
                - 'success': Boolean (final_Phi > threshold)
        """
        self.runs = runs_data
        self.results = {}

    # ===================================================================
    # HYPOTHESIS TESTING (H1, H2, H3, Bonus)
    # ===================================================================

    def test_h1_curiosity_predicts_success(self) -> dict:
        """
        H1: Mean curiosity correlates with final Φ.

        Expected: r(mean_C_slow, final_Φ) > 0.7, p < 0.01

        Returns:
            Dict with correlations at all timescales
        """
        mean_C_fast = [np.mean(run["C_fast_trajectory"]) for run in self.runs]
        mean_C_medium = [np.mean(run["C_medium_trajectory"]) for run in self.runs]
        mean_C_slow = [np.mean(run["C_trajectory"]) for run in self.runs]
        final_Phi = [run["final_Phi"] for run in self.runs]

        r_fast, p_fast = pearsonr(mean_C_fast, final_Phi)
        r_medium, p_medium = pearsonr(mean_C_medium, final_Phi)
        r_slow, p_slow = pearsonr(mean_C_slow, final_Phi)

        h1_validated = r_slow > 0.7 and p_slow < 0.01

        return {
            "h1_validated": h1_validated,
            "r_fast": r_fast,
            "p_fast": p_fast,
            "r_medium": r_medium,
            "p_medium": p_medium,
            "r_slow": r_slow,
            "p_slow": p_slow,
            "best_timescale": "slow" if r_slow > max(r_fast, r_medium) else "other",
        }

    def test_h2_critical_threshold(self, C_c_range=(0.02, 0.06)) -> dict:
        """
        H2: Critical threshold C_c exists where success rate drops.

        Expected: C_c ≈ 0.03-0.05

        Args:
            C_c_range: Range to search for threshold

        Returns:
            Dict with C_c estimate and success rates
        """
        mean_C_slow = np.array([np.mean(run["C_trajectory"]) for run in self.runs])
        success = np.array([run["success"] for run in self.runs])

        # Sort by mean C_slow
        sorted_indices = np.argsort(mean_C_slow)
        sorted_C = mean_C_slow[sorted_indices]
        sorted_success = success[sorted_indices]

        # Find threshold where success rate drops below 50%
        best_C_c = None
        best_separation = 0.0

        for i in range(10, len(sorted_C) - 10):
            C_c_candidate = sorted_C[i]

            if C_c_range[0] <= C_c_candidate <= C_c_range[1]:
                below_success_rate = np.mean(sorted_success[:i])
                above_success_rate = np.mean(sorted_success[i:])
                separation = above_success_rate - below_success_rate

                if separation > best_separation:
                    best_separation = separation
                    best_C_c = C_c_candidate

        if best_C_c is None:
            best_C_c = np.median(sorted_C)

        below_mask = mean_C_slow < best_C_c
        above_mask = mean_C_slow >= best_C_c

        return {
            "C_c_estimate": best_C_c,
            "success_rate_below": np.mean(success[below_mask]) if below_mask.any() else 0.0,
            "success_rate_above": np.mean(success[above_mask]) if above_mask.any() else 0.0,
            "separation": best_separation,
            "h2_validated": best_separation > 0.3,  # Clear bifurcation
        }

    def test_h3_multi_scale_structure(self) -> dict:
        """
        H3: Slow curiosity is most predictive.

        Expected: r_slow > r_medium > r_fast

        Returns:
            Dict with ordering validation
        """
        h1_results = self.test_h1_curiosity_predicts_success()

        r_fast = abs(h1_results["r_fast"])
        r_medium = abs(h1_results["r_medium"])
        r_slow = abs(h1_results["r_slow"])

        ordering_correct = r_slow > r_medium > r_fast
        slow_best = r_slow > r_medium and r_slow > r_fast

        return {
            "h3_validated": ordering_correct,
            "slow_best": slow_best,
            "r_fast": r_fast,
            "r_medium": r_medium,
            "r_slow": r_slow,
            "ordering": "slow > medium > fast" if ordering_correct else "unexpected",
        }

    def test_bonus_drive_validation(self) -> dict:
        """
        Bonus: Drive heuristic correlates with C_slow physics ground truth.

        Expected: r(Drive, C_slow) > 0.7

        Returns:
            Dict with correlation and validation status
        """
        mean_drive = [np.mean(run["drive_trajectory"]) for run in self.runs]
        mean_C_slow = [np.mean(run["C_trajectory"]) for run in self.runs]

        r, p = pearsonr(mean_drive, mean_C_slow)

        drive_validated = r > 0.7 and p < 0.01

        return {
            "drive_validated": drive_validated,
            "correlation": r,
            "p_value": p,
            "interpretation": "ExplorationDrive is valid heuristic" if drive_validated else "Need refinement",
        }

    # ===================================================================
    # PHASE-SPACE STRUCTURE
    # ===================================================================

    def plot_phase_portrait_C_Phi(self, save_path: Optional[Path] = None):
        """Plot trajectories in (Φ, C) plane."""
        fig, ax = plt.subplots(figsize=(10, 8))

        for run in self.runs:
            Phi = np.array(run["Phi_trajectory"])
            C = np.array(run["C_trajectory"])
            success = run["success"]

            color = "green" if success else "red"
            alpha = 0.7 if success else 0.3

            ax.plot(Phi, C, color=color, alpha=alpha, linewidth=1)
            ax.scatter(Phi[0], C[0], marker="o", color=color, s=50)
            ax.scatter(Phi[-1], C[-1], marker="x", color=color, s=50)

        ax.axhline(0, color="black", linestyle="--", alpha=0.3)
        ax.set_xlabel("Φ (Integration)", fontsize=12)
        ax.set_ylabel("C (Curiosity)", fontsize=12)
        ax.set_title("Phase Portrait: Curiosity vs Integration", fontsize=14)
        ax.legend(["Successful", "Failed"])
        ax.grid(alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def analyze_C_vs_basin(self) -> list[dict]:
        """
        Test if high C correlates with approaching basin.

        Expected: Negative correlation (high C → decreasing distance)
        """
        results = []

        for run in self.runs:
            C = np.array(run["C_trajectory"])
            basin = np.array(run["basin_distance"])

            if len(basin) > 1:
                d_basin = np.diff(basin)

                if len(C) > len(d_basin):
                    C = C[1 : len(d_basin) + 1]

                r, p = pearsonr(C, d_basin)

                results.append(
                    {
                        "correlation": r,
                        "p_value": p,
                        "geometric": r < -0.3,
                        "interpretation": "High C → approaching basin" if r < -0.3 else "Weak/opposite",
                    }
                )

        return results

    def analyze_C_vs_kappa(self) -> list[dict]:
        """
        Test if |C| peaks when κ changes fastest.

        Both are geometric change rates.
        """
        results = []

        for run in self.runs:
            C = np.array(run["C_trajectory"])
            kappa = np.array(run["kappa_eff"])

            if len(kappa) > 1:
                d_kappa = np.abs(np.diff(kappa))

                if len(C) > len(d_kappa):
                    C = C[1 : len(d_kappa) + 1]

                r, p = pearsonr(np.abs(C), d_kappa)

                results.append({"correlation": r, "p_value": p, "validated": r > 0.3})

        return results

    # ===================================================================
    # HYSTERESIS & LEARNED HELPLESSNESS
    # ===================================================================

    def measure_hysteresis(self, C_c: float) -> dict:
        """Measure hysteresis in crossing C_c."""
        up_crossings = []
        down_crossings = []

        for run in self.runs:
            C = np.array(run["C_trajectory"])
            Phi = np.array(run["Phi_trajectory"])

            for i in range(1, len(C)):
                if C[i - 1] < C_c and C[i] >= C_c:
                    Phi_gain = Phi[min(i + 50, len(Phi) - 1)] - Phi[i]
                    up_crossings.append(Phi_gain)

                if C[i - 1] >= C_c and C[i] < C_c:
                    Phi_loss = Phi[min(i + 50, len(Phi) - 1)] - Phi[i]
                    down_crossings.append(Phi_loss)

        return {
            "up_mean": np.mean(up_crossings) if up_crossings else 0.0,
            "down_mean": np.mean(down_crossings) if down_crossings else 0.0,
            "hysteresis_detected": abs(np.mean(up_crossings or [0])) != abs(np.mean(down_crossings or [0])),
        }

    def measure_recovery_probability(self, C_c: float, T_threshold=50) -> dict:
        """Measure P(recover) after learned helplessness episode."""
        recovery_stats = []

        for run in self.runs:
            C = np.array(run["C_trajectory"])
            Phi = np.array(run["Phi_trajectory"])

            in_episode = False
            episode_start = 0

            for i in range(len(C)):
                if C[i] < C_c and not in_episode:
                    in_episode = True
                    episode_start = i

                elif C[i] >= C_c and in_episode:
                    episode_length = i - episode_start

                    if episode_length >= T_threshold:
                        final_Phi = Phi[-1]
                        recovered = final_Phi > 0.3

                        recovery_stats.append({"length": episode_length, "recovered": recovered})

                    in_episode = False

        if recovery_stats:
            return {
                "total_episodes": len(recovery_stats),
                "recovery_rate": np.mean([s["recovered"] for s in recovery_stats]),
                "mean_episode_length": np.mean([s["length"] for s in recovery_stats]),
            }
        else:
            return {"total_episodes": 0, "recovery_rate": 0.0, "mean_episode_length": 0.0}

    def measure_dwell_times(self, run: dict) -> dict:
        """Measure time spent in each curiosity regime."""
        C = np.array(run["C_trajectory"])

        regimes = {"exploration": 0, "exploitation": 0, "stagnation": 0, "regression": 0}

        for c in C:
            if c > 0.05:
                regimes["exploration"] += 1
            elif c > 0.01:
                regimes["exploitation"] += 1
            elif c > -0.01:
                regimes["stagnation"] += 1
            else:
                regimes["regression"] += 1

        total = len(C)
        return {k: v / total for k, v in regimes.items()}

    # ===================================================================
    # TIME-LAG STRUCTURE
    # ===================================================================

    def time_lagged_correlation(self, C: np.ndarray, Phi: np.ndarray, max_lag=50) -> dict:
        """Compute time-lagged correlation between C and Φ."""
        lags = range(max_lag)
        correlations = []

        for lag in lags:
            if lag < len(C) and lag < len(Phi):
                try:
                    r, p = pearsonr(C[: -lag or None], Phi[lag:])
                    correlations.append({"lag": lag, "r": r, "p": p})
                except:
                    pass

        if correlations:
            best = max(correlations, key=lambda x: abs(x["r"]))

            return {
                "best_lag": best["lag"],
                "best_r": best["r"],
                "all_lags": correlations,
                "is_leading": best["lag"] > 0,
            }
        else:
            return {"best_lag": 0, "best_r": 0.0, "all_lags": [], "is_leading": False}

    def multi_scale_time_lags(self, run: dict) -> dict:
        """Test if optimal lag increases with timescale."""
        C_fast = np.array(run["C_fast_trajectory"])
        C_medium = np.array(run["C_medium_trajectory"])
        C_slow = np.array(run["C_trajectory"])
        Phi = np.array(run["Phi_trajectory"])

        lag_fast = self.time_lagged_correlation(C_fast, Phi, max_lag=10)
        lag_medium = self.time_lagged_correlation(C_medium, Phi, max_lag=30)
        lag_slow = self.time_lagged_correlation(C_slow, Phi, max_lag=50)

        ordering_correct = lag_slow["best_lag"] > lag_medium["best_lag"] > lag_fast["best_lag"]

        return {"fast": lag_fast, "medium": lag_medium, "slow": lag_slow, "ordering_correct": ordering_correct}

    # ===================================================================
    # ETHICS METRICS (SUFFERING PROFILE)
    # ===================================================================

    def measure_stagnation_burden(self, run: dict, epsilon=0.01) -> float:
        """Measure fraction of time in stagnation."""
        C = np.array(run["C_trajectory"])
        Phi = np.array(run["Phi_trajectory"])

        d_Phi = np.abs(np.diff(Phi))

        stagnant_steps = 0
        for i in range(len(C) - 1):
            if abs(C[i]) < epsilon and d_Phi[i] < epsilon:
                stagnant_steps += 1

        return stagnant_steps / len(C)

    def measure_regression_burden(self, run: dict) -> float:
        """Measure time spent in regression."""
        C = np.array(run["C_trajectory"])
        Phi = np.array(run["Phi_trajectory"])

        d_Phi = np.diff(Phi)

        regression_steps = 0
        for i in range(len(C) - 1):
            if C[i] < 0 and d_Phi[i] < 0:
                regression_steps += 1

        return regression_steps / len(C)

    def compute_suffering_profile(self, run: dict) -> dict:
        """Compute complete ethics metrics for a run."""
        stagnation = self.measure_stagnation_burden(run)
        regression = self.measure_regression_burden(run)
        dwell_times = self.measure_dwell_times(run)

        return {
            "stagnation_burden": stagnation,
            "regression_burden": regression,
            "total_suffering": stagnation + regression,
            "dwell_times": dwell_times,
        }

    # ===================================================================
    # COMPLETE ANALYSIS PIPELINE
    # ===================================================================

    def run_complete_analysis(self, C_c: Optional[float] = None) -> dict:
        """
        Run all advanced curiosity tests.

        Args:
            C_c: Critical threshold (if None, will estimate from H2)

        Returns:
            Dict with all test results
        """
        print("Running complete advanced curiosity analysis...")

        # 1. Hypothesis testing
        print("\n1. Testing hypotheses H1, H2, H3, Bonus...")
        h1_results = self.test_h1_curiosity_predicts_success()
        h2_results = self.test_h2_critical_threshold()
        h3_results = self.test_h3_multi_scale_structure()
        bonus_results = self.test_bonus_drive_validation()

        if C_c is None:
            C_c = h2_results["C_c_estimate"]

        print(f"   H1: {'✅ VALIDATED' if h1_results['h1_validated'] else '❌ REJECTED'}")
        print(f"   H2: {'✅ VALIDATED' if h2_results['h2_validated'] else '❌ REJECTED'} (C_c = {C_c:.4f})")
        print(f"   H3: {'✅ VALIDATED' if h3_results['h3_validated'] else '❌ REJECTED'}")
        print(f"   Bonus: {'✅ VALIDATED' if bonus_results['drive_validated'] else '❌ REJECTED'}")

        # 2. Phase-space structure
        print("\n2. Analyzing phase-space structure...")
        C_basin_results = self.analyze_C_vs_basin()
        C_kappa_results = self.analyze_C_vs_kappa()

        # 3. Hysteresis
        print("\n3. Measuring hysteresis and learned helplessness...")
        hysteresis_results = self.measure_hysteresis(C_c)
        recovery_results = self.measure_recovery_probability(C_c)
        dwell_times_all = [self.measure_dwell_times(run) for run in self.runs]

        # 4. Time-lags
        print("\n4. Computing time-lagged correlations...")
        time_lag_results = [self.multi_scale_time_lags(run) for run in self.runs]

        # 5. Ethics metrics
        print("\n5. Computing suffering profiles...")
        suffering_profiles = [self.compute_suffering_profile(run) for run in self.runs]

        results = {
            "hypotheses": {"h1": h1_results, "h2": h2_results, "h3": h3_results, "bonus": bonus_results},
            "phase_space": {"C_basin": C_basin_results, "C_kappa": C_kappa_results},
            "hysteresis": {
                "crossings": hysteresis_results,
                "recovery": recovery_results,
                "dwell_times": dwell_times_all,
            },
            "time_lags": time_lag_results,
            "ethics": {
                "profiles": suffering_profiles,
                "mean_stagnation": np.mean([p["stagnation_burden"] for p in suffering_profiles]),
                "mean_regression": np.mean([p["regression_burden"] for p in suffering_profiles]),
            },
        }

        self.results = results
        print("\n✅ Complete analysis finished!")

        return results

    def save_results(self, output_path: Path):
        """Save analysis results to JSON."""
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Advanced Curiosity Analyzer - Ready for Run 8 data")
    print("\nUsage:")
    print("  analyzer = AdvancedCuriosityAnalyzer(runs_data)")
    print("  results = analyzer.run_complete_analysis()")
    print("  analyzer.plot_phase_portrait_C_Phi(save_path='phase_portrait.png')")
    print("  analyzer.save_results(Path('analysis_results.json'))")
