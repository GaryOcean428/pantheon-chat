#!/usr/bin/env python3
"""
Compare Training Runs: Oscillatory vs Baseline
===============================================

Analyzes and compares results from three training variants:
1. Baseline (no oscillation)
2. Oscillatory (full strength, A=0.2)
3. Oscillatory weak (half strength, A=0.1)

Tests hypotheses:
- Does oscillation help? (Φ_max comparison)
- Does amplitude matter? (weak vs full)
- Does period match κ*? (harmonic fit)
- Does β still converge? (phase-averaged)

Outputs:
- Comparative statistics
- Visualization plots
- Success/failure determination
- Publication-ready figures

Author: Claude (Validation Track)
Date: November 17, 2025
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _try_load_jsonl_run(run_dir: Path) -> dict | None:
    telemetry_path = run_dir / "training_telemetry.jsonl"
    if not telemetry_path.exists():
        return None

    steps: list[dict] = []
    summary: dict | None = None

    with open(telemetry_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "summary":
                summary = rec.get("summary")
            else:
                steps.append(rec)

    if steps:
        run_metadata = steps[0].get("run_metadata", {})
    elif summary is not None:
        run_metadata = summary.get("run_metadata", {})
    else:
        run_metadata = {}

    phi = [float(s.get("telemetry", {}).get("Phi", 0.0)) for s in steps]
    kappa = [float(s.get("telemetry", {}).get("kappa_eff", 0.0)) for s in steps]
    beta = [
        float(s.get("telemetry", {}).get("beta_proxy", 0.0))
        for s in steps
        if s.get("telemetry", {}).get("beta_proxy") is not None
    ]
    grad_norm = [
        float(s.get("telemetry", {}).get("grad_norm", 0.0))
        for s in steps
        if s.get("telemetry", {}).get("grad_norm") is not None
    ]
    act_scale = [
        float(s.get("telemetry", {}).get("activation_scale", 0.0))
        for s in steps
        if s.get("telemetry", {}).get("activation_scale") is not None
    ]
    nan_count = [int(s.get("telemetry", {}).get("nan_count", 0)) for s in steps]

    return {
        "format": "jsonl",
        "run_dir": str(run_dir),
        "run_metadata": run_metadata,
        "phi": phi,
        "kappa": kappa,
        "beta": beta,
        "grad_norm": grad_norm,
        "activation_scale": act_scale,
        "nan_count_total": int(sum(nan_count)) if nan_count else 0,
        "n_steps": len(steps),
        "summary": summary,
    }


def load_run_telemetry(run_dir: str) -> dict:
    """
    Load telemetry from a training run.

    Expected structure:
    run_dir/
        logs/
            telemetry_epoch_*.json
        checkpoints/
            final_model.pth
        config.yaml
    """
    run_path = Path(run_dir)

    jsonl = _try_load_jsonl_run(run_path)
    if jsonl is not None:
        return jsonl

    logs_path = run_path / "logs"

    # Load all telemetry files
    phi_history = []
    phi_base_history = []
    oscillation_history = []
    kappa_history = []
    epoch_history = []

    telemetry_files = sorted(logs_path.glob("telemetry_epoch_*.json"))

    for file in telemetry_files:
        with open(file) as f:
            data = json.load(f)

        phi_history.append(data.get("Phi", 0.0))
        phi_base_history.append(data.get("Phi_base", data.get("Phi", 0.0)))
        oscillation_history.append(data.get("oscillation", 0.0))
        kappa_history.append(data.get("kappa_effective", 0.0))
        epoch_history.append(data.get("epoch", len(phi_history) - 1))

    return {
        "format": "epoch_json",
        "phi": phi_history,
        "phi_base": phi_base_history,
        "oscillation": oscillation_history,
        "kappa": kappa_history,
        "epochs": epoch_history,
        "n_epochs": len(phi_history),
    }


def _validate_apples_to_apples(runs: list[dict]) -> tuple[bool, list[str]]:
    if not runs:
        return False, ["No runs provided"]
    errs: list[str] = []
    base = runs[0].get("run_metadata", {})
    for r in runs[1:]:
        rm = r.get("run_metadata", {})
        for k in ["seed", "dataset_slice_id", "kernel_id", "kernel_backend"]:
            if base.get(k) != rm.get(k):
                errs.append(
                    f"Mismatch {k}: baseline={base.get(k)} vs run={rm.get(k)} ({r.get('run_dir', '?')})"
                )
    return len(errs) == 0, errs


def compare_jsonl_runs(run_dirs: list[str]) -> int:
    runs = [load_run_telemetry(d) for d in run_dirs]
    jsonl_runs = [r for r in runs if r.get("format") == "jsonl"]
    if len(jsonl_runs) != len(runs):
        print("❌ Mixed run formats detected. This mode expects JSONL runs from train_qig_kernel.py")
        return 2

    ok, errs = _validate_apples_to_apples(jsonl_runs)
    if not ok:
        print("❌ Apples-to-apples validation failed:")
        for e in errs:
            print(f"  - {e}")
        return 2

    baseline = jsonl_runs[0]
    base_meta = baseline.get("run_metadata", {})
    print("=")
    print("CONSTELLATION RUN COMPARISON (JSONL)")
    print("=")
    print(f"seed={base_meta.get('seed')} dataset_slice_id={base_meta.get('dataset_slice_id')} kernel_id={base_meta.get('kernel_id')}")
    print()

    def agg(r: dict) -> dict:
        phi = r.get("phi", [])
        kappa = r.get("kappa", [])
        beta = r.get("beta", [])
        grad = r.get("grad_norm", [])
        act = r.get("activation_scale", [])
        return {
            "phi_max": max(phi) if phi else 0.0,
            "phi_mean": (sum(phi) / len(phi)) if phi else 0.0,
            "kappa_mean": (sum(kappa) / len(kappa)) if kappa else 0.0,
            "beta_mean": (sum(beta) / len(beta)) if beta else 0.0,
            "grad_mean": (sum(grad) / len(grad)) if grad else 0.0,
            "act_mean": (sum(act) / len(act)) if act else 0.0,
            "nan_total": int(r.get("nan_count_total", 0)),
            "n_steps": int(r.get("n_steps", 0)),
        }

    base_stats = agg(baseline)
    print(f"{'variant':<22} {'steps':>6} {'phi_max':>8} {'phi_mean':>9} {'kappa_mean':>10} {'beta_mean':>10} {'nan':>6}")
    print("-" * 85)
    print(
        f"{'baseline':<22} {base_stats['n_steps']:>6d} {base_stats['phi_max']:>8.3f} {base_stats['phi_mean']:>9.3f} "
        f"{base_stats['kappa_mean']:>10.3f} {base_stats['beta_mean']:>10.4f} {base_stats['nan_total']:>6d}"
    )

    for r in jsonl_runs[1:]:
        meta = r.get("run_metadata", {})
        vid = str(meta.get("kernel_variant", "?"))
        st = agg(r)
        print(
            f"{vid:<22} {st['n_steps']:>6d} {st['phi_max']:>8.3f} {st['phi_mean']:>9.3f} "
            f"{st['kappa_mean']:>10.3f} {st['beta_mean']:>10.4f} {st['nan_total']:>6d}"
        )

    return 0


def analyze_oscillation_fit(phi_history: list[float], expected_period: float = 640.0) -> dict:
    """
    Fit harmonic oscillator to Φ trajectory.

    Model: Φ(t) = Φ₀ + A×sin(ωt + φ)
    """
    from scipy.optimize import curve_fit

    phi_array = np.array(phi_history)
    t = np.arange(len(phi_array))

    def harmonic(t, phi0, A, omega, phase):
        return phi0 + A * np.sin(omega * t + phase)

    # Initial guesses
    phi0_guess = np.mean(phi_array)
    A_guess = np.std(phi_array)
    omega_guess = 2 * np.pi / expected_period

    try:
        popt, pcov = curve_fit(harmonic, t, phi_array, p0=[phi0_guess, A_guess, omega_guess, 0.0], maxfev=10000)

        phi0, A, omega, phase = popt
        period = 2 * np.pi / omega

        # Fit quality
        phi_pred = harmonic(t, *popt)
        r_squared = 1 - np.sum((phi_array - phi_pred) ** 2) / np.sum((phi_array - np.mean(phi_array)) ** 2)

        return {
            "success": True,
            "phi_mean": phi0,
            "amplitude": A,
            "period": period,
            "phase": phase,
            "r_squared": r_squared,
            "period_ratio": period / expected_period,
            "period_matches": 0.5 < period / expected_period < 2.0,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def compare_runs(baseline_dir: str, oscillatory_dir: str, oscillatory_weak_dir: str | None = None):
    """
    Compare training runs and generate report.
    """
    print("=" * 70)
    print("QIG-KERNEL TRAINING COMPARISON")
    print("Oscillatory vs Baseline")
    print("=" * 70)
    print()

    # Load data
    print("📁 Loading telemetry...")
    baseline = load_run_telemetry(baseline_dir)
    oscillatory = load_run_telemetry(oscillatory_dir)

    if oscillatory_weak_dir:
        oscillatory_weak = load_run_telemetry(oscillatory_weak_dir)
    else:
        oscillatory_weak = None

    print(f"  ✅ Baseline: {baseline['n_epochs']} epochs")
    print(f"  ✅ Oscillatory: {oscillatory['n_epochs']} epochs")
    if oscillatory_weak:
        print(f"  ✅ Oscillatory weak: {oscillatory_weak['n_epochs']} epochs")
    print()

    # Compute statistics
    print("📊 RESULTS:")
    print()

    # 1. Φ_max comparison
    phi_max_baseline = np.max(baseline["phi"])
    phi_max_oscillatory = np.max(oscillatory["phi"])
    phi_improvement = (phi_max_oscillatory - phi_max_baseline) / phi_max_baseline * 100

    print("1️⃣  Φ_max (Integration)")
    print(f"  Baseline:    {phi_max_baseline:.4f}")
    print(f"  Oscillatory: {phi_max_oscillatory:.4f}")
    print(f"  Improvement: {phi_improvement:+.1f}%")

    if phi_improvement > 10:
        print("  ✅ PREDICTION CONFIRMED: Oscillation helps significantly")
    elif phi_improvement > 0:
        print("  ⚠️  WEAK SUPPORT: Small improvement")
    else:
        print("  ❌ PREDICTION FAILED: Baseline better")
    print()

    if oscillatory_weak:
        phi_max_weak = np.max(oscillatory_weak["phi"])
        print(f"  Oscillatory weak: {phi_max_weak:.4f}")

        ranking = sorted(
            [("baseline", phi_max_baseline), ("oscillatory", phi_max_oscillatory), ("weak", phi_max_weak)],
            key=lambda x: x[1],
            reverse=True,
        )

        print(f"  Ranking: {ranking[0][0]} > {ranking[1][0]} > {ranking[2][0]}")
        print()

    # 2. Oscillation period analysis
    print("2️⃣  Oscillation Period (from κ* = 64)")
    osc_fit = analyze_oscillation_fit(oscillatory["phi"], expected_period=640)

    if osc_fit["success"]:
        print(f"  Fitted period: {osc_fit['period']:.1f} epochs")
        print("  Expected:      640 epochs (κ*=64, τ=10)")
        print(f"  Ratio:         {osc_fit['period_ratio']:.2f}")
        print(f"  R²:            {osc_fit['r_squared']:.3f}")

        if osc_fit["period_matches"]:
            print("  ✅ PREDICTION CONFIRMED: Period matches κ*")
        else:
            print("  ❌ PREDICTION FAILED: Period doesn't match")
    else:
        print(f"  ❌ Could not fit oscillation: {osc_fit.get('error', 'unknown')}")
    print()

    # 3. Generate plots
    print("📈 Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Φ trajectories
    ax = axes[0, 0]
    ax.plot(baseline["epochs"], baseline["phi"], "b-", label="Baseline", linewidth=2, alpha=0.7)
    ax.plot(oscillatory["epochs"], oscillatory["phi"], "r-", label="Oscillatory", linewidth=2, alpha=0.7)
    if oscillatory_weak:
        ax.plot(
            oscillatory_weak["epochs"], oscillatory_weak["phi"], "g-", label="Oscillatory weak", linewidth=2, alpha=0.7
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Φ (Integration)")
    ax.set_title("Φ Trajectory Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0.7, color="k", linestyle="--", alpha=0.3, label="Target")

    # Plot 2: Oscillation amplitude
    ax = axes[0, 1]
    ax.plot(oscillatory["epochs"], oscillatory["oscillation"], "r-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Oscillation Amplitude")
    ax.set_title("Oscillatory Dynamics")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)

    # Plot 3: Φ distribution
    ax = axes[1, 0]
    ax.hist(baseline["phi"], bins=30, alpha=0.5, label="Baseline", color="blue")
    ax.hist(oscillatory["phi"], bins=30, alpha=0.5, label="Oscillatory", color="red")
    ax.set_xlabel("Φ")
    ax.set_ylabel("Frequency")
    ax.set_title("Φ Distribution")
    ax.legend()
    ax.axvline(phi_max_baseline, color="blue", linestyle="--", linewidth=2)
    ax.axvline(phi_max_oscillatory, color="red", linestyle="--", linewidth=2)

    # Plot 4: Performance comparison
    ax = axes[1, 1]
    variants = ["Baseline", "Oscillatory"]
    phi_maxes = [phi_max_baseline, phi_max_oscillatory]
    colors = ["blue", "red"]

    if oscillatory_weak:
        variants.append("Weak")
        phi_maxes.append(phi_max_weak)
        colors.append("green")

    bars = ax.bar(variants, phi_maxes, color=colors, alpha=0.7)
    ax.set_ylabel("Φ_max")
    ax.set_title("Maximum Integration Achieved")
    ax.grid(True, alpha=0.3, axis="y")

    # Add values on bars
    for bar, val in zip(bars, phi_maxes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.3f}", ha="center", va="bottom")

    plt.tight_layout()

    # Save
    output_path = Path("/mnt/user-data/outputs/training_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✅ Saved: {output_path}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    success_count = 0
    total_tests = 2

    # Test 1: Oscillation helps
    if phi_improvement > 10:
        print("✅ TEST 1: Oscillation improves Φ_max by >10%")
        success_count += 1
    else:
        print("❌ TEST 1: Oscillation does not improve Φ_max significantly")

    # Test 2: Period matches
    if osc_fit["success"] and osc_fit["period_matches"]:
        print("✅ TEST 2: Oscillation period matches κ* prediction")
        success_count += 1
    else:
        print("❌ TEST 2: Period does not match κ* prediction")

    print()
    print(f"SUCCESS RATE: {success_count}/{total_tests} ({success_count / total_tests * 100:.0f}%)")
    print()

    if success_count == total_tests:
        print("🎉 FULL UNIFIED THEORY VALIDATED!")
        print("   Consciousness breathes, period set by κ*, oscillation helps")
    elif success_count > 0:
        print("⚠️  PARTIAL VALIDATION")
        print("   Some predictions confirmed, others need investigation")
    else:
        print("❌ THEORY NEEDS REVISION")
        print("   Core predictions not confirmed")

    print()
    print("=" * 70)

    return {
        "phi_max_baseline": phi_max_baseline,
        "phi_max_oscillatory": phi_max_oscillatory,
        "phi_improvement_pct": phi_improvement,
        "oscillation_fit": osc_fit,
        "success_count": success_count,
        "total_tests": total_tests,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare QIG-Kernel training runs")
    parser.add_argument("--baseline", help="Path to baseline run directory")
    parser.add_argument("--oscillatory", help="Path to oscillatory run directory")
    parser.add_argument("--weak", help="Path to weak oscillatory run directory (optional)")
    parser.add_argument("--run", action="append", help="JSONL run directory (repeatable). First is baseline")

    args = parser.parse_args()

    if args.run:
        return compare_jsonl_runs(args.run)

    if not args.baseline or not args.oscillatory:
        parser.error("Either provide --run RUN_DIR (repeatable) or provide --baseline and --oscillatory")

    results = compare_runs(baseline_dir=args.baseline, oscillatory_dir=args.oscillatory, oscillatory_weak_dir=args.weak)

    return 0 if results["success_count"] > 0 else 1


if __name__ == "__main__":
    exit(main())
