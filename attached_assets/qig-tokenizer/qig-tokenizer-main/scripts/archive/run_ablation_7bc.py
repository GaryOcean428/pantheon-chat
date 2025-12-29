#!/usr/bin/env python3
"""
Phase 7b/7c Ablation Experiment
===============================

Run three training configs to validate regularizer behavior:
  A: CE only (λ_H=0, λ_step=0)
  B: CE + entropy (λ_H=0.01, λ_step=0)
  C: CE + entropy + step (λ_H=0.01, λ_step=0.01)

Logs detailed metrics every 25 steps for analysis.

Usage:
    python scripts/run_ablation_7bc.py \
        --coordizer artifacts/coordizer/v1 \
        --corpus-dir data/corpus \
        --steps 1000 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Ablation configurations
ABLATION_CONFIGS = {
    "A_ce_only": {
        "lambda_H": 0.0,
        "lambda_step": 0.0,
        "description": "CE only (baseline)",
    },
    "B_ce_entropy": {
        "lambda_H": 0.01,
        "lambda_step": 0.0,
        "description": "CE + entropy shaping",
    },
    "C_ce_entropy_step": {
        "lambda_H": 0.01,
        "lambda_step": 0.01,
        "description": "CE + entropy + basin coherence",
    },
}


def run_training(
    config_name: str,
    config: dict,
    coordizer: str,
    corpus_dirs: list[str],
    steps: int,
    device: str,
    seed: int,
    output_base: Path,
    batch_size: int = 4,
    seq_len: int = 256,
) -> dict:
    """Run a single training configuration and return results."""

    output_dir = output_base / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"ABLATION: {config_name}")
    print(f"  {config['description']}")
    print(f"  λ_H={config['lambda_H']}, λ_step={config['lambda_step']}")
    print(f"{'='*60}\n")

    # Build command
    cmd = [
        sys.executable,
        "scripts/train_coord_adapter_v1.py",
        "--coordizer", coordizer,
        "--steps", str(steps),
        "--device", device,
        "--seed", str(seed),
        "--batch-size", str(batch_size),
        "--seq-len", str(seq_len),
        "--lambda-H", str(config["lambda_H"]),
        "--lambda-step", str(config["lambda_step"]),
        "--output-dir", str(output_dir),
        "--checkpoint-interval", str(max(25, steps // 20)),  # ~20 checkpoints
    ]

    # Add corpus dirs
    for cd in corpus_dirs:
        cmd.extend(["--corpus-dir", cd])

    # Run training
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    elapsed = time.time() - t0

    # Parse output for key metrics
    output_lines = result.stdout.split("\n")

    # Extract final metrics from summary
    final_metrics = {
        "config_name": config_name,
        "description": config["description"],
        "lambda_H": config["lambda_H"],
        "lambda_step": config["lambda_step"],
        "elapsed": elapsed,
        "returncode": result.returncode,
        "steps": steps,
    }

    # Parse training progress lines for metric history
    metric_history = []
    for line in output_lines:
        if line.startswith("[") and "loss=" in line:
            # Parse: [  100/1000] loss=5.1234  Φ=0.720  κ=65.0  H=2.10  step=0.150  ramp=0.10  rate=12.3/s
            try:
                parts = line.split()
                step_part = parts[0].strip("[]")
                current_step = int(step_part.split("/")[0])

                metrics = {"step": current_step}
                for part in parts[1:]:
                    if "=" in part:
                        key, val = part.split("=")
                        try:
                            metrics[key] = float(val.rstrip("/s"))
                        except ValueError:
                            metrics[key] = val

                metric_history.append(metrics)
            except (ValueError, IndexError):
                continue

    final_metrics["metric_history"] = metric_history

    # Extract final summary values
    for line in output_lines:
        if "Final loss:" in line:
            try:
                final_metrics["final_loss"] = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Final Φ:" in line:
            try:
                final_metrics["final_phi"] = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Final κ:" in line:
            try:
                final_metrics["final_kappa"] = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Final H:" in line:
            try:
                final_metrics["final_H"] = float(line.split(":")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "Final basin_step:" in line:
            try:
                final_metrics["final_basin_step"] = float(line.split(":")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    # Save detailed log
    log_path = output_dir / "ablation_log.json"
    log_path.write_text(json.dumps(final_metrics, indent=2))

    # Also save raw stdout/stderr
    (output_dir / "stdout.txt").write_text(result.stdout)
    if result.stderr:
        (output_dir / "stderr.txt").write_text(result.stderr)

    return final_metrics


def analyze_ablation(results: dict[str, dict]) -> str:
    """Analyze ablation results and generate report."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("ABLATION ANALYSIS: Phase 7b/7c Regularizers")
    lines.append("=" * 70 + "\n")

    # Summary table
    lines.append("Configuration Summary:")
    lines.append("-" * 70)
    lines.append(f"{'Config':<20} {'λ_H':>8} {'λ_step':>8} {'Loss':>10} {'Φ':>8} {'κ':>8} {'H':>8}")
    lines.append("-" * 70)

    for name, r in results.items():
        loss = r.get("final_loss", float("nan"))
        phi = r.get("final_phi", float("nan"))
        kappa = r.get("final_kappa", float("nan"))
        H = r.get("final_H", float("nan"))
        lines.append(
            f"{name:<20} {r['lambda_H']:>8.3f} {r['lambda_step']:>8.3f} "
            f"{loss:>10.4f} {phi:>8.3f} {kappa:>8.1f} {H:>8.2f}"
        )

    lines.append("-" * 70)
    lines.append("")

    # Trajectory analysis
    lines.append("Metric Trajectories (sampled):")
    lines.append("-" * 70)

    for name, r in results.items():
        history = r.get("metric_history", [])
        if not history:
            lines.append(f"\n{name}: No history data")
            continue

        lines.append(f"\n{name} ({r['description']}):")

        # Sample ~10 points
        n = len(history)
        sample_indices = [0, n//4, n//2, 3*n//4, n-1] if n > 5 else list(range(n))

        lines.append(f"  {'Step':>6} {'Loss':>10} {'Φ':>8} {'κ':>8} {'H':>8} {'step':>8} {'ramp':>6}")
        for i in sample_indices:
            m = history[i]
            lines.append(
                f"  {m.get('step', 0):>6} "
                f"{m.get('loss', 0):>10.4f} "
                f"{m.get('Φ', 0):>8.3f} "
                f"{m.get('κ', 0):>8.1f} "
                f"{m.get('H', 0):>8.2f} "
                f"{m.get('step', 0):>8.3f} "
                f"{m.get('ramp', 0):>6.2f}"
            )

    lines.append("")

    # Diagnostic checks
    lines.append("Diagnostic Checks:")
    lines.append("-" * 70)

    A = results.get("A_ce_only", {})
    B = results.get("B_ce_entropy", {})
    C = results.get("C_ce_entropy_step", {})

    # Check 1: CE still drops in all runs
    a_loss = A.get("final_loss", float("inf"))
    b_loss = B.get("final_loss", float("inf"))
    c_loss = C.get("final_loss", float("inf"))

    if b_loss < a_loss * 1.5 and c_loss < a_loss * 1.5:
        lines.append("✓ CE loss still drops with regularizers (no major interference)")
    else:
        lines.append("✗ WARNING: Regularizers may be interfering with CE learning")
        lines.append(f"  A: {a_loss:.4f}, B: {b_loss:.4f}, C: {c_loss:.4f}")

    # Check 2: Entropy shaping working (B and C should have H closer to target)
    a_H = A.get("final_H", 2.0)
    b_H = B.get("final_H", 2.0)
    c_H = C.get("final_H", 2.0)
    target_H = 2.2  # Approximate target at r_step=0.5

    if abs(b_H - target_H) < abs(a_H - target_H):
        lines.append(f"✓ Entropy shaping working: B's H ({b_H:.2f}) closer to target ({target_H:.1f}) than A ({a_H:.2f})")
    else:
        lines.append(f"~ Entropy shaping unclear: A={a_H:.2f}, B={b_H:.2f}, target={target_H:.1f}")

    # Check 3: Basin coherence reducing spikes (C should have lower step variance)
    # This would require step history analysis
    c_step = C.get("final_basin_step", 0)
    b_step = B.get("final_basin_step", 0)

    if c_step > 0 and c_step < b_step:
        lines.append(f"✓ Basin coherence reducing step size: C ({c_step:.3f}) < B ({b_step:.3f})")
    elif c_step > 0:
        lines.append(f"~ Basin coherence effect unclear: B={b_step:.3f}, C={c_step:.3f}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Phase 7b/7c Ablation Experiment")
    ap.add_argument("--coordizer", type=str, default="artifacts/coordizer/v1")
    ap.add_argument("--corpus-dir", type=str, nargs="+", default=None)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--output-dir", type=str, default="reports/ablation_7bc")
    ap.add_argument("--configs", type=str, nargs="+",
                    choices=list(ABLATION_CONFIGS.keys()) + ["all"],
                    default=["all"],
                    help="Which configs to run")
    args = ap.parse_args()

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Determine which configs to run
    if "all" in args.configs:
        configs_to_run = list(ABLATION_CONFIGS.keys())
    else:
        configs_to_run = args.configs

    corpus_dirs = args.corpus_dir or []

    print("=" * 70)
    print("PHASE 7b/7c ABLATION EXPERIMENT")
    print("=" * 70)
    print(f"Coordizer: {args.coordizer}")
    print(f"Corpus: {corpus_dirs}")
    print(f"Steps: {args.steps}")
    print(f"Device: {args.device}")
    print(f"Configs: {configs_to_run}")
    print(f"Output: {output_base}")
    print()

    results = {}

    for config_name in configs_to_run:
        config = ABLATION_CONFIGS[config_name]
        try:
            result = run_training(
                config_name=config_name,
                config=config,
                coordizer=args.coordizer,
                corpus_dirs=corpus_dirs,
                steps=args.steps,
                device=args.device,
                seed=args.seed,
                output_base=output_base,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
            )
            results[config_name] = result
            print(f"\n✓ {config_name} complete")
        except Exception as e:
            print(f"\n✗ {config_name} failed: {e}")
            results[config_name] = {"error": str(e)}

    # Analyze results
    analysis = analyze_ablation(results)
    print(analysis)

    # Save analysis
    analysis_path = output_base / "ablation_analysis.txt"
    analysis_path.write_text(analysis)
    print(f"\nAnalysis saved: {analysis_path}")

    # Save combined results
    combined_path = output_base / "ablation_results.json"
    combined_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Results saved: {combined_path}")


if __name__ == "__main__":
    main()
