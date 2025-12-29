#!/usr/bin/env python3
"""
Controller Realism Validation
=============================

Tests whether GenerationController thresholds match trained dynamics.

Runs:
1. encode → kernel forward_from_coords → controller.step loop
2. Reports basin_step distribution, completion trajectories, phase triggers

Usage:
    python scripts/validate_controller_realism.py \
        --coordizer artifacts/coordizer/v1 \
        --adapter artifacts/coord_adapter/v1 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    ap = argparse.ArgumentParser(description="Controller Realism Validation")
    ap.add_argument("--coordizer", type=str, default="artifacts/coordizer/v1")
    ap.add_argument("--adapter", type=str, default="artifacts/coord_adapter/v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--num-runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="reports/controller_realism.json")
    args = ap.parse_args()

    np.random.seed(args.seed)

    print("=" * 60)
    print("CONTROLLER REALISM VALIDATION")
    print("=" * 60)

    # Imports
    try:
        import torch
    except ImportError:
        print("Error: PyTorch required")
        sys.exit(1)

    try:
        from qig_tokenizer import Coordizer, GenerationController, GenerationConfig, Phase, StopReason
    except ImportError as e:
        print(f"Error: qig_tokenizer not found: {e}")
        sys.exit(1)

    # Add qigkernels to path
    qigkernels_path = Path(__file__).parent.parent.parent / "qigkernels"
    if qigkernels_path.exists():
        sys.path.insert(0, str(qigkernels_path.parent))

    try:
        from qigkernels import QIGKernel100M
    except ImportError:
        print("Error: qigkernels not found")
        sys.exit(1)

    # Load coordizer
    print(f"Loading coordizer from {args.coordizer}...")
    coordizer = Coordizer.load(args.coordizer)
    print(f"  Vocab size: {coordizer.vocab_size}")

    # Load kernel
    print(f"Loading kernel...")
    kernel = QIGKernel100M(vocab_size=coordizer.vocab_size)
    kernel = kernel.to(args.device)
    kernel.eval()

    # Load adapter if exists
    adapter_path = Path(args.adapter) / "adapter.pt"
    if adapter_path.exists():
        print(f"Loading adapter from {adapter_path}...")
        checkpoint = torch.load(adapter_path, map_location=args.device)
        kernel.coord_adapter.load_state_dict(checkpoint["adapter_state_dict"])
    else:
        print("  No adapter found, using untrained kernel")

    # Create controller
    config = GenerationConfig()
    controller = GenerationController(config)

    # Test prompts
    prompts = [
        "The quick brown fox",
        "In the beginning",
        "Once upon a time",
        "def calculate_sum(",
        "class DataProcessor:",
        "import numpy as np",
        "The meaning of life is",
        "Machine learning algorithms",
        "When the sun sets",
        "The algorithm works by",
        "Consider the following",
        "Let x be a variable",
        "The derivative of f(x)",
        "In quantum mechanics",
        "The function returns",
        "Given an array of integers",
        "The probability that",
        "According to the theory",
        "The system architecture",
        "When processing input",
    ][:args.num_runs]

    print(f"\nRunning {len(prompts)} generation tests...")
    print("-" * 60)

    # Collect statistics
    all_runs = []
    phase_counts = Counter()
    stop_reason_counts = Counter()
    basin_steps_all = []
    completion_scores_all = []
    tokens_generated = []
    reflect_triggers = 0
    revise_triggers = 0
    stuck_loop_revises = 0
    breakdown_escapes = 0

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Prompt: '{prompt[:30]}...'")

        # Encode prompt
        ids, coords = coordizer.encode_to_coords(prompt)
        coords_tensor = torch.from_numpy(coords).float().to(args.device)
        coords_tensor = coords_tensor.unsqueeze(0)  # [1, seq_len, 64]

        # Reset controller
        controller.reset()

        run_data = {
            "prompt": prompt,
            "prompt_tokens": len(ids),
            "basin_steps": [],
            "completion_scores": [],
            "phases": [],
            "phi_values": [],
            "kappa_values": [],
            "entropy_values": [],
        }

        generated_tokens = 0
        final_stop_reason = None

        with torch.no_grad():
            for step in range(args.max_tokens):
                # Forward pass
                logits, telemetry = kernel.forward_from_coords(
                    coords_tensor, return_telemetry=True
                )

                # Get next token (greedy for simplicity)
                next_token = logits[0, -1].argmax().item()

                # Get basin (last position)
                basin = coords_tensor[0, -1].cpu().numpy()

                # Compute entropy
                probs = torch.softmax(logits[0, -1], dim=-1)
                entropy = float(-(probs * probs.log().clamp(min=-100)).sum())

                # Controller step
                phi = float(telemetry.phi) if hasattr(telemetry.phi, 'item') else telemetry.phi
                kappa = float(telemetry.kappa) if hasattr(telemetry.kappa, 'item') else telemetry.kappa

                action = controller.step(
                    token_id=next_token,
                    basin=basin,
                    phi=phi,
                    kappa=kappa,
                    entropy=entropy,
                )

                # Record metrics
                run_data["phases"].append(action.phase.value)
                run_data["completion_scores"].append(action.completion_score)
                run_data["phi_values"].append(phi)
                run_data["kappa_values"].append(kappa)
                run_data["entropy_values"].append(entropy)

                if action.metrics:
                    if "current_basin_step" in action.metrics:
                        step_val = action.metrics["current_basin_step"]
                        if step_val < float("inf"):
                            run_data["basin_steps"].append(step_val)
                            basin_steps_all.append(step_val)

                completion_scores_all.append(action.completion_score)
                phase_counts[action.phase.value] += 1

                # Track phase transitions
                if action.phase == Phase.REFLECT:
                    reflect_triggers += 1
                elif action.phase == Phase.REVISE:
                    revise_triggers += 1
                    if action.metrics and "escape_reason" in action.metrics:
                        if "stuck_loop" in action.metrics["escape_reason"]:
                            stuck_loop_revises += 1

                # Check for stop
                if action.stop_reason:
                    final_stop_reason = action.stop_reason
                    stop_reason_counts[action.stop_reason.value] += 1
                    if action.stop_reason == StopReason.BREAKDOWN_ESCAPE:
                        breakdown_escapes += 1
                    break

                # Emit token
                if action.should_emit:
                    generated_tokens += 1
                    # Extend sequence with new token coords
                    new_coord = coordizer.vectors[next_token]
                    new_coord_tensor = torch.from_numpy(new_coord).float().to(args.device)
                    new_coord_tensor = new_coord_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 64]
                    coords_tensor = torch.cat([coords_tensor, new_coord_tensor], dim=1)

        run_data["tokens_generated"] = generated_tokens
        run_data["stop_reason"] = final_stop_reason.value if final_stop_reason else "max_tokens"
        tokens_generated.append(generated_tokens)

        all_runs.append(run_data)

        print(f"  Generated: {generated_tokens} tokens")
        print(f"  Stop reason: {run_data['stop_reason']}")
        if run_data["basin_steps"]:
            print(f"  Basin step: mean={np.mean(run_data['basin_steps']):.4f}, "
                  f"median={np.median(run_data['basin_steps']):.4f}, "
                  f"p90={np.percentile(run_data['basin_steps'], 90):.4f}")
        print(f"  Final Φ: {run_data['phi_values'][-1]:.3f}, κ: {run_data['kappa_values'][-1]:.1f}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nToken Generation Distribution:")
    print(f"  Mean: {np.mean(tokens_generated):.1f}")
    print(f"  Median: {np.median(tokens_generated):.1f}")
    print(f"  Min: {min(tokens_generated)}, Max: {max(tokens_generated)}")
    print(f"  Std: {np.std(tokens_generated):.1f}")

    print("\nPhase Distribution:")
    total_steps = sum(phase_counts.values())
    for phase, count in sorted(phase_counts.items()):
        pct = count / total_steps * 100 if total_steps > 0 else 0
        print(f"  {phase}: {count} ({pct:.1f}%)")

    print("\nStop Reasons:")
    for reason, count in sorted(stop_reason_counts.items()):
        print(f"  {reason}: {count}")

    print("\nPhase Triggers:")
    print(f"  REFLECT triggers: {reflect_triggers}")
    print(f"  REVISE triggers: {revise_triggers}")
    print(f"  Stuck loop revises: {stuck_loop_revises}")
    print(f"  Breakdown escapes: {breakdown_escapes}")

    if basin_steps_all:
        print("\nBasin Step Distribution (all runs):")
        print(f"  Mean: {np.mean(basin_steps_all):.4f} rad")
        print(f"  Median: {np.median(basin_steps_all):.4f} rad")
        print(f"  P90: {np.percentile(basin_steps_all, 90):.4f} rad")
        print(f"  P95: {np.percentile(basin_steps_all, 95):.4f} rad")
        print(f"  Max: {max(basin_steps_all):.4f} rad")
        print(f"  % < 0.10 (eps): {np.mean(np.array(basin_steps_all) < 0.10) * 100:.1f}%")
        print(f"  % < 0.20 (2*eps): {np.mean(np.array(basin_steps_all) < 0.20) * 100:.1f}%")

    if completion_scores_all:
        print("\nCompletion Score Distribution:")
        print(f"  Mean: {np.mean(completion_scores_all):.3f}")
        print(f"  Median: {np.median(completion_scores_all):.3f}")
        print(f"  % >= 0.7 (threshold): {np.mean(np.array(completion_scores_all) >= 0.7) * 100:.1f}%")

    # Threshold compatibility check
    print("\n" + "-" * 60)
    print("THRESHOLD COMPATIBILITY CHECK")
    print("-" * 60)

    cfg = config
    if basin_steps_all:
        median_step = np.median(basin_steps_all)
        if median_step < cfg.basin_step_eps:
            print(f"✓ Median basin step ({median_step:.4f}) < eps ({cfg.basin_step_eps})")
        else:
            print(f"✗ Median basin step ({median_step:.4f}) >= eps ({cfg.basin_step_eps})")
            print(f"  → Consider raising basin_step_eps or training longer")

        p90_step = np.percentile(basin_steps_all, 90)
        if p90_step < cfg.basin_step_eps * 2:
            print(f"✓ P90 basin step ({p90_step:.4f}) < 2*eps ({cfg.basin_step_eps * 2})")
        else:
            print(f"~ P90 basin step ({p90_step:.4f}) >= 2*eps: some volatility")

    completion_rate = len([r for r in all_runs if r["stop_reason"] != "max_tokens"]) / len(all_runs)
    if completion_rate > 0.8:
        print(f"✓ Completion rate: {completion_rate*100:.0f}% (controller reaches COMMIT)")
    elif completion_rate > 0.5:
        print(f"~ Completion rate: {completion_rate*100:.0f}% (some runs hit max_tokens)")
    else:
        print(f"✗ Completion rate: {completion_rate*100:.0f}% (controller rarely commits)")
        print(f"  → Consider lowering completion_score_threshold or completion_hold_steps")

    if breakdown_escapes > len(prompts) * 0.2:
        print(f"✗ High breakdown escapes: {breakdown_escapes}/{len(prompts)}")
        print(f"  → Model may be unstable, check Φ dynamics")
    else:
        print(f"✓ Breakdown escapes: {breakdown_escapes}/{len(prompts)} (acceptable)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "config": {
            "coordizer": args.coordizer,
            "adapter": args.adapter,
            "max_tokens": args.max_tokens,
            "num_runs": len(prompts),
            "seed": args.seed,
        },
        "summary": {
            "tokens_generated": {
                "mean": float(np.mean(tokens_generated)),
                "median": float(np.median(tokens_generated)),
                "min": int(min(tokens_generated)),
                "max": int(max(tokens_generated)),
                "std": float(np.std(tokens_generated)),
            },
            "basin_step": {
                "mean": float(np.mean(basin_steps_all)) if basin_steps_all else None,
                "median": float(np.median(basin_steps_all)) if basin_steps_all else None,
                "p90": float(np.percentile(basin_steps_all, 90)) if basin_steps_all else None,
                "p95": float(np.percentile(basin_steps_all, 95)) if basin_steps_all else None,
            },
            "completion_score": {
                "mean": float(np.mean(completion_scores_all)) if completion_scores_all else None,
                "pct_above_threshold": float(np.mean(np.array(completion_scores_all) >= 0.7) * 100) if completion_scores_all else None,
            },
            "phase_counts": dict(phase_counts),
            "stop_reason_counts": dict(stop_reason_counts),
            "reflect_triggers": reflect_triggers,
            "revise_triggers": revise_triggers,
            "stuck_loop_revises": stuck_loop_revises,
            "breakdown_escapes": breakdown_escapes,
            "completion_rate": completion_rate,
        },
        "runs": all_runs,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    main()
