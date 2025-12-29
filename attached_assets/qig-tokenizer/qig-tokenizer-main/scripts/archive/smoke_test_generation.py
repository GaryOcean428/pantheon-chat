#!/usr/bin/env python3
"""
Smoke Test Generation
=====================

Run deterministic generation passes with fixed knobs to validate
controller behavior before tuning.

Reports:
- Completion reason
- Steps to commit
- Final Φ/κ band
- Entropy band over last window
- NN distance to nearest token basin

Usage:
    python scripts/smoke_test_generation.py \
        --coordizer artifacts/coordizer/v1 \
        --adapter artifacts/coord_adapter/v1 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def fisher_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Fisher-Rao (angular) distance."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    cos_angle = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def nearest_basin_distance(query_basin: np.ndarray, token_basins: np.ndarray) -> tuple[float, int]:
    """
    Find distance to nearest token basin.

    Returns:
        (distance, token_id) of nearest basin
    """
    query_norm = query_basin / (np.linalg.norm(query_basin) + 1e-10)
    basins_norm = token_basins / (np.linalg.norm(token_basins, axis=1, keepdims=True) + 1e-10)

    similarities = basins_norm @ query_norm
    nearest_idx = int(np.argmax(similarities))
    nearest_sim = similarities[nearest_idx]
    distance = float(np.arccos(np.clip(nearest_sim, -1.0, 1.0)))

    return distance, nearest_idx


def main():
    ap = argparse.ArgumentParser(description="Smoke Test Generation")
    ap.add_argument("--coordizer", type=str, default="artifacts/coordizer/v1")
    ap.add_argument("--adapter", type=str, default="artifacts/coord_adapter/v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--output", type=str, default="reports/smoke_test_generation.json")
    ap.add_argument("--prompts-file", type=str, default=None,
                    help="JSON file with prompt list")
    args = ap.parse_args()

    np.random.seed(args.seed)

    print("=" * 60)
    print("SMOKE TEST GENERATION")
    print("=" * 60)

    # Imports
    try:
        import torch
        import torch.nn.functional as F
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
    token_basins = coordizer.vectors.copy()
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

    # Create controller with geometry temp modulation enabled
    config = GenerationConfig()
    controller = GenerationController(config)

    # Test prompts
    if args.prompts_file and Path(args.prompts_file).exists():
        prompts = json.loads(Path(args.prompts_file).read_text())
    else:
        prompts = [
            "The fundamental theorem of calculus states that",
            "In machine learning, gradient descent",
            "def fibonacci(n):",
            "class NeuralNetwork(nn.Module):",
            "The quantum mechanical wave function",
            "When implementing a hash table",
            "According to Einstein's theory of relativity",
            "The time complexity of quicksort is",
            "In natural language processing",
            "The derivative of the loss function",
        ]

    print(f"\nRunning {len(prompts)} generation tests...")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print("-" * 60)

    all_results = []
    torch.manual_seed(args.seed)

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Prompt: '{prompt[:40]}...'")

        # Encode prompt
        ids, coords = coordizer.encode_to_coords(prompt)
        coords_tensor = torch.from_numpy(coords).float().to(args.device)
        coords_tensor = coords_tensor.unsqueeze(0)  # [1, seq_len, 64]

        # Reset controller
        controller.reset()

        result = {
            "prompt": prompt,
            "prompt_tokens": len(ids),
            "generated_text": "",
            "generated_tokens": [],
            "steps_to_commit": 0,
            "stop_reason": None,
            "final_phi": None,
            "final_kappa": None,
            "phi_band": {"min": float("inf"), "max": float("-inf"), "mean": 0},
            "kappa_band": {"min": float("inf"), "max": float("-inf"), "mean": 0},
            "entropy_last_window": [],
            "nn_distances": [],
            "trajectory_basins": [],
            "gibberish_bursts": 0,
            "premature_commits": 0,
        }

        phi_values = []
        kappa_values = []
        generated_ids = list(ids)

        with torch.no_grad():
            for step in range(args.max_tokens):
                # Forward pass
                logits, telemetry = kernel.forward_from_coords(
                    coords_tensor, return_telemetry=True
                )

                # Apply temperature
                scaled_logits = logits[0, -1] / args.temperature

                # Sample token
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                # Compute entropy
                log_probs = F.log_softmax(logits[0, -1], dim=-1)
                entropy = float(-(probs * log_probs).sum())

                # Get current basin
                basin = coords_tensor[0, -1].cpu().numpy()

                # NN distance check
                nn_dist, nn_token = nearest_basin_distance(basin, token_basins)
                result["nn_distances"].append(nn_dist)

                # Gibberish detection: high NN distance
                if nn_dist > 0.5:
                    result["gibberish_bursts"] += 1

                # Controller step
                phi = float(telemetry.phi) if hasattr(telemetry.phi, 'item') else telemetry.phi
                kappa = float(telemetry.kappa) if hasattr(telemetry.kappa, 'item') else telemetry.kappa

                phi_values.append(phi)
                kappa_values.append(kappa)

                action = controller.step(
                    token_id=next_token,
                    basin=basin,
                    phi=phi,
                    kappa=kappa,
                    entropy=entropy,
                )

                # Track last window entropy
                if len(result["entropy_last_window"]) < config.window_size:
                    result["entropy_last_window"].append(entropy)
                else:
                    result["entropy_last_window"].pop(0)
                    result["entropy_last_window"].append(entropy)

                # Check for stop
                if action.stop_reason:
                    result["stop_reason"] = action.stop_reason.value
                    result["steps_to_commit"] = step + 1

                    # Check for premature commit
                    if step < config.min_emit_tokens:
                        result["premature_commits"] += 1

                    break

                # Emit token
                if action.should_emit:
                    generated_ids.append(next_token)
                    result["generated_tokens"].append(next_token)

                    # Extend sequence
                    new_coord = coordizer.vectors[next_token]
                    new_coord_tensor = torch.from_numpy(new_coord).float().to(args.device)
                    new_coord_tensor = new_coord_tensor.unsqueeze(0).unsqueeze(0)
                    coords_tensor = torch.cat([coords_tensor, new_coord_tensor], dim=1)

                    # Track trajectory basin
                    result["trajectory_basins"].append(new_coord.tolist())

        # Decode generated text
        try:
            result["generated_text"] = coordizer.decode(generated_ids)
        except Exception:
            result["generated_text"] = f"[{len(result['generated_tokens'])} tokens]"

        # Compute final metrics
        if phi_values:
            result["final_phi"] = phi_values[-1]
            result["phi_band"] = {
                "min": min(phi_values),
                "max": max(phi_values),
                "mean": float(np.mean(phi_values)),
            }

        if kappa_values:
            result["final_kappa"] = kappa_values[-1]
            result["kappa_band"] = {
                "min": min(kappa_values),
                "max": max(kappa_values),
                "mean": float(np.mean(kappa_values)),
            }

        if result["stop_reason"] is None:
            result["stop_reason"] = "max_tokens"
            result["steps_to_commit"] = args.max_tokens

        # Don't store full trajectory basins in output (too large)
        result["trajectory_basins"] = len(result["trajectory_basins"])

        all_results.append(result)

        # Print summary
        print(f"  Generated: {len(result['generated_tokens'])} tokens")
        print(f"  Stop: {result['stop_reason']} at step {result['steps_to_commit']}")
        print(f"  Φ band: [{result['phi_band']['min']:.3f}, {result['phi_band']['max']:.3f}] (μ={result['phi_band']['mean']:.3f})")
        print(f"  κ band: [{result['kappa_band']['min']:.1f}, {result['kappa_band']['max']:.1f}] (μ={result['kappa_band']['mean']:.1f})")
        if result["nn_distances"]:
            print(f"  NN dist: median={np.median(result['nn_distances']):.3f}, p90={np.percentile(result['nn_distances'], 90):.3f}")
        if result["gibberish_bursts"] > 0:
            print(f"  ⚠ Gibberish bursts: {result['gibberish_bursts']}")
        print(f"  Output: '{result['generated_text'][:60]}...'")

    # Summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)

    # Aggregate metrics
    total_generated = sum(len(r["generated_tokens"]) for r in all_results)
    total_gibberish = sum(r["gibberish_bursts"] for r in all_results)
    total_premature = sum(r["premature_commits"] for r in all_results)

    stop_reasons = [r["stop_reason"] for r in all_results]
    steps_to_commit = [r["steps_to_commit"] for r in all_results]

    all_nn_dists = []
    for r in all_results:
        all_nn_dists.extend(r["nn_distances"])

    print(f"\nTokens generated: {total_generated}")
    print(f"Mean steps to commit: {np.mean(steps_to_commit):.1f}")

    print(f"\nStop reasons:")
    from collections import Counter
    for reason, count in Counter(stop_reasons).items():
        print(f"  {reason}: {count}")

    print(f"\nGibberish bursts (NN dist > 0.5): {total_gibberish}")
    print(f"Premature commits (< min_emit): {total_premature}")

    if all_nn_dists:
        print(f"\nNN Distance (all tokens):")
        print(f"  Median: {np.median(all_nn_dists):.4f} rad")
        print(f"  P90: {np.percentile(all_nn_dists, 90):.4f} rad")
        print(f"  % < 0.2: {np.mean(np.array(all_nn_dists) < 0.2) * 100:.1f}%")
        print(f"  % < 0.5: {np.mean(np.array(all_nn_dists) < 0.5) * 100:.1f}%")

    # Quality assessment
    print("\n" + "-" * 60)
    print("QUALITY ASSESSMENT")
    print("-" * 60)

    gibberish_rate = total_gibberish / max(total_generated, 1)
    if gibberish_rate < 0.05:
        print(f"✓ Low gibberish rate: {gibberish_rate*100:.1f}%")
    elif gibberish_rate < 0.15:
        print(f"~ Moderate gibberish rate: {gibberish_rate*100:.1f}%")
    else:
        print(f"✗ High gibberish rate: {gibberish_rate*100:.1f}%")

    commit_rate = len([r for r in all_results if r["stop_reason"] != "max_tokens"]) / len(all_results)
    if commit_rate > 0.8:
        print(f"✓ Good commit rate: {commit_rate*100:.0f}%")
    else:
        print(f"~ Commit rate: {commit_rate*100:.0f}%")

    if total_premature == 0:
        print("✓ No premature commits")
    else:
        print(f"✗ Premature commits: {total_premature}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "config": {
            "coordizer": args.coordizer,
            "adapter": args.adapter,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "summary": {
            "num_prompts": len(prompts),
            "total_generated": total_generated,
            "mean_steps_to_commit": float(np.mean(steps_to_commit)),
            "stop_reasons": dict(Counter(stop_reasons)),
            "gibberish_bursts": total_gibberish,
            "premature_commits": total_premature,
            "nn_distance": {
                "median": float(np.median(all_nn_dists)) if all_nn_dists else None,
                "p90": float(np.percentile(all_nn_dists, 90)) if all_nn_dists else None,
                "pct_under_0.2": float(np.mean(np.array(all_nn_dists) < 0.2) * 100) if all_nn_dists else None,
            },
            "quality": {
                "gibberish_rate": gibberish_rate,
                "commit_rate": commit_rate,
            },
        },
        "results": all_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    main()
