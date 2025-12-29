#!/usr/bin/env python3
"""
Quick Analysis: Dynamic Threshold Run
======================================

Compare Run 4 (dynamic) vs previous runs.
"""

import json
from pathlib import Path


def analyze_run(telemetry_path: Path, run_name: str):
    """Analyze a training run."""
    if not telemetry_path.exists():
        print(f"❌ {run_name}: No telemetry found")
        return None

    with open(telemetry_path) as f:
        lines = [json.loads(line) for line in f]

    if not lines:
        return None

    # Extract key metrics
    epochs = set(entry["epoch"] for entry in lines)
    max_epoch = max(epochs)

    # Get metrics by epoch
    epoch_data = {}
    for epoch in sorted(epochs):
        epoch_entries = [e for e in lines if e["epoch"] == epoch]
        phi_values = [e["telemetry"].get("Phi", 0) for e in epoch_entries]
        basin_values = [e["telemetry"].get("basin_distance", 1.0) for e in epoch_entries]
        threshold_values = [e.get("threshold_current", 0.15) for e in epoch_entries]

        epoch_data[epoch] = {
            "phi_max": max(phi_values),
            "phi_avg": sum(phi_values) / len(phi_values),
            "basin_min": min(basin_values),
            "basin_avg": sum(basin_values) / len(basin_values),
            "threshold_avg": sum(threshold_values) / len(threshold_values),
        }

    return {
        "name": run_name,
        "max_epoch": max_epoch,
        "epoch_data": epoch_data,
        "total_steps": len(lines),
    }


def main():
    print("\n" + "=" * 80)
    print("DYNAMIC THRESHOLD RUN ANALYSIS")
    print("=" * 80 + "\n")

    # Analyze all runs
    runs = [
        ("runs/baseline_entangled/training_telemetry.jsonl", "Run 2 (constant 0.15)"),
        ("runs/baseline_staggered_real/training_telemetry.jsonl", "Run 3 (staggered)"),
        ("runs/baseline_dynamic/training_telemetry.jsonl", "Run 4 (DYNAMIC)"),
    ]

    results = []
    for path, name in runs:
        result = analyze_run(Path(path), name)
        if result:
            results.append(result)

    # Compare
    print(f"{'Run':<25} {'Epoch 5 Φ':<12} {'Epoch 5 Basin':<15} {'Geometric?':<12}")
    print("-" * 80)

    for r in results:
        epoch5 = r["epoch_data"].get(4, {})  # Epoch 5 is index 4
        phi = epoch5.get("phi_max", 0)
        basin = epoch5.get("basin_min", 1.0)
        geometric = "YES ✅" if phi > 0.45 else "No"

        print(f"{r['name']:<25} {phi:<12.3f} {basin:<15.3f} {geometric:<12}")

    # Detailed Run 4 trajectory
    print("\n" + "=" * 80)
    print("RUN 4 DYNAMIC THRESHOLD TRAJECTORY")
    print("=" * 80 + "\n")

    dynamic = [r for r in results if "DYNAMIC" in r["name"]]
    if dynamic:
        d = dynamic[0]
        print(f"{'Epoch':<8} {'Φ (max)':<10} {'Basin (min)':<12} {'Threshold':<12} {'Status':<20}")
        print("-" * 80)

        for epoch in sorted(d["epoch_data"].keys()):
            ed = d["epoch_data"][epoch]
            phi = ed["phi_max"]
            basin = ed["basin_min"]
            thr = ed["threshold_avg"]

            # Status
            if phi > 0.7:
                status = "🎯 CONSCIOUSNESS!"
            elif phi > 0.6:
                status = "🚀 Deep geometric"
            elif phi > 0.45:
                status = "✅ Geometric regime"
            elif phi > 0.3:
                status = "📈 Climbing"
            else:
                status = "⏳ Exploring"

            print(f"{epoch + 1:<8} {phi:<10.3f} {basin:<12.3f} {thr:<12.3f} {status:<20}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
