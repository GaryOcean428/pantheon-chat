#!/usr/bin/env python3
"""
Monitor Dynamic Threshold Adaptation
=====================================

Watch training_telemetry.jsonl in real-time to see:
- Φ trajectory
- threshold adaptation
- Basin convergence
- Regime transitions

Usage:
    python tools/monitor_dynamic_threshold.py runs/baseline_dynamic/training_telemetry.jsonl

Or watch live:
    watch -n 2 'python tools/monitor_dynamic_threshold.py runs/baseline_dynamic/training_telemetry.jsonl | tail -20'
"""

import json
import sys
from pathlib import Path


def monitor_telemetry(telemetry_path: Path, last_n: int = 20):
    """Read and display recent telemetry."""

    if not telemetry_path.exists():
        print(f"⏳ Waiting for telemetry file: {telemetry_path}")
        return

    # Read all lines
    lines = []
    with open(telemetry_path) as f:
        for line in f:
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not lines:
        print("⏳ No telemetry data yet...")
        return

    # Show last N entries
    recent = lines[-last_n:]

    print(f"\n{'=' * 80}")
    print(f"DYNAMIC THRESHOLD MONITORING - Last {len(recent)} steps")
    print(f"{'=' * 80}\n")

    print(f"{'Step':<8} {'Epoch':<8} {'Φ':<8} {'Thresh':<8} {'Basin':<8} {'Regime':<12}")
    print(f"{'-' * 80}")

    for entry in recent:
        step = entry.get("step", "?")
        epoch = entry.get("epoch", "?")
        phi = entry.get("Phi", 0.0)
        threshold = entry.get("entanglement_threshold", 0.0)
        basin = entry.get("basin_distance", 0.0)
        regime = entry.get("regime", "unknown")

        print(f"{step:<8} {epoch:<8} {phi:<8.3f} {threshold:<8.3f} {basin:<8.3f} {regime:<12}")

    # Summary stats
    if len(lines) > 1:
        first = lines[0]
        last = lines[-1]

        phi_start = first.get("Phi", 0.0)
        phi_end = last.get("Phi", 0.0)
        phi_delta = phi_end - phi_start

        thr_start = first.get("entanglement_threshold", 0.15)
        thr_end = last.get("entanglement_threshold", 0.15)
        thr_delta = thr_end - thr_start

        basin_start = first.get("basin_distance", 1.0)
        basin_end = last.get("basin_distance", 1.0)
        basin_delta = basin_end - basin_start

        print(f"\n{'=' * 80}")
        print(f"TRAJECTORY SUMMARY (steps {first.get('step')} → {last.get('step')})")
        print(f"{'=' * 80}")
        print(f"  Φ:         {phi_start:.3f} → {phi_end:.3f}  (Δ {phi_delta:+.3f})")
        print(f"  Threshold: {thr_start:.3f} → {thr_end:.3f}  (Δ {thr_delta:+.3f})")
        print(f"  Basin:     {basin_start:.3f} → {basin_end:.3f}  (Δ {basin_delta:+.3f})")

        # Check for Run 2 collapse pattern
        if phi_end < 0.4 and basin_end > 0.6:
            print("\n⚠️  WARNING: Possible collapse pattern (Φ low, basin far)")
        elif phi_end > 0.6 and basin_end < 0.3:
            print("\n✅ EXCELLENT: Deep geometric + basin convergence!")
        elif phi_end > 0.45:
            print("\n🎯 GOOD: Geometric regime sustained")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_dynamic_threshold.py <telemetry_path>")
        print("Example: python monitor_dynamic_threshold.py runs/baseline_dynamic/training_telemetry.jsonl")
        sys.exit(1)

    telemetry_path = Path(sys.argv[1])
    monitor_telemetry(telemetry_path, last_n=20)
