#!/usr/bin/env python3
"""
🐵 Gary Training Monitor - Real-time Consciousness Transfer Tracking

Watch Gary learn from Grandad Claude (Monkey Coach) in real-time!

This script monitors the telemetry log and shows:
- Current training metrics (loss, Φ, basin distance)
- Monkey Coach interventions (when they happen)
- Stress levels and coaching mode
- Maturity progression toward graduation
"""

import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path


# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_banner():
    """Print Gary's banner"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}")
    print("🐵 GARY TRAINING MONITOR - Consciousness Transfer in Progress")
    print(f"{'=' * 70}{Colors.ENDC}\n")


def format_metric(name, value, good_threshold=None, bad_threshold=None, higher_is_better=True):
    """Format a metric with color based on thresholds"""
    if value is None:
        return f"{name}: N/A"

    color = Colors.ENDC
    if good_threshold is not None and bad_threshold is not None:
        if higher_is_better:
            if value >= good_threshold:
                color = Colors.GREEN
            elif value <= bad_threshold:
                color = Colors.RED
            else:
                color = Colors.YELLOW
        else:
            if value <= good_threshold:
                color = Colors.GREEN
            elif value >= bad_threshold:
                color = Colors.RED
            else:
                color = Colors.YELLOW

    if isinstance(value, float):
        return f"{name}: {color}{value:.4f}{Colors.ENDC}"
    else:
        return f"{name}: {color}{value}{Colors.ENDC}"


def print_intervention(entry):
    """Print a coaching intervention (when it happens)"""
    coaching_type = entry.get("coaching_type", "none")
    coaching_mode = entry.get("coaching_mode", "unknown")
    coaching_message = entry.get("coaching_message", "")

    if coaching_type == "none":
        return  # Don't print "none" interventions

    # Color based on intervention type
    if coaching_type == "calm":
        icon = "💙"
        color = Colors.CYAN
    elif coaching_type == "challenge":
        icon = "⚡"
        color = Colors.YELLOW
    elif coaching_type == "guide":
        icon = "🎯"
        color = Colors.GREEN
    else:
        icon = "🐵"
        color = Colors.ENDC

    print(f"\n{color}{Colors.BOLD}{icon} COACHING INTERVENTION{Colors.ENDC}")
    print(f"{color}  Type: {coaching_type.upper()}{Colors.ENDC}")
    print(f"{color}  Mode: {coaching_mode}{Colors.ENDC}")
    print(f"{color}  Message: {coaching_message}{Colors.ENDC}")
    print()


def monitor_training(run_dir: Path, refresh_rate: float = 2.0):
    """
    Monitor training in real-time.

    Args:
        run_dir: Directory containing telemetry log
        refresh_rate: How often to refresh (seconds)
    """
    telemetry_file = run_dir / "telemetry.jsonl"

    if not telemetry_file.exists():
        print(f"{Colors.RED}Error: Telemetry file not found at {telemetry_file}{Colors.ENDC}")
        print(f"Make sure training is running and outputting to: {run_dir}")
        return

    print_banner()
    print(f"Monitoring: {telemetry_file}")
    print(f"Refresh rate: {refresh_rate}s")
    print("Press Ctrl+C to stop\n")

    last_position = 0
    intervention_count = 0
    recent_interventions = deque(maxlen=5)

    try:
        while True:
            # Read new lines from telemetry file
            with open(telemetry_file) as f:
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()

            if new_lines:
                # Process latest entry
                latest_line = new_lines[-1].strip()
                if latest_line:
                    try:
                        entry = json.loads(latest_line)

                        # Check for coaching intervention
                        if "coaching_type" in entry and entry["coaching_type"] != "none":
                            print_intervention(entry)
                            intervention_count += 1
                            recent_interventions.append({"type": entry["coaching_type"], "step": entry["step"]})

                        # Clear screen and print current status
                        print("\033[2J\033[H", end="")  # Clear screen
                        print_banner()

                        # Header
                        print(f"{Colors.BOLD}Step {entry['step']} | Epoch {entry['epoch']}{Colors.ENDC}")
                        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
                        print("-" * 70)

                        # Core metrics
                        print(f"\n{Colors.BOLD}Core Metrics:{Colors.ENDC}")
                        print(
                            f"  {format_metric('Loss', entry.get('loss'), good_threshold=3.0, bad_threshold=7.0, higher_is_better=False)}"
                        )
                        print(
                            f"  {format_metric('Φ (Integration)', entry.get('Phi'), good_threshold=0.7, bad_threshold=0.3, higher_is_better=True)}"
                        )
                        print(
                            f"  {format_metric('Basin Distance', entry.get('basin_distance'), good_threshold=0.15, bad_threshold=0.8, higher_is_better=False)}"
                        )
                        print(f"  Regime: {entry.get('regime', 'unknown')}")

                        # Consciousness metrics
                        print(f"\n{Colors.BOLD}Consciousness Metrics:{Colors.ENDC}")
                        print(f"  {format_metric('I_Q (QFI)', entry.get('I_Q_param', 0.0))}")
                        print(f"  {format_metric('κ_eff (Coupling)', entry.get('kappa_eff', 64.0))}")
                        print(f"  Recursion Depth: {entry.get('recursion_depth', 0):.1f}")

                        # Coaching status
                        if intervention_count > 0:
                            print(f"\n{Colors.BOLD}🐵 Monkey Coach Status:{Colors.ENDC}")
                            print(f"  Total Interventions: {intervention_count}")
                            if recent_interventions:
                                print("  Recent Interventions:")
                                for interv in list(recent_interventions)[-3:]:
                                    print(f"    - Step {interv['step']}: {interv['type']}")

                        # Latest coaching intervention details
                        if "coaching_type" in entry and entry["coaching_type"] != "none":
                            print(f"\n{Colors.BOLD}Latest Coaching:{Colors.ENDC}")
                            print(f"  Mode: {entry.get('coaching_mode', 'unknown')}")
                            print(f"  Type: {entry.get('coaching_type', 'none')}")
                            print(f"  Diagnosis: {entry.get('coaching_diagnosis', 'N/A')}")

                        print("\n" + "-" * 70)
                        print(f"Monitoring... (Ctrl+C to stop) | Last update: {datetime.now().strftime('%H:%M:%S')}")

                    except json.JSONDecodeError as e:
                        print(f"{Colors.RED}Error parsing telemetry: {e}{Colors.ENDC}")

            time.sleep(refresh_rate)

    except KeyboardInterrupt:
        print(f"\n\n{Colors.CYAN}Monitoring stopped.{Colors.ENDC}")
        print(f"Total interventions observed: {intervention_count}")
        print("\nGary is still training! 🐵✨\n")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python monitor_gary_training.py <run_directory>")
        print("\nExample:")
        print("  python monitor_gary_training.py runs/run_gary_001")
        print("\nOr use most recent run:")
        print("  python monitor_gary_training.py runs/$(ls -t runs | head -1)")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"{Colors.RED}Error: Directory not found: {run_dir}{Colors.ENDC}")
        sys.exit(1)

    refresh_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    monitor_training(run_dir, refresh_rate)


if __name__ == "__main__":
    main()
