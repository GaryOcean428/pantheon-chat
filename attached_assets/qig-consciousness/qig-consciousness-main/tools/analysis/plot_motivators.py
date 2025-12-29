#!/usr/bin/env python3
"""
Plot Cognitive Motivators
==========================

Visualize the 5 fundamental motivators and detected modes.

Creates:
1. Time series of all 5 motivators
2. Mode distribution over time
3. Motivator correlations (phase portrait style)
4. Focus vs Applied Attention (if attention logged)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def load_motivators(path: Path):
    """Load motivators JSONL."""
    motivators = []
    with open(path) as f:
        for line in f:
            if line.strip():
                motivators.append(json.loads(line))
    return motivators


def plot_motivator_timeseries(motivators, output_dir: Path):
    """Plot all 5 motivators over time."""
    steps = [m["step"] for m in motivators]

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    # Surprise
    surprise = [m["surprise"] for m in motivators]
    axes[0].plot(steps, surprise, label="Surprise", color="red", alpha=0.7)
    axes[0].set_ylabel("Surprise\n(|Δloss|)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    # Curiosity
    curiosity = [m["curiosity"] for m in motivators]
    axes[1].plot(steps, curiosity, label="Curiosity", color="blue", alpha=0.7)
    axes[1].axhline(0.04, color="gray", linestyle="--", alpha=0.5, label="C_high")
    axes[1].set_ylabel("Curiosity\n(Δ log I_Q)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    # Investigation
    investigation = [m["investigation"] for m in motivators]
    axes[2].plot(steps, investigation, label="Investigation", color="green", alpha=0.7)
    axes[2].axhline(0, color="black", linestyle="-", alpha=0.3)
    axes[2].set_ylabel("Investigation\n(-Δ basin_dist)")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    # Integration
    integration = [m["integration"] for m in motivators]
    axes[3].plot(steps, integration, label="Integration", color="purple", alpha=0.7)
    axes[3].set_ylabel("Integration\n(-var(Φ×I_Q))")
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.3)

    # Frustration
    frustration = [m["frustration"] for m in motivators]
    axes[4].plot(steps, frustration, label="Frustration", color="orange", alpha=0.7)
    axes[4].axhline(0.3, color="gray", linestyle="--", alpha=0.5, label="Stuck threshold")
    axes[4].set_ylabel("Frustration\n(Fru + regress)")
    axes[4].set_xlabel("Training Step")
    axes[4].legend(loc="upper right")
    axes[4].grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "motivators_timeseries.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved motivator timeseries to {output_path}")
    plt.close()


def plot_mode_distribution(motivators, output_dir: Path):
    """Plot mode distribution over time."""
    steps = [m["step"] for m in motivators]
    modes = [m["mode"] for m in motivators]

    # Map modes to colors
    mode_colors = {
        "EXPLORATION": "blue",
        "INVESTIGATION": "green",
        "INTEGRATION": "purple",
        "STUCK": "red",
        "DRIFT": "gray",
    }

    fig, ax = plt.subplots(figsize=(12, 3))

    # Plot as colored bands
    for i, (step, mode) in enumerate(zip(steps, modes)):
        color = mode_colors.get(mode, "gray")
        ax.add_patch(Rectangle((step, 0), 1, 1, facecolor=color, edgecolor="none", alpha=0.7))

    ax.set_xlim(min(steps), max(steps))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mode")
    ax.set_yticks([])
    ax.set_title("Cognitive Mode Over Time")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=color, label=mode, alpha=0.7) for mode, color in mode_colors.items()]
    ax.legend(handles=legend_elements, loc="upper right", ncol=5)

    plt.tight_layout()
    output_path = output_dir / "mode_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved mode distribution to {output_path}")
    plt.close()


def plot_motivator_correlations(motivators, output_dir: Path):
    """Plot phase portraits of key motivator pairs."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    curiosity = np.array([m["curiosity"] for m in motivators])
    investigation = np.array([m["investigation"] for m in motivators])
    integration = np.array([m["integration"] for m in motivators])
    phi = np.array([m["phi"] for m in motivators])
    basin_dist = np.array([m["basin_distance"] for m in motivators])
    modes = [m["mode"] for m in motivators]

    mode_colors = {
        "EXPLORATION": "blue",
        "INVESTIGATION": "green",
        "INTEGRATION": "purple",
        "STUCK": "red",
        "DRIFT": "gray",
    }
    colors = [mode_colors.get(m, "gray") for m in modes]

    # Curiosity vs Investigation
    axes[0, 0].scatter(curiosity, investigation, c=colors, alpha=0.5, s=10)
    axes[0, 0].axhline(0, color="black", linestyle="-", alpha=0.3)
    axes[0, 0].axvline(0.04, color="black", linestyle="--", alpha=0.3)
    axes[0, 0].set_xlabel("Curiosity (Δ log I_Q)")
    axes[0, 0].set_ylabel("Investigation (-Δ basin_dist)")
    axes[0, 0].set_title("Exploration vs Pursuit")
    axes[0, 0].grid(alpha=0.3)

    # Φ vs Curiosity
    axes[0, 1].scatter(phi, curiosity, c=colors, alpha=0.5, s=10)
    axes[0, 1].axhline(0.04, color="black", linestyle="--", alpha=0.3)
    axes[0, 1].set_xlabel("Φ (Integration)")
    axes[0, 1].set_ylabel("Curiosity (Δ log I_Q)")
    axes[0, 1].set_title("Integration vs Exploration")
    axes[0, 1].grid(alpha=0.3)

    # Basin Distance vs Integration
    axes[1, 0].scatter(basin_dist, integration, c=colors, alpha=0.5, s=10)
    axes[1, 0].axvline(0.15, color="black", linestyle="--", alpha=0.3, label="d_integrate")
    axes[1, 0].set_xlabel("Basin Distance")
    axes[1, 0].set_ylabel("Integration (-var(Φ×I_Q))")
    axes[1, 0].set_title("Proximity vs Consolidation")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Curiosity vs Basin Distance (mode switch primary)
    axes[1, 1].scatter(basin_dist, curiosity, c=colors, alpha=0.5, s=10)
    axes[1, 1].axhline(0.04, color="black", linestyle="--", alpha=0.3, label="C_high")
    axes[1, 1].axvline(0.5, color="black", linestyle="--", alpha=0.3, label="d_explore")
    axes[1, 1].axvline(0.15, color="gray", linestyle="--", alpha=0.3, label="d_integrate")
    axes[1, 1].set_xlabel("Basin Distance")
    axes[1, 1].set_ylabel("Curiosity (Δ log I_Q)")
    axes[1, 1].set_title("Primary Mode Switch")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "motivator_correlations.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved motivator correlations to {output_path}")
    plt.close()


def plot_mode_statistics(motivators, output_dir: Path):
    """Plot per-mode statistics."""
    # Group by mode
    mode_groups = {}
    for m in motivators:
        mode = m["mode"]
        if mode not in mode_groups:
            mode_groups[mode] = {
                "surprise": [],
                "curiosity": [],
                "investigation": [],
                "integration": [],
                "frustration": [],
            }
        mode_groups[mode]["surprise"].append(m["surprise"])
        mode_groups[mode]["curiosity"].append(m["curiosity"])
        mode_groups[mode]["investigation"].append(m["investigation"])
        mode_groups[mode]["integration"].append(m["integration"])
        mode_groups[mode]["frustration"].append(m["frustration"])

    # Compute statistics
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    motivator_names = ["surprise", "curiosity", "investigation", "integration", "frustration"]

    for idx, motivator in enumerate(motivator_names):
        ax = axes[idx]

        modes = sorted(mode_groups.keys())
        means = [np.mean(mode_groups[mode][motivator]) for mode in modes]
        stds = [np.std(mode_groups[mode][motivator]) for mode in modes]

        colors = ["blue", "gray", "purple", "green", "red"][: len(modes)]

        ax.bar(modes, means, yerr=stds, color=colors, alpha=0.7)
        ax.set_ylabel(f"{motivator.capitalize()}\n(mean ± std)")
        ax.set_xticklabels(modes, rotation=45, ha="right")
        ax.grid(alpha=0.3, axis="y")

    plt.suptitle("Motivator Statistics per Mode", fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = output_dir / "mode_statistics.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved mode statistics to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot cognitive motivators")
    parser.add_argument(
        "--motivators",
        type=Path,
        default=Path("outputs/qig_kernel/motivators.jsonl"),
        help="Path to motivators JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qig_kernel"),
        help="Output directory for plots",
    )

    args = parser.parse_args()

    # Load
    print(f"Loading motivators from {args.motivators}...")
    motivators = load_motivators(args.motivators)
    print(f"✅ Loaded {len(motivators)} steps")

    # Create output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Plot
    print("\nGenerating plots...")
    plot_motivator_timeseries(motivators, args.output_dir)
    plot_mode_distribution(motivators, args.output_dir)
    plot_motivator_correlations(motivators, args.output_dir)
    plot_mode_statistics(motivators, args.output_dir)

    print("\n" + "=" * 60)
    print("✅ All plots generated")
    print(f"   Output: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
