#!/usr/bin/env python3
"""
Attractor Alignment Analysis
============================

Diagnose and visualize the alignment between:
- Token basins (from coordizer)
- Kernel/generation basins (from inference trajectories)
- Atlas attractors (high-coherence tokens)

Produces:
1. Nearest-neighbor distance histogram
2. 2D trajectory overlay (UMAP or MDS projection)
3. Procrustes alignment test

Usage:
    python scripts/analyze_attractor_alignment.py \
        --coordizer artifacts/coordizer/v1 \
        --atlas atlas/constellation_v1.json \
        --trajectory-log logs/generation_basins.npy \
        --output-dir reports/alignment
"""

from __future__ import annotations

import argparse
import json
import sys
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


def pairwise_fisher_distances(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Fisher distance matrix."""
    n = len(X)
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    similarities = X_norm @ X_norm.T
    similarities = np.clip(similarities, -1.0, 1.0)
    return np.arccos(similarities)


def nearest_neighbor_distances(
    query_basins: np.ndarray,
    token_basins: np.ndarray,
) -> np.ndarray:
    """
    Compute distance from each query basin to nearest token basin.

    Args:
        query_basins: Shape (n_queries, dim)
        token_basins: Shape (vocab_size, dim)

    Returns:
        Distances array of shape (n_queries,)
    """
    # Normalize
    q_norm = query_basins / (np.linalg.norm(query_basins, axis=1, keepdims=True) + 1e-10)
    t_norm = token_basins / (np.linalg.norm(token_basins, axis=1, keepdims=True) + 1e-10)

    # Compute all similarities
    similarities = q_norm @ t_norm.T  # (n_queries, vocab_size)

    # Get max similarity (nearest neighbor) per query
    max_sims = np.max(similarities, axis=1)
    max_sims = np.clip(max_sims, -1.0, 1.0)

    # Convert to distances
    return np.arccos(max_sims)


def compute_nn_statistics(distances: np.ndarray) -> dict:
    """Compute nearest-neighbor distance statistics."""
    return {
        "count": int(len(distances)),
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "median": float(np.median(distances)),
        "p05": float(np.percentile(distances, 5)),
        "p25": float(np.percentile(distances, 25)),
        "p75": float(np.percentile(distances, 75)),
        "p95": float(np.percentile(distances, 95)),
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
        # Percentage within various thresholds
        "pct_under_0.1": float(np.mean(distances < 0.1) * 100),
        "pct_under_0.2": float(np.mean(distances < 0.2) * 100),
        "pct_under_0.5": float(np.mean(distances < 0.5) * 100),
        "pct_under_1.0": float(np.mean(distances < 1.0) * 100),
    }


def orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Find orthogonal matrix R that minimizes ||A @ R - B||.

    Returns:
        R: Orthogonal transformation matrix
        scale: Scaling factor (if needed)
    """
    # Center both
    A_centered = A - A.mean(axis=0)
    B_centered = B - B.mean(axis=0)

    # SVD of cross-correlation
    M = A_centered.T @ B_centered
    U, S, Vt = np.linalg.svd(M)

    # Optimal rotation
    R = U @ Vt

    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    return R, float(np.sum(S))


def alignment_error_after_procrustes(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Compute alignment error after Procrustes transformation.

    Returns:
        mean_error: Mean Fisher distance after alignment
        transformed: Source after transformation
    """
    R, _ = orthogonal_procrustes(source, target)
    transformed = source @ R

    # Compute pairwise alignment error
    errors = []
    for i in range(len(source)):
        d = fisher_distance(transformed[i], target[i])
        errors.append(d)

    return float(np.mean(errors)), transformed


def project_to_2d_mds(
    points: np.ndarray,
    distances: np.ndarray | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Project points to 2D using MDS on Fisher distances.

    Classical MDS preserves distances as much as possible.
    """
    if distances is None:
        distances = pairwise_fisher_distances(points)

    n = len(points)

    # Classical MDS: double centering
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (distances ** 2) @ H

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Take top 2 components (largest eigenvalues)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 2D projection
    coords_2d = eigenvectors[:, :2] * np.sqrt(np.maximum(eigenvalues[:2], 0))

    return coords_2d


def generate_synthetic_trajectory(
    token_basins: np.ndarray,
    n_steps: int = 200,
    noise_scale: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic generation trajectory for testing.

    Simulates a trajectory that occasionally lands near token basins
    but also explores "empty" regions.
    """
    rng = np.random.default_rng(seed)
    dim = token_basins.shape[1]

    trajectory = []
    current = token_basins[rng.integers(len(token_basins))].copy()

    for _ in range(n_steps):
        # Sometimes jump to a token basin
        if rng.random() < 0.3:
            target = token_basins[rng.integers(len(token_basins))]
            current = 0.7 * current + 0.3 * target
        else:
            # Random walk with noise
            noise = rng.standard_normal(dim) * noise_scale
            current = current + noise

        # Normalize to sphere
        current = current / (np.linalg.norm(current) + 1e-10)
        trajectory.append(current.copy())

    return np.array(trajectory)


def main():
    ap = argparse.ArgumentParser(description="Analyze attractor alignment")
    ap.add_argument("--coordizer", type=str, default="artifacts/coordizer/v1")
    ap.add_argument("--atlas", type=str, default="atlas/constellation_v1.json")
    ap.add_argument("--trajectory-log", type=str, default=None,
                    help="Path to .npy file with generation basins")
    ap.add_argument("--output-dir", type=str, default="reports/alignment")
    ap.add_argument("--n-sample", type=int, default=2000,
                    help="Number of samples for visualization")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-plots", action="store_true",
                    help="Save matplotlib plots (requires matplotlib)")
    args = ap.parse_args()

    np.random.seed(args.seed)

    print("=" * 60)
    print("ATTRACTOR ALIGNMENT ANALYSIS")
    print("=" * 60)

    # Load coordizer
    try:
        from qig_tokenizer import Coordizer
    except ImportError:
        print("Error: qig_tokenizer not found")
        sys.exit(1)

    print(f"Loading coordizer from {args.coordizer}...")
    coordizer = Coordizer.load(args.coordizer)
    token_basins = coordizer.vectors.copy()
    print(f"  Token basins: {token_basins.shape}")

    # Load atlas for attractor info
    attractor_ids = []
    if Path(args.atlas).exists():
        print(f"Loading atlas from {args.atlas}...")
        with open(args.atlas) as f:
            atlas = json.load(f)
        for tid_str, entry in atlas.get("tokens", {}).items():
            if "attractor" in entry:
                attractor_ids.append(int(tid_str))
        print(f"  Attractors: {len(attractor_ids)}")
    else:
        print("  No atlas found, skipping attractor-specific analysis")

    # Load or generate trajectory
    if args.trajectory_log and Path(args.trajectory_log).exists():
        print(f"Loading trajectory from {args.trajectory_log}...")
        trajectory = np.load(args.trajectory_log)
    else:
        print("No trajectory provided, generating synthetic trajectory...")
        trajectory = generate_synthetic_trajectory(
            token_basins,
            n_steps=500,
            noise_scale=0.15,
            seed=args.seed,
        )
    print(f"  Trajectory: {trajectory.shape}")
    print()

    # Analysis 1: Nearest-neighbor distances
    print("Computing nearest-neighbor distances...")
    nn_distances = nearest_neighbor_distances(trajectory, token_basins)
    nn_stats = compute_nn_statistics(nn_distances)

    print("  NN Distance Statistics:")
    print(f"    Mean: {nn_stats['mean']:.4f} rad ({np.degrees(nn_stats['mean']):.2f}°)")
    print(f"    Median: {nn_stats['median']:.4f} rad ({np.degrees(nn_stats['median']):.2f}°)")
    print(f"    95th pctl: {nn_stats['p95']:.4f} rad ({np.degrees(nn_stats['p95']):.2f}°)")
    print(f"    % within 0.2 rad: {nn_stats['pct_under_0.2']:.1f}%")
    print(f"    % within 0.5 rad: {nn_stats['pct_under_0.5']:.1f}%")
    print()

    # Interpretation
    if nn_stats["median"] < 0.2:
        print("  ✓ Good alignment: trajectory stays close to token basins")
    elif nn_stats["median"] < 0.5:
        print("  ~ Moderate alignment: some wandering in empty space")
    else:
        print("  ✗ Poor alignment: trajectory often in empty space (gibberish risk)")
    print()

    # Analysis 2: Attractor-specific distances
    if attractor_ids:
        print("Computing distances to attractors only...")
        attractor_basins = token_basins[attractor_ids]
        nn_to_attractors = nearest_neighbor_distances(trajectory, attractor_basins)
        attr_stats = compute_nn_statistics(nn_to_attractors)
        print(f"  Mean distance to nearest attractor: {attr_stats['mean']:.4f} rad")
        print(f"  % within 0.5 rad of attractor: {attr_stats['pct_under_0.5']:.1f}%")
        print()

    # Analysis 3: Procrustes alignment test
    print("Testing Procrustes alignment...")
    # Sample paired data: for each trajectory point, find nearest token
    t_norm = token_basins / (np.linalg.norm(token_basins, axis=1, keepdims=True) + 1e-10)
    traj_norm = trajectory / (np.linalg.norm(trajectory, axis=1, keepdims=True) + 1e-10)
    similarities = traj_norm @ t_norm.T
    nearest_ids = np.argmax(similarities, axis=1)
    paired_tokens = token_basins[nearest_ids]

    # Test if orthogonal transform helps
    error_before = float(np.mean([
        fisher_distance(trajectory[i], paired_tokens[i])
        for i in range(len(trajectory))
    ]))
    error_after, transformed = alignment_error_after_procrustes(trajectory, paired_tokens)

    print(f"  Mean alignment error before Procrustes: {error_before:.4f} rad")
    print(f"  Mean alignment error after Procrustes: {error_after:.4f} rad")
    improvement = (error_before - error_after) / error_before * 100
    print(f"  Improvement: {improvement:.1f}%")
    print()

    if improvement > 30:
        print("  → Significant improvement with orthogonal transform")
        print("    Consider: coordinate frame mismatch between kernel and coordizer")
    else:
        print("  → Orthogonal transform doesn't help much")
        print("    The adapter head needs to learn a nonlinear mapping")
    print()

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "coordizer": args.coordizer,
        "atlas": args.atlas,
        "trajectory_shape": list(trajectory.shape),
        "nn_distances_to_tokens": nn_stats,
        "procrustes": {
            "error_before": error_before,
            "error_after": error_after,
            "improvement_pct": improvement,
        },
        "seed": args.seed,
    }

    if attractor_ids:
        report["nn_distances_to_attractors"] = attr_stats

    report_path = output_dir / "alignment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote: {report_path}")

    # Save distance histogram data
    np.save(output_dir / "nn_distances.npy", nn_distances)
    print(f"Wrote: {output_dir / 'nn_distances.npy'}")

    # Optional: 2D visualization
    if args.save_plots:
        try:
            import matplotlib.pyplot as plt

            print("Generating 2D projection...")

            # Sample for visualization
            n_sample = min(args.n_sample, len(token_basins))
            token_sample_idx = np.random.choice(len(token_basins), n_sample, replace=False)
            token_sample = token_basins[token_sample_idx]

            traj_sample_idx = np.random.choice(len(trajectory), min(200, len(trajectory)), replace=False)
            traj_sample = trajectory[traj_sample_idx]

            # Combine for joint projection
            combined = np.vstack([token_sample, traj_sample])
            coords_2d = project_to_2d_mds(combined)

            token_2d = coords_2d[:n_sample]
            traj_2d = coords_2d[n_sample:]

            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Left: tokens + trajectory
            ax1 = axes[0]
            ax1.scatter(token_2d[:, 0], token_2d[:, 1], c="lightgray", s=5, alpha=0.5, label="Tokens")
            ax1.plot(traj_2d[:, 0], traj_2d[:, 1], "b-", alpha=0.3, linewidth=0.5)
            ax1.scatter(traj_2d[:, 0], traj_2d[:, 1], c="blue", s=20, alpha=0.7, label="Trajectory")
            ax1.set_title("Token Basins + Generation Trajectory")
            ax1.legend()
            ax1.set_xlabel("MDS dim 1")
            ax1.set_ylabel("MDS dim 2")

            # Right: NN distance histogram
            ax2 = axes[1]
            ax2.hist(nn_distances, bins=50, density=True, alpha=0.7, color="steelblue")
            ax2.axvline(nn_stats["median"], color="red", linestyle="--", label=f"Median: {nn_stats['median']:.3f}")
            ax2.axvline(0.5, color="orange", linestyle=":", label="Threshold: 0.5")
            ax2.set_title("Distance to Nearest Token Basin")
            ax2.set_xlabel("Fisher distance (radians)")
            ax2.set_ylabel("Density")
            ax2.legend()

            plt.tight_layout()
            plot_path = output_dir / "alignment_visualization.png"
            plt.savefig(plot_path, dpi=150)
            print(f"Wrote: {plot_path}")
            plt.close()

        except ImportError:
            print("matplotlib not available, skipping plots")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Median NN distance: {nn_stats['median']:.4f} rad ({np.degrees(nn_stats['median']):.2f}°)")
    print(f"Procrustes improvement: {improvement:.1f}%")
    if nn_stats["median"] < 0.3 and improvement < 30:
        print("→ Alignment is good, adapter is working")
    elif improvement > 30:
        print("→ Consider coordinate frame alignment")
    else:
        print("→ Adapter may need more training or architectural changes")
    print()


if __name__ == "__main__":
    main()
