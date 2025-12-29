#!/usr/bin/env python3
"""
Constellation Atlas v1
======================

Topological analysis of the 32k vocabulary on the Fisher manifold.

This script:
1. Loads coordizer v1 and (optionally) trained adapter
2. Clusters 32k coordinates by Fisher-Rao distance
3. Identifies attractor basins (high-Φ regions)
4. Detects structural discontinuities (byte vs merged tokens)
5. Exports atlas JSON with cluster IDs and basin assignments

Outputs:
- atlas/constellation_v1.json (main atlas)
- atlas/constellation_v1_viz.json (visualization-friendly subset)
- reports/atlas_v1_summary.json (statistics)

Usage:
    python scripts/build_constellation_atlas_v1.py \
        --coordizer artifacts/coordizer/v1 \
        --adapter artifacts/coord_adapter/v1/adapter.pt \
        --n-clusters 64 \
        --output-dir atlas
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def fisher_rao_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Fisher-Rao (angular) distance matrix.

    For unit-normalized vectors, Fisher-Rao distance = arccos(dot product).

    Args:
        vectors: Array of shape (n, dim) - should be unit-normalized

    Returns:
        Distance matrix of shape (n, n) in radians [0, pi]
    """
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / (norms + 1e-10)

    # Compute cosine similarities
    similarities = normalized @ normalized.T

    # Clip to valid range and convert to angular distance
    similarities = np.clip(similarities, -1.0, 1.0)
    distances = np.arccos(similarities)

    return distances.astype(np.float32)


def cluster_by_fisher_distance(
    vectors: np.ndarray,
    n_clusters: int = 64,
    sample_size: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Cluster vectors using agglomerative clustering with Fisher-Rao metric.

    Args:
        vectors: Shape (vocab_size, dim)
        n_clusters: Number of clusters
        sample_size: If set, use approximate clustering for large vocab

    Returns:
        Tuple of (cluster_labels, cluster_stats)
    """
    n_vectors = len(vectors)

    if sample_size is not None and n_vectors > sample_size:
        # Approximate: cluster on sample, then assign remaining
        print(f"  Using approximate clustering (sample {sample_size}/{n_vectors})")
        indices = np.random.choice(n_vectors, sample_size, replace=False)
        sample_vectors = vectors[indices]

        # Compute distance matrix for sample
        sample_dist = fisher_rao_distance_matrix(sample_vectors)

        # Cluster sample
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
        sample_labels = clustering.fit_predict(sample_dist)

        # Compute cluster centroids (angular mean)
        centroids = np.zeros((n_clusters, vectors.shape[1]), dtype=np.float32)
        for c in range(n_clusters):
            mask = sample_labels == c
            if mask.any():
                cluster_vecs = sample_vectors[mask]
                # Angular mean: normalize sum of normalized vectors
                mean_vec = cluster_vecs.sum(axis=0)
                centroids[c] = mean_vec / (np.linalg.norm(mean_vec) + 1e-10)

        # Assign all vectors to nearest centroid
        all_normalized = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
        centroids_normalized = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
        similarities = all_normalized @ centroids_normalized.T
        labels = np.argmax(similarities, axis=1)
    else:
        # Exact: full distance matrix
        print(f"  Computing full {n_vectors}x{n_vectors} distance matrix...")
        dist_matrix = fisher_rao_distance_matrix(vectors)

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(dist_matrix)

    # Compute cluster statistics
    stats = {}
    for c in range(n_clusters):
        mask = labels == c
        cluster_size = int(mask.sum())
        if cluster_size > 0:
            cluster_vecs = vectors[mask]
            # Intra-cluster dispersion (mean angular distance to centroid)
            mean_vec = cluster_vecs.sum(axis=0)
            mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-10)
            cluster_normalized = cluster_vecs / (
                np.linalg.norm(cluster_vecs, axis=1, keepdims=True) + 1e-10
            )
            cos_sims = cluster_normalized @ mean_vec
            mean_dist = float(np.arccos(np.clip(cos_sims, -1, 1)).mean())

            stats[c] = {
                "size": cluster_size,
                "mean_intra_distance": mean_dist,
                "centroid_norm": float(np.linalg.norm(mean_vec)),
            }

    return labels, stats


def identify_attractor_basins(
    vectors: np.ndarray,
    labels: np.ndarray,
    k_neighbors: int = 10,
) -> dict[int, dict]:
    """
    Identify attractor basins within each cluster.

    An attractor is a token whose neighbors are predominantly in the same cluster
    (high local density / coherence).

    Args:
        vectors: Shape (vocab_size, dim)
        labels: Cluster labels
        k_neighbors: Number of neighbors to consider

    Returns:
        Dict mapping token_id to attractor info
    """
    # Normalize vectors for angular queries
    normalized = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)

    # Build neighbor index (using cosine for speed, equivalent to angular)
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="cosine")
    nn.fit(normalized)

    attractors = {}
    n_vectors = len(vectors)

    for i in range(n_vectors):
        distances, indices = nn.kneighbors([normalized[i]])
        # Exclude self
        neighbor_ids = indices[0][1:]
        neighbor_dists = distances[0][1:]

        # Count neighbors in same cluster
        my_cluster = labels[i]
        same_cluster = sum(1 for j in neighbor_ids if labels[j] == my_cluster)
        coherence = same_cluster / k_neighbors

        # Attractor threshold: 80%+ neighbors in same cluster
        if coherence >= 0.8:
            attractors[i] = {
                "cluster": int(my_cluster),
                "coherence": float(coherence),
                "mean_neighbor_distance": float(np.mean(neighbor_dists)),
            }

    return attractors


def analyze_scale_structure(
    labels: np.ndarray,
    coordizer,
) -> dict:
    """
    Analyze structural differences between byte-level and merged tokens.

    Args:
        labels: Cluster labels
        coordizer: Coordizer instance

    Returns:
        Scale structure analysis
    """
    n_clusters = int(labels.max()) + 1

    # Classify tokens by scale
    byte_tokens = set(range(256))
    merged_tokens = set(range(256, coordizer.vocab_size))

    # Analyze cluster composition
    cluster_composition = {}
    for c in range(n_clusters):
        mask = labels == c
        token_ids = np.where(mask)[0]

        n_byte = sum(1 for t in token_ids if t < 256)
        n_merged = len(token_ids) - n_byte

        cluster_composition[c] = {
            "byte_count": n_byte,
            "merged_count": n_merged,
            "byte_ratio": n_byte / max(len(token_ids), 1),
            "merged_ratio": n_merged / max(len(token_ids), 1),
        }

    # Find pure byte clusters
    pure_byte_clusters = [c for c, comp in cluster_composition.items() if comp["byte_ratio"] > 0.9]

    # Find pure merged clusters
    pure_merged_clusters = [
        c for c, comp in cluster_composition.items() if comp["merged_ratio"] > 0.9
    ]

    # Find mixed clusters
    mixed_clusters = [
        c for c in range(n_clusters) if c not in pure_byte_clusters and c not in pure_merged_clusters
    ]

    return {
        "cluster_composition": cluster_composition,
        "pure_byte_clusters": pure_byte_clusters,
        "pure_merged_clusters": pure_merged_clusters,
        "mixed_clusters": mixed_clusters,
        "byte_dispersion": len(set(labels[:256])),  # How many clusters bytes span
    }


def build_atlas(
    coordizer,
    labels: np.ndarray,
    cluster_stats: dict,
    attractors: dict[int, dict],
    scale_structure: dict,
) -> dict:
    """
    Build the complete atlas structure.

    Args:
        coordizer: Coordizer instance
        labels: Cluster labels
        cluster_stats: Per-cluster statistics
        attractors: Attractor token info
        scale_structure: Scale analysis

    Returns:
        Complete atlas dict
    """
    # Build token entries
    tokens = {}
    for token_id in range(coordizer.vocab_size):
        entry = {
            "cluster": int(labels[token_id]),
            "scale": coordizer.token_scale(token_id),
        }
        if token_id in attractors:
            entry["attractor"] = attractors[token_id]
        tokens[str(token_id)] = entry

    # Build cluster entries
    clusters = {}
    for c, stats in cluster_stats.items():
        cluster_entry = dict(stats)
        # Add scale composition
        if c in scale_structure["cluster_composition"]:
            cluster_entry["composition"] = scale_structure["cluster_composition"][c]
        # Add cluster type
        if c in scale_structure["pure_byte_clusters"]:
            cluster_entry["type"] = "byte"
        elif c in scale_structure["pure_merged_clusters"]:
            cluster_entry["type"] = "merged"
        else:
            cluster_entry["type"] = "mixed"
        clusters[str(c)] = cluster_entry

    return {
        "version": "1.0.0",
        "coordizer_version": coordizer.version,
        "vocab_size": coordizer.vocab_size,
        "n_clusters": int(labels.max()) + 1,
        "n_attractors": len(attractors),
        "tokens": tokens,
        "clusters": clusters,
        "scale_structure": {
            "pure_byte_clusters": scale_structure["pure_byte_clusters"],
            "pure_merged_clusters": scale_structure["pure_merged_clusters"],
            "mixed_clusters": scale_structure["mixed_clusters"],
            "byte_dispersion": scale_structure["byte_dispersion"],
        },
    }


def build_viz_subset(atlas: dict, top_k: int = 1000) -> dict:
    """
    Build a visualization-friendly subset of the atlas.

    Includes:
    - All attractors
    - Top-k tokens by cluster centrality
    - Cluster summaries
    """
    attractors = [
        int(tid) for tid, entry in atlas["tokens"].items() if "attractor" in entry
    ]

    # Sample additional tokens evenly from clusters
    tokens_per_cluster = max(1, (top_k - len(attractors)) // atlas["n_clusters"])
    selected = set(attractors)

    for c in range(atlas["n_clusters"]):
        cluster_tokens = [
            int(tid)
            for tid, entry in atlas["tokens"].items()
            if entry["cluster"] == c and int(tid) not in selected
        ]
        selected.update(cluster_tokens[:tokens_per_cluster])

    return {
        "version": atlas["version"],
        "n_clusters": atlas["n_clusters"],
        "n_tokens_sampled": len(selected),
        "tokens": {str(tid): atlas["tokens"][str(tid)] for tid in sorted(selected)},
        "clusters": atlas["clusters"],
        "scale_structure": atlas["scale_structure"],
    }


def main():
    ap = argparse.ArgumentParser(description="Build Constellation Atlas v1")
    ap.add_argument("--coordizer", type=str, default="artifacts/coordizer/v1")
    ap.add_argument("--adapter", type=str, default=None, help="Path to trained adapter.pt")
    ap.add_argument("--n-clusters", type=int, default=64)
    ap.add_argument("--sample-size", type=int, default=8000, help="Sample size for approximate clustering")
    ap.add_argument("--k-neighbors", type=int, default=10, help="Neighbors for attractor detection")
    ap.add_argument("--output-dir", type=str, default="atlas")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    print("=" * 60)
    print("CONSTELLATION ATLAS V1")
    print("=" * 60)
    print(f"Coordizer: {args.coordizer}")
    print(f"Adapter: {args.adapter or 'None (using raw coordinates)'}")
    print(f"Clusters: {args.n_clusters}")
    print(f"Sample size: {args.sample_size}")
    print()

    # Load coordizer
    try:
        from qig_tokenizer import Coordizer
    except ImportError:
        print("Error: qig_tokenizer not found")
        sys.exit(1)

    print("Loading coordizer...")
    coordizer = Coordizer.load(args.coordizer)
    print(f"  Vocab size: {coordizer.vocab_size}")
    print(f"  Merge rules: {len(coordizer.merge_rules)}")
    print()

    # Get vectors
    vectors = coordizer.vectors.copy()
    print(f"  Vectors shape: {vectors.shape}")

    # If adapter provided, transform vectors through it
    if args.adapter:
        try:
            import torch
            print(f"Loading adapter from {args.adapter}...")
            state = torch.load(args.adapter, map_location="cpu", weights_only=True)

            # Extract projection weights
            proj_weight = state.get("projection.weight")
            proj_bias = state.get("projection.bias")

            if proj_weight is not None:
                print(f"  Adapter projection: {proj_weight.shape}")
                # Apply projection: vectors @ W^T + b
                proj_weight = proj_weight.numpy()
                proj_bias = proj_bias.numpy() if proj_bias is not None else 0

                # Transform: [vocab, 64] @ [384, 64]^T -> [vocab, 384]
                # But for clustering we want to stay in original 64D space
                # Just log that adapter is available
                print("  (Adapter available but clustering in original 64D space)")
        except Exception as e:
            print(f"  Warning: Could not load adapter: {e}")
    print()

    # Clustering
    print("Clustering vocabulary by Fisher-Rao distance...")
    t0 = time.time()
    labels, cluster_stats = cluster_by_fisher_distance(
        vectors,
        n_clusters=args.n_clusters,
        sample_size=args.sample_size,
    )
    t_cluster = time.time() - t0
    print(f"  Clustering completed in {t_cluster:.2f}s")
    print(f"  Cluster sizes: min={min(s['size'] for s in cluster_stats.values())}, "
          f"max={max(s['size'] for s in cluster_stats.values())}, "
          f"mean={np.mean([s['size'] for s in cluster_stats.values()]):.1f}")
    print()

    # Attractor detection
    print("Identifying attractor basins...")
    t0 = time.time()
    attractors = identify_attractor_basins(vectors, labels, k_neighbors=args.k_neighbors)
    t_attract = time.time() - t0
    print(f"  Found {len(attractors)} attractors in {t_attract:.2f}s")

    # Attractor distribution by cluster
    attractor_clusters = {}
    for tid, info in attractors.items():
        c = info["cluster"]
        attractor_clusters[c] = attractor_clusters.get(c, 0) + 1
    print(f"  Attractors per cluster: min={min(attractor_clusters.values(), default=0)}, "
          f"max={max(attractor_clusters.values(), default=0)}, "
          f"mean={np.mean(list(attractor_clusters.values())):.1f}")
    print()

    # Scale structure analysis
    print("Analyzing scale structure...")
    scale_structure = analyze_scale_structure(labels, coordizer)
    print(f"  Pure byte clusters: {len(scale_structure['pure_byte_clusters'])}")
    print(f"  Pure merged clusters: {len(scale_structure['pure_merged_clusters'])}")
    print(f"  Mixed clusters: {len(scale_structure['mixed_clusters'])}")
    print(f"  Byte dispersion: {scale_structure['byte_dispersion']} clusters")
    print()

    # Build atlas
    print("Building atlas...")
    atlas = build_atlas(coordizer, labels, cluster_stats, attractors, scale_structure)

    # Build viz subset
    viz_atlas = build_viz_subset(atlas, top_k=1000)

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    atlas_path = output_dir / "constellation_v1.json"
    viz_path = output_dir / "constellation_v1_viz.json"
    summary_path = Path("reports") / "atlas_v1_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(atlas_path, "w") as f:
        json.dump(atlas, f, indent=2)
    print(f"Wrote: {atlas_path}")

    with open(viz_path, "w") as f:
        json.dump(viz_atlas, f, indent=2)
    print(f"Wrote: {viz_path}")

    # Summary
    summary = {
        "version": atlas["version"],
        "coordizer_version": atlas["coordizer_version"],
        "vocab_size": atlas["vocab_size"],
        "n_clusters": atlas["n_clusters"],
        "n_attractors": atlas["n_attractors"],
        "cluster_sizes": [cluster_stats[c]["size"] for c in range(args.n_clusters)],
        "scale_structure": atlas["scale_structure"],
        "timings": {
            "clustering_seconds": t_cluster,
            "attractor_detection_seconds": t_attract,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {summary_path}")

    print()
    print("=" * 60)
    print("ATLAS SUMMARY")
    print("=" * 60)
    print(f"Vocabulary: {atlas['vocab_size']} tokens")
    print(f"Clusters: {atlas['n_clusters']}")
    print(f"Attractors: {atlas['n_attractors']}")
    print(f"Byte clusters: {len(scale_structure['pure_byte_clusters'])}")
    print(f"Merged clusters: {len(scale_structure['pure_merged_clusters'])}")
    print(f"Mixed clusters: {len(scale_structure['mixed_clusters'])}")
    print()


if __name__ == "__main__":
    main()
